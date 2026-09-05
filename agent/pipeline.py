"""agent/pipeline.py：阶段一最小链路编排。

问题 → 检索(硬编码 Top-3) → LLM 生成语义查询(pydantic 校验, 失败重试 1 次)
     → 编译 SQL → DuckDB 只读执行 → 口径披露回答
全流程 Langfuse 埋点（retrieve / generate / compile / execute 四 span）。
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .compiler import CompileError, compile_query
from .config import Config, load_config, resolve
from .executor import Executor
from .llm import LLMClient, build_llm
from .prompts import build_system_prompt
from .retriever import Retriever
from .semantic_layer import load_semantic_layer
from .tracing import build_tracer
from .types import AgentState, SemanticQuery


class Pipeline:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or load_config()
        self.layer = load_semantic_layer(
            resolve(self.cfg.semantic_layer_path).parent,
            resolve(self.cfg.semantic_layer_path).parent / "schema/semantic_layer.schema.json",
        )
        self.retriever = Retriever(resolve(self.cfg.meta_dir))
        self.executor = Executor(resolve(self.cfg.duckdb_path))
        self.tracer = build_tracer(self.cfg)
        self._llm: LLMClient | None = None

    def _get_llm(self, use_alt: bool) -> LLMClient:
        if use_alt:
            return build_llm(self.cfg, use_alt=True)
        if self._llm is None:
            self._llm = build_llm(self.cfg)
        return self._llm

    # ---------- 语义查询生成（含 pydantic 校验 + 重试 1 次） ----------
    def _parse_semantic_query(self, raw: str) -> SemanticQuery:
        text = raw.strip()
        # 去掉可能的 markdown 代码围栏 ```json ... ```
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
        if m:
            text = m.group(1)
        return SemanticQuery.model_validate_json(text)

    def _generate(self, question: str, tables: list[dict], llm: LLMClient,
                  today: str) -> tuple[SemanticQuery | None, str | None, int]:
        system = build_system_prompt(self.layer, tables, today)
        last_error: str | None = None
        for attempt in range(2):  # 首次 + 重试 1 次
            user = question if attempt == 0 else (
                f"{question}\n\n注意：你上一次输出的语义查询 JSON 解析失败：{last_error}\n"
                "请只输出符合格式要求的合法 JSON。"
            )
            try:
                raw = llm.complete(system, user)
                return self._parse_semantic_query(raw), raw, attempt
            except (ValidationError, json.JSONDecodeError) as e:
                last_error = str(e)
            except Exception as e:  # LLMError 等：不重试，直接上抛
                raise
        return None, last_error, 1

    # ---------- 回答组装（口径披露） ----------
    _RATE_METRICS = {"repurchase_rate", "refund_rate", "gross_margin"}
    _INT_METRICS = {"orders_count"}

    def _format_value(self, metric: str, value: Any) -> str:
        if isinstance(value, str):
            return value
        if metric in self._RATE_METRICS:
            return f"{float(value) * 100:.2f}%"
        if metric in self._INT_METRICS:
            return f"{float(value):,.0f}"
        return f"{float(value):,.2f}"

    def _dim_suffix(self, sq: SemanticQuery) -> str:
        """从维度过滤生成可读后缀，如「服饰品类」「一线城市」。"""
        display = {n: d["display_name"] for n, d in self.layer.dimensions.items()}
        parts = []
        for f in sq.filters:
            v = f.value if isinstance(f.value, str) else "、".join(f.value)
            parts.append(f"{v}{display.get(f.dim, f.dim)}")
        return " · ".join(parts)

    def _assemble_answer(self, sq: SemanticQuery, value: Any) -> str:
        meta = self.layer.metrics[sq.metric]
        ym = sq.window.value
        suffix = self._dim_suffix(sq)
        head = f"【{meta['display_name']}】{ym}" + (f" · {suffix}" if suffix else "")
        if value is None:
            return f"{head} 无数据"
        return (
            f"{head} = {self._format_value(sq.metric, value)}\n"
            f"口径：{meta['description']}"
        )

    # ---------- 主入口 ----------
    def answer(self, question: str, *, use_alt: bool = False) -> dict:
        llm = self._get_llm(use_alt)
        today = self.cfg.reference_date or date.today().isoformat()
        state: AgentState = {
            "question": question,
            "retrieved_tables": [],
            "semantic_query": None,
            "compiled_sql": None,
            "execution_result": None,
            "execution_error": None,
            "retry_count": 0,
            "answer": "",
            "trace_id": "",
        }
        with self.tracer.trace(f"question: {question[:40]}") as t:
            # 1) 检索（阶段一硬编码 Top-3）
            with t.span("retrieve") as sp:
                tables = self.retriever.retrieve(question)
                state["retrieved_tables"] = [x["table"] for x in tables]
                sp.update(output={"tables": state["retrieved_tables"]})

            # 2) 语义查询生成（pydantic 校验 + 重试 1 次）
            sq, raw, attempts = None, None, 0
            with t.generation(
                "generate", model=llm.model, input={"question": question},
                output={"raw": None},
            ) as gen:
                try:
                    sq, raw, attempts = self._generate(question, tables, llm, today)
                except Exception as e:
                    state["execution_error"] = f"LLM 调用失败: {e}"
                    state["answer"] = f"无法生成语义查询：{e}"
                    t.set_trace_io(input={"question": question}, output={"error": str(e)})
                    return state
                state["semantic_query"] = sq
                state["retry_count"] = attempts
                usage = getattr(llm, "last_usage", None)
                gen.update(
                    input={"question": question, "retry": attempts},
                    output={"raw": raw, "semantic_query": sq.model_dump() if sq else None},
                    usage_details=usage,
                )

            if sq is None:
                state["execution_error"] = f"语义查询解析失败（已重试1次）：{raw}"
                state["answer"] = f"无法解析语义查询：{raw}"
                t.set_trace_io(input={"question": question}, output={"error": state["execution_error"]})
                return state

            # 3) 编译 SQL
            with t.span("compile") as sp:
                try:
                    sql = compile_query(sq, self.layer)
                except CompileError as e:
                    state["execution_error"] = str(e)
                    state["answer"] = f"编译失败：{e}"
                    return state
                state["compiled_sql"] = sql
                sp.update(output={"sql": sql})

            # 4) 只读执行
            with t.span("execute") as sp:
                try:
                    rows = self.executor.execute(sql)
                    value = rows[0][0] if rows else None
                except Exception as e:
                    state["execution_error"] = f"SQL 执行失败: {e}"
                    state["answer"] = f"执行失败：{e}"
                    sp.update(output={"error": str(e)})
                    return state
                state["execution_result"] = value
                sp.update(output={"result": value})

            # 5) 回答组装（口径披露）
            with t.span("answer") as sp:
                answer = self._assemble_answer(sq, value)
                state["answer"] = answer
                sp.update(output={"answer": answer})

            t.set_trace_io(input={"question": question}, output={"answer": answer})
            state["trace_id"] = t.trace_id
        return state


def run_one(question: str, *, cfg: Config | None = None, use_alt: bool = False) -> dict:
    return Pipeline(cfg).answer(question, use_alt=use_alt)
