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

from jinja2 import Environment, FileSystemLoader
from pydantic import ValidationError

from .compiler import CompileError, compile_query
from .config import Config, load_config, resolve
from .embedding import Embedder, Reranker
from .executor import Executor
from .llm import LLMClient, build_llm
from .prompts import build_system_prompt
from .prompt_budget import render_tables_budgeted
from .repair import try_repair
from .retriever import Retriever
from .semantic_layer import load_semantic_layer
from .tracing import build_tracer
from .types import AgentState, SemanticQuery
from .vector_store import VectorStore


# 检索注入 Token 预算（agnes maxInput 128k 的 1/8，预留指令/回答余量）
RETRIEVAL_BUDGET = 16000


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
        self._vector_store = None
        self._reranker = None
        emb = self.cfg.embedding or self.cfg.alt_embedding
        if emb:
            self._vector_store = VectorStore(
                emb.chroma_dir,
                Embedder(emb.provider, emb.base_url, emb.api_key, emb.model,
                         emb.dimensions, emb.instruction, emb.task))
            if self.cfg.rerank and self.cfg.rerank.api_key:
                self._reranker = Reranker(self.cfg.rerank.base_url,
                                          self.cfg.rerank.api_key,
                                          self.cfg.rerank.model)
        self._env = Environment(
            loader=FileSystemLoader(resolve(Path("agent/templates")))
        )
        self._answer_tpl = self._env.get_template("answer.j2")

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

    def _generate(self, question: str, schema_text: str, llm: LLMClient,
                  today: str) -> tuple[SemanticQuery | None, str | None, int]:
        system = build_system_prompt(self.layer, schema_text, today)
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

    def _assemble_answer(self, sq: SemanticQuery, value: Any,
                         retrieved_tables: list[str], sql: str) -> str:
        meta = self.layer.metrics[sq.metric]
        ym = sq.window.value
        suffix = self._dim_suffix(sq)
        if value is None:
            return f"【{meta['display_name']}】{ym}" + (f" · {suffix}" if suffix else "") + " 无数据"
        return self._answer_tpl.render(
            metric_name=meta["display_name"],
            month=ym,
            dim_suffix=suffix,
            value_text=self._format_value(sq.metric, value),
            caliber=meta["description"],
            retrieved_tables=retrieved_tables,
            sql=sql,
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
            # 1) 检索（v2 混合检索：向量+关键词 RRF；无 embedding 时回退硬编码）
            with t.span("retrieve") as sp:
                if self._vector_store is not None:
                    tables = self.retriever.retrieve_hybrid(
                        question, self._vector_store, top_tables=3,
                        vector_n=10, reranker=self._reranker)
                else:
                    tables = self.retriever.retrieve(question)  # 回退
                state["retrieved_tables"] = [x["table"] for x in tables]
                # Token 预算裁剪：先字段明细、后描述，外键永不裁
                schema_text, budget_reports = render_tables_budgeted(
                    tables, RETRIEVAL_BUDGET)
                sp.update(output={
                    "tables": state["retrieved_tables"],
                    "schema_tokens": len(schema_text),
                    "budget": RETRIEVAL_BUDGET,
                })

            # 2) 语义查询生成（pydantic 校验 + 重试 1 次）
            sq, raw, attempts = None, None, 0
            with t.generation(
                "generate", model=llm.model, input={"question": question},
                output={"raw": None},
            ) as gen:
                try:
                    sq, raw, attempts = self._generate(question, schema_text, llm, today)
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

            # 3) 编译 SQL + EXPLAIN 干跑校验（确定性优先：编译失败先走修复规则）
            sql = None
            with t.span("compile") as sp:
                try:
                    sql = compile_query(sq, self.layer)
                except CompileError as e:
                    state["execution_error"] = str(e)
                    state["answer"] = f"编译失败：{e}"
                    return state
                state["compiled_sql"] = sql
                sp.update(output={"sql": sql})

            # 4) EXPLAIN 干跑：编译通过后先验可行性（read_only 毫秒级）
            dry_error = None
            with t.span("dry_run") as sp:
                dry_error = self.executor.explain_dry_run(sql)
                sp.update(output={"ok": dry_error is None, "error": dry_error})

            # 若干跑失败 → 确定性修复规则表（阶段三纠错轨地基）
            if dry_error:
                repaired = try_repair(dry_error, sql)
                if repaired:
                    sql = repaired
                    state["compiled_sql"] = sql
                    # 修复后复验
                    dry_error = self.executor.explain_dry_run(sql)
                if dry_error:
                    state["execution_error"] = f"SQL 干跑校验失败（确定性修复未覆盖）：{dry_error}"
                    state["answer"] = f"SQL 不可执行：{dry_error}"
                    return state

            # 5) 只读执行
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

            # 6) 回答组装（口径披露 + 命中表 + SQL 溯源）
            with t.span("answer") as sp:
                answer = self._assemble_answer(
                    sq, value, state["retrieved_tables"], state["compiled_sql"])
                state["answer"] = answer
                sp.update(output={"answer": answer})

            t.set_trace_io(input={"question": question}, output={"answer": answer})
            state["trace_id"] = t.trace_id
        return state


def run_one(question: str, *, cfg: Config | None = None, use_alt: bool = False) -> dict:
    return Pipeline(cfg).answer(question, use_alt=use_alt)
