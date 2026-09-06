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
from .graph import DsaiGraph
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
        self._graph = None

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

    # ---------- 主入口（阶段三：委托 LangGraph 状态机） ----------
    def answer(self, question: str, *, use_alt: bool = False) -> dict:
        if self._graph is None:
            self._graph = DsaiGraph(self)
        return self._graph.answer(question, use_alt=use_alt)


def run_one(question: str, *, cfg: Config | None = None, use_alt: bool = False) -> dict:
    return Pipeline(cfg).answer(question, use_alt=use_alt)
