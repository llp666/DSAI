"""orchestration/pipeline.py：minimal chain orchestration.

question → retrieve (hardcoded Top-3) → LLM generates a semantic query (pydantic-validated, 1 retry)
        → compile SQL → DuckDB read-only execute → caliber-disclosing answer
Full flow traced via Langfuse (retrieve / generate / compile / execute spans).
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader
from pydantic import ValidationError

from compile.compiler import CompileError, compile_query
from config import Config, load_config, resolve
from retrieval.embedding import Embedder, Reranker
from compile.executor import Executor
from orchestration.graph import DsaiGraph
from orchestration.llm import LLMClient, build_llm
from orchestration.monitor import MonitoredLLM, QuestionMonitor
from orchestration.dialogue import route_dialogue
from orchestration.prompts import build_system_prompt, render_history
from retrieval.prompt_budget import render_tables_budgeted
from orchestration.repair import try_repair
from retrieval.retriever import Retriever
from data.semantic import load_semantic_layer
from tracing import build_tracer
from compile.types import AgentState, SemanticQuery
from retrieval.vector_store import VectorStore


# retrieval injection token budget (1/8 of agnes 128k maxInput, leaving room for instruction/answer)
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
            loader=FileSystemLoader(resolve(Path("tools/templates")))
        )
        self._answer_tpl = self._env.get_template("answer.j2")
        self._graph = None
        self._monitor: QuestionMonitor | None = None  # per-question budget (set at answer entry)

    def _get_llm(self, use_alt: bool) -> LLMClient:
        if use_alt:
            llm = build_llm(self.cfg, use_alt=True)
        elif self._llm is None:
            self._llm = build_llm(self.cfg)
            llm = self._llm
        else:
            llm = self._llm
        # budget wrapper: when a monitor is active, each call counts (overrun raises BudgetExceeded)
        if self._monitor is not None:
            return MonitoredLLM(llm, self._monitor)
        return llm

    def begin_question(self, **limits) -> QuestionMonitor:
        """Open the per-question budget monitor (answer entry); reuses the current one on repeat."""
        if self._monitor is None:
            self._monitor = QuestionMonitor(**limits)
        return self._monitor

    def end_question(self) -> dict | None:
        """Close the per-question monitor, return the budget snapshot, and reset."""
        if self._monitor is None:
            return None
        snap = self._monitor.snapshot()
        self._monitor = None
        return snap

    # ---------- semantic query generation (pydantic validation + 1 retry) ----------
    def _parse_semantic_query(self, raw: str) -> SemanticQuery:
        text = raw.strip()
        # strip possible markdown code fences ```json ... ```
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
        if m:
            text = m.group(1)
        return SemanticQuery.model_validate_json(text)

    def _generate(self, question: str, schema_text: str, llm: LLMClient,
                  today: str) -> tuple[SemanticQuery | None, str | None, int]:
        system = build_system_prompt(self.layer, schema_text, today)
        last_error: str | None = None
        for attempt in range(2):  # first try + 1 retry
            user = question if attempt == 0 else (
                f"{question}\n\n注意：你上一次输出的语义查询 JSON 解析失败：{last_error}\n"
                "请只输出符合格式要求的合法 JSON。"
            )
            try:
                raw = llm.complete(system, user)
                return self._parse_semantic_query(raw), raw, attempt
            except (ValidationError, json.JSONDecodeError) as e:
                last_error = str(e)
            except Exception:  # LLMError etc.: no retry, re-raise
                raise
        return None, last_error, 1

    # ---------- answer assembly (caliber disclosure) ----------
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
        """Build a readable suffix from dimension filters, e.g. 「服饰品类」「一线城市」."""
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
        # multi-row grouped result (per-category GMV etc.): one line per row "dim value: metric value"
        if isinstance(value, list):
            lines = [f"【{meta['display_name']}】{ym}"
                     + (f" · {suffix}" if suffix else "")]
            for row in value:
                if isinstance(row, list) and row:
                    dim_v = row[0]
                    val_v = row[1] if len(row) > 1 else row[0]
                    lines.append(f"  {dim_v}: {self._format_value(sq.metric, val_v)}")
            return "\n".join(lines) + "\n\n口径：" + meta["description"]
        return self._answer_tpl.render(
            metric_name=meta["display_name"],
            month=ym,
            dim_suffix=suffix,
            value_text=self._format_value(sq.metric, value),
            caliber=meta["description"],
            retrieved_tables=retrieved_tables,
            sql=sql,
        )

    # ---------- main entry (stage 3: delegate to the LangGraph state machine) ----------
    def answer(self, question: str, *, use_alt: bool = False,
               thread_id: str | None = None, history: list[dict] | None = None) -> dict:
        if self._graph is None:
            self._graph = DsaiGraph(self)
        return self._graph.answer(question, use_alt=use_alt,
                                  thread_id=thread_id, history=history)

    def answer_stream(self, question: str, *, use_alt: bool = False,
                      thread_id: str | None = None,
                      history: list[dict] | None = None):
        """Run the pipeline and return (result, chunks): ``chunks`` streams the answer text.

        The deterministic pipeline runs to completion first; the final answer is then rewritten +
        streamed by the LLM from the grounded result. Tool-insight, degrade and empty answers are
        echoed verbatim (no second LLM call).
        """
        result = self.answer(question, use_alt=use_alt, thread_id=thread_id, history=history)
        return result, self._answer_chunks(result)

    def chat_stream(self, question: str, *, thread_id: str | None = None,
                    history: list[dict] | None = None):
        """Chat entry: route business questions to the state machine, casual talk to plain LLM chat.

        Returns (result, chunks): ``result`` is the deterministic state-machine result for business
        questions (or {} for casual), ``chunks`` streams the answer text.
        """
        if route_dialogue(question) == "business":
            return self.answer_stream(question, thread_id=thread_id, history=history)
        return {}, self._casual_chunks(question, history)

    def _casual_chunks(self, question: str, history: list[dict] | None = None):
        """Stream a plain conversational reply (no semantic query / SQL)."""
        system = (
            "你是「电商智能问数 Agent」，一个面向电商业务数据的智能问答助手，"
            "能帮用户查询 GMV、销售额、订单、库存等经营指标。\n"
            "对话规则：\n"
            "- 始终以「电商智能问数 Agent」自称，绝不透露底层模型名称或厂商身份（如 agnes、OpenAI 等）。\n"
            "- 友好、简洁地回答问候、闲聊、自我介绍、功能帮助类问题。\n"
            "- 若用户提到数据或指标但表述不明确，引导其说清要查询的指标与时间范围。\n"
            "- 不编造任何数据或数字；涉及具体数值的问题交由数据查询链路回答。"
        )
        hist = render_history(history)
        user = question if not hist else f"对话历史：\n{hist}\n\n当前问题：{question}"
        yield from self._get_llm(False).stream_complete(system, user)

    def _answer_chunks(self, result: dict):
        """Yield the deterministic answer verbatim (tool insight / normal / degrade all ship)."""
        yield result.get("answer", "")


def run_one(question: str, *, cfg: Config | None = None, use_alt: bool = False) -> dict:
    return Pipeline(cfg).answer(question, use_alt=use_alt)