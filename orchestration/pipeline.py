"""orchestration/pipeline.py：minimal chain orchestration.

question → retrieve (hardcoded Top-3) → LLM generates a semantic query (pydantic-validated, 1 retry)
        → compile SQL → DuckDB read-only execute → caliber-disclosing answer
Full flow traced via Langfuse (retrieve / generate / compile / execute spans).
"""

from __future__ import annotations

import json
import queue
import re
import threading
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

# Technical appendix markers inside the deterministic answer that belong in the audit trail,
# not the chat bubble: 数据来源表 (source tables) + SQL 溯源 (SQL) + 已尝试 SQL (degrade trace).
# They always come last in the rendered template, so the chat answer cuts off at the first hit.
_TECH_APPENDIX_MARKERS = ("数据来源表：", "SQL 溯源：", "已尝试 SQL：")


def _strip_technical_appendix(text: str) -> str:
    """Cut the SQL / source-table appendix out of an answer for the chat bubble.

    ``state["answer"]`` keeps the full template (口径 + 数据来源表 + SQL 溯源) for the eval /
    audit trail; the chat UI only wants the human-readable head. No-op when no marker is present."""
    hit = [i for i in (text.find(m) for m in _TECH_APPENDIX_MARKERS) if i >= 0]
    if not hit:
        return text
    return text[: min(hit)].rstrip()


# System-prompt boilerplate a thinking model occasionally recites inside its chain-of-thought
# (「对话规则」 block from the casual prompt, semantic-query generator / tool-round contracts,
# repair reflectors). These lines are instructions, not reasoning — drop any reasoning delta
# containing one so the visible thinking panel never re-surfaces hidden system prompts.
# Chat-only filter: eval consumes the non-streaming answer(), not reasoning.
_REASONING_FILTER_MARKERS = (
    # casual system prompt（对话规则块）
    "你是「电商智能问数 Agent」", "对话规则", "电商智能问数 Agent」自称",
    "友好、简洁地回答", "若用户提到数据或指标", "不编造任何数据",
    # 语义查询生成器 system prompt
    "语义查询生成器", "可用指标", "可用维度", "相关表结构", "输出格式",
    "只输出合法 JSON", "今天的日期", "few-shot",
    # 工具轮 system prompt
    "适合调用诊断工具", "只用于「", "聚合计数题", "禁止编造数字",
    "真实数据来源", "基于工具返回的真实结果",
    # 纠错 / 空结果反思 system prompt
    "纠错器", "空结果反思器", "放宽口径",
)


def _sanitize_reasoning(chunk: str) -> str | None:
    """Drop a reasoning delta that echoes system-prompt boilerplate (None = filtered out)."""
    if any(m in chunk for m in _REASONING_FILTER_MARKERS):
        return None
    return chunk


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
                  today: str, on_reasoning=None) -> tuple[SemanticQuery | None, str | None, int]:
        system = build_system_prompt(self.layer, schema_text, today)
        last_error: str | None = None
        for attempt in range(2):  # first try + 1 retry
            user = question if attempt == 0 else (
                f"{question}\n\n注意：你上一次输出的语义查询 JSON 解析失败：{last_error}\n"
                "请只输出符合格式要求的合法 JSON。"
            )
            try:
                if on_reasoning is not None:
                    raw = llm.complete_stream(system, user, on_reasoning=on_reasoning)
                else:
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
               thread_id: str | None = None, history: list[dict] | None = None,
               on_reasoning=None, on_phase=None) -> dict:
        if self._graph is None:
            self._graph = DsaiGraph(self)
        return self._graph.answer(question, use_alt=use_alt,
                                  thread_id=thread_id, history=history,
                                  on_reasoning=on_reasoning, on_phase=on_phase)

    def answer_stream(self, question: str, *, use_alt: bool = False,
                      thread_id: str | None = None,
                      history: list[dict] | None = None,
                      result_box: dict | None = None):
        """Stream a business question as tagged (reasoning, content) chunks (Kimi-style).

        Returns (result_box, chunks): ``result_box`` (a dict, created if omitted) is filled with
        the final state-machine result when streaming ends; ``chunks`` is a generator of tagged
        chunks. The state machine runs in a background thread; its live chain-of-thought is
        pumped through a queue so the UI can render a collapsible thinking panel while the answer
        is still being generated. Phase markers ("phase", label) are emitted at each pipeline stage
        (理解问题 → 检索数据表 → 生成查询 → 执行计算) so the panel transitions smoothly instead of
        freezing during the retrieval wait.
        """
        if result_box is None:
            result_box = {}
        q: queue.Queue = queue.Queue()

        def _hook(chunk: str) -> None:
            # system-prompt echoes never reach the thinking panel (issue ②)
            cleaned = _sanitize_reasoning(chunk)
            if cleaned is not None:
                q.put(("reasoning", cleaned))

        def _phase(label: str) -> None:
            q.put(("phase", label))

        def _run() -> None:
            try:
                result_box["result"] = self.answer(
                    question, use_alt=use_alt, thread_id=thread_id,
                    history=history, on_reasoning=_hook, on_phase=_phase)
            except Exception as e:  # surface errors as the final content chunk
                result_box["error"] = e
            finally:
                q.put(("_done", None))

        def _stream():
            threading.Thread(target=_run, daemon=True).start()
            while True:
                kind, text = q.get()
                if kind == "_done":
                    break
                yield kind, text
            if "error" in result_box:
                yield "content", f"（出错了：{result_box['error']}）"
            else:
                # chat-visible answer: strip the SQL / source-table appendix (eval keeps the
                # full text), then hand it to the UI in small slices so it streams out smoothly
                # instead of dumping all at once.
                text = _strip_technical_appendix(result_box["result"].get("answer", ""))
                for i in range(0, len(text), 12):
                    yield "content", text[i:i + 12]

        return result_box, _stream()

    def chat_stream(self, question: str, *, thread_id: str | None = None,
                    history: list[dict] | None = None, result_box: dict | None = None):
        """Chat entry: business → state machine; real-time → web search; casual → plain LLM chat.

        Returns (result_box, chunks): ``result_box`` is filled with the state-machine result for
        business questions ({} for search/casual); ``chunks`` yields tagged (reasoning, content)
        chunks. Only business questions carry phase/reasoning — search & casual stream content
        straight out (no retrieval chain, no thinking panel).
        """
        route = route_dialogue(question)
        if route == "business":
            return self.answer_stream(question, thread_id=thread_id, history=history,
                                      result_box=result_box)
        if route == "search":
            return {}, self._search_chunks(question, history)
        return {}, self._casual_chunks(question, history)

    def _casual_chunks(self, question: str, history: list[dict] | None = None):
        """Stream a plain conversational reply (no semantic query / SQL); yields tagged chunks.

        Casual chat is NOT a business question: it skips the retrieve→think→answer chain.
        Thinking is disabled at the provider (chat_template_kwargs per docs/model.md) so the
        reply is fast, and only content chunks are yielded — no reasoning panel, no chance of
        the system prompt ("对话规则") leaking into the visible reply.
        """
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
        # no_thinking speeds the reply up at the API level; the content-only filter is
        # defensive — even a provider that ignores the toggle never yields reasoning here.
        for kind, text in self._get_llm(False).stream_complete(system, user, no_thinking=True):
            if kind == "content":
                yield kind, text

    def _search_chunks(self, question: str, history: list[dict] | None = None):
        """Real-time questions (天气/新闻/最新资讯…) → web search, then a sourced LLM summary.

        Same fast path as casual (content-only, thinking off), but grounded in live keenable
        search results — the search API is documented in docs/model.md (stage-3 reserved) and
        configured via SEARCH_API_KEY / SEARCH_BASE_URL. Degrades to an honest reply when the
        API is unconfigured or the query fails.
        """
        from tools.search import web_search
        from tools.contract import ToolError

        cfg = self.cfg.search
        if cfg is None:
            yield "content", ("（暂时无法回答实时问题：未配置搜索 API，"
                              "请在 .env 中设置 SEARCH_API_KEY）")
            return
        try:
            results = web_search(question, cfg)
        except ToolError as e:
            yield "content", f"（搜索失败：{e.as_text()}）"
            return
        if not results:
            yield "content", "（没有搜到与这个问题相关的结果，换个问法试试）"
            return
        sources = "\n".join(
            f"- {r['title']} | {r['url']}\n  {r['snippet']}" for r in results
        )
        system = (
            "你是电商智能问数 Agent。用户问了一个实时/资讯类问题，下面是网页搜索结果。\n"
            "请基于搜索结果，用自然语言简洁、准确地回答用户问题；\n"
            "- 在回答末尾附上「信息来源」列表（标题 + 链接）。\n"
            "- 若搜索结果不足以回答，如实说明，绝不编造。"
        )
        hist = render_history(history)
        user = f"问题：{question}\n\n网页搜索结果：\n{sources}"
        if hist:
            user = f"对话历史：\n{hist}\n\n{user}"
        for kind, text in self._get_llm(False).stream_complete(system, user, no_thinking=True):
            if kind == "content":
                yield kind, text


def run_one(question: str, *, cfg: Config | None = None, use_alt: bool = False) -> dict:
    return Pipeline(cfg).answer(question, use_alt=use_alt)