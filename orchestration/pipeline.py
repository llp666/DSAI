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
# not the chat bubble: 口径 (metric caliber / retrieval rules, agent-facing) + 数据来源表
# (source tables) + SQL 溯源 (SQL) + 已尝试 SQL (degrade trace). They always come last in the
# rendered template, so the chat answer cuts off at the first hit.
_TECH_APPENDIX_MARKERS = ("口径：", "数据来源表：", "SQL 溯源：", "已尝试 SQL：")


def _strip_technical_appendix(text: str) -> str:
    """Cut the caliber / SQL / source-table appendix out of an answer for the chat bubble.

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
    # 推理语言约束行（思考过程（reasoning…）——防止该指令被模型 echo 进面板
    "思考过程（reasoning", "用中文思考", "全程用中文",
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


# ---- trivial casual fast-path: 你好/hi/谢谢/再见 → instant canned reply (zero LLM latency) ----
# Whole-string short forms only: anything carrying a real request (「你好，帮我查下GMV」) falls
# through to the tool round. This kills the "still hangs a moment before replying" gap for the
# simplest chats the user called out, without touching the needs-thinking casual path.
_GREETING_RE = re.compile(
    r"^(?:你好|您好|哈喽|嗨|hello|hi|hey|早上好|下午好|晚上好|在吗|在不在)[呀啊哈!！。~～.…、\s]*$",
    re.IGNORECASE)
_THANKS_RE = re.compile(
    r"^(?:谢谢|感谢|多谢|thanks|thank you|谢谢啦|谢谢你|辛苦啦)[!！。~～.…、\s]*$",
    re.IGNORECASE)
_BYE_RE = re.compile(
    r"^(?:再见|拜拜|晚安|bye|回聊)[!！。~～.…、\s]*$",
    re.IGNORECASE)

_GREETING_REPLY = (
    "你好！我是电商智能问数 Agent，可以帮你查 GMV、销售额、订单、库存、营销 ROI 等经营数据。"
    "想查点什么？比如「上个月 GMV 是多少」或者「哪些 SKU 断货了」。"
)
_THANKS_REPLY = "不客气！有 GMV、销量、库存、广告投放等问题随时问我～"
_BYE_REPLY = "再见！有数据问题随时来找我 👋"


def _trivial_reply(question: str) -> str | None:
    """Instant canned reply for a pure greeting/thanks/goodbye (no LLM call → no gap).

    Returns None when the message isn't trivially short — it then goes through the real
    chat path (thinking + optional search_tool)."""
    q = question.strip()
    if not q:
        return None
    if _THANKS_RE.fullmatch(q):
        return _THANKS_REPLY
    if _BYE_RE.fullmatch(q):
        return _BYE_REPLY
    if _GREETING_RE.fullmatch(q):
        return _GREETING_REPLY
    return None


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
            if self.cfg.rerank and self.cfg.rerank.api_key and self.cfg.rerank_enabled:
                self._reranker = Reranker(self.cfg.rerank.base_url,
                                          self.cfg.rerank.api_key,
                                          self.cfg.rerank.model)
        self._env = Environment(
            loader=FileSystemLoader(resolve(Path("tools/templates")))
        )
        self._answer_tpl = self._env.get_template("answer.j2")
        self._graph = None
        self._monitor: QuestionMonitor | None = None  # per-question budget (set at answer entry)

    def warm_up(self) -> None:
        """Best-effort background warm-up so the first question skips the HTTP handshake.

        Measured: the first embedding of a process costs ~1970ms vs ~320ms warm — the whole
        difference is DNS + TLS. Called once at app start (``app.get_pipeline``), deliberately
        *not* from ``__init__`` so tests building a Pipeline directly don't spawn network threads.
        """
        if self._vector_store is None:
            return
        threading.Thread(target=self._vector_store.warm_up, daemon=True,
                         name="pipeline-warmup").start()

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
        """Chat entry: business → state machine; casual (incl. real-time questions) → LLM chat.

        Returns (result_box, chunks): ``result_box`` is filled with the state-machine result for
        business questions ({} for casual); ``chunks`` yields tagged (reasoning, content) chunks.
        Real-time questions (天气/新闻/汇率…) are the LLM's call inside the casual tool round —
        no keyword route reaches the search API anymore (stage 3-5).
        """
        route = route_dialogue(question)
        if route == "business":
            return self.answer_stream(question, thread_id=thread_id, history=history,
                                      result_box=result_box)
        return {}, self._casual_chunks(question, history)

    def _casual_chunks(self, question: str, history: list[dict] | None = None):
        """Stream a casual reply (no semantic query / SQL); yields tagged (reasoning, content) chunks.

        Two tiers:
        - 简单问候（你好/hi/谢谢/再见…）→ 0 LLM 延迟的即时回复：不调模型、无思考面板、无 gap;
        - 需要真正回答的闲聊 → thinking 开启的流式对话（思考过程展示在面板里，② 防泄漏），
          且工具轮让 LLM 自主决定是否调用 search_tool 查实时信息（③ 从关键词路由改为 LLM 决策）。
        """
        instant = _trivial_reply(question)
        if instant is not None:
            yield "content", instant
            return
        yield "phase", "正在思考…"
        # 语义层指标释义注入：概念题（什么是GMV/解释一下ROI）由 casual 链路回答时，
        # 优先用本系统语义层的口径定义（口径已固化、权威），LLM 背景知识只作补充；
        # 未知领域/实时热点仍由 LLM 自主决定调 search_tool 联网检索。
        # getattr 降级：MagicMock/精简桩可能没有 layer —— 释义块置空，其余规则不变。
        layer = getattr(self, "layer", None)
        metric_digest = ""
        if layer is not None:
            metric_digest = "\n".join(
                f"- {name}（{m['display_name']}）：{m['description']}"
                for name, m in layer.metrics.items()
            )
        system = (
            "你是「电商智能问数 Agent」，一个面向电商业务数据的智能问答助手，"
            "能帮用户查询 GMV、销售额、订单、库存等经营指标，也能联网查实时/资讯类问题。\n"
            "对话规则：\n"
            "- 始终以「电商智能问数 Agent」自称，绝不透露底层模型名称或厂商身份（如 agnes、OpenAI 等）。\n"
            "- 思考过程（reasoning）与作答必须使用与用户提问相同的语言——用户用中文提问时必须全程用中文思考与拆解，绝不切换为英文。\n"
            "- 友好、简洁地回答问候、闲聊、自我介绍、功能帮助类问题。\n"
            "- 解释概念/名词（如「什么是GMV」「ROI 是什么意思」）时，优先依据下方「本系统指标口径」"
            "作答——口径与数仓实现一致，解释完可顺带告诉用户可以直接问这个指标的数值；"
            "若问的概念不在清单里，用你的背景知识简洁解释。\n"
            "- 若问题需要当前/实时信息（天气、新闻、最新资讯、汇率、股价、热点等），"
            "调用 search_tool 搜索后再回答，并附上「信息来源」。\n"
            "- 若用户提到数据或指标但表述不明确，引导其说清要查询的指标与时间范围。\n"
            "- 不编造任何数据或数字；涉及具体数值的问题交由数据查询链路回答。\n"
            "\n本系统指标口径：\n" + metric_digest
        )
        hist = render_history(history)
        user = question if not hist else f"对话历史：\n{hist}\n\n当前问题：{question}"
        from tools.search import search_tool_schema

        llm = self._get_llm(False)
        # 概念题快链路（无工具轮）：释义已在 system 里，无需联网检索 —— 跳过工具 schema
        # 序列化与 LLM 的工具决策轮，直接流式作答（agnes 首字从 ~10s 压回 ~3.5s）。
        # 未知领域/实时热点不在此列：它们没有概念题意图，仍走带工具轮的完整 casual 路径。
        from orchestration.dialogue import _is_concept_question

        if _is_concept_question(question):
            for kind, text in llm.stream_complete(system, user):
                if kind == "reasoning":
                    cleaned = _sanitize_reasoning(text)
                    if cleaned is not None:
                        yield kind, cleaned
                else:
                    yield kind, text
            return
        # thinking 开启（有需要思考的闲聊展示思考过程），reasoning 经防泄漏过滤
        for kind, text in llm.stream_complete_with_tools(system, user, [search_tool_schema()]):
            if kind == "reasoning":
                cleaned = _sanitize_reasoning(text)
                if cleaned is not None:
                    yield kind, cleaned
            else:
                yield kind, text
        # LLM 自主选择了 search_tool → 用它的 query 联网搜索并按来源摘要回答
        query = None
        for tc in (getattr(llm, "last_tool_calls", None) or []):
            if tc.get("name") == "search_tool":
                query = (tc.get("arguments") or {}).get("query") or None
                if query:
                    break
        if query:
            yield from self._search_chunks(question, history, query=query)

    def _search_chunks(self, question: str, history: list[dict] | None = None,
                       *, query: str | None = None):
        """Web-search a reply grounded in live keenable results, then a sourced LLM summary.

        Entered when the LLM autonomously chose search_tool (stage 3-5) — not a keyword route.
        ``query`` is the LLM's chosen search term (falls back to the original question). Same
        fast path as casual (content-only, thinking off); degrades to an honest reply when the
        API is unconfigured or the query fails.
        """
        from tools.search import web_search
        from tools.contract import ToolError

        # 生成器惰性：第一个 yield 前不执行函数体 —— 先发「正在联网检索…」过渡提示，
        # 再执行阻塞的 web_search，UI 不白屏
        yield "phase", "正在联网检索…"

        cfg = self.cfg.search
        if cfg is None:
            yield "content", ("（暂时无法回答实时问题：未配置搜索 API，"
                              "请在 .env 中设置 SEARCH_API_KEY）")
            return
        q = query or question
        try:
            results = web_search(q, cfg)
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
            "你是电商智能问数 Agent。用户的问题需要联网实时信息，下面是网页搜索结果。\n"
            "请基于搜索结果，用自然语言简洁、准确地回答用户问题；\n"
            "- 在回答末尾附上「信息来源」列表（标题 + 链接）。\n"
            "- 若搜索结果不足以回答，如实说明，绝不编造。"
        )
        hist = render_history(history)
        user = f"问题：{q}\n\n网页搜索结果：\n{sources}"
        if hist:
            user = f"对话历史：\n{hist}\n\n{user}"
        for kind, text in self._get_llm(False).stream_complete(system, user, no_thinking=True):
            if kind == "content":
                yield kind, text


def run_one(question: str, *, cfg: Config | None = None, use_alt: bool = False) -> dict:
    return Pipeline(cfg).answer(question, use_alt=use_alt)