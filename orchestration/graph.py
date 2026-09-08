"""orchestration/graph.py：stage-3 LangGraph state machine (nine nodes + SQL tools + repair track).

Migrates stage-2's function chain into a StateGraph:
- main chain: intent → retrieve → generate → tools → judge → answer
- judge three-way conditional edges (four-priority decision):
    success → answer (normal assembly)
    repairable → repair (deterministic rules) → reflect (LLM reflection) → generate (same-round retry, ≤ max_retries)
    unrepairable → answer (degraded)
- tools node = ToolNode([compile, dry_run, execute], handle_tool_errors=True), tool exceptions wrapped
  as ToolMessage「Error: …」and fed back to repair/reflect/generate.

State and components are injected via Pipeline (graph.py doesn't import pipeline.py, avoiding a cycle).
"""

from __future__ import annotations

import json
import uuid
from datetime import date
from decimal import Decimal
from typing import Any, Literal

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from compile.compiler import CompileError, compile_query
from orchestration.error_classifier import build_repair_prompt, classify_error, format_available_dimensions
from orchestration.monitor import BudgetExceeded
from orchestration.preflight import preflight
from retrieval.prompt_budget import render_tables_budgeted
from orchestration.prompts import build_system_prompt
from orchestration.relevance import check_relevance
from retrieval.rewrite import rewrite
from retrieval.query_rewrite import rewrite_query
from tools.contract import ToolError, serialize_result
from orchestration.tool_router import route_tool_intent
from tools.inventory import inventory_diagnostic_tool
from tools.marketing import marketing_roi_funnel_tool
from tools.search import SEARCH_ARGS_SCHEMA
from compile.types import AgentState, SemanticQuery

# retrieval injection token budget (1/8 of agnes 128k maxInput, leaving room for instruction/answer)
RETRIEVAL_BUDGET = 16000
MAX_RETRIES = 2
# recursion limit: main chain 5 steps + per-round repair 5 steps × (MAX_RETRIES+1) + margin.
# overrun raises GraphRecursionError, caught in answer() and degraded rather than silently truncated
# (checkpointer keeps the full path visible in Langfuse)
RECURSION_LIMIT = 5 * (MAX_RETRIES + 1) + 5


class _NullCtx:
    """Empty context manager when tracing is off (compatible with Langfuse spans)."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def update(self, **kw):
        pass


# intent/domain classification (keyword → domain label; first match wins)
INTENT_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("营销域", ("roi", "渠道", "广告", "投放", "转化", "归因", "花费", "营销", "cpm", "ctr")),
    ("供应链域", ("库存", "断货", "呆滞", "补货", "在途", "供应商", "入库", "仓库", "动销", "采购")),
    ("用户域", ("用户", "复购", "新老", "画像", "城市", "线级", "会员", "客群")),
    ("商品域", ("sku", "品类", "商品", "毛利", "价格", "销量", "排行")),
    ("退款域", ("退款", "退货", "售后")),
    ("促销域", ("促销", "大促", "618", "双11", "活动", "优惠")),
    ("订单域", ("订单", "gmv", "成交", "支付", "销售额", "净销", "客单")),
]


def _classify_intent(question: str) -> str:
    q = question.lower()
    for domain, words in INTENT_RULES:
        if any(w.lower() in q for w in words):
            return domain
    return "通用"


def _parse_result(res: str):
    """Parse execute_tool's JSON row array: single row/col → scalar, else list[list]."""
    try:
        rows = json.loads(res)
    except json.JSONDecodeError:
        return res
    if isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], list) and len(rows[0]) == 1:
        return rows[0][0]
    return rows


def _is_empty(result) -> bool:
    """Empty verdict: None or empty list (rank/snapshot metrics return [] with no data)."""
    return result is None or (isinstance(result, list) and len(result) == 0)


def _looks_like_json(text: str) -> bool:
    """Rough check whether LLM output looks like JSON (a semantic-query object)."""
    t = text.strip()
    return t.startswith("{") or t.startswith("```json") or t.startswith("```")


def _jsonable_state(final: dict) -> dict:
    """Serialization guard: drop state fields that are unconsumed and not JSON-serializable.

    LangGraph accumulates AIMessage/ToolMessage objects into state.messages, which app/eval don't
    consume; a wholesale json.dumps(state) would fail on them. Drop messages only. semantic_query
    is a pydantic model — not a JSON scalar, but eval (run_eval/run_injection) consumes it as a
    model (.model_dump()/.dimensions/.window.value), so keep it as-is. The chart is already a JSON
    string upstream (_tool_chart); no go.Figure ever leaves.
    """
    return {k: v for k, v in final.items() if k != "messages"}


def _build_tool_user(question: str, tool_result: str | None) -> str:
    """Build the tool-round user message: first round is the question; later rounds append the result/error."""
    if not tool_result:
        return question
    if tool_result.startswith("工具调用失败"):
        return (f"问题：{question}\n\n"
                f"【诊断工具调用失败】\n{tool_result}\n\n"
                "请不要再调用诊断工具，改用语义查询 JSON 回答这个问题"
                "（引用语义层已定义的指标，如 marketing_roi 按渠道类型聚合 ROI）。")
    return (f"问题：{question}\n\n"
            f"【已调用的诊断工具结果】\n{tool_result}\n\n"
            "请基于以上工具结果，用自然语言给出最终洞察答案。"
            "若工具结果不足以回答，再输出语义查询 JSON。")


def _parse_reflect_empty(raw: str) -> dict:
    """Parse reflect_empty's LLM output: {"action": "confirm"|"relax", "reason": "..."}.

    Accepts only confirm/relax (no LLM-supplied new_window) — the relax window is deterministic,
    so the LLM can't quietly change calibers to force a number.
    """
    import re as _re

    text = raw.strip()
    m = _re.search(r"```(?:json)?\s*(.*?)\s*```", text, _re.S)
    if m:
        text = m.group(1)
    data = json.loads(text)
    action = data.get("action")
    if action not in ("confirm", "relax"):
        raise ValueError(f"reflect_empty 动作非法：{action!r}")
    return {"action": action, "reason": data.get("reason", "")}


def _relax_window(sq: SemanticQuery) -> SemanticQuery | None:
    """Deterministic window relax: single day → its month; month (widest) → None (no relax).

    Only one relax is allowed — day-empty → whole month — with metric/dimensions/filters unchanged.
    """
    if sq.window.type == "month":
        return None
    month = sq.window.value[:7]  # YYYY-MM-DD → YYYY-MM
    data = sq.model_dump()
    data["window"] = {"type": "month", "value": month}
    return SemanticQuery.model_validate(data)


class DsaiGraph:
    def __init__(self, pipeline: Any):
        self.p = pipeline
        self.cfg = pipeline.cfg
        self.layer = pipeline.layer
        self.retriever = pipeline.retriever
        self.executor = pipeline.executor
        self.tracer = pipeline.tracer
        self._vector_store = pipeline._vector_store
        self._reranker = pipeline._reranker
        self._t = None  # current trace session (set at answer entry, used by node spans)
        self._use_alt = False
        self._checkpointer = MemorySaver()
        self._build_tools()
        self._graph = self._build_graph()

    # ---------- three SQL tools (@tool + ToolNode error wrapping) ----------

    def _build_tools(self) -> None:
        executor = self.executor
        layer = self.layer

        @tool
        def compile_tool(semantic_query: dict) -> str:
            """把语义查询编译为可执行 SQL（编译器确定性展开）。"""
            sq = SemanticQuery.model_validate(semantic_query)
            try:
                return compile_query(sq, layer)
            except CompileError as e:
                raise RuntimeError(f"编译失败: {e}") from e

        @tool
        def dry_run_tool(sql: str) -> str:
            """EXPLAIN 干跑：校验 SQL 可执行性（只读、毫秒级）。"""
            err = executor.explain_dry_run(sql)
            if err is not None:
                raise RuntimeError(f"SQL 干跑校验失败: {err}")
            return "OK"

        @tool
        def execute_tool(sql: str) -> str:
            """执行只读 SQL，返回完整行结果（JSON 数组；Decimal→float 保类型）。"""
            try:
                rows = executor.execute(sql)
            except Exception as e:
                raise RuntimeError(f"SQL 执行失败: {e}") from e
            return json.dumps(rows, ensure_ascii=False,
                              default=lambda o: float(o) if isinstance(o, Decimal) else str(o))

        @tool
        def inventory_tool(args: dict) -> str:
            """库存诊断工具：输入 sku_id/category，计算可售天数（库存/近30天日均）、
            DOI（可售天数超90=呆滞）、近30天销量环比；输出逐 SKU 洞察 + DOI×环比散点象限图。
            用于「某 SKU 还够卖几天」「哪些 SKU 呆滞」「补货建议」类问题。"""
            try:
                res = inventory_diagnostic_tool(executor, args)
            except ToolError as e:
                raise RuntimeError(e.as_text()) from e
            return serialize_result(res)

        @tool
        def marketing_tool(args: dict) -> str:
            """营销ROI漏斗工具：拆解曝光→点击→加购→支付各环节转化（按渠道类型）。
            用于「投放漏斗」「转化链路」类问题。"""
            try:
                res = marketing_roi_funnel_tool(executor, args)
            except ToolError as e:
                raise RuntimeError(e.as_text()) from e
            return serialize_result(res)

        self._tools = [compile_tool, dry_run_tool, execute_tool,
                       inventory_tool, marketing_tool]
        self._tool_node = ToolNode(self._tools, handle_tool_errors=True)
        self._compile_tool, self._dry_run_tool, self._execute_tool = self._tools[:3]
        self._inventory_tool, self._marketing_tool = self._tools[3:]

    def _call_tool(self, name: str, args: dict, msgs: list, config) -> tuple[str | None, str | None]:
        """Call a tool; wrap errors into ToolMessage「Error: …」for the repair track. Returns (error, result).

        Diagnostic tools (inventory/marketing) call the underlying function directly (bypassing
        ToolNode's checkpointer-config dependency); the three SQL tools go through ToolNode.
        """
        if name in ("inventory_tool", "marketing_tool"):
            try:
                res = (inventory_diagnostic_tool(self.executor, args) if name == "inventory_tool"
                       else marketing_roi_funnel_tool(self.executor, args))
                content = serialize_result(res)
            except ToolError as e:
                content = f"Error: {e.as_text()}"
            return (content if content.startswith("Error:") else None), content
        if name == "search_tool":
            from tools.search import web_search_tool

            if self.cfg.search is None:
                return "搜索未配置（请在 .env 中设置 SEARCH_API_KEY）", None
            try:
                content = serialize_result(web_search_tool(self.cfg.search, args))
            except ToolError as e:
                content = f"Error: {e.as_text()}"
            return (content if content.startswith("Error:") else None), content
        call = AIMessage(content="", tool_calls=[{
            "name": name, "args": args, "id": f"call_{name}_{len(msgs)}", "type": "tool_call",
        }])
        out = self._tool_node.invoke({"messages": msgs + [call]}, config)
        tmsg = out["messages"][-1]
        msgs.extend([call, tmsg])
        content = str(tmsg.content)
        if content.startswith("Error:"):
            return content, None
        return None, content

    # ---------- optional diagnostic-tool call (stage 3-4E) ----------

    def _tool_schemas(self) -> list[dict]:
        """OpenAI-compatible tool definitions (for optional LLM tool calls).

        Built by hand (not convert_to_openai_function): ``args: dict`` becomes a v__args list under
        langchain, losing the inner JSON Schema. Construct directly from each tool module's args
        schema so the LLM sees the correct parameter definition.
        """
        from tools.inventory import INVENTORY_ARGS_SCHEMA
        from tools.marketing import MARKETING_ARGS_SCHEMA
        return [
            {"type": "function", "function": {
                "name": "inventory_tool",
                "description": ("库存诊断工具：输入 sku_id/category，计算可售天数（库存/近30天日均）、"
                                "DOI（可售天数超90=呆滞）、近30天销量环比；输出逐 SKU 洞察 + DOI×环比散点象限图。"
                                "用于「某 SKU 还够卖几天」「哪些 SKU 呆滞」「补货建议」类问题。"),
                "parameters": INVENTORY_ARGS_SCHEMA,
            }},
            {"type": "function", "function": {
                "name": "marketing_tool",
                "description": ("营销ROI漏斗工具：拆解曝光→点击→加购→支付各环节转化（按渠道类型）。"
                                "用于「投放漏斗」「转化链路」类问题。"),
                "parameters": MARKETING_ARGS_SCHEMA,
            }},
            {"type": "function", "function": {
                "name": "search_tool",
                "description": ("网页搜索工具：输入 query，返回相关网页的标题/链接/摘要。"
                                "用于实时/资讯类问题（天气、新闻、最新资讯、汇率、股价、热点等），"
                                "或需要当前信息才能回答的问题；调用后基于返回结果回答并附「信息来源」。"
                                "业务指标问题（GMV/销量/库存…）不要调用。"),
                "parameters": SEARCH_ARGS_SCHEMA,
            }},
        ]

    _TOOL_MAX_ROUNDS = 2  # cap on in-generate tool-call loops (against agnes duplicate tool_calls burning tokens)

    def _tool_system_prompt(self, state: AgentState) -> str:
        """Tool-round system prompt: semantic-query JSON contract + tool-call rules."""
        today = self.cfg.reference_date or date.today().isoformat()
        base = build_system_prompt(self.p.layer, state["schema_text"], today)
        return (
            "你是电商数据分析 Agent。先判断问题是否适合调用诊断工具（见下方 functions）。\n"
            "规则：\n"
            "- inventory_tool 只用于「逐 SKU 明细诊断」：具体 SKU 还够卖几天/是否呆滞，"
            "或「哪些/哪个 SKU 断货」列出 SKU 明细清单。\n"
            "- 聚合计数题（如「断货 SKU 数是多少」「有多少 SKU 断货」「库存预警 SKU 数」）要的是一个总数，"
            "不是 SKU 明细 → 不要调用 inventory_tool，直接输出语义查询 JSON（指标 stockout_skus_count）。\n"
            "- marketing_tool 只用于投放漏斗/转化链路（曝光→点击→加购→支付分环节转化）。\n"
            "- search_tool 只用于实时/资讯类问题（天气/新闻/最新资讯/汇率/股价等）或需要当前信息才能回答的问题，"
            "调用后基于返回结果回答并附「信息来源」；业务指标问题不要搜索。\n"
            "- 调用工具后，基于工具返回的真实结果，用自然语言输出洞察（可引用 sku_id/天数/环比/环节转化率）。\n"
            "- 工具报错/未实现/不足以回答 → 放弃工具，改输出下方「输出格式」的语义查询 JSON（引用已定义指标）。\n"
            "- 没有任何真实数据来源时禁止编造数字；无法回答就明确说明，绝不猜数。\n\n"
            f"{base}"
        )

    @staticmethod
    def _extract_chart(res: str) -> str:
        """Extract the Plotly figure JSON from a tool-return JSON (empty when no chart)."""
        try:
            payload = json.loads(res)
            chart = payload.get("chart") or {}
            if chart.get("kind") == "plotly_figure_json":
                return chart.get("figure_json", "")
        except Exception:
            pass
        return ""

    def _run_tool_round(self, state: AgentState, llm, config) -> tuple[dict, str | None, SemanticQuery | None]:
        """One round of "LLM may call tools": returns (state updates, insight answer or None, query or None).

        Flow: call the LLM with tools → if tool_calls, execute them (dedup) → feed the ToolResult back
        → the LLM converges on a final answer (insight) or a semantic-query JSON.

        Returns by priority:
        - tool_answer non-None → insight answer produced directly (→ answer, skips semantic query)
        - sq non-None → the round converged on a valid semantic-query JSON (generate_node uses it directly)
        - both None → the round produced nothing (only when not entering the tool round)
        """
        msgs = list(state.get("messages", []))
        start = len(msgs)
        schemas = self._tool_schemas()
        used_tool: str | None = None
        tool_answer: str | None = None
        sq: SemanticQuery | None = None
        reasoning_parts: list[str] = []
        upd: dict = {"messages": msgs[start:]}
        for round_i in range(self._TOOL_MAX_ROUNDS):
            content, tool_calls = llm.complete_with_tools(
                self._tool_system_prompt(state),
                _build_tool_user(state["question"], tool_answer), schemas)
            if getattr(llm, "last_reasoning", None):
                reasoning_parts.append(llm.last_reasoning)
            if not tool_calls:
                # LLM didn't call a tool → output is a semantic-query JSON or insight-answer text
                if _looks_like_json(content):
                    try:
                        sq = self.p._parse_semantic_query(content)
                    except Exception:
                        tool_answer = content  # not valid JSON → treat as insight answer
                    else:
                        if used_tool:
                            upd["tool_used"] = used_tool
                        upd["_reasoning"] = "\n\n".join(reasoning_parts)
                        return upd, None, sq  # semantic-query path (tool info recorded)
                else:
                    tool_answer = content or tool_answer  # plain text → insight answer
                break
            # has tool_calls: dedup then execute
            seen: set = set()
            last_res: str | None = None
            for tc in tool_calls:
                name = tc["name"]
                key = (name, json.dumps(tc.get("arguments", {}), sort_keys=True))
                if key in seen:
                    continue  # agnes occasionally duplicates tool_calls → execute once only
                seen.add(key)
                used_tool = name
                err, res = self._call_tool(name, tc["arguments"], msgs, config)
                last_res = f"工具调用失败：{err}" if err else res
                if err is None and res:
                    upd["_tool_chart"] = self._extract_chart(res)
            tool_answer = last_res  # full ToolResult JSON or error text fed back for the next round
        if used_tool:
            upd["tool_used"] = used_tool
        upd["messages"] = msgs[start:]
        upd["_reasoning"] = "\n\n".join(reasoning_parts)
        # tool failed and LLM didn't converge → fall back to semantic-query path (no raw error as the answer)
        if tool_answer and (tool_answer.startswith("工具调用失败") or tool_answer.startswith("Error:")):
            return upd, None, None
        return upd, tool_answer, sq

    def _emit_phase(self, label: str) -> None:
        """Fire a pipeline-stage label (no-op outside the chat streaming UI)."""
        cb = getattr(self, "_on_phase", None)
        if cb is not None:
            cb(label)

    # ---------- nodes ----------

    def _span(self, name: str, **kw):
        """Wrap a Langfuse span (no-op when tracing is off)."""
        if self._t is not None:
            return self._t.span(name, **kw)
        return _NullCtx()

    def intent_node(self, state: AgentState) -> dict:
        self._emit_phase("正在理解问题…")
        ref = date.fromisoformat(self.cfg.reference_date) if self.cfg.reference_date else date.today()
        q = rewrite(state["question"], ref)  # deterministic time completion
        # three-layer RAG rewrite: coreference (history) / synonym / intent — before retrieval
        q = rewrite_query(q, history=state.get("history"), llm=self.p._get_llm(self._use_alt))
        intent = _classify_intent(q)
        return {"question": q, "intent": intent}

    def retrieve_node(self, state: AgentState) -> dict:
        self._emit_phase("正在检索数据表…")
        with self._span("retrieve", output={"intent": state.get("intent")}):
            degraded = False
            tables = []
            if self._vector_store is not None:
                try:
                    tables = self.retriever.retrieve_hybrid(
                        state["question"], self._vector_store, top_tables=3,
                        vector_n=10, reranker=self._reranker)
                except Exception as e:
                    # embedding network failure → pure-keyword fallback (recorded, no crash)
                    degraded = True
                    tables = self.retriever.retrieve_keyword_only(state["question"])
            if not tables:
                tables = self.retriever.retrieve(state["question"])  # final hardcoded fallback
            schema_text, _ = render_tables_budgeted(tables, RETRIEVAL_BUDGET)
            return {
                "retrieved_tables": [t["table"] for t in tables],
                "schema_text": schema_text,
                "_retrieve_degraded": degraded,
            }

    def generate_node(self, state: AgentState, config=None) -> dict:
        # label the phase truthfully: a diagnostic-tool question runs a tool round, not a plain query
        intent = route_tool_intent(state["question"])
        self._emit_phase("正在调用诊断工具…" if (intent.hit or intent.needs_llm) else "正在生成查询…")
        # reflect already rewrote the query: pass through (no LLM override), straight to tools recompile
        if state.get("_reflect_fixed"):
            return {"execution_error": None, "_reflect_fixed": False}
        with self._span("generate", input={"retry": state.get("retry_count")}):
            llm = self.p._get_llm(self._use_alt)
            on_reasoning = (config or {}).get("configurable", {}).get("on_reasoning")
            # optional tool-call round (stage 3-4E): only when intent routing hits a diagnostic domain
            # (inventory/marketing funnel), avoiding an extra tools-armed LLM call for ordinary questions
            upd: dict = {}
            tool_answer: str | None = None
            sq: SemanticQuery | None = None
            if intent.hit or intent.needs_llm:
                upd, tool_answer, sq = self._run_tool_round(state, llm, config)
            if tool_answer is not None:
                # insight answer: go straight to answer, skip the semantic-query path
                return {**upd, "tool_answer": tool_answer, "semantic_query": None,
                        "execution_error": None, "stage": "tool"}
            if sq is not None:
                # tool round converged on a valid semantic-query JSON: straight to validate/tools
                return {**upd, "semantic_query": sq, "execution_error": None}
            today = self.cfg.reference_date or date.today().isoformat()
            question = state["question"]
            feedback = state.get("error_feedback", "")
            if feedback:
                question = (f"{question}\n\n【上次失败原因】\n{feedback}\n"
                            "请根据错误原因修正你的语义查询 JSON，只输出合法 JSON。")
            try:
                sq, raw, attempts = self.p._generate(question, state["schema_text"], llm, today,
                                                     on_reasoning=on_reasoning)
            except BudgetExceeded:
                raise  # budget: pass through so answer() degrades rather than swallowing it
            except Exception as e:
                return {"semantic_query": None, "execution_error": f"LLM 调用失败: {e}"}
            # merge tool-round + generation reasoning for the thinking panel
            reasoning = upd.get("_reasoning", "")
            if getattr(llm, "last_reasoning", None):
                reasoning = f"{reasoning}\n\n{llm.last_reasoning}".strip()
            return {**upd, "semantic_query": sq, "execution_error": None,
                    "_reasoning": reasoning}

    def tools_node(self, state: AgentState, config) -> dict:
        self._emit_phase("正在执行计算…")
        with self._span("tools", input={"n_errors": len(state.get("errors", []))}):
            msgs = list(state.get("messages", []))
            start = len(msgs)
            errors = list(state.get("errors", []))
            upd: dict = {}
            sq = state.get("semantic_query")
            if sq is not None:
                # 1) compile
                err, sql = self._call_tool("compile_tool",
                                           {"semantic_query": sq.model_dump()}, msgs, config)
                if err:
                    errors.append(err)
                else:
                    upd["compiled_sql"] = sql
                    # 2) dry-run
                    err, _ = self._call_tool("dry_run_tool", {"sql": sql}, msgs, config)
                    if err:
                        errors.append(err)
                    else:
                        # 3) execute
                        err, res = self._call_tool("execute_tool", {"sql": sql}, msgs, config)
                        if err:
                            errors.append(err)
                        else:
                            upd["execution_result"] = _parse_result(res)
            # add_messages reducer appends; return only this round's new messages
            return {**upd, "errors": errors, "messages": msgs[start:]}

    def validate_node(self, state: AgentState) -> dict:
        """Pre-check (before compile): four-layer preflight + relevance/intent check.

        preflight: date boundary / enum dictionary / granularity fallback / cutoff — deterministic
        interception; relevance (check_relevance): DSL-unsupported entity → hallucination (degrade),
        granularity mismatch → intent_mismatch (repair). Only pass through to tools on success.
        """
        q = state.get("question", "")
        sq = state.get("semantic_query")
        if sq is None:
            return {"hallucination": False, "intent_mismatch": False,
                    "preflight_kind": "pass", "preflight_reason": ""}
        # layer 1: four-layer preflight (deterministic, zero LLM)
        cutoff = self.layer.context.get("visible_data_cutoff")
        try:
            verdict = preflight(sq, self.layer, cutoff=cutoff)
        except Exception as e:
            # preflight crash → block + degrade (honest), don't crash the graph
            return {"hallucination": False, "intent_mismatch": False,
                    "preflight_kind": "cutoff",
                    "preflight_reason": f"预检异常：{e}",
                    "execution_error": f"预检异常：{e}"}
        if not verdict.ok:
            with self._span("validate", output={"kind": verdict.kind,
                                                "reason": verdict.reason[:120]}):
                if verdict.kind == "cutoff":
                    # window past the visible data cutoff → block + degrade (not silently empty)
                    return {"hallucination": False, "intent_mismatch": False,
                            "preflight_kind": "cutoff",
                            "preflight_reason": verdict.reason,
                            "execution_error": verdict.reason}
                if verdict.kind == "granularity":
                    # illegal dimension/metric combo → feed back to repair
                    return {"hallucination": False, "intent_mismatch": True,
                            "preflight_kind": "granularity",
                            "preflight_reason": verdict.reason,
                            "error_feedback": verdict.reason + " " + verdict.suggestion}
                # date / enum: invalid value/format → feed back to repair
                return {"hallucination": False, "intent_mismatch": True,
                        "preflight_kind": verdict.kind,
                        "preflight_reason": verdict.reason,
                        "error_feedback": verdict.reason + " " + verdict.suggestion}
        # layer 2: relevance/intent check
        rel = check_relevance(q, sq.metric, sq.dimensions, sq.filters)
        with self._span("validate", output={"kind": rel.kind,
                                            "reason": rel.reason}):
            if rel.kind == "hallucination":
                return {"hallucination": True, "intent_mismatch": False,
                        "preflight_kind": "pass", "preflight_reason": "",
                        "execution_error": rel.reason}
            if rel.kind == "granularity":
                return {"hallucination": False, "intent_mismatch": True,
                        "preflight_kind": "pass", "preflight_reason": "",
                        "error_feedback": rel.reason}
            return {"hallucination": False, "intent_mismatch": False,
                    "preflight_kind": "pass", "preflight_reason": ""}

    def judge_node(self, state: AgentState) -> dict:
        """judge node (after validate, before compile): preflight/relevance verdict for routing."""
        stage = self._judge_pre_cond(state)
        with self._span("judge", output={"stage": stage,
                                         "preflight": state.get("preflight_kind")}):
            return {"stage": stage}

    def judge_post_node(self, state: AgentState) -> dict:
        """judge node (after tools, post compile/execute): execution verdict."""
        stage = self._judge_cond(state)
        with self._span("judge", output={"stage": stage,
                                         "n_errors": len(state.get("errors", []))}):
            return {"stage": stage}

    def _judge_pre_cond(self, state: AgentState) -> Literal["answer", "repair", "tools"]:
        """post-validate (pre-compile) routing: hallucination/cutoff → degrade; fixable → repair; pass → tools."""
        if state.get("hallucination"):
            return "answer"
        if state.get("preflight_kind") == "cutoff":
            return "answer"
        if state.get("intent_mismatch") or state.get("preflight_kind") in ("date", "enum", "granularity"):
            return "repair"
        return "tools"

    def _judge_cond(self, state: AgentState) -> Literal["answer", "repair", "degrade", "reflect_empty"]:
        """post-tools (post compile/execute) priority:
        hallucination→degrade; empty result not yet relaxed→reflect_empty; granularity→repair;
        success→answer; generation failed→degrade; retries exhausted→degrade; errors→repair."""
        if state.get("hallucination"):
            return "degrade"
        if state.get("intent_mismatch") and state.get("retry_count", 0) < state.get("max_retries", MAX_RETRIES):
            return "repair"
        # empty-result reflection: only when "executed and returned []" and not yet relaxed
        if isinstance(state.get("execution_result"), list) \
                and len(state.get("execution_result")) == 0:
            if not state.get("_relaxed") and state.get("relax_attempts", 0) < MAX_RETRIES:
                return "reflect_empty"
            return "answer"
        if state.get("execution_result") is not None:
            return "answer"
        if state.get("semantic_query") is None:
            return "degrade"
        if state.get("retry_count", 0) >= state.get("max_retries", MAX_RETRIES):
            return "degrade"
        if state.get("errors"):
            return "repair"
        return "answer"

    def repair_node(self, state: AgentState) -> dict:
        """Error three-way classification. Rule ①: never patch SQL text directly — all fixes land at the
        semantic-query layer and recompile. Granularity mismatch isn't a compile error → skip classification."""
        # granularity mismatch: skip classification / try_repair, advance straight to reflect to add a dimension
        if state.get("intent_mismatch"):
            return {"retry_count": state.get("retry_count", 0) + 1}
        with self._span("repair", input={"n_errors": len(state.get("errors", []))}):
            errors = state.get("errors", [])
            err = errors[-1] if errors else "未知错误"
            sql = state.get("compiled_sql")
            cls = classify_error(err, sql or "")
            categories = list(state.get("error_categories", [])) + [cls.category]
            classifications = list(state.get("error_classifications", [])) + [
                {"category": cls.category, "entity": cls.entity,
                 "reason": cls.reason, "error": err[:200]}
            ]
            feedback = f"SQL 校验/执行失败：{err}"
            return {
                "error_categories": categories,
                "error_classifications": classifications,
                "error_feedback": feedback,
                "retry_count": state.get("retry_count", 0) + 1,
            }

    def reflect_empty_node(self, state: AgentState) -> dict:
        """Empty-result reflection: when execution returns [], let the LLM decide true-no-data vs relaxable.

        Two actions:
        - relax: deterministically widen the window (day → its month), set _relaxed and re-query.
        - confirm: that window truly has no data → honest degrade (answer renders "no data").
        """
        sq = state.get("semantic_query")
        if sq is None:
            # empty result without a query: degrade honestly (no reflection)
            return {"empty_result": True, "error_feedback": "无有效语义查询，按确认无数据降级"}
        with self._span("reflect_empty", input={"metric": sq.metric,
                                                "window": sq.window.value,
                                                "empty_result": True}):
            llm = self.p._get_llm(self._use_alt)
            system = (
                "你是电商数据分析 Agent 的空结果反思器。用户查询的语义查询执行后返回空结果（无匹配数据）。"
                "请判断这是「该窗口确实无数据（confirm）」还是「窗口过窄导致查空（relax）」。\n\n"
                "只输出合法 JSON，格式：\n"
                '{"action": "confirm"|"relax", "reason": "判断理由（<50字）"}\n'
                "注意：不要为了给出数字而放宽口径——放宽策略是确定的（单日→所在月），你只负责判定方向。"
            )
            user = (
                f"问题：{state['question']}\n"
                f"指标：{sq.metric}（{self.layer.metrics[sq.metric]['display_name']}）\n"
                f"窗口：{sq.window.type} {sq.window.value}\n"
                f"数据截止：{self.layer.context.get('visible_data_cutoff', '未知')}\n"
                "查询执行返回空结果。请判定 confirm 或 relax。"
            )
            try:
                raw = llm.complete(system, user)
                decision = _parse_reflect_empty(raw)
                if decision["action"] == "relax":
                    new_sq = _relax_window(sq)
                    if new_sq is None:
                        # month window is already widest → confirm no data
                        return {"empty_result": True,
                                "error_feedback": f"空结果反思：{sq.window.type} 窗口已最宽，确认无数据（{decision.get('reason', '')}）"}
                    return {"semantic_query": new_sq, "_relaxed": True,
                            "_reflect_fixed": True,  # pass new_sq straight to recompile, no LLM override
                            "relax_attempts": state.get("relax_attempts", 0) + 1,
                            "execution_result": None,
                            "empty_result": False,
                            "error_feedback": f"空结果反思：窗口放宽到 {new_sq.window.value}（{decision.get('reason', '')}）"}
                # confirm: no data → honest degrade
                return {"empty_result": True,
                        "error_feedback": f"空结果反思确认：{decision.get('reason', '该窗口无数据')}"}
            except BudgetExceeded:
                raise  # budget: pass to answer to degrade
            except Exception as e:
                # reflection failed: don't guess, degrade honestly
                return {"empty_result": True,
                        "error_feedback": f"空结果反思失败（{e}），按确认无数据降级"}

    def reflect_node(self, state: AgentState) -> dict:
        """Reflect by error type: granularity → feed back to add a dimension; reference/logic/dialect → LLM rewrite."""
        categories = state.get("error_categories", [])
        cat = categories[-1] if categories else "unknown"
        feedback = state.get("error_feedback", "")
        # granularity mismatch: question asks grouping/filtering but query has none → feed back to add a dim
        if state.get("intent_mismatch"):
            with self._span("reflect", output={"kind": "granularity",
                                               "reason": feedback[:150]}):
                llm = self.p._get_llm(self._use_alt)
                system = (
                    "你是电商数据分析 Agent 的纠错器。用户问题要求按维度分组或过滤，"
                    "但上一轮生成的语义查询没有任何维度/过滤（粒度错位）。"
                    "请根据下方「可用维度」清单，补上正确的维度/过滤后重写语义查询 JSON。\n\n"
                    f"可用维度：\n{format_available_dimensions(self.layer)}\n"
                    "只输出合法 JSON（metric/window/dimensions/filters），不要解释。"
                )
                user = f"问题：{state['question']}\n错位原因：{feedback}"
                try:
                    raw = llm.complete(system, user)
                    sq = self.p._parse_semantic_query(raw)
                    return {"semantic_query": sq, "_reflect_fixed": True,
                            "intent_mismatch": False,
                            "error_feedback": feedback + "\n（已补维度重写）"}
                except BudgetExceeded:
                    raise  # budget: pass to answer to degrade
                except Exception:
                    return {"intent_mismatch": False,
                            "error_feedback": feedback + "\n（补维度重写失败，降级）"}
        # reference/logic/dialect/unknown: LLM rewrites the query via the category prompt
        cls_records = state.get("error_classifications", [])
        reason = cls_records[-1].get("reason", "") if cls_records else ""
        with self._span("reflect", input={"feedback": feedback[:200], "category": cat,
                                          "reason": reason}):
            llm = self.p._get_llm(self._use_alt)
            metric = state["semantic_query"].metric if state.get("semantic_query") else ""
            error = state["errors"][-1] if state.get("errors") else ""
            system, user = build_repair_prompt(
                state["question"], metric, cat, error, self.layer, feedback)
            try:
                raw = llm.complete(system, user)
                sq = self.p._parse_semantic_query(raw)
                # rewrite succeeded: mark to skip generate, straight to tools with the fixed query
                return {"semantic_query": sq, "_reflect_fixed": True,
                        "error_feedback": feedback + "\n（已重写语义查询）"}
            except BudgetExceeded:
                raise  # budget: pass to answer to degrade
            except Exception:
                # rewrite failed: keep the original error feedback, let generate retry
                return {"error_feedback": feedback}

    def answer_node(self, state: AgentState) -> dict:
        with self._span("answer", output={"stage": state.get("stage")}):
            sq = state.get("semantic_query")
            # tool insight answer (stage 3-4E): produced directly by a diagnostic tool
            if state.get("tool_answer"):
                return {"answer": (
                    f"【诊断工具洞察】（{state.get('tool_used') or '工具'}）\n"
                    f"{state['tool_answer']}")}
            # empty result (executed and returned [], no execution error): honestly render "no data"
            empty = isinstance(state.get("execution_result"), list) \
                and len(state.get("execution_result")) == 0
            if not state.get("execution_error") and (state.get("empty_result") or empty):
                reason = state.get("error_feedback", "")
                window = sq.window.value if sq else ""
                relaxed_note = ("（已尝试放宽窗口，仍无数据）"
                                if state.get("_relaxed") else "（经空结果反思确认）")
                return {"answer": (
                    f"【{self.layer.metrics[sq.metric]['display_name'] if sq else '指标'}】"
                    f"{window} 无数据 {relaxed_note}\n"
                    f"{reason}\n"
                    "说明：查询如实执行，未做数据猜测或填充。")}
            # normal completion: query valid, no errors (execution_result may be None; _assemble_answer handles it)
            if sq is not None and not state.get("errors") and not state.get("execution_error") \
                    and not state.get("hallucination"):
                answer = self.p._assemble_answer(
                    sq, state.get("execution_result"),
                    state["retrieved_tables"], state.get("compiled_sql"))
                # soul risk ②: a relaxed caliber must be disclosed when yielding a number
                if state.get("_relaxed"):
                    note = state.get("error_feedback", "")
                    answer = (f"{answer}\n\n⚠ 口径说明：原始单日窗口无数据，"
                              f"已将窗口放宽到 {sq.window.value} 后给出当月数据。\n{note}")
                return {"answer": answer}
            # hallucination block: question is out of answerable range, return no wrong data
            if state.get("hallucination"):
                hint = (state.get("execution_error")
                        or "问题包含语义层无法表达的实体（如订单号/状态明细），无法回答")
                return {"answer": (
                    f"无法回答：该问题超出了当前可查询范围。\n"
                    f"原因：{hint}\n"
                    "建议：改为按月份/品类/渠道/城市等维度查询聚合指标（GMV/订单数/退款率等）。")}
            # cutoff block: data hasn't reached the visible boundary — honest, not silently empty
            if state.get("preflight_kind") == "cutoff":
                return {"answer": (
                    f"无法回答：查询窗口超出数据可见范围。\n"
                    f"原因：{state.get('preflight_reason') or state.get('execution_error', '')}\n"
                    "说明：数据仅更新到可见截止日，之后窗口无数据可查，不做猜测填充。")}
            # degrade: explicit note + tried SQL + error summary + manual-intervention advice
            err = (state.get("execution_error")
                   or (state["errors"][-1] if state.get("errors") else "未知错误"))
            error_summary = err[:300]
            sql = state.get("compiled_sql")
            categories = state.get("error_categories", [])
            cat_desc = "、".join(categories) if categories else "未分类"
            tried = (f"\n已尝试 SQL：\n{sql}" if sql else "\n（语义查询未编译成 SQL）")
            advice = (
                f"无法自动修复该问题（错误类别：{cat_desc}，已重试 {state.get('retry_count', 0)} 次）。\n"
                f"错误摘要：{error_summary}\n"
                f"{tried}\n"
                "建议：① 换个措辞重新提问（避免引用不支持的状态/字段）；"
                "② 若为数据问题，联系数仓管理员人工介入。"
            )
            return {"answer": f"无法回答：{advice}"}

    # ---------- graph construction ----------

    def _tool_answer_cond(self, state: AgentState) -> Literal["answer", "validate"]:
        """post-generate routing: tool insight answer (tool_answer non-empty) → answer; else validate."""
        if state.get("tool_answer"):
            return "answer"
        return "validate"

    def _reflect_empty_cond(self, state: AgentState) -> Literal["generate", "answer"]:
        """post-reflect_empty routing: relaxed window → re-query; confirm → honest degrade."""
        if state.get("_relaxed") and not state.get("empty_result"):
            return "generate"
        return "answer"

    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("intent", self.intent_node)
        g.add_node("retrieve", self.retrieve_node)
        g.add_node("generate", self.generate_node)
        g.add_node("tools", self.tools_node)
        g.add_node("validate", self.validate_node)
        g.add_node("judge", self.judge_node)            # pre-compile (after validate)
        g.add_node("judge_post", self.judge_post_node)  # post-compile (after tools)
        g.add_node("repair", self.repair_node)
        g.add_node("reflect", self.reflect_node)
        g.add_node("reflect_empty", self.reflect_empty_node)
        g.add_node("answer", self.answer_node)

        g.add_edge(START, "intent")
        g.add_edge("intent", "retrieve")
        g.add_edge("retrieve", "generate")
        # tool insight answer (stage 3-4E) goes straight to answer; else validate runs before compile
        g.add_conditional_edges(
            "generate", self._tool_answer_cond,
            {"answer": "answer", "validate": "validate"},
        )
        g.add_edge("validate", "judge")
        g.add_conditional_edges(
            "judge", self._judge_pre_cond,
            {"answer": "answer", "repair": "repair", "tools": "tools"},
        )
        g.add_edge("tools", "judge_post")
        g.add_conditional_edges(
            "judge_post", self._judge_cond,
            {"answer": "answer", "repair": "repair",
             "reflect_empty": "reflect_empty", "degrade": "answer"},
        )
        g.add_edge("repair", "reflect")
        g.add_edge("reflect", "generate")
        g.add_conditional_edges(
            "reflect_empty", self._reflect_empty_cond,
            {"generate": "generate", "answer": "answer"},
        )
        g.add_edge("answer", END)
        return g.compile(checkpointer=self._checkpointer)

    # ---------- entry ----------

    def answer(self, question: str, *, use_alt: bool = False,
               thread_id: str | None = None, history: list[dict] | None = None,
               on_reasoning=None, on_phase=None) -> dict:
        today = self.cfg.reference_date or date.today().isoformat()
        self._on_phase = on_phase  # per-question phase callback (chat streaming UI); None for eval/tests
        state: AgentState = {
            "question": question,
            "intent": "",
            "stage": "",
            "hallucination": False,
            "intent_mismatch": False,
            "preflight_kind": "pass",
            "preflight_reason": "",
            "empty_result": False,
            "_relaxed": False,
            "relax_attempts": 0,
            "schema_text": "",
            "retrieved_tables": [],
            "_retrieve_degraded": False,
            "semantic_query": None,
            "compiled_sql": None,
            "execution_result": None,
            "execution_error": None,
            "errors": [],
            "error_categories": [],
            "error_classifications": [],
            "error_feedback": "",
            "_reflect_fixed": False,
            "retry_count": 0,
            "max_retries": MAX_RETRIES,
            "tool_answer": "",
            "tool_used": "",
            "_tool_chart": "",
            "_reasoning": "",
            "messages": [],
            "history": history or [],
            "answer": "",
            "trace_id": "",
        }
        with self.tracer.trace(f"question: {question[:40]}") as t:
            self._t = t
            self._use_alt = use_alt
            t.set_trace_io(input={"question": question})
            config = {
                "configurable": {
                    # stable per-conversation thread_id retains the message channel across turns
                    "thread_id": thread_id or uuid.uuid4().hex[:12],
                    "on_reasoning": on_reasoning,  # live-thinking hook (chat UI); None for eval/tests
                },
                "recursion_limit": RECURSION_LIMIT,
            }
            monitor = self.p.begin_question()  # three-layer budget: rounds/tokens/timeout
            final = dict(state)  # fallback: safely accessible in finally on any exception path
            try:
                final = self._graph.invoke(state, config)
            except BudgetExceeded as e:
                # budget cut: honestly report exhaustion, no silent truncation
                err = f"单问预算耗尽被熔断：{e}"
                final = dict(state)
                final["execution_error"] = err
                final["errors"] = list(final.get("errors", [])) + [err]
                final["budget_exceeded"] = {"reason": e.reason, "used": e.used,
                                            "limit": e.limit}
                final["answer"] = (
                    f"无法回答：处理该问题超出预算上限（{e.reason}，已用 {e.used}，上限 {e.limit}）。\n"
                    "建议：缩小问题范围（如改为月粒度/减少追问轮次）后重试。")
                with self._span("judge", output={"stage": "degrade", "budget_exceeded": True}):
                    final["stage"] = "degrade"
            except GraphRecursionError as e:
                # recursion overrun: record the real error and degrade (Langfuse has the prior spans)
                err = f"递归深度超限（{RECURSION_LIMIT} 步，可能纠错循环未收敛）: {e}"
                final = dict(state)
                final["execution_error"] = err
                final["errors"] = list(final.get("errors", [])) + [err]
                final["answer"] = f"无法回答：{err}"
                with self._span("judge", output={"stage": "degrade", "recursion_exceeded": True}):
                    final["stage"] = "degrade"
            finally:
                final.setdefault("budget", monitor.snapshot())  # budget snapshot into final
                final = _jsonable_state(final)  # serialization guard: drop pydantic/message objects
                self.p.end_question()
                self._t = None
        final["trace_id"] = t.trace_id
        return final