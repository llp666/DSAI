"""agent/graph.py：阶段三 LangGraph 状态机（九节点 + SQL 工具 + 纠错轨）。

把阶段二 pipeline 的函数链迁为 StateGraph 状态机：
- 主链：intent → retrieve → generate → tools → judge → answer
- judge 三分类条件边（四优先级判定）：
    成功 → answer（正常组装）
    可修复 → repair（确定性规则）→ reflect（LLM 反思）→ generate（同轮重试，≤max_retries）
    不可修复 → answer（降级回答）
- tools 节点 = ToolNode([compile, dry_run, execute], handle_tool_errors=True)，
  工具异常包装为 ToolMessage「Error: …」真实错误文本回灌给 repair/reflect/generate。

状态与组件通过 Pipeline 注入（graph.py 不 import pipeline.py，避免循环）。
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

from .compiler import CompileError, compile_query
from .error_classifier import build_repair_prompt, classify_error, format_available_dimensions
from .monitor import BudgetExceeded
from .preflight import preflight
from .prompt_budget import render_tables_budgeted
from .prompts import build_system_prompt
from .relevance import check_relevance
from .rewrite import rewrite
from .tool_contract import ToolError, serialize_result
from .tool_router import route_tool_intent
from .tools_inventory import inventory_diagnostic_tool
from .tools_marketing import marketing_roi_funnel_tool
from .types import AgentState, SemanticQuery

# 检索注入 Token 预算（agnes maxInput 128k 的 1/8，预留指令/回答余量）
RETRIEVAL_BUDGET = 16000
MAX_RETRIES = 2
# 递归上限：主链 5 步 + 每轮 repair 5 步 × (MAX_RETRIES+1) + 余量。超限抛 GraphRecursionError，
# 由 answer() 捕获降级而非静默截断（配合 checkpointer 在 Langfuse 看完整路径）
RECURSION_LIMIT = 5 * (MAX_RETRIES + 1) + 5


class _NullCtx:
    """未启用 trace 时的空上下文（与 Langfuse span 兼容）。"""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def update(self, **kw):
        pass

# 意图/域分类（关键词 → 域标签；顺序匹配，先命中先得）
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
    """解析 execute_tool 返回的 JSON 行数组：单行单列→标量，多行→list[list]。"""
    try:
        rows = json.loads(res)
    except json.JSONDecodeError:
        return res
    if isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], list) and len(rows[0]) == 1:
        return rows[0][0]
    return rows


def _is_empty(result) -> bool:
    """空结果判定：None 或空列表（rank/snapshot 类指标无数据时返回 []）。"""
    return result is None or (isinstance(result, list) and len(result) == 0)


def _looks_like_json(text: str) -> bool:
    """粗判 LLM 输出是否像 JSON（语义查询对象）。"""
    t = text.strip()
    return t.startswith("{") or t.startswith("```json") or t.startswith("```")


def _jsonable_state(final: dict) -> dict:
    """③ 序列化防御：剔除 state 中不消费且不可 JSON 化的消息对象。

    LangGraph 把 AIMessage/ToolMessage 对象累积进 state.messages，app/eval 均不消费；
    一旦 telemetry/日志整体 json.dumps(state) 会因消息对象序列化失败。只剔除 messages。
    semantic_query 是 pydantic 模型，虽非 JSON 标量但被 eval（run_eval/run_injection）
    按模型消费（.model_dump()/.dimensions/.window.value），保留原样。
    chart 已在上游固化为 JSON 字符串（_tool_chart），全链无 go.Figure 对象出圈。
    """
    return {k: v for k, v in final.items() if k != "messages"}


def _build_tool_user(question: str, tool_result: str | None) -> str:
    """构造工具调用轮的 user 消息：首轮只给问题；后续把工具结果/错误一并给 LLM 收敛。

    工具成功 → 基于结果给自然语言洞察；工具错误 → 提示改用语义查询 JSON 走指标链路。
    """
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
    """解析 reflect_empty 的 LLM 输出：{"action": "confirm"|"relax", "reason": "..."}。

    只接受 confirm/relax 二选一，不接受 LLM 提供的 new_window——
    放宽窗口是确定性的（见 reflect_empty_node），防止 LLM 为出数悄悄改口径。
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
    """确定性放宽窗口：单日 → 该日所在月；month 已最宽 → 返回 None（不放宽）。

    放宽是确定性规则而非 LLM 自选，杜绝「为出数悄悄改口径」——
    只允许「单日查空 → 看当月整体」这一种合理放宽，且保持 metric/dimensions/filters 不变。
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
        self._t = None  # 当前 trace session（answer() 入口设置，节点 span 用）
        self._use_alt = False
        self._checkpointer = MemorySaver()
        self._build_tools()
        self._graph = self._build_graph()

    # ---------- SQL 三工具（@tool + ToolNode 错误包装） ----------

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
        """调用工具；错误包装成 ToolMessage「Error: …」回灌纠错轨，返回 (error, result)。

        诊断工具（inventory_tool/marketing_tool）直接调用底层函数（绕过 ToolNode 的
        checkpointer config 依赖）；SQL 三工具走 ToolNode（图内 tools_node 复用）。
        """
        if name in ("inventory_tool", "marketing_tool"):
            try:
                res = (inventory_diagnostic_tool(self.executor, args) if name == "inventory_tool"
                       else marketing_roi_funnel_tool(self.executor, args))
                content = serialize_result(res)
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

    # ---------- 诊断工具可选调用（阶段 3-4E） ----------

    def _tool_schemas(self) -> list[dict]:
        """OpenAI 兼容的工具定义（给 LLM 可选调用）。

        手工构造（不用 convert_to_openai_function）：`args: dict` 会被 langchain
        转成 v__args 列表，丢失内部 JSON Schema。这里直接从工具模块的入参
        JSON Schema 构造，保证 LLM 看到正确的参数定义。
        """
        from .tools_inventory import INVENTORY_ARGS_SCHEMA
        from .tools_marketing import MARKETING_ARGS_SCHEMA
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
        ]

    _TOOL_MAX_ROUNDS = 2  # generate 内工具调用循环上限（防 agnes 重复 tool_calls 烧 token）

    def _tool_system_prompt(self, state: AgentState) -> str:
        """工具轮 system prompt：语义查询 JSON 契约打底 + 工具调用规则。

        基础 prompt（build_system_prompt）已含指标口径/值字典/JSON 契约——
        LLM 在工具报错或不足时能回落到合法语义查询 JSON，不编造数字。
        """
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
            "- 调用工具后，基于工具返回的真实结果，用自然语言输出洞察（可引用 sku_id/天数/环比/环节转化率）。\n"
            "- 工具报错/未实现/不足以回答 → 放弃工具，改输出下方「输出格式」的语义查询 JSON（引用已定义指标）。\n"
            "- 没有任何真实数据来源时禁止编造数字；无法回答就明确说明，绝不猜数。\n\n"
            f"{base}"
        )

    @staticmethod
    def _extract_chart(res: str) -> str:
        """从工具返回 JSON 提取 Plotly figure JSON（无图返回空串，供 app 渲染）。"""
        try:
            payload = json.loads(res)
            chart = payload.get("chart") or {}
            if chart.get("kind") == "plotly_figure_json":
                return chart.get("figure_json", "")
        except Exception:
            pass
        return ""

    def _run_tool_round(self, state: AgentState, llm, config) -> tuple[dict, str | None, SemanticQuery | None]:
        """一轮「LLM 可选调工具」：返回 (state 更新 dict, 洞察答案或 None, 语义查询或 None)。

        流程：带 tools 调 LLM → 若返回 tool_calls 则执行工具（去重）→ 把 ToolResult 回灌 LLM →
        LLM 收敛输出最终答案（洞察式）或语义查询 JSON。工具错误经 ToolMessage 回灌纠错轨。

        返回值二选一（按优先级）：
        - tool_answer 非 None → 工具洞察答案直接产出（走 answer，跳过语义查询链路）
        - sq 非 None → 工具轮已收敛出合法语义查询 JSON（generate_node 直接用，不再二次生成）
        两者皆 None → 工具轮未产出（应只发生在未进工具轮，调用方无需处理）。
        """
        msgs = list(state.get("messages", []))
        start = len(msgs)
        schemas = self._tool_schemas()
        used_tool: str | None = None
        tool_answer: str | None = None
        sq: SemanticQuery | None = None
        upd: dict = {"messages": msgs[start:]}
        for round_i in range(self._TOOL_MAX_ROUNDS):
            content, tool_calls = llm.complete_with_tools(
                self._tool_system_prompt(state),
                _build_tool_user(state["question"], tool_answer), schemas)
            if not tool_calls:
                # LLM 未调工具 → 输出是语义查询 JSON 或洞察答案文本
                if _looks_like_json(content):
                    try:
                        sq = self.p._parse_semantic_query(content)
                    except Exception:
                        tool_answer = content  # 不是合法 JSON → 当洞察答案
                    else:
                        if used_tool:
                            upd["tool_used"] = used_tool
                        return upd, None, sq  # 语义查询链路（工具信息已记录）
                else:
                    # 纯文本 → 洞察答案
                    tool_answer = content or tool_answer
                break
            # 有 tool_calls：去重后执行
            seen: set = set()
            last_res: str | None = None
            for tc in tool_calls:
                name = tc["name"]
                key = (name, json.dumps(tc.get("arguments", {}), sort_keys=True))
                if key in seen:
                    continue  # agnes 偶发重复 tool_calls → 只执行一次
                seen.add(key)
                used_tool = name
                err, res = self._call_tool(name, tc["arguments"], msgs, config)
                last_res = f"工具调用失败：{err}" if err else res
                if err is None and res:
                    upd["_tool_chart"] = self._extract_chart(res)
            tool_answer = last_res  # 完整 ToolResult JSON 或错误文本回灌下一轮收敛
        if used_tool:
            upd["tool_used"] = used_tool
        upd["messages"] = msgs[start:]
        # 工具调用失败且 LLM 未收敛 → 回落到语义查询链路（不让原始错误当最终答案）
        if tool_answer and (tool_answer.startswith("工具调用失败") or tool_answer.startswith("Error:")):
            return upd, None, None
        return upd, tool_answer, sq

    # ---------- 节点 ----------

    def _span(self, name: str, **kw):
        """包一层 Langfuse span（未启用时 Noop）。"""
        if self._t is not None:
            return self._t.span(name, **kw)
        return _NullCtx()

    def intent_node(self, state: AgentState) -> dict:
        ref = date.fromisoformat(self.cfg.reference_date) if self.cfg.reference_date else date.today()
        q = rewrite(state["question"], ref)
        intent = _classify_intent(q)
        return {"question": q, "intent": intent}

    def retrieve_node(self, state: AgentState) -> dict:
        with self._span("retrieve", output={"intent": state.get("intent")}):
            degraded = False
            tables = []
            if self._vector_store is not None:
                try:
                    tables = self.retriever.retrieve_hybrid(
                        state["question"], self._vector_store, top_tables=3,
                        vector_n=10, reranker=self._reranker)
                except Exception as e:
                    # embedding 网络故障降级：纯关键词检索（记录进 trace，不崩溃）
                    degraded = True
                    tables = self.retriever.retrieve_keyword_only(state["question"])
            if not tables:
                tables = self.retriever.retrieve(state["question"])  # 最终回退硬编码
            schema_text, _ = render_tables_budgeted(tables, RETRIEVAL_BUDGET)
            return {
                "retrieved_tables": [t["table"] for t in tables],
                "schema_text": schema_text,
                "_retrieve_degraded": degraded,
            }

    def generate_node(self, state: AgentState, config=None) -> dict:
        # reflect 已重写语义查询：透传（不再调 LLM 覆盖修复结果），直接进 tools 重编译验证
        if state.get("_reflect_fixed"):
            return {"execution_error": None, "_reflect_fixed": False}
        with self._span("generate", input={"retry": state.get("retry_count")}):
            llm = self.p._get_llm(self._use_alt)
            # 工具可选调用轮（阶段 3-4E）：仅当意图路由命中诊断域（库存/营销漏斗）才进工具轮，
            # 避免普通问题多一次带 tools 的 LLM 调用（token 开销 / 误触发）
            upd: dict = {}
            tool_answer: str | None = None
            sq: SemanticQuery | None = None
            intent = route_tool_intent(state["question"])
            if intent.hit or intent.needs_llm:
                upd, tool_answer, sq = self._run_tool_round(state, llm, config)
            if tool_answer is not None:
                # 工具洞察答案：直接走 answer，跳过语义查询链路
                return {**upd, "tool_answer": tool_answer, "semantic_query": None,
                        "execution_error": None, "stage": "tool"}
            if sq is not None:
                # 工具轮已收敛出合法语义查询 JSON：直接进 validate/tools，不再二次生成
                return {**upd, "semantic_query": sq, "execution_error": None}
            today = self.cfg.reference_date or date.today().isoformat()
            question = state["question"]
            feedback = state.get("error_feedback", "")
            if feedback:
                question = (f"{question}\n\n【上次失败原因】\n{feedback}\n"
                            "请根据错误原因修正你的语义查询 JSON，只输出合法 JSON。")
            try:
                sq, raw, attempts = self.p._generate(question, state["schema_text"], llm, today)
            except BudgetExceeded:
                raise  # 熔断：透传让 answer 捕获降级，不吞成普通失败
            except Exception as e:
                return {"semantic_query": None, "execution_error": f"LLM 调用失败: {e}"}
            return {**upd, "semantic_query": sq, "execution_error": None}

    def tools_node(self, state: AgentState, config) -> dict:
        with self._span("tools", input={"n_errors": len(state.get("errors", []))}):
            msgs = list(state.get("messages", []))
            start = len(msgs)
            errors = list(state.get("errors", []))
            upd: dict = {}
            sq = state.get("semantic_query")
            if sq is not None:
                # 1) 编译
                err, sql = self._call_tool("compile_tool",
                                           {"semantic_query": sq.model_dump()}, msgs, config)
                if err:
                    errors.append(err)
                else:
                    upd["compiled_sql"] = sql
                    # 2) 干跑
                    err, _ = self._call_tool("dry_run_tool", {"sql": sql}, msgs, config)
                    if err:
                        errors.append(err)
                    else:
                        # 3) 执行
                        err, res = self._call_tool("execute_tool", {"sql": sql}, msgs, config)
                        if err:
                            errors.append(err)
                        else:
                            upd["execution_result"] = _parse_result(res)
            # add_messages reducer 会追加，只返回本轮新增消息
            return {**upd, "errors": errors, "messages": msgs[start:]}

    def validate_node(self, state: AgentState) -> dict:
        """前置校验（编译前）：四层预检 + 相关性/意图校验。

        预检（preflight）：日期边界/枚举字典/粒度回退/时效 cutoff——确定性拦截可预见的失败；
        相关性（check_relevance）：DSL 外实体 → hallucination（降级）；粒度错位 → intent_mismatch（repair）。
        全部通过才进 tools 编译执行，避免无效 LLM 重试与无效编译。
        """
        q = state.get("question", "")
        sq = state.get("semantic_query")
        if sq is None:
            return {"hallucination": False, "intent_mismatch": False,
                    "preflight_kind": "pass", "preflight_reason": ""}
        # 第一层：四层预检（确定性，零 LLM）
        cutoff = self.layer.context.get("visible_data_cutoff")
        try:
            verdict = preflight(sq, self.layer, cutoff=cutoff)
        except Exception as e:
            # 预检异常不崩溃：降级拦截（诚实告知），交由降级回答
            return {"hallucination": False, "intent_mismatch": False,
                    "preflight_kind": "cutoff",
                    "preflight_reason": f"预检异常：{e}",
                    "execution_error": f"预检异常：{e}"}
        if not verdict.ok:
            with self._span("validate", output={"kind": verdict.kind,
                                                "reason": verdict.reason[:120]}):
                if verdict.kind == "cutoff":
                    # 窗口超出可见数据截止：拦截降级（不静默查空）
                    return {"hallucination": False, "intent_mismatch": False,
                            "preflight_kind": "cutoff",
                            "preflight_reason": verdict.reason,
                            "execution_error": verdict.reason}
                if verdict.kind == "granularity":
                    # 维度/指标组合非法：回灌修复（repair 轨）
                    return {"hallucination": False, "intent_mismatch": True,
                            "preflight_kind": "granularity",
                            "preflight_reason": verdict.reason,
                            "error_feedback": verdict.reason + " " + verdict.suggestion}
                # date / enum：值/格式非法 → 回灌修复
                return {"hallucination": False, "intent_mismatch": True,
                        "preflight_kind": verdict.kind,
                        "preflight_reason": verdict.reason,
                        "error_feedback": verdict.reason + " " + verdict.suggestion}
        # 第二层：相关性/意图校验
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
        """judge 图节点（validate 后·编译前）：预检/相关性判定，供条件路由。"""
        stage = self._judge_pre_cond(state)
        with self._span("judge", output={"stage": stage,
                                         "preflight": state.get("preflight_kind")}):
            return {"stage": stage}

    def judge_post_node(self, state: AgentState) -> dict:
        """judge 图节点（tools 后·编译执行后）：执行结果判定。"""
        stage = self._judge_cond(state)
        with self._span("judge", output={"stage": stage,
                                         "n_errors": len(state.get("errors", []))}):
            return {"stage": stage}

    def _judge_pre_cond(self, state: AgentState) -> Literal["answer", "repair", "tools"]:
        """validate 后（编译前）路由：幻觉/预检 cutoff → 降级；粒度错位/预检可修 → repair；通过 → tools。"""
        if state.get("hallucination"):
            return "answer"
        if state.get("preflight_kind") == "cutoff":
            return "answer"
        if state.get("intent_mismatch") or state.get("preflight_kind") in ("date", "enum", "granularity"):
            return "repair"
        return "tools"

    def _judge_cond(self, state: AgentState) -> Literal["answer", "repair", "degrade", "reflect_empty"]:
        """tools 后（编译执行后）判定优先级：
        幻觉→degrade；执行过返回空列表且未放宽→reflect_empty；粒度错位→repair；
        成功→answer；生成失败→degrade；重试耗尽→degrade；有错→repair。"""
        if state.get("hallucination"):
            return "degrade"
        if state.get("intent_mismatch") and state.get("retry_count", 0) < state.get("max_retries", MAX_RETRIES):
            return "repair"
        # 空结果反思：仅当「执行过且返回空列表」且未放宽过才触发（None=未执行，走后续判定）
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
        """错误三分类分级。坑点①铁律：严禁直接补丁 SQL 文本——所有修复落语义查询层再重编译。
        粒度错位（intent_mismatch）不属编译错，跳过错误分类，直接计重试次数。"""
        # 粒度错位：不分类、不 try_repair，直接推进到 reflect 补维度
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
        """空结果反思：查询执行返回空结果（[]）时，让 LLM 判定是真无数据还是窗口可放宽。

        返回两种动作：
        - relax：确定性放宽窗口（单日 → 该日所在月）重查，设 _relaxed 重查。
          放宽是确定性规则而非 LLM 自选窗口，杜绝「为出数悄悄改口径」——
          LLM 只判二选一，不提供 new_window。
        - confirm：确认该窗口确实无数据 → 诚实降级（answer 渲染「无数据」，不猜测填充）
        """
        sq = state.get("semantic_query")
        if sq is None:
            # 无语义查询却空结果：直接诚实降级（不反思）
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
                        # 月窗口已是最宽语义，不放宽 → 确认无数据
                        return {"empty_result": True,
                                "error_feedback": f"空结果反思：{sq.window.type} 窗口已最宽，确认无数据（{decision.get('reason', '')}）"}
                    return {"semantic_query": new_sq, "_relaxed": True,
                            "_reflect_fixed": True,  # 透传 new_sq 直接重编译，不让 LLM 覆盖
                            "relax_attempts": state.get("relax_attempts", 0) + 1,
                            "execution_result": None,
                            "empty_result": False,
                            "error_feedback": f"空结果反思：窗口放宽到 {new_sq.window.value}（{decision.get('reason', '')}）"}
                # confirm：确认无数据，诚实降级
                return {"empty_result": True,
                        "error_feedback": f"空结果反思确认：{decision.get('reason', '该窗口无数据')}"}
            except BudgetExceeded:
                raise  # 熔断：透传给 answer 降级
            except Exception as e:
                # 反思失败：不猜测，诚实降级
                return {"empty_result": True,
                        "error_feedback": f"空结果反思失败（{e}），按确认无数据降级"}

    def reflect_node(self, state: AgentState) -> dict:
        """按错误类型反思：粒度错位→回灌补维度；引用/逻辑/方言→LLM 重写语义查询 JSON。"""
        categories = state.get("error_categories", [])
        cat = categories[-1] if categories else "unknown"
        feedback = state.get("error_feedback", "")
        # 粒度错位：问题要求分组/过滤但查询无维度 → 回灌让 LLM 补合法维度
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
                    raise  # 熔断：透传给 answer 降级
                except Exception:
                    return {"intent_mismatch": False,
                            "error_feedback": feedback + "\n（补维度重写失败，降级）"}
        # 引用/逻辑/方言/未知错：LLM 按分类 prompt 直接重写语义查询 JSON
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
                # 重写成功：标记跳过 generate，直接进 tools 用修复后的语义查询重编译验证
                return {"semantic_query": sq, "_reflect_fixed": True,
                        "error_feedback": feedback + "\n（已重写语义查询）"}
            except BudgetExceeded:
                raise  # 熔断：透传给 answer 降级
            except Exception:
                # 重写失败：保留原错误反馈，让 generate 自行尝试修正
                return {"error_feedback": feedback}

    def answer_node(self, state: AgentState) -> dict:
        with self._span("answer", output={"stage": state.get("stage")}):
            sq = state.get("semantic_query")
            # 工具洞察答案（阶段 3-4E）：诊断工具直接产出，跳过语义查询链路
            if state.get("tool_answer"):
                return {"answer": (
                    f"【诊断工具洞察】（{state.get('tool_used') or '工具'}）\n"
                    f"{state['tool_answer']}")}
            # 空结果（执行过返回空列表，且无执行错误）：诚实渲染「无数据」，不猜测填充
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
            # 正常完成：语义查询有效且无错误（execution_result 可为 None=无数据，_assemble_answer 处理）
            if sq is not None and not state.get("errors") and not state.get("execution_error") \
                    and not state.get("hallucination"):
                answer = self.p._assemble_answer(
                    sq, state.get("execution_result"),
                    state["retrieved_tables"], state.get("compiled_sql"))
                # ②灵魂风险：放宽口径出数必须披露——单日查空放宽到当月拿到数时，答案明示口径变化
                if state.get("_relaxed"):
                    note = state.get("error_feedback", "")
                    answer = (f"{answer}\n\n⚠ 口径说明：原始单日窗口无数据，"
                              f"已将窗口放宽到 {sq.window.value} 后给出当月数据。\n{note}")
                return {"answer": answer}
            # 幻觉拦截：明确提示问题超出可答范围（意图错位），不返回错数据
            if state.get("hallucination"):
                hint = (state.get("execution_error")
                        or "问题包含语义层无法表达的实体（如订单号/状态明细），无法回答")
                return {"answer": (
                    f"无法回答：该问题超出了当前可查询范围。\n"
                    f"原因：{hint}\n"
                    "建议：改为按月份/品类/渠道/城市等维度查询聚合指标（GMV/订单数/退款率等）。")}
            # 时效 cutoff 拦截：数据未到可见边界，诚实说明而非静默查空
            if state.get("preflight_kind") == "cutoff":
                return {"answer": (
                    f"无法回答：查询窗口超出数据可见范围。\n"
                    f"原因：{state.get('preflight_reason') or state.get('execution_error', '')}\n"
                    "说明：数据仅更新到可见截止日，之后窗口无数据可查，不做猜测填充。")}
            # 降级：明确提示 + 已试 SQL + 错误摘要 + 建议人工介入
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

    # ---------- 图构建 ----------

    def _tool_answer_cond(self, state: AgentState) -> Literal["answer", "validate"]:
        """generate 后路由：工具洞察答案（tool_answer 非空）→ 直达 answer；否则进 validate 预检。"""
        if state.get("tool_answer"):
            return "answer"
        return "validate"

    def _reflect_empty_cond(self, state: AgentState) -> Literal["generate", "answer"]:
        """reflect_empty 后路由：已放宽窗口 → 重查；confirm → 诚实降级回答。"""
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
        g.add_node("judge", self.judge_node)            # 编译前（validate 后）
        g.add_node("judge_post", self.judge_post_node)  # 编译后（tools 后）
        g.add_node("repair", self.repair_node)
        g.add_node("reflect", self.reflect_node)
        g.add_node("reflect_empty", self.reflect_empty_node)
        g.add_node("answer", self.answer_node)

        g.add_edge(START, "intent")
        g.add_edge("intent", "retrieve")
        g.add_edge("retrieve", "generate")
        # 工具洞察答案（阶段 3-4E）直达 answer；否则 validate 前置到编译前
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

    # ---------- 入口 ----------

    def answer(self, question: str, *, use_alt: bool = False) -> dict:
        today = self.cfg.reference_date or date.today().isoformat()
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
            "messages": [],
            "answer": "",
            "trace_id": "",
        }
        with self.tracer.trace(f"question: {question[:40]}") as t:
            self._t = t
            self._use_alt = use_alt
            t.set_trace_io(input={"question": question})
            config = {
                "configurable": {
                    "thread_id": f"{uuid.uuid4().hex[:12]}",  # 每问独立会话，checkpointer 存档
                },
                "recursion_limit": RECURSION_LIMIT,
            }
            monitor = self.p.begin_question()  # 三层熔断：轮次/token/超时
            final = dict(state)  # 兜底：任何异常路径下 finally 均可安全访问
            try:
                final = self._graph.invoke(state, config)
            except BudgetExceeded as e:
                # 预算熔断：诚实告知预算耗尽，不静默截断（trace 已含此前各节点 span）
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
                # 递归超限不静默：记录真实错误并降级回答（Langfuse 已含此前各节点 span）
                err = f"递归深度超限（{RECURSION_LIMIT} 步，可能纠错循环未收敛）: {e}"
                final = dict(state)
                final["execution_error"] = err
                final["errors"] = list(final.get("errors", [])) + [err]
                final["answer"] = f"无法回答：{err}"
                with self._span("judge", output={"stage": "degrade", "recursion_exceeded": True}):
                    final["stage"] = "degrade"
            finally:
                final.setdefault("budget", monitor.snapshot())  # _timed：预算快照进 final
                final = _jsonable_state(final)  # ③ 序列化闸：剔除 pydantic/消息对象
                self.p.end_question()
                self._t = None
        final["trace_id"] = t.trace_id
        return final
