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
from .prompt_budget import render_tables_budgeted
from .repair import try_repair
from .rewrite import rewrite
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
            """执行只读 SQL，返回首行首列结果（JSON 序列化，Decimal→float 保类型）。"""
            try:
                rows = executor.execute(sql)
            except Exception as e:
                raise RuntimeError(f"SQL 执行失败: {e}") from e
            value = rows[0][0] if rows else None
            return json.dumps(value, ensure_ascii=False,
                              default=lambda o: float(o) if isinstance(o, Decimal) else str(o))

        self._tools = [compile_tool, dry_run_tool, execute_tool]
        self._tool_node = ToolNode(self._tools, handle_tool_errors=True)
        self._compile_tool, self._dry_run_tool, self._execute_tool = self._tools

    def _call_tool(self, name: str, args: dict, msgs: list, config) -> tuple[str | None, str | None]:
        """通过 ToolNode 调用工具；错误包装成 ToolMessage，返回 (error, result)。"""
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
            if self._vector_store is not None:
                tables = self.retriever.retrieve_hybrid(
                    state["question"], self._vector_store, top_tables=3,
                    vector_n=10, reranker=self._reranker)
            else:
                tables = self.retriever.retrieve(state["question"])
            schema_text, _ = render_tables_budgeted(tables, RETRIEVAL_BUDGET)
            return {
                "retrieved_tables": [t["table"] for t in tables],
                "schema_text": schema_text,
            }

    def generate_node(self, state: AgentState) -> dict:
        with self._span("generate", input={"retry": state.get("retry_count")}):
            llm = self.p._get_llm(self._use_alt)
            today = self.cfg.reference_date or date.today().isoformat()
            question = state["question"]
            feedback = state.get("error_feedback", "")
            if feedback:
                question = (f"{question}\n\n【上次失败原因】\n{feedback}\n"
                            "请根据错误原因修正你的语义查询 JSON，只输出合法 JSON。")
            try:
                sq, raw, attempts = self.p._generate(question, state["schema_text"], llm, today)
            except Exception as e:
                return {"semantic_query": None, "execution_error": f"LLM 调用失败: {e}"}
            return {"semantic_query": sq, "execution_error": None}

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
                            try:
                                upd["execution_result"] = json.loads(res)
                            except json.JSONDecodeError:
                                upd["execution_result"] = res
            # add_messages reducer 会追加，只返回本轮新增消息
            return {**upd, "errors": errors, "messages": msgs[start:]}

    def judge_node(self, state: AgentState) -> dict:
        """judge 图节点：记录判定结果供 trace（条件路由在 _judge_cond）。"""
        stage = self._judge_cond(state)
        with self._span("judge", output={"stage": stage,
                                         "n_errors": len(state.get("errors", []))}):
            return {"stage": stage}

    def _judge_cond(self, state: AgentState) -> Literal["answer", "repair", "degrade"]:
        """四优先级：成功 → answer；LLM 生成失败 → degrade；重试耗尽 → degrade；有错误 → repair。"""
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
        with self._span("repair", input={"n_errors": len(state.get("errors", []))}):
            errors = state.get("errors", [])
            err = errors[-1] if errors else "未知错误"
            sql = state.get("compiled_sql")
            feedback = f"SQL 校验/执行失败：{err}"
            if sql:
                repaired = try_repair(err, sql)
                if repaired:
                    feedback += f"\n确定性修复已应用：\n{repaired}"
            return {
                "error_feedback": feedback,
                "retry_count": state.get("retry_count", 0) + 1,
            }

    def reflect_node(self, state: AgentState) -> dict:
        with self._span("reflect", input={"feedback": state.get("error_feedback", "")[:200]}):
            llm = self.p._get_llm(self._use_alt)
            system = (
                "你是电商数据分析 Agent 的纠错反思器。用户的问题经语义查询→SQL 链路失败，"
                "错误文本如下。请分析失败原因，给出修正后的语义查询要点（指标/过滤值/维度），"
                "不超过 3 句话。只输出分析，不要输出 JSON 或 SQL。"
            )
            user = f"问题：{state['question']}\n\n错误与修复上下文：\n{state.get('error_feedback', '')}"
            try:
                advice = llm.complete(system, user)
            except Exception as e:
                advice = f"（反思失败：{e}）"
            return {"error_feedback": state.get("error_feedback", "") + "\n反思建议：\n" + advice}

    def answer_node(self, state: AgentState) -> dict:
        with self._span("answer", output={"stage": state.get("stage")}):
            sq = state.get("semantic_query")
            # 正常完成：语义查询有效且无错误（execution_result 可为 None=无数据，_assemble_answer 处理）
            if sq is not None and not state.get("errors") and not state.get("execution_error"):
                answer = self.p._assemble_answer(
                    sq, state.get("execution_result"),
                    state["retrieved_tables"], state.get("compiled_sql"))
            else:
                err = (state.get("execution_error")
                       or (state["errors"][-1] if state.get("errors") else "未知错误"))
                answer = f"无法回答：{err}"
            return {"answer": answer}

    # ---------- 图构建 ----------

    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("intent", self.intent_node)
        g.add_node("retrieve", self.retrieve_node)
        g.add_node("generate", self.generate_node)
        g.add_node("tools", self.tools_node)
        g.add_node("judge", self.judge_node)
        g.add_node("repair", self.repair_node)
        g.add_node("reflect", self.reflect_node)
        g.add_node("answer", self.answer_node)

        g.add_edge(START, "intent")
        g.add_edge("intent", "retrieve")
        g.add_edge("retrieve", "generate")
        g.add_edge("generate", "tools")
        g.add_edge("tools", "judge")
        g.add_conditional_edges(
            "judge", self._judge_cond,
            {"answer": "answer", "repair": "repair", "degrade": "answer"},
        )
        g.add_edge("repair", "reflect")
        g.add_edge("reflect", "generate")
        g.add_edge("answer", END)
        return g.compile(checkpointer=self._checkpointer)

    # ---------- 入口 ----------

    def answer(self, question: str, *, use_alt: bool = False) -> dict:
        today = self.cfg.reference_date or date.today().isoformat()
        state: AgentState = {
            "question": question,
            "intent": "",
            "stage": "",
            "schema_text": "",
            "retrieved_tables": [],
            "semantic_query": None,
            "compiled_sql": None,
            "execution_result": None,
            "execution_error": None,
            "errors": [],
            "error_feedback": "",
            "retry_count": 0,
            "max_retries": MAX_RETRIES,
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
            try:
                final = self._graph.invoke(state, config)
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
                self._t = None
        final["trace_id"] = t.trace_id
        return final
