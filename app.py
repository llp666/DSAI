"""app.py：Streamlit 单页 Demo（阶段 3-4F）。

用法：streamlit run app.py
- 输入自然语言问题 → 展示 Agent 答案 + 阶段/工具/预算元信息
- 工具轨（inventory_diagnostic）额外渲染 Plotly 图（DOI×环比散点象限图）
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="DSAI 电商数据分析 Agent", page_icon="📊", layout="wide")


@st.cache_resource
def get_pipeline():
    from orchestration.pipeline import Pipeline
    return Pipeline()


def render_meta(result: dict) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("stage", result.get("stage") or "-")
    c2.metric("工具", result.get("tool_used") or "-")
    c3.metric("LLM 调用", result.get("budget", {}).get("calls", 0))
    c4.metric("Token", result.get("budget", {}).get("tokens", 0))
    c5.metric("耗时 s", result.get("budget", {}).get("elapsed_s", 0))
    if result.get("preflight_kind") and result.get("preflight_kind") != "pass":
        st.caption(f"预检：{result.get('preflight_kind')} — {result.get('preflight_reason', '')[:80]}")
    if result.get("error_categories"):
        st.caption(f"错误分类：{', '.join(result.get('error_categories'))}")


def render_chart(result: dict) -> None:
    """工具轨结果含 _tool_chart（Plotly figure JSON）时渲染散点象限图。"""
    chart = result.get("_tool_chart")
    if not chart:
        return
    try:
        import plotly.io as pio

        fig = pio.from_json(chart)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.caption(f"（图表渲染失败：{e}）")


def main() -> None:
    st.title("DSAI 电商数据分析 Agent")
    st.caption("语义查询 DSL + 诊断工具（库存/营销漏斗）· LangGraph 状态机 · 三层熔断")
    question = st.text_input("问题", value="SKU-14167349 还够卖几天？",
                             placeholder="例如：哪些SKU断货？上个月GMV是多少？")
    use_alt = st.checkbox("备选 LLM 供应商", value=False)
    if st.button("运行", type="primary"):
        with st.spinner("Agent 处理中…"):
            p = get_pipeline()
            result = p.answer(question, use_alt=use_alt)
        st.markdown("---")
        render_meta(result)
        if result.get("tool_answer"):
            st.markdown("### 诊断工具洞察")
            st.markdown(result["answer"])
            render_chart(result)  # 工具 chart 已由工具轮提取进 state（_tool_chart）
        else:
            st.markdown("### 答案")
            st.markdown(result.get("answer", ""))
        with st.expander("诊断信息"):
            st.json({k: result.get(k) for k in
                     ["intent", "stage", "tool_used", "retrieved_tables",
                      "preflight_kind", "error_categories", "budget",
                      "compiled_sql", "trace_id"]})
    st.markdown("---")
    st.caption("示例问题：哪些SKU断货？| 上个月GMV是多少？| 各品类净销售额？"
               "| 搜索广告渠道的ROI是多少？| SKU-14167349 还够卖几天？")


if __name__ == "__main__":
    main()
