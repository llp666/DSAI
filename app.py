"""app.py：ChatGPT-style chat interface.

- sidebar: new-chat button + past-conversation list (conversation history)
- main: chat bubbles + bottom input; business questions go to the state machine,
  casual talk goes to a plain LLM conversation (see orchestration/dialogue.route_dialogue)
"""

from __future__ import annotations

import uuid

import streamlit as st

st.set_page_config(page_title="电商智能问数 Agent", page_icon="📊", layout="wide")


@st.cache_resource
def get_pipeline():
    from orchestration.pipeline import Pipeline

    return Pipeline()


def render_chart(result: dict) -> None:
    """Render the inventory scatter-quadrant chart when the result carries a _tool_chart."""
    chart = result.get("_tool_chart")
    if not chart:
        return
    try:
        import plotly.io as pio

        fig = pio.from_json(chart)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.caption(f"（图表渲染失败：{e}）")


def _conversation_history(messages: list[dict]) -> list[dict]:
    """Pair consecutive user→assistant turns into the agent's history [{question, answer}]."""
    history = []
    pending = None
    for m in messages:
        if m["role"] == "user":
            pending = m["content"]
        elif m["role"] == "assistant" and pending is not None:
            history.append({"question": pending, "answer": m["content"]})
            pending = None
    return history


def main() -> None:
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
    if "current" not in st.session_state:
        st.session_state.current = None

    p = get_pipeline()

    # ---- sidebar: new chat + conversation history ----
    with st.sidebar:
        st.markdown("## 电商智能问数 Agent")
        if st.button("＋ 新建对话", use_container_width=True):
            st.session_state.current = None
            st.rerun()
        st.divider()
        for conv_id in reversed(list(st.session_state.conversations.keys())):
            conv = st.session_state.conversations[conv_id]
            if st.button(conv["title"][:24], key=conv_id, use_container_width=True):
                st.session_state.current = conv_id
                st.rerun()

    # ---- main chat area ----
    conv = st.session_state.conversations.get(st.session_state.current)
    messages = conv["messages"] if conv else []

    for m in messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant" and m.get("result"):
                render_chart(m["result"])

    if question := st.chat_input("问我一件事，例如：2026年7月GMV是多少？"):
        if st.session_state.current is None or st.session_state.current not in st.session_state.conversations:
            conv_id = uuid.uuid4().hex[:12]
            st.session_state.conversations[conv_id] = {
                "title": question[:20],
                "thread_id": uuid.uuid4().hex[:12],
                "messages": [],
            }
            st.session_state.current = conv_id
        conv = st.session_state.conversations[st.session_state.current]
        history = _conversation_history(conv["messages"])

        conv["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            result = {}
            try:
                with st.spinner("Agent 处理中…"):
                    result, chunks = p.chat_stream(
                        question, history=history, thread_id=conv["thread_id"])
                full = st.write_stream(chunks)
            except Exception as e:
                full = f"（出错了：{e}）"
                st.markdown(full)
                result = {}
            render_chart(result)
            conv["messages"].append(
                {"role": "assistant", "content": full, "result": result})


if __name__ == "__main__":
    main()