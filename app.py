"""app.py — Apple-style chat frontend for the e-commerce data agent.

Design: airy light canvas, frosted-glass sidebar, hairline borders,
gradient blue bubbles, floating capsule composer, rise/fade micro-motion.
Business logic (state machine / casual routing) is unchanged.
"""

from __future__ import annotations

import time
import uuid

import streamlit as st

st.set_page_config(
    page_title="电商智能问数 Agent",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = r"""
<style>
:root{
  --bg:#fbfbfd; --ink:#1d1d1f; --ink-2:#6e6e73; --ink-3:#86868b;
  --blue:#0071e3; --blue-2:#0a84ff;
  --hairline:rgba(0,0,0,.08);
  --shadow-card:0 1px 2px rgba(0,0,0,.04),0 18px 44px -18px rgba(0,0,0,.16);
  --ease:cubic-bezier(.2,.8,.2,1);
  --font:-apple-system,BlinkMacSystemFont,"SF Pro Display","SF Pro Text",
         "Helvetica Neue","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
}
html,body,[data-testid="stAppViewContainer"]{
  font-family:var(--font); color:var(--ink); -webkit-font-smoothing:antialiased;
}
[data-testid="stAppViewContainer"]{
  background-color:var(--bg);
  background-image:
    radial-gradient(720px 460px at 88% -8%, rgba(0,113,227,.07), transparent 60%),
    radial-gradient(640px 520px at -12% 110%, rgba(122,92,255,.06), transparent 60%);
}
.block-container{max-width:860px; padding:2.6rem 1.25rem 9rem}
[data-testid="stDecoration"],[data-testid="stToolbar"],#MainMenu,footer{display:none}
[data-testid="stHeader"]{
  background:rgba(251,251,253,.55);
  backdrop-filter:blur(18px) saturate(170%);
  -webkit-backdrop-filter:blur(18px) saturate(170%);
}
hr{border:none;border-top:1px solid var(--hairline)}
::-webkit-scrollbar{width:8px;height:8px}
::-webkit-scrollbar-thumb{background:rgba(0,0,0,.16);border-radius:99px}
::-webkit-scrollbar-thumb:hover{background:rgba(0,0,0,.28)}
@keyframes rise{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
@keyframes fade{from{opacity:0}to{opacity:1}}

/* ---------- sidebar: frosted glass ---------- */
[data-testid="stSidebar"]{
  background:rgba(255,255,255,.7);
  backdrop-filter:blur(24px) saturate(170%);
  -webkit-backdrop-filter:blur(24px) saturate(170%);
  border-right:1px solid var(--hairline);
}
.brand{font-size:1.05rem;font-weight:700;letter-spacing:-.01em;display:flex;
  align-items:center;gap:.55rem;margin:.2rem 0 1rem}
.brand-dot{width:12px;height:12px;border-radius:50%;
  background:conic-gradient(from 180deg,#0a84ff,#7a5cff,#ff375f,#0a84ff);
  box-shadow:0 0 12px rgba(10,132,255,.55)}
.side-label{font-size:.7rem;letter-spacing:.14em;text-transform:uppercase;
  color:var(--ink-3);font-weight:600;margin:1rem .2rem .4rem}
[data-testid="stSidebar"] .stButton button{
  width:100%;border:none;background:transparent;color:var(--ink);
  justify-content:flex-start;text-align:left;border-radius:14px;
  padding:.6rem .8rem;font-weight:500;transition:all .22s var(--ease);
}
[data-testid="stSidebar"] .stButton button p{
  overflow:hidden;white-space:nowrap;text-overflow:ellipsis;
}
[data-testid="stSidebar"] .stButton button:hover{background:rgba(0,0,0,.05);transform:translateX(2px)}
[data-testid="stSidebar"] .stButton button:active{transform:scale(.98)}
[data-testid="stSidebar"] .stButton button:disabled{
  opacity:1;cursor:default;color:var(--blue)!important;
  background:rgba(0,113,227,.10);font-weight:600;
}
.st-key-new_chat button{
  background:linear-gradient(120deg,#0a84ff,#0060df)!important;color:#fff!important;
  font-weight:600!important;justify-content:center!important;
  box-shadow:0 10px 24px -10px rgba(0,113,227,.6);
}
.st-key-new_chat button:hover{filter:brightness(1.07);transform:translateY(-1px)!important}

/* ---------- hero (empty state) ---------- */
.hero{text-align:center;padding:3.4rem 0 2.2rem;animation:fade .6s var(--ease) both}
.hero-chip{display:inline-flex;align-items:center;gap:.4rem;font-size:.8rem;font-weight:600;
  color:var(--ink-2);background:rgba(255,255,255,.85);border:1px solid var(--hairline);
  border-radius:999px;padding:.35rem .85rem}
.hero-title{font-size:clamp(2.4rem,6vw,3.6rem);line-height:1.1;font-weight:700;
  letter-spacing:-.02em;margin:1.1rem 0 .8rem}
.grad{background:linear-gradient(92deg,#0071e3 0%,#7a5cff 45%,#ff375f 100%);
  -webkit-background-clip:text;background-clip:text;color:transparent}
.hero-sub{color:var(--ink-2);font-size:1.05rem;max-width:32rem;margin:0 auto}

/* suggestion cards */
.st-key-sugg_0 button,.st-key-sugg_1 button,.st-key-sugg_2 button,.st-key-sugg_3 button{
  min-height:60px;white-space:pre-wrap;text-align:left;justify-content:flex-start;
  background:rgba(255,255,255,.92);border:1px solid var(--hairline);border-radius:18px;
  padding:.85rem 1rem;font-weight:600;color:var(--ink);
  box-shadow:0 1px 2px rgba(0,0,0,.03);
  transition:transform .25s var(--ease),box-shadow .25s var(--ease),border-color .25s;
}
.st-key-sugg_0 button:hover,.st-key-sugg_1 button:hover,
.st-key-sugg_2 button:hover,.st-key-sugg_3 button:hover{
  transform:translateY(-3px);border-color:rgba(0,113,227,.35);box-shadow:var(--shadow-card);
}

/* ---------- chat rows ---------- */
[data-testid="stChatMessage"]{animation:rise .45s var(--ease) both;background:transparent;padding:.35rem 0}
[data-testid="stChatMessageContent"]{padding:.85rem 1.15rem;border-radius:20px;line-height:1.65}
[data-testid="stMarkdownContainer"] p{margin-bottom:.35rem}

/* avatars: glyph hidden, redrawn via CSS */
[data-testid^="chatAvatarIcon"]{
  width:34px;height:34px;min-width:34px;border-radius:12px;font-size:0;
  display:flex;align-items:center;justify-content:center;
  box-shadow:0 8px 16px -8px rgba(0,0,0,.35);
}
[data-testid="chatAvatarIcon-assistant"]{background:#1d1d1f}
[data-testid="chatAvatarIcon-assistant"]::after{content:"\2726";font-size:15px;color:#7cb3ff}
[data-testid="chatAvatarIcon-user"]{background:linear-gradient(135deg,#0a84ff,#7a5cff)}
[data-testid="chatAvatarIcon-user"]::after{
  content:"";width:16px;height:16px;background:#fff;
  -webkit-mask:url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><circle cx="12" cy="8" r="4.2"/><path d="M4 20c1.2-4 4.2-6 8-6s6.8 2 8 6z"/></svg>') center/contain no-repeat;
  mask:url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><circle cx="12" cy="8" r="4.2"/><path d="M4 20c1.2-4 4.2-6 8-6s6.8 2 8 6z"/></svg>') center/contain no-repeat;
}

/* user bubble (right, blue gradient) */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]){flex-direction:row-reverse}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"]{
  width:fit-content;max-width:78%;margin-left:auto;
  background:linear-gradient(135deg,#0a84ff,#0068d8);color:#fff;
  border-radius:20px 20px 8px 20px;
  box-shadow:0 10px 26px -12px rgba(0,104,216,.55);
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] p{color:#fff}

/* assistant card (left, white) */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"]{
  width:fit-content;max-width:92%;background:#fff;
  border:1px solid var(--hairline);border-radius:20px 20px 20px 8px;
  box-shadow:var(--shadow-card);
}

/* plotly chart card */
[data-testid="stPlotlyChart"]{
  border:1px solid var(--hairline);border-radius:16px;overflow:hidden;
  background:#fff;box-shadow:0 10px 30px -18px rgba(0,0,0,.2);padding:6px;margin-top:.4rem;
}
[data-testid="stSpinner"]{color:var(--ink-2)}

/* thinking panel (Kimi-style collapsible) */
[data-testid="stExpander"]{
  border:1px solid var(--hairline);border-radius:14px;
  background:rgba(0,0,0,.02);box-shadow:none;margin:.3rem 0;
}
[data-testid="stExpander"] summary{font-size:.85rem;font-weight:600;color:var(--ink-2)}
@keyframes thinkPulse{0%,100%{opacity:1}50%{opacity:.42}}
/* live thinking panel: the open header (icon + label) blinks until the panel completes */
[data-testid="stExpander"] summary[aria-expanded="true"]{
  animation:thinkPulse 1.35s ease-in-out infinite;
}

/* ---------- composer: floating capsule ---------- */

[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"] {
  background: linear-gradient(180deg, rgba(251,251,253,0) 0%, var(--bg) 40%);
}
[data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"] > div {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}

/* 去掉外层灰框 */
[data-testid="stChatInput"] {
  position: relative !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
}

/* 唯一可见容器：白色胶囊 */
[data-testid="stChatInput"] > div {
  position: relative !important;
  background: rgba(255, 255, 255, 0.92) !important;
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid var(--hairline) !important;
  border-radius: 999px !important;
  box-shadow: 0 14px 40px -14px rgba(0, 0, 0, 0.22);
  min-height: 52px !important;
  display: flex !important;
  align-items: center !important;
  transition: box-shadow 0.25s var(--ease), border-color 0.25s var(--ease);
}
[data-testid="stChatInput"] > div:focus-within {
  border-color: rgba(10, 132, 255, 0.55) !important;
  box-shadow:
    0 0 0 4px rgba(10, 132, 255, 0.16),
    0 18px 44px -16px rgba(0, 0, 0, 0.24) !important;
}

[data-testid="stChatInput"] textarea {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  border-radius: 999px !important;
  min-height: 52px !important;
  padding: 14px 56px 14px 22px !important;
  font-size: 1rem !important;
  line-height: 1.4 !important;
}
[data-testid="stChatInput"] textarea:focus {
  border: none !important;
  box-shadow: none !important;
  outline: none !important;
}

/* 箭头：与胶囊同高，贴右侧内缘 */
button[data-testid="stChatInputSubmitButton"] {
  position: absolute !important;
  right: 18px !important;
  top: 50% !important;
  bottom: auto !important;
  transform: translateY(-50%) !important;

  width: 44px !important;
  height: 44px !important;
  min-width: 44px !important;
  min-height: 44px !important;
  max-height: 44px !important;

  background: linear-gradient(135deg, #0a84ff, #0060df) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 50% !important;
  box-shadow: 0 4px 12px -2px rgba(0, 113, 227, 0.45);
  z-index: 2;
  transition: transform 0.2s var(--ease), filter 0.2s, opacity 0.2s;
}
button[data-testid="stChatInputSubmitButton"]:hover {
  transform: translateY(-50%) scale(1.06) !important;
  filter: brightness(1.05);
}
button[data-testid="stChatInputSubmitButton"]:disabled {
  opacity: 0.4;
  filter: grayscale(0.15);
}

/* 按钮内图标居中 */
button[data-testid="stChatInputSubmitButton"] svg {
  width: 18px !important;
  height: 18px !important;
}

[data-testid="stIFrame"]{display:none}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

AVATARS = {"user": "🧑", "assistant": "✨"}

SUGGESTIONS = [
    ("📈", "2026年7月的 GMV 是多少？"),
    ("🏆", "上个月销量 Top 10 的商品"),
    ("📦", "哪些商品有库存积压风险？"),
    ("🧭", "你都能帮我做什么？"),
]


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


def _render_stream(chunks):
    """Consume tagged (reasoning, content) chunks → (answer_text, reasoning_text).

    Kimi-style: the model's chain-of-thought streams into a collapsible 「思考过程」
    panel that sits ABOVE the answer. The panel is a running st.status while thinking
    (spinner + pulsing header keep it alive across the thinking→answer gap), then
    collapses to a static ✓ once the answer arrives. Both streams render with a gentle
    typewriter cadence so they don't dump at once.
    """
    status = st.empty()
    status.caption("正在检索相关数据表…")
    think = None          # st.status container (lazy)
    think_box = None      # placeholder inside it (streams reasoning)
    think_text = ""
    answer = None         # lazy placeholder, created AFTER the thinking panel
    answer_text = ""
    for kind, chunk in chunks:
        if kind == "reasoning":
            if think is None:
                status.empty()
                think = st.status("💭 正在思考…", expanded=True)
                think_box = think.empty()
            think_text += chunk
            think_box.markdown(think_text)
            time.sleep(0.006)  # gentle cadence: smooth reveal, not a sudden dump
        elif kind == "content":
            if think is not None:
                think.update(label="💭 思考过程", state="complete", expanded=False)
            if answer is None:
                status.empty()
                answer = st.empty()
            answer_text += chunk
            answer.markdown(answer_text)
            time.sleep(0.004)
    return answer_text, think_text


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


def _select(conv_id: str) -> None:
    st.session_state.current = conv_id
    st.session_state._scroll = True


def _new_chat() -> None:
    st.session_state.current = None


def _suggest(text: str) -> None:
    st.session_state._suggested = text


def scroll_bottom() -> None:
    # st.iframe replaced st.components.v1.html (removed after 2026-06-01).
    st.iframe(
        """
        <script>
        (function () {
          var doc = window.parent.document;
          ['[data-testid="stAppViewContainer"]', '.main', 'section.main'].forEach(function (s) {
            var el = doc.querySelector(s);
            if (el) { el.scrollTo({top: el.scrollHeight, behavior: 'smooth'}); }
          });
          window.parent.scrollTo({top: doc.body.scrollHeight, behavior: 'smooth'});
        })();
        </script>
        """,
        height="content",
    )


def main() -> None:
    ss = st.session_state
    ss.setdefault("conversations", {})
    ss.setdefault("current", None)

    pipeline = get_pipeline()
    suggested = ss.pop("_suggested", None)

    # ---- sidebar: brand + new chat + history ----
    with st.sidebar:
        st.markdown(
            '<div class="brand"><span class="brand-dot"></span>电商智能问数</div>',
            unsafe_allow_html=True,
        )
        st.button("＋ 新建对话", key="new_chat", use_container_width=True, on_click=_new_chat)
        st.markdown('<div class="side-label">历史对话</div>', unsafe_allow_html=True)
        if not ss.conversations:
            st.caption("暂无历史对话")
        for conv_id in reversed(list(ss.conversations.keys())):
            conv = ss.conversations[conv_id]
            st.button(
                conv["title"][:24] or "新对话",
                key=f"conv_{conv_id}",
                use_container_width=True,
                disabled=(conv_id == ss.current),
                on_click=_select,
                args=(conv_id,),
            )

    conv = ss.conversations.get(ss.current)
    messages = conv["messages"] if conv else []

    # ---- hero + suggestions (empty state) ----
    if not messages and suggested is None:
        st.markdown(
            """
            <div class="hero">
              <div class="hero-chip">✦&nbsp;&nbsp;AI 数据问数助手</div>
              <div class="hero-title">把业务数据，<br><span class="grad">聊给你听。</span></div>
              <div class="hero-sub">GMV、销量、库存、趋势——用一句大白话提问，剩下的交给 Agent。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        cols = st.columns(2)
        for i, (icon, text) in enumerate(SUGGESTIONS):
            with cols[i % 2]:
                st.button(
                    f"{icon}  {text}",
                    key=f"sugg_{i}",
                    use_container_width=True,
                    on_click=_suggest,
                    args=(text,),
                )

    # ---- history ----
    for m in messages:
        with st.chat_message(m["role"], avatar=AVATARS[m["role"]]):
            if m["role"] == "assistant" and m.get("reasoning"):
                # thinking panel sits ABOVE the answer (Kimi-style)
                with st.status("💭 思考过程", state="complete", expanded=False):
                    st.markdown(m["reasoning"])
            st.markdown(m["content"])
            if m["role"] == "assistant" and m.get("result"):
                render_chart(m["result"])

    if ss.pop("_scroll", None):
        scroll_bottom()

    # ---- composer ----
    prompt = st.chat_input("问点什么，例如：2026年7月GMV是多少？") or suggested

    if prompt:
        if ss.current is None or ss.current not in ss.conversations:
            conv_id = uuid.uuid4().hex[:12]
            ss.conversations[conv_id] = {
                "title": prompt[:20],
                "thread_id": uuid.uuid4().hex[:12],
                "messages": [],
            }
            ss.current = conv_id
        conv = ss.conversations[ss.current]
        messages = conv["messages"]

        history = _conversation_history(messages)
        messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar=AVATARS["user"]):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=AVATARS["assistant"]):
            result_box: dict = {}
            reasoning = ""
            try:
                _, chunks = pipeline.chat_stream(
                    prompt, history=history, thread_id=conv["thread_id"], result_box=result_box
                )
                full, reasoning = _render_stream(chunks)
                result = result_box.get("result", {})
            except Exception as e:
                full = f"（出错了：{e}）"
                st.markdown(full)
                result = {}
            render_chart(result)

        messages.append({"role": "assistant", "content": full,
                         "reasoning": reasoning, "result": result})
        ss._scroll = True
        st.rerun()


if __name__ == "__main__":
    main()
