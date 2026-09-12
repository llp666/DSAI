"""app.py — Apple-style chat frontend for the e-commerce data agent.

Design: airy light canvas, frosted-glass sidebar, hairline borders,
gradient blue bubbles, floating capsule composer, rise/fade micro-motion.
Business logic (state machine / casual routing) is unchanged.
"""

from __future__ import annotations

import threading
import time
import uuid

import streamlit as st

from tools import dashboard

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
  --violet:#7a5cff; --green:#34C759; --red:#FF3B30; --orange:#FF9F0A;
  --glass:rgba(255,255,255,.68);
  --hairline:rgba(0,0,0,.08);
  --shadow-card:0 1px 2px rgba(0,0,0,.04),0 18px 44px -18px rgba(0,0,0,.16);
  --shadow-bento:0 1px 2px rgba(0,0,0,.03),0 24px 60px -26px rgba(24,32,72,.22);
  --ease:cubic-bezier(.2,.8,.2,1);
  --font:-apple-system,BlinkMacSystemFont,"SF Pro Display","SF Pro Text",
         "Helvetica Neue","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
  /* 数据数字字体：优先圆润/等宽数据字（SF Pro Rounded / DIN），逐级回落到系统字体 */
  --font-num:"SF Pro Rounded","SF Pro Display","DIN Alternate","DIN Condensed",
         "Segoe UI Variable Display","Segoe UI",-apple-system,"Helvetica Neue",
         "PingFang SC","Microsoft YaHei",sans-serif;
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
[data-testid="stDecoration"],#MainMenu,footer{display:none}
/* 侧边栏收起后，「展开」按钮在 stToolbar 内 —— 隐藏工具栏其他图标但保留它 */
[data-testid="stToolbar"]{visibility:hidden}
[data-testid="stToolbar"] [data-testid="stExpandSidebarButton"],
[data-testid="stToolbar"] [data-testid="stExpandSidebarButton"] *{visibility:visible}
/* 顶部悬浮 Header 透明化：不给毛玻璃遮罩——内容上滑时平滑穿行、无「虚方框」截断感
   （看板大卡滚动到顶时尤其明显）。收起侧边栏后的「展开」按钮仍可见（stToolbar 规则）。 */
[data-testid="stHeader"]{
  background:transparent;
  backdrop-filter:none;
  -webkit-backdrop-filter:none;
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
/* ---------- 顶栏（统一 48px 高度中轴线，左右严格同几何中线）----------
   左侧品牌在 st.columns 单行（stHorizontalBlock），右侧原生折叠按钮在绝对定位的
   stSidebarHeader —— 两者各自居中但容器不同，历史 bug：columns 因外层 padding:0
   从 y=0 贴顶起排（文字偏上），折叠按钮在固定高 header 内居中（中心线在下）→
   ~12px 垂直落差。根治：两边容器统一 48px 等高基准，列与内部纵向块强制垂直居中，
   品牌 margin 清零 → 文字与折叠箭头基于同一 48px 容器的几何中线对齐。 */
[data-testid="stSidebarContent"]{
  position:relative;padding:0!important;
  display:flex;flex-direction:column;height:100vh;overflow:hidden;
}
[data-testid="stSidebarUserContent"]{
  display:flex;flex-direction:column;flex:1 1 auto;min-height:0;padding:0!important;
}
[data-testid="stSidebarUserContent"]>[data-testid="stVerticalBlock"],
[data-testid="stSidebarUserContent"]>div{
  display:flex;flex-direction:column;flex:1 1 auto;min-height:0;padding:0!important;gap:.4rem;
}
/* 统一顶栏高度基准：折叠按钮 header 与品牌 columns 行同为 48px、内容居中 */
[data-testid="stSidebarHeader"]{
  position:absolute;top:0;right:0;height:48px!important;min-height:48px!important;
  display:flex;align-items:center;justify-content:flex-end;
  padding:0 .6rem 0 0;z-index:30;pointer-events:none;
}
[data-testid="stLogoSpacer"]{display:none}
[data-testid="stSidebarHeader"] [data-testid="stSidebarCollapseButton"]{
  pointer-events:auto;visibility:visible!important;opacity:1!important;
  display:inline-flex;align-items:center;justify-content:center;
  width:30px;height:30px;border-radius:8px;color:var(--blue);transition:background .2s;
}
[data-testid="stSidebarHeader"] [data-testid="stSidebarCollapseButton"]:hover{background:rgba(10,132,255,.10)}
/* 折叠箭头 svg 自带内联灰色（fadedText60）→ 强制品牌蓝；display:block 清行内基线 */
[data-testid="stSidebarHeader"] [data-testid="stSidebarCollapseButton"] svg{color:var(--blue)!important;display:block}
/* 品牌行 = st.columns([5,1]) 单行：左列品牌（flex:1 占满）；右列空置（折叠按钮由
   stSidebarHeader 绝对定位在同一 48px 顶带右上，右列仅预留其槽位宽 ≈ 3rem）。
   列与内部纵向块强制 flex 居中 + 等高 48px → 消除文字「顶天贴顶」。 */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]{
  display:flex!important;align-items:center!important;
  height:48px!important;min-height:48px!important;gap:.4rem!important;flex:0 0 auto;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]>[data-testid="stColumn"]{
  display:flex!important;align-items:center!important;
  height:48px!important;padding:0!important;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]>[data-testid="stColumn"]:first-child{flex:1 1 auto;min-width:0}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]>[data-testid="stColumn"]:last-child{
  flex:0 0 84px!important;width:84px!important;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stVerticalBlock"]{
  display:flex!important;align-items:center!important;justify-content:center!important;
  height:100%!important;gap:0!important;
}
/* 品牌行：左参考线 0.9rem，含渐变色圆点；margin 清零 → 基于 48px 容器几何中线 */
.brand{font-size:1.05rem;font-weight:700;letter-spacing:-.01em;display:flex;
  align-items:center;gap:.55rem;line-height:1;margin:0!important;padding-left:.9rem}
.brand-dot{width:12px;height:12px;flex:none;border-radius:50%;
  background:conic-gradient(from 180deg,#0a84ff,#7a5cff,#ff375f,#0a84ff);
  box-shadow:0 0 12px rgba(10,132,255,.55);
  transform:translateY(-1px)}
/* 左垂直参考线：一级入口 / 分类标题 / 会话项统一从 0.9rem 起排 */
.side-label{font-size:.7rem;letter-spacing:.14em;text-transform:uppercase;
  color:var(--ink-3);font-weight:600;margin:1rem 0 .5rem;padding-left:.9rem}
.side-empty{font-size:.8rem;color:var(--ink-3);margin:.1rem 0 .4rem;padding-left:.9rem}
[data-testid="stSidebar"] .stButton button{
  width:100%;border:none;background:transparent;color:var(--ink);
  justify-content:flex-start;text-align:left;border-radius:8px;
  padding:.58rem .9rem;font-weight:500;transition:all .22s var(--ease);
}
.st-key-nav_chat button,.st-key-nav_dash button,.st-key-nav_lib button{
  justify-content:flex-start!important;text-align:left!important;border-radius:8px;
  background:transparent!important;color:var(--ink)!important;padding:.5rem .9rem!important;
}
.st-key-nav_chat button:hover,.st-key-nav_dash button:hover,.st-key-nav_lib button:hover{background:rgba(0,0,0,.05)!important}
/* 图标内嵌于按钮 label（:material/...，每项单图标，无 CSS ::before 附加图标）；
   内层 label 容器一并左对齐（同历史会话项）——只改 button 的 justify-content 压不住
   内层居中，文字会右侧留大片空白、脱离 0.9rem 基准线。 */
.st-key-nav_chat button div,.st-key-nav_chat button span,.st-key-nav_chat button p,
.st-key-nav_dash button div,.st-key-nav_dash button span,.st-key-nav_dash button p,
.st-key-nav_lib button div,.st-key-nav_lib button span,.st-key-nav_lib button p{
  justify-content:flex-start!important;text-align:left!important;
}
/* 历史会话按钮文字严格左对齐（禁止居中）：Streamlit 按钮内部还有一层 display:flex;
   justify-content:center 的 label 容器 —— 只改 button 的 justify-content 压不住
   居中，必须连带覆盖内层 div/span/p 一起 flex-start + 左对齐。 */
[data-testid="stSidebar"] [class*="st-key-conv_"] button div,
[data-testid="stSidebar"] [class*="st-key-conv_"] button span,
[data-testid="stSidebar"] [class*="st-key-conv_"] button p{
  justify-content:flex-start!important;text-align:left!important;
}
[data-testid="stSidebar"] .stButton button p{
  overflow:hidden;white-space:nowrap;text-overflow:ellipsis;
  min-width:0;flex:0 1 auto;
}
[data-testid="stSidebar"] .stButton button:hover{background:rgba(0,0,0,.05)}
[data-testid="stSidebar"] .stButton button:active{transform:scale(.98)}
[data-testid="stSidebar"] .stButton button:disabled{
  opacity:1;cursor:default;color:var(--blue)!important;
  background:rgba(0,113,227,.10);font-weight:600;
}
/* 历史会话项：彻底左对齐 + 聊天气泡小图标 + 8px 圆角 + hover 交互动效 */
[data-testid="stSidebar"] [class*="st-key-conv_"] button{
  justify-content:flex-start!important;text-align:left!important;border-radius:8px;
}
[data-testid="stSidebar"] [class*="st-key-conv_"] button::before{
  content:"";display:inline-block;width:13px;height:13px;margin-right:.55rem;flex:none;
  background:currentColor;opacity:.55;vertical-align:-1.5px;
  -webkit-mask:url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M12 2.5c6 0 9.5 4 9.5 8.75 0 4.5-3.4 8.25-9.5 8.25-.9 0-1.8-.1-2.6-.3L5 21.5l.8-3.3C3.6 16.7 2.5 14.3 2.5 11.25 2.5 6.5 6 2.5 12 2.5z"/></svg>') center/contain no-repeat;
  mask:url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M12 2.5c6 0 9.5 4 9.5 8.75 0 4.5-3.4 8.25-9.5 8.25-.9 0-1.8-.1-2.6-.3L5 21.5l.8-3.3C3.6 16.7 2.5 14.3 2.5 11.25 2.5 6.5 6 2.5 12 2.5z"/></svg>') center/contain no-repeat;
}
[data-testid="stSidebar"] [class*="st-key-conv_"] button:hover{
  background:rgba(0,113,227,.08);
  box-shadow:inset 0 0 0 1px rgba(0,113,227,.16);
}
[data-testid="stSidebar"] [class*="st-key-conv_"] button:active{
  background:rgba(0,113,227,.14);transform:scale(.985);
}
/* 后台执行中的历史会话：旋转环替换气泡图标（完成后由监控触发整页 rerun 移除） */
[data-testid="stSidebar"] [class*="st-key-conv_run_"] .stButton button::before{
  content:"";display:inline-block;width:11px;height:11px;margin-right:.55rem;flex:none;
  border:2px solid rgba(10,132,255,.28);border-top-color:#0a84ff;border-radius:50%;
  background:none;-webkit-mask:none;mask:none;opacity:1;
  vertical-align:-1.5px;animation:spinRing .75s linear infinite;
}
@keyframes spinRing{to{transform:rotate(360deg)}}

.st-key-hist_scroll{flex:1 1 auto;min-height:0;overflow-y:auto;padding-bottom:1rem}
.st-key-hist_scroll::-webkit-scrollbar{width:6px}
.lib-card{max-width:620px;margin:5rem auto;padding:3rem 2rem;text-align:center;
  border:1px solid var(--hairline);border-radius:20px;background:rgba(255,255,255,.72);
  box-shadow:var(--shadow-card);backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px)}

.hero{text-align:center;padding:3.4rem 0 1.4rem;animation:fade .6s var(--ease) both}
.hero-chip{display:inline-flex;align-items:center;gap:.4rem;font-size:.8rem;font-weight:600;
  color:var(--ink-2);background:rgba(255,255,255,.85);border:1px solid var(--hairline);
  border-radius:999px;padding:.35rem .85rem}
/* hero 标题是打字机（st.iframe 内自含样式，见 _TYPEWRITER_HTML）——不再走 markdown */
.hero-sub{color:var(--ink-2);font-size:1.05rem;max-width:32rem;margin:0 auto}

/* suggestion cards */
.st-key-sugg_0 button,.st-key-sugg_1 button,.st-key-sugg_2 button,.st-key-sugg_3 button{
  min-height:60px;white-space:pre-wrap;text-align:center;justify-content:center;
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
[data-testid="stChatMessage"]{animation:rise .45s var(--ease) both;background:transparent;padding:.35rem 0;
  align-items:flex-end!important}  /* IM 习惯：头像与气泡底边严格平齐（多行时贴末行） */
/* user 气泡的上下内边距（原 .85rem 偏肥：单行短文本上下大片空隙、气泡臃肿）。
   收紧到上下 .5rem，紧贴内容自适应高度；助手卡不受影响（自带 padding:0）。 */
[data-testid="stChatMessageContent"]{padding:.5rem 1.15rem;border-radius:20px;line-height:1.6}
[data-testid="stMarkdownContainer"] p{margin-bottom:.35rem}
/* 修复「卡片高度崩塌 / 内容穿透底边框」根因：Streamlit 给 stMarkdownContainer 设
   margin-bottom:-1rem 抵消 <p> 默认 1rem 底边距；本应用把 p 底边距改成 .35rem 后
   补偿失效 → markdown 边框盒恒定探出容器 16px，长推理文字底部穿透卡片边框、与外层
   状态行压线。改为 margin-bottom:0，容器按内容自适应长高、底边线严密闭合。 */
[data-testid="stMarkdownContainer"]{margin-bottom:0!important}

/* 角色气泡（user/assistant 分区）。Streamlit 1.63 的 emoji 头像（🧑/✨）走
   AvatarType.EMOJI 分支 → 纯 div、不渲染 chatAvatarIcon-* testid（见静态 bundle
   的 xo() 分支），故原先用该 testid 门控的气泡 CSS 全程死链。改用可靠信号门控 ——
   stChatMessageContent 的 aria-label（固定英文 "Chat message from user/assistant"）。 */

/* assistant 消息：不整卡包气泡 —— 拆成「思考卡片」+「回复气泡」两个独立组件。
   助手内容容器透明、限宽 92%、左对齐、无内边距；两组件各自塑形（思考=expander 卡、
   回复=.st-key-reply_bubble），视觉层级分明、不共用同一容器。 */
[data-testid="stChatMessageContent"][aria-label="Chat message from assistant"]{
  max-width:92%;
  margin:0;margin-right:auto;
  padding:0;background:transparent;border:none;box-shadow:none;
}

/* 正式回复气泡：白色、发丝边框、四角统一大圆角（独立于思考卡片）。
   key 唯一 → 类名为 st-key-reply_bubble_<suffix>，用前缀部分匹配统一命中。
   上下内边距收紧到 .55rem（单行短回复不再上下大片空隙、气泡高度自适应内容）。 */
[class*="st-key-reply_bubble"]{
  border-radius:20px!important;
  border:1px solid var(--hairline)!important;
  background:#fff!important;
  padding:.55rem 1.05rem!important;
  box-shadow:var(--shadow-card);
}
[class*="st-key-reply_bubble"] p{margin-bottom:0}

/* user 气泡（右、蓝渐变、四角统一大圆角）。垂直居中修复：末尾 <p> 的 margin-bottom
   （5.6px）使单行文字上浮、下方留白大于上方——容器 flex 垂直居中 + 末段边距清零，
   上下内边距等高对称。 */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageContent"][aria-label="Chat message from user"]){flex-direction:row-reverse}
[data-testid="stChatMessageContent"][aria-label="Chat message from user"]{
  flex:0 0 auto;width:fit-content;max-width:78%;
  margin:0;margin-left:auto;
  display:flex;flex-direction:column;justify-content:center;
  background:linear-gradient(135deg,#0a84ff,#0068d8);color:#fff;
  border-radius:20px;
  box-shadow:0 10px 26px -12px rgba(0,104,216,.55);
}
[data-testid="stChatMessageContent"][aria-label="Chat message from user"] [data-testid="stMarkdownContainer"]{margin:0}
[data-testid="stChatMessageContent"][aria-label="Chat message from user"] [data-testid="stMarkdownContainer"] p{color:#fff;margin:0}

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
[data-testid="stExpander"] summary{
  font-size:.85rem;font-weight:600;color:var(--ink-2);
  display:flex;align-items:center;gap:.5rem;
  min-height:2.2rem;padding:.5rem 1rem .5rem 1.05rem!important;line-height:1.35;
  overflow:visible!important;
}
/* 状态图标（✅/💭/spinner）、标题文字、右侧折叠箭头四者同一几何中线：
   所有直接子项 flex:none + 居中；svg display:block 清除行内基线（svg 默认
   inline 携带 descender 空隙——「✅ 图标沉底」的根源）；emoji 字形行高归一。
   上下 padding 对称（.5rem/.5rem），折叠条内部留白上下均等。 */
[data-testid="stExpander"] summary>*{flex:0 0 auto;align-self:center}
[data-testid="stExpander"] summary svg{display:block}
[data-testid="stExpander"] summary [data-testid="stExpanderIcon"]{
  width:18px;height:18px;display:inline-flex;align-items:center;justify-content:center;
  line-height:1;overflow:visible!important;transform:none;
}
[data-testid="stExpander"] summary [data-testid="stExpanderIcon"] *{line-height:1}
[data-testid="stExpander"] summary span,[data-testid="stExpander"] summary p{
  margin:0;line-height:1.35;
}
@keyframes thinkPulse{0%,100%{opacity:1}50%{opacity:.42}}
/* live thinking panel: the open header (icon + label) blinks until the panel completes */
[data-testid="stExpander"] summary[aria-expanded="true"]{
  animation:thinkPulse 1.35s ease-in-out infinite;
}
/* 修复「最后一行文字压线/穿透底边框」：Streamlit stExpanderDetails 默认
   padding-top:8px / padding-bottom:0 —— 内容盒与卡片底边框齐平，多行推理正文的
   最后一行只靠 <p> 的 5.6px 底边距悬空（视觉上骑在底边框上）。显式补足
   padding-bottom:16px，最后一行与底边框之间保留完整 16px 空白间距。
   横向同理：默认无左右内边距，正文（尤其 markdown 表格/长 JSON token）贴到
   面板左右边框——补足 padding-left/right:16px，两侧各留 16px 呼吸空白。 */
[data-testid="stExpanderDetails"]{
  padding-bottom:16px!important;padding-left:16px!important;padding-right:16px!important;
}
/* 高度兜底：details 一律 height:auto / min-height:min-content / max-height:none /
   overflow:visible —— 无固定高度、无折叠动画高度锁、无 overflow 裁剪，容器高度
   严格由内部文字撑开（compact 用原生 details，无 max-height 动画，此处显式声明防回归）。 */
[data-testid="stExpander"] details{
  height:auto!important;min-height:min-content;max-height:none!important;overflow:visible!important;
}
/* 状态框（💭 Thinking…）内文字居中，上下左右对称 */
[data-testid="stStatusWidget"]{text-align:center}
[data-testid="stStatusWidgetButton"]{justify-content:center}
[data-testid="stStatusWidgetLabel"]{text-align:center}

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
/* 鼠标经过：一圈柔和光晕（静态、低强度） */
[data-testid="stChatInput"] > div:hover {
  border-color: rgba(10, 132, 255, 0.42) !important;
  box-shadow:
    0 0 0 4px rgba(10, 132, 255, 0.12),
    0 0 22px 1px rgba(10, 132, 255, 0.14),
    0 18px 44px -16px rgba(0, 0, 0, 0.24) !important;
}

/* 聚焦：一圈呼吸式柔和光晕（动态光影，持续缓慢放大/缩小的光晕环） */
[data-testid="stChatInput"] > div:focus-within {
  border-color: rgba(10, 132, 255, 0.62) !important;
  animation: glowPulse 2.2s var(--ease) infinite;
}
@keyframes glowPulse {
  0%, 100% {
    box-shadow:
      0 0 0 5px rgba(10, 132, 255, 0.14),
      0 0 26px 2px rgba(10, 132, 255, 0.16),
      0 18px 44px -16px rgba(0, 0, 0, 0.24) !important;
  }
  50% {
    box-shadow:
      0 0 0 9px rgba(10, 132, 255, 0.08),
      0 0 44px 8px rgba(10, 132, 255, 0.32),
      0 18px 44px -16px rgba(0, 0, 0, 0.24) !important;
  }
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

/* 看板状态行：flex 布局，状态文字在左、刷新按钮在右（对齐图表右侧边）。
   不用 float（Streamlit 列是 flex 容器，float 不生效）。 */
.st-key-dash_status{display:flex;justify-content:space-between;align-items:flex-end;gap:1rem;margin:.35rem 0 .5rem}
.st-key-dash_status .st-key-dash_refresh{width:auto;margin:0;flex:none}
.dash-title{display:flex;align-items:center;gap:.6rem;margin:.2rem 0 .1rem}
.dash-title .dot{width:12px;height:12px;border-radius:50%;
  background:conic-gradient(from 180deg,#0a84ff,#7a5cff,#ff375f,#0a84ff);
  box-shadow:0 0 12px rgba(10,132,255,.5)}
/* 经营看板副标题已按需求移除（.dash-sub） */
[data-testid="stMetric"]{
  background:rgba(0,0,0,.02);border:1px solid var(--hairline);border-radius:14px;
  padding:.6rem .85rem;
}

/* ---------- Bento Grid 设计系统（看板）----------
   标准网格：每行卡片统一定高（行内等高、行间不互相撑开），列宽按 Bento 配比。
   keyed 卡片容器（st-key-kpi_* / st-key-bento_* / st-key-sku_*）→ 毛玻璃 + 发丝边 + 弥散阴影。
   数字统一用 --font-num（圆润等宽数据字体）+ 字重 600（非默认粗黑）+ tabular-nums。 */
[class*="st-key-kpi_"],[class*="st-key-bento_"]{
  background:var(--glass)!important;
  backdrop-filter:blur(20px) saturate(160%);
  -webkit-backdrop-filter:blur(20px) saturate(160%);
  border:1px solid var(--hairline)!important;
  border-radius:16px!important;
  box-shadow:var(--shadow-bento)!important;
  transition:transform .3s var(--ease),box-shadow .3s var(--ease);
}
[class*="st-key-bento_"]:hover,[class*="st-key-kpi_"]:hover{
  transform:translateY(-3px);
  box-shadow:0 2px 3px rgba(0,0,0,.03),0 30px 70px -28px rgba(24,32,72,.3)!important;
}
/* 行内卡片等高：靠 keyed 卡片定高实现（不覆盖 Streamlit 的列宽权重！）。
   仅清理列内边距/最小宽，保证 Bento 网格严丝合缝。 */
[class*="st-key-bento_"] [data-testid="stHorizontalBlock"],
[class*="st-key-kpi_"] [data-testid="stHorizontalBlock"]{align-items:flex-start!important;gap:.7rem!important}
[class*="st-key-bento_"] [data-testid="stColumn"],
[class*="st-key-kpi_"] [data-testid="stColumn"]{padding:0!important;min-width:0!important}
/* 定高策略 = 「最小高度 + 内容自适应」（不再写死 height）：
   ① 内容短于 min-height → 同排卡恒等 min-height，天然等高，永不高低脚；
   ② 内容长于 min-height → 卡片自行增高，绝不溢出/切边/出内部滚动条。
   ⚠ 必须带 flex:0 0 auto —— Streamlit 的 emotion 类给这些容器设了 flex:1 1 0%，
   flex-basis 会覆盖 height（实测定高 144px 被拉伸成 154px 的根因）。
   ⚠ KPI 卡是 borderless 容器，Streamlit 不给内边距 → 必须自己补 padding，
   否则文字顶到边框（「毛利率顶在上边框、左侧零边距」的根因）。 */
[class*="st-key-kpi_"]{
  flex:0 0 auto!important;min-height:172px!important;height:auto!important;
  display:flex!important;flex-direction:column;justify-content:center;
  padding:1.05rem 1.15rem!important;
}
.st-key-bento_kpi{flex:0 0 auto!important}
.st-key-bento_trend,.st-key-bento_category{
  flex:0 0 auto!important;min-height:470px!important;height:auto!important;
  display:flex!important;flex-direction:column}
.st-key-bento_inventory,.st-key-bento_marketing{
  flex:0 0 auto!important;min-height:600px!important;height:auto!important;
  display:flex!important;flex-direction:column}
/* Hero 卡内环与文案同行居中 */
.st-key-kpi_hero [data-testid="stHorizontalBlock"]{align-items:center!important;gap:.5rem!important}

/* KPI 卡片内部（自定义 HTML：完全可控字号，杜绝 st.metric 默认 2.25rem 数字截断） */
.kpi-fill{display:flex;flex-direction:column;justify-content:center;gap:.4rem;min-width:0}
/* 双指标卡（毛利率/退款率）：每组独立成块、组间拉开间距，避免挤成一团 */
.kpi-split{gap:.75rem}
.kpi-metric{display:flex;flex-direction:column;gap:.15rem;min-width:0}
.kpi-split .kpi-value{font-size:1.3rem}
.kpi-label{font-size:.76rem;font-weight:600;color:var(--ink-3);letter-spacing:.02em;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.kpi-value{font-family:var(--font-num);font-size:1.5rem;font-weight:600;
  letter-spacing:-.025em;line-height:1.1;font-variant-numeric:tabular-nums;
  font-feature-settings:"tnum" 1;white-space:nowrap}
.kpi-foot{display:flex;align-items:center;gap:.4rem;min-width:0}
/* Hero 卡（跨列双倍宽）：GMV 大数字 + 净销售额/净利润副行 + 迷你走势 + 环比环 */
.kpi-hero{display:flex;flex-direction:column;justify-content:center;gap:.3rem;min-width:0}
.kpi-hero-num{font-family:var(--font-num);font-size:2.3rem;font-weight:600;
  letter-spacing:-.03em;line-height:1.05;font-variant-numeric:tabular-nums;
  font-feature-settings:"tnum" 1;white-space:nowrap}
.kpi-hero-sub{display:flex;flex-wrap:wrap;gap:.2rem .7rem;align-items:center;
  font-size:.78rem;color:var(--ink-2)}
.kpi-hero-sub b{font-family:var(--font-num);font-weight:600;
  font-variant-numeric:tabular-nums;color:var(--ink)}
.kpi-hero-spark{margin-top:.15rem}
.mom-ring{width:60px;height:60px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;flex:none}
.mom-ring-hole{width:46px;height:46px;border-radius:50%;background:#fff;
  display:flex;flex-direction:column;align-items:center;justify-content:center;line-height:1}
.mom-ring-val{font-family:var(--font-num);font-size:.7rem;font-weight:600;
  font-variant-numeric:tabular-nums}
.mom-ring-cap{font-size:.56rem;color:var(--ink-3);margin-top:.08rem}
/* 环比标签胶囊 */
.mom-chip{display:inline-block;font-size:.7rem;font-weight:600;border-radius:999px;
  padding:.1rem .45rem;font-variant-numeric:tabular-nums;white-space:nowrap}
.mom-up{color:#1f8b3d;background:rgba(52,199,89,.13)}
.mom-down{color:#c0271d;background:rgba(255,59,48,.12)}
.mom-flat{color:var(--ink-3);background:rgba(0,0,0,.05)}
/* 品类占比已并入可点击胶囊（st.pills 的 format_func），旧的只读占比胶囊样式已移除 */
/* SKU 卡片行（列表区独立滚动，不撑破定高卡片） */
.st-key-sku_list{max-height:440px;overflow-y:auto;padding-right:.2rem}
.st-key-sku_list::-webkit-scrollbar{width:6px}
[class*="st-key-sku_"]{border-radius:12px!important;box-shadow:none!important;
  background:rgba(255,255,255,.62)!important;border:1px solid var(--hairline)!important;
  padding:.5rem .7rem!important}
.sku-row{display:flex;justify-content:space-between;align-items:baseline;gap:.5rem}
.sku-meta{font-size:.72rem;color:var(--ink-3);white-space:nowrap}
.sku-advice{font-size:.76rem;color:var(--ink-2);margin-top:.1rem}
/* 库存分级筛选：单行 pills（标签自带计数，替代与静态胶囊重复的双行排版） */
.st-key-bento_inventory [data-testid="stPills"] button{border-radius:999px!important;
  font-size:.76rem!important;font-weight:600!important;padding:.15rem .6rem!important}

/* 只隐藏 scroll_bottom / scroll_top 的滚动脚本 iframe；hero 打字机 iframe（同为 stIFrame testid）保持可见 */
.st-key-scroll_js [data-testid="stIFrame"],
.st-key-scroll_top_js [data-testid="stIFrame"]{display:none}
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

# 打字机 hero 标题：st.iframe(srcdoc) 里跑 JS（st.markdown 的 <script> 会被清洗不执行）。
_TYPEWRITER_HTML = """\
<div class="tw"><span id="tw"></span><span class="caret"></span></div>
<style>
  html,body{margin:0;padding:0;height:100%;background:transparent;overflow:hidden;
    display:flex;align-items:center;justify-content:center}
  .tw{font:700 26px/1.35 -apple-system,BlinkMacSystemFont,"SF Pro Display","SF Pro Text",
      "Helvetica Neue","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
    letter-spacing:-.01em;white-space:nowrap;min-height:1.35em;margin:1rem 0 .5rem;
    background:linear-gradient(92deg,#0071e3 0%,#7a5cff 45%,#ff375f 100%);
    -webkit-background-clip:text;background-clip:text;color:transparent}
  .caret{display:inline-block;width:2px;height:1.02em;margin-left:3px;background:#0a84ff;
    vertical-align:-.15em;animation:blink 1s steps(1) infinite}
  @keyframes blink{50%{opacity:0}}
</style>
<script>
(function(){
  var el = document.getElementById('tw');
  if (!el) return;
  var phrases = ['把业务数据，聊给你听。','让数据开口说话。','看懂每一条生意脉络。','用大白话提问，Agent 替你算。'];
  var p = 0, i = 0, del = false;
  function tick(){
    var cur = phrases[p];
    if (!del){
      i += 1; el.textContent = cur.slice(0, i);
      if (i === cur.length){ del = true; setTimeout(tick, 2200); return; }
      setTimeout(tick, 110);
    } else {
      i -= 1; el.textContent = cur.slice(0, i);
      if (i === 0){ del = false; p = (p + 1) % phrases.length; }
      setTimeout(tick, 55);
    }
  }
  tick();
})();
</script>
"""


def _render_typewriter() -> None:
    """hero 打字机标题：st.iframe(srcdoc) 允许 JS 执行，实现多短语轮播打字效果。

    之前的实现把 <script> 写进 st.markdown(unsafe_allow_html=True)——Streamlit 会清洗
    掉脚本标签，标题恒为空、只剩 CSS 光标闪烁（bug #1 根因）。
    """
    st.iframe(_TYPEWRITER_HTML, width="stretch", height=72)


@st.cache_resource
def get_pipeline():
    from orchestration.pipeline import Pipeline

    p = Pipeline()
    # 后台预热 embedding 的 HTTP 连接：进程内首次嵌入实测 ~1970ms（全是 DNS+TLS 握手），
    # 热态只要 ~320ms。挪到启动时，用户的第一个问题不再付这笔钱。尽力而为，失败静默。
    p.warm_up()
    return p


def _render_figure_json(fig_json: str) -> None:
    """Render a Plotly figure JSON string as an inline chart."""
    if not fig_json:
        return
    try:
        import plotly.io as pio

        fig = pio.from_json(fig_json)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.caption(f"（图表渲染失败：{e}）")


def render_chart(result: dict) -> None:
    """Render the inventory scatter-quadrant chart when the result carries a _tool_chart."""
    chart = result.get("_tool_chart")
    if chart:
        _render_figure_json(chart)


def _stream_worker(pipeline, prompt, conv, view, result_box, msg) -> None:
    """后台线程：把流式回答逐块写回会话消息（ChatGPT 式「跳出回来继续回答」）。

    Streamlit 的 rerun 请求会打断当前脚本 run——但打断的只是「渲染」。只要生成跑在
    后台线程、结果逐块写进会话（msg/conv 是传入引用，线程不碰 st.session_state），
    切走再回来时内容早已累积，回答继续算完，不中断、不丢失。
    """
    try:
        history = _conversation_history(conv["messages"][:-1])  # 本问答尚未成对，先剔除
        _, chunks = pipeline.chat_stream(
            prompt, history=history, thread_id=conv["thread_id"], result_box=result_box)
        for kind, text in chunks:
            if msg.get("_timed_out"):   # 监控已判定响应超时 → 停止接收后续分块
                break
            if kind == "reasoning":
                msg["reasoning"] = msg.get("reasoning", "") + text
            elif kind == "content":
                msg["content"] = msg.get("content", "") + text
            elif kind == "phase":
                msg["phase"] = text
        if msg.get("_timed_out"):
            return
        result = result_box.get("result", {})
        msg["result"] = result
        # 工具轮推理（llm.last_reasoning 一次性返回，不经 on_reasoning 流式钩子）并入思考
        # 面板——仅当流式钩子未产出任何推理时并入（state["_reasoning"] 含已流式的生成器
        # 部分，再并入会重复）
        extra = result.get("_reasoning") or ""
        if extra and not msg.get("reasoning"):
            msg["reasoning"] = extra
        if view == "dashboard":
            try:
                msg["chart_json"] = dashboard.answer_chart(result, pipeline.layer, pipeline.executor)
            except Exception:
                msg["chart_json"] = None
    except Exception as e:
        if not msg.get("_timed_out"):   # 超时中断后不再追加错误尾巴污染提示
            msg["content"] = msg.get("content", "") + f"（出错了：{e}）"
    finally:
        msg["_streaming"] = False
        # 冻结耗时：完成态标题读这里，不再用 time.time()-_t0 每次重渲染重算
        # （fragment 0.5s 轮询期间那个值会永远递增——「耗时狂飙」的根因）。
        # setdefault：超时兜底已先冻结过时不覆盖。
        msg.setdefault("_elapsed", max(0.0, time.time() - msg.get("_t0", time.time())))


def _expire_inflight_reqs() -> None:
    """TTL 过期清理幂等请求身份（主线程侧，每次渲染路径调用）。

    注册满 30s 的身份自然过期——远超任何 rerun 级重放窗口（fragment 0.5s 轮询、
    重连重发都在秒级完成），同时不妨碍用户稍后真想重问同一句。
    不按「回答完成」注销：重放副本常在完成后才到达（在途时被闸门挡住、完成闸门
    已开——正是此前「完成即注销」版本仍偶发双提问的根因）。
    """
    ss = st.session_state
    seen = ss.get("_inflight_reqs")
    if not seen:
        return
    cutoff = time.time() - 30
    for rid in [r for r, ts in seen.items() if ts < cutoff]:
        del seen[rid]


_REVEAL_CHARS_PER_TICK = 160  # 每 0.5s tick 揭露的推理字符数。真实 LLM chunk 小 → 实时追平
                              # （逐字流式）；一次性大注入（工具轮/测试桩）→ 打字机式消化，
                              # 不再「phase → 一大段推理」的跳变。


def _advance_reveal(msg: dict) -> int:
    """打字机游标：推进「已显示」字符数并返回；reasoning 全量由调用方按游标切片。

    msg 是普通 dict（后台线程 append reasoning、fragment 主线程推进游标，GIL 保护）。
    截断对齐到最近换行边界，避免 markdown 结构（列表/加粗）被腰斩导致渲染跳动。
    """
    full = msg.get("reasoning") or ""
    shown = msg.get("_rs_shown") or 0
    if shown < len(full):
        target = min(len(full), shown + _REVEAL_CHARS_PER_TICK)
        if target < len(full):
            nl = full.rfind("\n", shown, target)
            if nl > shown:
                target = nl + 1
        msg["_rs_shown"] = target
        shown = target
    return shown


def _render_reply_bubble(text: str, bubble_key: str) -> None:
    """正式回复气泡：与「深度思考卡片」解耦的独立视觉组件。

    keyed bordered 容器 → 前端以 .st-key-reply_bubble_* 类名暴露给 CSS 统一塑形。
    容器 key 须唯一（Streamlit 会去重 widget user_key），故每条消息传各自稳定下标/标识；
    CSS 用 [class*="st-key-reply_bubble"] 部分匹配统一命中。
    """
    with st.container(border=True, key=f"reply_bubble_{bubble_key}"):
        st.markdown(text)


def _render_streaming_message(msg: dict, conv_id: str, msg_index: int = -1) -> None:
    """渲染在途（后台线程仍在写）的助手消息：思考卡片 + 部分回答气泡 + 光标。

    ⚠ 面板不能带 key：keyed st.expander 被本 fragment 每 0.5s 重建时，偶发触发前端
    delta 树 reconcil 竞态 → 同一条消息长出两个「💭 Thinking」面板（示例问题首点的
    已知 bug，~2/3 复现）。未 keyed 时按稳定路径就地更新——面板内的 reasoning
    markdown 每 tick 更新从未重复，证明该路径安全。代价：思考进行中手动收起后会在
    下一 tick 重新展开。面板内部结构恒定 → Streamlit 就地更新、不重建元素，
    避免 rise 动画反复重启造成的「变浅 / 思考一卡一卡」。

    思考卡片与正式回复气泡为两个独立组件：思考结束前由思考卡片承载（含中间步骤的
    加载文案，内聚在面板内）；答案一到，思考卡片收敛为 ✅ 收起态、回复另起气泡。
    """
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        reasoning = msg.get("reasoning") or ""
        phase = msg.get("phase") or ""
        content = msg.get("content") or ""
        shown = _advance_reveal(msg)

        if content:
            # 正式回复已产出（真正的完成时刻）：思考面板此时、且仅此时收敛为对勾
            # 收起态（组件一），回复正文另起气泡（组件二）。
            if reasoning:
                with st.expander("💭 已深度思考", icon=":material/check_circle:",
                                 expanded=False, type="compact"):
                    st.markdown(reasoning)
            _render_reply_bubble(content + "▍", str(msg_index))
            return

        # 尚无回复（含推理揭露完但 SQL/工具仍在执行的 gap 期）：思考面板一律保持
        # 进行态「💭 正在思考…」+ spinner——完成判定严格后延到 content 开始流式的
        # 瞬间。旧版在 gap 期就切「已深度思考」+对勾、内部却常驻「⏳ 正在执行
        # 查询…」→ 「宣布思考完毕」与「查询仍在进行」状态撕裂。gap 期当前进度
        # （正在执行查询…等）作为面板内动态思考进度展示；若后续有新增推理
        # （纠错重试轮），游标自然落后、正文继续打字机揭露。
        with st.expander("💭 正在思考…", icon="spinner", expanded=True, type="compact"):
            if shown:
                st.markdown(reasoning[:shown] + ("▍" if shown < len(reasoning) else ""))
            hint = phase or ("正在执行数据查询…" if reasoning and shown >= len(reasoning) else "")
            if hint:
                # 面板内只渲染阶段 caption；「正在思考…」已在标题里，不重复正文占位
                if hint != "正在思考…":
                    st.caption("⏳ " + hint)


@st.fragment(run_every=0.5)
def _chat_area(conv_id, view: str) -> None:
    """消息区（静态历史 + 在途流式回答）整体重绘的 fragment。

    fragment 重跑会把容器内元素清空重画（Streamlit 文档行为）——消息结构变化
    （新消息插入 / 完成态切换 / 视图切换）时整容器原子重画，不残留陈旧子树。
    此前静态区与流式区分属「整页 rerun」与「fragment」两条渲染路径，切换瞬间
    浏览器短暂并存「已更新的静态区」+「旧的流式面板」→ 双思考窗口幽灵子树
    （示例问题首点/思考中提问的已知 bug）。现统一由本 fragment 重绘，幽灵无根。
    完成的一瞬触一发整页 rerun，静态渲染与侧边栏旋转环随状态收敛；
    空闲（无在途）时早退，不空转重绘整个消息列表。视图切走时 fragment 卸载
    （停止轮询），后台线程照常写会话——回来时内容已累积。
    """
    ss = st.session_state
    # 视图切走后，遗留的 0.5s 轮询不再渲染/滚动本会话内容
    # （否则看板会被滚到底、聊天内容串场——跳看板「滚动条滑到底」的根因之一）。
    if conv_id != ss.view_current.get(view):
        return
    conv = ss.conversations.get(conv_id) if conv_id else None
    if conv is None:
        return
    messages = conv["messages"]
    if not messages:
        return
    _render_messages(messages)                 # 静态消息（在途的由下面流式渲染）
    in_flight = [(i, m) for i, m in enumerate(messages) if m.get("_streaming")]
    for i, m in in_flight:
        _render_streaming_message(m, conv_id, i)
    if in_flight:
        scroll_bottom()          # 流式期间自动跟随到底部
        ss._had_stream = True
    else:
        # 无在途（本轮完成）：注销幂等身份，同句重发不再被拦
        _expire_inflight_reqs()
        if ss.pop("_had_stream", False):
            ss._scroll = True
            st.rerun()           # fragment 内 rerun 整页重跑 → 完成消息静态渲染


def _expire_stale_streams(conversations: dict, timeout: float) -> bool:
    """把超时（>timeout 秒）仍在途的消息标为响应超时中断；返回是否有变化。

    后台线程挂起（LLM 卡死）时 _streaming 永远为 True → 旋转环/光标卡死、会话「卡掉」。
    这里兜底：置 _streaming=False 并标注中断（工作线程读 _timed_out 停止接收后续分块）。
    """
    changed = False
    for c in conversations.values():
        for m in c.get("messages", []):
            if m.get("_streaming") and time.time() - m.get("_t0", time.time()) > timeout:
                m["_streaming"] = False
                m["_timed_out"] = True
                # 超时即冻结耗时（工作线程可能仍挂起，其 finally 的 setdefault 不会再覆盖）
                m["_elapsed"] = time.time() - m.get("_t0", time.time())
                m["content"] = (m.get("content") or "") + "（响应超时，已中断。请重试一次～）"
                changed = True
    return changed


@st.fragment(run_every=0.5)
def _stream_status_monitor() -> None:
    """后台会话执行状态监控：任一会话在途标记变化（开始/结束）→ 整页 rerun。

    侧边栏只在整页 rerun 时重绘——后台会话（切走仍在算）完成不会触发当前视图的
    fragment 轮询，需要本监控主动 rerun 一下，让历史按钮的旋转环随执行状态启停。
    无变化时静默（只记快照，不 rerun），避免空转。
    另含响应超时兜底（_expire_stale_streams）——会话不永久卡在「旋转环 / ▍光标」。
    """
    STREAM_TIMEOUT = 120  # 秒；超过视为响应超时，停止在途标记并标注中断
    ss = st.session_state
    prev = ss.get("_bg_status")
    _expire_stale_streams(ss.conversations, STREAM_TIMEOUT)
    now = {cid: _has_in_flight(c["messages"]) for cid, c in ss.conversations.items()}
    ss["_bg_status"] = now
    if prev is not None and now != prev:
        st.rerun()


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


def _has_in_flight(messages: list[dict]) -> bool:
    """会话里是否存在后台线程仍在写的在途消息。"""
    return any(m.get("_streaming") for m in messages)


def _render_messages(messages: list[dict]) -> None:
    """历史消息渲染（聊天/看板共用）：user/assistant 气泡 + 思考面板 + 图表。

    图表优先回放消息里保存的 chart_json（看板问答转来的消息），否则走
    render_chart(result)（工具轨散点图）。在途消息（后台线程仍在写）由
    _chat_area 轮询渲染，这里跳过，避免重复/撕裂。
    """
    for i, m in enumerate(messages):
        if m.get("_streaming"):
            continue
        with st.chat_message(m["role"], avatar=AVATARS[m["role"]]):
            if m["role"] == "assistant" and m.get("reasoning"):
                # thinking panel sits ABOVE the answer (Kimi-style);
                # completed → collapsed default with elapsed time, expandable to review.
                # ⚠ 必须用 keyed st.expander：st.status 在 1.63 没有 key 参数，
                # 本 fragment 每 0.5s 重绘一次会重置展开态 → 点击永远打不开；
                # keyed expander 的开合状态存进 session_state，重绘不丢。
                # 耗时读流结束时冻结的 _elapsed（time.time()-_t0 会随重绘永远递增）。
                # 超时中断（_timed_out）→ 「💭 思考已中断」收起态：残留的流式思考窗
                # （spinner/正在思考…）不得幸存，与超时提示回复语义连贯。
                elapsed = m.get("_elapsed") or 0.0
                if m.get("_timed_out"):
                    label = "💭 思考已中断"
                else:
                    label = (f"💭 已深度思考（耗时 {elapsed:.0f} 秒）" if elapsed >= 1
                             else "💭 已深度思考")
                with st.expander(label, icon=":material/check_circle:", expanded=False,
                                 key=f"think_done_{i}", type="compact"):
                    st.markdown(m["reasoning"])
            if m["role"] == "assistant":
                # 正式回复气泡（独立组件）：与思考卡片解耦；图表随消息回放。
                if m.get("content"):
                    _render_reply_bubble(m["content"], str(i))
                if m.get("chart_json"):
                    _render_figure_json(m["chart_json"])
                elif m.get("result"):
                    render_chart(m["result"])
            else:
                st.markdown(m["content"])


def _select(conv_id: str, source: str) -> None:
    """点侧边栏历史：切到对应视图并激活该会话（看板会话进入提问态回放）。"""
    ss = st.session_state
    ss.view_current[source] = conv_id
    ss.view = "chat" if source == "chat" else "dashboard"
    if source == "dashboard":
        ss._dash["mode"] = "answer"   # 看板会话 → 提问态回放对话
    ss._hero_dismissed = False        # 切到空会话时 hero 允许复现
    ss._scroll = True


def _goto(target: str) -> None:
    ss = st.session_state
    ss.view = target
    if target == "chat":
        ss.view_current["chat"] = None
    elif target == "dashboard":
        ss._dash["mode"] = "default"
        ss.view_current["dashboard"] = None
    ss._hero_dismissed = False
    ss._scroll_top = True
    ss._scroll = False
    ss._had_stream = False


# 搜索弹窗 JS 桥：自动聚焦、ESC/遮罩关闭（转点隐藏的关闭钮）、键击级实时过滤
# （input 事件委托到 document——Streamlit rerun 重建输入框后监听不失效）。
def _render_sidebar(ss) -> None:
    with st.sidebar:
        # 顶栏单行：品牌（左列）+ 右列空槽（原生折叠按钮 « 由 stSidebarHeader 绝对
        # 定位在同一 44px 顶带右上 → 两元素同一垂直中线）。st.columns 是 Streamlit
        # 唯一保证水平单行的容器（stHorizontalBlock）。
        brand_col, icon_col = st.columns([5, 1], gap="small", vertical_alignment="center")
        with brand_col:
            st.markdown(
                '<div class="brand"><span class="brand-dot"></span>电商智能问数</div>',
                unsafe_allow_html=True,
            )
        with icon_col:
            pass  # 槽位留给绝对定位的折叠按钮（不在内容流内）
        st.button(":material/edit_square:  新聊天", key="nav_chat", use_container_width=True,
                  on_click=_goto, args=("chat",))
        st.button(":material/dashboard:  经营看板", key="nav_dash", use_container_width=True,
                  on_click=_goto, args=("dashboard",))
        st.button(":material/menu_book:  资料库", key="nav_lib", use_container_width=True,
                  on_click=_goto, args=("library",))
        _history_section(ss)


def _history_section(ss) -> None:
    """历史对话列表（搜索功能已下线——按用户决策做减法，列表独享垂直空间）。"""
    st.markdown('<div class="side-label">历史对话</div>', unsafe_allow_html=True)
    with st.container(key="hist_scroll"):
        if not ss.conversations:
            st.markdown('<div class="side-empty">暂无</div>', unsafe_allow_html=True)
        for conv_id in reversed(list(ss.conversations.keys())):
            conv = ss.conversations[conv_id]
            source = conv.get("source", "chat")
            active = conv_id == ss.view_current.get(source)
            label = f"{'● ' if active else ''}{conv['title'] or '新对话'}"
            btn_kw = dict(key=f"conv_{source}_{conv_id}", use_container_width=True,
                          on_click=_select, args=(conv_id, source))
            if _has_in_flight(conv["messages"]):
                with st.container(key=f"conv_run_{conv_id}"):
                    st.button(label, **btn_kw)
            else:
                st.button(label, **btn_kw)


def _render_library() -> None:
    st.markdown(
        '<div class="lib-card"><h2>📚 资料库</h2>'
        '<p>核心指标图表与分析快照将保存在这里。</p>'
        '<small>资料库功能即将上线</small></div>',
        unsafe_allow_html=True,
    )


def _suggest(text: str) -> None:
    st.session_state._suggested = text


def _prefill(text: str) -> None:
    """把建议文案预填进底部输入框（不自动提交，与 _suggest 的自动提交区分）。

    看板 fragment 内点按钮 → 置标记 + 整页 rerun；main() 在 chat_input 实例化前
    写 ss["composer"]（同 _composer_consumed 已验证模式）。
    """
    st.session_state["_prefill_prompt"] = text
    st.rerun(scope="app")


def scroll_bottom() -> None:
    # st.iframe replaced st.components.v1.html (removed after 2026-06-01).
    # 包一层 keyed 容器 → CSS 只隐藏这个滚动脚本 iframe，不误伤 hero 打字机 iframe。
    with st.container(key="scroll_js"):
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


def scroll_to_top() -> None:
    # 视图切换后回到顶部（看板从标题/概览开始，不沿用聊天页底部的滚动位置）。
    # 与 scroll_bottom 同机制：keyed 容器 + CSS 隐藏 iframe，st.iframe 允许 JS 执行。
    with st.container(key="scroll_top_js"):
        st.iframe(
            """
            <script>
            (function () {
              var doc = window.parent.document;
              ['[data-testid="stAppViewContainer"]', '.main', 'section.main'].forEach(function (s) {
                var el = doc.querySelector(s);
                if (el) { el.scrollTop = 0; }
              });
              window.parent.scrollTo(0, 0);
            })();
            </script>
            """,
            height="content",
        )


def _render_chat(suggested) -> None:
    """聊天视图：hero（空态打字机）+ 历史对话回放 + 在途回答轮询。输入框在 main() 统一处理。"""
    ss = st.session_state
    conv = ss.conversations.get(ss.view_current["chat"])
    messages = conv["messages"] if conv else []
    # ---- hero + suggestions (empty state) ----
    # 只在「从没提交过 + 当前会话无消息」时展示；_hero_dismissed 兜底：
    # 即使出现会话被重置的边角情况，四个示例按钮也绝不回头。
    if not messages and suggested is None and not ss.get("_hero_dismissed", False):
        st.markdown(
            '<div class="hero"><div class="hero-chip">✦&nbsp;&nbsp;AI 数据问数助手</div></div>',
            unsafe_allow_html=True,
        )
        _render_typewriter()
        st.markdown(
            '<div class="hero-sub">GMV、销量、库存、趋势——用一句大白话提问，剩下的交给 Agent。</div>',
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
    # 消息区全部由 _chat_area fragment 重绘（静态 + 在途）。fragment 容器内元素每次
    # 重跑清空重画 → 消息结构变化时原子替换，不残留陈旧子树（防双思考窗口幽灵）。
    # 此前静态区放整页 rerun、流式区放 fragment 两条路径，切换瞬间短暂并存导致幽灵。
    _chat_area(ss.view_current["chat"], "chat")   # 静态历史 + 在途回答轮询
    # 滚动：有在途回答时由 _chat_area 负责（避免 key='scroll_js' 重复）；
    # 静止时（提交/点历史/回答完成）这里滚一次到位。
    if ss.pop("_scroll", None) and not _has_in_flight(messages):
        scroll_bottom()


def main() -> None:
    ss = st.session_state
    ss.setdefault("conversations", {})
    ss.setdefault("view", "chat")
    ss.setdefault("view_current", {"chat": None, "dashboard": None})
    ss.setdefault("_dash", {"mode": "default", "cards": None})

    pipeline = get_pipeline()
    if getattr(pipeline, "_dash_box", None) is None:
        pipeline._dash_box = {}                     # 后台预装配看板卡片 → 首次跳看板丝滑
        threading.Thread(target=_prewarm_dashboard_cards,
                         args=(pipeline,), daemon=True).start()
    # 提交后的下一次 run、在任何 widget 实例化之前清空 composer。直接在提交的同一 run
    # 里置空会触发 StreamlitWidgetAlreadyInstantiatedError（widget 实例化后禁止改其
    # key），而放在 _submit_message 之后又是死代码（其末尾 st.rerun() 抛 RerunException
    # 立即中断脚本）。不清空 → chat_input 在每个全页 rerun 重放旧提交值 → 同一问题被
    # 重复提交/排队，会话冒出两条相同问答（「尤其看板视图」重复的根因）。
    if ss.pop("_composer_consumed", False):
        ss["composer"] = ""
    _pre = ss.pop("_prefill_prompt", None)
    if _pre:
        ss["composer"] = _pre      # 看板按钮预填：实例化前写值合法，提交由用户按回车
    _expire_inflight_reqs()   # 已完成的问答轮注销幂等身份（同句重发不再被拦）
    suggested = ss.pop("_suggested", None)

    _render_sidebar(ss)
    _stream_status_monitor()
    if ss.view == "dashboard":
        _render_dashboard(pipeline)
    elif ss.view == "library":
        _render_library()
    else:
        _render_chat(suggested)

    # 切换视图的滚动归位：进入看板、资料库或新聊天时回到顶部。
    # 视图切换后回到顶部（侧边栏入口统一设置此标记）。
    if ss.pop("_scroll_top", False):
        scroll_to_top()

    placeholder = (
        "问点什么，例如：2026年7月GMV是多少？"
        if ss.view in ("chat", "library")
        else "问点什么：统计/图表问题（如「画个各品类 GMV 图表」）、或直接闲聊都可以"
    )
    prompt = st.chat_input(placeholder, key="composer") or suggested
    if prompt:
        ss._composer_consumed = True
        _submit_message(pipeline, prompt, "chat" if ss.view == "library" else ss.view)
    if ss.get("_pending") and not any(
            _has_in_flight(c["messages"]) for c in ss.conversations.values()):
        q_prompt, q_view = ss["_pending"].pop(0)
        _submit_message(pipeline, q_prompt, q_view)


def _fingerprint_of(executor) -> str | None:
    """数仓指纹（对陈旧进程容错：运行中的旧版 tools.dashboard 若没有该函数 → None，
    不自动轮询、仅保留手动刷新；重启服务后恢复实时轮询）。"""
    fn = getattr(dashboard, "_warehouse_fingerprint", None)
    if fn is None:
        return None
    try:
        return fn(executor)
    except Exception:
        return None


def _warehouse_fingerprint(pipeline) -> str | None:
    return _fingerprint_of(pipeline.executor)


def _prewarm_dashboard_cards(pipeline) -> None:
    """后台预装配看板卡片：app 加载即开算，首次跳看板不再被 0.8s SQL 装配阻塞。

    用户停留在聊天页的间隙把三张卡算好放进 pipeline._dash_box（普通 dict，线程不碰
    session_state）；首次进看板时由 _adopt_prewarmed_cards 直接取用，切换即丝滑
    （否则切换那 0.8s 里浏览器仍显示旧聊天页的示例卡片，像「残影被带到看板」）。
    用独立只读 DuckDB 连接——不共享主线程 executor 的连接，避免并发查询竞态。
    """
    box = getattr(pipeline, "_dash_box", None)
    if not isinstance(box, dict) or box.get("cards"):
        return
    from compile.executor import Executor
    from config import resolve

    try:
        with Executor(resolve(pipeline.cfg.duckdb_path)) as ex:
            box["cards"] = dashboard.build_cards(pipeline.layer, ex)
            box["fingerprint"] = _fingerprint_of(ex)
    except Exception as e:
        box["cards"] = {"error": str(e)}
    finally:
        box["_warming"] = False


def _ensure_prewarm(pipeline) -> None:
    """看板卡片未就绪时确保后台线程在算（幂等：已在算则不重复开线程）。

    首次进看板且预装配尚未完成 → 补齐线程，本次切换零阻塞；卡片就绪后由
    _render_dashboard_cards 的 1s 轮询就地采纳。
    """
    box = getattr(pipeline, "_dash_box", None)
    if not isinstance(box, dict):
        pipeline._dash_box = {}
        box = pipeline._dash_box
    if box.get("cards") is None and not box.get("_warming"):
        box["_warming"] = True
        threading.Thread(target=_prewarm_dashboard_cards,
                         args=(pipeline,), daemon=True).start()


def _adopt_prewarmed_cards(pipeline, dash: dict) -> None:
    """把后台预装配好的卡片取进会话（只取用、不重建；MagicMock 桩不误取）。"""
    box = getattr(pipeline, "_dash_box", None)
    cards = box.get("cards") if isinstance(box, dict) else None
    if isinstance(cards, dict):
        dash["cards"] = cards
        dash["fingerprint"] = box.get("fingerprint")


def _rebuild_cards(pipeline, dash: dict) -> None:
    """重建看板卡片（确定性 SQL 装配；失败进 error 哨兵）并记录数仓指纹基线。"""
    try:
        dash["cards"] = dashboard.build_cards(pipeline.layer, pipeline.executor)
    except Exception as e:
        dash["cards"] = {"error": str(e)}
    dash["fingerprint"] = _warehouse_fingerprint(pipeline)


@st.fragment(run_every=30)
def _live_poll(pipeline) -> None:
    """看板实时跟随：状态行（数据截至/最后检查/↻刷新）+ 每 30s 轮询数仓指纹。

    指纹有变动（datagen 重跑 / 增量写入）→ 重建卡片 + st.rerun（fragment 内 rerun
    会重跑整个脚本），数字与图表实时跟随最新数据；静态快照则指纹恒定、零额外重绘。
    """
    dash = st.session_state["_dash"]
    cards = dash.get("cards") or {}
    cutoff = cards.get("cutoff") or "—"
    fp = _warehouse_fingerprint(pipeline)
    dash["last_check"] = time.strftime("%H:%M:%S")
    with st.container(key="dash_status"):
        st.caption(f"数据截至 {cutoff} · 每 30 秒自动检查数仓更新（最后检查 {dash['last_check']}）")
        if st.button("↻ 刷新", key="dash_refresh"):
            _rebuild_cards(pipeline, dash)
            st.rerun()
    if fp is not None and fp != dash.get("fingerprint"):
        dash["fingerprint"] = fp
        _rebuild_cards(pipeline, dash)
        st.rerun()


def _render_dashboard(pipeline) -> None:
    """看板视图：日常态展示三张告警/概览卡（图表轮播）；提问后收起卡片、累计展示对话。"""
    ss = st.session_state
    dash = ss["_dash"]

    if dash["mode"] == "answer":
        _render_dashboard_answer()
    else:
        st.markdown(
            "<style>.block-container{max-width:1120px;padding:1.25rem 1.25rem 9rem}</style>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="dash-title"><span class="dot"></span>'
            '<h1 style="margin:0;font-size:1.55rem;letter-spacing:-.02em">经营看板</h1></div>',
            unsafe_allow_html=True,
        )
        if dash.get("cards") is None:
            _adopt_prewarmed_cards(pipeline, dash)   # 优先取后台预装配结果（首次切换不被阻塞）
        if dash.get("cards") is None:
            _ensure_prewarm(pipeline)                # 未就绪 → 后台继续算，本次切换零阻塞
        _live_poll(pipeline)                         # 状态行（数据截至/最后检查/刷新）+ 30s 指纹轮询
        _render_dashboard_cards(pipeline)            # 1s 轮询：未就绪显示「加载中」，就绪后无缝出卡


def _render_dashboard_answer() -> None:
    """提问态：收起卡片，回放看板当前会话全部问答。"""
    ss = st.session_state
    conv = ss.conversations.get(ss.view_current["dashboard"])
    messages = conv["messages"] if conv else []
    _chat_area(ss.view_current["dashboard"], "dashboard")
    if ss.pop("_scroll", None) and not _has_in_flight(messages):
        scroll_bottom()   # 有在途回答时由 _chat_area 负责滚动


@st.fragment(run_every=1)
def _render_dashboard_cards(pipeline) -> None:
    """日常态：1s 图表轮播 fragment——每拍推进一帧重渲染图表（Streamlit delta 只发变化的图）。

    数仓数据变动由 _live_poll 的 30s 指纹轮询重建卡片 → 新数据从下一拍起轮播。
    卡片未就绪（首次进看板、预装配还在后台算）→ 显示「加载中」，下一拍采纳后就地出卡。
    经营概览整行大卡（KPI 磁贴 + 双图），下方库存告警 | 营销建议 半宽并排。
    """
    dash = st.session_state["_dash"]
    if dash.get("cards") is None:
        _adopt_prewarmed_cards(pipeline, dash)   # 预装配完成即采纳（不整页 rerun，就地替换）
        if dash.get("cards") is None:
            st.caption("（看板数据加载中…）")
            return
    cards = dash.get("cards") or {}
    if cards.get("error"):
        st.caption(f"（看板数据加载失败：{cards['error']}）")
        return

    tick = dash.get("anim", 0)
    dash["anim"] = tick + 1
    inv, ov, mk = cards.get("inventory"), cards.get("overview"), cards.get("marketing")

    # Bento Grid 三行：KPI 行（Hero + 紧凑卡）
    #          → 联动行（趋势点击月份下钻 | 品类点击填 Prompt）
    #          → 下钻行（库存 Tag 过滤 | 营销 ROI 盈亏基准线）
    _render_kpi_row(ov, tick)
    trend_col, cat_col = st.columns(2, gap="medium")
    with trend_col:
        _render_trend_card(ov, tick)
    with cat_col:
        _render_category_card(ov, tick)
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        _render_inventory_card(inv)
    with col2:
        _render_marketing_card(mk, tick)


# 环比「越低越好」的指标（渲染层反色：下降=绿）
_MOM_LOWER_BETTER = {"refund_rate"}


def _mom_chip(metric: str, mom, mom_text) -> str:
    """环比标签胶囊 HTML：按指标语义上色（退款率降=好）。"""
    if mom is None or not mom_text:
        return '<span class="mom-chip mom-flat">—</span>'
    good = (mom < 0) if metric in _MOM_LOWER_BETTER else (mom > 0)
    cls = "mom-up" if good else "mom-down"
    arrow = "↑" if mom > 0 else "↓"
    return f'<span class="mom-chip {cls}">{mom_text} {arrow}</span>'


def _mom_ring(mom) -> str:
    """Hero 卡环比环：conic-gradient 圆环，幅度 |mom| 映射（30% 充满）。"""
    if mom is None:
        return '<div class="mom-ring"><div class="mom-ring-hole"><span class="mom-ring-val">—</span></div></div>'
    deg = int(max(0.03, min(1.0, abs(mom) / 0.30)) * 360)
    color = "#FF3B30" if mom < 0 else "#34C759"
    return (
        f'<div class="mom-ring" style="background:conic-gradient({color} {deg}deg, rgba(0,0,0,.07) {deg}deg)">'
        f'<div class="mom-ring-hole"><span class="mom-ring-val" style="color:{color}">{mom * 100:+.1f}%</span>'
        f'<span class="mom-ring-cap">环比</span></div></div>'
    )


def _kpi_card(metric: str, tile: dict) -> None:
    """单张等宽 KPI 卡（自定义 HTML）：标签 + 大数字（数据字体）+ 环比胶囊。

    用自定义 HTML 而非 st.metric —— st.metric 的值默认 2.25rem，窄列必然截断
    （「客单价 1,68...」）；此处字号受控 + nowrap + tabular-nums，永不出省略号。
    """
    st.markdown(
        '<div class="kpi-fill">'
        f'<div class="kpi-label">{tile.get("display", metric)}</div>'
        f'<div class="kpi-value">{tile.get("text", "—")}</div>'
        f'<div class="kpi-foot">{_mom_chip(metric, tile.get("mom"), tile.get("mom_text"))}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def _kpi_hero(gmv: dict, net: dict, profit: dict) -> None:
    """Hero 超级卡：GMV 大数字 + 净销售额/净利润副行 + 环比环 + 迷你走势线。"""
    st.markdown(
        '<div class="kpi-hero">'
        '<div class="kpi-foot"><span class="kpi-label">'
        f'{gmv.get("display", "GMV")}</span>'
        f'{_mom_chip("gmv", gmv.get("mom"), gmv.get("mom_text"))}</div>'
        f'<div class="kpi-hero-num">{gmv.get("text", "—")}</div>'
        '<div class="kpi-hero-sub">'
        f'<span>净销售额 <b>{net.get("text", "—")}</b></span>'
        f'<span>净利润 <b>{profit.get("text", "—")}</b></span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def _render_kpi_row(ov: dict | None, tick: int) -> None:
    """Bento KPI 行：Hero（跨列双倍宽）+ 订单数 + 客单价 + 毛利率/退款率，四卡等高。

    Hero 内嵌 GMV 大数字、净销售额/净利润副行、环比环与迷你走势线；右侧三张紧凑卡
    各含一个大数字 + 环比胶囊。整行统一 128px（CSS 定高），列宽 [2,1,1,1]。
    """
    with st.container(border=True, key="bento_kpi"):
        st.markdown("### 📊 经营概览")
        if not ov:
            st.caption("（经营概览卡片加载失败）")
            return
        tiles = ov.get("tiles", {})
        hero, c1, c2, c3 = st.columns((2, 1, 1, 1), gap="small")
        with hero:
            with st.container(key="kpi_hero"):
                ring_col, body_col = st.columns((1, 4), gap="small")
                with ring_col:
                    st.markdown(_mom_ring(tiles.get("gmv", {}).get("mom")),
                                unsafe_allow_html=True)
                with body_col:
                    _kpi_hero(tiles.get("gmv", {}), tiles.get("net_sales_amount", {}),
                              tiles.get("net_profit", {}))
                _render_sparkline(tiles.get("gmv", {}).get("spark"))
        with c1:
            with st.container(key="kpi_orders"):
                _kpi_card("orders_count", tiles.get("orders_count", {}))
        with c2:
            with st.container(key="kpi_aov"):
                _kpi_card("avg_order_value", tiles.get("avg_order_value", {}))
        with c3:
            with st.container(key="kpi_margin"):
                gm, rr = tiles.get("gross_margin", {}), tiles.get("refund_rate", {})
                st.markdown(
                    '<div class="kpi-fill kpi-split">'
                    '<div class="kpi-metric">'
                    '<div class="kpi-foot"><span class="kpi-label">'
                    f'{gm.get("display", "毛利率")}</span>'
                    f'{_mom_chip("gross_margin", gm.get("mom"), gm.get("mom_text"))}</div>'
                    f'<div class="kpi-value">{gm.get("text", "—")}</div>'
                    '</div>'
                    '<div class="kpi-metric">'
                    '<div class="kpi-foot"><span class="kpi-label">'
                    f'{rr.get("display", "退款率")}</span>'
                    f'{_mom_chip("refund_rate", rr.get("mom"), rr.get("mom_text"))}</div>'
                    f'<div class="kpi-value">{rr.get("text", "—")}</div>'
                    '</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )


def _render_sparkline(spark) -> None:
    """Hero 卡迷你走势线（64px，无坐标轴）。陈旧进程无 _sparkline_chart → 静默跳过。"""
    if not spark:
        return
    fn = getattr(dashboard, "_sparkline_chart", None)
    if fn is None:
        return
    try:
        _render_figure_json(fn(list(spark)))
    except Exception:
        pass


def _render_trend_card(ov: dict | None, tick: int) -> None:
    """Bento 趋势卡：GMV 平滑面积图 + 月份 pills（点击 → 右侧品类图下钻该月）。"""
    with st.container(border=True, key="bento_trend"):
        st.markdown("### 📈 GMV 趋势")
        trend = (ov or {}).get("trend")
        if not trend or not trend.get("labels"):
            st.caption("（GMV 趋势图表加载失败）")
            return
        _render_frame_chart("line", trend["labels"], trend["values"],
                            trend.get("title", "GMV 近 6 月趋势"),
                            trend.get("y_title", "GMV"), tick,
                            trend.get("chart_json"), area=True, height=318)
        months = list(trend["labels"])
        st.pills("下钻月份", months, default=st.session_state.get("trend_month") or months[-1],
                 key="trend_month", label_visibility="collapsed")


def _render_category_card(ov: dict | None, tick: int) -> None:
    """Bento 品类卡：选定月的品类贡献（柱状 + 占比胶囊）+ 分析按钮。

    占比与筛选合并为**同一行可点击胶囊**（format_func 把占比写进标签）——旧版
    「图表上方一排只读占比胶囊 + 下方一排品类药丸」两排重复，已合并。
    月份来自趋势卡 pills（`trend_month`）；缺失时兜底当前月 `ov["category"]`。
    """
    with st.container(border=True, key="bento_category"):
        st.markdown("### 📊 各品类贡献")
        ov = ov or {}
        month = st.session_state.get("trend_month")
        series = ov.get("category_series") or {}
        data = series.get(month) if month else None
        if not data:
            data = ov.get("category")            # 兜底：当前月单月分布
            month = ov.get("month")
        if not data or not data.get("labels"):
            st.caption("（品类分布图表加载失败）")
            return
        pairs = sorted(zip(data["labels"], data["values"]), key=lambda kv: kv[1], reverse=True)
        total = sum(v for _, v in pairs) or 1.0
        share = {l: v / total for l, v in pairs}
        labels = [l for l, _ in pairs]
        _render_frame_chart("bar", labels, [v for _, v in pairs],
                            f"各品类 GMV（{month or ''}）", data.get("y_title", "GMV"),
                            tick, None, colors=_gradient_colors(len(labels)), height=250)
        chosen = st.pills("选择品类", labels, key="cat_pick", label_visibility="collapsed",
                          format_func=lambda l: f"{share.get(l, 0):.1%} {l}")
        if chosen:
            if st.button(f"🤖 分析「{chosen}」", key="cat_analyze"):
                _prefill(f"分析{chosen}品类的经营表现")


def _gradient_colors(n: int) -> list[str]:
    """品类柱状配色：品牌蓝 → 紫罗兰 的逐柱渐变。"""
    if n <= 0:
        return []
    if n == 1:
        return ["#0a84ff"]
    out = []
    for i in range(n):
        t = i / (n - 1)
        r = int(0x0a + (0x7a - 0x0a) * t)
        g = int(0x84 + (0x5c - 0x84) * t)
        b = int(0xff + (0xff - 0xff) * t)
        out.append(f"#{r:02x}{g:02x}{b:02x}")
    return out


def _inv_bucket(record: dict) -> str | None:
    """库存记录分桶（与 tools.dashboard._bucket_of 同语义）；陈旧进程兜底本地实现。"""
    fn = getattr(dashboard, "_bucket_of", None)
    if fn is not None:
        return fn(record)
    d = record.get("doi_days")
    if record.get("on_hand", 0) <= 0 or d == 0:
        return "断货"
    if d is not None and d <= 3:
        return "低库存"
    if d is None or d > 90:
        return "呆滞"
    return None


def _render_inventory_card(inv: dict | None) -> None:
    """Bento 库存告警卡：单行分级筛选（计数内嵌）+ SKU 卡片列表（区内滚动）。

    分级与计数合并为一行 pills（标签自带计数，替掉旧版「静态发光胶囊 + 筛选胶囊」
    两行重复排版）；列表区独立滚动，不撑破行内定高。
    """
    with st.container(border=True, key="bento_inventory"):
        st.markdown("### 🛎 库存告警")
        if not inv:
            st.caption("（库存告警卡片加载失败）")
            return
        counts = inv.get("counts", {})
        levels = ["断货", "低库存", "呆滞"]
        label_of = {"全部": "全部"} | {k: f"{k} {counts.get(k, 0)}" for k in levels}
        sel = st.pills("筛选", ["全部"] + levels, default="全部",
                       format_func=lambda o: label_of.get(o, o),
                       key="inv_filter", label_visibility="collapsed")
        sel = sel or "全部"
        for line in inv.get("insights", [])[:1]:
            st.caption(line)
        rows = inv.get("top") or []
        if sel != "全部":
            rows = [r for r in rows if _inv_bucket(r) == sel]
        with st.container(key="sku_list"):
            if not rows:
                st.caption("（该分级暂无 SKU）")
                return
            for r in rows[:6]:
                with st.container(border=True, key=f"sku_{r['sku_id']}"):
                    doi = r.get("doi_days")
                    st.markdown(
                        f'<div class="sku-row"><b>{r["sku_id"]}</b>'
                        f'<span class="sku-meta">{r.get("category", "")} · 在手 {r.get("on_hand", 0)} · '
                        f'可售 {doi if doi is not None else "—"} 天</span></div>'
                        f'<div class="sku-advice">{r.get("advice", "")}</div>',
                        unsafe_allow_html=True,
                    )
                    b1, b2 = st.columns(2)
                    if b1.button("🤖 提问", key=f"ask_{r['sku_id']}"):
                        _prefill(f"{r['sku_id']} 的库存情况如何？")
                    if b2.button("📦 催补货", key=f"urge_{r['sku_id']}"):
                        _prefill(f"请为 {r['sku_id']} 生成补货建议")


def _render_marketing_card(mk: dict | None, tick: int) -> None:
    """Bento 营销卡：渠道 ROI 柱状（盈亏平衡基准线 + 盈绿亏红配色）+ 洞察。"""
    with st.container(border=True, key="bento_marketing"):
        st.markdown("### 🚀 营销建议")
        if not mk:
            st.caption("（营销建议卡片加载失败）")
            return
        st.caption(mk.get("insight", ""))
        rows = mk.get("rows") or []
        if rows:
            labels = [r[0] for r in rows]
            values = [float(r[1]) for r in rows]
            colors = ["#34C759" if v >= 1.0 else "#FF3B30" for v in values]
            _render_frame_chart("bar", labels, values,
                                mk.get("title", "各渠道类型 ROI"),
                                mk.get("y_title", "ROI"), tick, mk.get("chart_json"),
                                colors=colors, hline=1.0, height=430, decimals=2)
        elif mk.get("chart_json"):
            _render_figure_json(mk["chart_json"])
        else:
            st.caption("（渠道 ROI 图表加载失败）")


def _play_progress(tick: int, n: int, hold: int = 2) -> int:
    """轮播帧位置：0..n-1 循环推进，到末帧后停留 hold 拍再回起点。"""
    if n <= 1:
        return 0
    return min(tick % (n + hold), n - 1)


def _render_frame_chart(kind: str, labels: list, values: list[float], title: str,
                        y_title: str, tick: int, fallback_json: str | None,
                        column=None, colors: list | None = None,
                        hline: float | None = None, area: bool = False,
                        height: int = 300, decimals: int = 0) -> None:
    """渲染第 tick 帧的动态图表（tools.dashboard._progress_chart 轮播帧）。

    陈旧进程兜底：运行中的旧版 tools.dashboard 若没有 _progress_chart → 回退静态
    chart_json，不崩页面；两者都不可用 → 单行提示。column=None 时直接渲染。
    colors/hline/area/height 为 Bento 换肤参数（透传；旧进程不支持则忽略）。
    """
    fn = getattr(dashboard, "_progress_chart", None)
    try:
        fig_json = (fn(kind, labels, values, title, y_title, _play_progress(tick, len(labels)),
                       colors=colors, hline=hline, area=area, height=height,
                       decimals=decimals)
                    if fn is not None else fallback_json)
    except Exception:
        fig_json = fallback_json
    if fig_json:
        if column is not None:
            with column:
                _render_figure_json(fig_json)
        else:
            _render_figure_json(fig_json)
    elif column is not None:
        with column:
            st.caption("（图表渲染失败）")
    else:
        st.caption("（图表渲染失败）")


def _submit_message(pipeline, prompt: str, view: str) -> None:
    """把一次提问交给后台线程流式回答，并写入当前会话（聊天/看板共用，会话隔离）。

    - 会话按来源（chat/dashboard）分桶 → 看板问答不混入聊天历史、互不中断；
    - 立即把 user + 在途 assistant 消息落进会话 → 视图切换不再丢回答；
    - 生成跑后台线程 → 切走再回来，回答继续算完（ChatGPT 式）；
    - 看板态完成时算图表（answer_chart）随消息保存，历史回放一致。

    幂等提交（双重提问根治）：请求身份 = (view, conv_id, prompt)。同一身份的请求只被
    接受一次，重复到达的一律是同一提交事件的重放/竞态副本（chat_input 前端 trigger
    值重放——`_request_full_app_rerun` 等交互路径会重放 trigger widget 状态、双击
    回车、WebSocket 重连重发），直接丢弃。身份释放只有两个途径：
    ① 同会话接受了「不同」的提问（用户转向新问题 → 旧身份作废）；
    ② 注册后 30s 自然过期（远超任何 rerun 级重放窗口；用户稍后真想重问同句不受阻）。
    ⚠ 不按「回答完成」释放——重放副本常在回答完成后才到达（流式期间在途被闸门挡住、
    完成闸门已开的窗口），这正是此前「3s 窗口/完成即注销」两版仍偶发双提问的根因。
    """
    ss = st.session_state
    key = view  # "chat" | "dashboard" —— 各自维护当前会话指针
    conv_id = ss.view_current[key]
    if conv_id is None or conv_id not in ss.conversations:
        conv_id = uuid.uuid4().hex[:12]
        ss.conversations[conv_id] = {
            "title": prompt[:20],
            "thread_id": uuid.uuid4().hex[:12],
            "source": view,          # 会话归属：聊天 / 看板（侧边栏分栏展示）
            "messages": [],
        }
        ss.view_current[key] = conv_id
    conv = ss.conversations[conv_id]
    messages = conv["messages"]

    # —— 幂等闸门：同身份请求在 TTL 内只放行一次 ——
    # seen 登记只发生在真正 append 消息处；入队路径不入 seen（否则队列 drain 时
    # _submit_message(同身份) 会撞自己的闸门被丢，排队提问永远不执行）。
    # 在途重放（同身份已 seen）直接丢弃；入队重复由 _pending 自身的去重挡。
    req_id = (view, conv_id, prompt)
    seen = ss.setdefault("_inflight_reqs", {})
    if req_id in seen:
        # 同一提交的重放副本（或用户极快重发同一句）：丢弃，但给出可见提示——
        # 静默吞掉用户输入比偶发误拦更糟。在途/排队中的不同问题不受影响（见下）。
        st.toast("该问题刚刚已提交，正在处理中～ 如需重问请稍等片刻或换个问法")
        st.rerun()
        return

    # 该会话有回答在途 → 不丢提问：排进等待队列（带提交时的视图），当前回答完成后
    # main() 自动追问。此前直接 toast+return，用户打的字被静默吞掉（「思考中提问被卡掉」）。
    if messages and messages[-1].get("_streaming"):
        pending = ss.setdefault("_pending", [])
        if not any(p == (prompt, view) for p in pending):
            pending.append((prompt, view))   # 队列去重：双击/重放的重复排队不再入
        st.toast("上一条还在回答，已排队，完成后自动追问～")
        st.rerun()
        return

    # 真正受理：登记幂等身份（TTL 内同句重放一律丢弃）；用户转向新问题 → 同会话
    # 旧身份作废（重问上一句不再被 30s 窗口误拦）。
    now = time.time()
    for other in [r for r in seen if r[0] == view and r[1] == conv_id and r[2] != prompt]:
        del seen[other]
    seen[req_id] = now
    messages.append({"role": "user", "content": prompt})
    msg = {"role": "assistant", "content": "", "reasoning": "",
           "phase": "", "_streaming": True, "_t0": time.time()}
    messages.append(msg)
    ss._hero_dismissed = True       # 一经提交，四个示例问题不再复现（直到新会话/切视图）
    result_box: dict = {}
    threading.Thread(
        target=_stream_worker, daemon=True,
        args=(pipeline, prompt, conv, view, result_box, msg),
    ).start()
    if view == "dashboard":
        ss._dash["mode"] = "answer"   # 看板态收起卡片，累计展示对话
    ss._scroll = True
    st.rerun()


if __name__ == "__main__":
    main()
