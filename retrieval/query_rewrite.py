"""retrieval/query_rewrite.py：RAG query rewrite (three layers).

- Layer 1 · coreference resolution + completion: use conversation history to complete
  referential / elliptical follow-ups (fixes context-incomplete questions).
- Layer 2 · synonym expansion + generalization: map colloquial phrases to canonical metric terms
  for better keyword matching (fixes keyword mismatch).
- Layer 3 · intent decomposition + structuring: clarify vague questions into a focused metric query
  (fixes vague questions).

Deterministic where possible; the LLM is only called for genuinely referential / vague inputs.
"""

from __future__ import annotations

import re

# colloquial phrase → canonical metric term (extend freely)
SYNONYMS = {
    "卖了多少钱": "销售额", "卖了多少": "销售额", "卖了多少钱的": "销售额",
    "赚了多少": "净利润", "赚了多少钱": "净利润",
    "买了多少": "订单数", "多少单": "订单数",
    "回头客": "复购率", "回购": "复购率",
    "成交了多少": "GMV", "毛利": "毛利率",
}

# a follow-up question that refers to the prior turn (ellipsis / pronoun)
_REFERENTIAL_RE = re.compile(r"^(那|那个|换成|再|也|还有|继续|这个|上次|上回|刚才|那么|那这)")

# a question that states no concrete metric → vague (trailing ？ optional)
_VAGUE_RE = re.compile(r"(怎么样|如何|好不好|什么水平|表现如何|情况如何|效果如何|卖得如何|卖得怎样)？?$")


def _recent_history_text(history: list[dict], limit: int = 4) -> str:
    lines = []
    for h in history[-limit:]:
        if not isinstance(h, dict):
            continue
        if h.get("question"):
            lines.append(f"问：{h['question']}")
        if h.get("answer"):
            lines.append(f"答：{str(h['answer'])[:200]}")
    return "\n".join(lines)


def layer1_coreference(question: str, history: list[dict] | None, llm) -> str:
    """指代消解与问句补全：把「那净销售额呢」补全为自包含问句（需要历史上下文）。"""
    if not history or not _REFERENTIAL_RE.match(question.strip()):
        return question
    if llm is None:
        return question
    system = (
        "你是查询改写器。把用户的追问改写成一句自包含、完整的问题，补全被省略的指标、时间、维度。"
        "只输出改写后的问题，不要解释。"
    )
    user = f"对话历史：\n{_recent_history_text(history)}\n\n当前追问：{question}\n改写后："
    try:
        out = llm.complete(system, user).strip()
        return out or question
    except Exception:
        return question


def layer2_synonyms(question: str) -> str:
    """同义扩展与泛化：把口语化说法标注为规范指标词，提升检索关键词匹配。"""
    for colloquial, canonical in SYNONYMS.items():
        if colloquial in question and canonical not in question:
            question = question.replace(colloquial, f"{colloquial}（{canonical}）", 1)
    return question


def layer3_intent(question: str, llm) -> str:
    """意图拆解与结构化：把模糊提问澄清为明确的指标查询。"""
    if not _VAGUE_RE.search(question):
        return question
    if llm is None:
        return question
    system = (
        "你是查询改写器。把模糊或含糊的问题澄清为一句明确的指标查询，"
        "指明要查的指标（如 GMV/净销售额/订单数/复购率）与时间范围。只输出改写后的问句，不要解释。"
    )
    user = f"问题：{question}\n明确化后："
    try:
        out = llm.complete(system, user).strip()
        return out or question
    except Exception:
        return question


def rewrite_query(question: str, *, history: list[dict] | None = None, llm=None) -> str:
    """Apply the three rewrite layers, in order, and return the rewritten question."""
    question = layer1_coreference(question, history, llm)
    question = layer2_synonyms(question)
    question = layer3_intent(question, llm)
    return question