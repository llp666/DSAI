"""agent/prompt_budget.py：Token 预算硬约束 + 优先级裁剪。

- tiktoken cl100k_base 近似计数（agnes 非 OpenAI 模型，用通用编码近似足够裁剪判断）；
- 裁剪优先级：先裁字段明细 → 后裁业务描述/示例问题；主键/外键列、表名/口径永不裁；
- 超限时逐步裁剪直到 ≤ budget，返回裁剪后的渲染文本 + 裁剪报告。
"""

from __future__ import annotations

import re

import tiktoken

_ENCODER = tiktoken.get_encoding("cl100k_base")

# 外键特征：列注释含「引用」或引用表名
_FK_RE = re.compile(r"引用\s*(?:dim|ods)\.\s*[a-z_]+", re.I)


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def _is_fk_column(column: dict, fks: list) -> bool:
    """外键列：出现在 fks 清单，或注释含引用关系。"""
    name = column.get("name", "")
    for fk in fks:
        on = fk.get("on", "") if isinstance(fk, dict) else ""
        if name in on or name in str(fk):
            return True
    return bool(_FK_RE.search(column.get("comment", "")))


def render_table_budgeted(doc: dict, budget: int) -> tuple[str, dict]:
    """渲染单表文档，按 budget 裁剪（tokens）。返回 (text, 裁剪报告)。"""
    report = {"table": doc["table"], "original_tokens": 0, "final_tokens": 0,
              "fields_cut": 0, "desc_cut": False, "sample_cut": False}
    fks = doc.get("fks", []) or []
    columns = doc.get("columns", []) or []

    # 分区：外键/主键列（永不裁） vs 普通列（可裁）
    essential_cols = [c for c in columns if c.get("name") in (doc.get("pk") or [])
                      or _is_fk_column(c, fks)]
    cuttable_cols = [c for c in columns if c not in essential_cols]

    def fields_block(cols: list) -> str:
        return "字段：\n" + "\n".join(
            f"    - {c['name']}: {c.get('type', '')} {c.get('comment', '')}".rstrip()
            for c in cols) + "\n"

    def full_fields(cuttable: list) -> str:
        """essential 列 + 当前 cuttable 列，合成单一「字段：」段（essential 排前）。"""
        return fields_block(essential_cols + cuttable)

    header = f"### 表 {doc['table']}\n描述：{doc.get('description', '')}\n"
    sample = doc.get("sample_questions", [])
    sample_text = f"示例问题：{'、'.join(sample)}\n" if sample else ""

    full = header + full_fields(cuttable_cols) + sample_text
    report["original_tokens"] = count_tokens(full)
    if report["original_tokens"] <= budget:
        report["final_tokens"] = report["original_tokens"]
        return full, report

    # 裁剪阶段 1：裁普通列（essential 永不裁，至少留 1 列防死循环）
    keep_cuttable = cuttable_cols[:]
    while len(keep_cuttable) > 1 and \
            count_tokens(header + full_fields(keep_cuttable) + sample_text) > budget:
        keep_cuttable = keep_cuttable[: max(1, len(keep_cuttable) // 2)]
    report["fields_cut"] = len(cuttable_cols) - len(keep_cuttable)

    # 裁剪阶段 2：裁示例问题
    cur = header + full_fields(keep_cuttable) + sample_text
    if count_tokens(cur) > budget:
        sample_text = ""
        report["sample_cut"] = True

    # 裁剪阶段 3：裁业务描述（最后，表名/外键列仍保留）
    cur = header + full_fields(keep_cuttable) + sample_text
    if count_tokens(cur) > budget:
        header = f"### 表 {doc['table']}\n"
        report["desc_cut"] = True

    final = header + full_fields(keep_cuttable) + sample_text
    report["final_tokens"] = count_tokens(final)
    return final, report


def render_tables_budgeted(tables: list[dict], budget: int) -> tuple[str, list[dict]]:
    """渲染多表，全局 Token 预算硬约束（tiktoken cl100k）。

    逐表渲染 + 裁剪，若仍超总预算则按顺序丢弃示例问题最少的表描述。
    返回 (渲染文本, 各表裁剪报告)。
    """
    parts = []
    reports = []
    for doc in tables:
        txt, rep = render_table_budgeted(doc, budget)
        parts.append(txt)
        reports.append(rep)
    full = "\n\n".join(parts)
    if count_tokens(full) <= budget:
        return full, reports
    # 仍超：逐表丢弃（先丢 token 最多的表描述），直到 ≤ budget
    order = sorted(range(len(parts)), key=lambda i: count_tokens(parts[i]), reverse=True)
    keep = list(range(len(parts)))
    for idx in order:
        if count_tokens("\n\n".join(parts[i] for i in keep)) <= budget:
            break
        keep.remove(idx)
    reports = [r for i, r in enumerate(reports) if i in keep]
    return "\n\n".join(parts[i] for i in sorted(keep)), reports
