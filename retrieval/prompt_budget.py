"""retrieval/prompt_budget.py：token-budget hard constraint + priority trimming.

- tiktoken cl100k_base approximate counting (not an OpenAI model; generic encoder suffices for trimming);
- trim priority: field details first → description/sample questions; PK/FK columns, table name/caliber never trimmed;
- over budget, trim stepwise until <= budget; return rendered text + trim report.
"""

from __future__ import annotations

import re

import tiktoken

_ENCODER = tiktoken.get_encoding("cl100k_base")

# FK signature: column comment contains "引用 <schema>.<table>" or references a table name
_FK_RE = re.compile(r"引用\s*(?:dim|ods)\.\s*[a-z_]+", re.I)


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def _is_fk_column(column: dict, fks: list) -> bool:
    """FK column: appears in the fks list, or its comment states a reference relation."""
    name = column.get("name", "")
    for fk in fks:
        on = fk.get("on", "") if isinstance(fk, dict) else ""
        if name in on or name in str(fk):
            return True
    return bool(_FK_RE.search(column.get("comment", "")))


def render_table_budgeted(doc: dict, budget: int) -> tuple[str, dict]:
    """Render one table doc, trimmed to budget tokens. Returns (text, report dict)."""
    report = {"table": doc["table"], "original_tokens": 0, "final_tokens": 0,
              "fields_cut": 0, "desc_cut": False, "sample_cut": False}
    fks = doc.get("fks", []) or []
    columns = doc.get("columns", []) or []

    # partition: essential (PK/FK, never trimmed) vs cuttable columns
    essential_cols = [c for c in columns if c.get("name") in (doc.get("pk") or [])
                      or _is_fk_column(c, fks)]
    cuttable_cols = [c for c in columns if c not in essential_cols]

    def fields_block(cols: list) -> str:
        return "字段：\n" + "\n".join(
            f"    - {c['name']}: {c.get('type', '')} {c.get('comment', '')}".rstrip()
            for c in cols) + "\n"

    def full_fields(cuttable: list) -> str:
        """essential columns + current cuttable columns, one "字段：" block (essential first)."""
        return fields_block(essential_cols + cuttable)

    header = f"### 表 {doc['table']}\n描述：{doc.get('description', '')}\n"
    sample = doc.get("sample_questions", [])
    sample_text = f"示例问题：{'、'.join(sample)}\n" if sample else ""

    full = header + full_fields(cuttable_cols) + sample_text
    report["original_tokens"] = count_tokens(full)
    if report["original_tokens"] <= budget:
        report["final_tokens"] = report["original_tokens"]
        return full, report

    # trim stage 1: cut ordinary columns (essential never cut, keep >= 1 to avoid a loop)
    keep_cuttable = cuttable_cols[:]
    while len(keep_cuttable) > 1 and \
            count_tokens(header + full_fields(keep_cuttable) + sample_text) > budget:
        keep_cuttable = keep_cuttable[: max(1, len(keep_cuttable) // 2)]
    report["fields_cut"] = len(cuttable_cols) - len(keep_cuttable)

    # trim stage 2: drop sample questions
    cur = header + full_fields(keep_cuttable) + sample_text
    if count_tokens(cur) > budget:
        sample_text = ""
        report["sample_cut"] = True

    # trim stage 3: drop business description (last; table name / FK columns kept)
    cur = header + full_fields(keep_cuttable) + sample_text
    if count_tokens(cur) > budget:
        header = f"### 表 {doc['table']}\n"
        report["desc_cut"] = True

    final = header + full_fields(keep_cuttable) + sample_text
    report["final_tokens"] = count_tokens(final)
    return final, report


def render_tables_budgeted(tables: list[dict], budget: int) -> tuple[str, list[dict]]:
    """Render multiple tables under a global token budget.

    Renders/trims per table; if still over budget, drops the table descriptions with the
    fewest sample questions in order. Returns (rendered text, per-table reports).
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
    # still over: drop tables one by one (drop the most token-heavy first) until <= budget
    order = sorted(range(len(parts)), key=lambda i: count_tokens(parts[i]), reverse=True)
    keep = list(range(len(parts)))
    for idx in order:
        if count_tokens("\n\n".join(parts[i] for i in keep)) <= budget:
            break
        keep.remove(idx)
    reports = [r for i, r in enumerate(reports) if i in keep]
    return "\n\n".join(parts[i] for i in sorted(keep)), reports