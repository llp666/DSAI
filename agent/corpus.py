"""agent/corpus.py：检索语料构建（table_docs + 场景卡 → 校验 → 向量化文本）。

- 加载 meta/table_docs.json 与 semantic_layer/scenarios.yaml；
- 渲染为可向量化 document（含表名/描述/字段/示例问题）；
- jsonschema 校验，不通过即拒收并抛错（进 CI 门禁）。
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from jsonschema import validate
from jsonschema.exceptions import ValidationError as JsValidationError

from .config import resolve
from .tables import has_data


class CorpusError(RuntimeError):
    pass


def _render_table(doc: dict) -> str:
    cols = "；".join(
        f"{c['name']}({c.get('comment', '')})" for c in doc.get("columns", [])
    )
    samples = "、".join(doc.get("sample_questions", []))
    return (
        f"表 {doc['table']}：{doc.get('description', '')}。"
        f"字段：{cols}。"
        f"所属领域：{doc.get('domain', '')}。"
        f"典型示例问题：{samples}。"
    )


def _render_scenario(sc: dict) -> str:
    tables = "、".join(sc.get("involved_tables", []))
    questions = "、".join(sc.get("typical_questions", []))
    notes = "；".join(sc.get("notes", []))
    return (
        f"场景 {sc['name']}（{sc.get('domain', '')}）："
        f"涉及表 {tables}。典型问题：{questions}。"
        f"注意事项：{notes}。"
    )


def build_corpus(meta_dir: Path, scenarios_path: Path,
                 schema_path: Path, updated_at: str) -> list[dict]:
    """构建入库语料条目列表（已通过 jsonschema 校验）。"""
    table_docs = json.loads(
        (resolve(meta_dir) / "table_docs.json").read_text(encoding="utf-8"))
    scenarios = yaml.safe_load(
        resolve(scenarios_path).read_text(encoding="utf-8"))["scenarios"]
    schema = json.loads(resolve(schema_path).read_text(encoding="utf-8"))

    entries: list[dict] = []
    for key, doc in table_docs.items():
        entry = {
            "id": key,
            "doc_type": "table",
            "domain": doc.get("domain", ""),
            "document": _render_table(doc),
            "metadata": {"updated_at": updated_at, "tables": [key],
                         "has_data": has_data(key)},
        }
        entries.append(entry)
    for sc in scenarios:
        entry = {
            "id": sc["id"],  # 已是 scn_ 前缀
            "doc_type": "scenario",
            "domain": sc.get("domain", ""),
            "document": _render_scenario(sc),
            "metadata": {
                "updated_at": updated_at,
                "tables": sc.get("involved_tables", []),
                "has_data": True,  # 场景卡涉及真实表
            },
        }
        entries.append(entry)

    for e in entries:
        try:
            validate(instance=e, schema=schema)
        except JsValidationError as err:
            raise CorpusError(
                f"语料条目校验失败 [{e['id']}]: "
                f"{'.'.join(str(x) for x in err.absolute_path) or '<root>'} {err.message}"
            ) from err
    return entries
