"""data/corpus.py：build the retrieval corpus (table_docs + scenario cards → validate → embeddable text).

- loads meta/table_docs.json and data/semantic_layer/scenarios.yaml;
- renders an embeddable document (table name / description / fields / sample questions);
- jsonschema-validates; rejects on failure (CI gate).
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from jsonschema import validate
from jsonschema.exceptions import ValidationError as JsValidationError

from config import resolve
from data.tables import has_data


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
    """Build the validated corpus entries."""
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
            "id": sc["id"],  # already scn_-prefixed
            "doc_type": "scenario",
            "domain": sc.get("domain", ""),
            "document": _render_scenario(sc),
            "metadata": {
                "updated_at": updated_at,
                "tables": sc.get("involved_tables", []),
                "has_data": True,  # scenario cards involve real tables
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