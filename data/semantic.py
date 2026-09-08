"""data/semantic.py：load + validate 5-block semantic-layer YAML, build entity graph for BFS joins.

- loads entities / relationships / metrics / dimensions / context YAML
- jsonschema validation (fail-fast, CI gate)
- builds an undirected entity-relationship graph for deterministic BFS shortest JOIN paths
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import yaml
from jsonschema import validate
from jsonschema.exceptions import ValidationError as JsValidationError

BLOCK_FILES = {
    "entities": "entities.yaml",
    "relationships": "relationships.yaml",
    "metrics": "metrics.yaml",
    "dimensions": "dimensions.yaml",
    "context": "context.yaml",
}


class SemanticLayerError(RuntimeError):
    pass


@dataclass
class SemanticLayer:
    entities: dict
    relationships: dict
    metrics: dict
    dimensions: dict
    context: dict
    graph: dict  # entity -> list[(neighbor_entity, relationship_name)]

    def resolve_table(self, entity: str) -> str:
        return self.entities[entity]["table"]

    def metric_names(self) -> list[str]:
        return list(self.metrics)

    def dimension_names(self) -> list[str]:
        return list(self.dimensions)

    def format_dimensions(self, with_location: bool = False) -> str:
        """Render the dimension list as prompt text; ``with_location`` appends table.column."""
        lines = []
        for name, d in self.dimensions.items():
            vals = " / ".join(str(v) for v in d["values"])
            loc = f"，表 {d['table']}.{d['column']}" if with_location else ""
            lines.append(f"- {name}（{d['display_name']}{loc}）：可选值 {vals}")
        return "\n".join(lines)


def _build_graph(entities: dict, relationships: dict) -> dict:
    graph = {e: [] for e in entities}
    for name, rel in relationships.items():
        graph[rel["left"]].append((rel["right"], name))
        graph[rel["right"]].append((rel["left"], name))
    # sort each adjacency list by neighbor name → deterministic BFS traversal order
    for e in graph:
        graph[e].sort(key=lambda x: x[0])
    return graph


def load_semantic_layer(dir_path: Path, schema_path: Path) -> SemanticLayer:
    data: dict = {}
    for block, fname in BLOCK_FILES.items():
        p = Path(dir_path) / fname
        if not p.exists():
            raise SemanticLayerError(f"语义层文件缺失：{fname}")
        loaded = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        # tolerate both "top-level key == block name" and "top-level is the block itself"
        if isinstance(loaded, dict) and block in loaded:
            loaded = loaded[block]
        data[block] = loaded

    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    try:
        validate(instance=data, schema=schema)
    except JsValidationError as e:
        loc = ".".join(str(x) for x in e.absolute_path) or "<root>"
        raise SemanticLayerError(
            f"语义层 schema 校验失败 [{loc}]: {e.message}"
        ) from e

    graph = _build_graph(data["entities"], data["relationships"])
    return SemanticLayer(**data, graph=graph)


def bfs_path(graph: dict, start: str, target: str) -> list[tuple[str, str, str]]:
    """Shortest BFS path start→target: [(from_entity, to_entity, relationship_name), ...]."""
    if start == target:
        return []
    prev: dict[str, tuple[str, str]] = {}  # entity -> (predecessor, relationship_name)
    visited = {start}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for nxt, rel in graph[cur]:
            if nxt in visited:
                continue
            visited.add(nxt)
            prev[nxt] = (cur, rel)
            if nxt == target:
                edges: list[tuple[str, str, str]] = []
                node = target
                while node != start:
                    parent, rel = prev[node]
                    edges.append((parent, node, rel))
                    node = parent
                edges.reverse()
                return edges
            queue.append(nxt)
    raise SemanticLayerError(f"关系图中 {start} → {target} 无路径可达")


def build_join_chain(graph: dict, start: str, targets: set[str]) -> list[tuple[str, str]]:
    """Deterministic JOIN chain covering all targets from start.

    Returns [(entity, relationship_name), ...]; each entity joins its predecessor on the
    relationship's ON condition. Targets iterate in name order → deterministic SQL.
    """
    if not targets:
        return []
    chain: list[tuple[str, str]] = []
    connected: set[str] = {start}
    for t in sorted(targets):
        if t in connected:
            continue
        for a, b, rel in bfs_path(graph, start, t):
            if b not in connected:
                connected.add(b)
                chain.append((b, rel))
    return chain


if __name__ == "__main__":
    import sys

    here = Path(__file__).resolve().parent
    layer = load_semantic_layer(
        here / "semantic_layer",
        here / "semantic_layer/schema/semantic_layer.schema.json",
    )
    print(f"语义层 schema 校验 OK：{len(layer.metrics)} 指标、"
          f"{len(layer.dimensions)} 维度、{len(layer.entities)} 实体、"
          f"{len(layer.relationships)} 关系")
    sys.exit(0)