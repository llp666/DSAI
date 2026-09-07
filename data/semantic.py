"""data/semantic.py：语义层五区块加载 + jsonschema 校验 + 关系图 BFS。

- 加载 entities/relationships/metrics/dimensions/context 五个 YAML；
- 用 jsonschema 做结构校验（fail-fast，进 CI 门禁）；
- 构建实体关系无向图，提供确定性 BFS 求最短 JOIN 路径（编译器依赖）。
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


def _build_graph(entities: dict, relationships: dict) -> dict:
    graph = {e: [] for e in entities}
    for name, rel in relationships.items():
        graph[rel["left"]].append((rel["right"], name))
        graph[rel["right"]].append((rel["left"], name))
    # 每个邻接列表按邻接实体名排序 → BFS 遍历顺序确定
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
        # 兼容「文件顶层带区块名」与「文件顶层即区块内容」两种写法
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
    """返回 start→target 的 BFS 最短路径：[(from实体, to实体, 关系名), ...]。"""
    if start == target:
        return []
    prev: dict[str, tuple[str, str]] = {}  # 实体 -> (前驱实体, 关系名)
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
    """从 start 出发覆盖所有 targets 的确定性 JOIN 链。

    返回 [(实体, 关系名), ...]：链上每个实体与其前驱按该关系名的 ON 条件 JOIN。
    targets 按名称排序 → 路径选择确定，同一查询永远编译出同一 SQL。
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
