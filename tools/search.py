"""tools/search.py：web-search tool for real-time questions (keenable).

Stage-3 reserved extension (docs/model.md): POST {base_url}/search with X-API-Key returns
[{title, url, snippet}]. Wired into chat_stream's real-time route (天气/新闻/最新资讯…),
so the agent answers those from live search results instead of refusing.

Two layers:
- web_search(): the network call (stdlib urllib, timeout 15s) → [{title, url, snippet}]
- web_search_tool(): wraps it in the diagnostic-tool contract (ToolResult / ToolError),
  reserved for a future LLM tool round in the graph.
"""

from __future__ import annotations

import json
from urllib.request import Request, urlopen

from tools.contract import ToolError, ToolResult, validate_args

# args JSON Schema (jsonschema-validated; for the contract wrapper)
SEARCH_ARGS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "搜索关键词/问题"},
        "top_n": {"type": "integer", "minimum": 1, "maximum": 8,
                  "description": "返回结果条数（默认 5）"},
    },
    "required": ["query"],
    "additionalProperties": False,
}


def web_search(query: str, cfg, top_n: int = 5) -> list[dict]:
    """Query the keenable search API; return [{title, url, snippet}] (top_n results).

    ``cfg`` is a SearchConfig (base_url + api_key). Any network/auth failure raises
    ToolError(query_failed) so the caller can degrade honestly instead of crashing."""
    body = json.dumps({"query": query}).encode("utf-8")
    req = Request(
        f"{cfg.base_url.rstrip('/')}/search",
        data=body,
        headers={"X-API-Key": cfg.api_key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise ToolError("query_failed", f"搜索失败：{e}", "稍后重试或换个问法") from e
    results = (payload.get("results") or [])[:top_n]
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("snippet", "")}
        for r in results
    ]


def web_search_tool(cfg, args: dict) -> ToolResult:
    """Contract wrapper: ToolResult(data=results, insights=title list) or ToolError(no_data)."""
    args = validate_args(SEARCH_ARGS_SCHEMA, args)
    rows = web_search(args["query"], cfg, args.get("top_n", 5))
    if not rows:
        raise ToolError("no_data", f"未搜到「{args['query']}」相关结果", "换个关键词试试")
    return ToolResult(
        data={"query": args["query"], "results": rows},
        insights=[r["title"] for r in rows],
        chart=None,
    )
