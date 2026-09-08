"""tools/search.py：web-search tool for real-time questions (keenable).

Stage-3 reserved extension (docs/model.md): POST {base_url}/search with X-API-Key returns
[{title, url, snippet}]. The LLM decides when to call it — the casual tool round and the
business tool round both arm search_tool (stage 3-5), so real-time questions (天气/新闻/
汇率…) are grounded in live results instead of refused.

Two layers:
- web_search(): the network call (stdlib urllib, timeout 15s) → [{title, url, snippet}]
- web_search_tool(): wraps it in the diagnostic-tool contract (ToolResult / ToolError)
- search_tool_schema(): OpenAI function definition armed in the LLM tool rounds
"""

from __future__ import annotations

import json
import time
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
    ToolError(query_failed) so the caller can degrade honestly instead of crashing. One
    retry with a fresh connection: the API is occasionally flaky at the TLS layer (record
    layer failure) — a single retry usually lands, which keeps the LLM-driven search path
    demo-robust.
    """
    body = json.dumps({"query": query}).encode("utf-8")
    last_exc: Exception | None = None
    for attempt in range(2):
        req = Request(
            f"{cfg.base_url.rstrip('/')}/search",
            data=body,
            headers={"X-API-Key": cfg.api_key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=15) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            break
        except Exception as e:
            last_exc = e
            if attempt == 0:
                time.sleep(1.0)  # brief backoff, then a fresh connection
    else:
        raise ToolError("query_failed", f"搜索失败：{last_exc}", "稍后重试或换个问法") from last_exc
    results = (payload.get("results") or [])[:top_n]
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("snippet", "")}
        for r in results
    ]


def search_tool_schema() -> dict:
    """OpenAI function schema so the LLM can autonomously decide to call web search.

    Armed in the casual tool round (stage 3-5) and the business tool round — real-time /
    current-information questions (天气/新闻/汇率/股价…) are the model's call, not a keyword
    router's.
    """
    return {
        "type": "function",
        "function": {
            "name": "search_tool",
            "description": ("网页搜索工具：输入 query，返回相关网页的标题/链接/摘要。"
                            "用于需要当前/实时信息才能回答的问题（天气、新闻、最新资讯、汇率、"
                            "股价、热点等），或你的知识可能过时、需要联网核实的问题；"
                            "业务指标问题（GMV/销量/库存…）不要调用。"),
            "parameters": SEARCH_ARGS_SCHEMA,
        },
    }


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
