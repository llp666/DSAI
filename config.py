"""config.py：environment loading & config (.env support, no dotenv dependency)."""

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def load_dotenv(path: str | Path = PROJECT_ROOT / ".env") -> None:
    """Minimal .env parser: KEY=VALUE, skips comments/blanks, never overrides existing env vars."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


@dataclass
class LLMConfig:
    provider: str
    model: str
    base_url: str
    api_key: str
    max_tokens: int = 16000
    max_input_tokens: int = 128000
    temperature: float = 0.0


@dataclass
class EmbeddingConfig:
    provider: str = "gitee"
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    dimensions: int = 1024
    instruction: str = ""
    task: str = "retrieval.query"
    chroma_dir: str = "warehouse/chroma"


@dataclass
class RerankConfig:
    base_url: str = ""
    api_key: str = ""
    model: str = ""


@dataclass
class SearchConfig:
    """Web-search API (keenable — docs/model.md, stage-3 reserved extension).

    Real-time questions (天气/新闻/最新…) are answered from live search results instead of
    the agent refusing. POST {base_url}/search with X-API-Key returns [{title, url, snippet}]."""
    base_url: str
    api_key: str


@dataclass
class Config:
    llm: LLMConfig
    alt_llm: LLMConfig | None
    langfuse_host: str | None
    langfuse_public_key: str | None
    langfuse_secret_key: str | None
    langfuse_enabled: bool
    duckdb_path: Path
    semantic_layer_path: Path
    meta_dir: Path
    reference_date: str | None = None
    # per-question Langfuse flush wait (seconds); 0 = fire-and-forget. The SDK exports from
    # its own background thread, so waiting only matters so short-lived processes (eval
    # scripts) don't exit before the last traces ship — cap it so an unreachable Langfuse
    # server can't stall the answer path (it retries with backoff for ~50s otherwise).
    langfuse_flush_timeout: float = 2.0
    embedding: EmbeddingConfig | None = None
    alt_embedding: EmbeddingConfig | None = None
    rerank: RerankConfig | None = None
    # rerank is opt-in. A/B on the 30-question retrieval set: RRF-only Recall@3 88.9% /
    # MRR 0.950 vs RRF+jina-rerank 88.9% / 0.894 — retrieve_hybrid truncates to Top-3
    # *before* reranking, so the reranker can only reorder (Recall@3 unchanged by
    # construction) and it reorders worse, at ~2s per question. Kept as a seam:
    # RERANK_ENABLED=true re-enables it, `eval.run_retrieval_eval --hybrid --rerank`
    # reproduces the A/B regardless of this switch.
    rerank_enabled: bool = False
    search: SearchConfig | None = None


def _llm_from_env(prefix: str) -> LLMConfig | None:
    api_key = os.environ.get(f"{prefix}_API_KEY", "").strip()
    model = os.environ.get(f"{prefix}_MODEL", "").strip()
    base_url = os.environ.get(f"{prefix}_BASE_URL", "").strip()
    provider = os.environ.get(f"{prefix}_PROVIDER", "").strip()
    if not api_key or not model or not base_url:
        return None
    return LLMConfig(
        provider=provider or "openai-compatible",
        model=model,
        base_url=base_url,
        api_key=api_key,
        max_tokens=int(os.environ.get(f"{prefix}_MAX_TOKENS", "16000")),
        max_input_tokens=int(os.environ.get(f"{prefix}_MAX_INPUT_TOKENS", "128000")),
        temperature=float(os.environ.get(f"{prefix}_TEMPERATURE", "0")),
    )


def _embedding_from_env(prefix: str, default_provider: str) -> EmbeddingConfig | None:
    emb = EmbeddingConfig(
        provider=os.environ.get(f"{prefix}_PROVIDER", default_provider),
        base_url=os.environ.get(f"{prefix}_BASE_URL", ""),
        api_key=os.environ.get(f"{prefix}_API_KEY", ""),
        model=os.environ.get(f"{prefix}_MODEL", ""),
        dimensions=int(os.environ.get(f"{prefix}_DIMENSIONS", "1024")),
        instruction=os.environ.get(f"{prefix}_INSTRUCTION", ""),
        task=os.environ.get(f"{prefix}_TASK", "retrieval.query"),
        chroma_dir=os.environ.get("CHROMA_DIR", "warehouse/chroma"),
    )
    return emb if emb.api_key and emb.model else None


def load_config(env_path: str | Path | None = None) -> Config:
    load_dotenv(env_path or PROJECT_ROOT / ".env")
    langfuse_enabled = os.environ.get("LANGFUSE_ENABLED", "false").lower() == "true"
    rerank = RerankConfig(
        base_url=os.environ.get("RERANK_BASE_URL", ""),
        api_key=os.environ.get("RERANK_API_KEY", ""),
        model=os.environ.get("RERANK_MODEL", ""),
    )
    search = SearchConfig(
        base_url=os.environ.get("SEARCH_BASE_URL", "https://api.keenable.ai/v1"),
        api_key=os.environ.get("SEARCH_API_KEY", ""),
    )
    return Config(
        llm=_llm_from_env("LLM"),
        alt_llm=_llm_from_env("ALT_LLM"),
        langfuse_host=os.environ.get("LANGFUSE_HOST") or None,
        langfuse_public_key=os.environ.get("LANGFUSE_PUBLIC_KEY") or None,
        langfuse_secret_key=os.environ.get("LANGFUSE_SECRET_KEY") or None,
        langfuse_enabled=langfuse_enabled,
        duckdb_path=Path(os.environ.get("DUCKDB_PATH", "warehouse/ecommerce.duckdb")),
        semantic_layer_path=Path(
            os.environ.get("SEMANTIC_LAYER_PATH", "data/semantic_layer/metrics.yaml")
        ),
        meta_dir=Path(os.environ.get("META_DIR", "meta")),
        reference_date=(os.environ.get("REFERENCE_DATE") or None),
        langfuse_flush_timeout=float(os.environ.get("LANGFUSE_FLUSH_TIMEOUT", "2")),
        embedding=_embedding_from_env("EMBEDDING", "gitee"),
        alt_embedding=_embedding_from_env("ALT_EMBEDDING", "jina"),
        rerank=(rerank if rerank.api_key and rerank.model else None),
        rerank_enabled=os.environ.get("RERANK_ENABLED", "false").lower() == "true",
        search=(search if search.api_key else None),
    )


def resolve(path: Path) -> Path:
    """Resolve a relative path against the project root."""
    return path if path.is_absolute() else PROJECT_ROOT / path