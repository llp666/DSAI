"""agent/config.py：环境变量加载与配置（.env 支持，无 dotenv 依赖）。"""

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: str | Path = PROJECT_ROOT / ".env") -> None:
    """极简 .env 解析：KEY=VALUE，忽略注释与空行，不覆盖已存在的环境变量。"""
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
    embedding: EmbeddingConfig | None = None
    alt_embedding: EmbeddingConfig | None = None
    rerank: RerankConfig | None = None


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


def load_config(env_path: str | Path | None = None) -> Config:
    load_dotenv(env_path or PROJECT_ROOT / ".env")
    langfuse_enabled = os.environ.get("LANGFUSE_ENABLED", "false").lower() == "true"
    emb = EmbeddingConfig(
        provider=os.environ.get("EMBEDDING_PROVIDER", "gitee"),
        base_url=os.environ.get("EMBEDDING_BASE_URL", ""),
        api_key=os.environ.get("EMBEDDING_API_KEY", ""),
        model=os.environ.get("EMBEDDING_MODEL", ""),
        dimensions=int(os.environ.get("EMBEDDING_DIMENSIONS", "1024")),
        instruction=os.environ.get("EMBEDDING_INSTRUCTION", ""),
        task=os.environ.get("EMBEDDING_TASK", "retrieval.query"),
        chroma_dir=os.environ.get("CHROMA_DIR", "warehouse/chroma"),
    )
    alt_emb = EmbeddingConfig(
        provider=os.environ.get("ALT_EMBEDDING_PROVIDER", "jina"),
        base_url=os.environ.get("ALT_EMBEDDING_BASE_URL", ""),
        api_key=os.environ.get("ALT_EMBEDDING_API_KEY", ""),
        model=os.environ.get("ALT_EMBEDDING_MODEL", ""),
        dimensions=int(os.environ.get("ALT_EMBEDDING_DIMENSIONS", "1024")),
        task=os.environ.get("ALT_EMBEDDING_TASK", "retrieval.query"),
        chroma_dir=os.environ.get("CHROMA_DIR", "warehouse/chroma"),
    )
    rerank = RerankConfig(
        base_url=os.environ.get("RERANK_BASE_URL", ""),
        api_key=os.environ.get("RERANK_API_KEY", ""),
        model=os.environ.get("RERANK_MODEL", ""),
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
            os.environ.get("SEMANTIC_LAYER_PATH", "semantic_layer/metrics.yaml")
        ),
        meta_dir=Path(os.environ.get("META_DIR", "meta")),
        reference_date=(os.environ.get("REFERENCE_DATE") or None),
        embedding=(emb if emb.api_key and emb.model else None),
        alt_embedding=(alt_emb if alt_emb.api_key and alt_emb.model else None),
        rerank=(rerank if rerank.api_key and rerank.model else None),
    )


def resolve(path: Path) -> Path:
    """将相对路径解析到项目根。"""
    return path if path.is_absolute() else PROJECT_ROOT / path
