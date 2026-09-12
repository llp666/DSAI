"""bootstrap.py：部署期自举——让「新环境 clone 下来就能跑」。

`warehouse/`、`meta/`、`warehouse/chroma` 都是 .gitignore 掉的生成物，所以任何新环境
（Streamlit Community Cloud / 别人的机器）拿到仓库时它们必然不存在，而启动链路对它们的
缺失反应并不一致：

- ``Retriever(meta_dir)`` 抛 FileNotFoundError（retrieval/retriever.py:13）
- ``Executor(db_path)`` 对 read_only=True 的连接抛「database does not exist」
  （compile/executor.py:5）
- ``VectorStore(chroma_dir)`` 反而 *不抛*：get_or_create_collection 建一个空集合，
  检索静默返回 0 条（retrieval/vector_store.py:28）——静默降级比报错更难查

本模块做两件事，**对本地开发零副作用**：

1. :func:`bridge_secrets` —— 把 Streamlit Secrets 注入 ``os.environ``。社区云只提供
   ``st.secrets``，而 config.py 只读 ``os.environ``/``.env``（config.py:126），
   没有这层桥接线上会「配置全空」而不是明确报错。
2. :func:`ensure_artifacts` —— 缺什么补什么（datagen 建库 → retrieval.ingest 建索引）。
   产物齐全时**立即返回，不导入 chromadb、不起子进程**，这是本地与测试不受影响的前提。

单独跑也行：``python bootstrap.py``（自建一套产物并打印进度）。
"""

from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
import threading
from pathlib import Path

from config import PROJECT_ROOT, load_config, resolve

# 演示/默认规模配置：行数远小于 scale_full，但日期区间一致（相对时间问法才成立）
DEFAULT_DATAGEN_CONFIG = "datagen/config/scale_demo.yaml"

# catalog 目录级空表（RAG 干扰项）张数（SPEC §8）。本地完整数仓是 270 张，语料
# 309 条 = 31 真实表 + 270 catalog + 8 场景卡——检索评测（Recall@3 88.9%）与
# 「数百张表」的规模前提都建立在这个数量上，少了它演示的检索行为对不上。
DEMO_CATALOG_TABLES = 270

# 冷启动时多个 session 可能同时进入 → 只让一个真去生成，其余等锁后复查
_lock = threading.Lock()

Progress = "callable(str) -> None"


def _log(progress, message: str) -> None:
    if progress is not None:
        try:
            progress(message)
        except Exception:
            pass


def bridge_secrets() -> int:
    """把 ``st.secrets`` 的标量键值写进 ``os.environ``（已存在的不覆盖），返回注入条数。

    必须在任何 ``load_config()`` 之前调用（app.py 顶层）。非 Streamlit 环境、
    没有 secrets.toml 时静默返回 0——本地照旧读 .env。
    """
    try:
        import streamlit as st

        items = list(st.secrets.items())
    except Exception:
        return 0  # 无 secrets 文件 / 非 Streamlit 运行上下文

    n = 0
    for key, value in items:
        if isinstance(value, (dict, list)):
            continue  # 嵌套结构不属于环境变量语义，跳过
        if key not in os.environ:
            os.environ[key] = str(value)
            n += 1
    return n


def _chroma_dir():
    """按配置解析 ChromaDB 目录；没有可用的 embedding 配置时返回 None。"""
    cfg = load_config()
    emb = cfg.embedding or cfg.alt_embedding
    return resolve(Path(emb.chroma_dir)) if emb else None


def _chroma_count(chroma_dir: Path) -> int:
    """已入库条数。

    直接以只读方式查 ChromaDB 的 sqlite（stdlib，实测 0.2ms），**不构造 PersistentClient**——
    后者要付 chromadb 的 import + 建连，实测 4.4s，而这个检查跑在每次脚本 rerun 上。
    目录/文件不存在返回 0，且不为此创建任何东西。
    """
    db = chroma_dir / "chroma.sqlite3"
    if not db.exists():
        return 0
    try:
        import sqlite3

        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            return con.execute(_CHROMA_COUNT_SQL, (_COLLECTION,)).fetchone()[0]
        finally:
            con.close()
    except Exception:
        # 未来 chromadb 改了表结构：宁可判为「已就绪」。误判「未就绪」的代价是每次冷启动
        # 重建几十秒，而误判「已就绪」只是检索降级——两害相权取其轻。
        return 1


# 与 retrieval/vector_store.py 的 COLLECTION 同名。此处刻意重复而不 import 那个模块：
# 它顶层 `import chromadb`，会把 4.4s 的导入拖进上面这个本该毫秒级的检查。
_COLLECTION = "dsai_corpus"
_CHROMA_COUNT_SQL = (
    "SELECT COUNT(*) FROM embeddings e "
    "JOIN segments s ON e.segment_id = s.id "
    "JOIN collections c ON s.collection = c.id "
    "WHERE c.name = ?"
)

# 进程内一旦确认齐备就不再复查（产物在进程生命周期内不会消失）
_ready = False


def missing_parts() -> set[str]:
    """返回缺失的部分，取值 ``{"db", "meta", "chroma"}`` 的子集。"""
    cfg = load_config()
    missing: set[str] = set()
    if not resolve(cfg.duckdb_path).exists():
        missing.add("db")
    if not (resolve(cfg.meta_dir) / "table_docs.json").exists():
        missing.add("meta")
    chroma = _chroma_dir()
    if chroma is None or _chroma_count(chroma) == 0:
        missing.add("chroma")
    return missing


def artifacts_ready() -> bool:
    """三项产物是否齐备（齐备时调用方应立即返回，不做任何额外动作）。

    结果在进程内缓存：这个函数位于每次 rerun 都会走到的启动路径上，必须廉价。
    """
    global _ready
    if _ready:
        return True
    if not missing_parts():
        _ready = True
    return _ready


def _datagen_cmd(cfg, *extra: str) -> list[str]:
    """datagen CLI 调用：路径显式传参，服从 DUCKDB_PATH / META_DIR 覆盖。"""
    db = resolve(cfg.duckdb_path)
    return [
        sys.executable, "-m", "datagen.cli",
        "--config", os.environ.get("DATAGEN_CONFIG", DEFAULT_DATAGEN_CONFIG),
        "--db", str(db),
        "--meta-root", str(resolve(cfg.meta_dir)),
        "--parquet-root", str(db.parent / "parquet"),
        *extra,
    ]


def _run(cmd: list[str], progress, label: str) -> bool:
    _log(progress, label)
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    for line in (proc.stdout or "").strip().splitlines()[-3:]:
        _log(progress, line)
    if proc.returncode != 0:
        _log(progress, f"失败（exit {proc.returncode}）："
                       f"{(proc.stderr or '').strip()[-500:]}")
        return False
    return True


def _run_datagen(progress) -> bool:
    """建数仓：先出数据，再补 catalog 空表（RAG 干扰项，见 DEMO_CATALOG_TABLES）。"""
    cfg = load_config()
    if not _run(_datagen_cmd(cfg), progress, "生成演示数仓（约 30~60s）…"):
        return False
    return _run(_datagen_cmd(cfg, "--catalog-only", str(DEMO_CATALOG_TABLES)), progress,
                f"补齐 catalog 空表 {DEMO_CATALOG_TABLES} 张（RAG 干扰项）…")


def _run_ingest(progress) -> bool:
    """建检索索引：复用 retrieval.ingest，把它的 stdout 逐行转给进度回调。"""
    _log(progress, "构建检索索引（embedding → ChromaDB，约 20~40s）…")
    buf = io.StringIO()
    try:
        from retrieval.ingest import main as ingest_main

        with contextlib.redirect_stdout(buf):
            rc = ingest_main()
    except Exception as e:  # 缺 EMBEDDING_* / 网络失败 / ChromaDB 出错
        for line in buf.getvalue().strip().splitlines():
            _log(progress, line)
        _log(progress, f"入库失败：{type(e).__name__}: {e}")
        return False
    for line in buf.getvalue().strip().splitlines():
        _log(progress, line)
    return rc == 0


def ensure_artifacts(progress=None) -> bool:
    """缺什么补什么；产物齐备时零开销返回 True。失败返回 False 并已通过 progress 说明。"""
    if artifacts_ready():
        return True

    with _lock:
        # 双检：等锁期间别的 session 可能已经把产物建好了
        missing = missing_parts()
        if not missing:
            return True
        _log(progress, "首次启动，检测到缺失产物：" + "、".join(sorted(missing)))

        if missing & {"db", "meta"} and not _run_datagen(progress):
            return False
        if missing_parts() & {"chroma"} and not _run_ingest(progress):
            return False

    return artifacts_ready()


if __name__ == "__main__":
    print(f"secrets → 环境变量：注入 {bridge_secrets()} 条")
    ok = ensure_artifacts(progress=lambda m: print(f"· {m}"))
    print("产物齐备 ✓" if ok else "自举失败 ✗")
    sys.exit(0 if ok else 1)
