"""agent/executor.py：DuckDB 只读执行器（对齐方案"Agent 侧一律 read_only=True"）。"""

from __future__ import annotations

from pathlib import Path

import duckdb


class Executor:
    def __init__(self, db_path: Path):
        # read_only=True：生成器独占读写，Agent 侧只读消费
        self._con = duckdb.connect(str(db_path), read_only=True)

    def execute(self, sql: str):
        """执行只读查询，返回行列表。"""
        return self._con.execute(sql).fetchall()

    def fetchone(self, sql: str):
        rows = self._con.execute(sql).fetchall()
        return rows[0] if rows else None

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> "Executor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
