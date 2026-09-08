"""compile/executor.py：DuckDB read-only executor (Agent side is always read_only=True)."""

from __future__ import annotations

from pathlib import Path

import duckdb


class Executor:
    def __init__(self, db_path: Path):
        # read_only=True: the generator owns writes, the Agent only reads
        self._con = duckdb.connect(str(db_path), read_only=True)

    def execute(self, sql: str):
        """Run a read-only query, return the row list."""
        return self._con.execute(sql).fetchall()

    def fetchone(self, sql: str):
        rows = self._con.execute(sql).fetchall()
        return rows[0] if rows else None

    def explain_dry_run(self, sql: str) -> str | None:
        """EXPLAIN dry-run: validate SQL executability (no data side effect), return error or None."""
        try:
            self._con.execute(f"EXPLAIN {sql}").fetchall()
            return None
        except Exception as e:
            return str(e)

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> "Executor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()