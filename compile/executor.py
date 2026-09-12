"""compile/executor.py：DuckDB read-only executor (Agent side is always read_only=True).

Threading: one ``DuckDBPyConnection`` is **not** thread-safe, and the agent genuinely shares
one Executor across threads — a chat answer runs in a background thread (``app._stream_worker``)
while the dashboard's 30s fingerprint poll runs on the Streamlit script thread. Worse than a
crash, the failure is silent: ``execute()`` returns the connection itself and ``fetchall()``
reads "the last result", so two threads interleaving between those calls read *each other's*
rows (reproduced: 6 threads running a labeled count query got each other's labels, plus
``IndexError`` on the drained result set). A wrong number in an answer beats no answer at all
only in how much damage it does.

So hand every thread its own cursor off the same database handle (DuckDB's documented
multi-thread pattern) rather than serializing with a lock: cursors keep the dashboard's 1s
render loop from ever waiting on a chat query.
"""

from __future__ import annotations

import threading
from pathlib import Path

import duckdb


class Executor:
    def __init__(self, db_path: Path):
        # read_only=True: the generator owns writes, the Agent only reads
        self._db = duckdb.connect(str(db_path), read_only=True)
        self._local = threading.local()

    @property
    def _con(self):
        """This thread's own cursor (created on first use).

        Kept as ``_con`` so existing call sites (``tools/inventory.py`` reaches for
        ``executor._con`` directly) get per-thread isolation for free.
        """
        con = getattr(self._local, "con", None)
        if con is None:
            con = self._db.cursor()
            self._local.con = con
        return con

    def execute(self, sql: str):
        """Run a read-only query, return the row list."""
        return self._con.execute(sql).fetchall()

    def fetchone(self, sql: str):
        rows = self.execute(sql)
        return rows[0] if rows else None

    def explain_dry_run(self, sql: str) -> str | None:
        """EXPLAIN dry-run: validate SQL executability (no data side effect), return error or None."""
        try:
            self.execute(f"EXPLAIN {sql}")
            return None
        except Exception as e:
            return str(e)

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "Executor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
