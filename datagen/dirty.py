"""dirty.py：受控脏数据注入（SPEC §6）。

先采样 ID 存 manifest，再按 ID 执行 UPDATE——可复现、可回滚（重跑即覆盖）、
可被纠错模块定向取用。注入完成后 validate 跳过已知脏行（manifest 白名单）。

四类注入：
- null_rate 3%：仅可选列 users.city_tier / refunds.refund_reason，外键列绝不注入；
- enum_drift 1.5%：2025-Q1 订单 order_status 混入旧字符串码（配置 mapping 如 {"4": "已完成"}）；
- tz_shift：2025-Q1 created_at −8h（模拟 ETL 把 UTC 原值当本地时间写入，全分区）；
- late_arrival 2%：orders.updated_at +2~5 天（状态同步延迟，不动 status_log）。
"""

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from datagen.rng import make_rng


def _sample_ids(con: duckdb.DuckDBPyConnection, table: str, col: str,
                where: str, rate: float, rng) -> list[str]:
    """从表按条件采样 rate 比例的 ID（先采样存 manifest，再注入）。"""
    if rate <= 0:
        return []
    n = con.execute(f"SELECT count(*) FROM {table} WHERE {where}").fetchone()[0]
    if n == 0:
        return []
    k = max(1, int(round(n * rate)))
    rows = con.execute(
        f"SELECT {col} FROM {table} WHERE {where} ORDER BY {col}").fetchall()
    ids = [r[0] for r in rows]
    idx = rng.choice(len(ids), size=min(k, len(ids)), replace=False)
    return [ids[i] for i in idx]


def _chunk(ids: list[str], size: int = 500):
    for i in range(0, len(ids), size):
        yield ids[i:i + size]


def inject_dirty(cfg, con: duckdb.DuckDBPyConnection, manifest_path: Path) -> dict:
    """执行四类注入，返回 dirty_manifest 结构（已写盘）。"""
    rng = make_rng(cfg.seed, "dirty")
    manifest: dict = {"null_rate": {}, "enum_drift": {}, "tz_shift": {}, "late_arrival": {}}

    # ---- 1. 空值注入（可选列，非外键）----
    # users.city_tier
    ids_u = _sample_ids(con, "dim.users", "user_id", "city_tier IS NOT NULL",
                        cfg.dirty.null_rate, rng)
    for chunk in _chunk(ids_u):
        con.execute("UPDATE dim.users SET city_tier = NULL WHERE user_id IN (SELECT unnest(?))", [chunk])
    manifest["null_rate"]["dim.users"] = {"ids": ids_u}

    # refunds.refund_reason
    ids_r = _sample_ids(con, "ods.refunds", "refund_id", "refund_reason IS NOT NULL",
                        cfg.dirty.null_rate, rng)
    for chunk in _chunk(ids_r):
        con.execute("UPDATE ods.refunds SET refund_reason = NULL WHERE refund_id IN (SELECT unnest(?))", [chunk])
    manifest["null_rate"]["ods.refunds"] = {"ids": ids_r}

    # ---- 2. 枚举漂移（2025-Q1 订单 order_status）----
    mapping = cfg.dirty.enum_drift.mapping  # {"4": "已完成"}
    part_start, part_end = cfg.dirty.enum_drift.partition.split("-"), None
    # partition 形如 "2025-Q1"
    year = int(part_start[0]); q = int(part_start[1][1])
    q_start = pd.Timestamp(f"{year}-{(q-1)*3+1:02d}-01")
    q_end = q_start + pd.DateOffset(months=3) - pd.Timedelta(days=1)
    # 只采样会真正被改写的行（order_status 命中 mapping 的旧值），
    # 否则采样到的非目标状态行不会变脏，注入行数与 manifest 对不上。
    mapping_keys = ", ".join(f"'{v}'" for v in mapping)
    where_q1 = (f"created_at::DATE BETWEEN '{q_start.date()}' AND '{q_end.date()}' "
                f"AND order_status IN ({mapping_keys})")
    ids_ed = _sample_ids(con, "ods.orders", "order_id", where_q1,
                         cfg.dirty.enum_drift.rate, rng)
    for old_val, new_val in mapping.items():
        for chunk in _chunk(ids_ed):
            con.execute(
                f"UPDATE ods.orders SET order_status = '{new_val}' "
                f"WHERE order_id IN (SELECT unnest(?)) AND order_status = '{old_val}'",
                [chunk])
    manifest["enum_drift"]["ods.orders"] = {
        "partition": cfg.dirty.enum_drift.partition,
        "range": [str(q_start.date()), str(q_end.date())],
        "ids": ids_ed, "mapping": mapping,
    }

    # ---- 3. 时区错位（2025-Q1 created_at −8h，全量）----
    if cfg.dirty.tz_shift.partition == cfg.dirty.enum_drift.partition:
        h = abs(cfg.dirty.tz_shift.hours)
        op = "-" if cfg.dirty.tz_shift.hours > 0 else "+"
        con.execute(f"""
            UPDATE ods.orders SET created_at = created_at {op} INTERVAL {h} HOUR
            WHERE {where_q1}
        """)
    manifest["tz_shift"]["ods.orders"] = {
        "range": [str(q_start.date()), str(q_end.date())],
        "shift_hours": cfg.dirty.tz_shift.hours,
    }

    # ---- 4. 迟到分区（orders.updated_at +2~5 天，2%）----
    rate = cfg.dirty.late_arrival.rate
    ids_la = _sample_ids(con, "ods.orders", "order_id", "updated_at IS NOT NULL", rate, rng)
    delays = rng.integers(2, 6, len(ids_la))
    for i, chunk in enumerate(_chunk(ids_la)):
        d = delays[i * 500:(i + 1) * 500]
        for cid, delay in zip(chunk, d):
            con.execute(f"UPDATE ods.orders SET updated_at = updated_at + INTERVAL {int(delay)} DAY "
                        f"WHERE order_id = '{cid}'")
    manifest["late_arrival"]["ods.orders"] = {"ids": ids_la, "tail_days": 3}

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def load_manifest(manifest_path: Path) -> dict:
    """读 dirty_manifest，供 validate 跳过已知脏行。"""
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {}


def skip_sql(manifest: dict) -> dict[str, str]:
    """把 manifest 转成 validate 可用的"跳过条件"片段（表 → SQL 谓词）。"""
    skips: dict[str, str] = {}
    for group in manifest.values():
        for table, info in group.items():
            ids = info.get("ids", [])
            if not ids:
                continue
            # 按表的主键列映射
            pk = {"dim.users": "user_id", "ods.refunds": "refund_id",
                  "ods.orders": "order_id"}.get(table, "order_id")
            id_list = ", ".join(f"'{x}'" for x in ids)
            cond = f"{pk} NOT IN ({id_list})"
            if table in skips:
                skips[table] += f" AND {cond}"
            else:
                skips[table] = cond
    # 时区错位：orders Q1 的 created_at 被 −8h，skip 需要排除 Q1 的时间校验
    return skips
