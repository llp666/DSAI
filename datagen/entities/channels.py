"""channels 渠道维度（SPEC §5.1）。

付费渠道数由 cfg.ads.channels 决定，按类型配额分摊：
  feeds 4 / search 3 / cps 3 / sms 2（12 个付费渠道）→ + 自然流量 8 个 = 20 行（SPEC §3.2）；
小配置（channels=6）→ feeds2/search2/cps1/sms1 + 自然 8 = 14 行。
demo_cases.negative_roi_channels 若不在清单则补为额外渠道（保证校验器断言命中）。
"""

import pandas as pd

PAID_TYPES = ["feeds", "search", "cps", "sms"]
CHANNEL_NAMES = {"feeds": "信息流广告", "search": "搜索广告", "cps": "导购联盟",
                 "sms": "短信营销", "organic": "自然流量"}
# 12 个付费渠道的类型配额（全量）
TYPE_QUOTA = [4, 3, 3, 2]
N_ORGANIC = 8


def _distribute(n_paid: int) -> list[int]:
    """把 n_paid 个付费渠道分到 4 类，每类至少 1，比例贴近 TYPE_QUOTA。"""
    total = sum(TYPE_QUOTA)
    base = [max(1, round(n_paid * q / total)) for q in TYPE_QUOTA]
    while sum(base) > n_paid:
        i = max(range(len(base)), key=lambda k: (base[k], -TYPE_QUOTA[k]))
        if base[i] == 1:
            break
        base[i] -= 1
    while sum(base) < n_paid:
        i = min(range(len(base)), key=lambda k: (base[k], -TYPE_QUOTA[k]))
        base[i] += 1
    return base


def gen_channels(cfg, negative_ids: list[str] | None = None) -> pd.DataFrame:
    counts = _distribute(cfg.ads.channels)
    rows: list[tuple[str, str, str]] = []  # (channel_id, channel_name, channel_type)
    for ctype, n in zip(PAID_TYPES, counts):
        for i in range(1, n + 1):
            rows.append((f"channel_{ctype}_{i:02d}", CHANNEL_NAMES[ctype], ctype))
    for i in range(1, N_ORGANIC + 1):
        rows.append((f"channel_organic_{i:02d}", CHANNEL_NAMES["organic"], "organic"))

    existing = {r[0] for r in rows}
    for neg in negative_ids or []:
        if neg not in existing:
            rows.append((neg, "负ROI渠道", "cps"))
    return pd.DataFrame(rows, columns=["channel_id", "channel_name", "channel_type"])
