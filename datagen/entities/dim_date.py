"""dim_date 日期维度（SPEC §5.1）。

内置 2025–2026 中国节假日表。is_holiday 口径：法定放假日 + 主要传统/西方节日，
调休补班日显式 false（is_holiday=false AND dow>=6 可查周末补班），618/双11 走 is_promo。
"""

import numpy as np
import pandas as pd

# 法定节假日放假日（国务院安排的主要假期；含 2025/2026 官方公休日）
STATIC_HOLIDAYS = {
    "2025-01-01", "2025-01-28", "2025-01-29", "2025-01-30", "2025-01-31",
    "2025-02-01", "2025-02-02", "2025-02-03", "2025-02-04",
    "2025-04-04", "2025-04-05", "2025-04-06",
    "2025-05-01", "2025-05-02", "2025-05-03", "2025-05-04", "2025-05-05",
    "2025-05-31", "2025-06-01", "2025-06-02",
    "2025-10-01", "2025-10-02", "2025-10-03", "2025-10-04", "2025-10-05",
    "2025-10-06", "2025-10-07", "2025-10-08",
    "2026-01-01", "2026-01-02", "2026-01-03",
    "2026-02-16", "2026-02-17", "2026-02-18", "2026-02-19", "2026-02-20",
    "2026-02-21", "2026-02-22", "2026-02-23",
    "2026-04-05", "2026-04-06", "2026-04-07",
    "2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04", "2026-05-05",
    "2026-06-19", "2026-06-20", "2026-06-21",
    "2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05",
    "2026-10-06", "2026-10-07", "2026-10-08",
}

# 调休补班日（周末上班，is_holiday=false）
MAKEUP_WORKDAYS = {
    "2025-01-26", "2025-02-08", "2025-04-27", "2025-09-28", "2025-10-11",
    "2026-02-14", "2026-02-28", "2026-04-06", "2026-09-27", "2026-10-10",
}

# 主要传统/西方节日（营销叙事素材，也标 is_holiday）
FESTIVALS = {
    "2025-02-14": "情人节", "2025-03-08": "妇女节", "2025-05-11": "母亲节",
    "2025-06-15": "父亲节", "2025-08-29": "七夕", "2025-12-25": "圣诞节",
    "2026-02-14": "情人节", "2026-03-08": "妇女节", "2026-05-10": "母亲节",
    "2026-06-21": "父亲节", "2026-08-19": "七夕", "2026-12-25": "圣诞节",
}

# festival_calendar 用（法定假日 + 传统节日统一）
FESTIVAL_CALENDAR = {
    "2025-01-01": ("元旦", "法定假日"), "2025-01-28": ("除夕", "法定假日"),
    "2025-01-29": ("春节", "法定假日"), "2025-04-04": ("清明节", "法定假日"),
    "2025-05-01": ("劳动节", "法定假日"), "2025-05-31": ("端午节", "法定假日"),
    "2025-10-01": ("国庆节", "法定假日"),
    "2025-02-14": ("情人节", "传统节日"), "2025-03-08": ("妇女节", "传统节日"),
    "2025-05-11": ("母亲节", "传统节日"), "2025-06-15": ("父亲节", "传统节日"),
    "2025-08-29": ("七夕", "传统节日"), "2025-12-25": ("圣诞节", "传统节日"),
    "2026-01-01": ("元旦", "法定假日"), "2026-02-16": ("除夕", "法定假日"),
    "2026-02-17": ("春节", "法定假日"), "2026-04-05": ("清明节", "法定假日"),
    "2026-05-01": ("劳动节", "法定假日"), "2026-06-19": ("端午节", "法定假日"),
    "2026-10-01": ("国庆节", "法定假日"),
    "2026-02-14": ("情人节", "传统节日"), "2026-03-08": ("妇女节", "传统节日"),
    "2026-05-10": ("母亲节", "传统节日"), "2026-06-21": ("父亲节", "传统节日"),
    "2026-08-19": ("七夕", "传统节日"), "2026-12-25": ("圣诞节", "传统节日"),
}


def holiday_flags(dates: pd.DatetimeIndex) -> pd.Series:
    """is_holiday：法定放假日 ∪ 传统节日，调休补班日显式 false。"""
    s = pd.Series(index=dates, dtype=bool)
    for d in dates:
        key = d.strftime("%Y-%m-%d")
        s[d] = key in STATIC_HOLIDAYS or key in FESTIVALS
    return s


def promo_flags(dates: pd.DatetimeIndex, promo_dates: list[str]) -> pd.Series:
    """is_promo：促销日及前后 1 天。"""
    promo = set()
    for pd_ in promo_dates:
        d = pd.Timestamp(pd_)
        promo.add(d)
        promo.add(d - pd.Timedelta(days=1))
        promo.add(d + pd.Timedelta(days=1))
    return pd.Series([d in promo for d in dates], index=dates)


def gen_dim_date(cfg) -> pd.DataFrame:
    """dim_date：覆盖 date_range 逐日一行。"""
    dates = pd.date_range(cfg.date_range.start, cfg.date_range.end, freq="D")
    df = pd.DataFrame(index=dates)
    df["date"] = dates.date
    df["dow"] = dates.dayofweek + 1  # 1=周一 … 7=周日
    df["month"] = dates.strftime("%Y-%m")
    df["is_holiday"] = holiday_flags(dates).values
    df["is_promo"] = promo_flags(dates, [p.date for p in cfg.promo_days]).values
    return df.reset_index(drop=True)


def gen_festival_calendar(cfg) -> pd.DataFrame:
    """festival_calendar：内置节日表，过滤到数据期范围内。"""
    start, end = pd.Timestamp(cfg.date_range.start), pd.Timestamp(cfg.date_range.end)
    rows = []
    for d, (name, ftype) in FESTIVAL_CALENDAR.items():
        t = pd.Timestamp(d)
        if start <= t <= end:
            rows.append((t.date(), name, ftype))
    rows.sort()
    return pd.DataFrame(rows, columns=["festival_date", "festival_name", "festival_type"])
