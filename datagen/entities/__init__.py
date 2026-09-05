"""entities 包：维度域生成模块。

每个 gen_* 以 make_rng(seed, module) 起手，模块名注册于 config.MODULES。
"""

from datagen.entities.channels import gen_channels
from datagen.entities.dim_date import gen_dim_date, gen_festival_calendar
from datagen.entities.products import gen_categories, gen_products
from datagen.entities.users import gen_users

__all__ = [
    "gen_channels",
    "gen_dim_date",
    "gen_festival_calendar",
    "gen_categories",
    "gen_products",
    "gen_users",
]
