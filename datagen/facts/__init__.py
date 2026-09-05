"""facts 包：事实域生成模块。"""

from datagen.facts.inventory import gen_inventory
from datagen.facts.marketing import gen_marketing_full
from datagen.facts.orders import gen_orders_full

__all__ = ["gen_orders_full", "gen_marketing_full", "gen_inventory"]
