"""随机流管理（SPEC §4.3）。

关键性质：按模块名哈希派生独立随机流。新增/删除任何模块都不影响
其他模块的随机序列——否则后加一张表会导致全库数据变化，评测集作废。
"""

import hashlib

import numpy as np


def make_rng(seed: int, module: str) -> np.random.Generator:
    """按 seed + module 名派生独立随机流。"""
    key = f"{seed}:{module}".encode()
    ent = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    return np.random.default_rng(ent)
