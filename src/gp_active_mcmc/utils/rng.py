from __future__ import annotations

import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)
