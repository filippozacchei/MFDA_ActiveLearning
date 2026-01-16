from __future__ import annotations
import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def jitter_psd(mat: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Add diagonal jitter to improve numerical stability."""
    d = mat.shape[0]
    return mat + eps * np.eye(d)
