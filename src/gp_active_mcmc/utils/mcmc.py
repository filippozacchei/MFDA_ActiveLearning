from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.floating]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int_]


def extract_samples(chain: dict[str, Any], *, chain_key: str) -> FloatArray:
    """Extract parameters from a tinyDA chain into an array of shape (n_steps, n_dim).

    Assumes `chain[chain_key]` is an iterable of "links" that expose `.parameters`.
    """
    try:
        links = chain[chain_key]
    except KeyError as e:
        raise KeyError(f"chain_key={chain_key!r} not found in chain object.") from e

    return np.asarray([link.parameters for link in links], dtype=float)


def move_fraction_from_samples(samples: ArrayLike) -> float:
    """Fraction of steps where the state changes.

    This is a fallback diagnostic when explicit acceptance flags are unavailable.
    It is *not* guaranteed to equal the true acceptance rate.
    """
    s = np.asarray(samples, dtype=float)
    if s.ndim != 2:
        raise ValueError(f"samples must be 2D. Got shape {s.shape}.")
    if s.shape[0] < 2:
        return 0.0
    moved = np.any(np.diff(s, axis=0) != 0.0, axis=1)
    return float(np.mean(moved))


def acceptance_rate_from_accepted(accepted: ArrayLike) -> float:
    """Acceptance rate computed from explicit acceptance flags."""
    a = np.asarray(accepted, dtype=bool).ravel()
    if a.ndim != 1:
        raise ValueError(f"accepted must be 1D. Got shape {a.shape}.")
    return float(np.mean(a))


def hf_call_fraction(used_hf: ArrayLike) -> float:
    """Fraction of steps using the high-fidelity model."""
    u = np.asarray(used_hf, dtype=bool).ravel()
    if u.ndim != 1:
        raise ValueError(f"used_hf must be 1D. Got shape {u.shape}.")
    return float(np.mean(u))


def mean_subchain_length(subchain_length: ArrayLike) -> float:
    s = np.asarray(subchain_length, dtype=float).ravel()
    if s.ndim != 1:
        raise ValueError(f"subchain_length must be 1D. Got shape {s.shape}.")
    return float(np.mean(s))


def posterior_rmse(samples: ArrayLike, theta_true: ArrayLike, *, burnin: int = 0) -> float:
    """Mean Euclidean distance of samples to `theta_true` after burn-in."""
    s = np.asarray(samples, dtype=float)
    if s.ndim != 2:
        raise ValueError(f"samples must be 2D. Got shape {s.shape}.")
    th = np.asarray(theta_true, dtype=float).ravel()
    if th.ndim != 1 or th.shape[0] != s.shape[1]:
        raise ValueError("theta_true must be 1D and match the parameter dimension of samples.")
    b = int(burnin)
    if b < 0 or b > s.shape[0]:
        raise ValueError("burnin must be between 0 and n_steps.")
    err = s[b:] - th[None, :]
    return float(np.mean(np.sqrt(np.sum(err**2, axis=1))))
