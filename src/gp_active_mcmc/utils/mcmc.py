from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int_]


def extract_samples(chain: dict[str, Any], *, chain_key: str) -> FloatArray:
    """Extract parameter samples from a tinyDA chain object.

    This is a small compatibility helper that converts the object returned by
    `tinyDA.sample(...)` into a plain NumPy array.

    Expected tinyDA structure
    -------------------------
    This function assumes that `chain[chain_key]` is an iterable of *links*, where each
    link exposes a `.parameters` attribute (a 1D parameter vector).

    Parameters
    ----------
    chain
        Object returned by `tinyDA.sample`. In practice this behaves like a dictionary
        that maps chain keys (e.g. ``"chain_0"`` or ``"chain_coarse_0"``) to sequences of
        chain links.
    chain_key
        Key identifying which chain to extract.

    Returns
    -------
    samples
        Sample matrix of shape ``(n_steps, n_dim)`` where:

        - ``n_steps`` is the number of MCMC iterations stored in the chain,
        - ``n_dim`` is the parameter dimension.

    Raises
    ------
    KeyError
        If `chain_key` is not present in the chain object.
    TypeError
        If the links do not expose a `.parameters` attribute or cannot be iterated.

    Notes
    -----
    This function performs no thinning or burn-in removal. Those should be done at the
    `MCMCChain` level (see [`MCMCChain.burn_in`][gp_active_mcmc.inference.chain.MCMCChain.burn_in]).
    """
    try:
        links = chain[chain_key]
    except KeyError as e:
        raise KeyError(f"chain_key={chain_key!r} not found in chain object.") from e

    try:
        return np.asarray([link.parameters for link in links], dtype=float)
    except Exception as e:
        raise TypeError(
            "Could not extract samples: expected chain[chain_key] to be an iterable of "
            "objects exposing a `.parameters` attribute."
        ) from e


def move_fraction_from_samples(samples: ArrayLike) -> float:
    """Compute the fraction of steps where the state changes.

    This is a *fallback* diagnostic used when explicit acceptance flags are not
    available from the sampler. It measures how often consecutive states differ.

    Parameters
    ----------
    samples
        Sample matrix of shape ``(n_steps, n_dim)``.

    Returns
    -------
    move_fraction
        Fraction of steps (between 0 and 1) for which the parameter vector changes
        compared to the previous step. Returns 0.0 if fewer than two samples are given.

    Raises
    ------
    ValueError
        If `samples` is not a 2D array.

    Notes
    -----
    - This is not guaranteed to equal the true Metropolis acceptance rate.
      For instance, deterministic updates, rounding, or other sampler internals may
      cause differences.
    - The first sample has no "previous step" and is not counted.
    """
    s = np.asarray(samples, dtype=float)
    if s.ndim != 2:
        raise ValueError(f"samples must be 2D. Got shape {s.shape}.")
    if s.shape[0] < 2:
        return 0.0

    moved = np.any(np.diff(s, axis=0) != 0.0, axis=1)
    return float(np.mean(moved))


def acceptance_rate_from_accepted(accepted: ArrayLike) -> float:
    """Compute acceptance rate from explicit acceptance flags.

    Parameters
    ----------
    accepted
        Boolean array-like of shape ``(n_steps,)`` indicating whether each proposal
        was accepted.

    Returns
    -------
    acceptance_rate
        Mean of the acceptance indicators.

    Raises
    ------
    ValueError
        If `accepted` is not one-dimensional.
    """
    a = np.asarray(accepted, dtype=bool).ravel()
    if a.ndim != 1:
        raise ValueError(f"accepted must be 1D. Got shape {a.shape}.")
    return float(np.mean(a))


def hf_call_fraction(used_hf: ArrayLike) -> float:
    """Compute the fraction of steps that used the high-fidelity (HF) model.

    Parameters
    ----------
    used_hf
        Boolean array-like of shape ``(n_steps,)`` where True indicates an HF evaluation
        occurred at that step.

    Returns
    -------
    hf_fraction
        Mean of the HF-usage indicators.

    Raises
    ------
    ValueError
        If `used_hf` is not one-dimensional.
    """
    u = np.asarray(used_hf, dtype=bool).ravel()
    if u.ndim != 1:
        raise ValueError(f"used_hf must be 1D. Got shape {u.shape}.")
    return float(np.mean(u))


def mean_subchain_length(subchain_length: ArrayLike) -> float:
    """Compute the mean subchain length (subsampling rate) over a history.

    Parameters
    ----------
    subchain_length
        Array-like subchain length history.

    Returns
    -------
    mean_length
        Mean of the provided subchain lengths.

    Raises
    ------
    ValueError
        If `subchain_length` is not one-dimensional.
    """
    s = np.asarray(subchain_length, dtype=float).ravel()
    if s.ndim != 1:
        raise ValueError(f"subchain_length must be 1D. Got shape {s.shape}.")
    return float(np.mean(s))


def posterior_rmse(samples: ArrayLike, theta_true: ArrayLike, *, burn_in: int = 0) -> float:
    """Compute a simple posterior error metric against a reference parameter.

    The reported value is the mean Euclidean distance between each sample and a
    provided reference parameter vector `theta_true`, after discarding burn-in.

    Parameters
    ----------
    samples
        Sample matrix of shape ``(n_steps, n_dim)``.
    theta_true
        Reference parameter vector of shape ``(n_dim,)``.
    burn_in
        Number of initial samples to discard before computing the statistic.

    Returns
    -------
    rmse
        Mean Euclidean distance from samples to `theta_true` (after burn-in).

    Raises
    ------
    ValueError
        If shapes are inconsistent or `burn_in` is outside `[0, n_steps]`.

    Notes
    -----
    Despite the name, this is not an RMSE in observation space; it is a parameter-space
    distance statistic that is sometimes convenient in toy problems with known truth.
    """
    s = np.asarray(samples, dtype=float)
    if s.ndim != 2:
        raise ValueError(f"samples must be 2D. Got shape {s.shape}.")

    th = np.asarray(theta_true, dtype=float).ravel()
    if th.ndim != 1 or th.shape[0] != s.shape[1]:
        raise ValueError("theta_true must be 1D and match the parameter dimension of samples.")

    b = int(burn_in)
    if b < 0 or b > s.shape[0]:
        raise ValueError("burn_in must be between 0 and n_steps.")

    err = s[b:] - th[None, :]
    return float(np.mean(np.sqrt(np.sum(err**2, axis=1))))
