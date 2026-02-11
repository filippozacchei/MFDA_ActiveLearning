# gp_active_mcmc/inference/chain.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gp_active_mcmc.utils.mcmc import (
    acceptance_rate_from_accepted,
    hf_call_fraction,
    mean_subchain_length,
    move_fraction_from_samples,
    posterior_rmse,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int_]


def _as_2d_float(samples: ArrayLike) -> FloatArray:
    """Convert an array-like to a 2D float array.

    Parameters
    ----------
    samples
        Candidate sample array. Expected shape is ``(n_steps, n_dim)``.

    Returns
    -------
    samples_2d
        Array of dtype float with shape ``(n_steps, n_dim)``.

    Raises
    ------
    ValueError
        If the input is not two-dimensional.
    """
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"samples must be 2D (n_steps, n_dim). Got shape {arr.shape}.")
    return arr


def _as_1d_bool(x: ArrayLike, *, name: str, n: int | None) -> BoolArray:
    """Validate and convert an aligned boolean vector.

    Parameters
    ----------
    x
        Input array-like.
    name
        Name used in error messages.
    n
        Expected length (typically `n_steps` of the chain).

    Returns
    -------
    x_bool
        1D boolean array of length `n`.

    Raises
    ------
    ValueError
        If the input is not 1D or has the wrong length.
    """
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D. Got shape {arr.shape}.")
    if n is not None and arr.shape[0] != n:
        raise ValueError(f"{name} length {arr.shape[0]} does not match n_steps {n}.")
    return np.asarray(arr, dtype=np.bool_)


def _as_1d_int(x: ArrayLike, *, name: str, n: int | None) -> IntArray:
    """Validate and convert an aligned integer vector.

    Parameters
    ----------
    x
        Input array-like.
    name
        Name used in error messages.
    n
        Expected length (typically `n_steps` of the chain).

    Returns
    -------
    x_int
        1D integer array of length `n`.

    Raises
    ------
    ValueError
        If the input is not 1D or has the wrong length.
    """
    arr = np.asarray(x, dtype=np.int_).ravel()
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D. Got shape {arr.shape}.")
    if n is not None and arr.shape[0] != n:
        raise ValueError(f"{name} length {arr.shape[0]} does not match n_steps {n}.")
    return arr


def _validate_burn_in(burn_in: int, *, n_steps: int) -> int:
    """Validate burn-in length and return it as an `int`."""
    b = int(burn_in)
    if b < 0:
        raise ValueError("burn_in must be >= 0.")
    if b > n_steps:
        raise ValueError(f"burn_in={b} is larger than chain length {n_steps}.")
    return b


def _validate_thin(thin: int) -> int:
    """Validate thinning factor and return it as an `int`."""
    t = int(thin)
    if t <= 0:
        raise ValueError("thin must be a positive integer.")
    return t


@dataclass(frozen=True, slots=True)
class ChainExtras:
    """Per-step metadata aligned with an [`MCMCChain`][gp_active_mcmc.inference.chain.MCMCChain].

    `ChainExtras` stores optional arrays aligned one-to-one with the sample matrix in
    [`MCMCChain`][gp_active_mcmc.inference.chain.MCMCChain]. Keeping these fields separate
    makes the core chain representation predictable while still supporting diagnostics.

    Attributes
    ----------
    used_hf
        Boolean array of length ``n_steps`` indicating whether the high-fidelity model
        was used at each step (active workflows).
    accepted
        Boolean array of length ``n_steps`` indicating whether each proposal was accepted.
    subchain_length
        Integer array of length ``n_steps`` recording the subchain length (or subsampling
        rate) history, when available (adaptive workflows).
    """

    used_hf: BoolArray | None = None
    accepted: BoolArray | None = None
    subchain_length: IntArray | None = None

    def slice(self, sl: slice) -> ChainExtras:
        """Return a sliced view of extras.

        Parameters
        ----------
        sl
            Slice to apply.

        Returns
        -------
        extras
            New `ChainExtras` with all available fields sliced consistently.
        """

        def _s_bool(v: BoolArray | None) -> BoolArray | None:
            return None if v is None else v[sl]

        def _s_int(v: IntArray | None) -> IntArray | None:
            return None if v is None else v[sl]

        return ChainExtras(
            used_hf=_s_bool(self.used_hf),
            accepted=_s_bool(self.accepted),
            subchain_length=_s_int(self.subchain_length),
        )


@dataclass(frozen=True, slots=True)
class MCMCChain:
    """Immutable container for MCMC samples and aligned diagnostics.

    The chain is represented as:

    - `samples`: a 2D array of shape ``(n_steps, n_dim)``
    - `extras`: optional per-step diagnostics aligned with samples

    The class provides lightweight post-processing utilities (burn-in removal,
    thinning, and summary statistics) without mutating the original object.
    """

    samples: FloatArray
    extras: ChainExtras = field(default_factory=ChainExtras)

    @classmethod
    def from_arrays(
        cls,
        *,
        samples: ArrayLike,
        used_hf: ArrayLike | None = None,
        accepted: ArrayLike | None = None,
        subchain_length: ArrayLike | None = None,
    ) -> MCMCChain:
        """Construct an `MCMCChain` from raw arrays.

        Parameters
        ----------
        samples
            Sample matrix of shape ``(n_steps, n_dim)``.
        used_hf
            Optional boolean vector of length ``n_steps``.
        accepted
            Optional boolean vector of length ``n_steps``.
        subchain_length
            Optional integer vector of length ``n_steps``.

        Returns
        -------
        chain
            A validated `MCMCChain`.
        """
        s = _as_2d_float(samples)
        n = int(s.shape[0])

        extras = ChainExtras(
            used_hf=(None if used_hf is None else _as_1d_bool(used_hf, name="used_hf", n=n)),
            accepted=(None if accepted is None else _as_1d_bool(accepted, name="accepted", n=n)),
            subchain_length=(
                None
                if subchain_length is None
                else _as_1d_int(
                    subchain_length, name="subchain_length", n=None
                )  # We do not know subchain length
            ),
        )
        return cls(samples=s, extras=extras)

    @property
    def n_steps(self) -> int:
        """Number of MCMC steps (rows of `samples`)."""
        return int(self.samples.shape[0])

    @property
    def n_dim(self) -> int:
        """Parameter dimension (columns of `samples`)."""
        return int(self.samples.shape[1])

    def burn_in(self, burn_in: int = 0) -> MCMCChain:
        """Drop the first `burn_in` samples and return a new chain."""
        b = _validate_burn_in(burn_in, n_steps=self.n_steps)
        sl = slice(b, None)
        return MCMCChain(samples=self.samples[sl], extras=self.extras.slice(sl))

    def thin(self, thin: int = 1) -> MCMCChain:
        """Thin the chain by keeping every `thin`-th sample."""
        t = _validate_thin(thin)
        sl = slice(None, None, t)
        return MCMCChain(samples=self.samples[sl], extras=self.extras.slice(sl))

    def summary(
        self,
        *,
        theta_true: ArrayLike | None = None,
        burn_in: int = 0,
    ) -> dict[str, Any]:
        """Compute lightweight diagnostic summary statistics.

        Parameters
        ----------
        theta_true
            Optional reference parameter vector. If provided, the RMSE between the posterior
            estimate and `theta_true` is reported.
        burn_in
            Burn-in used only for the posterior RMSE computation.

        Returns
        -------
        summary
            Dictionary of summary metrics.
        """
        out: dict[str, Any] = {"n_steps": self.n_steps, "n_dim": self.n_dim}

        if self.extras.accepted is not None:
            out["acceptance_rate"] = acceptance_rate_from_accepted(self.extras.accepted)
        else:
            out["move_fraction"] = move_fraction_from_samples(self.samples)

        if self.extras.used_hf is not None:
            out["hf_call_fraction"] = hf_call_fraction(self.extras.used_hf)
            out["n_hf_calls"] = int(np.sum(self.extras.used_hf))

        if self.extras.subchain_length is not None:
            out["mean_subchain_length"] = mean_subchain_length(self.extras.subchain_length)

        if theta_true is not None:
            out["posterior_rmse"] = posterior_rmse(self.samples, theta_true, burn_in=burn_in)

        return out


@dataclass(frozen=True, slots=True)
class SamplingResult:
    """Return type for sampling entrypoints.

    Attributes
    ----------
    chain
        The resulting MCMC chain.
    metadata
        Run metadata (configuration and bookkeeping). Intended to be lightweight and
        JSON-serialisable.
    """

    chain: MCMCChain
    metadata: dict[str, Any] = field(default_factory=dict)
