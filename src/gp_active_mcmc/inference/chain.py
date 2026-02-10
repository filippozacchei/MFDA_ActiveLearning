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

FloatArray = NDArray[np.floating]
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
    arr = np.asarray(samples, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"samples must be 2D (n_steps, n_dim). Got shape {arr.shape}.")
    return arr


def _as_1d_bool(x: ArrayLike, *, name: str, n: int) -> BoolArray:
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
    if arr.shape[0] != n:
        raise ValueError(f"{name} length {arr.shape[0]} does not match n_steps {n}.")
    return arr.astype(bool, copy=False)


def _as_1d_int(x: ArrayLike, *, name: str, n: int) -> IntArray:
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
    arr = np.asarray(x, dtype=int).ravel()
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D. Got shape {arr.shape}.")
    if arr.shape[0] != n:
        raise ValueError(f"{name} length {arr.shape[0]} does not match n_steps {n}.")
    return arr


def _validate_burnin(burnin: int, *, n_steps: int) -> int:
    """Validate burn-in length and return it as an `int`."""
    b = int(burnin)
    if b < 0:
        raise ValueError("burnin must be >= 0.")
    if b > n_steps:
        raise ValueError(f"burnin={b} is larger than chain length {n_steps}.")
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
        was used at each step (active workflows). In typical runs, these flags come from
        [`ActiveMCMCModel.log.used_hf`][gp_active_mcmc.inference.model.EvaluationLog].
    accepted
        Boolean array of length ``n_steps`` indicating whether each proposal was accepted,
        if the sampler provides explicit acceptance flags. If not available, acceptance-like
        information can be approximated using a move fraction computed from samples.
    subchain_length
        Integer array recording the subchain length (or subsampling rate) history. This is
        typically present only in adaptive runs using
        [`AdaptiveSubchain`][gp_active_mcmc.inference.adaptive_subchain.AdaptiveSubchain].
    """

    used_hf: BoolArray | None = None
    accepted: BoolArray | None = None
    subchain_length: IntArray | None = None

    def slice(self, sl: slice) -> "ChainExtras":
        """Return a sliced view of extras.

        This is used by [`MCMCChain.burnin`][gp_active_mcmc.inference.chain.MCMCChain.burnin] and
        [`MCMCChain.thin`][gp_active_mcmc.inference.chain.MCMCChain.thin] to keep extras aligned
        with sliced samples.

        Parameters
        ----------
        sl
            Slice to apply.

        Returns
        -------
        extras
            New `ChainExtras` with all available fields sliced consistently.
        """

        def _s(v):
            return None if v is None else v[sl]

        return ChainExtras(
            used_hf=_s(self.used_hf),
            accepted=_s(self.accepted),
            subchain_length=_s(self.subchain_length),
        )


@dataclass(frozen=True, slots=True)
class MCMCChain:
    """Immutable container for MCMC samples and aligned diagnostics.

    The chain is represented as:

    - `samples`: a 2D array of shape ``(n_steps, n_dim)``
    - `extras`: optional per-step diagnostics aligned with samples

    The class provides lightweight post-processing utilities (burn-in removal,
    thinning, and summary statistics) without mutating the original object.

    Notes
    -----
    This container is intentionally minimal: it performs no plotting and no I/O.
    Plotting utilities live in [`gp_active_mcmc.diagnostics`][gp_active_mcmc.diagnostics].

    See Also
    --------
    [`SamplingResult`][gp_active_mcmc.inference.chain.SamplingResult]
        Sampler return type that wraps an `MCMCChain` plus metadata.
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
    ) -> "MCMCChain":
        """Construct an `MCMCChain` from raw arrays.

        This constructor validates basic shapes and ensures that optional extras (if provided)
        have the correct length and dtype.

        Parameters
        ----------
        samples
            Sample matrix of shape ``(n_steps, n_dim)``.
        used_hf
            Optional boolean vector of length ``n_steps``.
        accepted
            Optional boolean vector of length ``n_steps``.
        subchain_length
            Optional integer vector of length ``n_steps``. If you pass a shorter history
            (e.g., only update events), store it elsewhere and do not claim per-step alignment.

        Returns
        -------
        chain
            A validated `MCMCChain`.

        Raises
        ------
        ValueError
            If `samples` is not 2D or if aligned arrays have inconsistent lengths.

        See Also
        --------
        [`ChainExtras`][gp_active_mcmc.inference.chain.ChainExtras]
            Container type used for optional aligned arrays.
        """
        s = _as_2d_float(samples)
        n = s.shape[0]
        extras = ChainExtras(
            used_hf=None if used_hf is None else _as_1d_bool(used_hf, name="used_hf", n=n),
            accepted=None if accepted is None else _as_1d_bool(accepted, name="accepted", n=n),
            subchain_length=None if subchain_length is None else subchain_length,
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

    def burnin(self, burnin: int = 0) -> "MCMCChain":
        """Drop the first `burnin` samples and return a new chain.

        Parameters
        ----------
        burnin
            Number of initial samples to discard.

        Returns
        -------
        chain
            New `MCMCChain` with burn-in removed.

        See Also
        --------
        [`MCMCChain.thin`][gp_active_mcmc.inference.chain.MCMCChain.thin]
            Thinning operation that also keeps extras aligned.
        """
        b = _validate_burnin(burnin, n_steps=self.n_steps)
        sl = slice(b, None)
        return MCMCChain(samples=self.samples[sl], extras=self.extras.slice(sl))

    def thin(self, thin: int = 1) -> "MCMCChain":
        """Thin the chain by keeping every `thin`-th sample.

        Parameters
        ----------
        thin
            Thinning factor. Must be a positive integer.

        Returns
        -------
        chain
            New `MCMCChain` with thinned samples and sliced extras.
        """
        t = _validate_thin(thin)
        sl = slice(None, None, t)
        return MCMCChain(samples=self.samples[sl], extras=self.extras.slice(sl))

    def summary(
        self,
        *,
        theta_true: ArrayLike | None = None,
        burnin: int = 0,
    ) -> dict[str, Any]:
        """Compute lightweight diagnostic summary statistics.

        Included metrics
        ----------------
        - `n_steps`, `n_dim`
        - acceptance information:
          - if `extras.accepted` is available: exact acceptance rate via
            [`acceptance_rate_from_accepted`][gp_active_mcmc.utils.mcmc.acceptance_rate_from_accepted]
          - otherwise: move fraction via
            [`move_fraction_from_samples`][gp_active_mcmc.utils.mcmc.move_fraction_from_samples]
        - HF usage:
          - `hf_call_fraction`, `n_hf_calls` if `extras.used_hf` is available, using
            [`hf_call_fraction`][gp_active_mcmc.utils.mcmc.hf_call_fraction]
        - adaptive subchain:
          - `mean_subchain_length` if `extras.subchain_length` is available, using
            [`mean_subchain_length`][gp_active_mcmc.utils.mcmc.mean_subchain_length]
        - optional accuracy:
          - `posterior_rmse` if `theta_true` is provided, using
            [`posterior_rmse`][gp_active_mcmc.utils.mcmc.posterior_rmse]

        Parameters
        ----------
        theta_true
            Optional reference parameter vector. If provided, the RMSE between the posterior
            estimate and `theta_true` is reported.
        burnin
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
            out["posterior_rmse"] = posterior_rmse(self.samples, theta_true, burnin=burnin)

        return out


@dataclass(frozen=True, slots=True)
class SamplingResult:
    """Return type for sampling entrypoints.

    The sampling entrypoints in
    [`gp_active_mcmc.inference.sampling`][gp_active_mcmc.inference.sampling]
    return a `SamplingResult` to keep the public API stable:

    - `chain`: the samples and aligned diagnostics as an
      [`MCMCChain`][gp_active_mcmc.inference.chain.MCMCChain]
    - `metadata`: lightweight bookkeeping information (iterations, chunk size, etc.)

    Attributes
    ----------
    chain
        The resulting MCMC chain.
    metadata
        Run metadata (configuration and bookkeeping). Intended to be lightweight and
        JSON-serialisable.

    See Also
    --------
    [`sample_active_chain`][gp_active_mcmc.inference.sampling.sample_active_chain]
        Fixed-subsampling sampler.
    [`sample_adaptive_active_chain`][gp_active_mcmc.inference.sampling.sample_adaptive_active_chain]
        Chunked sampler for adaptive subchain runs.
    """

    chain: MCMCChain
    metadata: dict[str, Any] = field(default_factory=dict)
