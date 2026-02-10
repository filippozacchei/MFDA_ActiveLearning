from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import tinyDA as tda
from numpy.typing import ArrayLike, NDArray

from gp_active_mcmc.inference.chain import MCMCChain, SamplingResult
from gp_active_mcmc.utils.mcmc import extract_samples

FloatArray = NDArray[np.floating]


@dataclass(frozen=True, slots=True)
class ChunkedMCMCConfig:
    """Configuration for chunked sampling.

    Chunked sampling is used when an algorithm needs to periodically re-enter `tinyDA`
    (i.e., call [`tda.sample`][tinyDA.sample] multiple times) rather than running one long
    sampling call.

    In this library, chunking is primarily used to support
    [`AdaptiveSubchain`][gp_active_mcmc.inference.adaptive_subchain.AdaptiveSubchain]:
    the subsampling rate (coarse steps per fine correction) can change between chunks.

    Parameters
    ----------
    chain_key
        Key used by `tinyDA` to identify the chain inside the returned object, e.g.
        ``"chain_0"`` or ``"chain_coarse_0"``.
    chunk_size
        Budget per chunk measured in *coarse evaluation units*.

        A "coarse evaluation unit" corresponds to one LF-first evaluation in the active model
        (one step in the coarse chain). In adaptive workflows we treat this as the primary
        computational budget and derive `tinyDA` iterations accordingly.

    See Also
    --------
    [`sample_adaptive_active_chain`][gp_active_mcmc.inference.sampling.sample_adaptive_active_chain]
        Chunked sampler that uses this configuration.
    """

    chain_key: str
    chunk_size: int = 500


def _extract_used_hf(model: Any) -> np.ndarray:
    """Extract HF usage flags from an active model.

    This helper exists for backward/forward compatibility across versions where the
    active model stored HF-usage metadata under different attribute names.

    Supported conventions
    ---------------------
    - `model.log.used_hf` (preferred; current API)
    - `model.used_hf_flags` (legacy)

    Parameters
    ----------
    model
        Active model instance used during sampling (typically an
        [`ActiveMCMCModel`][gp_active_mcmc.inference.model.ActiveMCMCModel]).

    Returns
    -------
    used_hf
        Boolean array aligned with coarse-chain samples.

    Raises
    ------
    AttributeError
        If no supported HF-usage attribute is found.
    """
    if hasattr(model, "log") and hasattr(model.log, "used_hf"):
        return np.asarray(model.log.used_hf, dtype=bool)
    if hasattr(model, "used_hf_flags"):
        return np.asarray(model.used_hf_flags, dtype=bool)
    raise AttributeError(
        "Model does not expose HF usage flags "
        "(expected model.log.used_hf or model.used_hf_flags)."
    )


def _extract_subchain_history(model: Any) -> np.ndarray | None:
    """Extract adaptive subchain length history if available.

    In adaptive runs, the model may store a history of chosen subchain lengths.
    This helper returns that history if present; otherwise it returns `None`.

    Parameters
    ----------
    model
        Active model instance used during sampling.

    Returns
    -------
    subchain_history
        Array of subchain lengths (typically one entry per coarse evaluation), or
        `None` if no history is available.

    See Also
    --------
    [`AdaptiveSubchainState.subchain_history`][gp_active_mcmc.inference.adaptive_subchain.AdaptiveSubchainState]
        Field where the adaptive policy records subchain length history.
    """
    adaptive = getattr(model, "adaptive", None)
    if adaptive is None:
        return None

    state = getattr(adaptive, "state", None)
    if state is None:
        return None

    hist = getattr(state, "subchain_history", None)
    if hist is None:
        return None

    return np.asarray(hist, dtype=int)


def sample_active_chain(
    *,
    model: Any,
    posterior: tda.Posterior | list[tda.Posterior],
    proposal: tda.Proposal,
    iterations: int,
    initial_parameters: ArrayLike,
    subsampling_rate: int,
    chain_key: str,
    n_chains: int = 1,
    force_sequential: bool = True,
    store_coarse_chain: bool = True,
) -> SamplingResult:
    """Run Active-(DA)-MCMC with a fixed subsampling rate (single `tinyDA` call).

    This is the main entrypoint when the subsampling rate is fixed throughout sampling.

    Interpretation of `posterior`
    -----------------------------
    The `posterior` argument determines the algorithmic mode:

    - If `posterior` is a single [`tinyDA.Posterior`][tinyDA.Posterior], the run corresponds to
      **MCMC-guided active learning** (single level).
    - If `posterior` is a list of two posteriors `[coarse, fine]`, the run corresponds to
      **delayed-acceptance MCMC (DA-MCMC) guided active learning**.

    In both cases, `subsampling_rate` controls the frequency of the fine correction:
    roughly, the fine posterior is evaluated every `subsampling_rate` coarse steps
    (depending on `tinyDA`'s internal delayed-acceptance implementation).

    Parameters
    ----------
    model
        Active model used during sampling. After sampling it is queried to extract HF usage
        flags for diagnostics. In typical workflows this is an
        [`ActiveMCMCModel`][gp_active_mcmc.inference.model.ActiveMCMCModel].
    posterior
        Either a single posterior (single-level) or a list of two posteriors `[coarse, fine]`
        for DA-MCMC.
    proposal
        Proposal passed to [`tinyDA.sample`][tinyDA.sample]. The proposal is deep-copied to
        avoid mutating the caller's instance.
    iterations
        Number of `tinyDA` iterations to run.
    initial_parameters
        Initial parameter vector.
    subsampling_rate
        Fine-correction frequency. Must be positive.
    chain_key
        Chain key used by [`extract_samples`][gp_active_mcmc.utils.mcmc.extract_samples] to locate the
        chain inside the object returned by `tinyDA`.
    n_chains
        Number of chains to run (passed to `tinyDA`).
    force_sequential
        If True, force sequential execution (useful for reproducibility).
    store_coarse_chain
        If True, store the coarse chain (when supported by `tinyDA`).

    Returns
    -------
    result
        [`SamplingResult`][gp_active_mcmc.inference.chain.SamplingResult] containing:

        - `chain`: an immutable [`MCMCChain`][gp_active_mcmc.inference.chain.MCMCChain]
        - `metadata`: a dict recording run configuration

    Raises
    ------
    ValueError
        If `iterations` or `subsampling_rate` are not positive.

    See Also
    --------
    [`sample_adaptive_active_chain`][gp_active_mcmc.inference.sampling.sample_adaptive_active_chain]
        Chunked sampler used when the subsampling rate changes over time.
    """
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    if subsampling_rate <= 0:
        raise ValueError("subsampling_rate must be positive.")

    chain_obj = tda.sample(
        posteriors=posterior,
        proposal=copy.deepcopy(proposal),
        iterations=int(iterations),
        n_chains=int(n_chains),
        force_sequential=bool(force_sequential),
        initial_parameters=np.asarray(initial_parameters, dtype=float),
        store_coarse_chain=bool(store_coarse_chain),
        subsampling_rate=int(subsampling_rate),
        adaptive_error_model=None,
    )

    samples = extract_samples(chain=chain_obj, chain_key=chain_key)
    used_hf = _extract_used_hf(model)

    chain = MCMCChain.from_arrays(samples=samples, used_hf=used_hf)
    return SamplingResult(
        chain=chain,
        metadata={
            "chain_key": chain_key,
            "iterations": int(iterations),
            "subsampling_rate": int(subsampling_rate),
            "n_chains": int(n_chains),
            "store_coarse_chain": bool(store_coarse_chain),
        },
    )


def sample_adaptive_active_chain(
    *,
    model: Any,
    posterior: tda.Posterior | list[tda.Posterior],
    proposal: tda.Proposal,
    n_coarse_evals: int,
    initial_parameters: ArrayLike,
    chain_key: str,
    config: ChunkedMCMCConfig,
    n_chains: int = 1,
    force_sequential: bool = True,
    store_coarse_chain: bool = True,
) -> SamplingResult:
    """Run adaptive DA-MCMC guided active learning using chunked sampling.

    This entrypoint supports the recommended workflow:

    - DA-MCMC (two posteriors: coarse + fine), and
    - an adaptive policy such as
      [`AdaptiveSubchain`][gp_active_mcmc.inference.adaptive_subchain.AdaptiveSubchain]
      attached to the active model.

    Why chunking is needed
    ----------------------
    In adaptive runs, the subsampling rate may change over time (because the subchain length
    is adapted online). Since [`tinyDA.sample`][tinyDA.sample] takes a fixed `subsampling_rate`
    per call, we run multiple shorter calls ("chunks") and update the subsampling rate between
    chunks.

    Budgeting
    ---------
    The overall budget is expressed as a total number of *coarse evaluation units*
    (`n_coarse_evals`). Each chunk consumes up to `config.chunk_size` coarse evaluations.

    Requirements
    ------------
    - The adaptive workflow requires `model.adaptive.state.subchain_length`.
    - DA-MCMC is mandatory in this mode: `posterior` should be `[coarse, fine]`.

    Parameters
    ----------
    model
        Active model with an adaptive policy. Must expose `model.adaptive.state.subchain_length`
        (see [`AdaptiveSubchainState.subchain_length`][gp_active_mcmc.inference.adaptive_subchain.AdaptiveSubchainState]).
    posterior
        Two-level posterior list `[coarse, fine]` (DA-MCMC).
    proposal
        Proposal passed to `tinyDA`. A working copy is deep-copied once and then reused across
        chunks to preserve proposal adaptation state.
    n_coarse_evals
        Total budget in coarse evaluation units.
    initial_parameters
        Initial parameter vector.
    chain_key
        Chain key used by [`extract_samples`][gp_active_mcmc.utils.mcmc.extract_samples].
    config
        Chunking configuration (`chunk_size` and `chain_key`).
    n_chains
        Number of chains. Currently only `n_chains=1` is supported for chunked adaptive runs.
    force_sequential
        If True, force sequential execution.
    store_coarse_chain
        If True, store the coarse chain.

    Returns
    -------
    result
        [`SamplingResult`][gp_active_mcmc.inference.chain.SamplingResult] with:

        - concatenated samples across chunks,
        - aligned HF usage flags,
        - optional subchain-length history (if available).

    Raises
    ------
    ValueError
        If budgets are non-positive, if `n_chains != 1`, or if adaptive state is missing.

    See Also
    --------
    [`ChunkedMCMCConfig`][gp_active_mcmc.inference.sampling.ChunkedMCMCConfig]
        Chunk configuration controlling `chunk_size`.
    [`ActiveMCMCModel`][gp_active_mcmc.inference.model.ActiveMCMCModel]
        Active model that provides `coarse` and `fine` callables for the two posteriors.
    """
    if n_coarse_evals <= 0:
        raise ValueError("n_coarse_evals must be positive.")
    if config.chunk_size <= 0:
        raise ValueError("config.chunk_size must be positive.")
    if n_chains != 1:
        raise ValueError("chunked adaptive sampling currently supports n_chains=1 only.")

    proposal_work = copy.deepcopy(proposal)
    theta_current = np.asarray(initial_parameters, dtype=float).copy()

    coarse_done = 0
    blocks: list[np.ndarray] = []
    used_hf_blocks: list[np.ndarray] = []

    used_hf_cursor = 0

    while coarse_done < n_coarse_evals:
        remaining = n_coarse_evals - coarse_done
        coarse_budget = min(config.chunk_size, remaining)

        adaptive = getattr(model, "adaptive", None)
        if adaptive is None or getattr(adaptive, "state", None) is None:
            raise ValueError("model.adaptive.state is required for adaptive sampling.")
        subchain_nominal = int(adaptive.state.subchain_length)
        if subchain_nominal <= 0:
            raise ValueError("adaptive.state.subchain_length must be positive.")

        subsampling_rate = min(subchain_nominal, coarse_budget)

        iterations = coarse_budget // subsampling_rate
        if iterations == 0:
            iterations = 1
            subsampling_rate = coarse_budget

        chain_obj = tda.sample(
            posteriors=posterior,
            proposal=proposal_work,
            iterations=int(iterations),
            n_chains=1,
            force_sequential=bool(force_sequential),
            initial_parameters=theta_current,
            store_coarse_chain=bool(store_coarse_chain),
            subsampling_rate=int(subsampling_rate),
            adaptive_error_model=None,
        )

        theta_block = extract_samples(chain=chain_obj, chain_key=chain_key)
        blocks.append(theta_block)

        used_hf_all = _extract_used_hf(model)
        used_hf_block = used_hf_all[used_hf_cursor : used_hf_cursor + theta_block.shape[0]]
        used_hf_cursor += theta_block.shape[0]
        used_hf_blocks.append(used_hf_block)

        theta_current = theta_block[-1]
        coarse_done += int(iterations) * int(subsampling_rate)

    samples = np.vstack(blocks) if blocks else np.zeros((0, 0), dtype=float)
    used_hf = (
        np.concatenate(used_hf_blocks)
        if used_hf_blocks
        else np.zeros((samples.shape[0],), dtype=bool)
    )
    subchain_hist = _extract_subchain_history(model)

    chain = MCMCChain.from_arrays(samples=samples, used_hf=used_hf, subchain_length=subchain_hist)
    return SamplingResult(
        chain=chain,
        metadata={
            "chain_key": chain_key,
            "n_coarse_evals": int(n_coarse_evals),
            "chunk_size": int(config.chunk_size),
            "n_chains": 1,
            "store_coarse_chain": bool(store_coarse_chain),
        },
    )
