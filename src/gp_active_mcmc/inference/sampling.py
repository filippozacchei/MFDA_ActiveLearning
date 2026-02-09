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

    Parameters
    ----------
    chain_key
        Key used by tinyDA to store the chain in the returned object.
    chunk_size
        Budget in *coarse evaluation units* per chunk.
    """

    chain_key: str
    chunk_size: int = 500


def _extract_used_hf(model: Any) -> np.ndarray:
    """Backward/forward compatible extraction of HF usage flags."""
    if hasattr(model, "log") and hasattr(model.log, "used_hf"):
        return np.asarray(model.log.used_hf, dtype=bool)
    if hasattr(model, "used_hf_flags"):
        return np.asarray(model.used_hf_flags, dtype=bool)
    raise AttributeError("Model does not expose HF usage flags (expected model.log.used_hf or model.used_hf_flags).")


def _extract_subchain_history(model: Any) -> np.ndarray | None:
    """Optional extraction of subchain length history (adaptive runs)."""
    adaptive = getattr(model, "adaptive", None)
    if adaptive is None:
        return None
    # supports AdaptiveSubchain(state=..., control=...)
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
    """Sample an active chain with a fixed subsampling rate (single tinyDA call)."""
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
    """Chunked sampling to support adaptive subchain lengths.

    This enforces a fixed *total* number of coarse evaluations (`n_coarse_evals`)
    while allowing the model to adjust the subsampling rate between chunks.
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

        # Pull the current subchain_length from the adaptive state.
        adaptive = getattr(model, "adaptive", None)
        if adaptive is None or getattr(adaptive, "state", None) is None:
            raise ValueError("model.adaptive.state is required for adaptive sampling.")
        subchain_nominal = int(adaptive.state.subchain_length)
        if subchain_nominal <= 0:
            raise ValueError("adaptive.state.subchain_length must be positive.")

        subsampling_rate = min(subchain_nominal, coarse_budget)

        # coarse_budget = iterations * subsampling_rate, choose iterations accordingly
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
    used_hf = np.concatenate(used_hf_blocks) if used_hf_blocks else np.zeros((samples.shape[0],), dtype=bool)
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
