from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import tinyDA as tda

from .active_mcmc_chain import ActiveMCMCChain
from .active_mcmc_model import AdaptiveActiveMCMCModel, ActiveMCMCModel
from .utils.mcmc import extract_samples
from .proposal import AdaptiveMetropolisShared


@dataclass(frozen=True)
class ChunkedMCMCConfig:
    """Configuration for chunked sampling.

    Parameters
    ----------
    chain_key
        Key used by tinyDA to store the chain in the returned object, e.g.
        "chain_0" or "chain_coarse_0".
    chunk_size
        Chunk size in *coarse evaluation* units (not tinyDA iterations).
        This controls how often we re-enter tinyDA.
    """

    chain_key: str
    chunk_size: int = 500


def sample_active_chain(
    model: ActiveMCMCModel,
    posterior: tda.Posterior | list[tda.Posterior],
    proposal: tda.Proposal,
    n_samples: int,
    n_chains: int,
    initial_parameter: np.ndarray,
    subsampling_rate: int,
    chain_key: str,
    *,
    force_sequential: bool = True,
    store_coarse_chain: bool = True,
    summary: bool = True,
    theta_true: np.ndarray | None = None,
) -> ActiveMCMCChain:
    """Sample an active chain with a fixed subsampling rate (single tinyDA call)."""
    chain_obj = tda.sample(
        posteriors=posterior,
        proposal=copy.deepcopy(proposal),
        iterations=n_samples,
        n_chains=n_chains,
        force_sequential=force_sequential,
        initial_parameters=initial_parameter,
        store_coarse_chain=store_coarse_chain,
        subsampling_rate=subsampling_rate,
        adaptive_error_model=None,
    )

    samples = extract_samples(chain=chain_obj, chain_key=chain_key)
    forward_calls = np.asarray(model.used_hf_flags, dtype=int)

    active_chain = ActiveMCMCChain(samples=samples, forward_calls=forward_calls)
    if summary:
        active_chain.info(theta_true=theta_true)
    return active_chain


def sample_adaptive_active_chain(
    model: AdaptiveActiveMCMCModel,
    posterior: tda.Posterior | list[tda.Posterior],
    proposal: AdaptiveMetropolisShared,
    n_coarse_evals: int,
    initial_parameter: np.ndarray,
    chain_key: str,
    chunk_size: int,
    *,
    n_chains: int = 1,
    force_sequential: bool = True,
    store_coarse_chain: bool = True,
    summary: bool = True,
    theta_true: np.ndarray | None = None,
) -> ActiveMCMCChain:
    """Sample an active chain in chunks to support adaptive subchain lengths.

    Intended for models that adapt `model.subchain_length` (e.g., AdaptiveActiveMCMCModel),
    while ensuring the *total* number of coarse evaluations equals `n_coarse_evals`.

    Proposal adaptivity:
    - If your proposal shares adaptive state across deepcopies (e.g., cov_bias/bias),
      then repeated calls to tinyDA (which may deepcopy internally) will still preserve
      adaptation.
    """
    if n_coarse_evals <= 0:
        raise ValueError("`n_coarse_evals` must be positive.")
    if chunk_size <= 0:
        raise ValueError("`config.chunk_size` must be positive.")
    if n_chains != 1:
        raise ValueError("Chunked sampling currently supports `n_chains=1` only.")

    # Deepcopy once: protects the caller and provides a stable object across chunks.
    proposal_work = copy.deepcopy(proposal)

    theta_current = np.array(initial_parameter, copy=True)
    coarse_done = 0

    blocks: list[np.ndarray] = []
    hf_blocks: list[np.ndarray] = []
    hf_cursor = 0

    while coarse_done < n_coarse_evals:
        remaining = n_coarse_evals - coarse_done
        coarse_budget = min(chunk_size, remaining)

        subchain_nominal = model.adaptive_state.subchain_length
        if subchain_nominal <= 0:
            raise ValueError("`model.subchain_length` must be positive.")

        # Use the model's suggested subchain length, but never exceed the remaining budget.
        subsampling_rate = min(subchain_nominal, coarse_budget)
        print(f"Adapting: new subsampling rate {subsampling_rate}")
        # tinyDA iterations for this chunk
        n_iter = coarse_budget // subsampling_rate
        if n_iter == 0:
            # Final tail: one iteration with reduced subsampling so we hit the budget exactly.
            n_iter = 1
            subsampling_rate = coarse_budget

        chain_obj = tda.sample(
            posteriors=posterior,
            proposal=proposal_work,
            iterations=n_iter,
            n_chains=1,
            force_sequential=force_sequential,
            initial_parameters=theta_current,
            store_coarse_chain=store_coarse_chain,
            subsampling_rate=subsampling_rate,
            adaptive_error_model=None,
        )

        theta_block = extract_samples(chain=chain_obj, chain_key=chain_key)

        hf_all = np.asarray(model.used_hf_flags, dtype=int)
        hf_block = hf_all[hf_cursor : hf_cursor + theta_block.shape[0]]
        hf_cursor += theta_block.shape[0]

        blocks.append(theta_block)
        hf_blocks.append(hf_block)

        theta_current = theta_block[-1]
        coarse_done += n_iter * subsampling_rate

    samples = np.vstack(blocks)
    forward_calls = (
        np.concatenate(hf_blocks) if hf_blocks else np.zeros((0,), dtype=int)
    )

    active_chain = ActiveMCMCChain(samples=samples, forward_calls=forward_calls)
    if summary:
        active_chain.info(theta_true=theta_true)
    return active_chain
