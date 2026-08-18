from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import tinyDA as tda
from numpy.typing import ArrayLike, NDArray

from gp_active_mcmc.inference.chain import MCMCChain, SamplingResult
from gp_active_mcmc.utils.mcmc import extract_samples

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


@dataclass(frozen=True, slots=True)
class ChunkedMCMCConfig:
    """Configuration for chunked sampling.

    Chunked sampling is used when an algorithm needs to periodically re-enter `tinyDA`
    (i.e., call `tda.sample` multiple times) rather than running one long
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


def _extract_used_hf(model: Any) -> FloatArray:
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
        "Model does not expose HF usage flags (expected model.log.used_hf or model.used_hf_flags)."
    )


def _extract_subchain_history(model: Any) -> FloatArray | None:
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


def _extract_chain_with_used_hf(*, chain_obj: Any, model: Any, chain_key: str) -> MCMCChain:
    """Extract one named chain from a `tinyDA` result, with a correctly-reconstructed
    `used_hf` flag per row. Three cases (see `sample_active_chain`'s docstring for the
    first two; the third is specific to `chain_coarse_i` in a two-posterior DA run):

    1. Natural per-coarse-step alignment (`used_hf` and `samples` the same length):
       used as-is.
    2. `used_hf` exactly one longer than `samples`: tinyDA's `DAChain.__init__`
       bootstraps by calling `coarse()` then immediately `fine()`-overwriting it once,
       *before* the sampling loop starts -- one extra `model.log` entry with no
       corresponding row in the extracted chain (confirmed empirically: this leading
       entry is always `True`, and the remaining entries fall exactly one `True` per
       `subsampling_rate`-sized block, matching the true `coarse()`/`fine()` call
       pattern). Dropping that one leading entry restores 1:1 alignment. Without this,
       `chain_coarse_i` (tinyDA's DA "coarse" chain, requested as a diagnostic
       trajectory via `diagnostic_chain_key` for trace plots) silently fell through to
       case 3 below and was reported as 100% HF -- every point in `ours`'s production
       trace plot showing as an HF call, when the vast majority are cheap LF steps.
    3. Otherwise (e.g. tinyDA's DA "fine" chain, `chain_fine_i`): every row trivially
       used HF, by construction of delayed acceptance (each outer iteration ends with
       exactly one HF correction).

    Factored out so the same logic can be applied to a second, purely diagnostic,
    chain extracted from the same `tinyDA` call (see `diagnostic_chain_key`).
    """
    samples = extract_samples(chain=chain_obj, chain_key=chain_key)
    used_hf = _extract_used_hf(model)

    if used_hf.shape[0] == samples.shape[0]:
        pass
    elif used_hf.shape[0] == samples.shape[0] + 1:
        used_hf = used_hf[1:]
    else:
        used_hf = np.ones(samples.shape[0], dtype=bool)

    return MCMCChain.from_arrays(samples=samples, used_hf=used_hf)


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
    diagnostic_chain_key: str | None = None,
) -> SamplingResult:
    """Run Active-(DA)-MCMC with a fixed subsampling rate (single `tinyDA` call).

    This is the main entrypoint when the subsampling rate is fixed throughout sampling.

    Interpretation of `posterior`
    -----------------------------
    The `posterior` argument determines the algorithmic mode:

    - If `posterior` is a single `tinyDA.Posterior`, the run corresponds to
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
        Proposal passed to `tinyDA.sample`. The proposal is deep-copied to
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
    diagnostic_chain_key
        Optional second chain key to *also* extract from the same `tinyDA` call, purely
        for diagnostics/plotting -- it has no effect on `chain` (the one used for
        inference) or on any budget accounting. The intended use is requesting tinyDA's
        DA "coarse" chain (`"chain_coarse_0"`) alongside a `chain_key` of `"chain_fine_0"`:
        the fine chain is the theoretically-valid posterior (one state per outer DA
        block), but it discards the intra-block low-fidelity trajectory, which is
        useful to see in a trace plot (e.g. colored by `used_hf`) even though those
        intermediate states don't individually carry the DA guarantee. If given, the
        result is available at `metadata["diagnostic_chain"]` as an `MCMCChain`.

    Returns
    -------
    result
        [`SamplingResult`][gp_active_mcmc.inference.chain.SamplingResult] containing:

        - `chain`: an immutable [`MCMCChain`][gp_active_mcmc.inference.chain.MCMCChain]
        - `metadata`: a dict recording run configuration (plus `"diagnostic_chain"` if
          `diagnostic_chain_key` was given)

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

    # `model.log.used_hf` (appended once per `coarse()`/`fine()` call) naturally aligns
    # 1:1 with `samples` for a single-posterior chain. For a two-posterior (DA) run,
    # it's off by exactly one against the DA "coarse" chain (one within-block
    # low-fidelity substep per genuine log entry, plus one extra bootstrap entry from
    # `tinyDA`'s `DAChain.__init__`), and unrelated in scale against the DA "fine"
    # chain (`chain_fine_i`: one state per outer DA iteration) -- there, every
    # fine-chain sample trivially "used HF" by construction of delayed acceptance
    # (each outer iteration ends with exactly one HF correction), regardless of the
    # model's internal per-coarse-step log (which has a different length and cannot be
    # sliced/aligned to match; naively slicing it, as an earlier version of this
    # function did, silently grabs the wrong tail of an unrelated array). See
    # `_extract_chain_with_used_hf` for the shape-based dispatch between these cases.
    #
    # Caveat: the fine-chain case undercounts if `coarse()` can *also* opportunistically
    # trigger extra HF evaluations within a block (the online-active-learning
    # uncertainty trigger) while `posterior` targets the fine chain -- that path is
    # disabled for a frozen model (`ActiveMCMCModel.frozen=True` short-circuits
    # `coarse()` before that check), which is the only case this is currently used for
    # (`sample_adaptive_then_frozen_chain`'s production phase). It would need
    # revisiting for a non-frozen two-posterior run using the fine chain.
    chain = _extract_chain_with_used_hf(chain_obj=chain_obj, model=model, chain_key=chain_key)

    metadata: dict[str, Any] = {
        "chain_key": chain_key,
        "iterations": int(iterations),
        "subsampling_rate": int(subsampling_rate),
        "n_chains": int(n_chains),
        "store_coarse_chain": bool(store_coarse_chain),
    }
    if diagnostic_chain_key is not None:
        metadata["diagnostic_chain"] = _extract_chain_with_used_hf(
            chain_obj=chain_obj, model=model, chain_key=diagnostic_chain_key
        )

    return SamplingResult(chain=chain, metadata=metadata)


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
    stop_check: Callable[[], bool] | None = None,
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
    is adapted online). Since `tinyDA.sample` takes a fixed `subsampling_rate`
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
    stop_check
        Optional zero-argument callable checked after every chunk. If it returns True, the
        loop stops early even though `n_coarse_evals` has not been reached, and
        `metadata["stopped_early"]` is set to True. Intended to carry a bound method such as
        `model.adaptive.has_converged`, so that the adaptive phase can be terminated as soon
        as the surrogate has converged rather than running for a fixed budget. See
        [`AdaptiveSubchain.has_converged`][gp_active_mcmc.inference.adaptive_subchain.AdaptiveSubchain.has_converged].

    Returns
    -------
    result
        [`SamplingResult`][gp_active_mcmc.inference.chain.SamplingResult] with:

        - concatenated samples across chunks,
        - aligned HF usage flags,
        - optional subchain-length history (if available),
        - `metadata["coarse_evals_used"]`: the actual number of coarse evaluations consumed
          (equal to `n_coarse_evals` unless `stop_check` triggered early termination).

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
    blocks: list[FloatArray] = []
    used_hf_blocks: list[FloatArray] = []

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

        if stop_check is not None and stop_check():
            break

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
            "coarse_evals_used": int(coarse_done),
            "stopped_early": bool(coarse_done < n_coarse_evals),
            "chunk_size": int(config.chunk_size),
            "n_chains": 1,
            "store_coarse_chain": bool(store_coarse_chain),
        },
    )


def sample_adaptive_then_frozen_chain(
    *,
    model: Any,
    posterior_factory: Callable[[Any], tda.Posterior | list[tda.Posterior]],
    proposal: tda.Proposal,
    n_coarse_evals: int,
    initial_parameters: ArrayLike,
    chain_key: str,
    config: ChunkedMCMCConfig,
    max_adapt_coarse_evals: int | None = None,
    production_chain_key: str | None = None,
    production_diagnostic_chain_key: str | None = None,
    n_chains: int = 1,
    force_sequential: bool = True,
    store_coarse_chain: bool = True,
) -> SamplingResult:
    """Run the full adapt -> freeze -> production workflow (paper Algorithm 3).

    This is the entrypoint that gives the "fixed-kernel online stage" guarantee its
    meaning: it runs [`sample_adaptive_active_chain`][gp_active_mcmc.inference.sampling.sample_adaptive_active_chain]
    until the surrogate has converged (or a budget is exhausted), then calls
    [`ActiveMCMCModel.freeze`][gp_active_mcmc.inference.model.ActiveMCMCModel.freeze] and
    spends the remaining budget with [`sample_active_chain`][gp_active_mcmc.inference.sampling.sample_active_chain]
    using a fixed subsampling rate and a surrogate that no longer learns. Only samples
    from the frozen production phase carry the DA-MCMC stationarity guarantee; the
    adaptive-phase samples are informative for diagnostics but should generally be
    discarded (like burn-in) for posterior inference.

    Convergence
    -----------
    The adaptive phase stops as soon as `model.adaptive.has_converged()` returns True
    (see [`AdaptiveSubchain.has_converged`][gp_active_mcmc.inference.adaptive_subchain.AdaptiveSubchain.has_converged],
    controlled by `AdaptiveSubchainControl.patience` and `target_error`), or once
    `max_adapt_coarse_evals` coarse evaluations have been spent, whichever comes first.
    If `max_adapt_coarse_evals` is None, the entire `n_coarse_evals` budget is available
    to the adaptive phase, and the production phase may end up empty (see
    `metadata["phase"] == "adapt_only"`).

    Parameters
    ----------
    model
        Active model with an adaptive policy (`model.adaptive` must be an
        [`AdaptiveSubchain`][gp_active_mcmc.inference.adaptive_subchain.AdaptiveSubchain]
        or otherwise expose `has_converged()`). Passed by reference to the adaptive phase
        and mutated in place (its surrogate is trained online); the production phase runs
        on a separate frozen copy obtained via `model.freeze()`.
    posterior_factory
        Callable that builds the `[coarse, fine]` posterior list (or a single posterior)
        for a given active model. Called once with `model` for the adaptive phase and once
        with the frozen copy for the production phase, so that both phases use posteriors
        bound to the correct `coarse`/`fine` callables. Typical usage:

        ```python
        def posterior_factory(m):
            return [
                tda.Posterior(prior, loglike_coarse, m.coarse),
                tda.Posterior(prior, loglike_fine, m.fine),
            ]
        ```
    proposal
        Proposal passed to both phases. Use an
        [`AdaptiveMetropolisShared`][gp_active_mcmc.inference.proposal.AdaptiveMetropolisShared]
        with `share_across_deepcopy=True` if proposal adaptation should carry over from the
        adaptive phase into production.
    n_coarse_evals
        Total budget in coarse evaluation units, shared between the adaptive and
        production phases.
    initial_parameters
        Initial parameter vector for the adaptive phase. The production phase is
        initialized from the last adaptive-phase sample.
    chain_key
        Chain key used to locate the *adaptive*-phase samples inside the object
        returned by `tinyDA` (e.g. tinyDA's DA "coarse" chain, `"chain_coarse_0"`).
        Only used for diagnostics and as the production phase's starting point; see
        `production_chain_key` for the samples that actually matter for inference.
    config
        Chunking configuration for the adaptive phase.
    max_adapt_coarse_evals
        Optional hard cap on the adaptive phase's budget (the paper's `N_adapt`,
        expressed in coarse evaluation units here). If None, defaults to `n_coarse_evals`.
    production_chain_key
        Chain key used for the *production* phase's samples -- the ones that carry the
        DA stationarity guarantee. For a two-posterior `[coarse, fine]` run this should
        be tinyDA's DA "fine" chain (e.g. `"chain_fine_0"`), **not** the same coarse
        chain used for the adaptive phase: tinyDA's coarse chain records every
        within-block low-fidelity substep, most of which are never individually
        checked against the true (fine/HF) likelihood, whereas the fine chain records
        exactly one state per outer DA iteration, each backed by a genuine HF-corrected
        accept/reject decision -- only the fine chain has `pi_HF` as its guaranteed
        invariant distribution. If None, defaults to `chain_key` (backwards-compatible,
        but only correct if the caller's `chain_key` already points at the fine chain).
    production_diagnostic_chain_key
        Optional second chain key to *also* extract during the production phase, purely
        for diagnostics/plotting (e.g. `"chain_coarse_0"` alongside a
        `production_chain_key` of `"chain_fine_0"`, to recover the intra-block
        low-fidelity trajectory the fine chain discards). Has no effect on `chain` or on
        any budget accounting; see `sample_active_chain`'s `diagnostic_chain_key`. If
        given, available at `metadata["production_diagnostic_chain"]`.
    n_chains, force_sequential, store_coarse_chain
        Passed through to both phases.

    Returns
    -------
    result
        [`SamplingResult`][gp_active_mcmc.inference.chain.SamplingResult] whose chain
        concatenates the adaptive-phase and production-phase samples (production last),
        and whose `metadata` records:

        - `converged`: whether `has_converged()` triggered the switch to production,
        - `n_adapt_samples` / `n_production_samples`: sizes of each phase, so callers can
          slice `chain.samples[n_adapt_samples:]` to get only the theoretically-justified
          production samples,
        - `frozen_subsampling_rate`: the subchain length frozen at the moment of switching,
        - `adapt_metadata` / `production_metadata`: the metadata dicts of each sub-call.

    Raises
    ------
    ValueError
        If `n_coarse_evals`, `max_adapt_coarse_evals` are non-positive, or if
        `model.adaptive` does not expose `has_converged()`.

    See Also
    --------
    [`ActiveMCMCModel.freeze`][gp_active_mcmc.inference.model.ActiveMCMCModel.freeze]
        Produces the frozen model used for the production phase.
    [`AdaptiveSubchain.has_converged`][gp_active_mcmc.inference.adaptive_subchain.AdaptiveSubchain.has_converged]
        Convergence criterion used to end the adaptive phase.
    """
    if n_coarse_evals <= 0:
        raise ValueError("n_coarse_evals must be positive.")

    adaptive = getattr(model, "adaptive", None)
    if adaptive is None or not hasattr(adaptive, "has_converged"):
        raise ValueError(
            "model.adaptive must expose has_converged() to use "
            "sample_adaptive_then_frozen_chain (e.g. an AdaptiveSubchain instance)."
        )

    adapt_budget = int(
        n_coarse_evals if max_adapt_coarse_evals is None else min(max_adapt_coarse_evals, n_coarse_evals)
    )
    if adapt_budget <= 0:
        raise ValueError("max_adapt_coarse_evals must be positive.")

    adapt_result = sample_adaptive_active_chain(
        model=model,
        posterior=posterior_factory(model),
        proposal=proposal,
        n_coarse_evals=adapt_budget,
        initial_parameters=initial_parameters,
        chain_key=chain_key,
        config=config,
        n_chains=n_chains,
        force_sequential=force_sequential,
        store_coarse_chain=store_coarse_chain,
        stop_check=adaptive.has_converged,
    )

    converged = bool(adaptive.has_converged())
    coarse_used = int(adapt_result.metadata["coarse_evals_used"])
    remaining = int(n_coarse_evals) - coarse_used

    frozen_model = model.freeze()

    if remaining <= 0:
        metadata = {
            "phase": "adapt_only",
            "converged": converged,
            "n_coarse_evals": int(n_coarse_evals),
            "n_adapt_samples": int(adapt_result.chain.n_steps),
            "n_production_samples": 0,
            "adapt_metadata": adapt_result.metadata,
        }
        return SamplingResult(chain=adapt_result.chain, metadata=metadata)

    subsampling_rate = int(adaptive.state.subchain_length)
    iterations = remaining // subsampling_rate
    if iterations == 0:
        iterations = 1
        subsampling_rate = remaining

    theta_last = adapt_result.chain.samples[-1]
    resolved_production_chain_key = production_chain_key if production_chain_key is not None else chain_key

    production_result = sample_active_chain(
        model=frozen_model,
        posterior=posterior_factory(frozen_model),
        proposal=proposal,
        iterations=int(iterations),
        initial_parameters=theta_last,
        subsampling_rate=int(subsampling_rate),
        chain_key=resolved_production_chain_key,
        n_chains=n_chains,
        force_sequential=force_sequential,
        store_coarse_chain=store_coarse_chain,
        diagnostic_chain_key=production_diagnostic_chain_key,
    )

    adapt_chain = adapt_result.chain
    prod_chain = production_result.chain

    samples = np.vstack([adapt_chain.samples, prod_chain.samples])

    adapt_used_hf = (
        adapt_chain.extras.used_hf
        if adapt_chain.extras.used_hf is not None
        else np.zeros(adapt_chain.n_steps, dtype=bool)
    )
    prod_used_hf = (
        prod_chain.extras.used_hf
        if prod_chain.extras.used_hf is not None
        else np.zeros(prod_chain.n_steps, dtype=bool)
    )
    used_hf = np.concatenate([adapt_used_hf, prod_used_hf])

    if adapt_chain.extras.subchain_length is not None:
        subchain_length: IntArray | None = np.concatenate(
            [
                adapt_chain.extras.subchain_length,
                np.full(prod_chain.n_steps, subsampling_rate, dtype=int),
            ]
        )
    else:
        subchain_length = None

    chain = MCMCChain.from_arrays(samples=samples, used_hf=used_hf, subchain_length=subchain_length)

    metadata = {
        "phase": "adapt_then_production",
        "chain_key": chain_key,
        "production_chain_key": resolved_production_chain_key,
        "n_coarse_evals": int(n_coarse_evals),
        "adapt_coarse_evals_used": coarse_used,
        "converged": converged,
        "n_adapt_samples": int(adapt_chain.n_steps),
        "n_production_samples": int(prod_chain.n_steps),
        "frozen_subsampling_rate": int(subsampling_rate),
        "adapt_metadata": adapt_result.metadata,
        "production_metadata": production_result.metadata,
    }
    if production_diagnostic_chain_key is not None:
        metadata["production_diagnostic_chain"] = production_result.metadata.get("diagnostic_chain")
    return SamplingResult(chain=chain, metadata=metadata)
