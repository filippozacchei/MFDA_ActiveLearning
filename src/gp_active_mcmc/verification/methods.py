"""The four comparison methods: hf_only, pretrained, online_active, adaptive_da.

Also defines `_ChunkState`, the resumable per-replicate state `harness.py`'s
round-based orchestration advances one chunk at a time -- defined here (not in
`harness.py`) because every `_init_*_state` factory below constructs one, and
`harness.py` in turn imports it from here, not the other way around.
"""

from __future__ import annotations

import contextlib
import copy
import functools
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import tinyDA as tda
from numpy.typing import NDArray

from gp_active_mcmc.inference import (
    ActiveGPLogLike,
    ActiveMCMCModel,
    AdaptiveSubchain,
    AdaptiveSubchainControl,
    AdaptiveSubchainState,
    ChunkedMCMCConfig,
    MCMCChain,
    sample_active_chain,
    sample_adaptive_active_chain,
    sample_adaptive_then_frozen_chain,
)
from gp_active_mcmc.surrogates import PODGPSurrogate
from gp_active_mcmc.surrogates.gp import KernelName
from gp_active_mcmc.utils.mcmc import extract_samples
from gp_active_mcmc.utils.rng import set_seed
from gp_active_mcmc.verification.design import (
    DEFAULT_ONLINE_LEARNING,
    OnlineLearningConfig,
    _surrogate_for_online_learning,
    active_learning_offline_design,
)
from gp_active_mcmc.verification.problem import Problem
from gp_active_mcmc.verification.sampling import make_proposal

FloatArray = NDArray[np.float64]

__all__ = [
    "run_adaptive_da",
    "run_hf_only",
    "run_online_active",
    "run_pretrained",
    "run_training_cost_comparison",
]


@contextlib.contextmanager
def _suppress_tinyda_output() -> Any:
    """Silence tinyDA's per-call console output (a `print` plus a `tqdm` bar) for the
    duration of one `with` block. `tinyDA.sample` has no verbosity flag to turn these
    off in the sequential path this package always uses.
    """
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        yield


def _posterior(problem: Problem, forward: Callable[[FloatArray], FloatArray], *, surrogate: bool) -> tda.Posterior:
    """The single ``tda.Posterior`` shared by every non-DA method: a Gaussian
    likelihood with covariance ``sigma_obs**2 * I`` wrapping `forward`.

    `surrogate=True` (a GP-backed `forward`, e.g. `model.coarse`) uses
    `ActiveGPLogLike`, which accounts for the surrogate's own predictive variance;
    `surrogate=False` (the raw HF model) uses plain `tda.AdaptiveGaussianLogLike`.
    """
    cov = (problem.sigma_obs**2) * np.eye(len(problem.y_obs))
    loglike = (
        ActiveGPLogLike(data=problem.y_obs, covariance=cov)
        if surrogate
        else tda.AdaptiveGaussianLogLike(data=problem.y_obs, covariance=cov)
    )
    return tda.Posterior(problem.prior, loglike, forward)


def _da_posteriors(problem: Problem, model: ActiveMCMCModel) -> list[tda.Posterior]:
    """The `[coarse, fine]` posterior pair `adaptive_da`'s two-level DA sampling uses."""
    return [
        _posterior(problem, model.coarse, surrogate=True),
        _posterior(problem, model.fine, surrogate=False),
    ]


@dataclass
class _ChunkState:
    """Resumable per-replicate state for one method's chunked, round-based MCMC run.

    The same shape covers all of: HF-only (`model=None`, no surrogate involved),
    pretrained/online_active (single posterior), and `adaptive_da`'s production phase
    (two-posterior DA, frozen model, fixed `subsampling_rate`) -- only the objects
    inside differ, not the chunking mechanics.

    Attributes
    ----------
    posterior
        `tda.Posterior` (single-posterior methods) or `list[tda.Posterior]`
        (`adaptive_da`'s two-level DA).
    proposal
        The (shared, adaptive) MCMC proposal.
    theta_current
        Current chain position, advanced in place by each chunk.
    model
        `ActiveMCMCModel`, or `None` for pure HF-only.
    subsampling_rate
        Coarse steps per outer iteration (DA block size); `1` for single-posterior
        methods.
    chain_key
        `tinyDA` chain key to extract per chunk.
    """

    posterior: Any
    proposal: Any
    theta_current: FloatArray
    model: Any | None = None
    subsampling_rate: int = 1
    chain_key: str = "chain_0"


# ---------------------------------------------------------------------------
# Method 1: HF-only Metropolis-Hastings (reference)
# ---------------------------------------------------------------------------


def run_hf_only(problem: Problem, *, iterations: int, seed: int, theta0: FloatArray | None = None) -> MCMCChain:
    """Run a single HF-only Metropolis-Hastings chain -- the ground-truth reference
    every other method is compared against.

    Parameters
    ----------
    problem
        Inverse-problem definition.
    iterations
        Number of MCMC iterations.
    seed
        Seed for the initial-parameter draw (if `theta0` is not given) and the
        underlying RNG.
    theta0
        Starting parameter vector. Drawn from `problem.prior` if not given.

    Returns
    -------
    chain
        The sampled chain (`used_hf` all `True`).
    """
    posterior = _posterior(problem, problem.hf_forward, surrogate=False)
    proposal = make_proposal(problem)

    rng = set_seed(seed)
    if theta0 is None:
        theta0 = np.asarray(problem.prior.rvs(random_state=rng), dtype=float)

    chain_obj = tda.sample(
        posteriors=posterior,
        proposal=proposal,
        iterations=int(iterations),
        n_chains=1,
        force_sequential=True,
        initial_parameters=theta0,
        adaptive_error_model=None,
    )
    samples = extract_samples(chain=chain_obj, chain_key="chain_0")
    used_hf = np.ones(samples.shape[0], dtype=bool)
    return MCMCChain.from_arrays(samples=samples, used_hf=used_hf)


# ---------------------------------------------------------------------------
# Method 2: pretrained surrogate (offline active-learning design) + frozen MH
# ---------------------------------------------------------------------------


def run_pretrained(
    problem: Problem,
    *,
    seed_X: FloatArray,
    seed_Y: FloatArray,
    gamma_threshold: float,
    pod_rank: int,
    kernel: KernelName,
    iterations: int,
    seed: int,
    theta0: FloatArray | None = None,
) -> tuple[MCMCChain, int, PODGPSurrogate]:
    """Train a surrogate purely offline (`active_learning_offline_design`), freeze it,
    then run a single Metropolis-Hastings chain against it (no HF correction).

    Parameters
    ----------
    problem
        Inverse-problem definition.
    seed_X, seed_Y
        Initial training design.
    gamma_threshold
        Worst-case predictive-standard-deviation stopping target for the offline
        design (see `active_learning_offline_design`).
    pod_rank
        POD basis rank.
    kernel
        GP kernel name.
    iterations
        Number of MCMC iterations.
    seed
        Seed for the offline-design RNG, the initial-parameter draw, and sampling.
    theta0
        Starting parameter vector. Drawn from `problem.prior` if not given.

    Returns
    -------
    chain, n_hf_spent, surrogate
        The sampled chain, the total number of HF evaluations spent training the
        surrogate, and the trained (frozen) `PODGPSurrogate`.
    """
    rng = set_seed(seed)
    surrogate = active_learning_offline_design(
        problem, seed_X, seed_Y, gamma_threshold=gamma_threshold, pod_rank=pod_rank, kernel=kernel, rng=rng
    )
    n_hf_spent = surrogate.gp.n_train

    model = ActiveMCMCModel(lf_model=surrogate, hf_model=problem.hf_forward, gamma_threshold=0.0, frozen=True)
    posterior = _posterior(problem, model.coarse, surrogate=True)
    proposal = make_proposal(problem)

    if theta0 is None:
        theta0 = np.asarray(problem.prior.rvs(random_state=rng), dtype=float)

    result = sample_active_chain(
        model=model,
        posterior=posterior,
        proposal=proposal,
        iterations=int(iterations),
        initial_parameters=theta0,
        subsampling_rate=1,
        chain_key="chain_0",
    )
    return result.chain, n_hf_spent, surrogate


# ---------------------------------------------------------------------------
# Method 3: Riccius-style online active learning (single posterior, no DA)
# ---------------------------------------------------------------------------


def run_online_active(
    problem: Problem,
    *,
    surrogate: PODGPSurrogate,
    gamma_threshold: float,
    iterations: int,
    seed: int,
    theta0: FloatArray | None = None,
    online_learning: OnlineLearningConfig = DEFAULT_ONLINE_LEARNING,
) -> tuple[MCMCChain, PODGPSurrogate]:
    """Run a single chain where the surrogate directly substitutes for the likelihood
    and is refined online via an uncertainty threshold, with no HF-correction step
    (Riccius et al. 2025, Algorithm 1).

    Parameters
    ----------
    problem
        Inverse-problem definition.
    surrogate
        Shared starting surrogate, deep-copied internally (see
        `_surrogate_for_online_learning`) so the caller's copy is untouched.
    gamma_threshold
        Predictive-variance trust threshold gating HF correction calls.
    iterations
        Number of MCMC iterations.
    seed
        Seed for the initial-parameter draw and sampling.
    theta0
        Starting parameter vector. Drawn from `problem.prior` if not given.
    online_learning
        POD/GP refit and rank settings for the online-learning surrogate.

    Returns
    -------
    chain, surrogate
        The sampled chain and the (online-updated) surrogate at the end of sampling.
    """
    model = ActiveMCMCModel(
        lf_model=_surrogate_for_online_learning(surrogate, online_learning),
        hf_model=problem.hf_forward,
        gamma_threshold=gamma_threshold,
    )
    posterior = _posterior(problem, model.coarse, surrogate=True)
    proposal = make_proposal(problem)

    rng = set_seed(seed)
    if theta0 is None:
        theta0 = np.asarray(problem.prior.rvs(random_state=rng), dtype=float)

    result = sample_active_chain(
        model=model,
        posterior=posterior,
        proposal=proposal,
        iterations=int(iterations),
        initial_parameters=theta0,
        subsampling_rate=1,
        chain_key="chain_0",
    )
    return result.chain, cast(PODGPSurrogate, model.lf_model)


# ---------------------------------------------------------------------------
# Method 4: adaptive delayed-acceptance MCMC with freeze-to-production
# ---------------------------------------------------------------------------


def run_adaptive_da(
    problem: Problem,
    *,
    surrogate: PODGPSurrogate,
    gamma_threshold: float,
    n_coarse_evals: int,
    max_adapt_coarse_evals: int,
    seed: int,
    theta0: FloatArray | None = None,
    online_learning: OnlineLearningConfig = DEFAULT_ONLINE_LEARNING,
    max_subchain: int = 10_000,
) -> tuple[MCMCChain, dict[str, Any], PODGPSurrogate]:
    """Adaptive delayed-acceptance MCMC with a freeze-to-production stage: an
    `AdaptiveSubchain` policy grows the coarse-to-fine subsampling rate as the
    surrogate proves trustworthy, then freezes the surrogate/rate once converged and
    switches to a plain, HF-corrected DA production chain.

    Parameters
    ----------
    problem
        Inverse-problem definition.
    surrogate
        Shared starting surrogate, deep-copied internally so the caller's copy is
        untouched.
    gamma_threshold
        Predictive-variance trust threshold gating HF correction calls.
    n_coarse_evals
        Total coarse-evaluation budget across both phases.
    max_adapt_coarse_evals
        Coarse-evaluation ceiling on the adaptive phase alone.
    seed
        Seed for the initial-parameter draw and sampling.
    theta0
        Starting parameter vector. Drawn from `problem.prior` if not given.
    online_learning
        POD/GP refit and rank settings for the online-learning surrogate.
    max_subchain
        Ceiling on `AdaptiveSubchain`'s subsampling rate (coarse steps between HF
        corrections), both during the adaptive phase and after freezing for
        production.

    Returns
    -------
    chain, metadata, surrogate
        The concatenated adapt+production chain, `sample_adaptive_then_frozen_chain`'s
        metadata dict, and the (possibly refit) surrogate.
    """
    state = AdaptiveSubchainState(subchain_length=10)
    control = AdaptiveSubchainControl(
        update_every=5,
        target_error=problem.sigma_obs,
        min_subchain=1,
        max_subchain=max_subchain,
        grow_factor=2.0,
        shrink_factor=0.5,
        patience=5,
    )
    hook = AdaptiveSubchain(state=state, control=control)

    model = ActiveMCMCModel(
        lf_model=_surrogate_for_online_learning(surrogate, online_learning),
        hf_model=problem.hf_forward,
        gamma_threshold=gamma_threshold,
        adaptive=hook,
    )

    proposal = make_proposal(problem)
    rng = set_seed(seed)
    if theta0 is None:
        theta0 = np.asarray(problem.prior.rvs(random_state=rng), dtype=float)

    result = sample_adaptive_then_frozen_chain(
        model=model,
        posterior_factory=functools.partial(_da_posteriors, problem),
        proposal=proposal,
        n_coarse_evals=n_coarse_evals,
        max_adapt_coarse_evals=max_adapt_coarse_evals,
        initial_parameters=theta0,
        chain_key="chain_coarse_0",  # adaptive phase: diagnostic only, discarded as burn-in
        production_chain_key="chain_fine_0",  # production phase: the actual HF-corrected posterior
        # Also pull the production phase's intra-block low-fidelity trajectory, purely
        # for trace plots -- `chain_fine_0` alone (the actual posterior) only exposes
        # one state per DA block, hiding every cheap coarse step in between.
        production_diagnostic_chain_key="chain_coarse_0",
        config=ChunkedMCMCConfig(chain_key="chain_coarse_0", chunk_size=100),
    )
    return result.chain, result.metadata, cast(PODGPSurrogate, model.lf_model)


# ---------------------------------------------------------------------------
# Training-cost comparison: `adaptive_da`'s online, MCMC-path-guided active learning
# vs. `pretrained`'s offline, global greedy-max-variance active learning -- how much
# HF budget each needs to reach a trustworthy surrogate. Training only; no downstream
# MCMC production phase runs on either side, since a DA-corrected posterior targets the
# truth regardless of how its surrogate was trained, making a posterior-accuracy
# comparison uninformative about training strategy specifically.
# ---------------------------------------------------------------------------


def run_training_cost_comparison(
    problem: Problem,
    *,
    seed_X: FloatArray,
    seed_Y: FloatArray,
    seed_surrogate: PODGPSurrogate,
    gamma_threshold: float,
    pod_rank: int,
    kernel: KernelName,
    max_adapt_coarse_evals: int,
    seed_base: int,
    online_learning: OnlineLearningConfig = DEFAULT_ONLINE_LEARNING,
    max_subchain: int = 10_000,
) -> tuple[dict[str, Any], PODGPSurrogate]:
    """Training-cost comparison for one problem instance: offline greedy active
    learning (`active_learning_offline_design`) vs. `adaptive_da`'s online adaptive
    phase, both starting from the same shared `seed_X`/`seed_Y` design and run to
    their own convergence criterion.

    `adaptive_da`'s adaptive phase is obtained via `run_adaptive_da` with
    `n_coarse_evals` capped at `max_adapt_coarse_evals`, so the production phase is
    skipped entirely rather than run and discarded.

    Parameters
    ----------
    problem
        Inverse-problem definition.
    seed_X, seed_Y
        Shared initial training design.
    seed_surrogate
        Shared starting surrogate for `adaptive_da`'s adaptive phase.
    gamma_threshold
        Predictive-variance/-standard-deviation trust threshold, shared by both sides.
    pod_rank
        POD basis rank.
    kernel
        GP kernel name, used by the offline side.
    max_adapt_coarse_evals
        Coarse-evaluation ceiling for `adaptive_da`'s adaptive phase.
    seed_base
        Base seed; the offline and online sides use `seed_base + 500`/`+ 600`
        respectively so a shared `seed_base` never collides across calls.
    online_learning, max_subchain
        Forwarded to `run_adaptive_da`.

    Returns
    -------
    metrics, offline_surrogate
        `metrics` is a JSON-serializable dict with `n_init`, `offline` (`n_hf_total`,
        `n_hf_extra` -- HF calls beyond the shared seed design, `wall_time_s`), and
        `online` (`n_hf_extra`, `coarse_evals_used`, `wall_time_s`, `converged`,
        `final_subchain_length`, `subchain_length_history`). `offline_surrogate` is the
        trained `PODGPSurrogate` (not JSON-serializable, kept separate).
    """
    n_init = int(seed_X.shape[0])

    t0 = time.time()
    offline_surrogate, offline_n_hf_total = _train_pretrained_surrogate(
        problem, seed_X=seed_X, seed_Y=seed_Y, gamma_threshold=gamma_threshold, pod_rank=pod_rank,
        kernel=kernel, seed=seed_base + 500,
    )
    offline_wall_time = time.time() - t0

    t0 = time.time()
    adapt_chain, adapt_meta, _adapt_surrogate = run_adaptive_da(
        problem, surrogate=seed_surrogate, gamma_threshold=gamma_threshold,
        n_coarse_evals=max_adapt_coarse_evals, max_adapt_coarse_evals=max_adapt_coarse_evals,
        seed=seed_base + 600, online_learning=online_learning, max_subchain=max_subchain,
    )
    online_wall_time = time.time() - t0

    online_used_hf = (
        adapt_chain.extras.used_hf if adapt_chain.extras.used_hf is not None
        else np.zeros(adapt_chain.n_steps, dtype=bool)
    )
    subchain_hist = adapt_chain.extras.subchain_length
    # No production phase ran here, so metadata is "adapt_only" shaped: the coarse-eval
    # count lives in the nested adapt_metadata, not the top-level key.
    coarse_evals_used = adapt_meta.get("adapt_coarse_evals_used")
    if coarse_evals_used is None:
        coarse_evals_used = adapt_meta["adapt_metadata"]["coarse_evals_used"]
    metrics: dict[str, Any] = {
        "n_init": n_init,
        "offline": {
            "n_hf_total": int(offline_n_hf_total),
            "n_hf_extra": int(offline_n_hf_total) - n_init,
            "wall_time_s": offline_wall_time,
        },
        "online": {
            "n_hf_extra": int(np.sum(online_used_hf)),
            "coarse_evals_used": int(coarse_evals_used),
            "wall_time_s": online_wall_time,
            "converged": bool(adapt_meta["converged"]),
            "final_subchain_length": int(subchain_hist[-1]) if subchain_hist is not None else None,
            "subchain_length_history": subchain_hist.tolist() if subchain_hist is not None else None,
        },
    }
    return metrics, offline_surrogate


# ---------------------------------------------------------------------------
# Chunk-ready state factories for harness.py's round-based orchestration.
# ---------------------------------------------------------------------------


def _init_hf_only_state(problem: Problem, *, seed: int, theta0: FloatArray | None = None) -> _ChunkState:
    posterior = _posterior(problem, problem.hf_forward, surrogate=False)
    proposal = make_proposal(problem)
    rng = set_seed(seed)
    if theta0 is None:
        theta0 = np.asarray(problem.prior.rvs(random_state=rng), dtype=float)
    return _ChunkState(posterior=posterior, proposal=proposal, theta_current=theta0, model=None)


def _train_pretrained_surrogate(
    problem: Problem,
    *,
    seed_X: FloatArray,
    seed_Y: FloatArray,
    gamma_threshold: float,
    pod_rank: int,
    kernel: KernelName,
    seed: int,
) -> tuple[PODGPSurrogate, int]:
    """Train `pretrained`'s offline design once: a fixed upfront cost shared by every
    replicate, not part of the monitored round loop.

    Returns
    -------
    surrogate, n_hf_spent
        The trained surrogate and the number of HF evaluations spent training it.
    """
    rng = set_seed(seed)
    surrogate = active_learning_offline_design(
        problem, seed_X, seed_Y, gamma_threshold=gamma_threshold, pod_rank=pod_rank, kernel=kernel, rng=rng
    )
    n_hf_spent = surrogate.gp.n_train
    return surrogate, n_hf_spent


def _init_pretrained_state_from_surrogate(
    problem: Problem, *, surrogate: PODGPSurrogate, seed: int, theta0: FloatArray | None = None
) -> _ChunkState:
    """Chunk-ready state for `pretrained`'s frozen, single-posterior sampling phase,
    from an already-trained surrogate shared across every replicate (deep-copied here
    so each replicate's `model.log` is independent, though the surrogate weights are
    identical).
    """
    model = ActiveMCMCModel(
        lf_model=copy.deepcopy(surrogate), hf_model=problem.hf_forward, gamma_threshold=0.0, frozen=True
    )
    posterior = _posterior(problem, model.coarse, surrogate=True)
    proposal = make_proposal(problem)

    rng = set_seed(seed)
    if theta0 is None:
        theta0 = np.asarray(problem.prior.rvs(random_state=rng), dtype=float)
    return _ChunkState(posterior=posterior, proposal=proposal, theta_current=theta0, model=model)


def _make_online_active_sync_hook(
    n_seed: int,
) -> Any:
    """Build a `run_until_rhat_converged` `post_round_hook` for `online_active_synced`:
    an `online_active` variant (still no delayed-acceptance correction) where every
    replicate's individually-collected HF points are pooled into a shared set and every
    replicate's surrogate is refit on that pool once per round.

    Targets inter-replicate surrogate disagreement specifically, without adding DA
    correction: a perfectly synchronized surrogate here still only converges the
    replicates to a consistently *biased* posterior, isolating the R-hat-convergence
    problem from the correctness problem `adaptive_da`'s DA correction solves.

    Every replicate starts as a deepcopy of the same seed surrogate, so only each
    replicate's acquired suffix (past `n_seed`) is pooled and deduplicated (exact-row
    match, since duplicates only arise from points already broadcast by a prior sync).
    `refit_pod()` is self-gated on the surrogate's own `pod_refit_max`; this hook peeks
    at replica 0's counter to skip pooling once that budget is spent, after which
    replicates resume learning independently.

    Also overwrites `hf_evals_per_chain` with the true pooled unique-point count each
    round, since each replicate's surrogate embeds every other replicate's HF calls
    too -- the per-chain tally `run_until_rhat_converged` keeps on its own would
    otherwise undercount the real cost.

    Parameters
    ----------
    n_seed
        Size of the shared offline seed design every replicate's surrogate started
        from.

    Returns
    -------
    sync
        A `post_round_hook`-shaped callable: ``(states, hf_evals_per_chain) ->
        (states, hf_evals_per_chain)``.
    """

    def _history(s: PODGPSurrogate) -> tuple[FloatArray, FloatArray]:
        # Every online_active surrogate reaching this hook was deep-copied from
        # seed_surrogate (see _init_online_active_state), which build_initial_surrogate
        # always seeds with non-None X_history/Y_history -- these are never None here.
        assert s.X_history is not None and s.Y_history is not None
        return s.X_history, s.Y_history

    def sync(states: list[_ChunkState], hf_evals_per_chain: list[int]) -> tuple[list[_ChunkState], list[int]]:
        surrogates = [cast(PODGPSurrogate, cast(ActiveMCMCModel, s.model).lf_model) for s in states]
        budget = surrogates[0].pod_refit_max
        if budget is not None and surrogates[0]._pod_refit_count >= budget:
            return states, hf_evals_per_chain
        seed_X = _history(surrogates[0])[0][:n_seed]
        seed_Y = _history(surrogates[0])[1][:n_seed]
        acquired_X = np.concatenate([_history(s)[0][n_seed:] for s in surrogates], axis=0)
        acquired_Y = np.concatenate([_history(s)[1][n_seed:] for s in surrogates], axis=0)
        if acquired_X.shape[0] > 0:
            _, unique_idx = np.unique(acquired_X, axis=0, return_index=True)
            unique_idx = np.sort(unique_idx)
            acquired_X = acquired_X[unique_idx]
            acquired_Y = acquired_Y[unique_idx]
        pooled_X = np.vstack([seed_X, acquired_X]) if acquired_X.shape[0] else seed_X
        pooled_Y = np.vstack([seed_Y, acquired_Y]) if acquired_Y.shape[0] else seed_Y

        for s in surrogates:
            s.X_history = pooled_X.copy()
            s.Y_history = pooled_Y.copy()
            s.refit_pod()

        # Overwrite (not accumulate): this is the true total unique HF spend so far.
        true_hf_count = int(pooled_X.shape[0] - n_seed)
        hf_evals_per_chain = [true_hf_count] * len(states)
        return states, hf_evals_per_chain

    return sync


def _init_online_active_state(
    problem: Problem,
    *,
    surrogate: PODGPSurrogate,
    gamma_threshold: float,
    seed: int,
    theta0: FloatArray | None = None,
    online_learning: OnlineLearningConfig = DEFAULT_ONLINE_LEARNING,
) -> _ChunkState:
    model = ActiveMCMCModel(
        lf_model=_surrogate_for_online_learning(surrogate, online_learning),
        hf_model=problem.hf_forward,
        gamma_threshold=gamma_threshold,
    )
    posterior = _posterior(problem, model.coarse, surrogate=True)
    proposal = make_proposal(problem)
    rng = set_seed(seed)
    if theta0 is None:
        theta0 = np.asarray(problem.prior.rvs(random_state=rng), dtype=float)
    return _ChunkState(posterior=posterior, proposal=proposal, theta_current=theta0, model=model)


def _init_adaptive_da_production_state(
    problem: Problem,
    *,
    surrogate: PODGPSurrogate,
    gamma_threshold: float,
    max_adapt_coarse_evals: int,
    seed: int,
    theta0: FloatArray | None = None,
    online_learning: OnlineLearningConfig = DEFAULT_ONLINE_LEARNING,
    max_subchain: int = 10_000,
) -> tuple[_ChunkState, dict[str, Any], MCMCChain]:
    """Run `adaptive_da`'s adaptive phase to its own `has_converged()` completion --
    unmonitored by cross-chain R-hat, since the target is non-stationary during
    adaptation -- then freeze and return a chunk-ready state for the same round-based,
    R-hat-monitored production loop the other methods use.

    Returns
    -------
    state, adapt_meta, adapt_chain
        `adapt_meta` carries adaptive-phase bookkeeping (`n_adapt_samples`,
        `adapt_coarse_evals_used`) for cost accounting; `adapt_chain` lets callers
        prepend it to the production trace for a full-resolution plot.
    """
    state_hook = AdaptiveSubchainState(subchain_length=10)
    control = AdaptiveSubchainControl(
        update_every=5,
        target_error=problem.sigma_obs,
        min_subchain=1,
        max_subchain=max_subchain,
        grow_factor=2.0,
        shrink_factor=0.5,
        patience=5,
    )
    hook = AdaptiveSubchain(state=state_hook, control=control)

    model = ActiveMCMCModel(
        lf_model=_surrogate_for_online_learning(surrogate, online_learning),
        hf_model=problem.hf_forward,
        gamma_threshold=gamma_threshold,
        adaptive=hook,
    )
    proposal = make_proposal(problem)
    rng = set_seed(seed)
    if theta0 is None:
        theta0 = np.asarray(problem.prior.rvs(random_state=rng), dtype=float)

    with _suppress_tinyda_output():
        adapt_result = sample_adaptive_active_chain(
            model=model,
            posterior=_da_posteriors(problem, model),
            proposal=proposal,
            n_coarse_evals=max_adapt_coarse_evals,
            initial_parameters=theta0,
            chain_key="chain_coarse_0",
            config=ChunkedMCMCConfig(chain_key="chain_coarse_0", chunk_size=100),
            stop_check=hook.has_converged,
        )
    adapt_meta = {
        "n_adapt_samples": int(adapt_result.chain.n_steps),
        "adapt_coarse_evals_used": int(adapt_result.metadata["coarse_evals_used"]),
        "converged_adaptive": bool(hook.has_converged()),
    }

    frozen_model = model.freeze()
    frozen_rate = int(hook.state.subchain_length)
    theta_last = adapt_result.chain.samples[-1]

    state = _ChunkState(
        posterior=_da_posteriors(problem, frozen_model),
        proposal=proposal,
        theta_current=theta_last,
        model=frozen_model,
        subsampling_rate=frozen_rate,
        chain_key="chain_fine_0",
    )
    return state, adapt_meta, adapt_result.chain


def _init_adaptive_da_production_state_from_frozen(
    problem: Problem, *, frozen_model: ActiveMCMCModel, frozen_rate: int, theta0: FloatArray
) -> _ChunkState:
    """Chunk-ready production state for `adaptive_da`, from an already-frozen
    model/rate shared across every replicate (deep-copied here so each replicate's
    `model.log` is independent, though the frozen surrogate weights and
    `subsampling_rate` are identical). `theta0` is shared too, typically the single
    adaptive phase's last state.
    """
    model_copy = copy.deepcopy(frozen_model)
    proposal = make_proposal(problem)
    return _ChunkState(
        posterior=_da_posteriors(problem, model_copy),
        proposal=proposal,
        theta_current=theta0,
        model=model_copy,
        subsampling_rate=frozen_rate,
        chain_key="chain_fine_0",
    )
