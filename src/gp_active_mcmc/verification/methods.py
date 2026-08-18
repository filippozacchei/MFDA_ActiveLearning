"""The comparison methods: hf_only, pretrained, adaptive_surrogate_mcmc, adaptive_stm.

Also defines `_ChunkState` (the per-replicate state `harness.py`'s round-based
orchestration resumes one chunk at a time) since every `_init_*_state` factory below
builds one.
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
    "run_adaptive_stm",
    "run_hf_only",
    "run_pretrained",
    "run_training_cost_comparison",
]


@contextlib.contextmanager
def _suppress_tinyda_output() -> Any:
    """Silences tinyDA's per-call print + tqdm bar (it has no verbosity flag)."""
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        yield


def _posterior(problem: Problem, forward: Callable[[FloatArray], FloatArray], *, surrogate: bool) -> tda.Posterior:
    """Gaussian-likelihood ``tda.Posterior`` (cov ``sigma_obs**2 * I``) wrapping
    `forward`. `surrogate=True` uses `ActiveGPLogLike`, which accounts for the GP's own
    predictive variance; `surrogate=False` uses plain `tda.AdaptiveGaussianLogLike`.
    """
    cov = (problem.sigma_obs**2) * np.eye(len(problem.y_obs))
    loglike = (
        ActiveGPLogLike(data=problem.y_obs, covariance=cov)
        if surrogate
        else tda.AdaptiveGaussianLogLike(data=problem.y_obs, covariance=cov)
    )
    return tda.Posterior(problem.prior, loglike, forward)


def _da_posteriors(problem: Problem, model: ActiveMCMCModel) -> list[tda.Posterior]:
    """The `[coarse, fine]` posterior pair `adaptive_stm`'s two-level DA sampling uses."""
    return [
        _posterior(problem, model.coarse, surrogate=True),
        _posterior(problem, model.fine, surrogate=False),
    ]


@dataclass
class _ChunkState:
    """Resumable per-replicate state for one method's chunked, round-based MCMC run.
    One shape for all of: HF-only (`model=None`), pretrained/adaptive_surrogate_mcmc
    (single posterior), and `adaptive_stm`'s production phase (two-posterior DA, frozen
    model, fixed `subsampling_rate`) -- only the objects inside differ, not the
    mechanics.
    """

    posterior: Any  # tda.Posterior, or list[tda.Posterior] for adaptive_stm's two-level DA
    proposal: Any
    theta_current: FloatArray
    model: Any | None = None  # ActiveMCMCModel, or None for pure HF-only
    subsampling_rate: int = 1  # DA block size; 1 for single-posterior methods
    chain_key: str = "chain_0"  # tinyDA chain key to extract per chunk


# Per-method seed offsets within one seed_base block -- single source of truth for
# harness.py and run_training_cost_comparison below, so a new method can't silently
# collide with an existing one (each entry must stay >= n_chains from its neighbors;
# enforced by test_seed_offsets_are_pairwise_disjoint).
_SEED_OFFSETS: dict[str, int] = {
    "hf_only": 100,
    "adaptive_surrogate_mcmc": 350,
    "adaptive_stm": 400,
    "training_cost_offline": 500,
    "training_cost_online": 600,
}


# --- Method 1: HF-only Metropolis-Hastings (reference) ---------------------


def run_hf_only(problem: Problem, *, iterations: int, seed: int, theta0: FloatArray | None = None) -> MCMCChain:
    """Runs a single HF-only MH chain (`used_hf` all `True`) -- the ground-truth
    reference every other method is compared against. `theta0` defaults to a
    `problem.prior` draw."""
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


# --- Method 2: pretrained surrogate (offline active-learning design) + frozen MH ---


def run_pretrained(
    problem: Problem,
    *,
    seed_X: FloatArray,
    seed_Y: FloatArray,
    gamma_threshold: float,
    kernel: KernelName,
    iterations: int,
    seed: int,
    theta0: FloatArray | None = None,
    rank_energy_threshold: float = 0.999,
    rank_max: int | None = None,
) -> tuple[MCMCChain, int, PODGPSurrogate]:
    """Trains a surrogate purely offline (`active_learning_offline_design`, POD rank
    adaptively re-derived at every refit -- no fixed-rank option), freezes it, then
    runs a single MH chain against it (no HF correction). Returns the chain, the total
    HF evaluations spent training, and the frozen surrogate."""
    rng = set_seed(seed)
    surrogate = active_learning_offline_design(
        problem, seed_X, seed_Y, gamma_threshold=gamma_threshold, kernel=kernel, rng=rng,
        rank_energy_threshold=rank_energy_threshold, rank_max=rank_max,
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


# --- Method 3: adaptive delayed-acceptance MCMC with freeze-to-production ---


def run_adaptive_stm(
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
    surrogate proves trustworthy, then freezes surrogate + rate once converged (or
    `max_adapt_coarse_evals` is hit) and switches to a plain, HF-corrected DA
    production chain for the rest of `n_coarse_evals`. Returns the concatenated
    adapt+production chain, `sample_adaptive_then_frozen_chain`'s metadata dict, and
    the (possibly refit) surrogate."""
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


# --- Training-cost comparison: adaptive_stm's online, MCMC-path-guided active
# learning vs. pretrained's offline, global greedy-max-variance active learning --
# how much HF budget each needs for a trustworthy surrogate. Training only, no
# downstream MCMC: a DA-corrected posterior targets the truth regardless of how its
# surrogate was trained, so a posterior-accuracy comparison wouldn't say anything
# about training strategy specifically. ---


def run_training_cost_comparison(
    problem: Problem,
    *,
    seed_X: FloatArray,
    seed_Y: FloatArray,
    seed_surrogate: PODGPSurrogate,
    gamma_threshold: float,
    kernel: KernelName,
    max_adapt_coarse_evals: int,
    seed_base: int,
    online_learning: OnlineLearningConfig = DEFAULT_ONLINE_LEARNING,
    max_subchain: int = 10_000,
) -> tuple[dict[str, Any], PODGPSurrogate]:
    """Offline greedy active learning (`active_learning_offline_design`) vs.
    `adaptive_stm`'s online adaptive phase (via `run_adaptive_stm` with `n_coarse_evals`
    capped at `max_adapt_coarse_evals`, so its production phase never runs), both from
    the same shared `seed_X`/`seed_Y` design.

    Returns
    -------
    metrics, offline_surrogate
        `metrics` (JSON-serializable): `n_init`, `offline` (`n_hf_total`, `n_hf_extra`,
        `wall_time_s`), `online` (`n_hf_extra`, `coarse_evals_used`, `wall_time_s`,
        `converged`, `final_subchain_length`, `subchain_length_history`).
        `offline_surrogate`: the trained `PODGPSurrogate`, kept separate since it isn't
        JSON-serializable.
    """
    n_init = int(seed_X.shape[0])

    t0 = time.time()
    # Offline side reuses online_learning's rank-derivation tuning: there's one
    # adaptive-rank policy in this package, not a separate one per method.
    offline_surrogate, offline_n_hf_total = _train_pretrained_surrogate(
        problem, seed_X=seed_X, seed_Y=seed_Y, gamma_threshold=gamma_threshold, kernel=kernel,
        seed=seed_base + _SEED_OFFSETS["training_cost_offline"],
        rank_energy_threshold=online_learning.rank_energy_threshold, rank_max=online_learning.rank_max,
    )
    offline_wall_time = time.time() - t0

    t0 = time.time()
    adapt_chain, adapt_meta, _adapt_surrogate = run_adaptive_stm(
        problem, surrogate=seed_surrogate, gamma_threshold=gamma_threshold,
        n_coarse_evals=max_adapt_coarse_evals, max_adapt_coarse_evals=max_adapt_coarse_evals,
        seed=seed_base + _SEED_OFFSETS["training_cost_online"], online_learning=online_learning,
        max_subchain=max_subchain,
    )
    online_wall_time = time.time() - t0

    online_used_hf = (
        adapt_chain.extras.used_hf if adapt_chain.extras.used_hf is not None
        else np.zeros(adapt_chain.n_steps, dtype=bool)
    )
    subchain_hist = adapt_chain.extras.subchain_length
    # adapt_meta["adapt_metadata"]["coarse_evals_used"] is present in both of
    # sample_adaptive_then_frozen_chain's metadata shapes ("adapt_only" and
    # "adapt_then_production") and is always the adaptive phase's own coarse-eval count
    # (confirmed against gp_active_mcmc/inference/sampling.py: the top-level
    # "adapt_coarse_evals_used" key, on the branch where it exists at all, is set to
    # exactly this same value) -- the quantity this comparison wants, since it measures
    # training cost for the adaptive phase specifically (see the module-level comment
    # above on why no production phase runs on either side of this comparison).
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


# --- Chunk-ready state factories for harness.py's round-based orchestration ---


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
    kernel: KernelName,
    seed: int,
    rank_energy_threshold: float = 0.999,
    rank_max: int | None = None,
) -> tuple[PODGPSurrogate, int]:
    """Trains `pretrained`'s offline design once -- a fixed upfront cost shared by
    every replicate, not part of the monitored round loop. Returns the trained
    surrogate and the HF evaluations spent training it."""
    rng = set_seed(seed)
    surrogate = active_learning_offline_design(
        problem, seed_X, seed_Y, gamma_threshold=gamma_threshold, kernel=kernel, rng=rng,
        rank_energy_threshold=rank_energy_threshold, rank_max=rank_max,
    )
    n_hf_spent = surrogate.gp.n_train
    return surrogate, n_hf_spent


def _make_adaptive_surrogate_mcmc_sync_hook(
    n_seed: int,
) -> Any:
    """Builds a `run_until_rhat_converged` `post_round_hook` for
    `adaptive_surrogate_mcmc`: pools every replicate's individually-collected HF points
    (past the shared `n_seed`-sized offline design, deduplicated) into one set each
    round and refits every replicate's surrogate on it -- still no DA correction, so
    this isolates inter-replicate disagreement from the separate correctness problem
    DA solves (a synchronized surrogate here still only converges replicates to a
    consistently *biased* posterior). Also overwrites `hf_evals_per_chain` with the
    true pooled unique-point count, since each surrogate embeds every replicate's HF
    calls and the per-chain tally would otherwise undercount. Stops pooling once
    `pod_refit_max` is spent (checked via replica 0's counter), after which replicates
    resume learning independently.
    """

    def _history(s: PODGPSurrogate) -> tuple[FloatArray, FloatArray]:
        # Every adaptive_surrogate_mcmc surrogate reaching this hook was deep-copied
        # from seed_surrogate (see _init_adaptive_surrogate_mcmc_state), which
        # build_initial_surrogate always seeds with non-None X_history/Y_history --
        # these are never None here.
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


def _init_adaptive_surrogate_mcmc_state(
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


def _init_adaptive_stm_production_state(
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
    """Runs `adaptive_stm`'s adaptive phase to its own `has_converged()` completion
    (unmonitored by cross-chain R-hat, since the target is non-stationary during
    adaptation), then freezes and returns a chunk-ready state for the same
    round-based, R-hat-monitored production loop the other methods use. `adapt_meta`
    carries `n_adapt_samples`/`adapt_coarse_evals_used` for cost accounting;
    `adapt_chain` lets callers prepend it to the production trace for a
    full-resolution plot."""
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


def _init_adaptive_stm_production_state_from_frozen(
    problem: Problem, *, frozen_model: ActiveMCMCModel, frozen_rate: int, theta0: FloatArray
) -> _ChunkState:
    """Chunk-ready production state for `adaptive_stm`, from a frozen model/rate
    shared across replicates (deep-copied so each replicate's `model.log` is
    independent). `theta0` is typically the shared adaptive phase's last state."""
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
