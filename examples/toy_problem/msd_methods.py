"""Shared library for the mass-spring-damper benchmark comparison.

Implements four inference strategies at a (roughly) matched high-fidelity (HF) budget:

1. HF-only Metropolis-Hastings (Algorithm 1) -- ground-truth reference and the
   "standard single-stage MCMC" baseline. See `run_hf_only`.
2. Pretrained surrogate + plain MH -- POD-GP surrogate trained *offline only* via a
   greedy max-variance active-learning acquisition, then frozen; matches the
   "Pretrained Model" arm in Riccius et al. (2025). See `run_pretrained`.
3. Riccius-style online active learning (single posterior, no delayed acceptance) --
   the surrogate is substituted directly for the likelihood and refined online via an
   uncertainty threshold, with no HF-correction step. Matches Algorithm 1 in
   Riccius et al. (2025), "Integration of Active Learning and MCMC Sampling for
   Efficient Bayesian Calibration of Mechanical Properties". See `run_online_active`.
4. Proposed: adaptive delayed-acceptance MCMC with a freeze-to-production stage
   (paper Algorithm 3). See `run_ours`.

This module has no `__main__`: it is imported by `run_sweep_msd.py` (the multi-seed
sweep driver), `check_convergence.py` (multi-chain R-hat/ESS diagnostic), and
`msd_benchmark.ipynb` (the walkthrough notebook).
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import tinyDA as tda
from msd import make_forward_model, make_observation, make_prior, make_timeline

from gp_active_mcmc.inference import (
    ActiveGPLogLike,
    ActiveMCMCModel,
    AdaptiveMetropolisShared,
    AdaptiveSubchain,
    AdaptiveSubchainControl,
    AdaptiveSubchainState,
    ChunkedMCMCConfig,
    MCMCChain,
    sample_active_chain,
    sample_adaptive_active_chain,
    sample_adaptive_then_frozen_chain,
)
from gp_active_mcmc.surrogates import POD, MultiOutputGP, PODGPSurrogate
from gp_active_mcmc.utils.mcmc import extract_samples, posterior_rmse
from gp_active_mcmc.utils.rng import set_seed

RESULTS_DIR = Path(__file__).parent / "results"

PARAM_NAMES = ("k", "c")

# ---------------------------------------------------------------------------
# Shared experiment configuration (single source of truth for all consumers)
# ---------------------------------------------------------------------------

KERNEL = "matern52"
GAMMA_THRESHOLD = 0.01
N_COARSE_EVALS = 10000
MAX_ADAPT_COARSE_EVALS = 400
BURN_IN = 100
N_REF_ITERATIONS = 3000
REF_BURN_IN = 300

# Deliberately under-trained offline designs + tight uncertainty trigger, under a wide
# prior: with a narrow prior or a large offline design, a naive surrogate already
# covers the plausible region densely enough that no method (including ones with no
# correction mechanism at all) ever needs to learn online, which defeats the point of
# comparing them. n_init sweep used for the sensitivity study.
#
# POD_RANK is fixed across every n_init setting and every method: it is a
# representational-capacity choice (how many modes the linear reduced basis keeps),
# independent of how much training data or GP-regression uncertainty there is. Coupling
# it to n_init (as an earlier version of this benchmark did) confounds the comparison --
# a too-low rank makes the surrogate confidently wrong in a way GP predictive variance
# cannot see (reconstruction bias from a truncated basis, not regression uncertainty),
# which previously caused e.g. `pretrained` at small n_init to "converge" to its
# stopping criterion almost immediately while being catastrophically biased.
N_INIT_SETTINGS = (10, 20, 40)
POD_RANK = 8

# ---------------------------------------------------------------------------
# Problem setup (shared across all methods and seeds)
# ---------------------------------------------------------------------------


@dataclass
class Problem:
    t: np.ndarray
    prior: Any
    theta_true: np.ndarray
    y_obs: np.ndarray
    sigma_obs: float
    hf_forward: Any


def build_problem(*, problem_seed: int, sigma_obs: float = 0.02) -> Problem:
    rng = set_seed(problem_seed)
    t = make_timeline(T=300, t_end=4.0)
    prior = make_prior()
    hf_forward = make_forward_model(t)

    theta_true = prior.rvs(random_state=rng)
    y_obs = make_observation(rng, theta_true, t, sigma_obs)

    return Problem(t=t, prior=prior, theta_true=theta_true, y_obs=y_obs, sigma_obs=sigma_obs, hf_forward=hf_forward)


def build_initial_surrogate(
    problem: Problem, rng: np.random.Generator, *, n_init: int, pod_rank: int, kernel: str = KERNEL
) -> tuple[PODGPSurrogate, np.ndarray, np.ndarray]:
    """Offline seed design shared by the surrogate-based methods (2, 3, 4)."""
    X = np.asarray([problem.prior.rvs(random_state=rng) for _ in range(n_init)], dtype=float)
    Y = np.asarray([problem.hf_forward(theta) for theta in X], dtype=float)

    pod = POD(rank=pod_rank).fit(Y)
    A = pod.transform(Y)
    gp = MultiOutputGP(X_train=X, Y_train=A, kernel=kernel, ard=True, noise_variance=1e-6)
    surrogate = PODGPSurrogate(pod=pod, gp=gp)
    return surrogate, X, Y


def make_proposal(problem: Problem, *, scale: float = 0.05) -> AdaptiveMetropolisShared:
    """Adaptive random-walk Metropolis (Haario et al. 2001), shared by all four methods.

    C0 = scale * prior.cov; covariance re-adapted from the running sample history every
    `period` steps; `sd=1` is a more conservative AM scaling than Haario's textbook
    default (min(1, 2.4**2/d)); `share_across_deepcopy=True` is required because the
    chunked/adaptive samplers deep-copy the proposal internally, and without it the
    adaptation state would reset every chunk instead of persisting.
    """
    return AdaptiveMetropolisShared(
        C0=scale * problem.prior.cov,
        period=100,
        share_across_deepcopy=True,
        adaptive=True,
        sd=1,
    )


# ---------------------------------------------------------------------------
# Method 1: HF-only Metropolis-Hastings (reference)
# ---------------------------------------------------------------------------


def run_hf_only(
    problem: Problem, *, iterations: int, seed: int, theta0: np.ndarray | None = None
) -> MCMCChain:
    cov = (problem.sigma_obs**2) * np.eye(len(problem.y_obs))
    loglike = tda.AdaptiveGaussianLogLike(data=problem.y_obs, covariance=cov)
    posterior = tda.Posterior(problem.prior, loglike, problem.hf_forward)
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


def active_learning_offline_design(
    problem: Problem,
    seed_X: np.ndarray,
    seed_Y: np.ndarray,
    *,
    gamma_threshold: float,
    pod_rank: int,
    kernel: str,
    rng: np.random.Generator,
    candidate_pool_size: int = 500,
    batch_size: int = 25,
    max_total_budget: int = 600,
    n_validation: int = 100,
) -> PODGPSurrogate:
    """Greedy max-predictive-variance acquisition, purely offline (no MCMC path).

    Stopping rule: keep acquiring until the surrogate's *own* predictive variance,
    evaluated over a FIXED validation grid of `n_validation` points (drawn once from
    the prior at the start, not resampled every iteration), has *worst-case* (max, not
    mean) per-point average variance at or below `gamma_threshold**2` -- the same
    criterion the online methods use to decide whether to trust the surrogate at all.
    Using a fixed grid and a max (rather than a fresh pool and a mean) means the
    criterion cannot be satisfied by a surrogate that is excellent almost everywhere
    but still has a pocket of high uncertainty somewhere in the prior's support, and it
    cannot drift between iterations just because a new random pool happened to avoid
    that pocket. This makes the offline budget self-determined by an actual worst-case
    accuracy target rather than an externally borrowed, run-to-run-variable number, and
    lets this baseline run independently of the other methods. `max_total_budget` is a
    safety valve in case the criterion is never satisfied (e.g. an unreachably tight
    threshold).

    Points are acquired in batches of `batch_size` rather than one at a time: each
    point within a batch is chosen by ranking a fresh candidate pool against the
    surrogate state at the *start* of the batch (not updated point-by-point within
    it), and the surrogate is refit once per batch rather than once per point. A
    single-point update refits the whole GP (O(n_train^3)), so refitting after every
    acquired point makes total cost scale with the *number of points added*, not just
    the final training-set size; batching turns that into `n_added / batch_size`
    refits instead. This is also a fairly standard/realistic simplification of batch
    active learning (real HF evaluations are rarely scheduled strictly one at a time).
    The acquisition candidate pool is resampled fresh each pick and is kept separate
    from the fixed validation grid, so validation is never contaminated by what the
    surrogate is trained on.
    """
    X_all = [seed_X.copy()]
    Y_all = [seed_Y.copy()]

    val_grid = np.asarray(problem.prior.rvs(size=n_validation, random_state=rng), dtype=float)

    pod = POD(rank=pod_rank).fit(seed_Y)
    A = pod.transform(seed_Y)
    gp = MultiOutputGP(X_train=seed_X, Y_train=A, kernel=kernel, ard=True, noise_variance=1e-6)
    surrogate = PODGPSurrogate(pod=pod, gp=gp)

    n_current = seed_X.shape[0]
    while n_current < max_total_budget:
        _val_mean, val_var = surrogate.predict(val_grid)
        if float(np.max(val_var.mean(axis=1))) <= gamma_threshold**2:
            break

        b = min(batch_size, max_total_budget - n_current)
        batch_X = []
        batch_Y = []
        for _ in range(b):
            pool = np.asarray(problem.prior.rvs(size=candidate_pool_size, random_state=rng), dtype=float)
            _mean, var = surrogate.predict(pool)
            score = var.mean(axis=1)
            theta_new = pool[int(np.argmax(score))]
            batch_X.append(theta_new)
            batch_Y.append(problem.hf_forward(theta_new))

        X_all.append(np.asarray(batch_X))
        Y_all.append(np.asarray(batch_Y))
        n_current += b

        X_final = np.vstack(X_all)
        Y_final = np.vstack(Y_all)
        pod = POD(rank=pod_rank).fit(Y_final)
        A_final = pod.transform(Y_final)
        gp = MultiOutputGP(X_train=X_final, Y_train=A_final, kernel=kernel, ard=True, noise_variance=1e-6)
        surrogate = PODGPSurrogate(pod=pod, gp=gp)

    return surrogate


def run_pretrained(
    problem: Problem,
    *,
    seed_X: np.ndarray,
    seed_Y: np.ndarray,
    gamma_threshold: float,
    pod_rank: int,
    kernel: str,
    iterations: int,
    seed: int,
    theta0: np.ndarray | None = None,
) -> tuple[MCMCChain, int, PODGPSurrogate]:
    rng = set_seed(seed)
    surrogate = active_learning_offline_design(
        problem, seed_X, seed_Y, gamma_threshold=gamma_threshold, pod_rank=pod_rank, kernel=kernel, rng=rng
    )
    n_hf_spent = surrogate.gp.n_train

    model = ActiveMCMCModel(lf_model=surrogate, hf_model=problem.hf_forward, gamma_threshold=0.0, frozen=True)

    cov = (problem.sigma_obs**2) * np.eye(len(problem.y_obs))
    loglike = ActiveGPLogLike(data=problem.y_obs, covariance=cov)
    posterior = tda.Posterior(problem.prior, loglike, model.coarse)
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
    theta0: np.ndarray | None = None,
) -> tuple[MCMCChain, PODGPSurrogate]:
    model = ActiveMCMCModel(
        lf_model=copy.deepcopy(surrogate), hf_model=problem.hf_forward, gamma_threshold=gamma_threshold
    )

    cov = (problem.sigma_obs**2) * np.eye(len(problem.y_obs))
    loglike = ActiveGPLogLike(data=problem.y_obs, covariance=cov)
    posterior = tda.Posterior(problem.prior, loglike, model.coarse)
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
    return result.chain, model.lf_model


# ---------------------------------------------------------------------------
# Method 4: proposed adaptive DA-MCMC with freeze
# ---------------------------------------------------------------------------


def run_ours(
    problem: Problem,
    *,
    surrogate: PODGPSurrogate,
    gamma_threshold: float,
    n_coarse_evals: int,
    max_adapt_coarse_evals: int,
    seed: int,
    theta0: np.ndarray | None = None,
) -> tuple[MCMCChain, dict[str, Any], PODGPSurrogate]:
    state = AdaptiveSubchainState(subchain_length=10)
    control = AdaptiveSubchainControl(
        update_every=5,
        target_error=problem.sigma_obs,
        min_subchain=1,
        max_subchain=100,
        grow_factor=2.0,
        shrink_factor=0.5,
        patience=5,
    )
    hook = AdaptiveSubchain(state=state, control=control)

    model = ActiveMCMCModel(
        lf_model=copy.deepcopy(surrogate),
        hf_model=problem.hf_forward,
        gamma_threshold=gamma_threshold,
        adaptive=hook,
    )

    cov = (problem.sigma_obs**2) * np.eye(len(problem.y_obs))
    loglike_coarse = ActiveGPLogLike(data=problem.y_obs, covariance=cov)
    loglike_fine = tda.AdaptiveGaussianLogLike(data=problem.y_obs, covariance=cov)

    def posterior_factory(m: ActiveMCMCModel) -> list[tda.Posterior]:
        return [
            tda.Posterior(problem.prior, loglike_coarse, m.coarse),
            tda.Posterior(problem.prior, loglike_fine, m.fine),
        ]

    proposal = make_proposal(problem)
    rng = set_seed(seed)
    if theta0 is None:
        theta0 = np.asarray(problem.prior.rvs(random_state=rng), dtype=float)

    result = sample_adaptive_then_frozen_chain(
        model=model,
        posterior_factory=posterior_factory,
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
    return result.chain, result.metadata, model.lf_model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def effective_burn_in(method: str, *, meta_ours: dict[str, Any] | None = None, default: int = BURN_IN) -> int:
    """Burn-in to use for posterior summaries/plots -- not just the convergence check.

    For `method == "ours"`, the adaptive phase is genuinely non-stationary (the
    surrogate is still being updated online), not just slowly mixing, so it must be
    discarded in full regardless of `default`: use `meta_ours["n_adapt_samples"]` (from
    `run_ours`'s returned metadata) rather than the shared constant. Using the shared
    `BURN_IN` for `ours` silently pools still-adapting, moving-target samples into
    "post-burn-in" posterior estimates and R-hat/ESS alike -- this is what caused
    `ours` to look artificially noisier/worse than plain MH despite mixing better once
    actually frozen. Other methods (which have no such freeze point) use `default`.
    """
    if method == "ours":
        if meta_ours is None:
            raise ValueError("meta_ours is required to compute burn-in for 'ours'.")
        return int(meta_ours["n_adapt_samples"])
    return default


def ours_coarse_eval_units(meta_ours: dict[str, Any]) -> int:
    """The true number of coarse-evaluation units underlying `run_ours`'s returned chain.

    `meta_ours["n_adapt_samples"] + meta_ours["n_production_samples"]` (i.e.
    `chain.n_steps`) is *not* this number: during the adaptive phase, tinyDA's coarse
    chain records one row per coarse evaluation, so that part is already correct. But
    during the frozen production phase, the returned chain is tinyDA's *fine* chain
    (one row per outer DA block, via the `chain_fine_0` fix), and each such row hides an
    entire block of `frozen_subsampling_rate` coarse evaluations that were spent
    reaching it -- those never appear as separate rows anywhere. Using `chain.n_steps`
    as the HF-call-fraction denominator therefore drastically undercounts production's
    true coarse-evaluation cost (by a factor of `frozen_subsampling_rate`), which
    systematically overstates `ours`'s HF-call fraction relative to the other three
    methods, whose chains genuinely are one row per coarse evaluation throughout. This
    reconstructs the correct count from metadata for use as `summarize`'s/
    `pooled_summarize`'s `n_coarse_eval_units` override.
    """
    n_adapt_coarse_evals = int(meta_ours["adapt_metadata"]["coarse_evals_used"])
    if meta_ours["phase"] == "adapt_only":
        return n_adapt_coarse_evals
    production_iterations = int(meta_ours["production_metadata"]["iterations"])
    frozen_rate = int(meta_ours["frozen_subsampling_rate"])
    return n_adapt_coarse_evals + production_iterations * frozen_rate


def ours_cumulative_coarse_evals(meta_ours: dict[str, Any]) -> np.ndarray:
    """Per-row cumulative coarse-evaluation count for `ours`'s full (adaptive +
    production) chain -- the x-axis a plot should use instead of raw row index.

    Raw row index is what every other method's chain uses as its x-axis (correctly:
    their chains are one row per coarse evaluation throughout), but for `ours` that's
    only true during the adaptive phase. During the frozen production phase each row
    is a whole DA block of `frozen_subsampling_rate` coarse evaluations, so plotting
    against row index compresses the entire production phase into a misleadingly short
    stretch of x-axis relative to how much budget it actually consumed -- e.g. on a
    cumulative HF-fraction plot, this makes `ours` look like it stops far short of the
    other methods' x-axis range even when it consumed a comparable coarse-eval budget,
    and makes its curve's slope during production look artificially steep. This is the
    same correction as `ours_coarse_eval_units`, applied per-row instead of as a single
    final total.
    """
    n_adapt = int(meta_ours["n_adapt_samples"])
    adapt_x = np.arange(1, n_adapt + 1, dtype=float)
    if meta_ours["phase"] == "adapt_only":
        return adapt_x
    n_adapt_coarse_evals = int(meta_ours["adapt_metadata"]["coarse_evals_used"])
    frozen_rate = int(meta_ours["frozen_subsampling_rate"])
    n_production = int(meta_ours["n_production_samples"])
    production_x = n_adapt_coarse_evals + frozen_rate * np.arange(1, n_production + 1, dtype=float)
    return np.concatenate([adapt_x, production_x])


def ours_full_resolution_trace(
    chain: MCMCChain, meta_ours: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full-resolution (x, samples, used_hf) trajectory for `ours`, across both phases.

    `chain` (as returned by `run_ours`) is *not* full resolution during production:
    it's built from `chain_fine_0` there, one row per DA block, which discards the many
    cheap intra-block low-fidelity coarse steps in between. This stitches the adaptive
    phase (already full resolution: `chain_coarse_0`, one row per coarse evaluation)
    with the production phase's separately-requested diagnostic chain
    (`meta_ours["production_diagnostic_chain"]`, populated because `run_ours` passes
    `production_diagnostic_chain_key="chain_coarse_0"`) to recover the true
    full-resolution path -- useful for a trace plot colored by `used_hf` that shows the
    actual mixing behaviour (many cheap LF steps between infrequent HF corrections),
    which the posterior-inference chain alone cannot show.

    Returns `x` in the same "coarse-evaluation iteration" units as
    `ours_cumulative_coarse_evals`/`plot_hf_fraction`, so it can share an x-axis with
    the other methods' (already full-resolution) traces.
    """
    n_adapt = int(meta_ours["n_adapt_samples"])
    adapt_samples = chain.samples[:n_adapt]
    adapt_used_hf = (
        chain.extras.used_hf[:n_adapt] if chain.extras.used_hf is not None else np.zeros(n_adapt, dtype=bool)
    )
    adapt_x = np.arange(1, n_adapt + 1, dtype=float)

    diagnostic_chain = meta_ours.get("production_diagnostic_chain")
    if diagnostic_chain is None:
        return adapt_x, adapt_samples, adapt_used_hf

    n_adapt_coarse_evals = int(meta_ours["adapt_metadata"]["coarse_evals_used"])
    prod_samples = diagnostic_chain.samples
    prod_used_hf = (
        diagnostic_chain.extras.used_hf
        if diagnostic_chain.extras.used_hf is not None
        else np.zeros(prod_samples.shape[0], dtype=bool)
    )
    prod_x = n_adapt_coarse_evals + np.arange(1, prod_samples.shape[0] + 1, dtype=float)

    x = np.concatenate([adapt_x, prod_x])
    samples = np.vstack([adapt_samples, prod_samples])
    used_hf = np.concatenate([adapt_used_hf, prod_used_hf])
    return x, samples, used_hf


def ours_multichain_trace(
    adapt_chain: MCMCChain, adapt_meta: dict[str, Any], production_chain: MCMCChain, frozen_rate: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stitched (x, samples, used_hf) trace for one `ours` replicate from the
    convergence-driven (section 5b) machinery -- the multi-chain analog of
    `ours_full_resolution_trace`, built from `run_convergence_driven_comparison`'s
    per-replicate pieces (`chains_by_method["ours_adapt"][i]`, `adapt_metas[i]`,
    `chains_by_method["ours"][i]`, and the frozen `subsampling_rate` recorded on that
    replicate's init state) rather than `run_ours`'s single-call metadata shape.

    Unlike `ours_full_resolution_trace`, the production segment here is at DA-block
    resolution (one row per block), not full coarse-eval resolution: section 5b's
    chunked production runs don't request the intra-block diagnostic chain (doing so
    for every chunk of every replicate would add a lot of extra memory/compute for what
    is meant to be a multi-chain overview plot, not the detailed single-chain one).
    `x` is still in coarse-evaluation-iteration units, so it lines up with the other
    methods' traces and with section 4's single-chain plot.
    """
    n_adapt = adapt_chain.n_steps
    adapt_x = np.arange(1, n_adapt + 1, dtype=float)
    adapt_used_hf = (
        adapt_chain.extras.used_hf if adapt_chain.extras.used_hf is not None else np.zeros(n_adapt, dtype=bool)
    )

    n_adapt_coarse_evals = int(adapt_meta["adapt_coarse_evals_used"])
    n_production = production_chain.n_steps
    production_x = n_adapt_coarse_evals + frozen_rate * np.arange(1, n_production + 1, dtype=float)
    production_used_hf = (
        production_chain.extras.used_hf
        if production_chain.extras.used_hf is not None
        else np.zeros(n_production, dtype=bool)
    )

    x = np.concatenate([adapt_x, production_x])
    samples = np.vstack([adapt_chain.samples, production_chain.samples])
    used_hf = np.concatenate([adapt_used_hf, production_used_hf])
    return x, samples, used_hf


# ---------------------------------------------------------------------------
# Distributional posterior-quality metrics (in addition to RMSE-to-truth)
# ---------------------------------------------------------------------------
#
# RMSE-to-truth only checks the posterior *mean* against a single point (theta_true);
# it says nothing about whether the spread/shape of the posterior is right, and it's
# meaningless as a way to compare a method against the actual quantity DA-exactness
# promises to preserve: the *distribution* of the HF-only posterior, not just its mean.
# These two metrics compare each method's pooled posterior against a reference
# distribution (in practice, the R-hat-validated HF-only posterior) under a Gaussian
# approximation of both -- exact for the closed-form formulas below, and a reasonable
# approximation here since this 2D posterior is close to elliptical.


def gaussian_wasserstein2(
    mean_a: np.ndarray, cov_a: np.ndarray, mean_b: np.ndarray, cov_b: np.ndarray
) -> float:
    """Closed-form 2-Wasserstein distance between N(mean_a, cov_a) and N(mean_b, cov_b).

    W2^2 = ||mean_a - mean_b||^2 + tr(cov_a + cov_b - 2*(cov_b^{1/2} cov_a cov_b^{1/2})^{1/2}).

    Symmetric and a true metric (unlike KL): captures both a location mismatch (the
    mean term) and a shape/spread mismatch (the trace term), in the same units as the
    parameters themselves (e.g. a W2 of 0.5 in `k` roughly means "the posteriors are
    about as different as if one were shifted by ~0.5").
    """
    from scipy.linalg import sqrtm

    mean_a = np.asarray(mean_a, dtype=float)
    mean_b = np.asarray(mean_b, dtype=float)
    cov_a = np.asarray(cov_a, dtype=float)
    cov_b = np.asarray(cov_b, dtype=float)

    diff = mean_a - mean_b
    cov_b_sqrt = sqrtm(cov_b)
    cross_term = sqrtm(cov_b_sqrt @ cov_a @ cov_b_sqrt)
    cross_term = np.real(cross_term)  # discard negligible complex parts from numerical sqrtm

    w2_sq = float(diff @ diff + np.trace(cov_a + cov_b - 2.0 * cross_term))
    return float(np.sqrt(max(w2_sq, 0.0)))


def gaussian_kl(mean_a: np.ndarray, cov_a: np.ndarray, mean_b: np.ndarray, cov_b: np.ndarray) -> float:
    """Closed-form KL(N(mean_a, cov_a) || N(mean_b, cov_b)).

    Asymmetric: KL(method || HF-only) penalizes the method's posterior for being
    confidently wrong (placing mass where HF-only doesn't) more than for being overly
    diffuse. In particular it blows up if the method's posterior is much narrower than
    HF-only's in some direction while also being off-center -- a failure mode RMSE and
    even W2 can under-report, since a narrow-but-wrong posterior is a worse UQ failure
    than a wide-but-centered one.
    """
    mean_a = np.asarray(mean_a, dtype=float)
    mean_b = np.asarray(mean_b, dtype=float)
    cov_a = np.asarray(cov_a, dtype=float)
    cov_b = np.asarray(cov_b, dtype=float)

    k = mean_a.shape[0]
    cov_b_inv = np.linalg.inv(cov_b)
    diff = mean_b - mean_a

    term_trace = float(np.trace(cov_b_inv @ cov_a))
    term_quad = float(diff @ cov_b_inv @ diff)
    sign_a, logdet_a = np.linalg.slogdet(cov_a)
    sign_b, logdet_b = np.linalg.slogdet(cov_b)
    term_logdet = float(logdet_b - logdet_a)

    return 0.5 * (term_trace + term_quad - k + term_logdet)


def summarize(
    chain: MCMCChain,
    problem: Problem,
    *,
    burn_in: int,
    reference_mean: np.ndarray | None = None,
    n_offline_hf: int = 0,
    n_coarse_eval_units: int | None = None,
) -> dict[str, Any]:
    """Summarize a chain's accuracy and total HF cost.

    `n_offline_hf` is the number of HF evaluations spent *before* this chain started
    (e.g. the shared `n_init` offline seed for online methods, or the full acquired
    training-set size for the pretrained baseline). Without it, `n_hf_calls` only
    counts HF calls made *during* sampling and understates the pretrained baseline's
    true cost, which is paid entirely upfront and invisible in `chain.extras.used_hf`.

    `n_coarse_eval_units` overrides the HF-call-fraction denominator's step count
    (`chain.n_steps`) for methods where a chain row isn't one coarse evaluation --
    currently only `ours`, via `ours_coarse_eval_units(meta_ours)`. Every other method's
    chain genuinely is one row per coarse evaluation, so the default (`None`, meaning
    "use `chain.n_steps`") is correct for them.
    """
    post = chain.burn_in(burn_in)
    n_hf_online = int(np.sum(chain.extras.used_hf)) if chain.extras.used_hf is not None else 0
    n_hf_total = n_offline_hf + n_hf_online
    steps_for_budget = chain.n_steps if n_coarse_eval_units is None else n_coarse_eval_units
    total_budget = n_offline_hf + steps_for_budget

    out: dict[str, Any] = {
        "n_steps": chain.n_steps,
        "posterior_mean": post.samples.mean(axis=0).tolist(),
        "posterior_std": post.samples.std(axis=0).tolist(),
        "rmse_to_truth": posterior_rmse(chain.samples, problem.theta_true, burn_in=burn_in),
        "n_offline_hf": n_offline_hf,
        "n_hf_calls_online": n_hf_online,
        "n_hf_calls": n_hf_total,
        "hf_call_fraction": (n_hf_total / total_budget) if total_budget > 0 else float("nan"),
    }
    if reference_mean is not None:
        out["mean_abs_dev_from_reference"] = np.abs(post.samples.mean(axis=0) - reference_mean).tolist()
    return out


def multichain_diagnostics(
    chains: list[MCMCChain], *, burn_in: int, param_names: tuple[str, ...] = PARAM_NAMES
) -> dict[str, Any]:
    """Gelman-Rubin R-hat and effective sample size across independent chains.

    All chains should share the same starting point `theta0` (only the sampling/active-
    learning randomness differs) so that R-hat measures within-vs-between-chain mixing
    rather than sensitivity to the initial point. Chains may have slightly different
    post-burn-in lengths (e.g. `ours`, whose adaptive subchain length varies run to
    run); they are truncated to the shortest chain's length since arviz requires equal
    draw counts across chains.
    """
    import arviz as az

    posts = [c.burn_in(burn_in).samples for c in chains]
    min_len = min(p.shape[0] for p in posts)
    if min_len < 2:
        raise ValueError(f"Need at least 2 post-burn-in draws per chain; got min_len={min_len}.")
    stacked = np.stack([p[:min_len] for p in posts], axis=0)  # (n_chains, n_draws, n_dim)

    idata = az.from_dict(posterior={name: stacked[:, :, i] for i, name in enumerate(param_names)})
    rhat = az.rhat(idata)
    ess = az.ess(idata)

    return {
        "n_chains": len(chains),
        "n_draws_per_chain": min_len,
        "rhat": {name: float(rhat[name].values) for name in param_names},
        "ess_bulk": {name: float(ess[name].values) for name in param_names},
    }


def find_burn_in_via_rhat(
    chains: list[MCMCChain],
    *,
    candidate_burn_ins: list[int] | None = None,
    rhat_threshold: float = 1.01,
    min_burn_in: int = 0,
    patience: int = 5,
    param_names: tuple[str, ...] = PARAM_NAMES,
) -> dict[str, Any]:
    """Find the smallest burn-in at which Gelman-Rubin R-hat (across `chains`) drops to
    `rhat_threshold` and stays there for `patience` consecutive larger candidates (not
    just the first dip below threshold, which could be a fluke).

    Note this deliberately does *not* require staying below threshold forever after:
    with only a handful of chains, R-hat itself is a noisy estimate (especially late in
    the sweep, where the remaining segment is short), so demanding zero violations all
    the way to the end is fragile -- a single late noisy blip would otherwise push the
    chosen burn-in out to nearly the full chain length even when convergence was
    reached much earlier. `patience` consecutive passes is a good-enough robustness
    check without that failure mode (mirrors the streak/patience criterion used
    elsewhere in this codebase, e.g. `AdaptiveSubchainControl.patience`).

    The returned `burn_in` doubles as the answer to "how many iterations does this
    method need before its chains have converged to a common distribution" -- smaller
    is more efficient, since fewer samples are wasted before the chain can be trusted.

    `min_burn_in` excludes candidates below a hard floor. For the proposed method
    (`ours`), this should be set to (at least) `n_adapt_samples`: during the adaptive
    phase the *target itself* is non-stationary (the surrogate is still being updated),
    so no amount of burn-in trimmed from within that phase can fix it -- the whole
    phase must be discarded regardless of what R-hat says about it.

    Returns a dict with `burn_in`, `rhat`, `ess_bulk`, `converged` (whether any
    candidate satisfied the threshold and held), and `sweep` (the full list of tried
    `{burn_in, rhat, ess_bulk, max_rhat}`, useful for plotting R-hat vs. assumed
    burn-in).
    """
    n_total = min(c.n_steps for c in chains)
    if candidate_burn_ins is None:
        step = max(1, n_total // 50)
        candidate_burn_ins = list(range(0, n_total - 10, step))

    candidates = sorted(b for b in set(candidate_burn_ins) if b >= min_burn_in and n_total - b >= 10)

    sweep: list[dict[str, Any]] = []
    for b in candidates:
        try:
            diag = multichain_diagnostics(chains, burn_in=b, param_names=param_names)
        except ValueError:
            continue
        sweep.append(
            {
                "burn_in": b,
                "rhat": diag["rhat"],
                "ess_bulk": diag["ess_bulk"],
                "max_rhat": max(diag["rhat"].values()),
            }
        )

    chosen = None
    for i, entry in enumerate(sweep):
        window = sweep[i : i + patience]
        if len(window) == patience and all(e["max_rhat"] <= rhat_threshold for e in window):
            chosen = entry
            break
    if chosen is None and sweep:
        # Never held converged for `patience` in a row within the tried range; report
        # the last (longest burn-in) point tried so the caller at least sees how close
        # it got.
        chosen = sweep[-1]

    return {
        "burn_in": chosen["burn_in"] if chosen else None,
        "rhat": chosen["rhat"] if chosen else None,
        "ess_bulk": chosen["ess_bulk"] if chosen else None,
        "converged": bool(chosen and chosen["max_rhat"] <= rhat_threshold),
        "sweep": sweep,
    }


def pooled_summarize(
    chains: list[MCMCChain],
    problem: Problem,
    *,
    burn_in: int,
    reference_mean: np.ndarray | None = None,
    reference_cov: np.ndarray | None = None,
    n_offline_hf: int | list[int] = 0,
    n_coarse_eval_units: int | list[int] | None = None,
) -> dict[str, Any]:
    """Like `summarize`, but pools post-burn-in samples across several chains (e.g. the
    replicate chains from a convergence check) into one richer posterior estimate,
    using a single validated `burn_in` (typically from `find_burn_in_via_rhat`) shared
    by all of them.

    `n_offline_hf` may be a single int (same offline cost for every chain, e.g. the
    shared `n_init` seed for the online methods) or a list with one entry per chain
    (e.g. `pretrained`, where each replicate trained its own offline design and so may
    have spent a different amount of HF budget getting there).

    `n_coarse_eval_units` is the pooled analog of `summarize`'s parameter of the same
    name: an override for each chain's step count in the HF-call-fraction denominator,
    needed for `ours` (see `ours_coarse_eval_units`). Same int-or-per-chain-list shape
    as `n_offline_hf`; `None` (the default, correct for the other three methods) uses
    each chain's own `n_steps`.

    `reference_cov`, given alongside `reference_mean`, additionally computes
    `wasserstein2_to_reference` and `kl_to_reference`: distributional distances (under
    a Gaussian approximation of both posteriors) to a reference posterior -- in
    practice, the R-hat-validated HF-only posterior (see `run_convergence_check`). This
    is the metric DA-exactness actually promises to control for `ours` (equality in
    distribution to the HF posterior), unlike `rmse_to_truth`, which only checks the
    mean against a single point and says nothing about whether the spread/shape is
    right.
    """
    if isinstance(n_offline_hf, int):
        n_offline_hf_list = [n_offline_hf] * len(chains)
    else:
        n_offline_hf_list = list(n_offline_hf)
    if len(n_offline_hf_list) != len(chains):
        raise ValueError("n_offline_hf list must have one entry per chain.")

    if n_coarse_eval_units is None:
        steps_list = [c.n_steps for c in chains]
    elif isinstance(n_coarse_eval_units, int):
        steps_list = [n_coarse_eval_units] * len(chains)
    else:
        steps_list = list(n_coarse_eval_units)
    if len(steps_list) != len(chains):
        raise ValueError("n_coarse_eval_units list must have one entry per chain.")

    posts = [c.burn_in(burn_in).samples for c in chains]
    pooled = np.concatenate(posts, axis=0)

    n_hf_online = sum(
        int(np.sum(c.extras.used_hf)) if c.extras.used_hf is not None else 0 for c in chains
    )
    n_hf_total = sum(n_offline_hf_list) + n_hf_online
    total_budget = sum(n_offline_hf_list) + sum(steps_list)

    posterior_mean = pooled.mean(axis=0)
    posterior_cov = np.cov(pooled, rowvar=False)
    out: dict[str, Any] = {
        "n_chains": len(chains),
        "n_pooled_samples": int(pooled.shape[0]),
        "posterior_mean": posterior_mean.tolist(),
        "posterior_std": pooled.std(axis=0).tolist(),
        "posterior_cov": posterior_cov.tolist(),
        "rmse_to_truth": float(np.mean(np.sqrt(np.sum((pooled - problem.theta_true[None, :]) ** 2, axis=1)))),
        "n_offline_hf": n_offline_hf_list,
        "n_hf_calls_online": n_hf_online,
        "n_hf_calls": n_hf_total,
        "hf_call_fraction": (n_hf_total / total_budget) if total_budget > 0 else float("nan"),
    }
    if reference_mean is not None:
        out["mean_abs_dev_from_reference"] = np.abs(posterior_mean - reference_mean).tolist()
    if reference_mean is not None and reference_cov is not None:
        out["wasserstein2_to_reference"] = gaussian_wasserstein2(
            posterior_mean, posterior_cov, reference_mean, reference_cov
        )
        out["kl_to_reference"] = gaussian_kl(posterior_mean, posterior_cov, reference_mean, reference_cov)
    return out


def run_convergence_check(
    problem: Problem,
    *,
    n_init: int,
    pod_rank: int,
    seed_X: np.ndarray,
    seed_Y: np.ndarray,
    seed_surrogate: PODGPSurrogate,
    n_chains: int,
    n_coarse_evals: int,
    rhat_threshold: float = 1.01,
    theta0: np.ndarray | None = None,
    seed_base: int = 0,
) -> tuple[dict[str, Any], dict[str, list[MCMCChain]], dict[str, list[PODGPSurrogate]], list[dict[str, Any]]]:
    """Run `n_chains` independent chains per method from a shared `theta0`, find each
    method's burn-in via `find_burn_in_via_rhat`, and pool the post-burn-in samples
    into a validated posterior summary via `pooled_summarize`.

    This is the single shared implementation used by both `check_convergence.py` and
    `msd_benchmark.ipynb`, so the two don't drift out of sync the way the notebook's
    posterior-summary burn-in once drifted from the convergence check's.

    Returns `(results, chains_by_method, surrogates_by_method, ours_metas)`:

    - `results`: JSON-serializable, keyed by method name ("hf_only", "pretrained",
      "online_active", "ours"), each entry the `find_burn_in_via_rhat` result plus a
      `"pooled"` key (the `pooled_summarize` output using that method's own validated
      burn-in).
    - `chains_by_method`: the raw `MCMCChain` lists (not JSON-serializable, kept
      separate), for callers that want to plot the pooled samples rather than just
      tabulate summary statistics.
    - `surrogates_by_method`: each replicate's trained `PODGPSurrogate`, keyed
      `"pretrained"`/`"online_active"`/`"ours"` only (`hf_only` has none), for
      "surrogate before/after" plots across all replicates.
    - `ours_metas`: each `ours` replicate's full `run_ours`-style metadata dict (not
      JSON-serializable -- it embeds a `production_diagnostic_chain` `MCMCChain` --
      kept separate for the same reason `chains_by_method` is). Lets callers build a
      full-resolution multi-chain trace for `ours` via `ours_full_resolution_trace`
      per `(chains_by_method["ours"][i], ours_metas[i])` pair, the same way section 4's
      single-chain trace plot does.
    """
    if theta0 is None:
        theta0 = np.asarray(problem.prior.rvs(random_state=set_seed(999 + n_init + seed_base)), dtype=float)

    results: dict[str, Any] = {}

    hf_chains = [
        run_hf_only(problem, iterations=n_coarse_evals, seed=seed_base + 100 + i, theta0=theta0)
        for i in range(n_chains)
    ]
    hf_conv = find_burn_in_via_rhat(hf_chains, rhat_threshold=rhat_threshold)
    hf_conv["pooled"] = pooled_summarize(hf_chains, problem, burn_in=hf_conv["burn_in"] or 0, n_offline_hf=0)
    results["hf_only"] = hf_conv

    # HF-only's own pooled posterior is the reference every other method is compared
    # against for the distributional metrics (wasserstein2_to_reference/kl_to_reference)
    # -- it's the R-hat-validated stand-in for "the true HF posterior" that DA-exactness
    # promises `ours` will match in distribution, not just in mean.
    hf_reference_mean = np.asarray(hf_conv["pooled"]["posterior_mean"])
    hf_reference_cov = np.asarray(hf_conv["pooled"]["posterior_cov"])

    pretrained_chains = []
    n_hf_pretrained_list = []
    pretrained_surrogates = []
    for i in range(n_chains):
        c, n_hf, s = run_pretrained(
            problem, seed_X=seed_X, seed_Y=seed_Y, gamma_threshold=GAMMA_THRESHOLD, pod_rank=pod_rank,
            kernel=KERNEL, iterations=n_coarse_evals, seed=seed_base + 200 + i, theta0=theta0,
        )
        pretrained_chains.append(c)
        n_hf_pretrained_list.append(n_hf)
        pretrained_surrogates.append(s)
    pre_conv = find_burn_in_via_rhat(pretrained_chains, rhat_threshold=rhat_threshold)
    pre_conv["pooled"] = pooled_summarize(
        pretrained_chains, problem, burn_in=pre_conv["burn_in"] or 0, n_offline_hf=n_hf_pretrained_list,
        reference_mean=hf_reference_mean, reference_cov=hf_reference_cov,
    )
    results["pretrained"] = pre_conv

    online_chains = []
    online_surrogates = []
    for i in range(n_chains):
        c, s = run_online_active(
            problem, surrogate=seed_surrogate, gamma_threshold=GAMMA_THRESHOLD, iterations=n_coarse_evals,
            seed=seed_base + 300 + i, theta0=theta0,
        )
        online_chains.append(c)
        online_surrogates.append(s)
    on_conv = find_burn_in_via_rhat(online_chains, rhat_threshold=rhat_threshold)
    on_conv["pooled"] = pooled_summarize(
        online_chains, problem, burn_in=on_conv["burn_in"] or 0, n_offline_hf=n_init,
        reference_mean=hf_reference_mean, reference_cov=hf_reference_cov,
    )
    results["online_active"] = on_conv

    ours_chains = []
    ours_metas = []
    ours_surrogates = []
    n_adapt_samples_per_chain = []
    for i in range(n_chains):
        c, meta, s = run_ours(
            problem, surrogate=seed_surrogate, gamma_threshold=GAMMA_THRESHOLD, n_coarse_evals=n_coarse_evals,
            max_adapt_coarse_evals=MAX_ADAPT_COARSE_EVALS, seed=seed_base + 400 + i, theta0=theta0,
        )
        ours_chains.append(c)
        ours_metas.append(meta)
        ours_surrogates.append(s)
        n_adapt_samples_per_chain.append(int(meta["n_adapt_samples"]))
    ours_min_burn_in = max(n_adapt_samples_per_chain)
    ours_conv = find_burn_in_via_rhat(ours_chains, rhat_threshold=rhat_threshold, min_burn_in=ours_min_burn_in)
    ours_conv["pooled"] = pooled_summarize(
        ours_chains, problem, burn_in=ours_conv["burn_in"] or 0, n_offline_hf=n_init,
        n_coarse_eval_units=[ours_coarse_eval_units(m) for m in ours_metas],
        reference_mean=hf_reference_mean, reference_cov=hf_reference_cov,
    )
    ours_conv["n_adapt_samples_per_chain"] = n_adapt_samples_per_chain
    results["ours"] = ours_conv

    chains_by_method = {
        "hf_only": hf_chains,
        "pretrained": pretrained_chains,
        "online_active": online_chains,
        "ours": ours_chains,
    }
    surrogates_by_method = {
        "pretrained": pretrained_surrogates,
        "online_active": online_surrogates,
        "ours": ours_surrogates,
    }
    return results, chains_by_method, surrogates_by_method, ours_metas


def print_convergence_table(results: dict[str, Any]) -> None:
    """Prints RMSE-to-truth alongside the two reference-based distributional metrics
    (`wasserstein2_to_reference`, `kl_to_reference` -- both vs. the HF-only pooled
    posterior; absent for `hf_only` itself, shown as `--`). See `pooled_summarize`'s
    docstring for why RMSE alone is not an adequate posterior-quality metric.
    """
    header = (
        f"{'method':<14}{'burn_in':>9}{'rhat_k':>9}{'rhat_c':>9}{'ess_k':>9}{'ess_c':>9}"
        f"{'rmse':>10}{'w2_ref':>10}{'kl_ref':>10}"
    )
    print(header)
    for name, r in results.items():
        if r["rhat"] is None:
            print(f"{name:<14}{'--':>9}")
            continue
        pooled = r.get("pooled", {})
        rmse = pooled.get("rmse_to_truth", float("nan"))
        w2 = pooled.get("wasserstein2_to_reference")
        kl = pooled.get("kl_to_reference")
        w2_str = f"{w2:>10.4f}" if w2 is not None else f"{'--':>10}"
        kl_str = f"{kl:>10.4f}" if kl is not None else f"{'--':>10}"
        print(
            f"{name:<14}{r['burn_in']:>9}{r['rhat']['k']:>9.3f}{r['rhat']['c']:>9.3f}"
            f"{r['ess_bulk']['k']:>9.1f}{r['ess_bulk']['c']:>9.1f}{rmse:>10.4f}{w2_str}{kl_str}"
        )


def print_convergence_driven_table(results: dict[str, Any]) -> None:
    """Cost-to-converge table: for each method, how much budget (coarse evals, HF
    calls) it needed to reach R-hat<=threshold (or "capped" if it never did), and the
    resulting accuracy -- the primary comparison for the paper's efficiency claim (vs.
    `print_convergence_table`'s fixed-budget comparison from `run_convergence_check`).
    """
    header = (
        f"{'method':<14}{'status':>10}{'rounds':>8}{'coarse_evals':>14}"
        f"{'n_hf':>8}{'rmse':>10}{'w2_ref':>10}{'kl_ref':>10}"
    )
    print(header)
    for name, r in results.items():
        pooled = r.get("pooled", {})
        rmse = pooled.get("rmse_to_truth", float("nan"))
        w2 = pooled.get("wasserstein2_to_reference")
        kl = pooled.get("kl_to_reference")
        w2_str = f"{w2:>10.4f}" if w2 is not None else f"{'--':>10}"
        kl_str = f"{kl:>10.4f}" if kl is not None else f"{'--':>10}"
        status = "converged" if r["converged"] else "capped"
        print(
            f"{name:<14}{status:>10}{r['rounds_run']:>8}{r['total_coarse_evals']:>14}"
            f"{pooled.get('n_hf_calls', 0):>8}{rmse:>10.4f}{w2_str}{kl_str}"
        )


# ---------------------------------------------------------------------------
# Convergence-driven comparison: run each method until R-hat converges (or a
# safety cap is hit), rather than at a fixed, arbitrarily-chosen budget.
# ---------------------------------------------------------------------------
#
# `run_convergence_check` above answers "given the same N_coarse_evals budget, how
# accurate is each method" -- but N_coarse_evals isn't the resource the paper is
# actually claiming to save (HF calls are), and it silently gives `ours` far fewer
# posterior draws than the other methods for the same nominal budget (each of its
# production rows is a whole DA block, not one coarse evaluation). The functions below
# instead answer "how much does each method cost to reach a fixed *quality* standard
# (R-hat<=1.01 across independent replicates)" -- the comparison that actually
# demonstrates a computational-efficiency claim, and the one a reviewer will look for.
#
# Implementation: `n_chains` replicate chains are advanced in synchronized rounds (one
# chunk of `chunk_size` coarse-evaluation units each, run in parallel across processes
# via `joblib`), with R-hat re-checked (reusing `find_burn_in_via_rhat` unmodified,
# since it already works correctly on a growing chain) after every round. This is a
# lock-step "map (parallel chunk) -> reduce (check R-hat) -> decide" loop rather than
# an asynchronous monitor-thread design: `tinyDA.sample()` isn't preemptible mid-call,
# so a monitor thread couldn't stop a chain any sooner than the next chunk boundary
# anyway -- the same granularity this lock-step loop already gives for free, without
# the added complexity of cross-process signaling.


@dataclass
class _ChunkState:
    """Resumable per-replicate state for one method's chunked, round-based MCMC run.

    The same shape covers all of: HF-only (`model=None`, no surrogate involved),
    pretrained/online_active (single posterior), and `ours`'s production phase
    (two-posterior DA, frozen model, fixed `subsampling_rate`) -- only the objects
    inside differ, not the chunking mechanics.
    """

    posterior: Any  # tda.Posterior (single) or list[tda.Posterior] (two-level DA)
    proposal: Any
    theta_current: np.ndarray
    model: Any | None = None  # ActiveMCMCModel; None for pure HF-only
    subsampling_rate: int = 1
    chain_key: str = "chain_0"


def _reset_model_log(model: Any | None) -> None:
    """Clear `model.log.used_hf` in place before a chunk runs.

    Necessary correctness fix for chunking (not needed by any single-shot `run_*`
    function, which only ever calls `sample_active_chain` once per model): `model.log`
    is a persistent, mutable object that keeps accumulating across every `coarse()`/
    `fine()` call made through it. `sample_active_chain`'s `used_hf` reconstruction
    dispatches on whether the log's length matches *this call's* sample count -- if the
    log weren't reset between chunks, chunk 2 onward would compare a cumulative
    (chunk-1 + chunk-2) log length against chunk 2's own (much shorter) sample count,
    which would almost always mismatch and silently fall back to "every sample used
    HF" -- correct for `ours`'s fine-chain case (where that fallback is actually always
    correct, chunked or not), but *wrong* for the single-posterior methods
    (pretrained/online_active), where a genuine per-step used_hf mix would be
    incorrectly reported as all-True after the first chunk.
    """
    if model is not None and hasattr(model, "log"):
        model.log.used_hf.clear()


def _run_chunk(state: _ChunkState, chunk_size: int) -> tuple[_ChunkState, MCMCChain, int]:
    """Advance one replicate chain by one chunk of `chunk_size` coarse-evaluation units.

    `chunk_size` is interpreted uniformly in coarse-evaluation units across all
    methods, matching the currency used everywhere else in this module: for
    `subsampling_rate=1` (HF-only, pretrained, online_active) that's `chunk_size`
    outer iterations; for `ours`'s production phase (`subsampling_rate=frozen_rate`)
    that's `chunk_size // frozen_rate` DA blocks (each spending `frozen_rate` coarse
    evaluations), so every method advances the same nominal budget per round.

    Returns `(new_state, chain_block, coarse_evals_spent)`. `coarse_evals_spent` is
    `iterations * state.subsampling_rate`, the *true* coarse-eval cost of this chunk --
    NOT the same as `chain_block.n_steps` whenever `subsampling_rate > 1` (`ours`'s
    production phase): each of its rows is a whole DA block, not one coarse eval, and
    when `subsampling_rate > chunk_size` the `max(1, ...)` floor below means a single
    chunk can cost far more than the nominal `chunk_size` target. Callers driving a
    budget cap must track this returned cost, not row counts, or a method with a large
    frozen subsampling rate can silently blow through the intended budget by a wide
    margin while still reporting a small, row-count-based "total".
    """
    iterations = max(1, chunk_size // state.subsampling_rate)
    coarse_evals_spent = iterations * state.subsampling_rate

    if state.model is None:
        # Pure HF-only: no ActiveMCMCModel, call tinyDA directly (mirrors run_hf_only;
        # note it does *not* deep-copy the proposal, unlike sample_active_chain --
        # harmless either way given AdaptiveMetropolisShared's shared state, but kept
        # consistent with the original for a like-for-like comparison).
        chain_obj = tda.sample(
            posteriors=state.posterior,
            proposal=state.proposal,
            iterations=int(iterations),
            n_chains=1,
            force_sequential=True,
            initial_parameters=state.theta_current,
            adaptive_error_model=None,
        )
        samples = extract_samples(chain=chain_obj, chain_key=state.chain_key)
        used_hf = np.ones(samples.shape[0], dtype=bool)
        chain = MCMCChain.from_arrays(samples=samples, used_hf=used_hf)
    else:
        _reset_model_log(state.model)
        result = sample_active_chain(
            model=state.model,
            posterior=state.posterior,
            proposal=state.proposal,
            iterations=int(iterations),
            initial_parameters=state.theta_current,
            subsampling_rate=state.subsampling_rate,
            chain_key=state.chain_key,
        )
        chain = result.chain

    new_state = replace(state, theta_current=chain.samples[-1])
    return new_state, chain, coarse_evals_spent


def _init_hf_only_state(problem: Problem, *, seed: int, theta0: np.ndarray | None = None) -> _ChunkState:
    cov = (problem.sigma_obs**2) * np.eye(len(problem.y_obs))
    loglike = tda.AdaptiveGaussianLogLike(data=problem.y_obs, covariance=cov)
    posterior = tda.Posterior(problem.prior, loglike, problem.hf_forward)
    proposal = make_proposal(problem)
    rng = set_seed(seed)
    if theta0 is None:
        theta0 = np.asarray(problem.prior.rvs(random_state=rng), dtype=float)
    return _ChunkState(posterior=posterior, proposal=proposal, theta_current=theta0, model=None)


def _init_pretrained_state(
    problem: Problem,
    *,
    seed_X: np.ndarray,
    seed_Y: np.ndarray,
    gamma_threshold: float,
    pod_rank: int,
    kernel: str,
    seed: int,
    theta0: np.ndarray | None = None,
) -> tuple[_ChunkState, int]:
    """Trains the offline design once (a fixed upfront cost, not part of the monitored
    round loop), then returns a chunk-ready state for the frozen, single-posterior
    sampling phase, plus the number of HF evaluations spent on that offline design."""
    rng = set_seed(seed)
    surrogate = active_learning_offline_design(
        problem, seed_X, seed_Y, gamma_threshold=gamma_threshold, pod_rank=pod_rank, kernel=kernel, rng=rng
    )
    n_hf_spent = surrogate.gp.n_train

    model = ActiveMCMCModel(lf_model=surrogate, hf_model=problem.hf_forward, gamma_threshold=0.0, frozen=True)
    cov = (problem.sigma_obs**2) * np.eye(len(problem.y_obs))
    loglike = ActiveGPLogLike(data=problem.y_obs, covariance=cov)
    posterior = tda.Posterior(problem.prior, loglike, model.coarse)
    proposal = make_proposal(problem)

    if theta0 is None:
        theta0 = np.asarray(problem.prior.rvs(random_state=rng), dtype=float)
    state = _ChunkState(posterior=posterior, proposal=proposal, theta_current=theta0, model=model)
    return state, n_hf_spent


def _init_online_active_state(
    problem: Problem,
    *,
    surrogate: PODGPSurrogate,
    gamma_threshold: float,
    seed: int,
    theta0: np.ndarray | None = None,
) -> _ChunkState:
    model = ActiveMCMCModel(
        lf_model=copy.deepcopy(surrogate), hf_model=problem.hf_forward, gamma_threshold=gamma_threshold
    )
    cov = (problem.sigma_obs**2) * np.eye(len(problem.y_obs))
    loglike = ActiveGPLogLike(data=problem.y_obs, covariance=cov)
    posterior = tda.Posterior(problem.prior, loglike, model.coarse)
    proposal = make_proposal(problem)
    rng = set_seed(seed)
    if theta0 is None:
        theta0 = np.asarray(problem.prior.rvs(random_state=rng), dtype=float)
    return _ChunkState(posterior=posterior, proposal=proposal, theta_current=theta0, model=model)


def _init_ours_production_state(
    problem: Problem,
    *,
    surrogate: PODGPSurrogate,
    gamma_threshold: float,
    max_adapt_coarse_evals: int,
    seed: int,
    theta0: np.ndarray | None = None,
) -> tuple[_ChunkState, dict[str, Any], MCMCChain]:
    """Runs `ours`'s adaptive phase to its own completion first (unmonitored by
    cross-chain R-hat -- the target is non-stationary during adaptation, so R-hat is
    meaningless there, and each replicate's adaptive phase should stop on its own
    `has_converged()` criterion regardless of what its siblings are doing), then
    freezes and returns a chunk-ready state for the same round-based, R-hat-monitored
    production loop used by the other three methods.

    Returns `(state, adapt_meta, adapt_chain)`: `adapt_meta` carries the adaptive-phase
    bookkeeping needed for cost accounting (`n_adapt_samples`,
    `adapt_coarse_evals_used`), and `adapt_chain` is kept so callers can prepend it to
    the production trace for a full-resolution plot, mirroring
    `ours_full_resolution_trace`'s adaptive-phase segment.
    """
    state_hook = AdaptiveSubchainState(subchain_length=10)
    control = AdaptiveSubchainControl(
        update_every=5,
        target_error=problem.sigma_obs,
        min_subchain=1,
        max_subchain=100,
        grow_factor=2.0,
        shrink_factor=0.5,
        patience=5,
    )
    hook = AdaptiveSubchain(state=state_hook, control=control)

    model = ActiveMCMCModel(
        lf_model=copy.deepcopy(surrogate), hf_model=problem.hf_forward, gamma_threshold=gamma_threshold, adaptive=hook
    )
    cov = (problem.sigma_obs**2) * np.eye(len(problem.y_obs))
    loglike_coarse = ActiveGPLogLike(data=problem.y_obs, covariance=cov)
    loglike_fine = tda.AdaptiveGaussianLogLike(data=problem.y_obs, covariance=cov)

    def posterior_factory(m: ActiveMCMCModel) -> list[tda.Posterior]:
        return [
            tda.Posterior(problem.prior, loglike_coarse, m.coarse),
            tda.Posterior(problem.prior, loglike_fine, m.fine),
        ]

    proposal = make_proposal(problem)
    rng = set_seed(seed)
    if theta0 is None:
        theta0 = np.asarray(problem.prior.rvs(random_state=rng), dtype=float)

    adapt_result = sample_adaptive_active_chain(
        model=model,
        posterior=posterior_factory(model),
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
        posterior=posterior_factory(frozen_model),
        proposal=proposal,
        theta_current=theta_last,
        model=frozen_model,
        subsampling_rate=frozen_rate,
        chain_key="chain_fine_0",
    )
    return state, adapt_meta, adapt_result.chain


def _concat_chain_blocks(blocks: list[MCMCChain]) -> MCMCChain:
    """Concatenate a replicate chain's accumulated per-round blocks into one chain.

    Each block's row 0 is tinyDA's echo of the `initial_parameters` it was given --
    not a new MCMC draw -- so for every block after the first, that row is *exactly*
    the previous block's last row (verified empirically: `chain.samples[0]` is
    bit-identical to whatever `initial_parameters` was passed in). Naively vstacking
    every block would duplicate one sample at every chunk boundary: a spurious
    zero-movement transition that both slightly overweights that theta value in the
    pooled posterior and understates the chain's true mixing (autocorrelation/ESS
    estimates get a fake "stuck" step injected once per chunk). Dropping row 0 from
    every block but the first removes the duplicate while keeping every genuine draw.
    """
    sample_parts = [blocks[0].samples] + [b.samples[1:] for b in blocks[1:]]
    samples = np.vstack(sample_parts)

    def _used_hf(b: MCMCChain) -> np.ndarray:
        return b.extras.used_hf if b.extras.used_hf is not None else np.zeros(b.n_steps, dtype=bool)

    used_hf_parts = [_used_hf(blocks[0])] + [_used_hf(b)[1:] for b in blocks[1:]]
    used_hf = np.concatenate(used_hf_parts)
    return MCMCChain.from_arrays(samples=samples, used_hf=used_hf)


def run_until_rhat_converged(
    init_fns: list[Callable[[], _ChunkState]],
    *,
    chunk_size: int,
    max_total_coarse_evals: int,
    rhat_threshold: float = 1.01,
    patience: int = 5,
    n_jobs: int | None = None,
) -> dict[str, Any]:
    """Run `len(init_fns)` independent replicate chains in synchronized rounds, each
    round advancing every chain by one `chunk_size`-coarse-eval chunk in parallel
    (`joblib`, one process per chain), stopping as soon as `find_burn_in_via_rhat`
    reports convergence across the accumulated chains-so-far (reused unmodified: it
    already does the right thing when called again on a longer chain) or
    `max_total_coarse_evals` is reached, whichever comes first.

    Returns a dict with `chains` (the final `list[MCMCChain]`, one per replicate,
    possibly still growing/non-converged if the cap was hit), `rounds_run`,
    `coarse_evals_per_chain` (each replicate's own true coarse-eval cost so far --
    *not* `chain.n_steps`; see `_run_chunk`'s docstring for why those differ whenever
    `subsampling_rate > 1`), `total_coarse_evals` (`max(coarse_evals_per_chain)`, used
    for the cap check and as a single summary number), `final_states` (each
    replicate's final `_ChunkState`, so callers can recover e.g. the final surrogate
    for methods where it keeps learning during this loop -- `online_active`; frozen
    methods' surrogates don't change here, so their initial state's model already has
    the final one), and the usual `find_burn_in_via_rhat` fields (`burn_in`, `rhat`,
    `ess_bulk`, `converged`).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if max_total_coarse_evals <= 0:
        raise ValueError("max_total_coarse_evals must be positive.")

    states = [f() for f in init_fns]
    n_chains = len(states)
    blocks: list[list[MCMCChain]] = [[] for _ in range(n_chains)]
    coarse_evals_per_chain = [0] * n_chains
    rounds_run = 0

    # A bare `joblib.Parallel(...)( ... )` call spins up a fresh worker pool (process
    # spawn + full re-import of every module used inside the workers, e.g. GPy/scipy)
    # and tears it down again on every call. Called once per *round* inside a while
    # loop, that overhead would dominate total runtime completely -- worse than just
    # running everything sequentially. Using `Parallel` as a context manager instead
    # keeps the same worker pool alive across every round in this call.
    with joblib.Parallel(n_jobs=n_jobs or n_chains) as parallel:
        while True:
            results = parallel(joblib.delayed(_run_chunk)(states[i], chunk_size) for i in range(n_chains))
            for i, (new_state, block, coarse_evals_spent) in enumerate(results):
                states[i] = new_state
                blocks[i].append(block)
                coarse_evals_per_chain[i] += coarse_evals_spent
            rounds_run += 1

            chains_so_far = [_concat_chain_blocks(b) for b in blocks]
            # Using max() (not min()) for the cap check is deliberate: it's the
            # comparison that actually bounds worst-case cost. A chain with a larger
            # subsampling_rate than its siblings can only overshoot chunk_size *more*
            # per round, never less, so capping on the fastest-growing chain is what
            # keeps this a genuine safety valve rather than a soft suggestion.
            total_coarse_evals = max(coarse_evals_per_chain)
            conv = find_burn_in_via_rhat(chains_so_far, rhat_threshold=rhat_threshold, patience=patience)

            if conv["converged"] or total_coarse_evals >= max_total_coarse_evals:
                return {
                    "chains": chains_so_far,
                    "rounds_run": rounds_run,
                    "coarse_evals_per_chain": coarse_evals_per_chain,
                    "total_coarse_evals": total_coarse_evals,
                    "final_states": states,
                    **conv,
                }


def run_convergence_driven_comparison(
    problem: Problem,
    *,
    n_init: int,
    pod_rank: int,
    seed_X: np.ndarray,
    seed_Y: np.ndarray,
    seed_surrogate: PODGPSurrogate,
    n_chains: int,
    chunk_size: int,
    max_total_coarse_evals: int,
    rhat_threshold: float = 1.01,
    patience: int = 5,
    theta0: np.ndarray | None = None,
    seed_base: int = 0,
    n_jobs: int | None = None,
) -> tuple[dict[str, Any], dict[str, list[MCMCChain]], dict[str, list[PODGPSurrogate]]]:
    """The convergence-driven analog of `run_convergence_check`: instead of comparing
    accuracy at a fixed, shared `n_coarse_evals` budget, runs each method until its
    replicate chains agree (R-hat<=`rhat_threshold`, `patience`-checked) or
    `max_total_coarse_evals` is hit, and reports the *cost* (HF calls, coarse evals) to
    get there alongside the resulting accuracy -- the comparison that actually
    demonstrates a computational-efficiency claim, and the one a reviewer will look
    for first, since it directly compares methods on the resource each is meant to
    save rather than an arbitrarily shared iteration count.

    Returns `(results, chains_by_method, surrogates_by_method)`, mirroring
    `run_convergence_check`'s signature so both can be JSON-dumped/plotted the same
    way:

    - `results`: JSON-serializable, keyed by method name ("hf_only", "pretrained",
      "online_active", "ours"), each entry containing `converged`, `rounds_run`,
      `coarse_evals_per_chain`, `total_coarse_evals`, `rhat`, `ess_bulk`, `burn_in`,
      and `pooled` (accuracy/cost summary via `pooled_summarize`, W2/KL-to-HF-only
      included for the surrogate-based methods). `ours`'s entry additionally has
      `adapt_metas` (per-replicate adaptive-phase bookkeeping).
    - `chains_by_method`: the raw replicate `MCMCChain` lists (not JSON-serializable,
      kept separate), for callers that want to plot the pooled samples. Also has
      `"ours_adapt"`: each replicate's adaptive-phase chain.
    - `surrogates_by_method`: each replicate's trained `PODGPSurrogate`, for
      "surrogate before/after" plots across all replicates -- keyed
      `"pretrained"`/`"online_active"`/`"ours"` only (`hf_only` has no surrogate).
      `pretrained`/`ours` surrogates are frozen during their monitored loop, so the
      *initial* per-replicate state's model already has the final surrogate;
      `online_active`'s keeps learning throughout, so its surrogate is read from
      `run_until_rhat_converged`'s `final_states` instead.
    """
    if theta0 is None:
        theta0 = np.asarray(problem.prior.rvs(random_state=set_seed(999 + n_init + seed_base)), dtype=float)

    results: dict[str, Any] = {}
    chains_by_method: dict[str, list[MCMCChain]] = {}
    surrogates_by_method: dict[str, list[PODGPSurrogate]] = {}

    hf_run = run_until_rhat_converged(
        [lambda i=i: _init_hf_only_state(problem, seed=seed_base + 100 + i, theta0=theta0) for i in range(n_chains)],
        chunk_size=chunk_size, max_total_coarse_evals=max_total_coarse_evals,
        rhat_threshold=rhat_threshold, patience=patience, n_jobs=n_jobs,
    )
    chains_by_method["hf_only"] = hf_run.pop("chains")
    hf_run.pop("final_states")  # no surrogate for HF-only; not JSON-serializable, must not leak into results
    hf_run["pooled"] = pooled_summarize(
        chains_by_method["hf_only"], problem, burn_in=hf_run["burn_in"] or 0, n_offline_hf=0
    )
    results["hf_only"] = hf_run

    hf_reference_mean = np.asarray(hf_run["pooled"]["posterior_mean"])
    hf_reference_cov = np.asarray(hf_run["pooled"]["posterior_cov"])

    # Pretrained's offline design (and its HF cost) is trained once per replicate,
    # up front, outside the monitored round loop -- it's a fixed cost, not something
    # the R-hat-driven stopping rule is meant to trade off against.
    pretrained_inits = []
    n_hf_pretrained_list = []
    for i in range(n_chains):
        state, n_hf = _init_pretrained_state(
            problem, seed_X=seed_X, seed_Y=seed_Y, gamma_threshold=GAMMA_THRESHOLD, pod_rank=pod_rank,
            kernel=KERNEL, seed=seed_base + 200 + i, theta0=theta0,
        )
        pretrained_inits.append(state)
        n_hf_pretrained_list.append(n_hf)
    pre_run = run_until_rhat_converged(
        [lambda s=s: s for s in pretrained_inits],
        chunk_size=chunk_size, max_total_coarse_evals=max_total_coarse_evals,
        rhat_threshold=rhat_threshold, patience=patience, n_jobs=n_jobs,
    )
    chains_by_method["pretrained"] = pre_run.pop("chains")
    pre_run.pop("final_states")  # frozen throughout: pretrained_inits already has the final surrogate
    surrogates_by_method["pretrained"] = [s.model.lf_model for s in pretrained_inits]
    pre_run["pooled"] = pooled_summarize(
        chains_by_method["pretrained"], problem, burn_in=pre_run["burn_in"] or 0, n_offline_hf=n_hf_pretrained_list,
        reference_mean=hf_reference_mean, reference_cov=hf_reference_cov,
    )
    results["pretrained"] = pre_run

    on_run = run_until_rhat_converged(
        [
            lambda i=i: _init_online_active_state(
                problem, surrogate=seed_surrogate, gamma_threshold=GAMMA_THRESHOLD,
                seed=seed_base + 300 + i, theta0=theta0,
            )
            for i in range(n_chains)
        ],
        chunk_size=chunk_size, max_total_coarse_evals=max_total_coarse_evals,
        rhat_threshold=rhat_threshold, patience=patience, n_jobs=n_jobs,
    )
    chains_by_method["online_active"] = on_run.pop("chains")
    # Unlike pretrained/ours, online_active's surrogate keeps learning throughout the
    # monitored loop -- its *final* state (not the initial one) has the surrogate that
    # actually produced the returned chains.
    surrogates_by_method["online_active"] = [s.model.lf_model for s in on_run.pop("final_states")]
    on_run["pooled"] = pooled_summarize(
        chains_by_method["online_active"], problem, burn_in=on_run["burn_in"] or 0, n_offline_hf=n_init,
        reference_mean=hf_reference_mean, reference_cov=hf_reference_cov,
    )
    results["online_active"] = on_run

    # ours: adaptive phase runs to each replicate's own completion first (outside the
    # monitored loop, same reasoning as pretrained's offline design), then production
    # is chunked and monitored exactly like the other three methods.
    ours_inits = []
    ours_adapt_metas = []
    ours_adapt_chains = []
    for i in range(n_chains):
        state, adapt_meta, adapt_chain = _init_ours_production_state(
            problem, surrogate=seed_surrogate, gamma_threshold=GAMMA_THRESHOLD,
            max_adapt_coarse_evals=MAX_ADAPT_COARSE_EVALS, seed=seed_base + 400 + i, theta0=theta0,
        )
        ours_inits.append(state)
        ours_adapt_metas.append(adapt_meta)
        ours_adapt_chains.append(adapt_chain)
    ours_run = run_until_rhat_converged(
        [lambda s=s: s for s in ours_inits],
        chunk_size=chunk_size, max_total_coarse_evals=max_total_coarse_evals,
        rhat_threshold=rhat_threshold, patience=patience, n_jobs=n_jobs,
    )
    chains_by_method["ours"] = ours_run.pop("chains")
    ours_run.pop("final_states")  # frozen throughout production: ours_inits already has the final surrogate
    surrogates_by_method["ours"] = [s.model.lf_model for s in ours_inits]
    # ours's production chains are DA blocks (subsampling_rate coarse evals each), not
    # one coarse eval per row -- `run_until_rhat_converged` already tracks each
    # replicate's true coarse-eval cost during production (coarse_evals_per_chain, for
    # exactly this reason), so only the adaptive phase's cost needs adding on top.
    ours_production_coarse_evals = [
        int(adapt_meta["adapt_coarse_evals_used"]) + production_cost
        for adapt_meta, production_cost in zip(ours_adapt_metas, ours_run["coarse_evals_per_chain"], strict=True)
    ]
    ours_run["pooled"] = pooled_summarize(
        chains_by_method["ours"], problem, burn_in=ours_run["burn_in"] or 0, n_offline_hf=n_init,
        n_coarse_eval_units=ours_production_coarse_evals,
        reference_mean=hf_reference_mean, reference_cov=hf_reference_cov,
    )
    ours_run["adapt_metas"] = ours_adapt_metas
    results["ours"] = ours_run
    chains_by_method["ours_adapt"] = ours_adapt_chains

    return results, chains_by_method, surrogates_by_method
