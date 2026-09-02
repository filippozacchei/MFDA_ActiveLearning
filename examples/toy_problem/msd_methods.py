"""Mass-spring-damper (MSD) problem glue for the shared `gp_active_mcmc.verification`
comparison harness.

Implements four inference strategies at a (roughly) matched high-fidelity (HF) budget:

1. HF-only Metropolis-Hastings -- ground-truth reference and the "standard
   single-stage MCMC" baseline. See `gp_active_mcmc.verification.run_hf_only`.
2. Pretrained surrogate + plain MH -- POD-GP surrogate trained *offline only* via a
   greedy max-variance active-learning acquisition, then frozen; matches the
   "Pretrained Model" arm in Riccius et al. (2025). See
   `gp_active_mcmc.verification.run_pretrained`.
3. Adaptive surrogate MCMC (Riccius, synced) -- the surrogate is substituted directly
   for the likelihood and refined online via an uncertainty threshold, with no
   HF-correction step, matching Algorithm 1 in Riccius et al. (2025), "Integration of
   Active Learning and MCMC Sampling for Efficient Bayesian Calibration of Mechanical
   Properties"; every replicate's individually-collected HF points are additionally
   pooled into a shared surrogate each round, isolating inter-replicate disagreement
   from the separate correctness problem DA correction solves. Built from
   `run_convergence_driven_comparison`'s orchestration (no standalone `run_*`
   function), not `gp_active_mcmc.verification.methods._init_adaptive_surrogate_mcmc_state`
   directly.
4. Proposed (adaptive_stm): adaptive delayed-acceptance MCMC with a freeze-to-production
   stage. See `gp_active_mcmc.verification.run_adaptive_stm`.

Only genuinely MSD-specific code lives here: the `Problem` dataclass (with its extra
`t` timeline field), `build_problem`, and this problem's calibrated constants
(`N_INIT`, `GAMMA_THRESHOLD`, ...). Every problem-agnostic function --
`run_hf_only`, `run_pretrained`, `run_adaptive_stm`, `run_convergence_driven_comparison`,
the metrics, the chunked-round harness -- lives in `gp_active_mcmc.verification` and is
re-exported here unchanged, so this module and `examples/navier_stokes/ns_problem.py`
both build on exactly the same tested, documented implementation rather than each
maintaining their own copy.

This module has no `__main__`: it is imported by `run.py` (the multi-seed sweep
driver) and `msd_benchmark.ipynb` (the paper's comparison notebook).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from msd import make_forward_model, make_observation, make_prior, make_timeline

from gp_active_mcmc.utils.rng import set_seed
from gp_active_mcmc.verification import (
    DEFAULT_ONLINE_LEARNING,
    ConvergenceConfig,
    OnlineLearningConfig,
    PODRankSelection,
    active_learning_offline_design,
    adaptive_stm_coarse_eval_units,
    adaptive_stm_cumulative_coarse_evals,
    adaptive_stm_full_resolution_trace,
    adaptive_stm_multichain_trace,
    build_initial_surrogate,
    effective_burn_in,
    find_burn_in_via_rhat,
    gaussian_kl,
    gaussian_wasserstein2,
    make_proposal,
    multichain_diagnostics,
    pooled_summarize,
    prepare_trace_data,
    print_convergence_driven_table,
    run_adaptive_stm,
    run_convergence_driven_comparison,
    run_hf_only,
    run_pretrained,
    run_training_cost_comparison,
    run_until_rhat_converged,
    select_pod_rank_and_seed_design,
    summarize,
)

__all__ = [
    "DEFAULT_ONLINE_LEARNING",
    "GAMMA_THRESHOLD",
    "KERNEL",
    "MAX_ADAPT_COARSE_EVALS",
    "MAX_SUBCHAIN",
    "N_INIT",
    "PARAM_NAMES",
    "POD_REFIT_EVERY",
    "POD_REFIT_MAX",
    "RANK_ENERGY_THRESHOLD",
    "RANK_MAX",
    "RESULTS_DIR",
    "ConvergenceConfig",
    "OnlineLearningConfig",
    "PODRankSelection",
    "Problem",
    "active_learning_offline_design",
    "adaptive_stm_coarse_eval_units",
    "adaptive_stm_cumulative_coarse_evals",
    "adaptive_stm_full_resolution_trace",
    "adaptive_stm_multichain_trace",
    "build_initial_surrogate",
    "build_problem",
    "effective_burn_in",
    "find_burn_in_via_rhat",
    "gaussian_kl",
    "gaussian_wasserstein2",
    "make_proposal",
    "multichain_diagnostics",
    "pooled_summarize",
    "prepare_trace_data",
    "print_convergence_driven_table",
    "run_adaptive_stm",
    "run_convergence_driven_comparison",
    "run_hf_only",
    "run_pretrained",
    "run_training_cost_comparison",
    "run_until_rhat_converged",
    "select_pod_rank_and_seed_design",
    "summarize",
]

RESULTS_DIR = Path(__file__).parent / "results"

PARAM_NAMES = ("k", "c")

# ---------------------------------------------------------------------------
# MSD-calibrated experiment configuration (single source of truth for all
# consumers of *this* problem -- gp_active_mcmc.verification's own functions take
# these as explicit arguments rather than defaulting to any problem's numbers).
# ---------------------------------------------------------------------------

KERNEL = "matern52"
GAMMA_THRESHOLD = 0.01
MAX_ADAPT_COARSE_EVALS = 1000
MAX_SUBCHAIN = 25
N_INIT = 25
POD_REFIT_EVERY = 25
POD_REFIT_MAX: int | None = None
RANK_ENERGY_THRESHOLD = 0.99
RANK_MAX: int | None = None

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
    scale: float
    param_names: tuple[str, ...] = PARAM_NAMES
    


def build_problem(*, problem_seed: int, sigma_obs: float = 0.1) -> Problem:
    rng = set_seed(problem_seed)
    t = make_timeline(T=300, t_end=4.0)
    prior = make_prior()
    hf_forward = make_forward_model(t)

    theta_true = prior.rvs(random_state=rng)
    y_obs = make_observation(rng, theta_true, t, sigma_obs)
    scale = 1.0

    return Problem(t=t, prior=prior, theta_true=theta_true, y_obs=y_obs, sigma_obs=sigma_obs, hf_forward=hf_forward, scale=scale)
