"""Navier-Stokes (backward-facing step) problem glue for the shared
`gp_active_mcmc.verification` comparison harness.

Mirrors `examples/toy_problem/msd_methods.py`'s role for MSD: only the physics/problem
setup (`Problem`, `build_problem`, this problem's calibrated constants) is defined
here. Every problem-agnostic function -- `run_hf_only`, `run_pretrained`,
`run_adaptive_stm`, `run_convergence_driven_comparison`, the metrics, the chunked-round
harness -- lives in `gp_active_mcmc.verification` and is re-exported unchanged, so this
module and `examples/toy_problem/msd_methods.py` both build on exactly the same tested,
documented implementation rather than each maintaining their own copy.

`theta = [h1, U_in, L_down]`: upstream channel height, inlet velocity, downstream
channel length (`h2`/`L_up` held fixed at `solver_hf.BFSGeometry`'s own defaults). The
observed quantity is the outlet streamwise-velocity profile `u_x(y)`, resampled to a
fixed length `T` (see `resample_profile`).

This module has no `__main__`: it is imported by `run.py` (the multi-seed sweep
driver, mirroring `examples/toy_problem/run.py`) and `ns_benchmark.ipynb` (this
problem's paper-figure notebook, mirroring `msd_benchmark.ipynb`). `backward.py` and
`forward.py` are separate, standalone single-run tutorials and intentionally do not
import from here.

Verification status
--------------------
`build_problem`/`make_forward_model` below (like `backward.py`/`forward.py`) depend on
`solver_hf.forward_model`, which imports `dolfinx`/`mpi4py`/`petsc4py`/`basix` -- a
FEniCSx stack this repository does not install as a dependency and that is not
available in every development environment. Nothing here has been executed; it is
verified by `mypy --strict` and by manual cross-checking against the execution-tested
MSD side (`examples/toy_problem/`), which exercises the exact same
`gp_active_mcmc.verification` call surface. The calibrated constants below are
therefore illustrative, not tuned by a real sweep -- scaled down from MSD's own
numbers to reflect that each HF evaluation here is a real FEM solve, not a
closed-form one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from resample import resample_profile
from scipy.stats import multivariate_normal
from solver_hf import forward_model as hf_solver

from gp_active_mcmc.surrogates.gp import KernelName
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

FloatArray = NDArray[np.float64]

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
    "T",
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

PARAM_NAMES = ("h1", "U_in", "L_down")

# ---------------------------------------------------------------------------
# Prior support -- single source of truth for this problem's bounds. Previously
# duplicated inconsistently between `backward.py` (U_in in [0.5, 1.5]) and
# `forward.py` (U_in in [0.25, 1.25]), a real bug (h1/L_down bounds already agreed).
# Both files stay untouched (future tutorials); this module's own inference prior
# below is independent of either.
# ---------------------------------------------------------------------------
H1_MIN, H1_MAX = 0.05, 0.15
U_MIN, U_MAX = 0.5, 1.50
L_MIN, L_MAX = 0.30, 0.50

# Fixed solver geometry (kept at solver_hf.BFSGeometry's own defaults; not inferred).
H2 = 0.20
L_UP = 0.10

# ---------------------------------------------------------------------------
# NS-calibrated experiment configuration (single source of truth for all consumers
# of *this* problem -- gp_active_mcmc.verification's own functions take these as
# explicit arguments rather than defaulting to any problem's numbers). Illustrative,
# not tuned by a real sweep -- see module docstring.
# ---------------------------------------------------------------------------
T = 100  # outlet-profile length after resample_profile
KERNEL: KernelName = "matern52"
GAMMA_THRESHOLD = 0.001  # 0.1 * the default sigma_obs below, matching MSD's convention
MAX_ADAPT_COARSE_EVALS = 150
MAX_SUBCHAIN = 20
N_INIT = 15
POD_REFIT_EVERY = 15
POD_REFIT_MAX: int | None = None
RANK_ENERGY_THRESHOLD = 0.999
RANK_MAX: int | None = None

# ---------------------------------------------------------------------------
# Problem setup (shared across all methods and seeds)
# ---------------------------------------------------------------------------


@dataclass
class Problem:
    """Satisfies `gp_active_mcmc.verification.Problem` structurally -- no `t` field:
    the outlet-profile abscissa `y` isn't a solver time input the way MSD's timeline
    is, so there is no natural equivalent to carry here."""

    prior: Any
    theta_true: FloatArray
    y_obs: FloatArray
    sigma_obs: float
    hf_forward: Any
    param_names: tuple[str, ...] = PARAM_NAMES


def make_prior() -> Any:
    """Gaussian prior over `theta = [h1, U_in, L_down]`, centered on the bounds'
    midpoint with a standard deviation of a quarter of each bound's width. Support is
    informative, not enforced (unlike MSD's `IndependentUniformPrior`): draws outside
    `[H1_MIN, H1_MAX]` etc. are possible but unlikely.
    """
    mean = np.array([0.5 * (H1_MIN + H1_MAX), 0.5 * (U_MIN + U_MAX), 0.5 * (L_MIN + L_MAX)])
    sigma = np.array([0.25 * (H1_MAX - H1_MIN), 0.25 * (U_MAX - U_MIN), 0.25 * (L_MAX - L_MIN)])
    return multivariate_normal(mean=mean, cov=np.diag(sigma**2))


def make_forward_model(*, T: int) -> Any:
    """Wraps `solver_hf.forward_model` (fixed `h2`/`L_up`) into the `theta -> y`
    callable `gp_active_mcmc.verification.Problem.hf_forward` expects: resamples the
    solver's raw `(y, u_x)` outlet profile to a fixed-length vector via
    `resample_profile`."""

    def f(theta: FloatArray) -> FloatArray:
        th = np.asarray(theta, dtype=float).ravel()
        if th.shape[0] != 3:
            raise ValueError("Expected theta = [h1, U_in, L_down].")
        y, u = hf_solver(float(th[0]), U_in=float(th[1]), h2=H2, L_up=L_UP, L_down=float(th[2]))
        return resample_profile(y, u, T=T)

    return f


def build_problem(*, problem_seed: int, sigma_obs: float = 0.01) -> Problem:
    rng = set_seed(problem_seed)
    prior = make_prior()
    hf_forward = make_forward_model(T=T)

    theta_true = np.asarray(prior.rvs(random_state=rng), dtype=float)
    y_clean = hf_forward(theta_true)
    y_obs = y_clean + sigma_obs * rng.standard_normal(size=T)

    return Problem(prior=prior, theta_true=theta_true, y_obs=y_obs, sigma_obs=sigma_obs, hf_forward=hf_forward)
