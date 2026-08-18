from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from gp_active_mcmc.verification.design import build_initial_surrogate
from gp_active_mcmc.verification.harness import (
    run_convergence_driven_comparison,
)
from gp_active_mcmc.verification.methods import _init_hf_only_state
from gp_active_mcmc.verification.rounds import ConvergenceConfig, run_until_rhat_converged

_T = np.linspace(0.0, 1.0, 15)

# See test_verification_methods.py for why this is needed (pre-existing, unrelated
# tinyDA-version mismatch, only ever surfaced by tests that call it for real).
_IGNORE_SUBSAMPLING_RATE_DEPRECATION = pytest.mark.filterwarnings(
    "ignore:.*subsampling_rate has been deprecated.*:UserWarning"
)

# With this few real MCMC samples (tiny chunk_size/n_coarse_evals, deliberately kept
# small for test speed), a chain occasionally has exactly zero within-window sample
# variance for one of arviz.rhat's candidate burn-ins, which arviz turns into a
# genuine (if benign) `RuntimeWarning: divide by zero` -- turned into a hard failure
# only because of this repo's own `filterwarnings = ["error", ...]` pytest config.
# Confirmed pre-existing and unrelated to this refactor: the same underlying
# tiny-sample-count non-determinism was directly observed (running the exact same
# post-refactor code twice in a row) while verifying the msd_methods.py ->
# gp_active_mcmc.verification move didn't change behavior.
_IGNORE_RHAT_DIVIDE_BY_ZERO = pytest.mark.filterwarnings(
    "ignore:divide by zero encountered in scalar divide:RuntimeWarning"
)


def _hf_forward(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    return np.exp(-theta[0] * _T) * np.sin(2 * np.pi * theta[1] * _T)


@dataclass
class _Problem:
    prior: Any
    theta_true: np.ndarray
    y_obs: np.ndarray
    sigma_obs: float
    hf_forward: Any
    param_names: tuple[str, ...]


def _make_problem() -> _Problem:
    prior = multivariate_normal(mean=[1.0, 1.0], cov=np.eye(2) * 0.09)
    rng = np.random.default_rng(0)
    theta_true = np.asarray(prior.rvs(random_state=rng), dtype=float)
    return _Problem(
        prior=prior, theta_true=theta_true, y_obs=_hf_forward(theta_true), sigma_obs=0.05, hf_forward=_hf_forward,
        param_names=("a", "b"),
    )


@_IGNORE_RHAT_DIVIDE_BY_ZERO
def test_run_until_rhat_converged_stops_on_max_total_coarse_evals_cap() -> None:
    problem = _make_problem()
    init_fns = [
        (lambda i=i: _init_hf_only_state(problem, seed=3000 + i)) for i in range(2)
    ]
    config = ConvergenceConfig(chunk_size=10, max_total_coarse_evals=25, rhat_threshold=1e-9, n_jobs=1)
    result = run_until_rhat_converged(init_fns, config=config, param_names=("a", "b"), verbose=False)
    assert result["stop_reason"] == "max_coarse_evals"
    assert not result["converged"]
    assert result["total_coarse_evals"] >= 25


@_IGNORE_RHAT_DIVIDE_BY_ZERO
def test_run_until_rhat_converged_stops_on_max_hf_evals_cap() -> None:
    problem = _make_problem()
    # hf_only states use HF on every single coarse step, so hf_evals_per_chain tracks
    # coarse_evals_per_chain 1:1 -- a max_hf_evals well below max_total_coarse_evals
    # must be what actually stops the loop.
    init_fns = [
        (lambda i=i: _init_hf_only_state(problem, seed=4000 + i)) for i in range(2)
    ]
    config = ConvergenceConfig(chunk_size=10, max_total_coarse_evals=1000, rhat_threshold=1e-9, n_jobs=1)
    result = run_until_rhat_converged(
        init_fns, config=config, param_names=("a", "b"), max_hf_evals=15, verbose=False,
    )
    assert result["stop_reason"] == "max_hf_evals"
    assert result["total_hf_evals"] >= 15


@_IGNORE_RHAT_DIVIDE_BY_ZERO
@_IGNORE_SUBSAMPLING_RATE_DEPRECATION
def test_run_convergence_driven_comparison_methods_param_restricts_execution() -> None:
    problem = _make_problem()
    rng = np.random.default_rng(5)
    seed_X = np.asarray([problem.prior.rvs(random_state=rng) for _ in range(6)], dtype=float)
    seed_surrogate, _X, _Y = build_initial_surrogate(problem, rng, n_init=6, kernel="rbf")

    convergence = ConvergenceConfig(chunk_size=10, max_total_coarse_evals=20, rhat_threshold=1e-9, min_ess=1.0, n_jobs=1)
    results, chains_by_method, surrogates_by_method = run_convergence_driven_comparison(
        problem, seed_X=seed_X, seed_surrogate=seed_surrogate,
        n_chains=2, gamma_threshold=1e6, max_adapt_coarse_evals=15,
        convergence=convergence, methods=("hf_only",), seed_base=5000,
    )
    assert set(results.keys()) == {"hf_only"}
    assert set(chains_by_method.keys()) == {"hf_only"}
    assert surrogates_by_method == {}


@_IGNORE_RHAT_DIVIDE_BY_ZERO
@_IGNORE_SUBSAMPLING_RATE_DEPRECATION
def test_run_convergence_driven_comparison_adaptive_stm_dict_keys() -> None:
    problem = _make_problem()
    rng = np.random.default_rng(6)
    seed_X = np.asarray([problem.prior.rvs(random_state=rng) for _ in range(6)], dtype=float)
    seed_surrogate, _X, _Y = build_initial_surrogate(problem, rng, n_init=6, kernel="rbf")

    convergence = ConvergenceConfig(chunk_size=10, max_total_coarse_evals=20, rhat_threshold=1e-9, min_ess=1.0, n_jobs=1)
    results, chains_by_method, surrogates_by_method = run_convergence_driven_comparison(
        problem, seed_X=seed_X, seed_surrogate=seed_surrogate,
        n_chains=2, gamma_threshold=1e6, max_adapt_coarse_evals=15,
        convergence=convergence, methods=("hf_only", "adaptive_stm"),
        seed_base=6000,
    )
    assert "adaptive_stm" in results
    assert "ours" not in results
    assert "adaptive_stm_adapt" in chains_by_method
    assert "adaptive_stm" in chains_by_method
    assert "adaptive_stm" in surrogates_by_method
