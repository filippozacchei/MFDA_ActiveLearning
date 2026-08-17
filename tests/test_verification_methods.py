from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from gp_active_mcmc.verification.design import build_initial_surrogate
from gp_active_mcmc.verification.methods import (
    run_adaptive_da,
    run_hf_only,
    run_online_active,
    run_pretrained,
    run_training_cost_comparison,
)

_T = np.linspace(0.0, 1.0, 15)

# gp_active_mcmc.inference.sampling.sample_active_chain calls tinyDA's tda.sample()
# with the now-deprecated `subsampling_rate` kwarg (tinyDA 0.9.21 prefers
# `subchain_length`); every other test in this repo mocks tda.sample rather than
# calling it for real, so this warning has never surfaced before. Pre-existing,
# unrelated to this refactor -- silenced here rather than routed around, since these
# tests deliberately exercise the real tinyDA call path.
_IGNORE_SUBSAMPLING_RATE_DEPRECATION = pytest.mark.filterwarnings(
    "ignore:.*subsampling_rate has been deprecated.*:UserWarning"
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


def _make_problem() -> _Problem:
    prior = multivariate_normal(mean=[1.0, 1.0], cov=np.eye(2) * 0.09)
    rng = np.random.default_rng(0)
    theta_true = np.asarray(prior.rvs(random_state=rng), dtype=float)
    return _Problem(
        prior=prior, theta_true=theta_true, y_obs=_hf_forward(theta_true), sigma_obs=0.05, hf_forward=_hf_forward
    )


def test_run_hf_only_returns_chain_of_expected_length() -> None:
    problem = _make_problem()
    chain = run_hf_only(problem, iterations=20, seed=1)
    assert chain.n_steps == 21  # tinyDA echoes initial_parameters as row 0


def test_run_hf_only_used_hf_always_true() -> None:
    problem = _make_problem()
    chain = run_hf_only(problem, iterations=15, seed=2)
    assert chain.extras.used_hf is not None
    assert np.all(chain.extras.used_hf)


@_IGNORE_SUBSAMPLING_RATE_DEPRECATION
def test_run_pretrained_returns_trained_surrogate_and_hf_count() -> None:
    problem = _make_problem()
    rng = np.random.default_rng(3)
    seed_X = np.asarray([problem.prior.rvs(random_state=rng) for _ in range(6)], dtype=float)
    seed_Y = np.asarray([problem.hf_forward(x) for x in seed_X], dtype=float)

    chain, n_hf_spent, surrogate = run_pretrained(
        problem, seed_X=seed_X, seed_Y=seed_Y, gamma_threshold=1e6, pod_rank=2, kernel="rbf",
        iterations=10, seed=4,
    )
    assert n_hf_spent == 6  # gamma_threshold trivially satisfied -> no extra HF points
    assert chain.n_steps == 11
    assert surrogate.gp.n_train == 6


@_IGNORE_SUBSAMPLING_RATE_DEPRECATION
def test_run_online_active_returns_updated_surrogate_copy() -> None:
    problem = _make_problem()
    rng = np.random.default_rng(5)
    surrogate, _X, _Y = build_initial_surrogate(problem, rng, n_init=6, pod_rank=2, kernel="rbf")

    chain, updated = run_online_active(
        problem, surrogate=surrogate, gamma_threshold=1e6, iterations=15, seed=6,
    )
    assert chain.n_steps == 16
    assert updated is not surrogate  # _surrogate_for_online_learning deep-copies


@_IGNORE_SUBSAMPLING_RATE_DEPRECATION
def test_run_adaptive_da_returns_chain_metadata_and_surrogate() -> None:
    problem = _make_problem()
    rng = np.random.default_rng(7)
    surrogate, _X, _Y = build_initial_surrogate(problem, rng, n_init=6, pod_rank=2, kernel="rbf")

    chain, meta, updated = run_adaptive_da(
        problem, surrogate=surrogate, gamma_threshold=1e6, n_coarse_evals=30, max_adapt_coarse_evals=30, seed=8,
    )
    assert meta["phase"] in {"adapt_only", "adapt_then_production"}
    assert meta["n_adapt_samples"] > 0
    assert chain.n_steps > 0
    assert updated is not surrogate


@_IGNORE_SUBSAMPLING_RATE_DEPRECATION
def test_run_training_cost_comparison_return_type() -> None:
    # Regression test: this function must return a 2-tuple (dict, PODGPSurrogate), not
    # a bare dict -- its declared return type was silently wrong before this move (the
    # code has always returned a tuple; only the annotation was stale, invisible
    # because examples/ was never mypy-checked).
    problem = _make_problem()
    rng = np.random.default_rng(9)
    seed_X = np.asarray([problem.prior.rvs(random_state=rng) for _ in range(6)], dtype=float)
    seed_Y = np.asarray([problem.hf_forward(x) for x in seed_X], dtype=float)
    seed_surrogate, _X, _Y = build_initial_surrogate(problem, rng, n_init=6, pod_rank=2, kernel="rbf")

    result = run_training_cost_comparison(
        problem, seed_X=seed_X, seed_Y=seed_Y, seed_surrogate=seed_surrogate, gamma_threshold=1e6,
        pod_rank=2, kernel="rbf", max_adapt_coarse_evals=20, seed_base=1000,
    )
    assert isinstance(result, tuple)
    assert len(result) == 2
    metrics, offline_surrogate = result
    assert isinstance(metrics, dict)
    assert "offline" in metrics and "online" in metrics
    assert offline_surrogate.gp.n_train >= 6
