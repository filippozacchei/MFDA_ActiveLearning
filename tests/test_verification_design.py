from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import multivariate_normal

from gp_active_mcmc.verification.design import (
    OnlineLearningConfig,
    _surrogate_for_online_learning,
    active_learning_offline_design,
    build_initial_surrogate,
    select_pod_rank_and_seed_design,
)

_T = np.linspace(0.0, 1.0, 20)


def _hf_forward(theta: np.ndarray) -> np.ndarray:
    # Deliberately nonlinear in theta (damped-oscillator-shaped, like a toy physics
    # model): an exactly-linear forward model gives the GP's ARD lengthscales no
    # genuine scale to find, which drives GPy's optimizer into a numerically unstable
    # (overflow) regime on such tiny synthetic training sets.
    theta = np.asarray(theta, dtype=float)
    return np.exp(-theta[0] * _T) * np.sin(2 * np.pi * theta[1] * _T)


@dataclass
class _Problem:
    prior: Any
    theta_true: np.ndarray
    y_obs: np.ndarray
    sigma_obs: float
    hf_forward: Any


def _make_problem(seed: int = 0) -> _Problem:
    prior = multivariate_normal(mean=[1.0, 1.0], cov=np.eye(2) * 0.09)
    rng = np.random.default_rng(seed)
    theta_true = np.asarray(prior.rvs(random_state=rng), dtype=float)
    return _Problem(
        prior=prior, theta_true=theta_true, y_obs=_hf_forward(theta_true), sigma_obs=0.05, hf_forward=_hf_forward
    )


def test_build_initial_surrogate_shapes() -> None:
    problem = _make_problem()
    rng = np.random.default_rng(1)
    surrogate, X, Y = build_initial_surrogate(problem, rng, n_init=8, pod_rank=2, kernel="rbf")
    assert X.shape == (8, 2)
    assert Y.shape == (8, 20)
    assert surrogate.gp.n_train == 8
    assert surrogate.gp.n_retrain_max == 0
    assert surrogate.X_history is not None and surrogate.X_history.shape == (8, 2)
    assert surrogate.Y_history is not None and surrogate.Y_history.shape == (8, 20)


def test_select_pod_rank_and_seed_design_respects_energy_threshold() -> None:
    problem = _make_problem()
    rng = np.random.default_rng(2)
    selection = select_pod_rank_and_seed_design(problem, rng, n_init=15, energy_threshold=0.95)
    assert selection.n_init == 15
    assert selection.X.shape == (15, 2)
    assert selection.energy_curve[selection.pod_rank - 1] >= 0.95


def test_select_pod_rank_and_seed_design_respects_r_max_cap() -> None:
    problem = _make_problem()
    rng = np.random.default_rng(3)
    selection = select_pod_rank_and_seed_design(problem, rng, n_init=15, energy_threshold=0.999999, r_max_cap=3)
    assert selection.pod_rank <= 3


def test_active_learning_offline_design_stops_on_gamma_threshold() -> None:
    problem = _make_problem()
    rng = np.random.default_rng(4)
    seed_X = np.asarray([problem.prior.rvs(random_state=rng) for _ in range(10)], dtype=float)
    seed_Y = np.asarray([problem.hf_forward(x) for x in seed_X], dtype=float)
    surrogate = active_learning_offline_design(
        problem, seed_X, seed_Y, gamma_threshold=1e6, pod_rank=2, kernel="rbf", rng=rng,
    )
    assert surrogate.gp.n_train == 10  # criterion trivially satisfied immediately, no extra points acquired


def test_active_learning_offline_design_respects_max_total_budget() -> None:
    problem = _make_problem()
    rng = np.random.default_rng(5)
    seed_X = np.asarray([problem.prior.rvs(random_state=rng) for _ in range(5)], dtype=float)
    seed_Y = np.asarray([problem.hf_forward(x) for x in seed_X], dtype=float)
    surrogate = active_learning_offline_design(
        problem, seed_X, seed_Y, gamma_threshold=1e-12, pod_rank=2, kernel="rbf", rng=rng,
        batch_size=5, max_total_budget=15,
    )
    assert surrogate.gp.n_train == 15  # never reaches the (unreachable) criterion, capped by max_total_budget


def test_surrogate_for_online_learning_deepcopies_and_sets_fields() -> None:
    problem = _make_problem()
    rng = np.random.default_rng(6)
    surrogate, _X, _Y = build_initial_surrogate(problem, rng, n_init=6, pod_rank=2, kernel="rbf")

    config = OnlineLearningConfig(
        pod_refit_every=5, pod_refit_max=3, adaptive_rank=True, rank_energy_threshold=0.99, rank_max=4,
    )
    lf = _surrogate_for_online_learning(surrogate, config)
    assert lf is not surrogate
    assert lf.pod_refit_every == 5
    assert lf.pod_refit_max == 3
    assert lf.adaptive_rank is True
    assert lf.rank_energy_threshold == 0.99
    assert lf.rank_max == 4
    # Original untouched.
    assert surrogate.pod_refit_every is None
    assert surrogate.adaptive_rank is False
