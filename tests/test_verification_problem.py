from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gp_active_mcmc.verification.problem import Problem


def _hf_forward(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    return np.array([theta[0] + theta[1], theta[0] - theta[1]], dtype=float)


@dataclass
class _DummyProblem:
    prior: Any
    theta_true: np.ndarray
    y_obs: np.ndarray
    sigma_obs: float
    hf_forward: Any
    param_names: tuple[str, ...]


@dataclass
class _DummyProblemWithExtraField:
    """Mirrors a domain-specific Problem (e.g. MSD's, with its own `t` field) that
    carries extra state beyond the shared protocol."""

    prior: Any
    theta_true: np.ndarray
    y_obs: np.ndarray
    sigma_obs: float
    hf_forward: Any
    param_names: tuple[str, ...]
    t: np.ndarray


def _make_problem() -> _DummyProblem:
    theta_true = np.array([1.0, 2.0])
    return _DummyProblem(
        prior=None, theta_true=theta_true, y_obs=_hf_forward(theta_true), sigma_obs=0.1, hf_forward=_hf_forward,
        param_names=("a", "b"),
    )


def test_dataclass_problem_satisfies_protocol() -> None:
    problem = _make_problem()
    assert isinstance(problem, Problem)


def test_missing_field_fails_protocol_check() -> None:
    @dataclass
    class _Incomplete:
        prior: Any
        theta_true: np.ndarray
        y_obs: np.ndarray
        # sigma_obs deliberately omitted
        hf_forward: Any
        param_names: tuple[str, ...]

    incomplete = _Incomplete(
        prior=None, theta_true=np.zeros(2), y_obs=np.zeros(2), hf_forward=_hf_forward, param_names=("a", "b")
    )
    assert not isinstance(incomplete, Problem)


def test_msd_like_problem_with_extra_field_still_satisfies() -> None:
    theta_true = np.array([1.0, 2.0])
    problem = _DummyProblemWithExtraField(
        prior=None, theta_true=theta_true, y_obs=_hf_forward(theta_true), sigma_obs=0.1,
        hf_forward=_hf_forward, param_names=("a", "b"), t=np.linspace(0.0, 1.0, 10),
    )
    assert isinstance(problem, Problem)
