from __future__ import annotations

import numpy as np
import pytest

from gp_active_mcmc.utils.mcmc import (
    acceptance_rate_from_accepted,
    extract_samples,
    hf_call_fraction,
    mean_subchain_length,
    move_fraction_from_samples,
    posterior_rmse,
)


class _Link:
    def __init__(self, parameters: np.ndarray):
        self.parameters = parameters


def test_extract_samples() -> None:
    chain = {"chain_0": [_Link(np.array([1.0, 2.0])), _Link(np.array([3.0, 4.0]))]}
    samples = extract_samples(chain, chain_key="chain_0")
    assert samples.shape == (2, 2)
    np.testing.assert_allclose(samples[0], [1.0, 2.0])

    with pytest.raises(KeyError):
        _ = extract_samples(chain, chain_key="missing")


def test_move_fraction_from_samples() -> None:
    s = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    # diffs: [no move, move] => 0.5
    assert move_fraction_from_samples(s) == pytest.approx(0.5)


def test_acceptance_rate_from_accepted() -> None:
    accepted = np.array([True, False, True, True])
    assert acceptance_rate_from_accepted(accepted) == pytest.approx(0.75)


def test_hf_call_fraction() -> None:
    used_hf = np.array([0, 1, 1, 0], dtype=int)
    assert hf_call_fraction(used_hf) == pytest.approx(0.5)


def test_mean_subchain_length() -> None:
    sub = np.array([2, 4, 6])
    assert mean_subchain_length(sub) == pytest.approx(4.0)


def test_posterior_rmse() -> None:
    samples = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    theta_true = np.array([0.0, 0.0])
    rmse = posterior_rmse(samples, theta_true, burn_in=1)
    # distances: [1, sqrt(2)] -> mean
    assert rmse == pytest.approx((1.0 + np.sqrt(2.0)) / 2.0)
