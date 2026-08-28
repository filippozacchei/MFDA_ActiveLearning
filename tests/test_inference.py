from __future__ import annotations

import copy

import numpy as np
import pytest

from gp_active_mcmc.inference.adaptive_subchain import (
    AdaptiveSubchain,
    AdaptiveSubchainControl,
    AdaptiveSubchainState,
)
from gp_active_mcmc.inference.coarse_output import CoarseOutput
from gp_active_mcmc.inference.likelihood import ActiveGPLogLike
from gp_active_mcmc.inference.model import ActiveMCMCModel
from gp_active_mcmc.inference.proposal import AdaptiveMetropolisShared


def _manual_gaussian_loglike(y: np.ndarray, m: np.ndarray, C: np.ndarray) -> float:
    """Reference implementation using the closed-form MVN log-likelihood, including the
    -0.5*log|C| normalizing term (up to the theta-independent -(n/2)*log(2*pi) constant
    ActiveGPLogLike also omits -- see its docstring for why that one's fine to drop but
    the log-determinant term isn't, given C varies with theta)."""
    y = np.asarray(y, dtype=float).ravel()
    m = np.asarray(m, dtype=float).ravel()
    C = np.asarray(C, dtype=float)

    sign, logdet = np.linalg.slogdet(C)
    if sign <= 0:
        raise ValueError("Covariance must be positive definite.")
    diff = y - m
    quad = float(diff @ np.linalg.solve(C, diff))
    return float(-0.5 * quad - 0.5 * logdet)


def test_likelihood_mean_only_uses_obs_cov() -> None:
    y_obs = np.array([0.2, -0.1, 0.05])
    C_obs = np.diag([0.1, 0.2, 0.3])

    like = ActiveGPLogLike(y_obs, C_obs)

    y_mean = np.array([0.25, -0.05, 0.0])
    ll = like.loglike(y_mean)

    ll_ref = _manual_gaussian_loglike(y_obs, y_mean, C_obs)
    np.testing.assert_allclose(ll, ll_ref, rtol=1e-12, atol=1e-12)

    np.testing.assert_allclose(like.total_cov, C_obs, rtol=0, atol=0)
    np.testing.assert_allclose(like.cov_inverse, np.linalg.inv(C_obs), rtol=1e-12, atol=1e-12)


def test_likelihood_adds_predictive_variance_diagonal() -> None:
    y_obs = np.array([0.0, 0.0, 0.0])
    C_obs = np.diag([0.1, 0.2, 0.3])
    like = ActiveGPLogLike(y_obs, C_obs)

    mean = np.array([0.0, 0.0, 0.0])
    var = np.array([0.01, 0.02, 0.03])
    pred = CoarseOutput(mean, var)

    ll = like.loglike(pred)

    C_total = C_obs + np.diag(var)
    ll_ref = _manual_gaussian_loglike(y_obs, mean, C_total)
    np.testing.assert_allclose(ll, ll_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(like.total_cov, C_total, rtol=0, atol=0)


def test_likelihood_cov_bias_is_added_if_present() -> None:
    y_obs = np.array([0.0, 0.0])
    C_obs = np.eye(2) * 0.2
    like = ActiveGPLogLike(y_obs, C_obs)

    like.cov_bias = np.eye(2) * 0.1  # optional attribute (tinyDA adaptation style)

    y_mean = np.array([0.0, 0.0])
    ll = like.loglike(y_mean)

    C_total = C_obs + like.cov_bias
    ll_ref = _manual_gaussian_loglike(y_obs, y_mean, C_total)
    np.testing.assert_allclose(ll, ll_ref, rtol=1e-12, atol=1e-12)


def test_likelihood_rejects_variance_wrong_shape() -> None:
    y_obs = np.array([0.0, 0.0, 0.0])
    C_obs = np.eye(3) * 0.1
    like = ActiveGPLogLike(y_obs, C_obs)

    pred_bad = CoarseOutput(np.zeros(3), np.ones(3))

    # Force a shape mismatch by monkeypatching variance (simulates user error / bad surrogate)
    pred_bad.variance = np.ones(2)

    with pytest.raises(ValueError, match="variance must have the same shape as mean"):
        _ = like.loglike(pred_bad)


def test_likelihood_rejects_negative_variance() -> None:
    y_obs = np.array([0.0, 0.0])
    C_obs = np.eye(2) * 0.2
    like = ActiveGPLogLike(y_obs, C_obs)

    # CoarseOutput itself might already validate this; test likelihood's guard too by patching
    pred = CoarseOutput(np.zeros(2), np.ones(2))
    pred.variance = np.array([0.1, -0.2])

    with pytest.raises(ValueError, match="variance must be non-negative"):
        _ = like.loglike(pred)


def test_likelihood_rejects_mean_length_mismatch() -> None:
    y_obs = np.array([1.0, 2.0, 3.0])
    C_obs = np.eye(3)
    like = ActiveGPLogLike(y_obs, C_obs)

    with pytest.raises(ValueError, match="does not match data length"):
        _ = like.loglike(np.array([1.0, 2.0]))


def test_likelihood_rejects_non_1d_mean() -> None:
    y_obs = np.array([1.0, 2.0])
    C_obs = np.eye(2)
    like = ActiveGPLogLike(y_obs, C_obs)

    with pytest.raises(ValueError, match="must be 1D"):
        _ = like.loglike(np.array([[1.0, 2.0]]))


def test_deepcopy_shares_instance_when_enabled() -> None:
    prop = AdaptiveMetropolisShared.__new__(AdaptiveMetropolisShared)
    prop._share_across_deepcopy = True
    prop.some_state = np.array([1.0, 2.0, 3.0])

    prop2 = copy.deepcopy(prop)

    assert prop2 is prop
    assert prop2.some_state is prop.some_state


def test_deepcopy_clones_instance_when_disabled() -> None:
    prop = AdaptiveMetropolisShared.__new__(AdaptiveMetropolisShared)
    prop._share_across_deepcopy = False
    prop.some_state = np.array([1.0, 2.0, 3.0])
    prop.nested = {"a": [1, 2], "b": np.array([0.0, 1.0])}

    prop2 = copy.deepcopy(prop)

    assert prop2 is not prop
    assert isinstance(prop2, AdaptiveMetropolisShared)

    # Arrays and nested structures should be deep-copied
    assert prop2.some_state is not prop.some_state
    np.testing.assert_allclose(prop2.some_state, prop.some_state)

    assert prop2.nested is not prop.nested
    assert prop2.nested["a"] is not prop.nested["a"]
    assert prop2.nested["b"] is not prop.nested["b"]
    np.testing.assert_allclose(prop2.nested["b"], prop.nested["b"])


def test_setup_proposal_preserves_am_recursor_across_repeated_calls() -> None:
    """Regression test: `tinyDA.Chain`/`DAChain` call `setup_proposal` unconditionally
    at construction, and this library's chunked round-based harness constructs a new
    one every round on the *same* (never-deep-copied) proposal object -- so
    `setup_proposal` fires every round too. The base `AdaptiveMetropolis.setup_proposal`
    unconditionally rebuilds `AM_recursor` from scratch each call; left unguarded, the
    Haario covariance-shape estimate would never accumulate past one round's samples,
    undermining `share_across_deepcopy`'s "adaptation continues across chunks" contract
    via a path `__deepcopy__` alone doesn't cover."""
    prop = AdaptiveMetropolisShared(C0=np.eye(2), period=100, adaptive=True, sd=1)
    prop.setup_proposal(parameters=np.array([1.0, 2.0]))
    recursor_after_first = prop.AM_recursor

    # Feed it a few updates, as `adapt()` would during real sampling.
    prop.AM_recursor.update(np.array([1.1, 2.1]))
    prop.AM_recursor.update(np.array([0.9, 1.9]))
    t_before_second_setup = prop.AM_recursor.t

    # A second setup_proposal call -- as happens every round in the chunked harness --
    # must NOT discard the accumulated moments.
    prop.setup_proposal(parameters=np.array([1.05, 2.05]))

    assert prop.AM_recursor is recursor_after_first
    assert prop.AM_recursor.t == t_before_second_setup


def test_adapt_preserves_acceptance_window_across_chain_resets() -> None:
    """Regression test: `tinyDA.Chain`/`DAChain` pass their own `accepted`/
    `accepted_coarse` list into `adapt()` -- a list that starts fresh (`[]`) every time
    a `Chain`/`DAChain` is constructed, i.e. every round in this library's chunked
    harness. Without tracking the window here instead, `GaussianRandomWalk.adapt`'s
    `kwargs["accepted"][-period:]` slice would draw on far fewer than `period` real
    outcomes right after a round boundary."""
    prop = AdaptiveMetropolisShared(C0=np.eye(2), period=10, adaptive=True, sd=1)
    prop.setup_proposal(parameters=np.array([1.0, 2.0]))

    # Round 1: a Chain-owned `accepted` list that grows to 5 entries, then gets
    # discarded (as happens when the next round constructs a fresh Chain).
    round1_accepted: list[bool] = []
    for outcome in [True, False, True, True, False]:
        round1_accepted.append(outcome)
        prop.adapt(parameters=np.zeros(2), parameters_previous=np.zeros(2), accepted=round1_accepted)

    # Round 2: a brand-new list, as a freshly-constructed Chain would pass.
    round2_accepted: list[bool] = []
    for outcome in [False, True, True, False, True]:
        round2_accepted.append(outcome)
        prop.adapt(parameters=np.zeros(2), parameters_previous=np.zeros(2), accepted=round2_accepted)

    # 10 calls total, period=10 -- the internal window must hold all 10 real outcomes,
    # not just round 2's 5 (what kwargs["accepted"] alone would have offered).
    assert list(prop._recent_accepted) == [True, False, True, True, False, False, True, True, False, True]


class DummySurrogate:
    def __init__(self, *, var_scale: float = 0.0):
        self.var_scale = float(var_scale)
        self.update_calls: list[tuple[np.ndarray, np.ndarray]] = []

    def predict(self, theta: np.ndarray):
        s = float(np.sum(theta))
        mean = np.array([s, s + 1.0], dtype=float)
        var = np.full_like(mean, self.var_scale, dtype=float)
        return mean, var

    def update(self, theta: np.ndarray, y: np.ndarray) -> None:
        self.update_calls.append((np.asarray(theta, dtype=float), np.asarray(y, dtype=float)))


class DummyHF:
    def __call__(self, theta: np.ndarray) -> np.ndarray:
        s = float(np.sum(theta))
        return np.array([2.0 * s, 2.0 * s + 1.0], dtype=float)


def test_coarse_returns_coarseoutput_when_uncertainty_small() -> None:
    lf = DummySurrogate(var_scale=1e-6)
    hf = DummyHF()
    model = ActiveMCMCModel(lf_model=lf, hf_model=hf, gamma_threshold=1e-2)

    out = model.coarse(np.array([1.0, 2.0]))
    assert isinstance(out, CoarseOutput)
    assert out.shape == (2,)
    assert hasattr(out, "variance")
    assert out.variance.shape == out.shape

    assert model.log.used_hf == [False]
    assert len(lf.update_calls) == 0


def test_coarse_escalates_to_hf_when_uncertainty_large() -> None:
    lf = DummySurrogate(var_scale=1.0)
    hf = DummyHF()
    model = ActiveMCMCModel(lf_model=lf, hf_model=hf, gamma_threshold=0.1)  # threshold^2 = 0.01

    out = model.coarse(np.array([1.0, 2.0]))
    np.testing.assert_allclose(out, np.array([6.0, 7.0]))  # sum=3 -> [6,7]

    assert model.log.used_hf == [True]
    assert len(lf.update_calls) == 1


def test_fine_replaces_last_flag_by_default() -> None:
    lf = DummySurrogate(var_scale=1e-6)
    hf = DummyHF()
    model = ActiveMCMCModel(lf_model=lf, hf_model=hf, gamma_threshold=1e-2)

    _ = model.coarse(np.array([1.0, 2.0]))
    assert model.log.used_hf == [False]

    _ = model.fine(np.array([1.0, 2.0]))  # upgrade
    assert model.log.used_hf == [True]
    assert len(lf.update_calls) == 1


def test_fine_appends_flag_when_replace_last_false() -> None:
    lf = DummySurrogate(var_scale=1e-6)
    hf = DummyHF()
    model = ActiveMCMCModel(lf_model=lf, hf_model=hf, gamma_threshold=1e-2)

    _ = model.coarse(np.array([1.0, 2.0]))
    assert model.log.used_hf == [False]

    _ = model.fine(np.array([1.0, 2.0]), replace_last=False)
    assert model.log.used_hf == [False, True]
    assert len(lf.update_calls) == 1


def test_adaptive_subchain_has_converged_delegates_to_state() -> None:
    state = AdaptiveSubchainState(subchain_length=10)
    control = AdaptiveSubchainControl(update_every=1, target_error=0.1, patience=2)
    hook = AdaptiveSubchain(state=state, control=control)

    assert hook.has_converged() is False

    for _ in range(2):
        state.hf_errors.append(0.01)
        state.step()
        state.update_subchain(control)

    assert hook.has_converged() is True


def test_frozen_coarse_never_escalates_to_hf() -> None:
    lf = DummySurrogate(var_scale=1.0)  # would normally escalate
    hf = DummyHF()
    model = ActiveMCMCModel(lf_model=lf, hf_model=hf, gamma_threshold=0.1, frozen=True)

    out = model.coarse(np.array([1.0, 2.0]))
    assert isinstance(out, CoarseOutput)
    assert model.log.used_hf == [False]
    assert len(lf.update_calls) == 0


def test_frozen_fine_evaluates_hf_but_does_not_update_surrogate() -> None:
    lf = DummySurrogate(var_scale=1e-6)
    hf = DummyHF()
    model = ActiveMCMCModel(lf_model=lf, hf_model=hf, gamma_threshold=0.1, frozen=True)

    y = model.fine(np.array([1.0, 2.0]))
    np.testing.assert_allclose(y, np.array([6.0, 7.0]))  # HF still evaluated
    assert model.log.used_hf == [True]
    assert len(lf.update_calls) == 0  # but surrogate untouched


def test_freeze_returns_independent_frozen_copy() -> None:
    lf = DummySurrogate(var_scale=1.0)
    hf = DummyHF()
    state = AdaptiveSubchainState(subchain_length=10)
    control = AdaptiveSubchainControl(update_every=1, target_error=0.01)
    hook = AdaptiveSubchain(state=state, control=control)

    model = ActiveMCMCModel(lf_model=lf, hf_model=hf, gamma_threshold=0.1, adaptive=hook)

    frozen = model.freeze()

    assert frozen.frozen is True
    assert frozen.adaptive is None
    assert frozen.lf_model is not model.lf_model  # deep-copied, not shared

    # Sampling with the frozen copy must not touch the original surrogate or its log.
    _ = frozen.coarse(np.array([1.0, 2.0]))
    assert len(lf.update_calls) == 0
    assert model.log.used_hf == []
    assert frozen.log.used_hf == [False]


def test_adaptive_hook_updates_state_on_coarse_and_fine() -> None:
    lf = DummySurrogate(var_scale=1e-6)
    hf = DummyHF()

    state = AdaptiveSubchainState(subchain_length=10)
    control = AdaptiveSubchainControl(update_every=1, target_error=0.01)
    hook = AdaptiveSubchain(state=state, control=control)

    model = ActiveMCMCModel(lf_model=lf, hf_model=hf, gamma_threshold=1e-2, adaptive=hook)

    _ = model.coarse(np.array([1.0, 2.0]))
    assert state.subchain_history == [10]
    assert state.total_hf_steps == 0

    _ = model.fine(np.array([1.0, 2.0]))
    assert state.total_hf_steps == 1
    assert len(state.hf_errors) == 1
    assert model.log.used_hf == [True]
    assert len(lf.update_calls) == 1
