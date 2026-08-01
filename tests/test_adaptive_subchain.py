from __future__ import annotations

import numpy as np
import pytest

from gp_active_mcmc.inference.adaptive_subchain import (
    AdaptiveSubchainControl,
    AdaptiveSubchainState,
)


def test_adaptive_state_append_length_and_step() -> None:
    state = AdaptiveSubchainState(subchain_length=10)

    state.append_length()
    state.append_length()
    assert state.subchain_history == [10, 10]

    assert state.total_hf_steps == 0
    state.step()
    state.step()
    assert state.total_hf_steps == 2


def test_append_error_computes_rmse() -> None:
    state = AdaptiveSubchainState(subchain_length=10)

    lf = np.array([0.0, 1.0, 2.0])
    hf = np.array([0.0, 2.0, 2.0])
    state.append_error(lf, hf)

    # rmse = sqrt(mean([0,1,0])) = sqrt(1/3)
    assert len(state.hf_errors) == 1
    np.testing.assert_allclose(state.hf_errors[0], np.sqrt(1.0 / 3.0), rtol=1e-12, atol=1e-12)


def test_append_error_rejects_shape_mismatch() -> None:
    state = AdaptiveSubchainState(subchain_length=10)
    with pytest.raises(ValueError, match="same shape"):
        state.append_error(np.zeros(3), np.zeros(2))


def test_update_subchain_shrinks_when_error_high() -> None:
    control = AdaptiveSubchainControl(
        update_every=2,
        target_error=0.1,
        min_subchain=1,
        max_subchain=100,
        grow_factor=2.0,
        shrink_factor=0.5,
    )
    state = AdaptiveSubchainState(subchain_length=10)

    # two HF steps to trigger update
    state.hf_errors.append(1.0)  # above target
    state.step()
    state.step()
    state.update_subchain(control)

    assert state.subchain_length == 5  # 10 * 0.5
    assert state._hf_since_update == 0


def test_update_subchain_grows_when_error_low() -> None:
    control = AdaptiveSubchainControl(
        update_every=2,
        target_error=1.0,
        min_subchain=1,
        max_subchain=100,
        grow_factor=2.0,
        shrink_factor=0.5,
    )
    state = AdaptiveSubchainState(subchain_length=10)

    state.hf_errors.append(0.1)  # below target
    state.step()
    state.step()
    state.update_subchain(control)

    assert state.subchain_length == 20  # 10 * 2.0
    assert state._hf_since_update == 0


def test_update_subchain_tracks_stable_streak_and_resets_on_miss() -> None:
    control = AdaptiveSubchainControl(update_every=1, target_error=0.1, patience=3)
    state = AdaptiveSubchainState(subchain_length=10)

    for err in (0.05, 0.05):  # below target: streak grows
        state.hf_errors.append(err)
        state.step()
        state.update_subchain(control)
    assert state.stable_streak == 2
    assert state.n_updates == 2
    assert not state.has_converged(control)

    state.hf_errors.append(1.0)  # above target: streak resets
    state.step()
    state.update_subchain(control)
    assert state.stable_streak == 0
    assert state.n_updates == 3
    assert not state.has_converged(control)


def test_has_converged_true_once_patience_reached() -> None:
    control = AdaptiveSubchainControl(update_every=1, target_error=0.1, patience=3)
    state = AdaptiveSubchainState(subchain_length=10)

    for _ in range(3):
        state.hf_errors.append(0.01)  # below target
        state.step()
        state.update_subchain(control)

    assert state.stable_streak == 3
    assert state.has_converged(control)


def test_patience_must_be_positive() -> None:
    with pytest.raises(ValueError, match="patience must be positive"):
        AdaptiveSubchainControl(patience=0)


def test_update_subchain_respects_bounds() -> None:
    control = AdaptiveSubchainControl(
        update_every=1,
        target_error=0.0,
        min_subchain=3,
        max_subchain=8,
        grow_factor=10.0,
        shrink_factor=0.1,
    )
    state = AdaptiveSubchainState(subchain_length=5)

    # err <= target => grow, but bounded by max_subchain
    state.hf_errors.append(0.0)
    state.step()
    state.update_subchain(control)
    assert state.subchain_length == 8

    # err > target => shrink, but bounded by min_subchain
    state.hf_errors.append(1.0)
    state.step()
    state.update_subchain(control)
    assert state.subchain_length == 3
