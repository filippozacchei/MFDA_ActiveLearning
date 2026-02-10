from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from gp_active_mcmc.diagnostics.surrogate import plot_error_vs_uncertainty, plot_prediction_at_theta
from gp_active_mcmc.diagnostics.mcmc import (
    plot_chain_2d,
    plot_cumulative_hf_fraction,
    plot_subchain_length_history,
    plot_surrogate_error_history,
)
from gp_active_mcmc.diagnostics.pod import pod_energy_from_snapshots, plot_pod_energy


class DummyModel:
    def predict(self, theta):
        t = np.linspace(0.0, 1.0, 5)
        mean = np.sin(t)
        var = 0.01 * np.ones_like(mean)
        return mean, var


def test_plot_error_vs_uncertainty_returns_fig_ax() -> None:
    fig, ax = plot_error_vs_uncertainty([0.1, 0.2], [0.3, 0.4])
    assert fig is not None and ax is not None

    with pytest.raises(ValueError):
        _ = plot_error_vs_uncertainty([0.1], [0.2, 0.3])


def test_plot_prediction_at_theta_returns_fig_ax() -> None:
    model = DummyModel()
    t = np.linspace(0.0, 1.0, 5)
    y_obs = np.zeros_like(t)
    fig, ax = plot_prediction_at_theta(model, theta=np.array([0.0, 0.0]), t=t, y_obs=y_obs)
    assert fig is not None and ax is not None


def test_plot_cumulative_hf_fraction_handles_empty() -> None:
    out = plot_cumulative_hf_fraction([], burnin=0)
    assert out is None

    out2 = plot_cumulative_hf_fraction(np.array([True, False, True]), burnin=1)
    assert out2 is not None


def test_plot_subchain_length_history_validation() -> None:
    assert plot_subchain_length_history([]) is None
    with pytest.raises(ValueError):
        _ = plot_subchain_length_history([0, 1, 2])


def test_plot_surrogate_error_history_validation() -> None:
    assert plot_surrogate_error_history([]) is None
    with pytest.raises(ValueError):
        _ = plot_surrogate_error_history([0.1, -0.2])


def test_plot_chain_2d_shapes() -> None:
    samples = np.random.default_rng(0).normal(size=(20, 2))
    fig, ax = plot_chain_2d(samples)
    assert fig is not None and ax is not None

    with pytest.raises(ValueError):
        _ = plot_chain_2d(np.random.normal(size=(20, 3)))

    with pytest.raises(ValueError):
        _ = plot_chain_2d(samples, used_hf=np.zeros(19))


def test_pod_energy_monotone() -> None:
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(20, 50))
    frac, cum = pod_energy_from_snapshots(Y)
    assert frac.ndim == 1 and cum.ndim == 1
    assert abs(cum[-1] - 1.0) < 1e-10
    assert np.all(np.diff(cum) >= -1e-12)


def test_plot_pod_energy_returns_fig_ax_pairs() -> None:
    rng = np.random.default_rng(1)
    Y = rng.normal(size=(10, 30))
    (fig1, ax1), (fig2, ax2) = plot_pod_energy(Y, r_max=10)
    assert fig1 is not None and ax1 is not None
    assert fig2 is not None and ax2 is not None
