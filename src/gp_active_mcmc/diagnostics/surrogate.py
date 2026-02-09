from __future__ import annotations

from typing import Protocol

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from numpy.typing import ArrayLike


class PredictMeanVar(Protocol):
    def predict(self, theta: ArrayLike):
        """Return (mean, variance) in observation space."""
        ...


def plot_prediction_at_theta(
    model: PredictMeanVar,
    theta: ArrayLike,
    t: ArrayLike,
    y_obs: ArrayLike,
    *,
    y_true: ArrayLike | None = None,
    title: str | None = None,
    z: float = 2.0,
) -> tuple[Figure, Axes]:
    """Plot surrogate prediction at a given parameter with uncertainty band."""
    theta_arr = np.asarray(theta, dtype=float)
    t_arr = np.asarray(t, dtype=float).ravel()
    y_obs_arr = np.asarray(y_obs, dtype=float).ravel()

    y_mean, y_var = model.predict(theta_arr)
    y_mean = np.asarray(y_mean, dtype=float).ravel()
    y_var = np.asarray(y_var, dtype=float).ravel()

    if y_mean.shape != t_arr.shape or y_obs_arr.shape != t_arr.shape or y_var.shape != t_arr.shape:
        raise ValueError("t, y_obs, y_mean, and y_var must all have the same 1D shape.")
    if np.any(y_var < 0):
        raise ValueError("y_var must be non-negative.")

    y_std = np.sqrt(y_var)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_arr, y_obs_arr, "k.", alpha=0.4, label="observations")
    ax.plot(t_arr, y_mean, lw=2, label="surrogate mean")
    ax.fill_between(
        t_arr,
        y_mean - z * y_std,
        y_mean + z * y_std,
        alpha=0.25,
        label=rf"$\pm {z}\sigma$",
    )

    if y_true is not None:
        y_true_arr = np.asarray(y_true, dtype=float).ravel()
        if y_true_arr.shape != t_arr.shape:
            raise ValueError("y_true must have the same shape as t.")
        ax.plot(t_arr, y_true_arr, lw=2, label="true profile")

    ax.set_xlabel("t")
    ax.set_ylabel("y")
    if title is not None:
        ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig, ax


def plot_error_vs_uncertainty(
    mean_std: ArrayLike,
    rmse_vals: ArrayLike,
    *,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Scatter plot of error (RMSE) vs predicted uncertainty (mean std)."""
    x = np.asarray(mean_std, dtype=float).ravel()
    y = np.asarray(rmse_vals, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError("mean_std and rmse_vals must have the same shape.")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, alpha=0.85)
    ax.set_xlabel("Mean predictive standard deviation")
    ax.set_ylabel("RMSE")
    if title is not None:
        ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    return fig, ax
