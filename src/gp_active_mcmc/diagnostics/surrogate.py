from __future__ import annotations

from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


class PredictMeanVar(Protocol):
    """Protocol for models that provide predictive mean and variance.

    This protocol is used by diagnostics/plotting utilities to accept any model
    that supports:

    - `predict(theta) -> (mean, variance)`

    where `mean` and `variance` are arrays in *observation space* (i.e. aligned with
    the time grid `t` used for plotting).

    Notes
    -----
    The model can be a POD-GP surrogate, a GP-only
    model, or any wrapper that exposes the same interface.
    """

    def predict(self, theta: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Predict mean and marginal variance in observation space.

        Parameters
        ----------
        theta
            Parameter vector of shape ``(n_dim,)``.

        Returns
        -------
        mean, variance
            Mean prediction and marginal (pointwise) variance, both 1D arrays of
            shape ``(n_obs,)`` aligned with the observation grid.
        """
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
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot a predictive mean with an uncertainty band at a fixed parameter value.

    This function is primarily intended for documentation and quick diagnostics:
    it visualises (i) observed data, (ii) the model's predictive mean, and (iii) a
    pointwise uncertainty band based on the model's predictive variance.

    Parameters
    ----------
    model
        Any object implementing `predict` with mean and variance output.
        The model must provide `predict(theta) -> (mean, variance)` in observation space.
    theta
        Parameter vector at which the prediction is evaluated.
    t
        1D grid for the observation/prediction (e.g. time). Must have the same length
        as the predicted mean and observation vectors.
    y_obs
        Observed data vector aligned with `t`. This is typically the noisy observation.
    y_true
        Optional "true" profile aligned with `t` (e.g. the noiseless HF output) for reference.
    title
        Optional plot title.
    z
        Width of the uncertainty band in standard deviations. The band is drawn as
        `mean ± z * std`, where `std = sqrt(variance)`.
    show
        If True, call `plt.show()` before returning. In library usage and documentation
        builds, leaving this False is recommended (MkDocs/Jupyter will render the figure).

    Returns
    -------
    fig, ax
        Matplotlib figure and axes. The caller may further customise or save them.

    Raises
    ------
    ValueError
        If `t`, `y_obs`, `mean`, and `variance` do not have the same 1D shape, or if
        `variance` contains negative entries.

    See Also
    --------
    [`plot_error_vs_uncertainty`][gp_active_mcmc.diagnostics.surrogate.plot_error_vs_uncertainty]
        Scatter plot assessing whether predicted uncertainty correlates with error.
    """
    theta_arr = np.asarray(theta, dtype=float)
    t_arr = np.asarray(t, dtype=float).ravel()
    y_obs_arr = np.asarray(y_obs, dtype=float).ravel()

    y_mean, y_var = model.predict(theta_arr)
    y_mean = np.asarray(y_mean, dtype=float).ravel()
    y_var = np.asarray(y_var, dtype=float).ravel()

    if y_mean.shape != t_arr.shape or y_obs_arr.shape != t_arr.shape or y_var.shape != t_arr.shape:
        raise ValueError("t, y_obs, y_mean, and y_var must all have the same 1D shape.")
    if np.any(y_var < 0.0):
        raise ValueError("y_var must be non-negative.")

    y_std = np.sqrt(y_var)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_arr, y_obs_arr, "k.", alpha=0.4, label="observations")
    ax.plot(t_arr, y_mean, lw=2, label="predictive mean")
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

    if show:
        plt.show()

    return fig, ax


def plot_error_vs_uncertainty(
    mean_std: ArrayLike,
    rmse_vals: ArrayLike,
    *,
    title: str | None = None,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot RMSE against predicted uncertainty.

    This diagnostic is a quick calibration check: if the model's predictive
    uncertainty is informative, larger predicted standard deviations should
    typically correspond to larger realised errors.

    Parameters
    ----------
    mean_std
        1D array of "typical" predictive standard deviations per test point, e.g.
        the mean of the pointwise predictive std across the observation grid.
    rmse_vals
        1D array of realised errors (RMSE) per test point.
    title
        Optional plot title.
    show
        If True, call `plt.show()` before returning.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.

    Raises
    ------
    ValueError
        If `mean_std` and `rmse_vals` do not have the same shape.
    """
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

    if show:
        plt.show()

    return fig, ax
