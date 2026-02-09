from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_prediction_at_theta(
    emul,
    theta: np.ndarray,
    t: np.ndarray,
    y_obs: np.ndarray,
    *,
    y_true: np.ndarray | None = None,
    title: str | None = None,
) -> None:
    """Surrogate prediction with ±2σ band."""
    y_hat, y_var = emul.predict(theta)
    y_std = np.sqrt(y_var)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, y_obs, "k.", alpha=0.4, label="observations")
    ax.plot(t, y_hat, lw=2, label="surrogate mean")
    ax.fill_between(
        t,
        y_hat - 2.0 * y_std,
        y_hat + 2.0 * y_std,
        alpha=0.25,
        label=r"$\pm 2\sigma$",
    )
    if isinstance(y_true,np.ndarray):
        ax.plot(t,y_true,lw=2, label="true profile")

    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    plt.show()


def plot_error_vs_uncertainty(
    mean_std: np.ndarray,
    rmse_vals: np.ndarray,
) -> None:
    """Error vs predicted uncertainty."""
    plt.figure()
    plt.scatter(mean_std, rmse_vals, alpha=0.85)
    plt.xlabel("Mean predictive standard deviation")
    plt.ylabel("RMSE")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True)
    plt.show()

    corr = np.corrcoef(mean_std, rmse_vals)[0, 1]
    print("Correlation =", float(corr))


def binned_reliability(
    mean_std: np.ndarray,
    rmse_vals: np.ndarray,
    *,
    n_bins: int = 5,
) -> None:
    """Binned reliability curve."""
    idx = np.argsort(mean_std)
    u_sorted = mean_std[idx]
    e_sorted = rmse_vals[idx]

    bins = np.array_split(np.arange(len(mean_std)), n_bins)
    u_bin = [u_sorted[b].mean() for b in bins]
    e_bin = [e_sorted[b].mean() for b in bins]

    plt.figure()
    plt.plot(u_bin, e_bin, marker="o")
    plt.xlabel("Mean predicted uncertainty (bin)")
    plt.ylabel("Mean RMSE (bin)")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True)
    plt.show()
