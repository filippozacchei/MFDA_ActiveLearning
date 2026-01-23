# =========================
# Plotting utilities
# =========================
import matplotlib.pyplot as plt
import numpy as np

def plot_rmse_and_uncertainty(
    rmse_hist: np.ndarray,
    mean_std_hist: np.ndarray,
    fname: str | None = None,
):
    """
    Plot RMSE vs true solution and mean surrogate uncertainty.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(rmse_hist, label="RMSE vs θ_true")
    ax.plot(mean_std_hist, label="Mean surrogate std")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Error / Uncertainty")
    ax.set_title("Surrogate accuracy and uncertainty")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    plt.show()
        
def plot_prediction_at_theta(
    emul,
    theta: np.ndarray,
    t: np.ndarray,
    y_obs: np.ndarray,
    title: str,
    fname: str,
):
    """
    Plot surrogate prediction with uncertainty bands at a fixed theta.
    """
    y_hat, y_std = emul.predict(theta)

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

    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    plt.show()
    
def plot_chain_2d(
    chain: np.ndarray,
    used_forward: np.ndarray,
    theta_true: np.ndarray,
    names: tuple[str, str] = ("A", "f"),
    fname: str | None = None,
):
    """
    Plot chain in the first two parameter dimensions,
    distinguishing GP vs forward-evaluated samples.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    gp_idx = np.where(~used_forward)[0] + 1
    fw_idx = np.where(used_forward)[0] + 1

    if gp_idx.size > 0:
        ax.scatter(
            chain[gp_idx, 0],
            chain[gp_idx, 1],
            s=15,
            alpha=0.4,
            label="GP",
        )

    if fw_idx.size > 0:
        ax.scatter(
            chain[fw_idx, 0],
            chain[fw_idx, 1],
            s=40,
            marker="x",
            label="Forward",
        )

    ax.scatter(
        theta_true[0],
        theta_true[1],
        s=140,
        marker="*",
        c="k",
        edgecolors="w",
        label=r"$\theta_{\mathrm{true}}$",
    )

    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.set_title("Final chain (θ₁ vs θ₂)")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    plt.show()
