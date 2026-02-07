# %% [markdown]
# # HF surrogate demo (POD–GP) — outlet profile with 2 inputs (h1, U_in)
#
# This mirrors the style of your POD–GP toy demo, but for the **HF Navier–Stokes**
# outlet profile QoI:
# - inputs: theta = [h1, U_in]
# - output: outlet streamwise velocity profile u_x(y), resampled to T=100 points
# - surrogate: POD(rank=5) + GP on POD coefficients (MultiOutputGP)
#
# Assumption
# ----------
# Your HF solver provides:
#   forward_model(h1: float, U_in: float) -> (y: array, u: array)

# %% Imports
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split

from gp_active_mcmc.gp import MultiOutputGP
from gp_active_mcmc.pod import POD
from gp_active_mcmc.podgp import PODGPSurrogate
from gp_active_mcmc.utils.rng import set_seed

from gp_active_mcmc.diagnostics.surrogate import plot_prediction_at_theta
from gp_active_mcmc.diagnostics.metrics import rmse, coverage
import matplotlib.pyplot as plt

# %% [markdown]
# ## Configuration

# %%
SEED = 7
rng = set_seed(SEED)

# QoI
T = 100  # outlet profile length after resampling

# Dataset
N_SNAPSHOTS = 60  # tune based on HF cost

# POD/GP
POD_RANK = 5
GP_KERNEL = "matern52"
USE_ARD = True
N_RETRAIN_MAX = 0
UPDATE_EVERY = 25

# Train/test
TEST_SIZE = 0.25
RANDOM_STATE = 0

# Input bounds (sampling support)
H1_MIN, H1_MAX = 0.05, 0.15
U_MIN, U_MAX = 0.5, 1.5

# %% [markdown]
# ## Utilities

# %%
def resample_profile(y: np.ndarray, u: np.ndarray, *, T: int) -> np.ndarray:
    """Resample u(y) onto a uniform grid on [min(y), max(y)] of length T."""
    y = np.asarray(y).ravel()
    u = np.asarray(u).ravel()
    if y.size != u.size:
        raise ValueError("y and u must have same length")
    if y.size < 2:
        raise ValueError("Need at least two points to resample")

    if not np.all(np.diff(y) >= 0):
        idx = np.argsort(y)
        y = y[idx]
        u = u[idx]

    y_new = np.linspace(float(y.min()), float(y.max()), T)
    return np.interp(y_new, y, u)


def make_prior_box_gaussian(
    *, h1_min: float, h1_max: float, u_min: float, u_max: float
) -> multivariate_normal:
    """Independent Gaussian prior roughly matching the box via ±2σ ≈ half-range."""
    mean = np.array([0.5 * (h1_min + h1_max), 0.5 * (u_min + u_max)])
    sig = np.array([0.25 * (h1_max - h1_min), 0.25 * (u_max - u_min)])
    cov = np.diag(sig**2)
    return multivariate_normal(mean=mean, cov=cov)


def clip_box(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    X[:, 0] = np.clip(X[:, 0], H1_MIN, H1_MAX)
    X[:, 1] = np.clip(X[:, 1], U_MIN, U_MAX)
    return X


def generate_dataset(
    *,
    solver: Callable[[float, float], tuple[np.ndarray, np.ndarray]],
    prior: multivariate_normal,
    n: int,
    T: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    X : (n,2) with columns [h1, U_in]
    Y : (n,T) outlet profiles
    """
    X = np.asarray(prior.rvs(size=n, random_state=rng), dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    X = clip_box(X)

    Y = np.zeros((n, T), dtype=float)
    for i in range(n):
        h1, u_in = float(X[i, 0]), float(X[i, 1])
        y, u = solver(h1, U_in=u_in)
        Y[i, :] = resample_profile(y, u, T=T)
    return X, Y


# %% [markdown]
# ## Main (POD–GP)

# %%
def main() -> None:
    # --- import HF solver here (adjust path) ------------------------------
    from utils.hf_boussinesq import forward_model as hf_solver  # noqa: WPS433

    prior = make_prior_box_gaussian(
        h1_min=H1_MIN, h1_max=H1_MAX, u_min=U_MIN, u_max=U_MAX
    )

    # --------------------------------------------------------------
    # Dataset generation
    # --------------------------------------------------------------
    X, Y = generate_dataset(solver=hf_solver, prior=prior, n=N_SNAPSHOTS, T=T, rng=rng)

    # --------------------------------------------------------------
    # Train–test split
    # --------------------------------------------------------------
    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # --------------------------------------------------------------
    # POD + GP surrogate construction
    # --------------------------------------------------------------
    pod = POD(r=POD_RANK).fit(Y_tr)
    A_tr = pod.project(Y_tr)[:, :POD_RANK]

    gps = MultiOutputGP(
        X_train=X_tr,
        Y_train=A_tr,
        kernel=GP_KERNEL,
        ard=USE_ARD,
        n_retrain_max=N_RETRAIN_MAX,
        update_every=UPDATE_EVERY,
    )
    emul = PODGPSurrogate(pod=pod, gp=gps)

    # --------------------------------------------------------------
    # Test-set performance and calibration
    # --------------------------------------------------------------
    z50, z90, z95 = 0.67449, 1.64485, 1.95996

    n_test = X_te.shape[0]
    test_rmse = np.zeros(n_test)
    mean_pred_std = np.zeros(n_test)
    cov50 = np.zeros(n_test)
    cov90 = np.zeros(n_test)
    cov95 = np.zeros(n_test)

    for i in range(n_test):
        y_hat, y_var = emul.predict(X_te[i])   # (T,), (T,)
        y_std = np.sqrt(np.maximum(y_var, 1e-14))
        y_true = Y_te[i]

        test_rmse[i] = rmse(y_hat, y_true)
        mean_pred_std[i] = float(np.mean(y_std))

        cov50[i] = coverage(y_true, y_hat, y_std, z=z50)
        cov90[i] = coverage(y_true, y_hat, y_std, z=z90)
        cov95[i] = coverage(y_true, y_hat, y_std, z=z95)

    metrics = {
        "rmse_mean": float(test_rmse.mean()),
        "rmse_median": float(np.median(test_rmse)),
        "mean_uncertainty": float(mean_pred_std.mean()),
        "coverage_50": float(cov50.mean()),
        "coverage_90": float(cov90.mean()),
        "coverage_95": float(cov95.mean()),
    }

    print("HF POD–GP surrogate (no co-kriging)")
    print(f"POD rank r = {POD_RANK}")
    print(f"N_train = {X_tr.shape[0]}, N_test = {X_te.shape[0]}, T = {T}, input_dim = 2")
    for k, v in metrics.items():
        print(f"{k:>18s} : {v:.6f}")

    # --------------------------------------------------------------
    # Representative cases (best / median / worst)
    # --------------------------------------------------------------
    idx_best = int(np.argmin(test_rmse))
    idx_worst = int(np.argmax(test_rmse))
    idx_median = int(np.argsort(test_rmse)[len(test_rmse) // 2])

    # dummy abscissa for plotting (outlet sample index)
    t_plot = np.arange(T)

    for label, idx in [("best", idx_best), ("median", idx_median), ("worst", idx_worst)]:
        theta = X_te[idx]
        y_true = Y_te[idx]
        plot_prediction_at_theta(
            emul,
            theta,
            t_plot,
            y_true,
            title=f"HF POD–GP — {label} test case (h1={theta[0]:.4f}, U_in={theta[1]:.3f})",
        )

    # --------------------------------------------------------------
    # One train vs one test overlay (simple)
    # --------------------------------------------------------------
    def plot_truth_pred(theta: np.ndarray, y_true: np.ndarray, *, tag: str) -> None:
        y_hat, _ = emul.predict(theta)
        plt.figure()
        plt.plot(y_true, label="truth")
        plt.plot(y_hat, "--", label="pred")
        plt.title(f"HF profile — {tag} (h1={theta[0]:.4f}, U_in={theta[1]:.3f})")
        plt.xlabel("outlet sample index")
        plt.ylabel(r"$u_x$")
        plt.legend()
        plt.tight_layout()
        plt.show()

    plot_truth_pred(X_tr[0], Y_tr[0], tag="train example")
    plot_truth_pred(X_te[0], Y_te[0], tag="test example")


if __name__ == "__main__":
    main()
