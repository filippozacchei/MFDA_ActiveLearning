# %% [markdown]
# # HF Navier–Stokes surrogate demo (POD–GP)
#
# This notebook-style example shows how to build and validate a POD–GP surrogate for a
# **high-fidelity** Navier–Stokes quantity of interest (QoI).
#
# **QoI**
# : Outlet streamwise velocity profile \(u_x(y)\), resampled to a fixed length \(T\).
#
# **Inputs**
# : \(\theta = [h_1, U_{\mathrm{in}}]\)
#
# **Surrogate**
# : POD(rank=\(r\)) on profiles \(u_x(y)\), then a multi-output GP mapping
# \(\theta \mapsto \mathbf{a}\) (POD coefficients). Reconstruction yields the predicted profile.
#
# **Assumptions**
# - You have a FEniCSx/DOLFINx solver wrapper:
#   `hf_solver(h1: float, *, U_in: float, h2: float, L_up: float, L_down: float) -> (y, u_x)`.
# - `resample_profile(y, u, T=T)` returns a vector of length \(T\).
#
# This example is intentionally small (documentation-friendly runtime). For a usable surrogate,
# increase `N_SNAPSHOTS`, tune the mesh/time settings, and consider caching HF runs.

# %% Imports
from __future__ import annotations

from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split

from gp_active_mcmc.surrogates import MultiOutputGP, POD, PODGPSurrogate
from gp_active_mcmc.utils.rng import set_seed
from gp_active_mcmc.utils.metrics import rmse, coverage
from gp_active_mcmc.diagnostics.surrogate import plot_prediction_at_theta
from gp_active_mcmc.diagnostics.pod import plot_pod_energy

# Local Navier–Stokes utilities (example-specific; not part of gp_active_mcmc)
from utils.outlet import resample_profile
from utils.solver import forward_model as hf_solver


# %% [markdown]
# ## Configuration
#
# The prior below is only used as a *sampling distribution* to generate an initial dataset.
# It is not yet an inference setup (no likelihood, no MCMC here).

# %%
SEED = 7
rng = set_seed(SEED)

# QoI
T = 100  # outlet profile length after resampling

# Dataset size (tune based on HF cost)
N_SNAPSHOTS = 50
TEST_SIZE = 0.5
RANDOM_STATE = 0

# POD/GP
POD_RANK = 5
GP_KERNEL = "matern52"
USE_ARD = True
N_RETRAIN_MAX = 0
UPDATE_EVERY = 25

# Fixed solver geometry parameters (keep fixed unless you want higher input dimension)
H2 = 0.20
L_UP = 0.10
L_DOWN = 0.40
U_IN_DEFAULT = 1.5

# Input bounds (support of the sampling distribution)
H1_MIN, H1_MAX = 0.05, 0.15
U_MIN, U_MAX = 0.25, 1.25
L_MIN, L_MAX = 0.3, 0.5

# %% [markdown]
# ## Sampling distribution for \(\theta = [h_1, U_{\mathrm{in}}]\)
#
# We use a Gaussian restricted to a plausible range *by rejection*. This is a pragmatic
# choice for dataset generation; you can replace it with Latin Hypercube sampling or a
# bounded distribution if preferred.

# %%
theta_mean = np.array(
    [0.5 * (H1_MIN + H1_MAX), 0.5 * (U_MIN + U_MAX), 0.5 * (L_MIN + L_MAX)], dtype=float
)
theta_sig = np.array(
    [0.25 * (H1_MAX - H1_MIN), 0.25 * (U_MAX - U_MIN), 0.25 * (L_MAX - L_MIN)], dtype=float
)
theta_cov = np.diag(theta_sig**2)

prior = multivariate_normal(mean=theta_mean, cov=theta_cov)


def _sample_theta_bounded(
    *,
    prior: multivariate_normal,
    n: int,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    rng: np.random.Generator,
    max_tries: int = 100_000,
) -> np.ndarray:
    """Sample n points from a Gaussian prior, rejecting outside bounds.

    Parameters
    ----------
    prior
        SciPy multivariate normal used for proposal samples.
    n
        Number of accepted samples.
    bounds
        ((h1_min, h1_max), (u_min, u_max)).
    rng
        NumPy Generator.
    max_tries
        Hard cap to avoid infinite loops.

    Returns
    -------
    X
        Array of shape (n, 2).
    """
    (h1_min, h1_max), (u_min, u_max), (l_min, l_max) = bounds
    X = np.empty((n, 3), dtype=float)

    accepted = 0
    tries = 0
    while accepted < n:
        tries += 1
        if tries > max_tries:
            raise RuntimeError(
                "Too many rejections when sampling bounded prior. "
                "Consider widening the Gaussian or using a bounded design."
            )
        th = np.asarray(prior.rvs(random_state=rng), dtype=float).ravel()
        if th.shape != (3,):
            continue
        h1, u_in, l_do = float(th[0]), float(th[1]), float(th[2])
        if (h1_min <= h1 <= h1_max) and (u_min <= u_in <= u_max) and (l_min <= l_do <= l_max):
            X[accepted] = th
            accepted += 1

    return X


# %% [markdown]
# ## Dataset generation
#
# We build snapshot pairs:
#
# - inputs: \(X_i = [h_{1,i}, U_{\mathrm{in}, i}]\),
# - outputs: \(Y_i = u_x(y)\) sampled at the outlet and resampled to length \(T\).


# %%
def generate_dataset(
    *,
    solver: Callable[..., tuple[np.ndarray, np.ndarray]],
    prior: multivariate_normal,
    n: int,
    T: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a training dataset for the outlet-profile surrogate.

    Returns
    -------
    X
        Array of shape (n, 2) with columns [h1, U_in].
    Y
        Array of shape (n, T) with resampled outlet profiles.
    """
    X = _sample_theta_bounded(
        prior=prior,
        n=n,
        bounds=((H1_MIN, H1_MAX), (U_MIN, U_MAX), (L_MIN, L_MAX)),
        rng=rng,
    )

    Y = np.zeros((n, T), dtype=float)
    for i in range(n):
        h1, u_in, l_do = float(X[i, 0]), float(X[i, 1]), float(X[i, 2])
        y, u = solver(h1, U_in=u_in, h2=H2, L_up=L_UP, L_down=l_do)
        Y[i, :] = resample_profile(y, u, T=T)

    return X, Y


# %% [markdown]
# ## Build POD–GP surrogate
#
# - Fit POD on training profiles \(Y_{\mathrm{train}}\).
# - Train a multi-output GP on POD coefficients \(A_{\mathrm{train}}\).
# - Wrap into `PODGPSurrogate` to expose `predict(theta)`.


# %%
def build_surrogate(*, X_tr: np.ndarray, Y_tr: np.ndarray) -> PODGPSurrogate:
    pod = POD(rank=POD_RANK).fit(Y_tr)
    A_tr = pod.transform(Y_tr)[:, :POD_RANK]

    gp = MultiOutputGP(
        X_train=X_tr,
        Y_train=A_tr,
        kernel=GP_KERNEL,
        ard=USE_ARD,
        noise_variance=1e-6,
        update_every=UPDATE_EVERY,
        n_retrain_max=N_RETRAIN_MAX,
    )

    return PODGPSurrogate(pod=pod, gp=gp)


# %% [markdown]
# ## Evaluate predictive accuracy and calibration
#
# The surrogate returns `(mean, var)` for the profile. We report:
#
# - per-profile RMSE, and
# - empirical coverage for Gaussian predictive intervals at 50%, 90%, and 95%.


# %%
def evaluate_surrogate(
    *,
    surrogate: PODGPSurrogate,
    X_te: np.ndarray,
    Y_te: np.ndarray,
) -> dict[str, float]:
    z50, z90, z95 = 0.67449, 1.64485, 1.95996

    n_test = X_te.shape[0]
    test_rmse = np.zeros(n_test)
    mean_pred_std = np.zeros(n_test)
    cov50 = np.zeros(n_test)
    cov90 = np.zeros(n_test)
    cov95 = np.zeros(n_test)

    for i in range(n_test):
        y_hat, y_var = surrogate.predict(X_te[i])  # (T,), (T,)
        y_std = np.sqrt(np.maximum(y_var, 1e-14))
        y_true = Y_te[i]

        test_rmse[i] = rmse(y_hat, y_true)
        mean_pred_std[i] = float(np.mean(y_std))

        cov50[i] = coverage(y_true, y_hat, y_std, z=z50)
        cov90[i] = coverage(y_true, y_hat, y_std, z=z90)
        cov95[i] = coverage(y_true, y_hat, y_std, z=z95)

    return {
        "rmse_mean": float(test_rmse.mean()),
        "rmse_median": float(np.median(test_rmse)),
        "mean_uncertainty": float(mean_pred_std.mean()),
        "coverage_50": float(cov50.mean()),
        "coverage_90": float(cov90.mean()),
        "coverage_95": float(cov95.mean()),
        "idx_best": float(int(np.argmin(test_rmse))),
        "idx_median": float(int(np.argsort(test_rmse)[len(test_rmse) // 2])),
        "idx_worst": float(int(np.argmax(test_rmse))),
    }


# %% [markdown]
# ## Main


# %%
def main() -> None:
    # --------------------------------------------------------------
    # Dataset
    # --------------------------------------------------------------
    X, Y = generate_dataset(solver=hf_solver, prior=prior, n=N_SNAPSHOTS, T=T, rng=rng)

    X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    plot_pod_energy(Y_tr, r_max=len(Y_tr))

    # --------------------------------------------------------------
    # Surrogate
    # --------------------------------------------------------------
    surrogate = build_surrogate(X_tr=X_tr, Y_tr=Y_tr)

    # --------------------------------------------------------------
    # Metrics
    # --------------------------------------------------------------
    metrics = evaluate_surrogate(surrogate=surrogate, X_te=X_te, Y_te=Y_te)

    print("HF POD–GP surrogate for outlet profile")
    print(f"inputs: [h1, U_in], fixed (h2={H2}, L_up={L_UP}, L_down={L_DOWN})")
    print(f"POD rank r = {POD_RANK}")
    print(f"N_train = {X_tr.shape[0]}, N_test = {X_te.shape[0]}, T = {T}")
    for k in [
        "rmse_mean",
        "rmse_median",
        "mean_uncertainty",
        "coverage_50",
        "coverage_90",
        "coverage_95",
    ]:
        print(f"{k:>18s} : {metrics[k]:.6f}")

    # --------------------------------------------------------------
    # Visual inspection: best / median / worst
    # --------------------------------------------------------------
    t_plot = np.arange(T, dtype=float)
    idx_best = int(metrics["idx_best"])
    idx_median = int(metrics["idx_median"])
    idx_worst = int(metrics["idx_worst"])

    for label, idx in [("best", idx_best), ("median", idx_median), ("worst", idx_worst)]:
        theta = X_te[idx]
        y_true = Y_te[idx]
        plot_prediction_at_theta(
            surrogate,
            theta,
            t_plot,
            y_true,
            title=(
                "HF POD–GP outlet profile — "
                f"{label} test case (h1={theta[0]:.4f}, U_in={theta[1]:.3f})"
            ),
            show=True,
        )


if __name__ == "__main__":
    main()
