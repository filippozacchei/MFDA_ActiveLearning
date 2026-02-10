# %% [markdown]
# # HF Navier–Stokes surrogate demo (POD–GP)
#
# This tutorial builds and validates a POD–GP surrogate for a **high-fidelity**
# Navier–Stokes quantity of interest (QoI): the outlet streamwise velocity profile
# \(u_x(y)\), resampled to a fixed length \(T\).
#
# ## FEniCSx / DOLFINx installation (example-specific)
#
# This tutorial depends on FEniCSx (DOLFINx) and its HPC stack (MPI + PETSc), plus Gmsh.
# A typical approach is to use a dedicated environment:
#
# **Conda (recommended for reproducibility)**
# - Create an environment that provides: `dolfinx`, `petsc4py`, `mpi4py`, `gmsh`, `pyvista`
# - Ensure MPI is consistent across packages (OpenMPI or MPICH).
#
# **System packages**
# - If using system MPI/PETSc, ensure `petsc4py` is compiled against the same PETSc.
#
# > The Navier–Stokes example is *not* part of the core `gp_active_mcmc` library API.
# > It is a heavy tutorial intended for advanced usage and requires a working PDE stack.
#
# ## Animation
#
# When run interactively, this notebook also generates a velocity-magnitude animation
# (GIF/MP4) and saves it under `docs/assets/navier-stokes/`.
# During MkDocs builds, expensive PDE runs are skipped and pre-generated assets are embedded.


# %% Imports
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split

from gp_active_mcmc.surrogates import MultiOutputGP, POD, PODGPSurrogate
from gp_active_mcmc.utils.rng import set_seed
from gp_active_mcmc.utils.metrics import rmse, coverage
from gp_active_mcmc.diagnostics.surrogate import plot_prediction_at_theta

# Example-specific utilities
from examples.navier_stokes.utils.solver import (
    solve_ipcs_bfs,
    forward_model as hf_solver,
    MFTimeConfig,
)
from examples.navier_stokes.utils.types import BFSGeometry
from examples.navier_stokes.utils.animation import FieldSampleGrid, make_velocity_animation

# %% [markdown]
# ## Runtime control
#
# This notebook can be executed by MkDocs during documentation builds. Since FEniCSx runs
# are expensive and environment-dependent, we **skip** the heavy parts when building docs.
#
# - Interactive run: `RUN_EXPENSIVE=True` → generate figures + animation and save assets.
# - MkDocs run: set `MKDOCS_BUILD=1` → `RUN_EXPENSIVE=False` → embed existing assets.

# %%
RUN_EXPENSIVE = True

# Where to save assets that the docs will embed
ASSET_DIR = Path("docs/assets/navier-stokes")
ASSET_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Configuration

# %%
rng = set_seed(7)

T = 100  # outlet profile length after resampling
N_SNAPSHOTS = 50  # increase for accuracy (HF cost!)
TEST_SIZE = 0.5
RANDOM_STATE = 0

POD_RANK = 5
GP_KERNEL = "matern52"
USE_ARD = True

# Standard geometry / BC setup (adjust as needed)
geom = BFSGeometry(h1=0.10, h2=0.20, L_up=0.10, L_down=0.40)
U_IN_DEFAULT = 1.5

# A shorter run for tutorial assets
time_cfg = MFTimeConfig(dt=1e-3, t_end=2.0, progress=True)

# Sampling support for training data (bounded via rejection sampling)
H1_MIN, H1_MAX = 0.05, 0.15
U_MIN, U_MAX = 0.5, 1.5
L_MIN, L_MAX = 0.3, 0.5

theta_mean = np.array(
    [0.5 * (H1_MIN + H1_MAX), 0.5 * (U_MIN + U_MAX), 0.5 * (L_MIN + L_MAX)], dtype=float
)
theta_sig = np.array(
    [0.25 * (H1_MAX - H1_MIN), 0.25 * (U_MAX - U_MIN), 0.25 * (L_MAX - L_MIN)], dtype=float
)
theta_cov = np.diag(theta_sig**2)
prior = multivariate_normal(mean=theta_mean, cov=theta_cov)

# %% [markdown]
# ## HF solver wrapper (outlet profile resampled to length T)
#
# We call the IPCS solver and post-process the outlet velocity profile to a fixed-length vector.

# ### Velocity animation (standard geometry)
#
# ![](../assets/navier-stokes/velocity.gif)

if RUN_EXPENSIVE:
    prof, mesh, L, H, frames = solve_ipcs_bfs(
        geom=geom,
        U_in=U_IN_DEFAULT,
        time=time_cfg,
        store_velocity_frames=True,
        frame_stride=5,
    )

    grid = FieldSampleGrid(
        x=np.linspace(0.0, L, 240),
        y=np.linspace(0.0, H, 120),
    )

    out_gif = ASSET_DIR / "velocity.gif"
    make_velocity_animation(
        mesh=mesh,
        frames=frames,
        grid=grid,
        outpath=out_gif,
        interval_ms=80,
        title="BFS: velocity magnitude",
    )
    print(f"Saved animation to {out_gif}")

# %% [markdown]
# ## Bounded sampling (for dataset generation)


# %%
def sample_theta_bounded(*, n: int, rng: np.random.Generator) -> np.ndarray:
    X = np.empty((n, 3), dtype=float)
    accepted = 0
    tries = 0
    while accepted < n:
        tries += 1
        th = np.asarray(prior.rvs(random_state=rng), dtype=float).ravel()
        if th.shape != (2,):
            continue
        h1, u, l_do = float(th[0]), float(th[1]), float(th[2])
        if (H1_MIN <= h1 <= H1_MAX) and (U_MIN <= u <= U_MAX) and (L_MIN <= l_do <= L_MAX):
            X[accepted] = th
            accepted += 1
        if tries > 200_000:
            raise RuntimeError("Too many rejections; widen the Gaussian or use a bounded design.")
    return X


# %% [markdown]
# ## Dataset generation (expensive)
#
# During MkDocs builds this cell is skipped; the docs will show pre-generated assets instead.

# %%
if RUN_EXPENSIVE:
    X = sample_theta_bounded(n=N_SNAPSHOTS, rng=rng)
    Y = np.zeros((N_SNAPSHOTS, T), dtype=float)
    for i in range(N_SNAPSHOTS):
        Y[i, :] = hf_solver(h1=X[i, 0], U_in=X[i, 1], L_down=X[i, 2])

    X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# %% [markdown]
# ## POD–GP surrogate (training)

# %%
if RUN_EXPENSIVE:
    pod = POD(rank=POD_RANK).fit(Y_tr)
    A_tr = pod.transform(Y_tr)[:, :POD_RANK]

    gp = MultiOutputGP(
        X_train=X_tr,
        Y_train=A_tr,
        kernel=GP_KERNEL,
        ard=USE_ARD,
        noise_variance=1e-6,
        update_every=25,
        n_retrain_max=0,
    )
    surrogate = PODGPSurrogate(pod=pod, gp=gp)

# %% [markdown]
# ## Test metrics + representative plots (saved to docs/assets)
#
# We compute RMSE and coverage, then save best/median/worst plots as PNG files.

# %%
if RUN_EXPENSIVE:
    z50, z90, z95 = 0.67449, 1.64485, 1.95996

    test_rmse = []
    mean_pred_std = []
    cov50, cov90, cov95 = [], [], []

    for theta, y_true in zip(X_te, Y_te, strict=True):
        y_hat, y_var = surrogate.predict(theta)
        y_std = np.sqrt(np.maximum(y_var, 1e-14))

        test_rmse.append(rmse(y_hat, y_true))
        mean_pred_std.append(float(np.mean(y_std)))
        cov50.append(coverage(y_true, y_hat, y_std, z=z50))
        cov90.append(coverage(y_true, y_hat, y_std, z=z90))
        cov95.append(coverage(y_true, y_hat, y_std, z=z95))

    test_rmse = np.asarray(test_rmse)
    order = np.argsort(test_rmse)
    idx_best = int(order[0])
    idx_median = int(order[len(order) // 2])
    idx_worst = int(order[-1])

    print("HF POD–GP surrogate (Navier–Stokes outlet profile)")
    print(f"POD rank: {POD_RANK}, N_train={X_tr.shape[0]}, N_test={X_te.shape[0]}, T={T}")
    print(f"RMSE mean   : {float(np.mean(test_rmse)):.6f}")
    print(f"RMSE median : {float(np.median(test_rmse)):.6f}")
    print(f"Coverage 50 : {float(np.mean(cov50)):.3f}")
    print(f"Coverage 90 : {float(np.mean(cov90)):.3f}")
    print(f"Coverage 95 : {float(np.mean(cov95)):.3f}")

    t_plot = np.arange(T, dtype=float)

    for label, idx, fname in [
        ("best", idx_best, "podgp_best.png"),
        ("median", idx_median, "podgp_median.png"),
        ("worst", idx_worst, "podgp_worst.png"),
    ]:
        theta = X_te[idx]
        y_true = Y_te[idx]
        fig, ax = plot_prediction_at_theta(
            surrogate,
            theta,
            t_plot,
            y_true,
            title=f"POD–GP prediction ({label})  h1={theta[0]:.4f}, U_in={theta[1]:.3f}",
            show=False,
        )
        fig.savefig(ASSET_DIR / fname, dpi=160)
        plt.close(fig)

# %% [markdown]
# ## Velocity animation for the standard geometry
#
# We run one HF simulation for a representative configuration and save an animation.
# This is skipped during MkDocs builds.
#
# The animation is saved to:
# - `docs/assets/navier-stokes/velocity.gif`


# %% [markdown]
# ## Embedded results (works in the hosted documentation)
#
# If you are reading this page in the hosted documentation, the plots and animation below
# are loaded from `docs/assets/navier-stokes/`. If you run the notebook interactively,
# re-running the expensive cells will regenerate these files.

# %%
# Nothing to execute here.

# %% [markdown]
# ### Representative surrogate predictions
#
# ![](../assets/navier-stokes/podgp_best.png)
# ![](../assets/navier-stokes/podgp_median.png)
# ![](../assets/navier-stokes/podgp_worst.png)
