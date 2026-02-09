# %% [markdown]
# # Dataset generation (LF/MF/HF) with nested designs
#
# Generates:
# - 125 LF samples
# - 25 MF samples (subset of LF)
# - 5 HF samples (subset of MF)
#
# Saves:
# data/lf.npz, data/mf.npz, data/hf.npz with arrays:
# - X: (N,1) inputs (h1)
# - Y: (N,ny) outlet velocity profile resampled to ny points

# %% Imports
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


# %% Data container
@dataclass(frozen=True)
class Dataset:
    X: np.ndarray  # (N,1)
    Y: np.ndarray  # (N,ny)


# %% Design
@dataclass(frozen=True)
class NestedDesign:
    X_lf: np.ndarray  # (n_lf,1)
    X_mf: np.ndarray  # (n_mf,1) subset of LF
    X_hf: np.ndarray  # (n_hf,1) subset of MF
    idx_mf_in_lf: np.ndarray
    idx_hf_in_mf: np.ndarray
    idx_hf_in_lf: np.ndarray


def make_nested_design(
    *,
    n_lf: int,
    n_mf: int,
    n_hf: int,
    h_min: float,
    h_max: float,
    seed: int,
) -> NestedDesign:
    if not (0 < n_hf <= n_mf <= n_lf):
        raise ValueError("Require 0 < n_hf <= n_mf <= n_lf")
    if not (h_min < h_max):
        raise ValueError("Require h_min < h_max")

    rng = np.random.default_rng(seed)

    X_lf = rng.uniform(h_min, h_max, size=(n_lf, 1))
    X_lf = np.sort(X_lf, axis=0)

    idx_mf_in_lf = np.sort(rng.choice(n_lf, size=n_mf, replace=False))
    X_mf = X_lf[idx_mf_in_lf]

    idx_hf_in_mf = np.sort(rng.choice(n_mf, size=n_hf, replace=False))
    X_hf = X_mf[idx_hf_in_mf]
    idx_hf_in_lf = idx_mf_in_lf[idx_hf_in_mf]

    return NestedDesign(
        X_lf=X_lf,
        X_mf=X_mf,
        X_hf=X_hf,
        idx_mf_in_lf=idx_mf_in_lf,
        idx_hf_in_mf=idx_hf_in_mf,
        idx_hf_in_lf=idx_hf_in_lf,
    )


# %% Profile utilities
def resample_profile(y: np.ndarray, u: np.ndarray, *, ny: int) -> np.ndarray:
    y = np.asarray(y).ravel()
    u = np.asarray(u).ravel()
    if y.size != u.size:
        raise ValueError("y and u must have same length")

    if not np.all(np.diff(y) >= 0):
        idx = np.argsort(y)
        y = y[idx]
        u = u[idx]

    y_new = np.linspace(float(y.min()), float(y.max()), ny)
    return np.interp(y_new, y, u)


def build_dataset(
    *,
    solver: Callable[[float], tuple[np.ndarray, np.ndarray]],
    h1_values: np.ndarray,  # (N,1)
    ny: int,
    allow_fail: bool = False,
) -> Dataset:
    h1_values = np.asarray(h1_values)
    if h1_values.ndim == 1:
        h1_values = h1_values[:, None]
    if h1_values.ndim != 2 or h1_values.shape[1] != 1:
        raise ValueError("h1_values must be (N,1)")

    Y = np.zeros((h1_values.shape[0], ny), dtype=float)
    for i, h1 in enumerate(h1_values[:, 0]):
        try:
            y, u = solver(float(h1))
        except Exception as e:
            if allow_fail:
                Y[i, :] = np.nan
                continue
            raise RuntimeError(f"Solver raised at h1={h1}: {e}") from e

        if not np.all(np.isfinite(u)):
            if allow_fail:
                Y[i, :] = np.nan
                continue
            raise RuntimeError(f"Solver returned non-finite values at h1={h1}")

        Y[i, :] = resample_profile(y, u, ny=ny)

    return Dataset(X=h1_values, Y=Y)


def save_npz_dataset(path: Path, ds: Dataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, X=ds.X, Y=ds.Y)


# %% Main
def main() -> None:
    # settings
    ny = 100
    U_in = 1.5
    design = make_nested_design(
        n_lf=125,
        n_mf=25,
        n_hf=5,
        h_min=0.05,
        h_max=0.15,
        seed=7,
    )

    # import solvers (expected signature: forward_model(h, u=1.5) OR forward_model(h, u))
    from utils.lf_potential import forward_model as lf_solver
    from utils.mf_ipcs import forward_model as mf_solver
    from utils.hf_boussinesq import forward_model as hf_solver

    # IMPORTANT: pass velocity as positional to avoid keyword mismatch
    ds_lf = build_dataset(
        solver=lambda h: lf_solver(h, U_in=U_in), h1_values=design.X_lf, ny=ny
    )
    ds_mf = build_dataset(
        solver=lambda h: mf_solver(h, U_in=U_in), h1_values=design.X_mf, ny=ny
    )
    ds_hf = build_dataset(
        solver=lambda h: hf_solver(h, U_in=U_in), h1_values=design.X_hf, ny=ny
    )

    save_npz_dataset(Path("data/lf.npz"), ds_lf)
    save_npz_dataset(Path("data/mf.npz"), ds_mf)
    save_npz_dataset(Path("data/hf.npz"), ds_hf)


if __name__ == "__main__":
    main()
