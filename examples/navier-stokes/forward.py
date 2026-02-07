# %% [markdown]
# # Forward surrogate demo (LF / MF / HF) — no POD, train/test split
#
# - LF: split train/test
# - MF: split train/test
# - HF: keep all in train (HF=5 is too small to split)
#
# Fit AR co-kriging on TRAIN only and plot one example per split.

# %% Imports
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from gp_active_mcmc.gp import AutoRegressiveCoKrigingN


KernelName = Literal["rbf", "matern32", "matern52"]


# %% Dataset I/O
@dataclass(frozen=True)
class Dataset:
    X: np.ndarray  # (N,1)
    Y: np.ndarray  # (N,T)


def _ensure_X(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2 or X.shape[1] != 1:
        raise ValueError(f"X must be (N,1) or (N,), got {X.shape}")
    return X


def _ensure_Y(Y: np.ndarray, *, T: int) -> np.ndarray:
    Y = np.asarray(Y)
    if Y.ndim != 2 or Y.shape[1] != T:
        raise ValueError(f"Y must be (N,{T}), got {Y.shape}")
    return Y


def load_npz(path: Path, *, T: int) -> Dataset:
    data = np.load(path)
    X = _ensure_X(data["X"])
    Y = _ensure_Y(data["Y"], T=T)
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"Row mismatch in {path}: X has {X.shape[0]}, Y has {Y.shape[0]}"
        )
    return Dataset(X=X, Y=Y)


# %% Model wrapper (pointwise = T independent AR models)
@dataclass
class PointwiseARCoKriging:
    models: list[AutoRegressiveCoKrigingN]
    T: int

    def predict(self, X: np.ndarray, *, level: int) -> tuple[np.ndarray, np.ndarray]:
        X = _ensure_X(X)
        N = X.shape[0]
        mu = np.zeros((N, self.T))
        var = np.zeros((N, self.T))
        for t, m in enumerate(self.models):
            mu_t, var_t = m.predict(X, level=level)
            mu[:, t] = mu_t
            var[:, t] = var_t
        return mu, var


def fit_pointwise_ar_cokriging(
    X_levels: list[np.ndarray],
    Y_levels: list[np.ndarray],
    *,
    T: int,
    kernel_base: KernelName = "matern52",
    kernel_delta: KernelName = "matern52",
    ard: bool = True,
    n_retrain_max: int = 0,
    update_every: int = 25,
) -> PointwiseARCoKriging:
    X_levels = [_ensure_X(X) for X in X_levels]
    Y_levels = [_ensure_Y(Y, T=T) for Y in Y_levels]

    models: list[AutoRegressiveCoKrigingN] = []
    for t in range(T):
        models.append(
            AutoRegressiveCoKrigingN(
                X_levels=X_levels,
                Y_levels=[Y[:, t] for Y in Y_levels],
                kernel_base=kernel_base,
                kernel_delta=kernel_delta,
                ard=ard,
                n_retrain_max=n_retrain_max,
                update_every=update_every,
            )
        )
    return PointwiseARCoKriging(models=models, T=T)


# %% Split helper
def split_dataset(
    ds: Dataset, *, test_frac: float, seed: int
) -> tuple[Dataset, Dataset]:
    rng = np.random.default_rng(seed)
    n = ds.X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(test_frac * n))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return Dataset(ds.X[train_idx], ds.Y[train_idx]), Dataset(
        ds.X[test_idx], ds.Y[test_idx]
    )


# %% Plot
def plot_truth_pred(
    *, name: str, split: str, h1: float, y_true: np.ndarray, y_pred: np.ndarray
) -> None:
    plt.figure()
    plt.plot(y_true, label=f"{name} truth")
    plt.plot(y_pred, "--", label=f"{name} pred")
    plt.title(f"{name} — {split} (h1={h1:.4f})")
    plt.xlabel("outlet sample index")
    plt.ylabel(r"$u_x$")
    plt.legend()
    plt.tight_layout()
    plt.show()


# %% Main
def main() -> None:
    T = 100
    seed = 7
    test_frac = 0.8

    data_dir = Path("data")
    lf = load_npz(data_dir / "lf.npz", T=T)
    mf = load_npz(data_dir / "mf.npz", T=T)
    hf = load_npz(data_dir / "hf.npz", T=T)

    lf_tr, lf_te = split_dataset(lf, test_frac=test_frac, seed=seed + 0)
    mf_tr, mf_te = split_dataset(mf, test_frac=test_frac, seed=seed + 1)
    hf_tr, hf_te = split_dataset(hf, test_frac=test_frac, seed=seed + 2)

    model = fit_pointwise_ar_cokriging(
        X_levels=[lf_tr.X, mf_tr.X, hf_tr.X],
        Y_levels=[lf_tr.Y, mf_tr.Y, hf_tr.Y],
        T=T,
        kernel_base="matern52",
        kernel_delta="matern52",
        ard=True,
        n_retrain_max=0,
        update_every=25,
    )

    # LF: one train + one test
    x = lf_tr.X[0:1]
    y_pred, _ = model.predict(x, level=0)
    plot_truth_pred(
        name="LF", split="train", h1=float(x[0, 0]), y_true=lf_tr.Y[0], y_pred=y_pred[0]
    )

    x = lf_te.X[0:1]
    y_pred, _ = model.predict(x, level=0)
    plot_truth_pred(
        name="LF", split="test", h1=float(x[0, 0]), y_true=lf_te.Y[0], y_pred=y_pred[0]
    )

    # MF: one train + one test
    x = mf_tr.X[0:1]
    y_pred, _ = model.predict(x, level=1)
    plot_truth_pred(
        name="MF", split="train", h1=float(x[0, 0]), y_true=mf_tr.Y[0], y_pred=y_pred[0]
    )

    x = mf_te.X[0:1]
    y_pred, _ = model.predict(x, level=1)
    plot_truth_pred(
        name="MF", split="test", h1=float(x[0, 0]), y_true=mf_te.Y[0], y_pred=y_pred[0]
    )

    # HF: only train
    x = hf_tr.X[0:1]
    y_pred, _ = model.predict(x, level=2)
    plot_truth_pred(
        name="HF", split="train", h1=float(x[0, 0]), y_true=hf_tr.Y[0], y_pred=y_pred[0]
    )

    x = hf_te.X[0:1]
    y_pred, _ = model.predict(x, level=2)
    plot_truth_pred(
        name="HF", split="test", h1=float(x[0, 0]), y_true=hf_te.Y[0], y_pred=y_pred[0]
    )


if __name__ == "__main__":
    main()
