from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from numpy.typing import ArrayLike


def _is_sequence_but_not_array(x: Any) -> bool:
    return isinstance(x, Sequence) and not isinstance(x, (np.ndarray, str, bytes))


def _as_1d_bool(x: ArrayLike, *, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D. Got shape {arr.shape}.")
    return arr.astype(bool, copy=False)


def plot_cumulative_hf_fraction(
    used_hf: ArrayLike | Sequence[ArrayLike],
    *,
    burnin: int = 0,
    labels: Sequence[str] | None = None,
    title: str = "High-fidelity usage over iterations",
) -> tuple[Figure, Axes] | None:
    """Plot cumulative HF fraction over iterations.

    Returns (fig, ax). If nothing can be plotted (empty after burn-in), returns None.
    """
    if burnin < 0:
        raise ValueError("burnin must be non-negative.")

    series: list[np.ndarray] = []
    if _is_sequence_but_not_array(used_hf):
        for x in used_hf:  # type: ignore[union-attr]
            series.append(_as_1d_bool(x, name="used_hf item"))
    else:
        series.append(_as_1d_bool(used_hf, name="used_hf"))

    if labels is not None and len(labels) != len(series):
        raise ValueError("labels length must match number of series.")

    fig, ax = plt.subplots(figsize=(7, 4))
    plotted_any = False

    for i, flags in enumerate(series):
        if flags.size == 0 or burnin >= flags.size:
            continue
        tail = flags[burnin:].astype(float)
        it = np.arange(1, tail.size + 1, dtype=float)
        cum = np.cumsum(tail) / it
        ax.plot(it, cum, label=None if labels is None else labels[i])
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return None

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cumulative HF fraction")
    ax.set_title(title)
    ax.grid(True)
    if labels is not None:
        ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_subchain_length_history(subchain_length: ArrayLike) -> tuple[Figure, Axes] | None:
    """Plot adaptive subchain length history. Returns None for empty input."""
    s = np.asarray(subchain_length, dtype=int).ravel()
    if s.size == 0:
        return None
    if np.any(s <= 0):
        raise ValueError("All subchain lengths must be positive.")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(s, marker="o", ms=3)
    ax.set_xlabel("Coarse iteration")
    ax.set_ylabel("Subchain length")
    ax.set_title("Adaptive subchain length history")
    ax.grid(True)
    fig.tight_layout()
    return fig, ax


def plot_surrogate_error_history(
    errors: ArrayLike,
    *,
    target: float | None = None,
    title: str = "Surrogate accuracy over time",
) -> tuple[Figure, Axes] | None:
    """Plot surrogate–HF discrepancy over fine iterations."""
    e = np.asarray(errors, dtype=float).ravel()
    if e.size == 0:
        return None
    if np.any(e < 0):
        raise ValueError("Errors must be non-negative.")
    if target is not None and target < 0:
        raise ValueError("target must be non-negative.")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(e, label="surrogate error")
    if target is not None:
        ax.axhline(target, linestyle="--", linewidth=1.5, label="target")
    ax.set_xlabel("Fine iteration")
    ax.set_ylabel("Error")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig, ax


def plot_chain_2d(
    samples: ArrayLike,
    *,
    used_hf: ArrayLike | None = None,
    theta_true: ArrayLike | None = None,
    title: str | None = None,
    names: tuple[str, str] = ("θ₁", "θ₂"),
) -> tuple[Figure, Axes]:
    """2D scatter plot of samples with optional HF markers and true parameter."""
    chain = np.asarray(samples, dtype=float)
    if chain.ndim != 2 or chain.shape[1] != 2:
        raise ValueError(f"samples must have shape (n_steps, 2). Got shape {chain.shape}.")

    uhf = None if used_hf is None else _as_1d_bool(used_hf, name="used_hf")
    if uhf is not None and uhf.shape[0] != chain.shape[0]:
        raise ValueError("used_hf must have the same length as number of samples.")

    tt = None if theta_true is None else np.asarray(theta_true, dtype=float).ravel()
    if tt is not None and tt.shape != (2,):
        raise ValueError("theta_true must be shape (2,).")

    fig, ax = plt.subplots(figsize=(6, 5))

    if uhf is None:
        ax.scatter(chain[:, 0], chain[:, 1], s=25, alpha=0.6)
    else:
        sur_idx = np.where(~uhf)[0]
        hf_idx = np.where(uhf)[0]
        if sur_idx.size:
            ax.scatter(chain[sur_idx, 0], chain[sur_idx, 1], s=20, alpha=0.4, label="surrogate")
        if hf_idx.size:
            ax.scatter(chain[hf_idx, 0], chain[hf_idx, 1], s=40, marker="x", label="high-fidelity")
        ax.legend()

    if tt is not None:
        ax.scatter(tt[0], tt[1], s=140, marker="*", c="k", label=r"$\theta_{\mathrm{true}}$")
        ax.legend()

    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    if title is not None:
        ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    return fig, ax
