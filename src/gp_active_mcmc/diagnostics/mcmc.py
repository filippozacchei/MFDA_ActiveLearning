from __future__ import annotations

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from collections.abc import Sequence
from typing import Any


# ---------------------------------------------------------------------
# Forward / high-fidelity usage diagnostics
# ---------------------------------------------------------------------


def plot_cumulative_hf_fraction(
    used_hf: npt.ArrayLike | Sequence[npt.ArrayLike],
    *,
    burnin: int = 0,
    labels: Sequence[str] | None = None,
) -> None:
    """
    Plot the cumulative fraction of high-fidelity evaluations.

    This function accepts either:
    - a single 1D array-like of HF flags, or
    - a sequence of 1D array-likes (multiple runs), optionally labelled.

    Parameters
    ----------
    used_hf
        Boolean or {0,1} array-like indicating high-fidelity usage per iteration,
        or a sequence of such arrays.
    burnin
        Number of initial samples to discard (non-negative).
    labels
        Optional labels for multiple runs. If provided, its length must match
        the number of runs.

    Notes
    -----
    The function silently returns (no plot) if, after burn-in, there are no points
    to plot for *all* provided series.
    """
    if burnin < 0:
        raise ValueError("`burnin` must be non-negative.")

    # Normalize input to a list of 1D numpy arrays (duck-typed, no fragile isinstance checks)
    series: list[np.ndarray] = []
    if _is_sequence_but_not_array(used_hf):
        for x in used_hf:  # type: ignore[union-attr]
            series.append(_as_1d_float_array(x, name="used_hf item"))
    else:
        series.append(_as_1d_float_array(used_hf, name="used_hf"))

    if labels is not None and len(labels) != len(series):
        raise ValueError(
            "`labels` length must match the number of series in `used_hf`."
        )

    plt.figure(figsize=(8, 4))

    plotted_any = False
    for i, flags in enumerate(series):
        if flags.ndim != 1:
            raise ValueError("Each `used_hf` series must be 1D.")
        if flags.size == 0 or burnin >= flags.size:
            continue

        tail = flags[burnin:]
        iterations = np.arange(1, tail.size + 1, dtype=float)
        cumulative_fraction = np.cumsum(tail) / iterations

        if labels is None:
            plt.plot(iterations, cumulative_fraction)
        else:
            plt.plot(iterations, cumulative_fraction, label=labels[i])

        plotted_any = True

    if not plotted_any:
        plt.close()
        return

    plt.xlabel("Iteration")
    plt.ylabel("Cumulative HF fraction")
    plt.title("High-fidelity usage over iterations")
    plt.grid(True)

    if labels is not None:
        plt.legend()

    plt.tight_layout()
    plt.show()


def _as_1d_float_array(x: npt.ArrayLike, *, name: str) -> np.ndarray:
    """Convert array-like to a 1D float numpy array with basic validation."""
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"`{name}` must be 1D (got ndim={arr.ndim}).")
    # Convert bool/int flags to float for cumulative sum/division
    return arr.astype(float, copy=False)


def _is_sequence_but_not_array(x: Any) -> bool:
    """True for sequences (e.g., list/tuple) but False for numpy arrays."""
    return isinstance(x, Sequence) and not isinstance(x, (np.ndarray, str, bytes))


# ---------------------------------------------------------------------
# Adaptive subchain diagnostics
# ---------------------------------------------------------------------
def plot_subchain_length_history(subchain_lengths: list[int]) -> None:
    """
    Plot the evolution of adaptive subchain lengths.

    Parameters
    ----------
    subchain_lengths : list[int]
        Subchain length at each coarse iteration.
    """
    if not subchain_lengths:
        return

    assert all(l > 0 for l in subchain_lengths), "All subchain lengths must be positive"

    plt.figure(figsize=(8, 4))
    plt.plot(subchain_lengths, marker="o", ms=3)
    plt.xlabel("Coarse iteration")
    plt.ylabel("Subchain length")
    plt.title("Adaptive subchain length history")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Surrogate accuracy diagnostics
# ---------------------------------------------------------------------
def plot_surrogate_error_history(
    errors: list[float],
    *,
    target: float | None = None,
) -> None:
    """
    Plot surrogate–HF discrepancy over fine iterations.

    Parameters
    ----------
    errors : list[float]
        Error metric per fine iteration.
    target : float or None, optional
        Target error threshold to display.
    """
    if not errors:
        return

    assert all(e >= 0 for e in errors), "Errors must be non-negative"

    plt.figure(figsize=(8, 4))
    plt.plot(errors, label="Surrogate error")

    if target is not None:
        assert target >= 0, "`target` must be non-negative"
        plt.axhline(
            target,
            linestyle="--",
            linewidth=1.5,
            label="Target error",
        )

    plt.xlabel("Fine iteration")
    plt.ylabel("Error")
    plt.title("Surrogate accuracy over time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# MCMC chain visualization
# ---------------------------------------------------------------------
def plot_chain(
    chain: np.ndarray,
    *,
    used_hf: np.ndarray | None = None,
    theta_true: np.ndarray | None = None,
    title: str | None = None,
    names: tuple[str, str] = ("θ₁", "θ₂"),
) -> None:
    """
    Two-dimensional visualization of an MCMC chain.

    Parameters
    ----------
    chain : np.ndarray
        Array of shape (N, 2) containing chain samples.
    used_hf : np.ndarray or None, optional
        Boolean array indicating high-fidelity usage.
    theta_true : np.ndarray or None, optional
        True parameter value (2D).
    title : str or None, optional
        Plot title.
    names : tuple[str, str], optional
        Axis labels.
    """
    assert (
        chain.ndim == 2 and chain.shape[1] == 2
    ), f"`chain` must have shape (N, 2) instead has shape: {chain.shape}"

    if used_hf is not None:
        assert (
            used_hf.shape[0] == chain.shape[0] or used_hf.shape[0] == chain.shape[0] + 1
        ), "`used_hf` must match chain length"
    if used_hf.shape[0] == chain.shape[0] + 1:
        chain = np.vstack([chain[0], chain])

    if theta_true is not None:
        assert theta_true.shape == (2,), "`theta_true` must be 2-D"

    fig, ax = plt.subplots(figsize=(6, 5))

    if used_hf is None:
        ax.scatter(chain[:, 0], chain[:, 1], s=25, alpha=0.6)
    else:
        hf_idx = np.where(used_hf)[0]
        sur_idx = np.where(~used_hf)[0]

        if sur_idx.size:
            ax.scatter(
                chain[sur_idx, 0],
                chain[sur_idx, 1],
                s=20,
                alpha=0.4,
                label="Surrogate",
            )

        if hf_idx.size:
            ax.scatter(
                chain[hf_idx, 0],
                chain[hf_idx, 1],
                s=40,
                marker="x",
                label="High-fidelity",
            )

    if theta_true is not None:
        ax.scatter(
            theta_true[0],
            theta_true[1],
            s=140,
            marker="*",
            c="k",
            label=r"$\theta_{\mathrm{true}}$",
        )

    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    plt.show()
