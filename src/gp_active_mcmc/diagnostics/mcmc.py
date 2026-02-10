from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike


def _is_sequence_but_not_array(x: Any) -> bool:
    """Return True for non-string Sequences that are not NumPy arrays.

    This is used to distinguish a single 1D array of flags from a list/tuple of
    multiple series.

    Parameters
    ----------
    x
        Candidate object.

    Returns
    -------
    is_seq
        True if `x` is a `Sequence` (e.g. list/tuple) but not a NumPy array and not a
        string/bytes object.
    """
    return isinstance(x, Sequence) and not isinstance(x, (np.ndarray, str, bytes))


def _as_1d_bool(x: ArrayLike, *, name: str) -> np.ndarray:
    """Convert an array-like to a 1D boolean array.

    Parameters
    ----------
    x
        Input array-like.
    name
        Name used in error messages.

    Returns
    -------
    flags
        1D boolean array.

    Raises
    ------
    ValueError
        If `x` is not one-dimensional.
    """
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
    show: bool = False,
) -> tuple[Figure, Axes] | None:
    """Plot cumulative HF usage fraction over iterations.

    Given a boolean series `used_hf` indicating whether the high-fidelity (HF) model was
    used at each MCMC step, this function plots the cumulative HF fraction.

    The function supports plotting multiple runs by passing a sequence of arrays.

    Parameters
    ----------
    used_hf
        Either a single boolean array of shape ``(n_steps,)`` or a sequence of such arrays.
        True indicates that the HF model was used at that step.
    burnin
        Number of initial steps to discard before computing the cumulative fraction.
        Must be non-negative.
    labels
        Optional labels for each series when `used_hf` is a sequence. If provided, its
        length must match the number of series.
    title
        Title for the plot.
    show
        If True, call `plt.show()` before returning. For MkDocs/Jupyter usage, leaving this
        False is recommended.

    Returns
    -------
    fig, ax or None
        Returns `(fig, ax)` if at least one series contains data after burn-in.
        Returns `None` if nothing can be plotted (e.g. all series empty after burn-in).

    Raises
    ------
    ValueError
        If `burnin` is negative or if `labels` has the wrong length.

    Notes
    -----
    In active-learning runs, HF usage flags are typically recorded by
    [`ActiveMCMCModel.log.used_hf`][gp_active_mcmc.inference.model.EvaluationLog] and
    attached to the chain extras.

    See Also
    --------
    [`plot_subchain_length_history`][gp_active_mcmc.diagnostics.chain.plot_subchain_length_history]
        Visualise how the adaptive subchain length changes over time.
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

    if show:
        plt.show()

    return fig, ax


def plot_subchain_length_history(
    subchain_length: ArrayLike,
    *,
    title: str = "Adaptive subchain length history",
    show: bool = False,
) -> tuple[Figure, Axes] | None:
    """Plot the adaptive subchain length history.

    Parameters
    ----------
    subchain_length
        1D integer array of subchain lengths. Values must be positive.
    title
        Plot title.
    show
        If True, call `plt.show()` before returning.

    Returns
    -------
    fig, ax or None
        Returns `(fig, ax)` if `subchain_length` is non-empty, otherwise `None`.

    Raises
    ------
    ValueError
        If any subchain length is non-positive.

    See Also
    --------
    [`AdaptiveSubchainState.subchain_history`][gp_active_mcmc.inference.adaptive_subchain.AdaptiveSubchainState]
        Field where the adaptive policy records the subchain length per coarse call.
    """
    s = np.asarray(subchain_length, dtype=int).ravel()
    if s.size == 0:
        return None
    if np.any(s <= 0):
        raise ValueError("All subchain lengths must be positive.")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(s, marker="o", ms=3)
    ax.set_xlabel("Coarse iteration")
    ax.set_ylabel("Subchain length")
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_surrogate_error_history(
    errors: ArrayLike,
    *,
    target: float | None = None,
    title: str = "Surrogate accuracy over time",
    show: bool = False,
) -> tuple[Figure, Axes] | None:
    """Plot surrogate–HF discrepancy over fine evaluations.

    Parameters
    ----------
    errors
        1D array of non-negative error values (e.g. RMSE between LF mean and HF output)
        recorded at fine evaluations.
    target
        Optional target error level. If provided, a horizontal reference line is drawn.
    title
        Plot title.
    show
        If True, call `plt.show()` before returning.

    Returns
    -------
    fig, ax or None
        Returns `(fig, ax)` if `errors` is non-empty, otherwise `None`.

    Raises
    ------
    ValueError
        If any error is negative or if `target` is negative.

    See Also
    --------
    [`AdaptiveSubchainState.hf_errors`][gp_active_mcmc.inference.adaptive_subchain.AdaptiveSubchainState]
        Error history recorded by the adaptive policy.
    """
    e = np.asarray(errors, dtype=float).ravel()
    if e.size == 0:
        return None
    if np.any(e < 0.0):
        raise ValueError("Errors must be non-negative.")
    if target is not None and target < 0.0:
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

    if show:
        plt.show()

    return fig, ax


def plot_chain_2d(
    samples: ArrayLike,
    *,
    used_hf: ArrayLike | None = None,
    theta_true: ArrayLike | None = None,
    title: str | None = None,
    names: tuple[str, str] = ("θ₁", "θ₂"),
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Scatter plot of a 2D chain, optionally highlighting HF steps.

    Parameters
    ----------
    samples
        Sample array of shape ``(n_steps, 2)``.
    used_hf
        Optional boolean array of shape ``(n_steps,)`` indicating whether HF was used at
        each step. If provided, surrogate-only steps and HF steps are plotted with
        different markers.
    theta_true
        Optional reference parameter of shape ``(2,)`` to overlay as a star marker.
    title
        Optional title.
    names
        Axis labels for the two parameters.
    show
        If True, call `plt.show()` before returning.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.

    Raises
    ------
    ValueError
        If `samples` is not shape ``(n_steps, 2)``, if `used_hf` has the wrong length,
        or if `theta_true` is not shape ``(2,)``.

    Notes
    -----
    This function is intended for quick diagnostics and documentation examples. For
    higher-dimensional chains, prefer pair plots or marginal traces.

    See Also
    --------
    [`plot_cumulative_hf_fraction`][gp_active_mcmc.diagnostics.chain.plot_cumulative_hf_fraction]
        Cumulative HF usage fraction over time.
    """
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

    if show:
        plt.show()

    return fig, ax
