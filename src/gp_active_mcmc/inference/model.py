from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gp_active_mcmc.inference.coarse_output import CoarseOutput
from gp_active_mcmc.protocols import ActiveSurrogate, HighFidelityModel

FloatArray = NDArray[np.floating]


def _as_1d_theta(theta: ArrayLike) -> FloatArray:
    """Convert `theta` to a 1D float array.

    Parameters
    ----------
    theta
        Candidate parameter vector.

    Returns
    -------
    theta_1d
        1D array of dtype float.

    Raises
    ------
    ValueError
        If `theta` is not 1D.
    """
    arr = np.asarray(theta, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"theta must be 1D. Got shape {arr.shape}.")
    return arr


def _as_1d_float(x: ArrayLike, *, name: str) -> FloatArray:
    """Convert an array-like to a 1D float array via ravel.

    Parameters
    ----------
    x
        Input array-like.
    name
        Name used for error messages.

    Returns
    -------
    x_1d
        1D array of dtype float.

    Raises
    ------
    ValueError
        If the result is not 1D after ravel (should be rare, but guarded).
    """
    arr = np.asarray(x, dtype=float).ravel()
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D after ravel. Got shape {arr.shape}.")
    return arr


@dataclass(slots=True)
class EvaluationLog:
    """Evaluation bookkeeping for Active-MCMC.

    Attributes
    ----------
    used_hf
        Boolean flag per coarse evaluation indicating whether the HF model
        was used for that step.
    """

    used_hf: list[bool] = field(default_factory=list)

    def append(self, used_hf: bool) -> None:
        """Append a flag indicating whether HF was used."""
        self.used_hf.append(bool(used_hf))

    def replace_last(self, used_hf: bool) -> None:
        """Replace the most recent flag, or append if the log is empty."""
        if self.used_hf:
            self.used_hf[-1] = bool(used_hf)
        else:
            self.append(bool(used_hf))


class AdaptiveHook(Protocol):
    """Optional callback interface for adaptive control.

    The model calls this hook to allow external logic (e.g. adaptive subchain
    control) to observe events and update its internal state.

    Notes
    -----
    - `on_coarse_call` is called once at the start of `ActiveMCMCModel.coarse`.
    - `on_fine_call` is called inside `ActiveMCMCModel.fine` **before** the LF
      model is updated with the HF observation.
    """

    def on_coarse_call(self) -> None:
        """Called at the start of a coarse evaluation."""

    def on_fine_call(
        self,
        *,
        y_hf: FloatArray,
        y_lf: FloatArray,
    ) -> None:
        """Called during fine evaluation before updating the surrogate.

        Parameters
        ----------
        y_hf
            HF model output at `theta`.
        y_lf
            LF predictive mean at `theta` **before** the LF model is updated.
        """


@dataclass(slots=True)
class ActiveMCMCModel:
    """ActiveMCMCModel.

    Coupled low-/high-fidelity forward model for multi-fidelity Active MCMC.

    This object implements two evaluation modes that are typically called by an
    MCMC likelihood:

    - `coarse(theta)`: evaluate the LF surrogate and return a `CoarseOutput`
      (mean + variance). If the surrogate uncertainty is too large, fall back
      to HF, update the surrogate, and return the HF output.
    - `fine(theta)`: always evaluate HF, update the surrogate, and return HF.

    HF usage is recorded in `log.used_hf`, aligned with the *coarse* evaluations.

    Parameters
    ----------
    lf_model
        Low-fidelity surrogate implementing `predict(theta) -> (mean, var)` and
        `update(theta, y_hf)`.
    hf_model
        High-fidelity forward model callable as `hf_model(theta) -> y`.
    gamma_threshold
        Uncertainty threshold. A coarse call triggers HF if
        `mean(y_var) > gamma_threshold**2`.
    log
        Evaluation log. If omitted, a fresh log is created.
    adaptive
        Optional hook for adaptive logic (e.g. adaptive subchain length). The hook
        receives notifications during `coarse` and `fine`.

    Notes
    -----
    Return types:
    - `coarse` returns either `CoarseOutput(mean, var)` (LF) or a 1D numpy array (HF).
    - `fine` always returns a 1D numpy array (HF).

    The HF-vs-LF decision uses the *average* predictive variance, which is a simple
    scalar proxy for model uncertainty. Alternative decision rules can be implemented
    by subclassing or by replacing this component with a different policy object.
    """

    lf_model: ActiveSurrogate
    hf_model: HighFidelityModel
    gamma_threshold: float
    log: EvaluationLog = field(default_factory=EvaluationLog)
    adaptive: AdaptiveHook | None = None

    def __post_init__(self) -> None:
        if self.gamma_threshold < 0.0:
            raise ValueError("gamma_threshold must be non-negative.")

    def coarse(self, theta: ArrayLike) -> np.ndarray | CoarseOutput:
        """Coarse (LF-first) evaluation with optional HF correction.

        Parameters
        ----------
        theta
            Parameter vector of shape `(n_dim,)`.

        Returns
        -------
        y
            If LF is used: `CoarseOutput(mean, var)` where both are 1D arrays of
            shape `(n_obs,)`. If HF is used: a 1D numpy array of shape `(n_obs,)`.

        Raises
        ------
        ValueError
            If shapes returned by the surrogate are inconsistent.
        """
        th = _as_1d_theta(theta)

        if self.adaptive is not None:
            self.adaptive.on_coarse_call()

        y_mean, y_var = self.lf_model.predict(th)
        mean = _as_1d_float(y_mean, name="y_mean")
        var = _as_1d_float(y_var, name="y_var")

        if mean.shape != var.shape:
            raise ValueError(
                f"Surrogate returned mean/var with different shapes: {mean.shape} vs {var.shape}."
            )

        avg_var = float(np.mean(var))
        if avg_var > self.gamma_threshold**2:
            y_hf = _as_1d_float(self.hf_model(th), name="y_hf")
            self.lf_model.update(th, y_hf)
            self.log.append(True)
            return y_hf

        self.log.append(False)
        return CoarseOutput(mean, var)

    def fine(self, theta: ArrayLike, *, replace_last: bool = True) -> np.ndarray:
        """Fine (HF) evaluation and surrogate update.

        Parameters
        ----------
        theta
            Parameter vector of shape `(n_dim,)`.
        replace_last
            If True, replaces the most recent entry in `log.used_hf` with True.
            This is convenient when `fine` is called as a correction following a
            previous `coarse` evaluation at the same step. If False, a new log
            entry is appended.

        Returns
        -------
        y_hf
            HF model output as a 1D numpy array of shape `(n_obs,)`.
        """
        th = _as_1d_theta(theta)

        y_hf = _as_1d_float(self.hf_model(th), name="y_hf")

        if self.adaptive is not None:
            lf_mean, _lf_var = self.lf_model.predict(th)
            lf_mean = _as_1d_float(lf_mean, name="lf_mean_before_update")
            self.adaptive.on_fine_call(y_hf=y_hf, y_lf=lf_mean)

        self.lf_model.update(th, y_hf)

        if replace_last:
            self.log.replace_last(True)
        else:
            self.log.append(True)

        return y_hf
