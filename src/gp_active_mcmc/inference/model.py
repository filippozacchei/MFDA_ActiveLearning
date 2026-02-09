from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gp_active_mcmc.protocols import ActiveSurrogate, HighFidelityModel
from gp_active_mcmc.inference.coarse_output import CoarseOutput

FloatArray = NDArray[np.floating]


def _as_1d_theta(theta: ArrayLike) -> FloatArray:
    arr = np.asarray(theta, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"theta must be 1D. Got shape {arr.shape}.")
    return arr


def _as_1d_float(x: ArrayLike, *, name: str) -> FloatArray:
    arr = np.asarray(x, dtype=float).ravel()
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D after ravel. Got shape {arr.shape}.")
    return arr


@dataclass(slots=True)
class EvaluationLog:
    """Bookkeeping for Active-MCMC evaluations."""
    used_hf: list[bool] = field(default_factory=list)

    def append(self, used_hf: bool) -> None:
        self.used_hf.append(bool(used_hf))

    def replace_last(self, used_hf: bool) -> None:
        if self.used_hf:
            self.used_hf[-1] = bool(used_hf)
        else:
            self.append(bool(used_hf))


class AdaptiveHook(Protocol):
    """Hook interface for adaptive Active-MCMC control."""

    def on_coarse_call(self) -> None:
        """Called at the start of a coarse evaluation."""

    def on_fine_call(self, *, theta: FloatArray, y_hf: FloatArray, lf_mean_before_update: FloatArray) -> None:
        """Called during fine evaluation before updating the surrogate."""


@dataclass(slots=True)
class ActiveMCMCModel:
    """Active-MCMC model coupling a surrogate (LF) with a high-fidelity (HF) model.

    The model exposes two modes:

    - coarse(theta): use LF unless average predictive variance exceeds gamma_threshold^2,
      in which case evaluate HF and update LF.
    - fine(theta): always evaluate HF and update LF.

    The model records HF usage in `log.used_hf`.
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
        th = _as_1d_theta(theta)

        if self.adaptive is not None:
            self.adaptive.on_coarse_call()

        y_mean, y_var = self.lf_model.predict(th)
        mean = _as_1d_float(y_mean, name="y_mean")
        var = _as_1d_float(y_var, name="y_var")

        if mean.shape != var.shape:
            raise ValueError(f"Surrogate returned mean/var with different shapes: {mean.shape} vs {var.shape}.")

        avg_var = float(np.mean(var))
        if avg_var > self.gamma_threshold**2:
            y_hf = _as_1d_float(self.hf_model(th), name="y_hf")
            self.lf_model.update(th, y_hf)
            self.log.append(True)
            return y_hf

        self.log.append(False)
        return CoarseOutput(mean, var)

    def fine(self, theta: ArrayLike, *, replace_last: bool = True) -> np.ndarray:
        th = _as_1d_theta(theta)

        # Evaluate HF
        y_hf = _as_1d_float(self.hf_model(th), name="y_hf")

        # Adaptive hook uses LF prediction *before* updating LF
        if self.adaptive is not None:
            lf_mean, _ = self.lf_model.predict(th)
            lf_mean = _as_1d_float(lf_mean, name="lf_mean_before_update")
            self.adaptive.on_fine_call(y_hf=y_hf, y_lf=lf_mean)

        # Update surrogate after hook
        self.lf_model.update(th, y_hf)

        # Bookkeeping: optionally upgrade the last coarse evaluation
        if replace_last:
            self.log.replace_last(True)
        else:
            self.log.append(True)

        return y_hf
