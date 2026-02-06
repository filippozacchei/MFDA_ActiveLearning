# active_mcmc.py
from __future__ import annotations

import numpy as np

from .protocols import ActiveSurrogate, HighFidelityModel
from .coarse_output import CoarseOutput
from .adaptive_config import AdaptiveState, AdaptiveControl


class ActiveMCMCModel:
    """MCMC model wrapper providing coarse (surrogate) and fine (HF) evaluations."""

    def __init__(
        self,
        lf_model: ActiveSurrogate,
        hf_model: HighFidelityModel,
        gamma_threshold: float,
    ):
        self.lf_model = lf_model
        self.hf_model = hf_model
        assert gamma_threshold >= 0, "gamma_threshold must be a non negative float"
        self.gamma_threshold = gamma_threshold

        self.used_hf_flags: list[bool] = []

    def _append_hf(self):
        self.used_hf_flags.append(True)

    def _append_lf(self):
        self.used_hf_flags.append(False)

    def _update_lf(self, theta: np.ndarray, y: np.ndarray) -> None:
        self.lf_model.update(theta, y)

    def coarse(self, theta: np.ndarray) -> np.ndarray | CoarseOutput:
        """Return surrogate prediction, or HF if uncertainty exceeds threshold."""
        y_pred, var = self.lf_model.predict(theta)
        avg_var = float(np.mean(var))

        if avg_var > self.gamma_threshold**2:
            self._append_hf()
            y_fine = self.fine(theta)  # to remove after bebugging
            return y_fine

        self._append_lf()
        return CoarseOutput(y_pred, var)

    def fine(self, theta: np.ndarray) -> np.ndarray:
        """Return HF prediction, updating the surrogate."""
        y = self.hf_model(theta)
        self.used_hf_flags.pop()
        self._update_lf(theta, y)
        self._append_hf()
        return y


class AdaptiveActiveMCMCModel(ActiveMCMCModel):
    """Adaptive MCMC that adjusts HF subchain length based on surrogate prediction error."""

    def __init__(
        self,
        lf_model: ActiveSurrogate,
        hf_model: HighFidelityModel,
        gamma_threshold: float,
        initial_adaptive_state: AdaptiveState,
        adaptive_control: AdaptiveControl,
    ):
        super().__init__(lf_model, hf_model, gamma_threshold)
        self.adaptive_control = adaptive_control
        self.adaptive_state = initial_adaptive_state

    def coarse(self, theta: np.ndarray) -> np.ndarray | CoarseOutput:
        self.adaptive_state.append_length()
        return super().coarse(theta)

    def fine(self, theta: np.ndarray) -> np.ndarray:
        self.adaptive_state.step()
        return super().fine(theta)

    def _update_lf(self, theta: np.ndarray, y: np.ndarray) -> None:
        y_pred, _ = self.lf_model.predict(theta)
        self.adaptive_state.append_error(y_pred, y)
        self.adaptive_state.update_subchain(self.adaptive_control)
        super()._update_lf(theta, y)
