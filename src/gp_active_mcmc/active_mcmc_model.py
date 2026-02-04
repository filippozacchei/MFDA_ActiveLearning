from __future__ import annotations

from typing import Protocol
import numpy as np

from .coarse_output import CoarseOutput


# ---------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------


class SurrogateModel(Protocol):
    """Low-fidelity surrogate with uncertainty estimates."""

    def predict(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return prediction mean and predictive variance."""

    def update(self, theta: np.ndarray, y: np.ndarray) -> None:
        """Update surrogate using high-fidelity data."""


class HighFidelityModel(Protocol):
    """High-fidelity forward model."""

    def predict(self, theta: np.ndarray) -> np.ndarray:
        """Return high-fidelity prediction."""


# ---------------------------------------------------------------------
# Base active MCMC model
# ---------------------------------------------------------------------


class ActiveMCMCModel:
    """
    Wrapper providing coarse (surrogate) and fine (HF) model evaluations.
    The surrogate is updated whenever HF data are used.
    """

    def __init__(
        self,
        surrogate: SurrogateModel,
        hf_model: HighFidelityModel,
        variance_threshold: float,
        update_on_fine: bool = True,
    ):
        self.surrogate = surrogate
        self.hf_model = hf_model
        self.variance_threshold = variance_threshold
        self.update_on_fine = update_on_fine

        self.n_hf_calls = 0
        self.used_hf: list[bool] = []

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def coarse(self, theta: np.ndarray):
        """Return surrogate output or HF output if uncertainty is large."""
        mean, var = self.surrogate.predict(theta)

        if self._requires_hf(var):
            return self._evaluate_hf(theta)

        self._record_no_hf()
        return CoarseOutput(mean, var)

    def fine(self, theta: np.ndarray) -> np.ndarray:
        """Return HF output and optionally update surrogate."""
        y = self._evaluate_hf(theta, record=False)

        if self.update_on_fine:
            self.surrogate.update(theta, y)

        return y

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _requires_hf(self, var: np.ndarray) -> bool:
        return float(np.mean(var)) > self.variance_threshold

    def _evaluate_hf(self, theta: np.ndarray, record: bool = True) -> np.ndarray:
        y = self.hf_model.predict(theta)

        self.surrogate.update(theta, y)
        self.n_hf_calls += 1

        if record:
            self.used_hf.append(True)

        return y

    def _record_no_hf(self) -> None:
        self.used_hf.append(False)


# ---------------------------------------------------------------------
# Adaptive active MCMC model
# ---------------------------------------------------------------------


class AdaptiveActiveMCMCModel(ActiveMCMCModel):
    """
    Active MCMC model with adaptive HF subchain length control.

    Subchain length increases when surrogate error is small and
    decreases when surrogate error is large.
    """

    def __init__(
        self,
        surrogate: SurrogateModel,
        hf_model: HighFidelityModel,
        variance_threshold: float,
        initial_subchain: int = 10,
        adapt_rate: float = 0.1,
        target_error: float = 0.01,
        update_every: int = 10,
        max_error_history: int = 50,
        min_subchain: int = 1,
        max_subchain: int = 100,
        update_on_fine: bool = True,
        max_steps: int | None = None,
    ):
        super().__init__(
            surrogate=surrogate,
            hf_model=hf_model,
            variance_threshold=variance_threshold,
            update_on_fine=update_on_fine,
        )

        self.subchain_length = initial_subchain
        self.adapt_rate = adapt_rate
        self.target_error = target_error
        self.update_every = update_every
        self.min_subchain = min_subchain
        self.max_subchain = max_subchain
        self.max_steps = max_steps

        self._errors: list[float] = []
        self._max_error_history = max_error_history
        self._step = 0
        self.subchain_history: list[int] = []

    # -----------------------------------------------------------------
    # Overrides
    # -----------------------------------------------------------------

    def coarse(self, theta: np.ndarray):
        self._step += 1
        return super().coarse(theta)

    def fine(self, theta: np.ndarray) -> np.ndarray:
        if self._hf_disabled():
            mean, _ = self.surrogate.predict(theta)
            return mean

        y = self.hf_model.predict(theta)
        if self.update_on_fine:
            self._update_surrogate(theta, y)

        return y

    # -----------------------------------------------------------------
    # Adaptive logic
    # -----------------------------------------------------------------

    def _update_surrogate(self, theta: np.ndarray, y: np.ndarray) -> None:
        mean, _ = self.surrogate.predict(theta)
        self.surrogate.update(theta, y)
        self._record_error(np.mean(np.abs(mean - y)))

    def _record_error(self, error: float) -> None:
        self._errors.append(error)
        if len(self._errors) > self._max_error_history:
            self._errors.pop(0)

        if not self._should_adapt():
            return

        self._adapt_subchain_length()

    def _should_adapt(self) -> bool:
        return self._step % self.update_every == 0 and len(self._errors) > 5

    def _adapt_subchain_length(self) -> None:
        npe = np.mean(self._errors) / self.target_error
        delta = np.clip(npe - 1.0, -1.0, 1.0)

        subsample_rate = 1.0 / self.subchain_length
        subsample_rate *= np.exp(self.adapt_rate * delta)
        subsample_rate = np.clip(
            subsample_rate,
            1.0 / self.max_subchain,
            1.0 / self.min_subchain,
        )

        self.subchain_length = int(round(1.0 / subsample_rate))
        self.subchain_history.append(self.subchain_length)

    def _hf_disabled(self) -> bool:
        return self.max_steps is not None and self._step >= self.max_steps
