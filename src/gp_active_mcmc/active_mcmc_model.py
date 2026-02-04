from typing import Protocol
import numpy as np

from .coarse_output import CoarseOutput


class ActiveLF(Protocol):
    """Protocol for a surrogate model with active learning capability."""

    def predict(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return surrogate prediction and associated uncertainty."""

    def update(self, theta: np.ndarray, y: np.ndarray) -> None:
        """Update surrogate with a high-fidelity (HF) observation."""


class HF(Protocol):
    """Protocol for a high-fidelity forward model."""

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        """Return high-fidelity model prediction."""


class ActiveMCMCModel:
    """
    MCMC model wrapper providing coarse (surrogate) and fine (HF) evaluations.
    Updates the surrogate whenever HF is evaluated.
    """

    def __init__(
        self,
        lf: ActiveLF,
        hf: HF,
        gamma_var: float,
        update_fine: bool = True,
    ):
        self.lf = lf
        self.hf = hf
        self.gamma_var = gamma_var
        self.update_fine = update_fine

        self.n_hf = 0
        self.used_hf: list[bool] = []

    def coarse(self, theta: np.ndarray) -> np.ndarray:
        """Return surrogate prediction, optionally calling HF if uncertainty is high."""
        y_pred, var = self.lf.predict(theta)
        u_bar = float(np.mean(var))

        if u_bar > self.gamma_var:
            y = self.hf(theta)
            self.update_lf(theta, y)
            self.n_hf += 1
            self.used_hf.append(True)
            return y

        self.used_hf.append(False)
        return CoarseOutput(y_pred, var)

    def fine(self, theta: np.ndarray) -> np.ndarray:
        """Return HF prediction, optionally updating the surrogate."""
        y = self.hf(theta)
        if self.update_fine:
            self.update_lf(theta, y)
        return y

    def update_lf(self, theta: np.ndarray, y: np.ndarray) -> None:
        """Update surrogate with a new HF observation."""
        self.lf.update(theta, y)


class AdaptiveActiveMCMCModel(ActiveMCMCModel):
    """
    Adaptive MCMC model that adjusts the HF subchain length based
    on surrogate prediction error.
    Subchain length increases when surrogate error is small and
    wdecreases when error is high.
    """

    def __init__(
        self,
        lf: ActiveLF,
        hf: HF,
        gamma_var: float,
        initial_subchain: int = 10,
        adapt_rate: float = 0.1,
        max_err_hist: int = 50,
        update_every: int = 1,
        target_error: float = 0.01,
        min_subchain: int = 1,
        max_subchain: int = 100,
        update_fine: bool = True,
        max_steps: int | None = None,
    ):
        super().__init__(lf, hf, gamma_var, update_fine)
        self.subchain_length = initial_subchain
        self.adapt_rate = adapt_rate
        self.hf_errors: list[float] = []
        self.max_err_hist = max_err_hist
        self.update_every = update_every
        self.target_error = target_error
        self.min_subchain = min_subchain
        self.max_subchain = max_subchain
        self.max_steps = max_steps
        self.total_steps = 0
        self.subchain_lengths = []
        self.subsample_rate = 1 / self.subchain_length

    def coarse(self, theta):
        self.total_steps += 1
        self.subchain_lengths.append(self.subchain_length)
        print(self.subchain_length)
        return super().coarse(theta)

    def fine(self, theta: np.ndarray):
        if self.max_steps is not None and self.total_steps >= self.max_steps:
            y_pred, _ = self.lf.predict(theta)
            return y_pred

        y = self.hf(theta)
        if self.update_fine:
            self.update_lf(theta, y)
        return y

    def record_error(self, error: float) -> None:
        """Record HF prediction error and adapt subchain length accordingly."""
        self.hf_errors.append(error)
        print(error)
        if len(self.hf_errors) > self.max_err_hist:
            self.hf_errors.pop(0)

        if self.total_steps % self.update_every != 0 or len(self.hf_errors) <= 5:
            return

        # normalized prediction error relative to target
        npe = np.mean(self.hf_errors) / self.target_error
        delta = np.clip(npe - 1.0, -1.0, 1.0)

        self.subsample_rate *= np.exp(self.adapt_rate * delta)
        self.subsample_rate = np.clip(
            self.subsample_rate, 1.0 / self.max_subchain, 1.0 / self.min_subchain
        )
        self.subchain_length = int(1.0 / self.subsample_rate)

    def update_lf(self, theta: np.ndarray, y: np.ndarray) -> None:
        """Update surrogate and record the prediction error for adaptive control."""
        y_pred, _ = self.lf.predict(theta)
        self.lf.update(theta, y)
        self.record_error(np.mean(np.abs(y_pred - y)))
