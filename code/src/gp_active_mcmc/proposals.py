import copy
import numpy as np
from abc import ABC, abstractmethod

# =========================================
# Proposal classes
# =========================================

class BaseProposal(ABC):
    """Base class for proposal strategies."""

    @abstractmethod
    def propose(self, theta: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def update(self, theta_new: np.ndarray, accepted: bool) -> None:
        pass

    def copy(self) -> "BaseProposal":
        """Return a deep copy of the proposal."""
        return copy.deepcopy(self)


class RWMProposal(BaseProposal):
    def __init__(self, cov: np.ndarray, step_scale: float = 0.1) -> None:
        self.cov = cov
        self.step_scale = step_scale

    def propose(self, theta: np.ndarray) -> np.ndarray:
        return theta + np.random.multivariate_normal(np.zeros_like(theta), self.step_scale**2 * self.cov)

    def update(self, theta_new: np.ndarray, accepted: bool) -> None:
        pass  # no adaptation


class AdaptiveRWMProposal(RWMProposal):
    def __init__(self, cov: np.ndarray, step_scale: float = 0.1, target_accept: float = 0.25, adapt_window: int = 50) -> None:
        super().__init__(cov, step_scale)
        self.target_accept = target_accept
        self.adapt_window = adapt_window
        self.mean_theta = None
        self.n_updates = 0
        self.history = []

    def propose(self, theta: np.ndarray) -> np.ndarray:
        return theta + np.random.multivariate_normal(np.zeros_like(theta), self.step_scale**2 * self.cov)

    def update(self, theta_new: np.ndarray, accepted: bool) -> None:
        self.n_updates += 1
        if self.mean_theta is None:
            self.mean_theta = theta_new.copy()
            return
        delta = theta_new - self.mean_theta
        self.mean_theta += delta / self.n_updates
        self.cov += np.outer(delta, theta_new - self.mean_theta)

        self.history.append(accepted)
        if len(self.history) >= self.adapt_window:
            local_acc = np.mean(self.history[-self.adapt_window:])
            gamma = 1.0 / np.sqrt(self.n_updates)
            self.step_scale *= np.exp(gamma * (local_acc - self.target_accept))
            self.step_scale = np.clip(self.step_scale, 1e-5, 10.0)

