from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class AdaptiveControl:
    adapt_rate: float = 0.1
    update_every: int = 100
    target_error: float = 0.01
    min_subchain: int = 1
    max_subchain: int = 100
    max_steps: Optional[int] = None


@dataclass
class AdaptiveState:
    total_steps: int = 0
    subchain_length: int = 10
    subsample_rate: float = 0.1
    hf_errors: deque[float] = field(default_factory=lambda: deque(maxlen=50))
    subchain_history: list[int] = field(default_factory=list)

    def step(self):
        self.total_steps += 1

    def append_length(self):
        self.subchain_history.append(self.subchain_length)

    def to_update(self, control: AdaptiveControl) -> bool:
        return self.total_steps % control.update_every == 0 and len(self.hf_errors) > 5

    def append_error(self, y_pred: np.ndarray, y: np.ndarray):
        self.hf_errors.append(np.mean(np.abs(y_pred - y)))

    def update_subchain(self, control: AdaptiveControl):
        if self.to_update(control):
            print("I am adapting")
            normalized_error = np.mean(self.hf_errors) / control.target_error
            delta = np.clip(normalized_error - 1.0, -1.0, 1.0)
            self.subsample_rate *= np.exp(control.adapt_rate * delta)
            self.subsample_rate = np.clip(
                self.subsample_rate,
                1.0 / control.max_subchain,
                1.0 / control.min_subchain,
            )
            self.subchain_length = int(1.0 / self.subsample_rate)
