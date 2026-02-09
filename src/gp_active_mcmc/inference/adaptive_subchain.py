from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating]


@dataclass(frozen=True, slots=True)
class AdaptiveSubchainControl:
    """Control parameters for adaptive HF subchain length.

    Semantics
    ---------
    The adaptive mechanism adjusts the HF subchain length based on an error
    statistic computed when HF is evaluated.

    - If the error is above `target_error`, the subchain length is reduced
      (more frequent HF corrections).
    - If the error is below `target_error`, the subchain length is increased
      (less frequent HF corrections).

    Updates occur every `update_every` HF steps.
    """

    update_every: int = 10
    target_error: float = 0.01
    min_subchain: int = 1
    max_subchain: int = 10_000
    grow_factor: float = 2.0
    shrink_factor: float = 0.5

    def __post_init__(self) -> None:
        if self.update_every <= 0:
            raise ValueError("update_every must be positive.")
        if self.target_error < 0.0:
            raise ValueError("target_error must be non-negative.")
        if self.min_subchain <= 0:
            raise ValueError("min_subchain must be positive.")
        if self.max_subchain < self.min_subchain:
            raise ValueError("max_subchain must be >= min_subchain.")
        if self.grow_factor <= 1.0:
            raise ValueError("grow_factor must be > 1.0.")
        if not (0.0 < self.shrink_factor < 1.0):
            raise ValueError("shrink_factor must be in (0, 1).")


@dataclass(slots=True)
class AdaptiveSubchainState:
    """Mutable adaptive state for HF subchain control.

    Parameters
    ----------
    subchain_length
        Current HF subchain length (how many coarse steps are taken between HF evaluations).

    Stored history
    --------------
    subchain_history
        Records subchain lengths at each coarse call.
    hf_errors
        Records the HF-vs-LF error statistic computed at each fine call.
    total_hf_steps
        Counts the number of HF evaluations.

    Notes
    -----
    The error statistic used here is RMSE between LF mean and HF output.
    """

    subchain_length: int = 10
    subchain_history: list[int] = field(default_factory=list)
    hf_errors: list[float] = field(default_factory=list)
    total_hf_steps: int = 0

    # internal counter for update scheduling
    _hf_since_update: int = 0

    def __post_init__(self) -> None:
        if self.subchain_length <= 0:
            raise ValueError("subchain_length must be positive.")

    def append_length(self) -> None:
        """Record the current subchain length (called at each coarse evaluation)."""
        self.subchain_history.append(int(self.subchain_length))

    def step(self) -> None:
        """Advance the HF step counters (called exactly once per fine evaluation)."""
        self.total_hf_steps += 1
        self._hf_since_update += 1

    def append_error(self, lf_mean: FloatArray, y_hf: FloatArray) -> None:
        """Compute and record an error statistic between LF mean and HF output."""
        lf = np.asarray(lf_mean, dtype=float).ravel()
        hf = np.asarray(y_hf, dtype=float).ravel()
        if lf.shape != hf.shape:
            raise ValueError(f"lf_mean and y_hf must have same shape. Got {lf.shape} vs {hf.shape}.")

        rmse = float(np.sqrt(np.mean((lf - hf) ** 2)))
        self.hf_errors.append(rmse)

    def update_subchain(self, control: AdaptiveSubchainControl) -> None:
        """Update subchain length according to the control policy (periodic)."""
        if self._hf_since_update < control.update_every:
            return
        if not self.hf_errors:
            return

        err = float(self.hf_errors[-1])

        if err > control.target_error:
            new_len = int(np.floor(self.subchain_length * control.shrink_factor))
        else:
            new_len = int(np.ceil(self.subchain_length * control.grow_factor))

        new_len = max(control.min_subchain, min(control.max_subchain, new_len))
        self.subchain_length = int(new_len)

        # reset scheduling counter
        self._hf_since_update = 0

@dataclass(slots=True)
class AdaptiveSubchain:
    """Adaptive subchain length driven by AdaptiveState/AdaptiveControl.

    Semantics
    ---------
    - Each coarse call records the current subchain length (history).
    - Each fine call:
      - appends a prediction error (LF mean vs HF),
      - updates the subchain length policy,
      - increments total HF steps.
    """

    state: AdaptiveSubchainState
    control: AdaptiveSubchainControl

    def on_coarse_call(self) -> None:
        self.state.append_length()

    def on_fine_call(self, *, y_hf: FloatArray, y_lf: FloatArray) -> None:
        self.state.append_error(y_lf, y_hf)
        self.state.update_subchain(self.control)
        self.state.step()