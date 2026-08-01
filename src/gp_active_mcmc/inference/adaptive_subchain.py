from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AdaptiveSubchainControl:
    """Hyperparameters for adaptive subchain-length control.

    This control block specifies *how* the adaptive policy reacts to observed LF-HF
    discrepancy during sampling.

    In DA-MCMC guided active learning, the sampler periodically performs a **fine**
    (HF) correction. The number of coarse steps taken between fine corrections is the
    **subchain length** (often called the subsampling rate).

    The update rule is:

    - if the most recent LF-HF error is **above** `target_error`, decrease the subchain
      length (more frequent HF corrections);
    - if the error is **below** `target_error`, increase the subchain length
      (less frequent HF corrections).

    Updates are attempted every `update_every` HF evaluations.

    Parameters
    ----------
    update_every
        Number of HF evaluations between policy updates. Must be positive.
    target_error
        Target level for the LF-HF error statistic (non-negative).
    min_subchain
        Lower bound on the subchain length (minimum spacing between HF corrections is 1).
    max_subchain
        Upper bound on the subchain length.
    grow_factor
        Multiplicative factor used when error is below target. Must be > 1.
    shrink_factor
        Multiplicative factor used when error is above target. Must be in (0, 1).
    patience
        Number of *consecutive* policy updates with error at or below `target_error`
        required before the adaptive phase is considered converged (see
        [`AdaptiveSubchainState.has_converged`][gp_active_mcmc.inference.adaptive_subchain.AdaptiveSubchainState.has_converged]).
        Must be positive. Because the streak only advances on policy-update events (which
        themselves only fire after `update_every` HF evaluations), `patience` also acts as
        an implicit warm-up: convergence cannot be declared before at least `patience`
        updates have been observed.

    Notes
    -----
    The default values are tuned for toy problems. In real applications, `target_error`
    should reflect acceptable surrogate error in the observation space used by the likelihood.

    See Also
    --------
    [AdaptiveSubchain][gp_active_mcmc.inference.adaptive_subchain.AdaptiveSubchain]
        Adaptive policy that uses this control block.
    [ActiveMCMCModel][gp_active_mcmc.inference.model.ActiveMCMCModel]
        The active model that calls the policy hooks during coarse/fine evaluations.
    """

    update_every: int = 10
    target_error: float = 0.01
    min_subchain: int = 1
    max_subchain: int = 10_000
    grow_factor: float = 2.0
    shrink_factor: float = 0.5
    patience: int = 5

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
        if self.patience <= 0:
            raise ValueError("patience must be positive.")


@dataclass(slots=True)
class AdaptiveSubchainState:
    """Mutable state for adaptive subchain control.

    This object stores the *current* subchain length and the histories needed for
    diagnostics and adaptation.

    Parameters
    ----------
    subchain_length
        Current subchain length (number of coarse steps between HF corrections).
        Must be positive.

    Attributes
    ----------
    subchain_history
        Records the subchain length at each coarse evaluation (aligned with coarse calls).
    hf_errors
        LF-HF error values computed at each fine evaluation.
    total_hf_steps
        Total number of HF evaluations performed.
    stable_streak
        Number of consecutive policy updates (see `update_subchain`) whose error was at
        or below `target_error`. Reset to 0 on any update above target. Used by
        [`has_converged`][gp_active_mcmc.inference.adaptive_subchain.AdaptiveSubchainState.has_converged]
        to decide when the adaptive phase can be terminated and the surrogate frozen.
    n_updates
        Total number of policy-update events that have occurred.
    _hf_since_update
        Internal counter tracking HF evaluations since the last update.

    Notes
    -----
    The error statistic used here is **RMSE** between the LF predictive mean and the HF
    output in observation space:

    $$
        \\mathrm{RMSE}(\\mu_{\\mathrm{LF}}, y_{\\mathrm{HF}}) =
        \\sqrt{\\frac{1}{n}\\sum_{i=1}^{n}(\\mu_{\\mathrm{LF}, i} - y_{\\mathrm{HF}, i})^2}.
    $$

    See Also
    --------
    [AdaptiveSubchainControl][gp_active_mcmc.inference.adaptive_subchain.AdaptiveSubchainControl]
        Control parameters used to update the subchain length.
    """

    subchain_length: int = 10
    subchain_history: list[int] = field(default_factory=list)
    hf_errors: list[float] = field(default_factory=list)
    total_hf_steps: int = 0
    stable_streak: int = 0
    n_updates: int = 0

    _hf_since_update: int = 0

    def __post_init__(self) -> None:
        if self.subchain_length <= 0:
            raise ValueError("subchain_length must be positive.")

    def append_length(self) -> None:
        """Record the current subchain length.

        This should be called exactly once per *coarse* evaluation so that
        `subchain_history` remains aligned with coarse calls.
        """
        self.subchain_history.append(int(self.subchain_length))

    def step(self) -> None:
        """Advance HF counters.

        This should be called exactly once per *fine* evaluation.
        """
        self.total_hf_steps += 1
        self._hf_since_update += 1

    def append_error(self, lf_mean: FloatArray, y_hf: FloatArray) -> None:
        """Compute and record an LF-HF error statistic (RMSE).

        Parameters
        ----------
        lf_mean
            LF predictive mean at the current parameter value (before surrogate update).
        y_hf
            HF model output at the current parameter value.

        Raises
        ------
        ValueError
            If `lf_mean` and `y_hf` have different shapes.
        """
        lf = np.asarray(lf_mean, dtype=float).ravel()
        hf = np.asarray(y_hf, dtype=float).ravel()
        if lf.shape != hf.shape:
            raise ValueError(
                f"lf_mean and y_hf must have same shape. Got {lf.shape} vs {hf.shape}."
            )

        rmse = float(np.sqrt(np.mean((lf - hf) ** 2)))
        self.hf_errors.append(rmse)

    def update_subchain(self, control: AdaptiveSubchainControl) -> None:
        """Update the subchain length according to `control`.

        The update is performed only when:

        - at least `control.update_every` HF evaluations have occurred since the last update, and
        - at least one error value is available.

        Parameters
        ----------
        control
            Policy hyperparameters controlling update frequency, bounds, and scaling.
        """
        if self._hf_since_update < control.update_every:
            return
        if not self.hf_errors:
            return

        err = float(self.hf_errors[-1])

        if err > control.target_error:
            new_len = int(np.floor(self.subchain_length * control.shrink_factor))
            self.stable_streak = 0
        else:
            new_len = int(np.ceil(self.subchain_length * control.grow_factor))
            self.stable_streak += 1

        new_len = max(control.min_subchain, min(control.max_subchain, new_len))
        self.subchain_length = int(new_len)

        self.n_updates += 1
        self._hf_since_update = 0

    def has_converged(self, control: AdaptiveSubchainControl) -> bool:
        """Whether the adaptive phase has converged and the surrogate can be frozen.

        Convergence is declared once the LF-HF error has stayed at or below
        `control.target_error` for `control.patience` *consecutive* policy updates
        (see [`update_subchain`][gp_active_mcmc.inference.adaptive_subchain.AdaptiveSubchainState.update_subchain]).

        Parameters
        ----------
        control
            Policy hyperparameters, in particular `patience`.

        Returns
        -------
        converged
            True if the stable streak has reached `control.patience`.
        """
        return self.stable_streak >= control.patience


@dataclass(slots=True)
class AdaptiveSubchain:
    """Adaptive subchain policy for DA-MCMC guided active learning.

    `AdaptiveSubchain` implements the hook interface
    [AdaptiveHook][gp_active_mcmc.inference.model.AdaptiveHook] used by
    [ActiveMCMCModel][gp_active_mcmc.inference.model.ActiveMCMCModel].

    Semantics
    ---------
    - On each **coarse** call: record the current subchain length.
    - On each **fine** call:
      1. compute and store LF-HF error (currently RMSE in observation space),
      2. update the subchain length if the update schedule triggers,
      3. increment HF counters.

    Intended usage
    --------------
    This policy is intended to be used with:

    - **two posteriors** `[coarse, fine]` (DA-MCMC is mandatory), and
    - a chunked sampler such as
      [sample_adaptive_active_chain][gp_active_mcmc.inference.sampling.sample_adaptive_active_chain],
      because `tinyDA` requires a fixed subsampling rate within each call.

    Parameters
    ----------
    state
        Mutable adaptive state (current length, histories, counters).
    control
        Control parameters (targets, bounds, and update schedule).

    See Also
    --------
    [ActiveMCMCModel][gp_active_mcmc.inference.model.ActiveMCMCModel]
        The active model that triggers `on_coarse_call` / `on_fine_call`.
    [ChunkedMCMCConfig][gp_active_mcmc.inference.sampling.ChunkedMCMCConfig]
        Chunk configuration used by the adaptive sampler.
    """

    state: AdaptiveSubchainState
    control: AdaptiveSubchainControl

    def on_coarse_call(self) -> None:
        """Hook called by the active model at the start of a coarse evaluation."""
        self.state.append_length()

    def on_fine_call(self, *, y_hf: FloatArray, y_lf: FloatArray) -> None:
        """Hook called by the active model during fine evaluation.

        Parameters
        ----------
        y_hf
            HF output at the current parameter value.
        y_lf
            LF predictive mean at the current parameter value (before surrogate update).
        """
        self.state.append_error(y_lf, y_hf)
        self.state.update_subchain(self.control)
        self.state.step()

    def has_converged(self) -> bool:
        """Whether the adaptive phase has converged (see `AdaptiveSubchainState.has_converged`)."""
        return self.state.has_converged(self.control)
