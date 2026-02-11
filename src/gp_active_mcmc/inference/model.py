from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gp_active_mcmc.inference.coarse_output import CoarseOutput
from gp_active_mcmc.protocols import ActiveSurrogate, HighFidelityModel

FloatArray = NDArray[np.float64]


def _as_1d_theta(theta: ArrayLike) -> FloatArray:
    """Convert a parameter vector to a 1D float array.

    This helper enforces the basic contract used throughout the inference API:
    parameters are treated as a 1D numeric vector.

    Parameters
    ----------
    theta
        Candidate parameter vector. Any array-like is accepted as long as it can
        be converted to a 1D array.

    Returns
    -------
    theta_1d
        1D array of dtype float.

    Raises
    ------
    ValueError
        If `theta` is not one-dimensional after conversion.
    """
    arr = np.asarray(theta, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"theta must be 1D. Got shape {arr.shape}.")
    return arr


def _as_1d_float(x: ArrayLike, *, name: str) -> FloatArray:
    """Convert an array-like to a 1D float array (ravelled).

    This is used to normalise outputs from forward models and surrogates into a
    consistent 1D representation (e.g., a trajectory sampled on a time grid).

    Parameters
    ----------
    x
        Input array-like.
    name
        Name used in error messages to make failures easier to diagnose.

    Returns
    -------
    x_1d
        1D array of dtype float.

    Raises
    ------
    ValueError
        If the result is not one-dimensional after ravel.
    """
    arr = np.asarray(x, dtype=float).ravel()
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D after ravel. Got shape {arr.shape}.")
    return arr


@dataclass(slots=True)
class EvaluationLog:
    """Minimal evaluation log for [`ActiveMCMCModel`][gp_active_mcmc.inference.model.ActiveMCMCModel].

    The active model records whether each *coarse* evaluation used the HF model.
    This metadata is useful for diagnostics (HF fraction, where corrections occur)
    and for sampler bookkeeping.

    Notes
    -----
    The log is aligned with calls to
    [`ActiveMCMCModel.coarse`][gp_active_mcmc.inference.model.ActiveMCMCModel.coarse].
    If a sampler performs a fine correction after a coarse step at the same MCMC iteration,
    the fine step can overwrite the last entry via [`replace_last`][gp_active_mcmc.inference.model.EvaluationLog.replace_last].

    Attributes
    ----------
    used_hf
        Boolean flag per coarse evaluation:

        - False: LF surrogate was used (no HF correction)
        - True: HF model was used (either triggered in `coarse` or via `fine`)
    """

    used_hf: list[bool] = field(default_factory=list)

    def append(self, used_hf: bool) -> None:
        """Append a boolean HF-usage flag."""
        self.used_hf.append(bool(used_hf))

    def replace_last(self, used_hf: bool) -> None:
        """Replace the most recent HF-usage flag.

        If the log is empty, this method appends a new entry. This behaviour
        is convenient for samplers that call [`ActiveMCMCModel.fine`][gp_active_mcmc.inference.model.ActiveMCMCModel.fine]
        as a correction following a prior coarse evaluation in the same MCMC step.
        """
        if self.used_hf:
            self.used_hf[-1] = bool(used_hf)
        else:
            self.append(bool(used_hf))


class AdaptiveHook(Protocol):
    """Callback interface for adaptive policies (e.g., adaptive subchains).

    [`ActiveMCMCModel`][gp_active_mcmc.inference.model.ActiveMCMCModel] is responsible for
    coupling LF and HF. Adaptive logic (such as choosing a changing subchain length) is
    expressed as an external hook so that the active model remains small and testable.

    The hook receives notifications during model evaluations:

    - [`on_coarse_call`][gp_active_mcmc.inference.model.AdaptiveHook.on_coarse_call] is called at the start of
      [`ActiveMCMCModel.coarse`][gp_active_mcmc.inference.model.ActiveMCMCModel.coarse].
    - [`on_fine_call`][gp_active_mcmc.inference.model.AdaptiveHook.on_fine_call] is called inside
      [`ActiveMCMCModel.fine`][gp_active_mcmc.inference.model.ActiveMCMCModel.fine], **before**
      updating the surrogate with the new HF observation.

    Notes
    -----
    The hook is deliberately narrow: it observes what happened and updates its own
    state; it does not perform I/O and it does not change the model outputs directly.

    See Also
    --------
    [`AdaptiveSubchain`][gp_active_mcmc.inference.adaptive_subchain.AdaptiveSubchain]
        Default adaptive policy provided by the library.
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
            HF model output at the current `theta`.
        y_lf
            LF predictive mean at the current `theta`, computed **before** the LF
            model is updated with `(theta, y_hf)`.
        """


@dataclass(slots=True)
class ActiveMCMCModel:
    """Couple a low-fidelity surrogate (LF) with a high-fidelity model (HF).

    [`ActiveMCMCModel`][gp_active_mcmc.inference.model.ActiveMCMCModel] is the core component
    of the library. It implements the LF/HF coupling required by active-learning MCMC workflows
    and exposes two callables intended to be used in MCMC posteriors:

    - [`ActiveMCMCModel.coarse`][gp_active_mcmc.inference.model.ActiveMCMCModel.coarse] —
      LF-first evaluation with an uncertainty trigger that may fall back to HF.
    - [`ActiveMCMCModel.fine`][gp_active_mcmc.inference.model.ActiveMCMCModel.fine] —
      HF evaluation (always) and surrogate update.

    In practice, users typically:

    1. Build an LF surrogate (e.g., POD-GP).
    2. Wrap LF + HF in an `ActiveMCMCModel`.
    3. Choose an inference mode by deciding which posterior(s) to pass to a sampler.
    4. Run a sampler and analyse both samples and HF-usage diagnostics.

    Choosing the inference mode
    ---------------------------
    The *posterior argument* determines how the sampler interacts with the active model.

    **Single posterior (MCMC-guided active learning)**
        Pass a single posterior using `model.coarse` as the forward model. HF calls happen
        internally whenever the uncertainty trigger activates.

        - `posterior = Posterior(prior, loglike, model.coarse)`
        - sampler:
          [`sample_active_chain`][gp_active_mcmc.inference.sampling.sample_active_chain]

    **Two posteriors (DA-MCMC guided active learning)**
        Pass two posteriors: coarse (LF-first) and fine (HF). This corresponds to
        delayed-acceptance MCMC (DA-MCMC).

        - `posterior = [Posterior(..., model.coarse), Posterior(..., model.fine)]`
        - sampler:
          [`sample_active_chain`][gp_active_mcmc.inference.sampling.sample_active_chain]

    **Adaptive DA-MCMC (recommended)**
        Use DA-MCMC (two posteriors) and pass an adaptive subchain policy via `adaptive=...`.
        The adaptive policy monitors LF-HF discrepancy and adjusts how often fine (HF)
        corrections are applied.

        - DA-MCMC is mandatory: you must use two posteriors.
        - adaptive policy:
          [`AdaptiveSubchain`][gp_active_mcmc.inference.adaptive_subchain.AdaptiveSubchain]
        - sampler:
          [`sample_adaptive_active_chain`][gp_active_mcmc.inference.sampling.sample_adaptive_active_chain]

    Parameters
    ----------
    lf_model
        Low-fidelity surrogate implementing:

        - `predict(theta) -> (mean, var)` where both arrays have shape `(n_obs,)`
        - `update(theta, y_hf)` to incorporate new HF evaluations

    hf_model
        High-fidelity forward model callable as `hf_model(theta) -> y` where `y` has shape `(n_obs,)`.
    gamma_threshold
        Uncertainty threshold used by [`coarse`][gp_active_mcmc.inference.model.ActiveMCMCModel.coarse].
        A coarse call triggers HF if `mean(y_var) > gamma_threshold**2`.
    log
        Evaluation log used to record HF usage aligned with coarse evaluations.
        See [`EvaluationLog`][gp_active_mcmc.inference.model.EvaluationLog].
    adaptive
        Optional adaptive hook (e.g., adaptive subchain logic). When provided, the hook is notified
        during coarse and fine evaluations. See [`AdaptiveHook`][gp_active_mcmc.inference.model.AdaptiveHook].

    Returns and types
    -----------------
    - [`coarse`][gp_active_mcmc.inference.model.ActiveMCMCModel.coarse] returns either:
      - a [`CoarseOutput`][gp_active_mcmc.inference.coarse_output.CoarseOutput] if LF is used, or
      - a 1D numpy array (HF output) if HF is triggered.
    - [`fine`][gp_active_mcmc.inference.model.ActiveMCMCModel.fine] always returns a 1D numpy array (HF output).

    Notes
    -----
    The uncertainty trigger in `coarse` is intentionally simple and cheap: it uses the mean predictive
    variance over outputs as a scalar criterion.

    See Also
    --------
    [`ActiveGPLogLike`][gp_active_mcmc.inference.likelihood.ActiveGPLogLike]
        Likelihood that inflates observation covariance when `CoarseOutput.variance` is present.
    [`ChunkedMCMCConfig`][gp_active_mcmc.inference.sampling.ChunkedMCMCConfig]
        Chunking configuration required for adaptive subchain sampling.
    """

    lf_model: ActiveSurrogate
    hf_model: HighFidelityModel
    gamma_threshold: float
    log: EvaluationLog = field(default_factory=EvaluationLog)
    adaptive: AdaptiveHook | None = None

    def __post_init__(self) -> None:
        if self.gamma_threshold < 0.0:
            raise ValueError("gamma_threshold must be non-negative.")

    def coarse(self, theta: ArrayLike) -> FloatArray | CoarseOutput:
        """Evaluate the coupled model in LF-first (coarse) mode.

        Workflow
        --------
        1. Compute LF predictive mean and variance at `theta`.
        2. If LF uncertainty is large (`mean(var) > gamma_threshold**2`), evaluate HF.
        3. If HF was used, update the LF surrogate with `(theta, y_hf)`.
        4. Record HF usage in `log.used_hf`.

        Parameters
        ----------
        theta
            Parameter vector of shape `(n_dim,)`.

        Returns
        -------
        out
            If LF is used, returns a [`CoarseOutput`][gp_active_mcmc.inference.coarse_output.CoarseOutput]
            containing `(mean, variance)` where both are 1D arrays of shape `(n_obs,)`.

            If HF is triggered, returns the HF output as a 1D numpy array of shape `(n_obs,)`.

        Raises
        ------
        ValueError
            If the surrogate returns mean/variance arrays with inconsistent shapes.

        See Also
        --------
        [`ActiveGPLogLike`][gp_active_mcmc.inference.likelihood.ActiveGPLogLike]
            Uses `CoarseOutput.variance` to inflate the observation covariance.
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

    def fine(self, theta: ArrayLike, *, replace_last: bool = True) -> FloatArray:
        """Evaluate the coupled model in HF (fine) mode and update the surrogate.

        This method always evaluates the HF model and then updates the LF surrogate.
        It is typically used as the *fine* level in DA-MCMC, either periodically or
        according to an adaptive subchain policy.

        Parameters
        ----------
        theta
            Parameter vector of shape `(n_dim,)`.
        replace_last
            If True, replace the most recent entry in `log.used_hf` with True.
            This is convenient when `fine()` is called as a correction following a
            prior `coarse()` evaluation at the same MCMC step.
            If False, append a new entry to the log.

        Returns
        -------
        y_hf
            HF model output as a 1D numpy array of shape `(n_obs,)`.

        See Also
        --------
        [`AdaptiveSubchain`][gp_active_mcmc.inference.adaptive_subchain.AdaptiveSubchain]
            Adaptive policy notified during fine evaluations (LF-HF discrepancy monitoring).
        """
        th = _as_1d_theta(theta)

        y_hf = _as_1d_float(self.hf_model(th), name="y_hf")

        if self.adaptive is not None:
            lf_mean, _ = self.lf_model.predict(th)
            lf_mean = _as_1d_float(lf_mean, name="lf_mean_before_update")
            self.adaptive.on_fine_call(y_hf=y_hf, y_lf=lf_mean)

        self.lf_model.update(th, y_hf)

        if replace_last:
            self.log.replace_last(True)
        else:
            self.log.append(True)

        return y_hf
