from __future__ import annotations

import copy
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
        A coarse call triggers HF if `mean(y_var) > gamma_threshold**2`. Ignored when `frozen=True`.
    log
        Evaluation log used to record HF usage aligned with coarse evaluations.
        See [`EvaluationLog`][gp_active_mcmc.inference.model.EvaluationLog].
    adaptive
        Optional adaptive hook (e.g., adaptive subchain logic). When provided, the hook is notified
        during coarse and fine evaluations. See [`AdaptiveHook`][gp_active_mcmc.inference.model.AdaptiveHook].
        Should be `None` when `frozen=True`: a frozen model is no longer adapting anything, so
        there is nothing for the hook to observe.
    frozen
        If True, the model no longer learns from HF evaluations: `coarse` always returns the
        surrogate prediction (the `gamma_threshold` fallback is disabled) and neither `coarse`
        nor `fine` call `lf_model.update(...)`. This turns the pair `(coarse, fine)` into a
        genuine fixed-kernel delayed-acceptance model — `fine` still evaluates the HF model for
        the DA correction step, it just stops teaching the surrogate. Use
        [`freeze`][gp_active_mcmc.inference.model.ActiveMCMCModel.freeze] to obtain a frozen
        copy of an adapted model rather than setting this by hand.

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
    frozen: bool = False

    # Memoizes the most recent `coarse`/`fine` call so a repeat at the identical `theta`
    # can skip real work instead of redoing it -- see the guards inside `coarse`/`fine`
    # for why this repeat happens routinely (every chunk boundary in this library's
    # chunked round loops) and what it costs when left unguarded. `compare=False` /
    # `repr=False`: these are a cache, not part of this model's value -- comparing two
    # `ActiveMCMCModel`s or printing one shouldn't depend on, or dump, cached arrays.
    _last_coarse_theta: FloatArray | None = field(default=None, repr=False, compare=False)
    _last_coarse_output: FloatArray | CoarseOutput | None = field(default=None, repr=False, compare=False)
    _last_fine_theta: FloatArray | None = field(default=None, repr=False, compare=False)
    _last_fine_output: FloatArray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.gamma_threshold < 0.0:
            raise ValueError("gamma_threshold must be non-negative.")

    def freeze(self) -> ActiveMCMCModel:
        """Return a frozen copy of this model, snapshotting the current surrogate state.

        The returned model shares the same `hf_model` and `gamma_threshold`, but:

        - `lf_model` is a deep copy of the current surrogate, so further training of `self`
          (e.g. if sampling continues) does not affect the frozen copy;
        - `frozen=True`, so neither `coarse` nor `fine` update the surrogate, and `coarse`
          never falls back to HF;
        - `adaptive=None` and `log` is a fresh `EvaluationLog`, since a frozen model has
          nothing left to adapt and its HF usage should be tracked independently of the
          adaptive phase that produced it.

        This is the mechanism used to transition from the adaptive online stage to the
        fixed-kernel production stage: run adaptively until convergence, call `freeze()`,
        then sample with the frozen model and a fixed subsampling rate using
        [`sample_active_chain`][gp_active_mcmc.inference.sampling.sample_active_chain].
        """
        return ActiveMCMCModel(
            lf_model=copy.deepcopy(self.lf_model),
            hf_model=self.hf_model,
            gamma_threshold=self.gamma_threshold,
            frozen=True,
        )

    def coarse(self, theta: ArrayLike) -> FloatArray | CoarseOutput:
        """Evaluate the coupled model in LF-first (coarse) mode.

        Workflow
        --------
        1. Compute LF predictive mean and variance at `theta`.
        2. If `frozen`, always return the surrogate prediction (steps 3-4 below are skipped).
           Otherwise, if LF uncertainty is large (`mean(var) > gamma_threshold**2`), evaluate HF.
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

        if self._last_coarse_theta is not None and np.array_equal(th, self._last_coarse_theta):
            # Repeat of the most recently evaluated point -- see `fine`'s matching guard
            # for why this happens routinely and what it's protecting against. Cheaper
            # here than in `fine` (no HF call at stake unless this repeat would again
            # trigger the HF fallback below, which a point the surrogate was *just*
            # queried at is unlikely to need), but still worth skipping: it avoids a
            # spurious `lf_model.update()` call -- teaching the surrogate a duplicate
            # training point -- whenever the fallback does trigger.
            # `coarse` only ever returns a bare `y_hf` array (HF fallback) or a
            # `CoarseOutput` (LF used); the log entry mirrors which one this was.
            self.log.append(not isinstance(self._last_coarse_output, CoarseOutput))
            return self._last_coarse_output  # type: ignore[return-value]

        y_mean, y_var = self.lf_model.predict(th)
        mean = _as_1d_float(y_mean, name="y_mean")
        var = _as_1d_float(y_var, name="y_var")

        if mean.shape != var.shape:
            raise ValueError(
                f"Surrogate returned mean/var with different shapes: {mean.shape} vs {var.shape}."
            )

        if self.frozen:
            self.log.append(False)
            out: FloatArray | CoarseOutput = CoarseOutput(mean, var)
            self._last_coarse_theta, self._last_coarse_output = th.copy(), out
            return out

        avg_var = float(np.mean(var))
        if avg_var > self.gamma_threshold**2:
            y_hf = _as_1d_float(self.hf_model(th), name="y_hf")
            self.lf_model.update(th, y_hf)
            self.log.append(True)
            self._last_coarse_theta, self._last_coarse_output = th.copy(), y_hf
            return y_hf

        self.log.append(False)
        out = CoarseOutput(mean, var)
        self._last_coarse_theta, self._last_coarse_output = th.copy(), out
        return out

    def fine(self, theta: ArrayLike, *, replace_last: bool = True) -> FloatArray:
        """Evaluate the coupled model in HF (fine) mode and update the surrogate.

        This method always evaluates the HF model, then updates the LF surrogate unless
        `frozen=True`. It is typically used as the *fine* level in DA-MCMC, either
        periodically or according to an adaptive subchain policy. When `frozen=True`, this
        becomes the HF correction step of a standard (non-learning) delayed-acceptance
        chain: the surrogate no longer changes, but HF is still evaluated for the
        acceptance ratio.

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

        if self._last_fine_theta is not None and np.array_equal(th, self._last_fine_theta):
            # This library's chunked round loops (`rounds.py`'s `_run_chunk`, and
            # `sample_adaptive_active_chain`'s own chunking) re-enter tinyDA by
            # constructing a fresh `Chain`/`DAChain` every chunk. `tinyDA.DAChain.__init__`
            # always re-evaluates *both* posteriors at the chain's starting point to seed
            # its first link -- even though that point is exactly where the previous
            # chunk already ended, so nothing new is learned there. Left unguarded, that
            # costs: one real, wasted HF call every chunk boundary; a spurious duplicate
            # training point taught to the surrogate (`lf_model.update` on a `theta`
            # already in its training set); and, whenever an adaptive policy is attached,
            # one non-informative repeat fed into it (`AdaptiveSubchain.on_fine_call`)
            # that can skew its subchain-length decisions -- exactly the phase where that
            # policy is trying to build an accurate picture of LF-HF discrepancy.
            #
            # NOTE: that last point cuts both ways in practice. Re-scoring the surrogate
            # at a point it was *just* trained on tends to look artificially good (a GP
            # fits its own training points almost exactly), so the "spurious" repeat was
            # quietly making `AdaptiveSubchain.has_converged()`'s streak easier to build
            # up -- skipping it makes the adapt phase more honest but can also make it
            # need a larger `max_adapt_coarse_evals` to reach the same declared
            # convergence than before this guard existed. Budget accordingly.
            #
            # Reusing the cached HF output from that identical, immediately-preceding
            # call avoids all three costs above. This only catches the common case where
            # the chain's last step was accepted (so the truly most recent `fine()` call
            # *is* this theta) -- a rejected last step means the most recent call was for
            # a different, since-discarded proposal, which this cache simply won't match;
            # that's a missed optimization, never an incorrect one, since a cache hit
            # only ever returns output `fine` itself already produced for this exact
            # `theta`.
            assert self._last_fine_output is not None  # set together with _last_fine_theta below
            y_hf = self._last_fine_output
        else:
            y_hf = _as_1d_float(self.hf_model(th), name="y_hf")

            if self.adaptive is not None:
                lf_mean, _ = self.lf_model.predict(th)
                lf_mean = _as_1d_float(lf_mean, name="lf_mean_before_update")
                self.adaptive.on_fine_call(y_hf=y_hf, y_lf=lf_mean)

            if not self.frozen:
                self.lf_model.update(th, y_hf)
                # `coarse`'s cache holds a surrogate *prediction*, which this update can
                # change at any theta (including whatever's currently cached there) --
                # unlike `fine`'s own cache above, which holds the real, deterministic HF
                # output and stays valid no matter what the surrogate does. Invalidate it
                # rather than risk `coarse` handing out a prediction from before the
                # surrogate just learned this exact point.
                self._last_coarse_theta = None
                self._last_coarse_output = None

            self._last_fine_theta, self._last_fine_output = th.copy(), y_hf

        if replace_last:
            self.log.replace_last(True)
        else:
            self.log.append(True)

        return y_hf
