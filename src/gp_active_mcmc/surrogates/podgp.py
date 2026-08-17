from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .gp import MultiOutputGP
from .pod import POD, pod_energy

FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class PODGPSurrogate:
    """POD-GP surrogate for trajectory- or field-valued model outputs.

    This surrogate implements a common two-step construction for high-dimensional outputs:

    1. **Compression (POD)**: map each snapshot/trajectory ``y`` to a low-dimensional
       coefficient vector ``a ∈ R^r``.
    2. **Regression (GP)**: learn the mapping ``θ → a`` using a Gaussian process.

    The surrogate then provides predictions in the original observation space by
    reconstructing:
    """

    r"""
    \[\hat{y(\theta)} = \mu + \hat{a(\theta)}\Phi,\]

    where $\mu$ is the POD mean (stored in `pod.mean_`) and $\Phi$ are the POD modes
    (rows of `pod.components_`). In code, the reconstruction is performed by
    [`POD.inverse_transform`][gp_active_mcmc.pod.POD.inverse_transform].

    Uncertainty propagation
    -----------------------
    The underlying GP returns a predictive *marginal* variance for each POD coefficient.
    This class maps coefficient variances to an **output-space pointwise variance**
    under an independence assumption across coefficients:

    \[\mathrm{Var}[y_i] \approx \sum_{j=1}^{r} \Phi_{j,i}^2\, \mathrm{Var}[a_j]\],

    where ``i`` indexes the observation component (e.g., time index) and ``j`` the POD mode.

    This is a pragmatic approximation that is fast and works well for diagnostics
    (e.g., uncertainty bands), but it does not represent correlated output uncertainty.

    Parameters
    ----------
    pod
        Fitted POD object. Must be fitted before calling [`predict`][gp_active_mcmc.podgp.PODGPSurrogate.predict]
        or [`update`][gp_active_mcmc.podgp.PODGPSurrogate.update].
    gp
        Multi-output GP model trained to predict POD coefficients.
        It must implement:

        - ``predict(theta) -> (mean_a, var_a)`` with shapes ``(n, r)``,
        - ``update(theta_new, a_new)`` for online updates.

    coeff_var_floor
        Small non-negative floor applied to coefficient variances returned by the GP.
        This prevents numerical issues when variances become exactly zero.
    y_var_floor
        Small non-negative floor applied to the reconstructed output variance.
    X_history, Y_history
        Every raw `(theta, y_true)` HF pair ever passed to [`update`][gp_active_mcmc.podgp.PODGPSurrogate.update],
        accumulated for [`refit_pod`][gp_active_mcmc.podgp.PODGPSurrogate.refit_pod] (see there for why this
        needs to be stored separately -- `gp`'s own stored targets are POD *coefficients*,
        not the raw snapshots, so they can't be used to refit the basis later). Seed this
        with the offline design's own `(X, Y)` at construction time, or `refit_pod`'s
        basis is only ever built from points collected *after* construction. `None`
        (the default) means "nothing accumulated yet"; `update` initialises it lazily.
    pod_refit_every
        If set, [`update`][gp_active_mcmc.podgp.PODGPSurrogate.update] calls
        [`refit_pod`][gp_active_mcmc.podgp.PODGPSurrogate.refit_pod] automatically every
        `pod_refit_every` accumulated points. `None` (the default) disables this --
        `update` then behaves exactly as before, only ever refining the GP within the
        basis fixed at construction time.
    pod_refit_max
        Caps the total number of times [`refit_pod`][gp_active_mcmc.podgp.PODGPSurrogate.refit_pod]
        actually does its work over this surrogate's *lifetime* -- persistent, not
        reset by anything, and enforced by `refit_pod` itself regardless of what
        triggers the call (`update`'s own `pod_refit_every` cadence, or an external
        caller invoking it directly, e.g. a cross-replicate sync hook). `None` (the
        default) means unbounded. This used to be enforced only at `update`'s call
        site with its own separate counter, which meant an external direct caller of
        `refit_pod` had no cap at all -- self-gating here closes that gap.

        This is also the *only* thing that ever changes the GP's hyperparameters:
        `gp.n_retrain_max` should be left at `0` on every `MultiOutputGP` this
        surrogate is built with, so `update`'s incremental re-optimisation path never
        fires on its own between refits. Every `refit_pod` call (up to `pod_refit_max`
        of them) grants the rebuilt GP a genuine, from-scratch hyperparameter search --
        one trigger, one cap, no separate warm-started or incremental in-between state
        to reason about.
    adaptive_rank
        If `True`, each `refit_pod` call also re-derives the POD rank itself (via
        `pod_energy` on the *full* accumulated `Y_history`, same criterion as
        `select_pod_rank_and_seed_design`'s one-shot offline choice) instead of
        reusing whatever rank the surrogate started with. Why this matters: the
        initial rank is only ever as good as the small offline sample it was picked
        from -- confirmed directly to badly underestimate the true out-of-sample rank
        needed (e.g. 12% of the prior exceeding the noise floor at a rank picked this
        way from 25 points). As real HF data accumulates through active learning, a
        larger sample gives a better-informed rank estimate; `False` (the default)
        keeps today's behaviour of a rank fixed at construction time. Cheap to add:
        `refit_pod` already rebuilds the GP from scratch every time regardless (a
        changed basis makes the old GP's targets stale either way), so letting the
        *rank* also change during that same rebuild doesn't add a new mechanism.
    rank_energy_threshold
        Cumulative-energy threshold used when `adaptive_rank` is set (default `0.999`,
        matching `select_pod_rank_and_seed_design`'s own default).
    rank_max
        Upper bound on how large `adaptive_rank` may grow the rank (default `None`,
        meaning `min(20, n_history - 1, n_obs)` -- mirrors
        `select_pod_rank_and_seed_design`'s own `r_max_cap=20` default).

    Notes
    -----
    This class is intended to satisfy the library's surrogate protocol
    ([`ActiveSurrogate`][gp_active_mcmc.protocols.ActiveSurrogate]):

    - [`predict`][gp_active_mcmc.podgp.PODGPSurrogate.predict] returns `(mean, var)` in observation space.
    - [`update`][gp_active_mcmc.podgp.PODGPSurrogate.update] ingests a new HF snapshot and updates the GP
      in coefficient space.

    See Also
    --------
    [`POD`][gp_active_mcmc.pod.POD]
        POD compression model.
    [`MultiOutputGP`][gp_active_mcmc.gp.MultiOutputGP]
        GP model used for coefficient regression.
    """

    pod: POD
    gp: MultiOutputGP
    coeff_var_floor: float = 1e-12
    y_var_floor: float = 1e-14
    X_history: FloatArray | None = None
    Y_history: FloatArray | None = None
    pod_refit_every: int | None = None
    pod_refit_max: int | None = None
    adaptive_rank: bool = False
    rank_energy_threshold: float = 0.999
    rank_max: int | None = None
    _since_pod_refit: int = field(default=0, repr=False)
    _pod_refit_count: int = field(default=0, repr=False)

    def _predict_coeffs(self, theta: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Predict POD coefficients (mean and variance) from parameters.

        Parameters
        ----------
        theta
            Parameter vector of shape ``(d,)`` or batch of shape ``(n, d)``.

        Returns
        -------
        mean_a
            Predictive mean of POD coefficients, shape ``(n, r)``.
        var_a
            Predictive marginal variance of POD coefficients, shape ``(n, r)``.
            A small floor is applied for numerical robustness.
        """
        theta2 = np.atleast_2d(np.asarray(theta, dtype=float))  # (n, d)
        mean_a, var_a = self.gp.predict(theta2)  # (n, r), (n, r)

        var_a = np.maximum(var_a, float(self.coeff_var_floor))
        return mean_a, var_a

    def _reconstruct_var(self, var_a: FloatArray) -> FloatArray:
        """Map coefficient variance to pointwise output variance.

        Parameters
        ----------
        var_a
            Coefficient variances with shape ``(r,)`` or ``(n, r)``.

        Returns
        -------
        y_var
            Output pointwise variance with shape ``(n_obs,)`` or ``(n, n_obs)``.

        Raises
        ------
        RuntimeError
            If the POD object is not fitted.
        ValueError
            If `var_a` is not 1D or 2D.
        """
        if not self.pod.is_fitted:
            raise RuntimeError("POD is not fitted. Fit POD before using PODGPSurrogate.")
        if self.pod.components_ is None:
            raise RuntimeError("POD is not fitted (components_ missing).")

        # pod.components_ has shape (r, n_obs); transpose to (n_obs, r).
        Phi = self.pod.components_.T  # (n_obs, r)
        var = np.asarray(var_a, dtype=float)

        if var.ndim == 1:
            # (n_obs, r) @ (r,) -> (n_obs,)
            y_var = (Phi**2) @ var
            res1: FloatArray = np.maximum(y_var, float(self.y_var_floor))
            return res1

        if var.ndim == 2:
            # (n, r) @ (r, n_obs) -> (n, n_obs)
            y_var = var @ (Phi**2).T
            res2: FloatArray = np.maximum(y_var, float(self.y_var_floor))
            return res2

        raise ValueError(f"var_a must be 1D or 2D. Got shape {var.shape}.")

    def predict(self, theta: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Predict mean output and pointwise variance in observation space.

        Parameters
        ----------
        theta
            Parameters of shape ``(d,)`` or batch of shape ``(n, d)``.

        Returns
        -------
        y_mean
            Predictive mean in observation space.

            - shape ``(n_obs,)`` if `theta` is 1D
            - shape ``(n, n_obs)`` if `theta` is 2D
        y_var
            Predictive pointwise variance in observation space, same shape as `y_mean`.

        Notes
        -----
        The returned variance is obtained by propagating coefficient-wise variances
        through the POD reconstruction under an independence approximation.
        """
        theta_arr = np.asarray(theta, dtype=float)
        is_single = theta_arr.ndim == 1

        mean_a, var_a = self._predict_coeffs(theta_arr)  # (n, r), (n, r)
        y_mean = self.pod.inverse_transform(mean_a)  # (n, n_obs)
        y_var = self._reconstruct_var(var_a)  # (n, n_obs)

        if is_single:
            return y_mean[0], y_var[0]
        return y_mean, y_var

    def update(self, theta: ArrayLike, y_true: ArrayLike) -> None:
        """Update the surrogate with one new high-fidelity observation.

        This method projects the new HF snapshot into POD coefficient space and updates
        the GP with a single new training point.

        Parameters
        ----------
        theta
            Parameter vector of shape ``(d,)`` (or ``(1, d)``).
        y_true
            HF snapshot/trajectory in observation space, shape ``(n_obs,)`` (or ``(1, n_obs)``).

        Raises
        ------
        ValueError
            If a batch (more than one observation) is provided.
        RuntimeError
            If the POD object is not fitted.
        """
        if not self.pod.is_fitted:
            raise RuntimeError("POD is not fitted. Fit POD before calling update().")

        theta2 = np.atleast_2d(np.asarray(theta, dtype=float))  # (1, d) expected
        y2 = np.atleast_2d(np.asarray(y_true, dtype=float))  # (1, n_obs)

        if theta2.shape[0] != 1 or y2.shape[0] != 1:
            raise ValueError("update expects a single theta and a single snapshot y_true.")

        a_true = self.pod.transform(y2)[0]  # (r,)
        self.gp.update(theta2, a_true)

        self.X_history = theta2.copy() if self.X_history is None else np.vstack([self.X_history, theta2])
        self.Y_history = y2.copy() if self.Y_history is None else np.vstack([self.Y_history, y2])

        if self.pod_refit_every is not None:
            self._since_pod_refit += 1
            if self._since_pod_refit >= self.pod_refit_every:
                self.refit_pod()  # no-ops on its own once pod_refit_max is spent
                self._since_pod_refit = 0

    def refit_pod(self, *, rank: int | None = None) -> None:
        """Refit the POD compression basis from the *full* accumulated HF history
        (`X_history`/`Y_history`: the seed design plus every point learned since), then
        re-project that same history through the new basis and retrain the GP from
        scratch on the reprojected coefficients.

        Why this exists
        ----------------
        `update` alone only ever refines the GP's mapping *into a fixed coefficient
        space*: `self.pod.transform(y2)` projects each new HF snapshot onto the basis
        fixed at construction time, and whatever output-space energy that snapshot has
        *outside* that basis is discarded before the GP ever sees it -- no amount of
        additional HF data or GP retraining can recover it. That's an irreducible
        representation-error floor that `update` cannot touch. `active_learning_offline_design`
        already refits both `pod` and `gp` from scratch once per batch for exactly this
        reason; this method gives the same capability to the online, per-HF-call path
        (`ActiveMCMCModel.coarse`/`.fine` -> `lf_model.update`), so it can also let its
        actively-selected HF points reshape the basis, not just the GP fit within it.

        Self-gated on `pod_refit_max` (see its field docstring) -- safe to call
        directly (e.g. from an external cross-replicate sync hook) as well as via
        `update`'s own cadence; both draw from the same persistent counter, so the cap
        means the same thing regardless of what's triggering the call. A no-op once
        `pod_refit_max` is spent (keeps the current `pod`/`gp` as-is). Every call that
        isn't a no-op grants the rebuilt GP a genuine, from-scratch hyperparameter
        optimisation -- this is the only place this surrogate's GP hyperparameters
        ever change; `gp.n_retrain_max` should be `0` so `update`'s own incremental
        path never fires between refits.

        Parameters
        ----------
        rank
            POD rank to refit at. Defaults to `self.pod.rank` (same capacity, re-fit
            from more/different data) unless `self.adaptive_rank` is set, in which
            case it defaults instead to a freshly-derived rank (see `adaptive_rank`'s
            field docstring). Pass an explicit value to override either default.

        Raises
        ------
        RuntimeError
            If `update` has never been called (nothing accumulated to refit from).
        """
        if self.X_history is None or self.Y_history is None:
            raise RuntimeError("No accumulated history to refit from -- call update() at least once first.")
        if self.pod_refit_max is not None and self._pod_refit_count >= self.pod_refit_max:
            return
        self._pod_refit_count += 1

        if rank is not None:
            r = int(rank)
        elif self.adaptive_rank:
            r_max = self.rank_max
            if r_max is None:
                r_max = 20
            r_max = max(1, min(r_max, self.X_history.shape[0] - 1, self.Y_history.shape[1]))
            energy_curve = pod_energy(
                self.Y_history, r_max=r_max, randomized=self.pod.randomized,
                n_oversamples=self.pod.n_oversamples, n_iter=self.pod.n_iter, random_state=self.pod.random_state,
            )
            r = int(np.searchsorted(energy_curve, self.rank_energy_threshold) + 1)
            r = min(r, r_max)
        else:
            r = int(self.pod.rank)
        new_pod = POD(
            rank=r,
            randomized=self.pod.randomized,
            n_oversamples=self.pod.n_oversamples,
            n_iter=self.pod.n_iter,
            random_state=self.pod.random_state,
        ).fit(self.Y_history)
        A = new_pod.transform(self.Y_history)

        new_gp = MultiOutputGP(
            X_train=self.X_history,
            Y_train=A,
            kernel=self.gp.kernel,
            ard=self.gp.ard,
            noise_variance=self.gp.noise_variance,
            update_every=self.gp.update_every,
            n_retrain_max=self.gp.n_retrain_max,
            max_iters=self.gp.max_iters,
        )
        self.pod = new_pod
        self.gp = new_gp

    def log_likelihood(self) -> float:
        """Return the summed marginal log-likelihood of the underlying GP(s)."""
        return self.gp.log_likelihood()

    def copy(self) -> PODGPSurrogate:
        """Return a deep copy of the surrogate.

        This is useful when a workflow needs independent surrogate state, e.g. when
        running truly independent chains or experiments.

        Notes
        -----
        Many active-learning workflows instead prefer *shared* state across deepcopies
        (see [`AdaptiveMetropolisShared`][gp_active_mcmc.inference.proposal.AdaptiveMetropolisShared]).
        """
        return PODGPSurrogate(
            pod=copy.deepcopy(self.pod),
            gp=copy.deepcopy(self.gp),
            coeff_var_floor=self.coeff_var_floor,
            y_var_floor=self.y_var_floor,
            X_history=None if self.X_history is None else self.X_history.copy(),
            Y_history=None if self.Y_history is None else self.Y_history.copy(),
            pod_refit_every=self.pod_refit_every,
            pod_refit_max=self.pod_refit_max,
            adaptive_rank=self.adaptive_rank,
            rank_energy_threshold=self.rank_energy_threshold,
            rank_max=self.rank_max,
        )
