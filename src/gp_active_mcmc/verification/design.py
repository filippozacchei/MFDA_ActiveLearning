"""Offline/seed surrogate construction shared by every surrogate-based method."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gp_active_mcmc.surrogates import POD, MultiOutputGP, PODGPSurrogate, pod_energy
from gp_active_mcmc.surrogates.gp import KernelName
from gp_active_mcmc.verification.problem import Problem

FloatArray = NDArray[np.float64]

__all__ = [
    "DEFAULT_ONLINE_LEARNING",
    "OnlineLearningConfig",
    "PODRankSelection",
    "active_learning_offline_design",
    "build_initial_surrogate",
    "select_pod_rank_and_seed_design",
]


@dataclass(frozen=True)
class OnlineLearningConfig:
    """How an online-learning surrogate's POD basis and GP hyperparameters get
    refreshed as new HF points arrive, and how the adaptively-re-derived POD rank
    (`PODGPSurrogate.refit_pod`/`.adaptive_rank` -- always on, the methodological
    default) is tuned. Shared by every `adaptive_surrogate_mcmc`/`adaptive_stm` call
    site -- this just bundles those knobs into one value instead of four.
    """

    pod_refit_every: int | None = None  # refit every N accumulated HF points; None disables refitting
    pod_refit_max: int | None = None  # cap on total refit_pod() calls per surrogate lifetime; None = unbounded
    rank_energy_threshold: float = 0.999  # cumulative-energy threshold refit_pod() uses to pick the rank
    rank_max: int | None = None  # upper bound on the adaptively-derived rank; None uses refit_pod()'s fallback


# The "no refitting" default, as a module-level singleton (not a fresh call in every
# signature) so ruff's B008 doesn't flag `OnlineLearningConfig()` as a mutable-looking
# default -- it's frozen, so sharing one instance across every caller is safe.
DEFAULT_ONLINE_LEARNING = OnlineLearningConfig()


def _adaptive_pod_rank(Y: FloatArray, *, n_current: int, energy_threshold: float, r_max_cap: int | None) -> int:
    """Smallest POD rank whose cumulative explained energy over `Y` reaches
    `energy_threshold` -- the same criterion `PODGPSurrogate.refit_pod`'s adaptive-rank
    branch uses online, applied here to a plain HF sample. `r_max_cap` (`None` falls
    back to 20, matching `refit_pod`'s own fallback) and `n_current` (the sample size)
    both additionally cap the returned rank.
    """
    r_max = 20 if r_max_cap is None else r_max_cap
    r_max = max(1, min(r_max, n_current - 1, Y.shape[1]))
    energy_curve = pod_energy(Y, r_max=r_max)
    r = int(np.searchsorted(energy_curve, energy_threshold) + 1)
    return min(r, r_max)


def build_initial_surrogate(
    problem: Problem,
    rng: np.random.Generator,
    *,
    n_init: int,
    kernel: KernelName,
    rank_energy_threshold: float = 0.999,
    rank_max: int | None = None,
) -> tuple[PODGPSurrogate, FloatArray, FloatArray]:
    """Draws an `n_init`-point offline seed design and fits the shared surrogate every
    surrogate-based method starts learning from, at a POD rank adaptively derived from
    that seed sample (`_adaptive_pod_rank` -- there's no fixed-rank option: every
    surrogate this package builds re-derives its rank, first here and then at every
    `PODGPSurrogate.refit_pod()` call). The GP's `n_retrain_max` is fixed at 0:
    hyperparameters only ever change via a later `refit_pod()` call (see
    `_surrogate_for_online_learning`). Returns the fitted surrogate and the seed
    design's inputs/outputs (shapes ``(n_init, d)`` and ``(n_init, n_obs)``)."""
    X = np.asarray([problem.prior.rvs(random_state=rng) for _ in range(n_init)], dtype=float)
    Y = np.asarray([problem.hf_forward(theta) for theta in X], dtype=float)

    pod_rank = _adaptive_pod_rank(Y, n_current=n_init, energy_threshold=rank_energy_threshold, r_max_cap=rank_max)
    pod = POD(rank=pod_rank).fit(Y)
    A = pod.transform(Y)
    gp = MultiOutputGP(X_train=X, Y_train=A, kernel=kernel, ard=True, noise_variance=1e-6, n_retrain_max=0)
    # Seed X_history/Y_history with the design itself so a later refit_pod() draws
    # from the full history, not just points added after this surrogate is copied.
    surrogate = PODGPSurrogate(pod=pod, gp=gp, X_history=X.copy(), Y_history=Y.copy())
    return surrogate, X, Y


@dataclass
class PODRankSelection:
    """Result of `select_pod_rank_and_seed_design`."""

    pod_rank: int  # smallest rank whose cumulative explained energy reaches the requested threshold
    n_init: int  # sample size used, equal to X.shape[0]
    X: FloatArray  # the HF sample, reusable as-is as build_initial_surrogate/run_pretrained's seed_X/seed_Y
    Y: FloatArray
    energy_curve: FloatArray  # cumulative explained energy; index r - 1 corresponds to rank r


def select_pod_rank_and_seed_design(
    problem: Problem,
    rng: np.random.Generator,
    *,
    n_init: int = 60,
    energy_threshold: float = 0.999,
    r_max_cap: int = 20,
) -> PODRankSelection:
    """Picks a POD rank from a single, plain `n_init`-size HF sample -- a one-shot
    model-selection step, not part of any method's active-learning budget. Rank is the
    smallest one whose cumulative explained energy (`gp_active_mcmc.surrogates.pod_energy`)
    reaches `energy_threshold`, additionally capped by `r_max_cap`. The sample doubles
    as the seed design (`X`/`Y`) passed to every method. Raises `ValueError` if
    `energy_threshold` isn't in ``(0, 1)``."""
    if not 0 < energy_threshold < 1:
        raise ValueError("energy_threshold must be in (0, 1).")

    X = np.asarray([problem.prior.rvs(random_state=rng) for _ in range(n_init)], dtype=float)
    Y = np.asarray([problem.hf_forward(theta) for theta in X], dtype=float)

    r_max = max(1, min(r_max_cap, n_init - 1, Y.shape[1]))
    energy_curve = pod_energy(Y, r_max=r_max)
    pod_rank = int(np.searchsorted(energy_curve, energy_threshold) + 1)
    pod_rank = min(pod_rank, r_max)

    return PODRankSelection(pod_rank=pod_rank, n_init=n_init, X=X, Y=Y, energy_curve=energy_curve)


def _surrogate_for_online_learning(surrogate: PODGPSurrogate, config: OnlineLearningConfig) -> PODGPSurrogate:
    """Deep-copies `surrogate` for a replicate that will call `.update()` during MCMC,
    applying `config`'s refit/rank settings to the copy's matching fields."""
    lf = copy.deepcopy(surrogate)
    lf.pod_refit_every = config.pod_refit_every
    lf.pod_refit_max = config.pod_refit_max
    lf.rank_energy_threshold = config.rank_energy_threshold
    lf.rank_max = config.rank_max
    return lf


def active_learning_offline_design(
    problem: Problem,
    seed_X: FloatArray,
    seed_Y: FloatArray,
    *,
    gamma_threshold: float,
    kernel: KernelName,
    rng: np.random.Generator,
    candidate_pool_size: int = 500,
    batch_size: int = 25,
    max_total_budget: int = 600,
    n_validation: int = 100,
    rank_energy_threshold: float = 0.999,
    rank_max: int | None = None,
) -> PODGPSurrogate:
    """Greedy max-predictive-variance acquisition, purely offline (no MCMC path).
    """
    X_all = [seed_X.copy()]
    Y_all = [seed_Y.copy()]

    val_grid = np.asarray(problem.prior.rvs(size=n_validation, random_state=rng), dtype=float)

    pod_rank = _adaptive_pod_rank(
        seed_Y, n_current=seed_X.shape[0], energy_threshold=rank_energy_threshold, r_max_cap=rank_max,
    )
    pod = POD(rank=pod_rank).fit(seed_Y)
    A = pod.transform(seed_Y)
    gp = MultiOutputGP(X_train=seed_X, Y_train=A, kernel=kernel, ard=True, noise_variance=1e-6)
    surrogate = PODGPSurrogate(pod=pod, gp=gp)

    n_current = seed_X.shape[0]
    while n_current < max_total_budget:
        _val_mean, val_var = surrogate.predict(val_grid)
        if float(np.max(val_var.mean(axis=1))) <= gamma_threshold**2:
            break

        b = min(batch_size, max_total_budget - n_current)
        batch_X = []
        batch_Y = []
        for _ in range(b):
            pool = np.asarray(problem.prior.rvs(size=candidate_pool_size, random_state=rng), dtype=float)
            _mean, var = surrogate.predict(pool)
            score = var.mean(axis=1)
            theta_new = pool[int(np.argmax(score))]
            batch_X.append(theta_new)
            batch_Y.append(problem.hf_forward(theta_new))

        X_all.append(np.asarray(batch_X))
        Y_all.append(np.asarray(batch_Y))
        n_current += b

        X_final = np.vstack(X_all)
        Y_final = np.vstack(Y_all)
        pod_rank = _adaptive_pod_rank(
            Y_final, n_current=n_current, energy_threshold=rank_energy_threshold, r_max_cap=rank_max,
        )
        pod = POD(rank=pod_rank).fit(Y_final)
        A_final = pod.transform(Y_final)
        gp = MultiOutputGP(X_train=X_final, Y_train=A_final, kernel=kernel, ard=True, noise_variance=1e-6)
        surrogate = PODGPSurrogate(pod=pod, gp=gp)

    return surrogate
