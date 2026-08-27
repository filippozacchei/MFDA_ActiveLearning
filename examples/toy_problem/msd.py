from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import multivariate_normal


class IndependentUniformPrior:
    """Independent uniform prior over a bounded box.

    Matches the subset of the scipy.stats frozen-distribution interface used
    elsewhere in this codebase (`rvs`, `pdf`, `logpdf`, plus `mean`/`cov` for scaling
    the MCMC proposal covariance), so it is a drop-in replacement for
    `scipy.stats.multivariate_normal` wherever `prior` is used.
    """

    def __init__(self, low: ArrayLike, high: ArrayLike):
        self.low = np.asarray(low, dtype=float)
        self.high = np.asarray(high, dtype=float)
        if self.low.shape != self.high.shape or self.low.ndim != 1:
            raise ValueError("low and high must be 1D arrays of the same shape.")
        if np.any(self.high <= self.low):
            raise ValueError("high must be strictly greater than low in every dimension.")
        self.dim = int(self.low.shape[0])
        self._log_density = float(-np.sum(np.log(self.high - self.low)))

    @property
    def mean(self) -> np.ndarray:
        return 0.5 * (self.low + self.high)

    @property
    def cov(self) -> np.ndarray:
        var = (self.high - self.low) ** 2 / 12.0
        return np.diag(var)

    def rvs(self, size: int | None = None, random_state: np.random.Generator | None = None) -> np.ndarray:
        rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
        if size is None:
            return rng.uniform(self.low, self.high)
        return rng.uniform(self.low, self.high, size=(size, self.dim))

    def logpdf(self, x: ArrayLike) -> float:
        x = np.asarray(x, dtype=float)
        if np.any(x < self.low) or np.any(x > self.high):
            return float("-inf")
        return self._log_density

    def pdf(self, x: ArrayLike) -> float:
        return float(np.exp(self.logpdf(x)))

# Mass is fixed/known (physically the easy quantity to measure directly, e.g. by
# weighing). Only stiffness k and damping c are inferred: with mass free as well, only
# the combinations wn = sqrt(k/m) and zeta = c/(2*sqrt(mk)) are identifiable from a
# single free-decay trajectory, leaving a near-flat ridge in (m, k, c) space that makes
# the posterior slow to pin down. Fixing m removes that degeneracy.
MASS = 1.0


def msd_forward(theta: np.ndarray, t: np.ndarray, *, x0: float = 1.0, m: float = MASS) -> np.ndarray:
    """
    Free-vibration response of a mass-spring-damper system, released from rest.

    theta = [k, c]
      k : stiffness (N/m), must be > 0
      c : damping coefficient (N s/m), must be >= 0
    m   : mass (kg), fixed/known.

    Governs m*x'' + c*x' + k*x = 0, x(0) = x0, x'(0) = 0.

    Output:
      x(t): 1D array with same shape as t
    """
    # Safety: keep the model defined even if Gaussian sampling yields non-physical draws
    # (mirrors the clamping already used in toy_forward).
    k = max(float(theta[0]), 1e-6)
    c = max(float(theta[1]), 0.0)

    wn = np.sqrt(k / m)
    zeta = c / (2.0 * np.sqrt(m * k))

    if zeta < 1.0 - 1e-9:
        # Underdamped: decaying oscillation.
        wd = wn * np.sqrt(1.0 - zeta**2)
        x = x0 * np.exp(-zeta * wn * t) * (
            np.cos(wd * t) + (zeta / np.sqrt(1.0 - zeta**2)) * np.sin(wd * t)
        )
    elif zeta > 1.0 + 1e-9:
        # Overdamped: two decaying exponentials, no oscillation.
        r1 = wn * (-zeta + np.sqrt(zeta**2 - 1.0))
        r2 = wn * (-zeta - np.sqrt(zeta**2 - 1.0))
        A = x0 * r2 / (r2 - r1)
        B = x0 - A
        x = A * np.exp(r1 * t) + B * np.exp(r2 * t)
    else:
        # Critically damped.
        x = x0 * (1.0 + wn * t) * np.exp(-wn * t)

    return x


def make_forward_model(t: np.ndarray, *, x0: float = 1.0, m: float = MASS) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap the mass-spring-damper forward model with the timeline baked in."""

    def _forward(theta: np.ndarray) -> np.ndarray:
        return msd_forward(theta, t, x0=x0, m=m)

    return _forward


def make_timeline(T: int = 300, t_end: float = 4.0) -> np.ndarray:
    return np.linspace(0.0, t_end, T)


def make_prior() -> Any:
    """Wide, independent uniform prior on (k, c).

    Wide and uniform on purpose: with a small offline design, a narrow (or Gaussian)
    prior either lets a handful of points cover the plausible region densely enough
    that no method ever needs to learn online, or concentrates the "hard" region only
    in the tails. A wide uniform box instead spreads the offline design points evenly
    across the whole domain (i.i.d. uniform draws, unlike Gaussian draws which cluster
    near the mean), so under-coverage is a property of the whole domain and genuine
    online learning matters everywhere, not just far from the mean.

    `c` is centred well above the old [0.2, 1.8] range (which gave Q~4-8, i.e. 4-5
    visible oscillation cycles over the observation window) and instead spans damping
    ratios roughly zeta in [0.2, 1.0] (Q roughly 0.5-2.5): near-equilibrium is reached
    within the observation window across (almost) the whole box. A linear POD basis
    struggles to represent a *family* of phase/frequency-shifted, slowly-decaying
    oscillations with few modes (related to the Kolmogorov n-width issue for
    wave/transport-dominated problems); heavier damping keeps the trajectory family
    close to a smooth, quickly-decaying shape that a handful of POD modes can span
    well, without removing the "wide range of possible shapes" difficulty the wide box
    is there to create.
    """
    # return IndependentUniformPrior(low=[10.0, 0.1], high=[100.0, 2.0])

    # Gaussian alternative, same nominal range as the uniform box above (mean = box
    # midpoint, sigma = 0.25 * box width per dimension, so +-2 sigma roughly spans the
    # original [10, 100] x [0.1, 2.0] box) -- matches the convention already used for
    # examples/navier_stokes's Gaussian prior. Trades away the "even offline-design
    # coverage everywhere" property the docstring above argues for (draws cluster near
    # the mean instead), so treat this as an experiment, not a replacement.
    return multivariate_normal(mean=[50.0, 1.0], cov=np.diag([20**2, 0.25**2]))


def make_observation(
    rng: np.random.Generator,
    theta_true: np.ndarray,
    t: np.ndarray,
    sigma_obs: float,
    *,
    x0: float = 1.0,
    m: float = MASS,
) -> np.ndarray:
    x_clean = msd_forward(theta_true, t, x0=x0, m=m)
    return x_clean + rng.normal(0.0, sigma_obs, size=x_clean.shape)
