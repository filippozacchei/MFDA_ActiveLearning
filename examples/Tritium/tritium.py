"""Tritium diffusion benchmark (Achlys via UM-Bridge).

Wraps the UM-Bridge Docker benchmark into the same interface used by the toy and
beam examples:

- ``make_forward_model(url, ...)`` → callable  ``theta → y``
- ``make_observation(rng, theta_true, ...)`` → noisy observation
- prior bounds helpers

The Docker container must be running before calling the forward model::

    docker run -it -p 4243:4243 linusseelinger/benchmark-achlys:latest

References
----------
https://um-bridge-benchmarks.readthedocs.io/en/docs/inverse-benchmarks/achlys.html
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import umbridge


# =====================================================================
#  Parameter space
# =====================================================================

# Names and uniform prior bounds from the benchmark description
PARAM_NAMES = ["E1", "E2", "E3", "n1", "n2"]

PRIOR_BOUNDS = np.array([
    [0.70, 1.00],     # E1  (detrapping energy, eV)
    [0.90, 1.30],     # E2
    [1.10, 1.75],     # E3
    [5e-4, 5e-3],     # n1  (intrinsic trap density)
    [1e-4, 1e-3],     # n2
])

N_PARAMS = len(PARAM_NAMES)
N_OUTPUT = 500  # benchmark returns 500 time-points of tritium flux


# =====================================================================
#  UM-Bridge client helpers
# =====================================================================

def _get_model(url: str, model_name: str = "forward") -> umbridge.HTTPModel:
    """Connect to the UM-Bridge server and return the requested model."""
    return umbridge.HTTPModel(url, model_name)


def make_forward_model(
    url: str = "http://localhost:4243",
    model_name: str = "forward",
    obs_idx: np.ndarray | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a callable ``theta → y`` backed by the UM-Bridge Achlys server.

    Parameters
    ----------
    url
        UM-Bridge server URL.
    model_name
        Model name exposed by the server (``"forward"`` or ``"posterior"``).
    obs_idx
        Optional index array to sub-select output components.  If ``None``,
        return the full 500-point flux output.
    """
    model = _get_model(url, model_name)

    def _forward(theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float).ravel().tolist()
        result = model([theta])            # UM-Bridge expects list-of-lists
        y = np.asarray(result[0], dtype=float)
        if obs_idx is not None:
            y = y[obs_idx]
        return y

    return _forward


def make_time_grid(n_pts: int = 500) -> np.ndarray:
    """Return the time grid matching the benchmark output.

    The Achlys benchmark paper uses a time range of ~25 000 s with 500 equally
    spaced output points.  Adjust ``t_end`` if the server uses a different range.
    """
    return np.linspace(0.0, 25_000.0, n_pts)


def make_observation(
    rng: np.random.Generator,
    theta_true: np.ndarray,
    forward: Callable[[np.ndarray], np.ndarray],
    sigma_obs: float,
    obs_idx: np.ndarray | None = None,
) -> np.ndarray:
    """Generate a synthetic noisy observation.

    Parameters
    ----------
    rng
        Numpy random generator.
    theta_true
        True parameter vector.
    forward
        Forward model callable (already configured with ``obs_idx`` if needed).
    sigma_obs
        Observation noise standard deviation.
    obs_idx
        If the forward model returns the full state, sub-select with this index.
    """
    y_clean = forward(theta_true)
    if obs_idx is not None:
        y_clean = y_clean[obs_idx]
    return y_clean + rng.normal(0.0, sigma_obs, size=y_clean.shape)


def sample_prior(rng: np.random.Generator, n: int = 1) -> np.ndarray:
    """Draw ``n`` samples from the uniform prior.

    Returns shape ``(n, 5)`` or ``(5,)`` if ``n == 1``.
    """
    lo = PRIOR_BOUNDS[:, 0]
    hi = PRIOR_BOUNDS[:, 1]
    samples = rng.uniform(lo, hi, size=(n, N_PARAMS))
    return samples[0] if n == 1 else samples
