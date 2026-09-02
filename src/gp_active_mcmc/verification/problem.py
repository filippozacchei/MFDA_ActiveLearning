"""The `Problem` protocol shared by every verification harness function."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from gp_active_mcmc.protocols import HighFidelityModel

FloatArray = NDArray[np.float64]

__all__ = ["Problem"]


@runtime_checkable
class Problem(Protocol):
    """Structural interface for an inverse problem usable with this package's
    comparison harness. Any object exposing these six attributes qualifies, checked
    structurally (no inheritance needed); a concrete problem may carry extra fields of
    its own (a time grid, a mesh, ...), simply invisible to the harness.
    """

    prior: Any  # duck-types a scipy.stats frozen distribution: .rvs, .pdf/.logpdf, .mean, .cov
    theta_true: FloatArray  # ground-truth parameter vector, shape (d,)
    y_obs: FloatArray  # observed data the likelihood conditions on, shape (n_obs,)
    sigma_obs: float  # observation-noise standard deviation (i.i.d. Gaussian noise)
    hf_forward: HighFidelityModel  # high-fidelity forward model, theta -> y
    param_names: tuple[str, ...]  # one name per dim of theta_true, e.g. ("k", "c")
    scale: float # proposal scale
