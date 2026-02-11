# gp_active_mcmc/protocols.py
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@runtime_checkable
class ActiveSurrogate(Protocol):
    """Surrogate model interface used by active learning / Active-MCMC components.

    Implementations typically provide a predictive mean and an uncertainty
    quantification (e.g., predictive variance or covariance).
    """

    def predict(self, theta: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Predict model output at parameter(s) ``theta``.

        Parameters
        ----------
        theta
            Parameter vector or batch of parameters. Common shapes:
            - (d,) for one parameter vector
            - (n, d) for a batch of n vectors

        Returns
        -------
        mean
            Predictive mean. Common shapes:
            - (m,) or (T, m) for one theta
            - (n, m) or (n, T, m) for batched theta
        uncertainty
            Uncertainty summary aligned with ``mean``. This can be predictive
            variance (same shape as mean) or a covariance representation, but
            the interpretation must be documented by the implementation.
        """

    def update(self, theta: ArrayLike, y: ArrayLike) -> None:
        """Update the surrogate using a new high-fidelity observation.

        Parameters
        ----------
        theta
            Parameter vector (shape (d,)) or batch (shape (n, d)).
        y
            High-fidelity observation(s) corresponding to theta. Shape must be
            consistent with the surrogate's output representation.
        """


@runtime_checkable
class HighFidelityModel(Protocol):
    """Forward model interface for high-fidelity evaluations."""

    def __call__(self, theta: ArrayLike) -> FloatArray:
        """Evaluate the forward model at parameter(s) ``theta``."""
