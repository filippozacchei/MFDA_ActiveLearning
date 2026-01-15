from __future__ import annotations
import numpy as np


def rwm_proposal(rng: np.random.Generator,
                theta: np.ndarray,
                cov: np.ndarray,
                step_scale: float) -> np.ndarray:
    z = rng.multivariate_normal(mean=np.zeros(theta.size), cov=cov)
    return theta + step_scale * z
