from __future__ import annotations
from dataclasses import dataclass
import numpy as np


class Prior:
    def logpdf(self, theta: np.ndarray) -> float:
        raise NotImplementedError
    
    def sample(self, rng: np.random.Generator)  -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class BoxUniformPrior(Prior):
    low: np.ndarray
    high: np.ndarray

    def logpdf(self, theta: np.ndarray) -> float:
        if np.all(theta >= self.low) and np.all(theta <= self.high):
            return 0.0  
        return -np.inf


@dataclass(frozen=True)
class GaussianPrior(Prior):
    mean: np.ndarray
    cov: np.ndarray

    def logpdf(self, theta: np.ndarray) -> float:
        x = theta - self.mean
        d = x.size
        L = np.linalg.cholesky(self.cov + 1e-12 * np.eye(d))
        z = np.linalg.solve(L, x)
        quad = float(z @ z)
        logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
        return float(-0.5 * (quad + logdet + d * np.log(2.0 * np.pi)))

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        theta = rng.multivariate_normal(self.mean, self.cov)
        return theta

@dataclass(frozen=True)
class SumPrior(Prior):
    priors: tuple[Prior, ...]

    def logpdf(self, theta: np.ndarray) -> float:
        lp = 0.0
        for p in self.priors:
            v = p.logpdf(theta)
            if np.isneginf(v):
                return -np.inf
            lp += v
        return float(lp)
