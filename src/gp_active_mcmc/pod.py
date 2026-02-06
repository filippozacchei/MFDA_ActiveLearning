from __future__ import annotations
import numpy as np

try:
    from sklearn.utils.extmath import randomized_svd

    _HAS_RAND_SVD = True
except ImportError:
    _HAS_RAND_SVD = False


class POD:
    """
    Proper Orthogonal Decomposition over the time dimension.

    Fit on snapshots Y: (N, T) => basis Phi: (T, r), mean: (T,)
    """

    def __init__(
        self,
        r: int,
        randomized: bool = True,
        n_oversamples: int = 10,
        n_iter: int = 2,
        random_state: int | None = 0,
    ):
        self.r = int(r)
        self.randomized = bool(randomized)
        self.n_oversamples = int(n_oversamples)
        self.n_iter = int(n_iter)
        self.random_state = random_state

        self.mean_: np.ndarray | None = None
        self.phi_: np.ndarray | None = None
        self.singular_values_: np.ndarray | None = None

    def fit(self, Y: np.ndarray) -> POD:
        assert Y.ndim == 2, "Y must be 2D (N, T)."

        self.mean_ = Y.mean(axis=0)
        Yc = Y - self.mean_

        if self.randomized and _HAS_RAND_SVD:
            # randomized SVD
            U, S, Vt = randomized_svd(
                Yc,
                n_components=self.r,
                n_oversamples=self.n_oversamples,
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
        else:
            # exact SVD
            U, S, Vt = np.linalg.svd(Yc, full_matrices=False)
            S = S[: self.r]
            Vt = Vt[: self.r]

        self.phi_ = Vt.T  # (T, r)
        self.singular_values_ = S.copy()
        return self

    def project(self, Y: np.ndarray) -> np.ndarray:
        assert self.phi_ is not None and self.mean_ is not None, "Call fit() first."
        assert Y.ndim == 2, "Y must be 2D."
        return (Y - self.mean_) @ self.phi_

    def reconstruct(self, A: np.ndarray) -> np.ndarray:
        assert self.phi_ is not None and self.mean_ is not None, "Call fit() first."
        assert (
            A.ndim == 2 and A.shape[1] == self.phi_.shape[1]
        ), f"A must be 2D with second dimension {self.phi_.shape[1]}"
        return self.mean_ + A @ self.phi_.T

    def energy(self, Y: np.ndarray, r_max: int | None = None) -> np.ndarray:
        assert Y.ndim == 2, "Y must be 2D (N, T)."

        Yc = Y - Y.mean(axis=0)
        m = min(Yc.shape)
        if r_max is None:
            r_max = m
        r_max = int(min(r_max, m))

        if self.randomized and _HAS_RAND_SVD:
            _, S, _ = randomized_svd(
                Yc,
                n_components=r_max,
                n_oversamples=self.n_oversamples,
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
        else:
            _, S, _ = np.linalg.svd(Yc, full_matrices=False)
            S = S[:r_max]

        total_energy = np.sum(S**2)
        if total_energy == 0:
            return np.zeros_like(S)
        return np.cumsum(S**2) / total_energy
