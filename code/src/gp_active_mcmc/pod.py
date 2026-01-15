from __future__ import annotations
import numpy as np

class POD:
    """
    POD over the time dimension.
    Fit on snapshots Y: (N, T) => basis Phi: (T, r), mean: (T,)
    """
    def __init__(self, r: int):
        self.r = int(r)
        self.mean_: np.ndarray | None = None
        self.phi_: np.ndarray | None = None

    def fit(self, Y: np.ndarray) -> "POD":
        if Y.ndim != 2:
            raise ValueError("Y must be (N, T).")
        self.mean_ = Y.mean(axis=0)
        Yc = Y - self.mean_
        # economy SVD
        _, _, Vt = np.linalg.svd(Yc, full_matrices=False)
        self.phi_ = Vt[: self.r].T  # (T, r)
        return self

    def project(self, Y: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.phi_ is None:
            raise RuntimeError("Call fit() first.")
        Yc = Y - self.mean_
        return Yc @ self.phi_  # (N, r)

    def reconstruct(self, A: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.phi_ is None:
            raise RuntimeError("Call fit() first.")
        return self.mean_ + A @ self.phi_.T  # (..., T)

    def energy(self, Y: np.ndarray) -> np.ndarray:
        """
        Cumulative energy curve from SVD singular values (for rank selection).
        """
        Yc = Y - Y.mean(axis=0)
        _, S, _ = np.linalg.svd(Yc, full_matrices=False)
        e = (S**2) / np.sum(S**2)
        return np.cumsum(e)
