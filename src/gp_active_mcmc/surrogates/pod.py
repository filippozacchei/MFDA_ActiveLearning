from __future__ import annotations

"""Proper Orthogonal Decomposition (POD).

Snapshots are represented as a 2D array ``Y`` with shape (n_snapshots, n_time).
The POD basis is computed from the mean-centered snapshot matrix.

This implementation follows scikit-learn conventions:
- fitted attributes end with an underscore
- components_ has shape (rank, n_time) (rows are components)
- transform returns coefficients with shape (n_snapshots, rank)
- inverse_transform reconstructs snapshots with shape (n_snapshots, n_time)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.utils.extmath import randomized_svd


FloatArray = NDArray[np.floating]


def _as_2d_float_array(x: ArrayLike, *, name: str) -> FloatArray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array. Got shape {arr.shape}.")
    return arr


def _require_fitted(is_fitted: bool) -> None:
    if not is_fitted:
        raise RuntimeError("POD is not fitted yet. Call 'fit(Y)' before using this method.")


@dataclass(slots=True)
class POD:
    """POD estimator (scikit-learn style).

    Parameters
    ----------
    rank
        Number of modes to retain (clipped to min(Y.shape)).
    randomized
        Use randomized SVD (recommended for large snapshot matrices).
    n_oversamples, n_iter, random_state
        Randomized SVD parameters (passed to sklearn.utils.extmath.randomized_svd).

    Attributes
    ----------
    mean_
        Mean snapshot over training set, shape (n_time,).
    components_
        POD modes, shape (rank, n_time).
    singular_values_
        Singular values for retained modes, shape (rank,).
    explained_energy_
        Cumulative explained energy for retained modes, shape (rank,).
    """

    rank: int
    randomized: bool = True
    n_oversamples: int = 10
    n_iter: int = 2
    random_state: Optional[int] = 0

    mean_: Optional[FloatArray] = None
    components_: Optional[FloatArray] = None
    singular_values_: Optional[FloatArray] = None
    explained_energy_: Optional[FloatArray] = None

    @property
    def is_fitted(self) -> bool:
        return self.mean_ is not None and self.components_ is not None and self.singular_values_ is not None

    @property
    def n_time_(self) -> int:
        _require_fitted(self.is_fitted)
        assert self.mean_ is not None  # guarded
        return int(self.mean_.shape[0])

    @property
    def rank_(self) -> int:
        _require_fitted(self.is_fitted)
        assert self.components_ is not None  # guarded
        return int(self.components_.shape[0])

    def fit(self, Y: ArrayLike) -> "POD":
        """Fit POD basis from snapshots.

        Parameters
        ----------
        Y
            Snapshot matrix, shape (n_snapshots, n_time).

        Returns
        -------
        self
        """
        Y_arr = _as_2d_float_array(Y, name="Y")

        r = int(self.rank)
        if r <= 0:
            raise ValueError(f"rank must be positive. Got {r}.")
        r = min(r, min(Y_arr.shape))

        mean = Y_arr.mean(axis=0)
        Yc = Y_arr - mean

        if self.randomized:
            _U, S, Vt = randomized_svd(
                Yc,
                n_components=r,
                n_oversamples=self.n_oversamples,
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
        else:
            _U, S_full, Vt_full = np.linalg.svd(Yc, full_matrices=False)
            S = S_full[:r]
            Vt = Vt_full[:r]

        self.mean_ = np.asarray(mean, dtype=float)
        self.singular_values_ = np.asarray(S, dtype=float)
        self.components_ = np.asarray(Vt, dtype=float)  # (r, n_time)

        total = float(np.sum(self.singular_values_**2))
        if total == 0.0:
            self.explained_energy_ = np.zeros_like(self.singular_values_)
        else:
            self.explained_energy_ = np.cumsum(self.singular_values_**2) / total

        return self

    def transform(self, Y: ArrayLike) -> FloatArray:
        """Project snapshots onto POD basis."""
        _require_fitted(self.is_fitted)
        Y_arr = _as_2d_float_array(Y, name="Y")

        assert self.mean_ is not None and self.components_ is not None  # guarded

        if Y_arr.shape[1] != self.mean_.shape[0]:
            raise ValueError(
                f"Y has incompatible time dimension: got {Y_arr.shape[1]}, expected {self.mean_.shape[0]}."
            )

        return (Y_arr - self.mean_) @ self.components_.T  # (n, r)

    def inverse_transform(self, A: ArrayLike) -> FloatArray:
        """Reconstruct snapshots from POD coefficients."""
        _require_fitted(self.is_fitted)
        A_arr = _as_2d_float_array(A, name="A")

        assert self.mean_ is not None and self.components_ is not None  # guarded

        if A_arr.shape[1] != self.rank_:
            raise ValueError(
                f"A has incompatible coefficient dimension: got {A_arr.shape[1]}, expected {self.rank_}."
            )

        return self.mean_ + A_arr @ self.components_  # (n, n_time)

    def fit_transform(self, Y: ArrayLike) -> FloatArray:
        """Fit POD and return coefficients for the same snapshots."""
        return self.fit(Y).transform(Y)


def pod_energy(
    Y: ArrayLike,
    *,
    r_max: int | None = None,
    randomized: bool = True,
    n_oversamples: int = 10,
    n_iter: int = 2,
    random_state: int | None = 0,
) -> FloatArray:
    """Compute cumulative POD energy curve for snapshot matrix Y."""
    Y_arr = _as_2d_float_array(Y, name="Y")
    Yc = Y_arr - Y_arr.mean(axis=0)

    max_rank = min(Yc.shape)
    r = max_rank if r_max is None else min(int(r_max), max_rank)

    if randomized:
        _U, S, _Vt = randomized_svd(
            Yc,
            n_components=r,
            n_oversamples=n_oversamples,
            n_iter=n_iter,
            random_state=random_state,
        )
    else:
        _U, S_full, _Vt_full = np.linalg.svd(Yc, full_matrices=False)
        S = S_full[:r]

    S = np.asarray(S, dtype=float)
    total = float(np.sum(S**2))
    if total == 0.0:
        return np.zeros_like(S)
    return np.cumsum(S**2) / total
