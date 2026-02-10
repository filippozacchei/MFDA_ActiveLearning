from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.utils.extmath import randomized_svd

"""Proper Orthogonal Decomposition (POD).

This module provides a compact, scikit-learn style implementation of Proper Orthogonal
Decomposition (POD), often used to compress trajectory-valued outputs into a small set of
coefficients.

Data model
----------
Snapshots are represented as a 2D array ``Y`` with shape ``(n_snapshots, n_obs)``:

- each row is one snapshot (e.g. a trajectory sampled on a time grid),
- each column is an observation component.

The POD basis is computed from the **mean-centered** snapshot matrix.

scikit-learn conventions
------------------------
- fitted attributes end with an underscore (e.g. ``mean_``),
- ``components_`` has shape ``(rank, n_obs)`` (rows are POD modes),
- ``transform`` returns coefficients with shape ``(n_snapshots, rank)``,
- ``inverse_transform`` reconstructs snapshots with shape ``(n_snapshots, n_obs)``.

In this library, POD is typically used to compress high-dimensional model outputs before
training a surrogate model (e.g. a GP) on POD coefficients.
"""

FloatArray = NDArray[np.floating]


def _as_2d_float_array(x: ArrayLike, *, name: str) -> FloatArray:
    """Convert an array-like to a 2D float array.

    Parameters
    ----------
    x
        Input array-like.
    name
        Name used in error messages.

    Returns
    -------
    x_2d
        2D float array.

    Raises
    ------
    ValueError
        If `x` is not two-dimensional.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array. Got shape {arr.shape}.")
    return arr


def _require_fitted(is_fitted: bool) -> None:
    """Raise if the POD instance has not been fitted."""
    if not is_fitted:
        raise RuntimeError("POD is not fitted yet. Call 'fit(Y)' before using this method.")


@dataclass(slots=True)
class POD:
    """Proper Orthogonal Decomposition (POD) estimator.

    The POD basis is computed via an SVD of the mean-centered snapshot matrix:

    - compute the column-wise mean `mean`,
    - form the centered matrix `Yc = Y - mean`,
    - take the first `rank` right singular vectors as POD modes.

    Parameters
    ----------
    rank
        Number of modes to retain. The effective rank is clipped to
        ``min(n_snapshots, n_obs)``.
    randomized
        If True, use randomized SVD (often faster for large snapshot matrices).
        If False, use a deterministic full SVD (`numpy.linalg.svd`).
    n_oversamples, n_iter, random_state
        Randomized SVD parameters passed to `sklearn.utils.extmath.randomized_svd`.

    Attributes
    ----------
    mean_
        Mean snapshot over the training set, shape ``(n_obs,)``.
    components_
        POD modes, shape ``(rank, n_obs)`` (rows are modes).
    singular_values_
        Retained singular values, shape ``(rank,)``.
    explained_energy_
        Cumulative explained energy, shape ``(rank,)`` in `[0, 1]`.

    Notes
    -----
    The explained energy is based on squared singular values. If `S` denotes the singular
    values of `Yc`, then the cumulative explained energy is

    .. math::

        E(k) = \\frac{\\sum_{i=1}^{k} S_i^2}{\\sum_{i} S_i^2}.

    This is commonly used to choose a rank by thresholding `E(k)` (e.g. 0.99).
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
        """Whether `fit` has been called and fitted attributes are available."""
        return (
            self.mean_ is not None
            and self.components_ is not None
            and self.singular_values_ is not None
        )

    @property
    def n_time_(self) -> int:
        """Observation dimension of the fitted snapshots.

        The name `n_time_` is kept for compatibility with trajectory use cases, but it
        should be interpreted as the number of observation components (`n_obs`).
        """
        _require_fitted(self.is_fitted)
        assert self.mean_ is not None  # guarded
        return int(self.mean_.shape[0])

    @property
    def rank_(self) -> int:
        """Effective retained rank after fitting."""
        _require_fitted(self.is_fitted)
        assert self.components_ is not None  # guarded
        return int(self.components_.shape[0])

    def fit(self, Y: ArrayLike) -> "POD":
        """Fit a POD basis from snapshots.

        Parameters
        ----------
        Y
            Snapshot matrix with shape ``(n_snapshots, n_obs)``.

        Returns
        -------
        self
            Fitted instance.

        Raises
        ------
        ValueError
            If `rank` is not positive or if `Y` is not 2D.
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
        self.components_ = np.asarray(Vt, dtype=float)  # (r, n_obs)

        total = float(np.sum(self.singular_values_**2))
        if total == 0.0:
            self.explained_energy_ = np.zeros_like(self.singular_values_)
        else:
            self.explained_energy_ = np.cumsum(self.singular_values_**2) / total

        return self

    def transform(self, Y: ArrayLike) -> FloatArray:
        """Project snapshots onto the POD basis.

        Parameters
        ----------
        Y
            Snapshot matrix of shape ``(n_snapshots, n_obs)``.

        Returns
        -------
        A
            POD coefficients of shape ``(n_snapshots, rank)``.

        Raises
        ------
        RuntimeError
            If the POD instance is not fitted.
        ValueError
            If `Y` has an incompatible observation dimension.
        """
        _require_fitted(self.is_fitted)
        Y_arr = _as_2d_float_array(Y, name="Y")

        assert self.mean_ is not None and self.components_ is not None  # guarded

        if Y_arr.shape[1] != self.mean_.shape[0]:
            raise ValueError(
                f"Y has incompatible time dimension: got {Y_arr.shape[1]}, expected {self.mean_.shape[0]}."
            )

        return (Y_arr - self.mean_) @ self.components_.T  # (n, r)

    def inverse_transform(self, A: ArrayLike) -> FloatArray:
        """Reconstruct snapshots from POD coefficients.

        Parameters
        ----------
        A
            POD coefficient matrix of shape ``(n_snapshots, rank)``.

        Returns
        -------
        Y_hat
            Reconstructed snapshots of shape ``(n_snapshots, n_obs)``.

        Raises
        ------
        RuntimeError
            If the POD instance is not fitted.
        ValueError
            If `A` has an incompatible coefficient dimension.
        """
        _require_fitted(self.is_fitted)
        A_arr = _as_2d_float_array(A, name="A")

        assert self.mean_ is not None and self.components_ is not None  # guarded

        if A_arr.shape[1] != self.rank_:
            raise ValueError(
                "A has incompatible coefficient dimension: "
                f"got {A_arr.shape[1]}, expected {self.rank_}."
            )

        return self.mean_ + A_arr @ self.components_  # (n, n_obs)

    def fit_transform(self, Y: ArrayLike) -> FloatArray:
        """Fit POD and return coefficients for the same snapshots.

        Parameters
        ----------
        Y
            Snapshot matrix of shape ``(n_snapshots, n_obs)``.

        Returns
        -------
        A
            POD coefficients of shape ``(n_snapshots, rank)``.
        """
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
    """Compute the cumulative POD energy curve for a snapshot matrix.

    This is a lightweight helper for choosing a POD rank without instantiating a
    [`POD`][gp_active_mcmc.surrogates.POD] object.

    Parameters
    ----------
    Y
        Snapshot matrix with shape ``(n_snapshots, n_obs)``.
    r_max
        Maximum rank to compute. If None, uses `min(n_snapshots, n_obs)`.
    randomized
        If True, use randomized SVD. If False, use deterministic SVD.
    n_oversamples, n_iter, random_state
        Randomized SVD parameters.

    Returns
    -------
    explained_energy
        1D array of length `r` (the computed rank). Entry `k` is the cumulative explained
        energy up to mode `k` (0-indexed). Values are in `[0, 1]`.

    Raises
    ------
    ValueError
        If `Y` is not 2D or if `r_max` is not positive when provided.
    """
    Y_arr = _as_2d_float_array(Y, name="Y")
    Yc = Y_arr - Y_arr.mean(axis=0)

    max_rank = min(Yc.shape)
    if r_max is None:
        r = max_rank
    else:
        r = int(r_max)
        if r <= 0:
            raise ValueError("r_max must be positive when provided.")
        r = min(r, max_rank)

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
