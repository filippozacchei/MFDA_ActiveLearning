from __future__ import annotations

import numpy as np
import pytest

from gp_active_mcmc.surrogates.gp import MultiOutputGP
from gp_active_mcmc.surrogates.pod import POD, pod_energy
from gp_active_mcmc.surrogates.podgp import PODGPSurrogate


def _make_snapshots(*, n_snapshots: int = 40, n_time: int = 80, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n_time)

    # Low-rank signal: 3 temporal modes
    modes = np.vstack(
        [
            np.sin(2 * np.pi * t),
            np.cos(2 * np.pi * t),
            np.sin(4 * np.pi * t),
        ]
    )  # (3, n_time)

    coeffs = rng.normal(size=(n_snapshots, 3))  # (n_snapshots, 3)
    signal = coeffs @ modes  # (n_snapshots, n_time)
    noise = 0.01 * rng.normal(size=(n_snapshots, n_time))
    return signal + noise


def test_fit_sets_attributes_and_shapes() -> None:
    Y = _make_snapshots()
    pod = POD(rank=5, randomized=False).fit(Y)

    assert pod.is_fitted
    assert pod.mean_ is not None
    assert pod.components_ is not None
    assert pod.singular_values_ is not None
    assert pod.explained_energy_ is not None

    n_time = Y.shape[1]
    r = min(5, min(Y.shape))
    assert pod.mean_.shape == (n_time,)
    assert pod.components_.shape == (r, n_time)
    assert pod.singular_values_.shape == (r,)
    assert pod.explained_energy_.shape == (r,)


def test_transform_inverse_transform_roundtrip_low_error() -> None:
    Y = _make_snapshots()
    pod = POD(rank=6, randomized=False).fit(Y)

    A = pod.transform(Y)
    Y_hat = pod.inverse_transform(A)

    assert A.shape == (Y.shape[0], pod.rank_)
    assert Y_hat.shape == Y.shape

    rel_err = np.linalg.norm(Y - Y_hat) / np.linalg.norm(Y)
    assert rel_err < 0.1


def test_fit_transform_matches_fit_then_transform() -> None:
    Y = _make_snapshots()

    pod = POD(rank=4, randomized=False)
    A1 = pod.fit_transform(Y)

    pod2 = POD(rank=4, randomized=False).fit(Y)
    A2 = pod2.transform(Y)

    np.testing.assert_allclose(A1, A2, rtol=1e-12, atol=1e-12)


def test_inverse_transform_requires_correct_rank() -> None:
    Y = _make_snapshots()
    pod = POD(rank=4, randomized=False).fit(Y)

    A = pod.transform(Y)
    A_bad = A[:, :2]  # wrong rank

    with pytest.raises(ValueError, match="incompatible coefficient dimension"):
        _ = pod.inverse_transform(A_bad)


def test_transform_requires_matching_time_dimension() -> None:
    Y = _make_snapshots()
    pod = POD(rank=4, randomized=False).fit(Y)

    Y_bad = Y[:, :-1]  # wrong n_time

    with pytest.raises(ValueError, match="incompatible time dimension"):
        _ = pod.transform(Y_bad)


def test_energy_curve_monotone_and_bounded() -> None:
    Y = _make_snapshots()
    e = pod_energy(Y, randomized=False)

    assert e.ndim == 1
    assert np.all(e >= 0.0)
    assert np.all(e <= 1.0 + 1e-12)
    assert np.all(np.diff(e) >= -1e-12)  # monotone nondecreasing


def test_methods_raise_before_fit() -> None:
    Y = _make_snapshots()
    pod = POD(rank=3, randomized=False)

    with pytest.raises(RuntimeError, match="not fitted"):
        _ = pod.transform(Y)

    with pytest.raises(RuntimeError, match="not fitted"):
        _ = pod.inverse_transform(np.zeros((Y.shape[0], 3)))


def test_randomized_fit_smoke_test() -> None:
    """Randomized SVD path should run and produce consistent shapes."""
    Y = _make_snapshots()
    pod = POD(rank=5, randomized=True, random_state=0).fit(Y)

    assert pod.is_fitted
    assert pod.components_ is not None
    assert pod.components_.shape == (min(5, min(Y.shape)), Y.shape[1])


def test_multioutput_gp_predict_shapes_and_positive_var() -> None:
    rng = np.random.default_rng(0)
    n, d, m = 30, 2, 5
    X = rng.normal(size=(n, d))
    Y = rng.normal(size=(n, m))

    gp = MultiOutputGP(X, Y, kernel="matern52", ard=True, update_every=0, n_retrain_max=0)

    Xq = rng.normal(size=(7, d))
    mean, var = gp.predict(Xq)

    assert mean.shape == (7, m)
    assert var.shape == (7, m)
    assert np.all(var >= 0.0)


def test_multioutput_gp_update_increases_training_size() -> None:
    rng = np.random.default_rng(1)
    n, d, m = 20, 3, 4
    X = rng.normal(size=(n, d))
    Y = rng.normal(size=(n, m))

    gp = MultiOutputGP(X, Y, kernel="rbf", ard=True, update_every=0, n_retrain_max=0)

    n_before = gp.n_train
    x_new = rng.normal(size=(1, d))
    y_new = rng.normal(size=(m,))

    gp.update(x_new, y_new)
    assert gp.n_train == n_before + 1


def _make_dataset(*, n: int = 40, d: int = 2, n_time: int = 80, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))

    t = np.linspace(0.0, 1.0, n_time)
    modes = np.vstack(
        [
            np.sin(2 * np.pi * t),
            np.cos(2 * np.pi * t),
            np.sin(4 * np.pi * t),
        ]
    )  # (3, n_time)

    # coefficients depend smoothly on X
    a = np.column_stack(
        [
            0.8 * X[:, 0] + 0.1 * rng.normal(size=n),
            -0.5 * X[:, 1] + 0.1 * rng.normal(size=n),
            0.3 * (X[:, 0] - X[:, 1]) + 0.1 * rng.normal(size=n),
        ]
    )  # (n, 3)

    Y = a @ modes + 0.01 * rng.normal(size=(n, n_time))
    return X, Y


def test_podgp_predict_shapes() -> None:
    X, Y = _make_dataset()

    pod = POD(rank=5, randomized=False).fit(Y)
    A = pod.transform(Y)  # (n, r)

    gp = MultiOutputGP(X, A, kernel="matern52", ard=True, update_every=0, n_retrain_max=0)
    surr = PODGPSurrogate(pod=pod, gp=gp)

    theta = X[0]
    y_mean, y_var = surr.predict(theta)

    assert y_mean.shape == (Y.shape[1],)
    assert y_var.shape == (Y.shape[1],)
    assert np.all(y_var >= 0.0)

    Y_mean_b, Y_var_b = surr.predict(X[:5])
    assert Y_mean_b.shape == (5, Y.shape[1])
    assert Y_var_b.shape == (5, Y.shape[1])


def test_podgp_update_increases_training_size() -> None:
    X, Y = _make_dataset(seed=1)

    pod = POD(rank=6, randomized=False).fit(Y)
    A = pod.transform(Y)

    gp = MultiOutputGP(X, A, kernel="rbf", ard=True, update_every=0, n_retrain_max=0)
    surr = PODGPSurrogate(pod=pod, gp=gp)

    n_before = gp.n_train
    surr.update(X[0], Y[0])
    assert gp.n_train == n_before + 1
