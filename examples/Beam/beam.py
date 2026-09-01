from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.linalg import solve

def make_spatial_grid(n_pts: int = 31, length: float = 1.0) -> np.ndarray:
    """Return the 1D spatial grid for the beam."""
    return np.linspace(0.0, length, n_pts)


def build_piecewise_logE(theta: np.ndarray, x: np.ndarray, length: float = 1.0) -> np.ndarray:
    """
    Piecewise-constant log-stiffness field on 3 equal subintervals.

    theta = [m1, m2, m3]
    """
    theta = np.asarray(theta, dtype=float).reshape(-1)
    if theta.shape != (3,):
        raise ValueError("theta must have shape (3,)")

    xi = x / length
    logE = np.empty_like(x)

    logE[(xi >= 0.0) & (xi <= 1.0 / 3.0)] = theta[0]
    logE[(xi > 1.0 / 3.0) & (xi <= 2.0 / 3.0)] = theta[1]
    logE[(xi > 2.0 / 3.0) & (xi <= 1.0)] = theta[2]

    return logE

def second_derivative_matrix(n: int, h: float) -> np.ndarray:
    """Centered finite-difference approximation of the second derivative."""
    D2 = np.zeros((n, n), dtype=float)

    for i in range(1, n - 1):
        D2[i, i - 1] = 1.0 / h**2
        D2[i, i] = -2.0 / h**2
        D2[i, i + 1] = 1.0 / h**2

    return D2

def build_beam_operator(E: np.ndarray, h: float) -> np.ndarray:
    """
    Build the beam operator:
        K(E) ~ D2 @ diag(E) @ D2

    with cantilever boundary conditions:
        u(0)   = 0
        u'(0)  = 0
        u''(L) = 0
        u'''(L)= 0
    """
    n = len(E)
    D2 = second_derivative_matrix(n, h)
    A = D2 @ np.diag(E) @ D2

    # u(0) = 0
    A[0, :] = 0.0
    A[0, 0] = 1.0

    # u'(0) = 0
    A[1, :] = 0.0
    A[1, 0] = -3.0 / (2.0 * h)
    A[1, 1] =  4.0 / (2.0 * h)
    A[1, 2] = -1.0 / (2.0 * h)

    # u''(L) = 0
    A[-2, :] = 0.0
    A[-2, -1] =  1.0 / h**2
    A[-2, -2] = -2.0 / h**2
    A[-2, -3] =  1.0 / h**2

    # u'''(L) = 0
    A[-1, :] = 0.0
    A[-1, -1] =  1.0 / h**3
    A[-1, -2] = -3.0 / h**3
    A[-1, -3] =  3.0 / h**3
    A[-1, -4] = -1.0 / h**3

    return A

def build_load_vector(x: np.ndarray, load: float | np.ndarray = -1.0) -> np.ndarray:
    """Distributed load: scalar for uniform, array for pointwise."""
    load = np.asarray(load, dtype=float)
    if load.ndim == 0:
        return load.item() * np.ones_like(x, dtype=float)
    if load.shape != x.shape:
        raise ValueError(f"load shape {load.shape} != x shape {x.shape}")
    return load


def beam_forward(
    theta: np.ndarray,
    x: np.ndarray,
    load: float | np.ndarray = -1.0,
) -> np.ndarray:
    """
    Beam forward model.

    Parameters
    ----------
    theta : ndarray, shape (3,)
        Log-stiffness parameters [m1, m2, m3].
    x : ndarray, shape (n_pts,)
        Spatial grid.
    load : float or ndarray
        Distributed load: scalar for uniform, array for pointwise.

    Returns
    -------
    u : ndarray, shape (n_pts,)
        Beam displacement evaluated on the grid x.
    """
    h = x[1] - x[0]

    logE = build_piecewise_logE(theta, x, length=float(x[-1] - x[0]))
    E = np.exp(logE)

    A = build_beam_operator(E, h)
    rhs = build_load_vector(x, load=load)

    # homogeneous BC rows 


    rhs[0] = 0.0
    rhs[1] = 0.0
    rhs[-2] = 0.0
    rhs[-1] = 0.0

    u = solve(A, rhs)
    return u

def make_observation_operator(n_pts: int, obs_idx: np.ndarray) -> np.ndarray:
    """Build observation matrix B selecting entries of the state vector."""
    obs_idx = np.asarray(obs_idx, dtype=int)
    B = np.zeros((len(obs_idx), n_pts), dtype=float)

    for j, i in enumerate(obs_idx):
        B[j, i] = 1.0

    return B

def make_forward_model(
    x: np.ndarray,
    obs_idx: np.ndarray | None = None,
    load: float | np.ndarray = -1.0,
    return_full_state: bool = False,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Wrap the beam forward model with the spatial grid baked in.

    If return_full_state is False, returns y = B @ u.
    If return_full_state is True, returns the full state u.
    """
    if not return_full_state:
        if obs_idx is None:
            raise ValueError("obs_idx must be provided when return_full_state=False")
        B = make_observation_operator(len(x), obs_idx)

    def _forward(theta: np.ndarray) -> np.ndarray:
        u = beam_forward(theta, x, load=load)
        if return_full_state:
            return u
        return B @ u

    return _forward

def make_observation(
    rng: np.random.Generator,
    theta_true: np.ndarray,
    x: np.ndarray,
    sigma_obs: float,
    obs_idx: np.ndarray,
    load: float | np.ndarray = -1.0,
) -> np.ndarray:
    """
    Generate noisy synthetic observations:
        y_obs = B @ u(theta_true) + noise
    """
    forward = make_forward_model(
        x=x,
        obs_idx=obs_idx,
        load=load,
        return_full_state=False,
    )
    y_clean = forward(theta_true)
    return y_clean + rng.normal(0.0, sigma_obs, size=y_clean.shape)