# examples/navier_stokes/utils/outlet.py
from __future__ import annotations

import numpy as np
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells

from .types import OutletProfile


"""Outlet sampling utilities (QoI extraction).

This module defines the quantity of interest (QoI) used in the Navier–Stokes example:
the outlet streamwise velocity profile u_x(y) at the plane x = L.

Design goals
------------
- Return a small, NumPy-friendly object suitable for surrogate modelling.
- Keep the sampling logic explicit and robust to points that fall outside the mesh
  due to floating-point tolerances.

Notes on dolfinx point evaluation
---------------------------------
Evaluating a finite-element function at arbitrary points requires locating the cell
containing each point. We use:

- `bb_tree` to accelerate spatial queries,
- `compute_collisions_points` / `compute_colliding_cells` to map points to candidate cells,
- `Function.eval(point, cells)` to evaluate on an owning cell.

Points that do not collide with any cell are left as NaN by default (configurable),
because silently filling zeros can hide geometry/mesh issues.
"""


def sample_outlet_u_x(
    *,
    mesh,
    u_function,  # dolfinx.fem.Function on a vector-valued space
    L: float,
    H: float,
    ny: int = 100,
    fill_value: float = np.nan,
    atol: float = 1e-12,
) -> OutletProfile:
    """Sample the streamwise velocity component u_x on the outlet line x = L.

    Parameters
    ----------
    mesh
        dolfinx mesh.
    u_function
        Velocity field as a dolfinx function on a vector space (shape (gdim,)).
    L
        Outlet x-coordinate (typically total domain length).
    H
        Outlet height (upper y-coordinate).
    ny
        Number of sampling points along y in [0, H].
    fill_value
        Value to use if a sampling point cannot be located in the mesh.
        Default is NaN to make failures visible.
    atol
        Absolute tolerance used to "snap" x-coordinates to exactly L to mitigate
        floating-point accumulation.

    Returns
    -------
    profile
        Outlet profile containing:
        - y: sampling coordinates (ny,),
        - u_x: sampled streamwise velocity (ny,).

    Raises
    ------
    ValueError
        If `ny < 2`, `H <= 0`, or `L < 0`.
    """
    if ny < 2:
        raise ValueError("ny must be >= 2.")
    if H <= 0.0:
        raise ValueError("H must be positive.")
    if L < 0.0:
        raise ValueError("L must be non-negative.")

    y = np.linspace(0.0, float(H), int(ny))

    # Points need shape (n, 3) for dolfinx geometry utilities.
    # For 2D meshes, z is ignored but must be present.
    x = np.full_like(y, float(L))
    if atol > 0.0:
        x[np.isclose(x, L, atol=atol)] = float(L)

    pts = np.column_stack([x, y, np.zeros_like(y)])

    tree = bb_tree(mesh, mesh.geometry.dim)
    candidates = compute_collisions_points(tree, pts)
    collisions = compute_colliding_cells(mesh, candidates, pts)

    u_x = np.full_like(y, float(fill_value), dtype=float)

    # Evaluate on the first colliding cell, if any.
    for i in range(y.size):
        cell_ids = collisions.links(i)
        if len(cell_ids) == 0:
            continue
        val = u_function.eval(pts[i], cell_ids[:1])
        # val is a vector (u_x, u_y, ...) depending on gdim; we want streamwise component.
        u_x[i] = float(val[0])

    return OutletProfile(y=y, u_x=u_x)


def resample_profile(y: np.ndarray, u: np.ndarray, *, T: int) -> np.ndarray:
    """Resample a 1D profile onto a uniform grid using linear interpolation.

    This is useful when downstream workflows expect a fixed-length vector, e.g.:
    - POD on outlet profiles,
    - GP training on coefficients,
    - likelihood evaluations on a fixed observation grid.

    Parameters
    ----------
    y
        Coordinates (n,). May be unsorted.
    u
        Profile values (n,), aligned with `y`.
    T
        Number of points in the resampled grid.

    Returns
    -------
    u_new
        Resampled profile of length `T`.

    Raises
    ------
    ValueError
        If shapes mismatch, if `T < 2`, or if fewer than two samples are provided.
    """
    y = np.asarray(y, dtype=float).ravel()
    u = np.asarray(u, dtype=float).ravel()

    if y.size != u.size:
        raise ValueError("y and u must have the same length.")
    if y.size < 2:
        raise ValueError("Need at least two points to resample.")
    if int(T) < 2:
        raise ValueError("T must be >= 2.")

    # Ensure monotone y for np.interp
    if not np.all(np.diff(y) >= 0.0):
        idx = np.argsort(y)
        y = y[idx]
        u = u[idx]

    y_new = np.linspace(float(y.min()), float(y.max()), int(T))
    return np.interp(y_new, y, u)
