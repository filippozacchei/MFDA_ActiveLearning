# src/your_pkg/cfd/outlet.py
from __future__ import annotations

import numpy as np
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells

from .types import OutletProfile


def sample_outlet_u_x(
    *,
    mesh,
    u_function,  # dolfinx.fem.Function on vector space
    L: float,
    H: float,
    ny: int = 100,
) -> OutletProfile:
    """
    Sample the streamwise velocity component u_x on the outlet line x=L.
    """
    y = np.linspace(0.0, H, ny)
    pts = np.column_stack([np.full_like(y, L), y, np.zeros_like(y)])

    tree = bb_tree(mesh, mesh.geometry.dim)
    cand = compute_collisions_points(tree, pts)
    coll = compute_colliding_cells(mesh, cand, pts)

    u_x = np.zeros_like(y)
    for i in range(ny):
        links = coll.links(i)
        if len(links) > 0:
            uu = u_function.eval(pts[i], links[:1])
            u_x[i] = uu[0]

    return OutletProfile(y=y, u_x=u_x)

def resample_profile(y: np.ndarray, u: np.ndarray, *, T: int) -> np.ndarray:
    y = np.asarray(y).ravel()
    u = np.asarray(u).ravel()
    if y.size != u.size:
        raise ValueError("y and u must have same length")
    if y.size < 2:
        raise ValueError("Need at least two points to resample")
    if not np.all(np.diff(y) >= 0):
        idx = np.argsort(y)
        y = y[idx]
        u = u[idx]
    y_new = np.linspace(float(y.min()), float(y.max()), T)
    return np.interp(y_new, y, u)