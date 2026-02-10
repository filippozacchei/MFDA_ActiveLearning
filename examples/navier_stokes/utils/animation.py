# examples/navier-stokes/utils/animation.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells


@dataclass(frozen=True, slots=True)
class FieldSampleGrid:
    """A uniform sampling grid for 2D fields."""

    x: np.ndarray  # (nx,)
    y: np.ndarray  # (ny,)

    @property
    def points(self) -> np.ndarray:
        X, Y = np.meshgrid(self.x, self.y, indexing="xy")
        pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
        return pts


def _eval_on_grid(mesh, f, grid: FieldSampleGrid) -> np.ndarray:
    """Evaluate a dolfinx Function on a structured grid (returns values at grid points)."""
    pts = grid.points
    tree = bb_tree(mesh, mesh.geometry.dim)
    cand = compute_collisions_points(tree, pts)
    coll = compute_colliding_cells(mesh, cand, pts)

    vals = np.full((pts.shape[0],), np.nan, dtype=float)

    for i in range(pts.shape[0]):
        cell_ids = coll.links(i)
        if len(cell_ids) == 0:
            continue
        v = f.eval(pts[i], cell_ids[:1])
        # v is vector valued for velocity; we plot speed
        vals[i] = float(np.linalg.norm(v))

    return vals.reshape((grid.y.size, grid.x.size))


def make_velocity_animation(
    *,
    mesh,
    frames: list,  # list[dolfinx.fem.Function] (velocity)
    grid: FieldSampleGrid,
    outpath: str,
    interval_ms: int = 80,
    vmin: float | None = None,
    vmax: float | None = None,
    title: str = "Velocity magnitude",
) -> str:
    """Create an animation of velocity magnitude on a structured grid.

    Parameters
    ----------
    mesh
        dolfinx mesh associated with the functions.
    frames
        Velocity snapshots (dolfinx Function objects).
    grid
        Structured sampling grid.
    outpath
        Output file path. Use `.gif` for GIF.
    interval_ms
        Delay between frames (ms).
    vmin, vmax
        Fixed color limits. If None, computed from first frame.
    title
        Figure title.

    Returns
    -------
    outpath
        Path to the created animation.
    """
    outpath = str(outpath)
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)

    data0 = _eval_on_grid(mesh, frames[-1], grid)
    if vmin is None:
        vmin = float(np.nanmin(data0))
    if vmax is None:
        vmax = float(np.nanmax(data0))

    fig, ax = plt.subplots(figsize=(7, 3))
    im = ax.imshow(
        data0,
        origin="lower",
        extent=(grid.x.min(), grid.x.max(), grid.y.min(), grid.y.max()),
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r"$\|u\|$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    fig.tight_layout()

    def update(k: int):
        dat = _eval_on_grid(mesh, frames[k], grid)
        im.set_data(dat)
        ax.set_title(f"{title} (frame {k+1}/{len(frames)})")
        return (im,)

    anim = FuncAnimation(fig, update, frames=len(frames), interval=interval_ms, blit=False)

    if outpath.lower().endswith(".gif"):
        anim.save(outpath, writer=PillowWriter(fps=max(1, int(1000 / interval_ms))))
    else:
        # For MP4 you need ffmpeg installed in the environment.
        anim.save(outpath)

    plt.close(fig)
    return outpath
