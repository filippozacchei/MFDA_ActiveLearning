from __future__ import annotations

from pathlib import Path

import numpy as np

from utils.types import BFSGeometry
from utils.solver import MFTimeConfig, solve_ipcs_bfs
from utils.animation import FieldSampleGrid, make_velocity_animation


def main() -> None:
    # Standard geometry + BC (your defaults)
    geom = BFSGeometry(h1=0.10, h2=0.20, L_up=0.10, L_down=0.40)
    U_in = 1.5

    # Keep runtime reasonable for a tutorial animation
    time = MFTimeConfig(dt=1e-3, t_end=2.0, progress=True)

    prof, mesh, L, H, frames = solve_ipcs_bfs(
        geom=geom,
        U_in=U_in,
        time=time,
        outlet_ny=150,
        store_velocity_frames=True,
        frame_stride=10,  # record every 5 steps
    )

    # Plot window: slightly padded rectangle around the domain
    grid = FieldSampleGrid(
        x=np.linspace(0.0, L, 240),
        y=np.linspace(0.0, H, 120),
    )

    out = Path("examples/navier_stokes/outputs/velocity.gif")
    make_velocity_animation(
        mesh=mesh,
        frames=frames,
        grid=grid,
        outpath=out,
        interval_ms=80,
        title="BFS IPCS: velocity magnitude",
    )

    print(f"Saved animation to: {out}")
    print(f"Outlet profile sampled at ny={prof.y.size} points.")


if __name__ == "__main__":
    main()
