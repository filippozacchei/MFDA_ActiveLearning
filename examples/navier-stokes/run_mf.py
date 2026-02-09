# examples/navier-stokes/cfd/run_mf.py
from __future__ import annotations

import matplotlib.pyplot as plt

from utils.mf_ipcs import solve_ipcs_bfs
from utils.types import BFSGeometry


def plot_vs_height(*, heights: list[float], U_in: float) -> None:
    profiles = []
    for h2 in heights:
        prof = solve_ipcs_bfs(geom=BFSGeometry(h1=h2), U_in=U_in)
        profiles.append((h2, prof))

    plt.figure()
    for h2, prof in profiles:
        plt.plot(prof.u_x, prof.y, label=f"h2={h2:.2f}")
    plt.xlabel("uₓ (outlet)")
    plt.ylabel("y")
    plt.legend()
    plt.title("MF: outlet velocity profile vs downstream height")
    plt.tight_layout()
    plt.show()


def plot_vs_inlet_velocity(*, h2: float, inlet_velocities: list[float]) -> None:
    profiles = []
    for U_in in inlet_velocities:
        prof = solve_ipcs_bfs(geom=BFSGeometry(h1=h2), U_in=U_in)
        profiles.append((U_in, prof))

    plt.figure()
    for U_in, prof in profiles:
        plt.plot(prof.u_x, prof.y, label=f"U_in={U_in:.2f}")
    plt.xlabel("uₓ (outlet)")
    plt.ylabel("y")
    plt.legend()
    plt.title("MF: outlet velocity profile vs inlet velocity")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # plot_vs_height(heights=[0.025, 0.08, 0.10, 0.12, 0.175], U_in=1.5)
    plot_vs_inlet_velocity(h2=0.15, inlet_velocities=[0.25, 0.5, 0.75, 1.0, 1.25])
