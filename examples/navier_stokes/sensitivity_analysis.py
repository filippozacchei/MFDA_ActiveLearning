# examples/navier-stokes/cfd/run_mf.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

from utils.solver import solve_ipcs_bfs
from utils.types import BFSGeometry


@dataclass(frozen=True, slots=True)
class PlotConfig:
    """Plot configuration for the MF Navier–Stokes BFS example."""

    figsize: tuple[float, float] = (6.0, 4.5)
    show: bool = True
    save_dir: str | None = None  # e.g. "figures"; if None, no saving
    dpi: int = 200


def _maybe_save(fig, *, cfg: PlotConfig, name: str) -> None:
    if cfg.save_dir is None:
        return
    outdir = Path(cfg.save_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / name, dpi=cfg.dpi, bbox_inches="tight")


def plot_vs_upstream_height(
    *,
    h1_values: Iterable[float],
    U_in: float,
    geom_base: BFSGeometry = BFSGeometry(),
    cfg: PlotConfig = PlotConfig(),
) -> None:
    """Compare outlet profiles while varying downstream channel height h2."""
    profiles = []
    for h1 in h1_values:
        geom = BFSGeometry(h1=h1, h2=geom_base.h2, L_up=geom_base.L_up, L_down=geom_base.L_down)
        prof = solve_ipcs_bfs(geom=geom, U_in=float(U_in))
        profiles.append((h1, prof))

    fig, ax = plt.subplots(figsize=cfg.figsize)
    for h2, prof in profiles:
        ax.plot(prof.u_x, prof.y, label=f"h2={h2:.3f}")
    ax.set_xlabel(r"$u_x$ at outlet")
    ax.set_ylabel("y")
    ax.set_title("MF: outlet profile vs upstream height")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    _maybe_save(fig, cfg=cfg, name="mf_outlet_vs_h2.png")
    if cfg.show:
        plt.show()
    else:
        plt.close(fig)


def plot_vs_inlet_velocity(
    *,
    inlet_velocities: Iterable[float],
    h1: float,
    geom_base: BFSGeometry = BFSGeometry(),
    cfg: PlotConfig = PlotConfig(),
) -> None:
    """Compare outlet profiles while varying inlet velocity U_in."""
    profiles = []
    for U_in in inlet_velocities:
        geom = BFSGeometry(h1=h1, h2=geom_base.h2, L_up=geom_base.L_up, L_down=geom_base.L_down)
        prof = solve_ipcs_bfs(geom=geom, U_in=float(U_in))
        profiles.append((float(U_in), prof))

    fig, ax = plt.subplots(figsize=cfg.figsize)
    for U_in, prof in profiles:
        ax.plot(prof.u_x, prof.y, label=f"U_in={U_in:.2f}")
    ax.set_xlabel(r"$u_x$ at outlet")
    ax.set_ylabel("y")
    ax.set_title("MF: outlet profile vs inlet velocity")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    _maybe_save(fig, cfg=cfg, name="mf_outlet_vs_uin.png")
    if cfg.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    plot_vs_upstream_height(h1_values=[0.05, 0.075, 0.1, 0.125, 0.15], U_in=1.5)

    plot_vs_inlet_velocity(
        h1=0.1,
        inlet_velocities=[0.5, 0.75, 1.0, 1.25, 1.5],
        cfg=PlotConfig(show=True, save_dir=None),
    )
