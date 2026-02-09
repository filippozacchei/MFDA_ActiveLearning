"""Plotting utilities for diagnostics.

All functions return matplotlib (fig, ax) objects and never call `plt.show()`.
"""

from __future__ import annotations

from gp_active_mcmc.diagnostics.mcmc import (
    plot_chain_2d,
    plot_cumulative_hf_fraction,
    plot_subchain_length_history,
    plot_surrogate_error_history,
)
from gp_active_mcmc.diagnostics.pod import plot_pod_energy, pod_energy_from_snapshots
from gp_active_mcmc.diagnostics.surrogate import (
    plot_error_vs_uncertainty,
    plot_prediction_at_theta,
)

__all__ = [
    "plot_chain_2d",
    "plot_cumulative_hf_fraction",
    "plot_subchain_length_history",
    "plot_surrogate_error_history",
    "pod_energy_from_snapshots",
    "plot_pod_energy",
    "plot_prediction_at_theta",
    "plot_error_vs_uncertainty",
]
