"""Plotting functions for the mass-spring-damper benchmark's paper figures.

Shared by `msd_benchmark.ipynb` and `run_sweep_convergence_driven.py`, so the per-seed
figure gallery the sweep script saves to `results/figures/` and the figures shown
inline in the notebook are produced by the exact same code.

Four methods appear across these plots, but not all in every one:
`hf_only` (ground truth), `pretrained`, `online_active` (Riccius-style, no DA
correction), `ours`. The posterior-accuracy figures compare `hf_only`/`online_active`/
`ours` only (`pretrained`'s frozen, uncorrected posterior isn't a fair comparison
against DA-corrected chains -- see the notebook's introduction); `pretrained` appears
in the surrogate-comparison and training-cost figures instead, where the comparison is
between training strategies, not posteriors.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

METHOD_COLORS = {
    "hf_only": "0.35",
    "pretrained": "tab:red",
    "online_active": "tab:orange",
    "ours": "tab:blue",
}
METHOD_MARKERS = {"pretrained": "s", "online_active": "^", "ours": "o"}
METHOD_LABELS = {
    "hf_only": "MH with HF",
    "pretrained": "Pretrained (offline)",
    "online_active": "Riccius-style online",
    "ours": "Ours (adaptive DA + freeze)",
}
SURROGATE_METHODS = ("pretrained", "online_active", "ours")
POSTERIOR_METHODS = ("online_active", "ours")


def plot_surrogate_comparison(problem, seed_surrogate, surrogates, *, title_suffix: str = "") -> Figure:
    """Surrogate prediction at `theta_true`, before vs. after training, for the three
    surrogate-based methods (`pretrained`/`online_active`/`ours`) side by side.

    `seed_surrogate`: the shared offline design's surrogate, before any method-specific
    training -- the "before" reference in every panel. `surrogates`: dict with one
    trained ("after") surrogate per method in `SURROGATE_METHODS`.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(SURROGATE_METHODS), figsize=(6 * len(SURROGATE_METHODS), 4), sharey=True)
    t = problem.t
    y_true = problem.hf_forward(problem.theta_true)
    mean_before, var_before = seed_surrogate.predict(problem.theta_true)

    for ax, name in zip(axes, SURROGATE_METHODS, strict=True):
        ax.plot(t, problem.y_obs, ".", ms=2, alpha=0.3, color="0.5", label="observations")
        ax.plot(t, y_true, color="k", lw=1.5, label="true profile")
        ax.plot(t, mean_before, color="tab:gray", lw=1.5, label="surrogate (before)")
        ax.fill_between(
            t, mean_before - 2 * np.sqrt(var_before), mean_before + 2 * np.sqrt(var_before),
            color="tab:gray", alpha=0.2,
        )

        mean_after, var_after = surrogates[name].predict(problem.theta_true)
        ax.plot(t, mean_after, color=METHOD_COLORS[name], lw=1.5, label="surrogate (after)")
        ax.fill_between(
            t, mean_after - 2 * np.sqrt(var_after), mean_after + 2 * np.sqrt(var_after),
            color=METHOD_COLORS[name], alpha=0.25,
        )

        ax.set_title(METHOD_LABELS[name])
        ax.set_xlabel("t")
    axes[0].set_ylabel("x(t)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Surrogate prediction at theta_true{title_suffix}")
    fig.tight_layout()
    return fig


def plot_posterior_scatter(problem, chains_by_method, burn_ins: dict[str, int], *, title_suffix: str = "") -> Figure:
    """Pooled posterior scatter: `hf_only` (grey hexbin, the ground truth for this
    problem instance) vs. `online_active` and `ours` (scatter), each pooled across its
    replicate chains at its own R-hat-validated burn-in.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ref_post = np.concatenate([c.burn_in(burn_ins["hf_only"]).samples for c in chains_by_method["hf_only"]], axis=0)
    ax.hexbin(ref_post[:, 0], ref_post[:, 1], gridsize=40, cmap="Greys", mincnt=1, alpha=0.55, zorder=1)

    for name in POSTERIOR_METHODS:
        pooled = np.concatenate(
            [c.burn_in(burn_ins[name]).samples for c in chains_by_method[name]], axis=0
        )
        ax.scatter(
            pooled[:, 0], pooled[:, 1], s=16, alpha=0.35, facecolors="none", edgecolors=METHOD_COLORS[name],
            marker=METHOD_MARKERS[name], linewidths=0.8, label=METHOD_LABELS[name], zorder=3,
        )
    ax.scatter(*problem.theta_true, s=180, c="k", marker="*", zorder=5, label=r"$\theta_{\mathrm{true}}$")
    ax.set_xlabel("k")
    ax.set_ylabel("c")
    ax.set_title(f"Posterior comparison{title_suffix}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_training_cost_boxplot(offline_extra: np.ndarray, online_extra: np.ndarray, *, n_init: int) -> Figure:
    """Training-cost comparison across seeds: extra HF calls beyond the shared offline
    seed design, `pretrained` (offline, global active learning) vs. `ours` (online,
    MCMC-path-guided adaptive phase)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.boxplot(
        [offline_extra, online_extra], tick_labels=["Pretrained\n(offline)", "Ours\n(online adaptive)"]
    )
    rng = np.random.default_rng(0)
    for i, vals in enumerate([offline_extra, online_extra], start=1):
        ax.scatter(np.full(len(vals), i) + rng.uniform(-0.05, 0.05, len(vals)), vals, alpha=0.5, s=20, color="0.3", zorder=3)
    ax.set_ylabel(f"Extra HF calls beyond shared\nn_init={n_init} seed design")
    ax.set_title(f"Training-cost comparison across {len(offline_extra)} seeds")
    fig.tight_layout()
    return fig


def plot_accuracy_boxplots(
    w2_by_method: dict[str, np.ndarray], kl_by_method: dict[str, np.ndarray], *,
    method_order: tuple[str, ...] = POSTERIOR_METHODS,
) -> Figure:
    """Posterior-accuracy comparison across seeds: Wasserstein-2 distance to the
    `hf_only` reference (left) and KL divergence to it (right), one box per method."""
    import matplotlib.pyplot as plt

    n = len(next(iter(w2_by_method.values())))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].boxplot([w2_by_method[m] for m in method_order], tick_labels=[METHOD_LABELS[m] for m in method_order])
    axes[0].set_ylabel("Wasserstein-2 distance to HF-only reference")
    axes[1].boxplot([kl_by_method[m] for m in method_order], tick_labels=[METHOD_LABELS[m] for m in method_order])
    axes[1].set_ylabel("KL divergence to HF-only reference")
    fig.suptitle(f"Posterior-accuracy comparison across {n} seeds")
    fig.tight_layout()
    return fig
