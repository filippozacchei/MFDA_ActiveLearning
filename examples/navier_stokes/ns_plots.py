"""Plotting functions for the Navier-Stokes (BFS outlet-profile) benchmark's paper
figures.

Mirrors `examples/toy_problem/msd_plots.py`'s role and API exactly, generalized in two
places MSD's 2-D problem didn't need to: no `problem.t` (NS's `Problem` has no time
axis -- see `ns_methods.Problem`'s docstring), and `theta` is 3-D here
(`h1, U_in, L_down`) rather than 2-D, so the 2-D scatter plots show the first two
parameters (`param_names[:2]`) and the trace plot has one column per parameter
instead of a hardcoded two.

Shared by `ns_benchmark.ipynb` and `run.py`, so the per-seed figure gallery the sweep
script saves to `results/figures/` and the figures shown inline in the notebook are
produced by the exact same code.

Four methods appear across these plots, but not all in every one: `hf_only` (ground
truth), `pretrained`, `adaptive_surrogate_mcmc` (Riccius, synced, no DA correction),
`adaptive_stm`. The posterior-accuracy figures compare
`hf_only`/`adaptive_surrogate_mcmc`/`adaptive_stm` only (`pretrained`'s frozen,
uncorrected posterior isn't a fair comparison against DA-corrected chains -- see
`msd_benchmark.ipynb`'s introduction, which explains this in full); `pretrained`
appears in the surrogate-comparison and training-cost figures instead, where the
comparison is between training strategies, not posteriors.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

METHOD_COLORS = {
    "hf_only": "0.35",
    "pretrained": "tab:red",
    "adaptive_surrogate_mcmc": "tab:green",
    "adaptive_stm": "tab:blue",
}
METHOD_MARKERS = {"pretrained": "s", "adaptive_surrogate_mcmc": "D", "adaptive_stm": "o"}
METHOD_LABELS = {
    "hf_only": "MH with HF",
    "pretrained": "Pretrained (offline)",
    "adaptive_surrogate_mcmc": "Adaptive surrogate MCMC (Riccius)",
    "adaptive_stm": "Adaptive STM (ours)",
}
SURROGATE_METHODS = ("pretrained", "adaptive_surrogate_mcmc", "adaptive_stm")
POSTERIOR_METHODS = ("adaptive_surrogate_mcmc", "adaptive_stm")
TRACE_METHODS = ("hf_only", "adaptive_surrogate_mcmc", "adaptive_stm")
LF_COLOR = "tab:blue"
HF_COLOR = "tab:red"


def plot_surrogate_comparison(
    problem, seed_surrogate, surrogates, *, methods: tuple[str, ...] = SURROGATE_METHODS, title_suffix: str = ""
) -> Figure:
    """Surrogate prediction at `theta_true`, before vs. after training, for `methods`
    (default all three surrogate-based methods:
    `pretrained`/`adaptive_surrogate_mcmc`/`adaptive_stm`) side by side.

    `seed_surrogate`: the shared offline design's surrogate, before any method-specific
    training -- the "before" reference in every panel. `surrogates[name]` is either one
    surrogate (`pretrained`: a single global design) or a list of per-replicate
    surrogates (`adaptive_surrogate_mcmc`/`adaptive_stm` under
    `run_convergence_driven_comparison`'s default decentralized runs, one entry per
    chain) -- every chain's curve is overlaid, with the min/max envelope across chains
    shaded, so this is the plot for checking at a glance whether independently-trained
    chains agree.
    """
    import matplotlib.pyplot as plt

    axes_1d = plt.subplots(1, len(methods), figsize=(6 * len(methods), 4), sharey=True)[1]
    axes = np.atleast_1d(axes_1d)
    fig = axes[0].figure
    t = np.arange(problem.y_obs.shape[0])  # NS has no problem.t; index along the resampled outlet line
    y_true = problem.hf_forward(problem.theta_true)
    mean_before, var_before = seed_surrogate.predict(problem.theta_true)

    for ax, name in zip(axes, methods, strict=True):
        ax.plot(t, problem.y_obs, ".", ms=2, alpha=0.3, color="0.5", label="observations")
        ax.plot(t, y_true, color="k", lw=1.5, label="true profile")
        ax.plot(t, mean_before, color="tab:gray", lw=1.5, label="surrogate (before)")
        ax.fill_between(
            t, mean_before - 2 * np.sqrt(var_before), mean_before + 2 * np.sqrt(var_before),
            color="tab:gray", alpha=0.2,
        )

        entry = surrogates[name]
        chain_surrogates = entry if isinstance(entry, list) else [entry]
        means_after = np.stack([s.predict(problem.theta_true)[0] for s in chain_surrogates])

        if len(chain_surrogates) == 1:
            var_after = chain_surrogates[0].predict(problem.theta_true)[1]
            ax.plot(t, means_after[0], color=METHOD_COLORS[name], lw=1.5, label="surrogate (after)")
            ax.fill_between(
                t, means_after[0] - 2 * np.sqrt(var_after), means_after[0] + 2 * np.sqrt(var_after),
                color=METHOD_COLORS[name], alpha=0.25,
            )
        else:
            for i, mean_after in enumerate(means_after):
                ax.plot(
                    t, mean_after, color=METHOD_COLORS[name], lw=1.0, alpha=0.6,
                    label=f"surrogate (after) -- {len(chain_surrogates)} chains" if i == 0 else None,
                )
            ax.fill_between(
                t, means_after.min(axis=0), means_after.max(axis=0), color=METHOD_COLORS[name], alpha=0.15,
                label="inter-chain spread",
            )

        ax.set_title(METHOD_LABELS[name])
        ax.set_xlabel("outlet sample index")
    axes[0].set_ylabel("u_x(y)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Surrogate prediction at theta_true{title_suffix}")
    fig.tight_layout()
    return fig


def plot_training_points_scatter(
    problem, seed_surrogate, surrogates_by_method, *, methods: tuple[str, ...] = SURROGATE_METHODS,
    title_suffix: str = "",
) -> Figure:
    """Where each method's acquired HF training points land in parameter space (first
    two of `problem.param_names`, `h1`/`U_in`), vs. the shared offline seed design
    every method starts from.

    `surrogates_by_method[name]` is either one surrogate (`pretrained`: a single
    global design) or a list of per-replicate surrogates
    (`adaptive_surrogate_mcmc`/`adaptive_stm` under `run_convergence_driven_comparison`'s
    default decentralized runs, one entry per chain) -- pooled here across replicates,
    so this is the plot to check whether independently-trained chains explored a
    consistent region of parameter space or scattered apart from each other and from
    `pretrained`'s single global design.
    """
    import matplotlib.pyplot as plt

    n_seed = seed_surrogate.X_history.shape[0]
    seed_X = seed_surrogate.X_history

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(
        seed_X[:, 0], seed_X[:, 1], s=20, color="0.5", marker="x", linewidths=1.0,
        label=f"seed design (n={n_seed})", zorder=2,
    )

    for name in methods:
        entry = surrogates_by_method[name]
        chain_surrogates = entry if isinstance(entry, list) else [entry]
        acquired = np.concatenate([s.X_history[n_seed:] for s in chain_surrogates], axis=0)
        if acquired.shape[0] == 0:
            continue
        ax.scatter(
            acquired[:, 0], acquired[:, 1], s=16, alpha=0.4, facecolors="none",
            edgecolors=METHOD_COLORS[name], marker=METHOD_MARKERS.get(name, "o"), linewidths=0.8,
            label=f"{METHOD_LABELS[name]} (n={acquired.shape[0]}, {len(chain_surrogates)} chain(s))", zorder=3,
        )

    ax.scatter(
        problem.theta_true[0], problem.theta_true[1], s=180, c="k", marker="*", zorder=5,
        label=r"$\theta_{\mathrm{true}}$",
    )
    ax.set_xlabel(problem.param_names[0])
    ax.set_ylabel(problem.param_names[1])
    ax.set_title(f"Acquired HF training points{title_suffix}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_posterior_scatter(
    problem, chains_by_method, burn_ins: dict[str, int], *,
    methods: tuple[str, ...] = POSTERIOR_METHODS, title_suffix: str = "",
) -> Figure:
    """Pooled posterior scatter: `hf_only` (grey hexbin, the ground truth for this
    problem instance) vs. `methods` (default `adaptive_surrogate_mcmc`/`adaptive_stm`),
    each pooled across its replicate chains at its own R-hat-validated burn-in.

    `theta` is 3-D here (`param_names`); this shows the first two dimensions
    (`h1`, `U_in`) only -- the same 2-D-scatter convention `backward.py`'s original
    demo used.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ref_post = np.concatenate([c.burn_in(burn_ins["hf_only"]).samples for c in chains_by_method["hf_only"]], axis=0)
    ax.hexbin(ref_post[:, 0], ref_post[:, 1], gridsize=40, cmap="Greys", mincnt=1, alpha=0.55, zorder=1)

    for name in methods:
        pooled = np.concatenate(
            [c.burn_in(burn_ins[name]).samples for c in chains_by_method[name]], axis=0
        )
        ax.scatter(
            pooled[:, 0], pooled[:, 1], s=16, alpha=0.35, facecolors="none", edgecolors=METHOD_COLORS[name],
            marker=METHOD_MARKERS[name], linewidths=0.8, label=METHOD_LABELS[name], zorder=3,
        )
    ax.scatter(problem.theta_true[0], problem.theta_true[1], s=180, c="k", marker="*", zorder=5, label=r"$\theta_{\mathrm{true}}$")
    ax.set_xlabel(problem.param_names[0])
    ax.set_ylabel(problem.param_names[1])
    ax.set_title(f"Posterior comparison{title_suffix}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_training_cost_boxplot(offline_extra: np.ndarray, online_extra: np.ndarray, *, n_init: int) -> Figure:
    """Training-cost comparison across seeds: extra HF calls beyond the shared offline
    seed design, `pretrained` (offline, global active learning) vs. `adaptive_stm`
    (online, MCMC-path-guided adaptive phase)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.boxplot(
        [offline_extra, online_extra], tick_labels=["Pretrained\n(offline)", "Adaptive STM\n(online adaptive)"]
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
    `hf_only` reference (left) and KL divergence to it (right), one box per method.

    Each method's box is typically drawn from a *different* number of seeds (only
    the ones where it converged), so the title reports one count per method rather
    than a single combined "n seeds" that would silently describe only whichever
    method happened to be first.
    """
    import matplotlib.pyplot as plt

    counts = ", ".join(f"{METHOD_LABELS[m]}: {len(w2_by_method[m])}" for m in method_order)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].boxplot([w2_by_method[m] for m in method_order], tick_labels=[METHOD_LABELS[m] for m in method_order])
    axes[0].set_ylabel("Wasserstein-2 distance to HF-only reference")
    axes[0].set_yscale('log')
    axes[1].boxplot([kl_by_method[m] for m in method_order], tick_labels=[METHOD_LABELS[m] for m in method_order])
    axes[1].set_ylabel("KL divergence to HF-only reference")
    plt.yscale('log')
    fig.suptitle(f"Posterior-accuracy comparison (converged seeds -- {counts})")
    fig.tight_layout()
    return fig


def plot_traces(
    problem, traces: dict, burn_ins: dict, *, mode: str = "full", title_suffix: str = ""
) -> Figure:
    """Trace plots (one row per method, one column per parameter), overlaying every
    replicate chain, colored by whether each point required an HF evaluation (blue =
    cheap LF/surrogate step, red = expensive HF call) -- shows the actual mixing
    behaviour (many cheap LF steps between infrequent HF corrections for `adaptive_stm`)
    that a posterior scatter alone can't.

    `traces`/`burn_ins`: from `gp_active_mcmc.verification.prepare_trace_data`. One
    column per entry in `problem.param_names` (3 here, vs. MSD's 2).
    `mode="full"`: every sample, with a vertical line at each method's burn-in.
    `mode="post_burn_in"`: only samples after each method's own burn-in.
    """
    import matplotlib.pyplot as plt

    n_params = len(problem.param_names)
    fig, axes = plt.subplots(
        len(TRACE_METHODS), n_params, figsize=(5.5 * n_params, 2.6 * len(TRACE_METHODS)), sharex=False,
    )
    for r, name in enumerate(TRACE_METHODS):
        chain_traces = traces[name]
        b = burn_ins[name]
        for p in range(n_params):
            ax = axes[r, p]
            for i, (x, samples, used_hf) in enumerate(chain_traces):
                if mode == "post_burn_in":
                    mask = x > b
                    x, samples, used_hf = x[mask], samples[mask], used_hf[mask]
                ax.plot(x, samples[:, p], color="0.85", lw=0.3, zorder=1)
                ax.scatter(
                    x[~used_hf], samples[~used_hf, p], s=3, color=LF_COLOR, alpha=0.35, zorder=2,
                    label="LF step" if (i == 0 and r == 0 and p == 1) else None,
                )
                ax.scatter(
                    x[used_hf], samples[used_hf, p], s=10, color=HF_COLOR, alpha=0.6, zorder=3,
                    label="HF call" if (i == 0 and r == 0 and p == 1) else None,
                )
            ax.axhline(problem.theta_true[p], color="k", ls="--", lw=1)
            if mode == "full":
                ax.axvline(b, color="tab:green", ls=":", lw=1.5)
            if r == 0:
                ax.set_title(problem.param_names[p])
            if p == 0:
                ax.set_ylabel(f"{METHOD_LABELS.get(name, name)}\n({len(chain_traces)} chains)", fontsize=9)
            if r == 0 and p == 1:
                ax.legend(fontsize=7, loc="upper right", markerscale=2)
    for p in range(n_params):
        axes[-1, p].set_xlabel("Coarse-evaluation iteration")
    mode_suffix = " (full range, burn-in marked)" if mode == "full" else " (post burn-in)"
    fig.suptitle(f"Trace plots (LF/HF colored){mode_suffix}{title_suffix}")
    fig.tight_layout()
    return fig
