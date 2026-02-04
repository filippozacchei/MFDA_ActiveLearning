import numpy as np
import matplotlib.pyplot as plt


def plot_active_mcmc_diagnostics(
    model,
    burnin: int = 0,
    max_points: int | None = None,
):
    """
    Plot diagnostics for an Active / Adaptive MCMC run.

    Parameters
    ----------
    model
        Instance of ActiveMCMCModel or AdaptiveActiveMCMCModel.
    burnin
        Number of initial steps to discard in diagnostics.
    max_points
        Optional cap on number of points plotted (for long runs).
    """

    used_hf = np.asarray(model.used_hf, dtype=float)

    if burnin > 0:
        used_hf = used_hf[burnin:]

    if max_points is not None:
        used_hf = used_hf[:max_points]

    steps = np.arange(1, len(used_hf) + 1)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    # ------------------------------------------------------------------
    # 1. HF usage fraction
    # ------------------------------------------------------------------
    hf_fraction = np.cumsum(used_hf) / steps
    axes[0].plot(steps, hf_fraction)
    axes[0].set_ylabel("HF call fraction")
    axes[0].set_title("High-fidelity usage")

    # ------------------------------------------------------------------
    # 2. Subchain length evolution (adaptive only)
    # ------------------------------------------------------------------
    if hasattr(model, "subchain_history") and len(model.subchain_history) > 0:
        axes[1].plot(model.subchain_history)
        axes[1].set_ylabel("Subchain length")
        axes[1].set_title("Adaptive subchain length")
    else:
        axes[1].text(0.5, 0.5, "No adaptive subchain data", ha="center", va="center")
        axes[1].set_axis_off()

    # ------------------------------------------------------------------
    # 3. Surrogate error history
    # ------------------------------------------------------------------
    if hasattr(model, "_errors") and len(model._errors) > 0:
        axes[2].plot(model._errors)
        axes[2].axhline(
            getattr(model, "target_error", 0.0),
            linestyle="--",
        )
        axes[2].set_ylabel("Surrogate error")
        axes[2].set_title("HF–surrogate error")
    else:
        axes[2].text(0.5, 0.5, "No error history available", ha="center", va="center")
        axes[2].set_axis_off()

    axes[-1].set_xlabel("Iteration")

    plt.tight_layout()
    plt.show()
