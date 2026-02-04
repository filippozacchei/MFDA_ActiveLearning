import numpy as np
import matplotlib.pyplot as plt


def plot_active_mcmc_diagnostics(
    model,
    burnin: int = 0,
    max_points: int | None = None,
    scatter: bool = True,
):
    """
    Plot diagnostics for an Active / Adaptive MCMC run.

    Parameters
    ----------
    model : ActiveMCMC or AdaptiveActiveMCMC
        The MCMC model instance.
    burnin : int
        Number of initial steps to discard.
    max_points : int | None
        Maximum number of points to plot (for long runs).
    scatter : bool
        Whether to plot scatter of coarse vs fine evaluations.
    """

    # -------------------------------------------------
    # Extract HF usage flags
    # -------------------------------------------------
    used_hf = np.asarray(getattr(model, "used_hf_flags", []), dtype=float)
    if burnin > 0:
        used_hf = used_hf[burnin:]
    if max_points is not None:
        used_hf = used_hf[:max_points]

    total_steps = len(used_hf)
    steps = np.arange(1, total_steps + 1)

    # -------------------------------------------------
    # Scatter of coarse vs fine (optional)
    # -------------------------------------------------
    if scatter:
        coarse_vals = []
        fine_vals = []
        for i, flag in enumerate(used_hf):
            # assuming model stores last predictions or use placeholder
            val = getattr(model, "last_theta_values", None)
            if val is not None:
                if flag:
                    fine_vals.append(val[i])
                else:
                    coarse_vals.append(val[i])

        plt.figure(figsize=(6, 6))
        if coarse_vals:
            plt.scatter(range(len(coarse_vals)), coarse_vals, label="Coarse", alpha=0.6)
        if fine_vals:
            plt.scatter(range(len(fine_vals)), fine_vals, label="Fine", alpha=0.6)
        plt.xlabel("Step index")
        plt.ylabel("Prediction")
        plt.title("Coarse vs Fine predictions")
        plt.legend()
        plt.grid(True)
        plt.show()

    # -------------------------------------------------
    # HF cumulative fraction
    # -------------------------------------------------
    hf_fraction = np.cumsum(used_hf) / steps
    plt.figure(figsize=(8, 4))
    plt.plot(steps, hf_fraction, color="tab:blue")
    plt.xlabel("Iteration")
    plt.ylabel("Cumulative HF-call fraction")
    plt.title("High-fidelity usage over iterations")
    plt.grid(True)
    plt.show()

    # -------------------------------------------------
    # Adaptive subchain length
    # -------------------------------------------------
    subchain_lengths = getattr(model, "adaptive_state", None)
    if subchain_lengths is not None:
        subchain_history = getattr(model.adaptive_state, "subchain_history", [])
        if subchain_history:
            plt.figure(figsize=(8, 4))
            plt.plot(subchain_history, color="tab:orange")
            plt.xlabel("Coarse iterations")
            plt.ylabel("Subchain length")
            plt.title("Adaptive subchain length evolution")
            plt.grid(True)
            plt.show()

    # -------------------------------------------------
    # HF–surrogate error history
    # -------------------------------------------------
    hf_errors = getattr(model, "adaptive_state", None)
    if hf_errors is not None:
        hf_errors = getattr(model.adaptive_state, "hf_errors", [])
        if hf_errors:
            plt.figure(figsize=(8, 4))
            plt.plot(hf_errors, color="tab:green")
            target_error = getattr(model.adaptive_control, "target_error", 0.0)
            plt.axhline(target_error, linestyle="--", color="red", label="Target error")
            plt.xlabel("Fine iterations")
            plt.ylabel("Surrogate error")
            plt.title("HF–surrogate error")
            plt.legend()
            plt.grid(True)
            plt.show()
