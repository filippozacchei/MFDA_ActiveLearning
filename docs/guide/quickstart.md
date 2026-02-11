# Quickstart

This quickstart introduces the typical workflow implemented in `gp_active_mcmc`:

1. build a **low-fidelity (LF) surrogate**,
1. couple it with a **high-fidelity (HF) forward model** through an **active model**,
1. run **Active / DA-MCMC** (optionally with **adaptive subchains**) and inspect diagnostics.

The package is organised around a separation of concerns:

- **Surrogates** (`gp_active_mcmc.surrogates`): POD, Gaussian processes, and POD–GP coupling.
- **Inference** (`gp_active_mcmc.inference`): active model, likelihoods, proposals, adaptive subchain policy, and sampler entrypoints.
- **Diagnostics** (`gp_active_mcmc.diagnostics`): plotting helpers returning `(fig, ax)` and never calling `plt.show()`.
- **Utils** (`gp_active_mcmc.utils`): numerical helpers (no plotting).

---

## Installation

### Development install

From the repository root:

```bash
pip install -e .
```

Run the unit tests:

```bash
pytest -q
```

---

## Core ideas

### 1) The active model is the core abstraction

`ActiveMCMCModel` is the heart of the library. It couples:

- an **LF surrogate** (fast, uncertain), and
- an **HF model** (accurate, expensive),

and exposes two callables that can be plugged into `tinyDA.Posterior`:

- `model.coarse(theta)`: LF-first; may trigger HF if surrogate uncertainty is large.
- `model.fine(theta)`: always HF; also updates the surrogate.

When HF is evaluated, the LF surrogate is updated online. This is the mechanism that enables _active learning during inference_.

See: \[`ActiveMCMCModel`\][gp_active_mcmc.inference.model.ActiveMCMCModel].

### 2) Your choice of posterior(s) determines the inference scheme

In `gp_active_mcmc`, the _posterior argument_ is not just a technicality: it defines the algorithmic mode.

#### Single posterior → MCMC-guided active learning (single-level)

Use only the coarse model:

- `posterior = Posterior(prior, loglike, model.coarse)`
- run with a fixed-rate sampler such as:
  \[`sample_active_chain`\][gp_active_mcmc.inference.sampling.sample_active_chain]

In this mode, HF calls happen only when triggered inside `coarse(...)`.

#### Two posteriors → DA-MCMC guided active learning (delayed acceptance)

Use coarse and fine posteriors:

- `posterior = [Posterior(..., model.coarse), Posterior(..., model.fine)]`
- run with:
  \[`sample_active_chain`\][gp_active_mcmc.inference.sampling.sample_active_chain]

This corresponds to delayed-acceptance MCMC (DA-MCMC), where the fine level corrects the coarse approximation periodically (controlled by `subsampling_rate`).

### 3) Adaptive subchains (recommended) require DA-MCMC

`AdaptiveSubchain` adapts the **HF correction frequency** online by monitoring LF–HF discrepancy (e.g., RMSE between LF mean and HF output when HF is evaluated).

Important constraints:

- **DA-MCMC is mandatory**: you must pass **two posteriors** `[coarse, fine]`.
- Adaptation implies the subsampling rate can change over time, so sampling must be **chunked**:
  \[`sample_adaptive_active_chain`\][gp_active_mcmc.inference.sampling.sample_adaptive_active_chain]
  with \[`ChunkedMCMCConfig`\][gp_active_mcmc.inference.sampling.ChunkedMCMCConfig].

See: \[`AdaptiveSubchain`\][gp_active_mcmc.inference.adaptive_subchain.AdaptiveSubchain].

---

## Minimal end-to-end workflow (toy problem)

The toy problem is a lightweight baseline that does not require external PDE solvers.

A typical workflow is:

1. **Define a time grid** and the HF forward model `y = f_hf(theta)`.
1. **Generate an initial design** (parameter samples) and HF snapshots.
1. **Fit a POD–GP surrogate** (LF model).
1. **Wrap LF + HF in `ActiveMCMCModel`**.
1. **Choose the inference mode** by selecting one posterior (single-level) or two posteriors (DA-MCMC).
1. **Optionally enable adaptive subchains** (DA-MCMC only).
1. **Run sampling** and inspect diagnostics (HF usage, subchain history, trace plots).

The documentation includes notebook-style tutorials:

- **Forward toy**: build and validate a POD–GP surrogate.
  See: `Tutorials → Forward toy (POD-GP)`.
- **Backward toy**: Bayesian inversion with Active / DA-MCMC and adaptive subchains.
  See: `Tutorials → Backward toy (Active-MCMC)`.

---

## API pointers

If you want to navigate the implementation, these pages are the most useful entrypoints:

- \[`Inference`\]\[gp_active_mcmc.inference\]: active model, samplers, likelihood, proposals, result containers.
- \[`Surrogates`\]\[gp_active_mcmc.surrogates\]: POD, GP, POD–GP surrogate.
- \[`Diagnostics`\]\[gp_active_mcmc.diagnostics\]: plotting helpers returning `(fig, ax)`.
- \[`Utils`\]\[gp_active_mcmc.utils\]: numerical helpers.
