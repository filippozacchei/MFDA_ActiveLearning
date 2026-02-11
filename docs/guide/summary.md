# Concepts: inference and surrogates

This page provides a conceptual (scientific) overview of the main building blocks in `gp_active_mcmc`. It is intended to complement the API reference with the “why” and “how the pieces fit together”.

---

## Notation

Let

- $\theta \in \mathbb{R}^d$ be the parameter vector,
- $y \in \mathbb{R}^{n}$ be the model output in _observation space_ (e.g., a time series sampled on a grid),
- $y_{\text{obs}} \in \mathbb{R}^{n}$ be observed data,
- $C_{\text{obs}} \in \mathbb{R}^{n \times n}$ be the observation-noise covariance,
- $f_{\text{HF}}(\theta)$ be the high-fidelity forward model,
- $f_{\text{LF}}(\theta)$ be the low-fidelity surrogate (learned approximation).

The library targets workflows where $f_{\text{HF}}$ is accurate but expensive, while $f_{\text{LF}}$ is cheap but imperfect.

---

## Inference: active learning inside MCMC

### The core abstraction: the active model

`ActiveMCMCModel` is the heart of the inference layer. It couples a low-fidelity surrogate and a high-fidelity forward model and exposes two callables designed to be plugged into `tinyDA` posteriors:

- `coarse(theta)`: evaluate LF first; optionally trigger HF if LF is deemed unreliable,
- `fine(theta)`: always evaluate HF and update LF using the new HF information.

See: [ActiveMCMCModel][gp_active_mcmc.inference.model.ActiveMCMCModel].

#### Coarse evaluation and HF triggering

In the default implementation, the coarse evaluation returns either:

- an LF prediction with uncertainty as a [CoarseOutput][gp_active_mcmc.inference.coarse_output.CoarseOutput], or
- a raw HF output $y_{\text{HF}}$ when a trigger condition activates.

A typical trigger is based on the surrogate’s predictive variance:

$$
\text{trigger HF if} \quad \frac{1}{n}\sum_{i=1}^{n} v_i(\theta) > \gamma^2,
$$

where $v(\theta)$ is the marginal predictive variance returned by the surrogate and $\gamma$ is a user parameter (see `gamma_threshold`).

This is intentionally cheap: it compresses uncertainty into a single scalar that is easy to monitor during sampling.

#### Fine evaluation and online surrogate updates

A fine evaluation computes $y_{\text{HF}} = f_{\text{HF}}(\theta)$ and updates the surrogate with the new pair $(\theta, y_{\text{HF}})$. This “learn while sampling” mechanism is the library’s active-learning component.

---

## Choosing the inference mode (posterior selection)

In `gp_active_mcmc`, the **posterior argument is algorithmic**: it determines how `tinyDA` orchestrates coarse vs fine evaluations.

### Mode A — single posterior (MCMC-guided active learning)

You pass a single posterior that uses the coarse model:

- `posterior = Posterior(prior, loglike, model.coarse)`
- run: [sample_active_chain][gp_active_mcmc.inference.sampling.sample_active_chain]

Interpretation:

- The chain is driven by LF evaluations.
- HF calls happen only when `ActiveMCMCModel.coarse` triggers them internally.

When to use:

- When you want the simplest workflow and do not need a formal delayed-acceptance mechanism.

### Mode B — two posteriors (DA-MCMC guided active learning)

You pass a list of two posteriors `[coarse, fine]`:

- coarse posterior uses `model.coarse`,
- fine posterior uses `model.fine`,
- run: [sample_active_chain][gp_active_mcmc.inference.sampling.sample_active_chain]

Interpretation:

- This corresponds to delayed-acceptance MCMC (DA-MCMC): proposals are first screened with the cheap level and then corrected by the expensive level at a frequency controlled by `subsampling_rate`.

When to use:

- When HF must participate in the acceptance/rejection mechanism in a principled delayed-acceptance scheme.

### Mode C — adaptive DA-MCMC (recommended)

You use DA-MCMC (two posteriors) **and** attach an adaptive subchain policy to the active model:

- construct [AdaptiveSubchain][gp_active_mcmc.inference.adaptive_subchain.AdaptiveSubchain],
- pass it to `ActiveMCMCModel(..., adaptive=...)`,
- use two posteriors `[coarse, fine]`,
- run: [sample_adaptive_active_chain][gp_active_mcmc.inference.sampling.sample_adaptive_active_chain]

Key constraint:

- **DA-MCMC is mandatory** for adaptive subchains (two posteriors required).

Why chunking is needed:

- `tinyDA` takes a fixed `subsampling_rate` per `sample(...)` call. If the subchain length changes online, we must re-enter `tinyDA` in chunks and update the subsampling rate between calls via [ChunkedMCMCConfig][gp_active_mcmc.inference.sampling.ChunkedMCMCConfig].

---

## Likelihood: accounting for surrogate uncertainty

When LF predictions come with predictive variance, it is often desirable to reflect that uncertainty in the likelihood.

`ActiveGPLogLike` is a Gaussian log-likelihood that supports **variance inflation** when the prediction carries a `.variance` attribute (typically [CoarseOutput][gp_active_mcmc.inference.coarse_output.CoarseOutput]):

$$
C_{\text{total}}(\theta) = C_{\text{obs}} + \operatorname{diag}(v(\theta)) ; (+, C_{\text{bias}}).
$$

This yields a likelihood that penalises uncertain surrogate predictions less strongly than confident predictions, reducing the risk of overconfident LF guidance.

See: [ActiveGPLogLike][gp_active_mcmc.inference.likelihood.ActiveGPLogLike].

---

## Adaptive subchains: controlling HF correction frequency

The adaptive subchain policy monitors LF–HF discrepancy during fine calls and updates a “subchain length” (coarse steps between fine corrections).

In the default implementation, the discrepancy is an RMSE in observation space:

$$
\mathrm{err}(\theta) = \sqrt{\frac{1}{n}\sum_{i=1}^{n}\left(\mu_{\text{LF},i}(\theta) - y_{\text{HF},i}(\theta)\right)^2},
$$

and the policy adjusts the subchain length every `update_every` HF evaluations:

- if `err > target_error`: shorten subchain (more frequent HF),
- else: lengthen subchain (less frequent HF).

See:

- [AdaptiveSubchain][gp_active_mcmc.inference.adaptive_subchain.AdaptiveSubchain]
- [AdaptiveSubchainState][gp_active_mcmc.inference.adaptive_subchain.AdaptiveSubchainState]
- [AdaptiveSubchainControl][gp_active_mcmc.inference.adaptive_subchain.AdaptiveSubchainControl]

---

## Results and diagnostics (what you can measure)

Sampling entrypoints return a [SamplingResult][gp_active_mcmc.inference.chain.SamplingResult] containing:

- an [MCMCChain][gp_active_mcmc.inference.chain.MCMCChain] with the sample matrix,
- optional aligned extras [ChainExtras][gp_active_mcmc.inference.chain.ChainExtras] (HF usage flags, acceptance flags, subchain history).

These extras enable diagnostics such as:

- HF call fraction (cost proxy),
- subchain length history (adaptive behaviour),
- chain visualisations (trace/pair plots).

See: [gp_active_mcmc.diagnostics][gp_active_mcmc.diagnostics].

---

# Surrogates: POD, GP, and POD–GP

The surrogate layer provides a standard reduced-order modelling pipeline:

1. compress snapshots with POD,
1. learn the map $\theta \mapsto$ POD coefficients with a GP,
1. reconstruct predictions in observation space.

The main user-facing surrogate is [PODGPSurrogate][gp_active_mcmc.surrogates.podgp.PODGPSurrogate].

---

## POD: reduced-order representation

Given snapshot trajectories/fields assembled as a matrix

$$
Y \in \mathbb{R}^{N \times n},
$$

where rows correspond to parameter samples and columns correspond to observation components, POD computes an orthonormal basis $\Phi \in \mathbb{R}^{n \times r}$ (with $r \ll n$) such that

$$
y(\theta) \approx \bar{y} + \Phi a(\theta),
$$

where $\bar{y}$ is the mean snapshot and $a(\theta) \in \mathbb{R}^r$ are POD coefficients.

In this library:

- `POD.fit(Y)` estimates $\bar{y}$ and $\Phi$,
- `POD.transform(Y)` returns coefficients $A$,
- `POD.inverse_transform(A)` reconstructs in observation space.

See: [POD][gp_active_mcmc.surrogates.pod.POD].

---

## GP regression for POD coefficients

After POD, the learning problem becomes:

$$
\theta \mapsto a(\theta) \in \mathbb{R}^r.
$$

The library implements:

- [SingleOutputGP][gp_active_mcmc.surrogates.gp.SingleOutputGP] for scalar outputs,
- [MultiOutputGP][gp_active_mcmc.surrogates.gp.MultiOutputGP] as independent GPs per output dimension (one GP per coefficient).

This design trades modelling simplicity for robustness:

- each coefficient is treated independently,
- predictive means and marginal variances are returned per coefficient.

---

## POD–GP coupling and uncertainty propagation

[PODGPSurrogate][gp_active_mcmc.surrogates.podgp.PODGPSurrogate] combines POD and MultiOutputGP:

- GP predicts coefficient mean and variance:
  $$
  \mu_a(\theta), ; v_a(\theta) \in \mathbb{R}^r,
  $$
- mean reconstruction:
  $$
  \mu_y(\theta) = \bar{y} + \Phi \mu_a(\theta),
  $$
- variance propagation (diagonal / marginal approximation):
  $$
  v_y(\theta) \approx \sum_{j=1}^{r} \Phi_{\cdot j}^2 , v_{a,j}(\theta),
  $$
  i.e., coefficient uncertainties are mapped to pointwise output variance assuming coefficients are independent.

This is exactly what the active inference layer needs:

- a predictive mean in observation space,
- an aligned marginal predictive variance for uncertainty-aware likelihoods.

See: [PODGPSurrogate][gp_active_mcmc.surrogates.podgp.PODGPSurrogate].

---

## Practical guidance: choosing POD rank and budgets

- POD rank $r$ controls the bias–variance trade-off:
  - too small: high truncation error (biased surrogate),
  - too large: coefficients become harder for the GP to learn reliably.
- A common workflow is:
  - inspect POD energy curves (retained variance),
  - validate on a held-out set (RMSE + coverage),
  - use active inference to expand training data in regions visited by the posterior.

See diagnostics helpers:

- [plot_pod_energy][gp_active_mcmc.diagnostics.pod.plot_pod_energy]
- [plot_prediction_at_theta][gp_active_mcmc.diagnostics.surrogate.plot_prediction_at_theta]
- [plot_error_vs_uncertainty][gp_active_mcmc.diagnostics.surrogate.plot_error_vs_uncertainty]

---
