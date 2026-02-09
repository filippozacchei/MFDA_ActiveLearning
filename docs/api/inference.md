# Inference

This section contains the inference-facing API: the active model, adaptive subchain policy,
samplers, likelihood, proposal, and result containers.

---

## Active model (core)

Use `ActiveMCMCModel` to couple a low-fidelity surrogate with a high-fidelity model.

::: gp_active_mcmc.inference.model.ActiveMCMCModel

---

## Adaptive subchain policy (optional)

Use `AdaptiveSubchain` to adapt the HF correction frequency online based on LF–HF discrepancy.

::: gp_active_mcmc.inference.adaptive_subchain.AdaptiveSubchain

### AdaptiveSubchainState

Stores the adaptive state (current subchain length, error history).

::: gp_active_mcmc.inference.adaptive_subchain.AdaptiveSubchainState

### AdaptiveSubchainControl

Stores control parameters for the adaptation policy.

::: gp_active_mcmc.inference.adaptive_subchain.AdaptiveSubchainControl

---

## Samplers

High-level entrypoints that run tinyDA and return a `SamplingResult`.

::: gp_active_mcmc.inference.sampling.sample_active_chain

::: gp_active_mcmc.inference.sampling.sample_adaptive_active_chain

::: gp_active_mcmc.inference.sampling.ChunkedMCMCConfig

---

## Likelihood

Gaussian log-likelihood that optionally inflates observation covariance using surrogate predictive variance.

::: gp_active_mcmc.inference.likelihood.ActiveGPLogLike

---

## Proposal

Adaptive proposal used with tinyDA.

::: gp_active_mcmc.inference.proposal.AdaptiveMetropolisShared

---

## Results

### ChainExtras

Per-step metadata aligned with samples (HF usage flags, acceptance flags, subchain length history).

::: gp_active_mcmc.inference.chain.ChainExtras

### MCMCChain

Immutable container for samples plus aligned per-step extras.

::: gp_active_mcmc.inference.chain.MCMCChain

### SamplingResult

Sampler output: `(chain, metadata)`.

::: gp_active_mcmc.inference.chain.SamplingResult
