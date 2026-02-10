"""gp_active_mcmc: Active learning / multi-fidelity MCMC with GP surrogates.

Public API
----------
High-level entrypoints are exposed at package level so users can:

- build surrogates (`gp_active_mcmc.surrogates`)
- run sampling (`gp_active_mcmc.sample_active_chain`, `gp_active_mcmc.sample_adaptive_active_chain`)
- access results (`gp_active_mcmc.SamplingResult`, `gp_active_mcmc.MCMCChain`)
"""
