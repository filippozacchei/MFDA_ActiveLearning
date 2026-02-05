from dataclasses import dataclass
import tinyDA as tda
import numpy as np
from .utils.mcmc import (
    extract_samples,
    extract_acceptance_rate,
    extract_forward_frac,
    posterior_rmse,
)
from .active_mcmc_model import ActiveMCMCModel


@dataclass
class ActiveMCMCChain:
    samples: np.ndarray
    forward_calls: np.ndarray

    def burnin(self, burnin: int = 0) -> "ActiveMCMCChain":
        samples = self.samples[burnin:]
        forward_Calls = self.forward_calls[burnin:]
        return ActiveMCMCChain(samples=samples, forward_calls=forward_Calls)

    def thin(self, thin: int = 0) -> "ActiveMCMCChain":
        samples = self.samples[::thin]
        forward_Calls = self.forward_calls[::thin]
        return ActiveMCMCChain(samples=samples, forward_calls=forward_Calls)

    def info(self, theta_true: np.ndarray | None):
        accept_rate = extract_acceptance_rate(self.samples)
        forward_frac = extract_forward_frac(self.forward_calls)
        if isinstance(theta_true, np.ndarray):
            rmse = posterior_rmse(self.samples, theta_true)

        print(f"  Acceptance rate      : {accept_rate:.3f}")
        print(f"  Forward-call fraction: {forward_frac:.3f}")

        if isinstance(theta_true, np.ndarray):
            print(f"  RMSE vs theta_true   : {rmse:.5f}")

        print(f"  Total forward calls   : {len(self.forward_calls)}")
        print(f"  Total chain length    : {len(self.samples)}")


def sample_active_chain(
    model: ActiveMCMCModel,
    posterior: tda.Posterior | list[tda.Posterior],
    proposal: tda.Proposal,
    n_samples: int,
    n_chains: int,
    initial_parameter: np.ndarray,
    subsampling_rate: int,
    chain_key: str,
    force_sequential: bool = True,
    store_coarse_chain: bool = True,
    summary: bool = True,
    theta_true: np.ndarray | None = None,
):
    chain = tda.sample(
        posteriors=posterior,
        proposal=proposal,
        iterations=n_samples,
        n_chains=n_chains,
        force_sequential=force_sequential,
        initial_parameters=initial_parameter,
        store_coarse_chain=store_coarse_chain,
        subsampling_rate=subsampling_rate,
        adaptive_error_model=None,
    )

    samples = extract_samples(chain=chain, chain_key=chain_key)
    forward_calls = np.array(model.used_hf_flags)
    chain = ActiveMCMCChain(samples=samples, forward_calls=forward_calls)
    if summary:
        chain.info(theta_true=theta_true)
    return chain
