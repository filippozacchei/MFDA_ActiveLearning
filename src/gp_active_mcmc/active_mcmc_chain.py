from dataclasses import dataclass

import numpy as np
from .utils.mcmc import (
    extract_acceptance_rate,
    extract_forward_frac,
    extract_mean_subchain_length,
    posterior_rmse,
)


@dataclass
class ActiveMCMCChain:
    samples: np.ndarray
    forward_calls: np.ndarray

    def burnin(self, burnin: int = 0) -> "ActiveMCMCChain":
        samples = self.samples[burnin:]
        forward_calls = self.forward_calls[burnin:]
        return ActiveMCMCChain(samples=samples, forward_calls=forward_calls)

    def thin(self, thin: int = 0) -> "ActiveMCMCChain":
        samples = self.samples[::thin]
        forward_calls = self.forward_calls[::thin]
        return ActiveMCMCChain(samples=samples, forward_calls=forward_calls)

    def info(self, theta_true: np.ndarray | None):
        accept_rate = extract_acceptance_rate(self.samples)
        forward_frac = extract_forward_frac(self.forward_calls)
        if isinstance(theta_true, np.ndarray):
            rmse = posterior_rmse(self.samples, theta_true)

        print(f"  Acceptance rate      : {accept_rate:.3f}")
        print(f"  Forward-call fraction: {forward_frac:.3f}")

        if isinstance(theta_true, np.ndarray):
            print(f"  Posterior RMSE vs theta_true   : {rmse:.5f}")
        n_hf = int(np.sum(self.forward_calls))  # works for bool or 0/1
        print(f"  Total HF forward calls: {n_hf}")
        print(f"  Total chain length    : {len(self.samples)}")


class AdaptiveActiveMCMCChain(ActiveMCMCChain):
    subchain_lengths: list[int]

    def burnin(self, burnin: int = 0) -> "ActiveMCMCChain":
        subchain_lenghts = self.subchain_lengths[burnin:]
        super = super().burnin(burnin)
        return AdaptiveActiveMCMCChain(
            samples=super.samples,
            forward_calls=super.forward_calls,
            subchain_lenghts=subchain_lenghts,
        )

    def thin(self, thin: int = 0) -> "ActiveMCMCChain":
        subchain_lenghts = self.subchain_lengths[::int]
        super = super().burnin(thin)
        return AdaptiveActiveMCMCChain(
            samples=super.samples,
            forward_calls=super.forward_calls,
            subchain_lenghts=subchain_lenghts,
        )

    def info(self, theta_true: np.ndarray | None):
        mean_length = extract_mean_subchain_length(
            subchain_lengths=self.subchain_lengths
        )
        print(f"  Average Subchain Length: {mean_length:.3f}")
        super().info(theta_true=theta_true)
