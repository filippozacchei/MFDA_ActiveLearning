import numpy as np


def extract_samples(chain: dict, *, chain_key: str) -> np.ndarray:
    """Convert a tinyDA chain into a (n_samples, n_params) numpy array."""
    return np.array([link.parameters for link in chain[chain_key]])


def extract_acceptance_rate(samples: np.ndarray) -> float:
    accepted = np.any(np.diff(samples, axis=0) != 0, axis=1)
    return accepted.mean()


def extract_forward_frac(forward_calls: np.ndarray) -> float:
    return forward_calls.mean()


def posterior_rmse(
    samples: np.ndarray, theta_true: np.ndarray, burnin: int = 0
) -> float:
    return np.mean(np.sqrt(np.sum((samples[burnin:] - theta_true) ** 2, axis=1)))
