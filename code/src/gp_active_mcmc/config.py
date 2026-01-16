from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class AlgorithmConfig:
    n_total: int = 1000
    gamma_var: float = 0.04         # threshold on predictive VARIANCE (not std)
    gamma_L_ratio: float = 2.5      # hyperparameter re-fit trigger
    n_retrain_max: int = 50
    step_scale: float = 0.1
    random_seed: int = 42
