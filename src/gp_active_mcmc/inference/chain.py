from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gp_active_mcmc.utils.mcmc import (
    acceptance_rate_from_accepted,
    hf_call_fraction,
    mean_subchain_length,
    move_fraction_from_samples,
    posterior_rmse,
)


FloatArray = NDArray[np.floating]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int_]


def _as_2d_float(samples: ArrayLike) -> FloatArray:
    arr = np.asarray(samples, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"samples must be 2D (n_steps, n_dim). Got shape {arr.shape}.")
    return arr


def _as_1d_bool(x: ArrayLike, *, name: str, n: int) -> BoolArray:
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D. Got shape {arr.shape}.")
    if arr.shape[0] != n:
        raise ValueError(f"{name} length {arr.shape[0]} does not match n_steps {n}.")
    return arr.astype(bool, copy=False)


def _as_1d_int(x: ArrayLike, *, name: str, n: int) -> IntArray:
    arr = np.asarray(x, dtype=int).ravel()
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D. Got shape {arr.shape}.")
    if arr.shape[0] != n:
        raise ValueError(f"{name} length {arr.shape[0]} does not match n_steps {n}.")
    return arr


def _validate_burnin(burnin: int, *, n_steps: int) -> int:
    b = int(burnin)
    if b < 0:
        raise ValueError("burnin must be >= 0.")
    if b > n_steps:
        raise ValueError(f"burnin={b} is larger than chain length {n_steps}.")
    return b


def _validate_thin(thin: int) -> int:
    t = int(thin)
    if t <= 0:
        raise ValueError("thin must be a positive integer.")
    return t


@dataclass(frozen=True, slots=True)
class ChainExtras:
    """Optional aligned arrays recorded per MCMC step."""

    used_hf: BoolArray | None = None
    accepted: BoolArray | None = None
    subchain_length: IntArray | None = None

    def slice(self, sl: slice) -> "ChainExtras":
        def _s(v):
            return None if v is None else v[sl]

        return ChainExtras(
            used_hf=_s(self.used_hf),
            accepted=_s(self.accepted),
            subchain_length=_s(self.subchain_length),
        )


@dataclass(frozen=True, slots=True)
class MCMCChain:
    """Immutable chain container."""

    samples: FloatArray
    extras: ChainExtras = field(default_factory=ChainExtras)

    @classmethod
    def from_arrays(
        cls,
        *,
        samples: ArrayLike,
        used_hf: ArrayLike | None = None,
        accepted: ArrayLike | None = None,
        subchain_length: ArrayLike | None = None,
    ) -> "MCMCChain":
        s = _as_2d_float(samples)
        n = s.shape[0]
        extras = ChainExtras(
            used_hf=None if used_hf is None else _as_1d_bool(used_hf, name="used_hf", n=n),
            accepted=None if accepted is None else _as_1d_bool(accepted, name="accepted", n=n),
            subchain_length=None
            if subchain_length is None
            else _as_1d_int(subchain_length, name="subchain_length", n=n),
        )
        return cls(samples=s, extras=extras)

    @property
    def n_steps(self) -> int:
        return int(self.samples.shape[0])

    @property
    def n_dim(self) -> int:
        return int(self.samples.shape[1])

    def burnin(self, burnin: int = 0) -> "MCMCChain":
        b = _validate_burnin(burnin, n_steps=self.n_steps)
        sl = slice(b, None)
        return MCMCChain(samples=self.samples[sl], extras=self.extras.slice(sl))

    def thin(self, thin: int = 1) -> "MCMCChain":
        t = _validate_thin(thin)
        sl = slice(None, None, t)
        return MCMCChain(samples=self.samples[sl], extras=self.extras.slice(sl))

    def summary(self, *, theta_true: ArrayLike | None = None, burnin: int = 0) -> dict[str, Any]:
        out: dict[str, Any] = {"n_steps": self.n_steps, "n_dim": self.n_dim}

        if self.extras.accepted is not None:
            out["acceptance_rate"] = acceptance_rate_from_accepted(self.extras.accepted)
        else:
            out["move_fraction"] = move_fraction_from_samples(self.samples)

        if self.extras.used_hf is not None:
            out["hf_call_fraction"] = hf_call_fraction(self.extras.used_hf)
            out["n_hf_calls"] = int(np.sum(self.extras.used_hf))

        if self.extras.subchain_length is not None:
            out["mean_subchain_length"] = mean_subchain_length(self.extras.subchain_length)

        if theta_true is not None:
            out["posterior_rmse"] = posterior_rmse(self.samples, theta_true, burnin=burnin)

        return out


@dataclass(frozen=True, slots=True)
class SamplingResult:
    """Sampler output (chain + metadata)."""

    chain: MCMCChain
    metadata: dict[str, Any] = field(default_factory=dict)
