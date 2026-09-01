"""
Global high-fidelity reference sampler for UM-Bridge muq-beam.

Purpose
-------
The usual local Random-Walk Metropolis exact chain can get trapped on one part
of the strongly correlated muq-beam posterior ridge. This script builds a
GLOBAL independence-Metropolis proposal from already available samples
(active DA, old exact, optional exact-from-active-right), but the acceptance
probability uses ONLY the exact UM-Bridge log posterior. Therefore the Markov
chain targets the exact high-fidelity posterior, provided the proposal has
support over the relevant posterior region.

Run from the MFDA_ActiveLearning repository root, for example:

    python TESI/global_exact_reference_from_active.py \
      --active-npz results/muq_beam_fine_chain_test/samples_and_diagnostics.npz \
      --right-exact-npz results/exact_from_active_right/exact_from_active_right.npz \
      --n-steps 30000 \
      --n-centers 600 \
      --component-scale 0.03 \
      --outdir results/muq_beam_global_exact_reference

The proposal is a uniform mixture of Gaussians centered on subsampled points
from the available chains. The MH ratio includes q(current)/q(proposed), so the
proposal does NOT define the target; UM-Bridge does.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

# Import utilities from your Beam.py. This script is intended to live in TESI/.
from Beam import (  # type: ignore
    MUQ_DIM,
    UMBridgeLogPosterior,
    compare_samples,
    make_plots,
)

FloatArray = NDArray[np.float64]


def as_jsonable(obj: Any) -> Any:
    """Convert numpy objects to JSON-friendly Python objects."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): as_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [as_jsonable(v) for v in obj]
    return obj


def finite_rows(x: FloatArray) -> FloatArray:
    x = np.asarray(x, dtype=float).reshape(-1, MUQ_DIM)
    keep = np.all(np.isfinite(x), axis=1)
    return x[keep]


def tail(x: FloatArray, burn: int = 0, frac: float = 1.0) -> FloatArray:
    x = finite_rows(x)
    if x.shape[0] == 0:
        return x
    b = max(0, min(int(burn), x.shape[0] - 1))
    f = min(max(float(frac), 1e-6), 1.0)
    start_frac = int(math.floor((1.0 - f) * x.shape[0]))
    start = max(b, start_frac)
    return x[start:]


def evenly_spaced(x: FloatArray, k: int) -> FloatArray:
    x = finite_rows(x)
    if x.shape[0] == 0 or int(k) <= 0:
        return np.zeros((0, MUQ_DIM), dtype=float)
    if int(k) >= x.shape[0]:
        return x.copy()
    idx = np.unique(np.linspace(0, x.shape[0] - 1, num=int(k), dtype=int))
    return x[idx].copy()


def load_sources(
    active_npz: Path,
    right_exact_npz: Path | None,
    active_burn: int,
    active_tail_frac: float,
    old_exact_burn: int,
    old_exact_tail_frac: float,
    right_exact_burn: int,
    right_exact_tail_frac: float,
) -> dict[str, FloatArray]:
    if not active_npz.exists():
        raise FileNotFoundError(f"Cannot find active npz: {active_npz}")
    data = np.load(active_npz)
    sources: dict[str, FloatArray] = {}

    if "active_samples" in data:
        sources["active_da"] = tail(
            np.asarray(data["active_samples"], dtype=float),
            burn=active_burn,
            frac=active_tail_frac,
        )
    if "exact_samples" in data:
        sources["old_exact"] = tail(
            np.asarray(data["exact_samples"], dtype=float),
            burn=old_exact_burn,
            frac=old_exact_tail_frac,
        )

    if right_exact_npz is not None and right_exact_npz.exists():
        rd = np.load(right_exact_npz)
        if "exact_new_samples" in rd:
            sources["right_exact"] = tail(
                np.asarray(rd["exact_new_samples"], dtype=float),
                burn=right_exact_burn,
                frac=right_exact_tail_frac,
            )

    sources = {k: v for k, v in sources.items() if v.shape[0] > 0}
    if not sources:
        raise RuntimeError("No usable sample sources found.")
    return sources


def build_mixture_centers(sources: dict[str, FloatArray], n_centers: int) -> tuple[FloatArray, dict[str, int]]:
    names = list(sources.keys())
    per = max(1, int(math.ceil(int(n_centers) / len(names))))
    pieces: list[FloatArray] = []
    counts: dict[str, int] = {}
    for name in names:
        pts = evenly_spaced(sources[name], per)
        pieces.append(pts)
        counts[name] = int(pts.shape[0])
    centers = np.vstack(pieces)
    if centers.shape[0] > int(n_centers):
        # Keep points evenly spaced through the concatenated list.
        centers = evenly_spaced(centers, int(n_centers))
    return centers, counts


def regularized_cov(samples: FloatArray, scale: float, jitter: float = 1e-10) -> FloatArray:
    x = finite_rows(samples)
    if x.shape[0] <= MUQ_DIM + 2:
        base = np.eye(MUQ_DIM)
    else:
        base = np.cov(x.T)
    base = np.asarray(base, dtype=float)
    if base.shape != (MUQ_DIM, MUQ_DIM) or not np.all(np.isfinite(base)):
        base = np.eye(MUQ_DIM)
    base = 0.5 * (base + base.T)
    vals, vecs = np.linalg.eigh(base)
    floor = max(float(np.max(np.abs(vals))) * 1e-8, jitter)
    vals = np.maximum(vals, floor)
    base = (vecs * vals) @ vecs.T
    cov = float(scale) * base
    cov = 0.5 * (cov + cov.T) + jitter * np.eye(MUQ_DIM)
    return cov


@dataclass
class GaussianMixtureProposal:
    centers: FloatArray
    cov: FloatArray
    rng: np.random.Generator

    def __post_init__(self) -> None:
        self.centers = finite_rows(self.centers)
        self.cov = np.asarray(self.cov, dtype=float)
        self.cov = 0.5 * (self.cov + self.cov.T) + 1e-12 * np.eye(MUQ_DIM)
        self.inv_cov = np.linalg.inv(self.cov)
        sign, logdet = np.linalg.slogdet(self.cov)
        if sign <= 0:
            raise RuntimeError("Proposal covariance is not positive definite.")
        self.log_norm = -0.5 * (MUQ_DIM * np.log(2.0 * np.pi) + logdet)
        self.n_centers = int(self.centers.shape[0])
        if self.n_centers == 0:
            raise RuntimeError("Mixture proposal has zero centers.")

    def sample(self) -> FloatArray:
        idx = int(self.rng.integers(0, self.n_centers))
        return self.rng.multivariate_normal(self.centers[idx], self.cov).astype(float)

    def logpdf(self, theta: FloatArray) -> float:
        th = np.asarray(theta, dtype=float).reshape(MUQ_DIM)
        diff = self.centers - th.reshape(1, -1)
        quad = np.einsum("ij,jk,ik->i", diff, self.inv_cov, diff, optimize=True)
        return float(logsumexp(self.log_norm - 0.5 * quad) - np.log(self.n_centers))


@dataclass
class IMHResult:
    samples: FloatArray
    logp: FloatArray
    accepted: NDArray[np.bool_]
    proposal_logq: FloatArray
    acceptance_rate: float


def independence_mh_exact(
    logpost: UMBridgeLogPosterior,
    proposal: GaussianMixtureProposal,
    theta0: FloatArray,
    n_steps: int,
    report_every: int = 1000,
) -> IMHResult:
    n_steps = int(n_steps)
    samples = np.empty((n_steps, MUQ_DIM), dtype=float)
    logp_vals = np.empty(n_steps, dtype=float)
    logq_vals = np.empty(n_steps, dtype=float)
    accepted = np.zeros(n_steps, dtype=bool)

    cur = np.asarray(theta0, dtype=float).reshape(MUQ_DIM).copy()
    cur_lp = float(logpost.logp(cur))
    cur_lq = proposal.logpdf(cur)
    if not np.isfinite(cur_lp):
        raise RuntimeError(f"Initial theta has non-finite log posterior: {cur}, logp={cur_lp}")

    for i in range(n_steps):
        prop = proposal.sample()
        prop_lp = float(logpost.logp(prop))
        prop_lq = proposal.logpdf(prop)
        log_alpha = prop_lp - cur_lp + cur_lq - prop_lq
        if np.isfinite(prop_lp) and np.isfinite(prop_lq) and np.log(proposal.rng.uniform()) < log_alpha:
            cur = prop
            cur_lp = prop_lp
            cur_lq = prop_lq
            accepted[i] = True
        samples[i] = cur
        logp_vals[i] = cur_lp
        logq_vals[i] = cur_lq

        if report_every > 0 and (i + 1) % int(report_every) == 0:
            acc = float(np.mean(accepted[: i + 1]))
            print(f"[global-exact-imh] step {i + 1}/{n_steps}, acceptance={acc:.3f}, current={cur}, logp={cur_lp:.4g}")

    return IMHResult(samples, logp_vals, accepted, logq_vals, float(np.mean(accepted)))


def choose_initial_center(centers: FloatArray, mode: str) -> FloatArray:
    c = finite_rows(centers)
    if mode == "left":
        return c[int(np.argmin(c[:, 0]))].copy()
    if mode == "right":
        return c[int(np.argmax(c[:, 0]))].copy()
    # median along the ridge coordinate m1
    order = np.argsort(c[:, 0])
    return c[order[len(order) // 2]].copy()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Global exact reference for muq-beam using exact IMH with active-informed proposal")
    p.add_argument("--active-npz", default="results/muq_beam_fine_chain_test/samples_and_diagnostics.npz")
    p.add_argument("--right-exact-npz", default="results/exact_from_active_right/exact_from_active_right.npz")
    p.add_argument("--outdir", default="results/muq_beam_global_exact_reference")
    p.add_argument("--url", default="http://localhost:4243")
    p.add_argument("--model-name", default="posterior")
    p.add_argument("--seed", type=int, default=2025)

    p.add_argument("--n-steps", type=int, default=30000)
    p.add_argument("--report-every", type=int, default=1000)
    p.add_argument("--n-centers", type=int, default=600)
    p.add_argument("--component-scale", type=float, default=0.03,
                   help="Gaussian component covariance = component_scale * empirical covariance of centers")
    p.add_argument("--theta0", choices=["left", "median", "right"], default="median")

    p.add_argument("--active-burn", type=int, default=50)
    p.add_argument("--active-tail-frac", type=float, default=1.0)
    p.add_argument("--old-exact-burn", type=int, default=0)
    p.add_argument("--old-exact-tail-frac", type=float, default=1.0)
    p.add_argument("--right-exact-burn", type=int, default=5000)
    p.add_argument("--right-exact-tail-frac", type=float, default=1.0)

    p.add_argument("--burn-reference", type=int, default=2000)
    p.add_argument("--burn-active-compare", type=int, default=50)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    active_npz = Path(args.active_npz)
    right_exact_npz = Path(args.right_exact_npz) if args.right_exact_npz else None

    print("=== Global exact reference via independence MH ===")
    print(f"active_npz={active_npz}")
    print(f"right_exact_npz={right_exact_npz}")
    print(f"outdir={outdir}")

    sources = load_sources(
        active_npz=active_npz,
        right_exact_npz=right_exact_npz,
        active_burn=args.active_burn,
        active_tail_frac=args.active_tail_frac,
        old_exact_burn=args.old_exact_burn,
        old_exact_tail_frac=args.old_exact_tail_frac,
        right_exact_burn=args.right_exact_burn,
        right_exact_tail_frac=args.right_exact_tail_frac,
    )
    for name, arr in sources.items():
        print(f"[source] {name}: {arr.shape}, m1 range {arr[:,0].min():.4g}..{arr[:,0].max():.4g}")

    centers, center_counts = build_mixture_centers(sources, args.n_centers)
    cov = regularized_cov(centers, scale=args.component_scale)
    print(f"[proposal] centers={centers.shape[0]}, counts={center_counts}")
    print(f"[proposal] component covariance diag={np.diag(cov)}")

    proposal = GaussianMixtureProposal(centers=centers, cov=cov, rng=rng)
    theta0 = choose_initial_center(centers, mode=args.theta0)
    print(f"[initial] theta0={theta0}")

    logpost = UMBridgeLogPosterior(args.url, args.model_name)
    info = logpost.check()
    print(f"[umbridge] connected: {info}")
    theta0_lp = float(logpost.logp(theta0))
    print(f"[initial] logp(theta0)={theta0_lp:.6g}, logq(theta0)={proposal.logpdf(theta0):.6g}")

    calls_before = logpost.n_calls
    result = independence_mh_exact(
        logpost=logpost,
        proposal=proposal,
        theta0=theta0,
        n_steps=args.n_steps,
        report_every=args.report_every,
    )
    hf_calls = logpost.n_calls - calls_before
    print(f"[global-exact-imh] done, acceptance={result.acceptance_rate:.3f}, HF calls={hf_calls}")

    active_samples = sources.get("active_da", np.zeros((0, MUQ_DIM), dtype=float))
    old_exact_samples = sources.get("old_exact", np.zeros((0, MUQ_DIM), dtype=float))
    right_exact_samples = sources.get("right_exact", np.zeros((0, MUQ_DIM), dtype=float))

    comparisons: dict[str, Any] = {}
    if active_samples.shape[0] > 0:
        comparisons["global_reference_vs_active_da"] = compare_samples(
            exact_samples=result.samples,
            active_samples=active_samples,
            burn_exact=args.burn_reference,
            burn_active=args.burn_active_compare,
        )
        print(f"[comparison global ref vs active DA] {comparisons['global_reference_vs_active_da']}")
    if old_exact_samples.shape[0] > 0:
        comparisons["global_reference_vs_old_exact"] = compare_samples(
            exact_samples=result.samples,
            active_samples=old_exact_samples,
            burn_exact=args.burn_reference,
            burn_active=0,
        )
        print(f"[comparison global ref vs old exact] {comparisons['global_reference_vs_old_exact']}")
    if right_exact_samples.shape[0] > 0:
        comparisons["global_reference_vs_right_exact"] = compare_samples(
            exact_samples=result.samples,
            active_samples=right_exact_samples,
            burn_exact=args.burn_reference,
            burn_active=0,
        )
        print(f"[comparison global ref vs right exact] {comparisons['global_reference_vs_right_exact']}")

    np.savez_compressed(
        outdir / "global_exact_reference.npz",
        centers=centers,
        component_cov=cov,
        theta0=theta0,
        theta0_logp=np.array([theta0_lp]),
        samples=result.samples,
        logp=result.logp,
        proposal_logq=result.proposal_logq,
        accepted=result.accepted,
        active_samples=active_samples,
        old_exact_samples=old_exact_samples,
        right_exact_samples=right_exact_samples,
    )

    metrics = {
        "args": vars(args),
        "umbridge": info,
        "hf_calls": int(hf_calls),
        "acceptance_rate": float(result.acceptance_rate),
        "theta0": theta0,
        "theta0_logp": theta0_lp,
        "proposal": {
            "n_centers": int(centers.shape[0]),
            "center_counts": center_counts,
            "component_scale": float(args.component_scale),
            "component_cov": cov,
            "component_cov_diag": np.diag(cov),
        },
        "source_shapes": {k: list(v.shape) for k, v in sources.items()},
        "comparisons": comparisons,
        "reference_summary": {
            "mean": np.mean(result.samples[max(0, min(args.burn_reference, result.samples.shape[0]-1)):], axis=0),
            "std": np.std(result.samples[max(0, min(args.burn_reference, result.samples.shape[0]-1)):], axis=0, ddof=1),
            "logp_mean": float(np.mean(result.logp[max(0, min(args.burn_reference, result.logp.shape[0]-1)):])) ,
            "logp_max": float(np.max(result.logp)),
        },
    }
    (outdir / "metrics.json").write_text(json.dumps(as_jsonable(metrics), indent=2), encoding="utf-8")

    if not args.no_plots:
        # Plot active DA against the global exact reference.
        if active_samples.shape[0] > 0:
            avs = outdir / "active_vs_global_reference"
            avs.mkdir(exist_ok=True)
            make_plots(
                outdir=avs,
                exact_samples=result.samples,
                active_samples=active_samples,
                used_hf=None,
                val_true=None,
                val_mean=None,
                val_var=None,
                burn_exact=args.burn_reference,
                burn_active=args.burn_active_compare,
            )
            print(f"[plots] active vs global reference: {avs}")
        # Plot old exact against the global exact reference.
        if old_exact_samples.shape[0] > 0:
            ovs = outdir / "old_exact_vs_global_reference"
            ovs.mkdir(exist_ok=True)
            make_plots(
                outdir=ovs,
                exact_samples=result.samples,
                active_samples=old_exact_samples,
                used_hf=None,
                val_true=None,
                val_mean=None,
                val_var=None,
                burn_exact=args.burn_reference,
                burn_active=0,
            )
            print(f"[plots] old exact vs global reference: {ovs}")
        if right_exact_samples.shape[0] > 0:
            rvs = outdir / "right_exact_vs_global_reference"
            rvs.mkdir(exist_ok=True)
            make_plots(
                outdir=rvs,
                exact_samples=result.samples,
                active_samples=right_exact_samples,
                used_hf=None,
                val_true=None,
                val_mean=None,
                val_var=None,
                burn_exact=args.burn_reference,
                burn_active=0,
            )
            print(f"[plots] right exact vs global reference: {rvs}")

    print("=== done ===")
    print(f"Saved arrays:  {outdir / 'global_exact_reference.npz'}")
    print(f"Saved metrics: {outdir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
