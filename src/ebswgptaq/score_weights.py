"""
Turn per-position sensitivity scores into calibration weights.

Design rules taken from the pivot document
------------------------------------------
1. Weighting is a weighted least-squares objective, so weights are normalised to
   **mean 1** over the pool. That keeps the ridge damping on a comparable scale
   across variants (v2 §9.3).
2. The primary mapping is `w_t = exp(beta * z_t)` with a *modest fixed beta*, not a
   tuned one. A target-ESS bisection is available but is a secondary control, not
   the headline claim (v2 §4.2).
3. Controls must preserve the *distribution* of weights and break only the
   *alignment*: `shuffle_within_sequence` permutes weights inside each sequence,
   `random` draws weights from the same empirical distribution.
4. Sink positions (BOS / attention sink) are neutralised before standardisation,
   because their activation norm dominates the Gram matrix (v1 §2.3).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .config import WeightConfig


def kish_ess(w: torch.Tensor) -> float:
    """Kish effective sample size (sum w)^2 / sum w^2."""
    wf = w.detach().to(torch.float32).flatten().clamp_min(0.0)
    s1 = wf.sum()
    s2 = wf.pow(2).sum()
    if s2 <= 0:
        return 0.0
    return float(s1 * s1 / s2)


def ess_fraction(w: torch.Tensor) -> float:
    n = w.numel()
    return kish_ess(w) / max(n, 1)


def concentration_stats(w: torch.Tensor) -> Dict[str, float]:
    """Everything the results table needs about the weight distribution."""
    wf = w.detach().to(torch.float32).flatten().clamp_min(0.0)
    n = wf.numel()
    if n == 0:
        return {}
    total = float(wf.sum())
    order = torch.sort(wf, descending=True).values
    top1 = max(1, int(round(0.01 * n)))
    top10 = max(1, int(round(0.10 * n)))
    return {
        "n": n,
        "mean": float(wf.mean()),
        "max": float(wf.max()),
        "median": float(wf.median()),
        "min": float(wf.min()),
        "p99": float(torch.quantile(wf, torch.tensor(0.99, dtype=torch.float32))),
        "ess": kish_ess(wf),
        "ess_fraction": ess_fraction(wf),
        "mass_top1pct": float(order[:top1].sum() / max(total, 1e-12)),
        "mass_top10pct": float(order[:top10].sum() / max(total, 1e-12)),
        "max_over_median": float(wf.max() / max(float(wf.median()), 1e-12)),
    }


def winsorize(z: torch.Tensor, quantile: float) -> torch.Tensor:
    if quantile is None or quantile <= 0.0 or quantile >= 1.0:
        return z
    hi = torch.quantile(z, torch.tensor(quantile, dtype=torch.float32))
    return torch.clamp(z, max=float(hi))


def standardize(z: torch.Tensor) -> torch.Tensor:
    mu = z.mean()
    sd = z.std(unbiased=False)
    if float(sd) < 1e-12:
        return torch.zeros_like(z)
    return (z - mu) / sd


def exp_weights(z: torch.Tensor, beta: float) -> torch.Tensor:
    """w = exp(beta*z) normalised to mean 1. Always positive, never zero."""
    w = torch.exp(float(beta) * z.double() if z.dtype == torch.float64 else float(beta) * z)
    m = w.mean()
    if float(m) <= 0:
        return torch.ones_like(z)
    return w / m


def solve_beta_for_ess(
    z: torch.Tensor,
    target_ess_fraction: float,
    beta_max: float = 50.0,
    iters: int = 30,
) -> Tuple[float, float]:
    """
    30-step bisection for the inverse temperature whose induced weights hit a target
    normalised ESS (v1 Theorem 1, retained here only as a secondary control).

    ESS_fraction(beta) is strictly decreasing for non-constant z, so bisection is
    valid. Returns (beta, achieved_ess_fraction).
    """
    if target_ess_fraction >= 1.0:
        return 0.0, 1.0
    if float(z.std(unbiased=False)) < 1e-12:
        return 0.0, 1.0

    zf = z.detach().to(torch.float64)
    n = zf.numel()
    m = int((zf == zf.max()).sum())
    floor = m / n
    if target_ess_fraction <= floor:
        return float(beta_max), floor

    lo, hi = 0.0, float(beta_max)

    def ess_at(beta: float) -> float:
        return ess_fraction(exp_weights(zf, beta))

    if ess_at(hi) > target_ess_fraction:
        return hi, ess_at(hi)

    beta = 0.5 * hi
    for _ in range(iters):
        beta = 0.5 * (lo + hi)
        if ess_at(beta) > target_ess_fraction:
            lo = beta
        else:
            hi = beta
    return beta, ess_at(beta)


def _apply_sink_mask(positions: torch.Tensor, valid: Optional[torch.Tensor],
                     cfg: WeightConfig) -> torch.Tensor:
    """Positions to neutralise: BOS/sink by default."""
    mask = torch.zeros_like(positions, dtype=torch.bool)
    if cfg.sink_mask:
        for p in cfg.sink_positions:
            mask |= positions == int(p)
    return mask


def build_weights(
    scores: torch.Tensor,
    positions: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
    cfg: Optional[WeightConfig] = None,
    sequence_id: Optional[torch.Tensor] = None,
) -> Dict[str, object]:
    """
    Scores -> mean-1 weights for one sequence (or one whole pool if sequence_id=None).

    `scores` may contain NaN/negative values for positions with no valid target;
    those are given the neutral weight 1.0.
    """
    cfg = cfg or WeightConfig()
    s = scores.detach().to(torch.float32).flatten().clone()
    pos = positions.flatten()

    finite = torch.isfinite(s)
    if valid is not None:
        finite &= valid.flatten().bool()
    if cfg.signal == "none":
        return {
            "weights": torch.ones_like(s),
            "beta": 0.0,
            "stats": concentration_stats(torch.ones_like(s)),
            "note": "uniform baseline",
        }

    # neutralise sink positions by masking them out of the statistics
    sink = _apply_sink_mask(pos, valid, cfg)
    stat_mask = finite & (~sink)

    if int(stat_mask.sum()) < 4:
        return {
            "weights": torch.ones_like(s),
            "beta": 0.0,
            "stats": concentration_stats(torch.ones_like(s)),
            "note": "too few valid scores; fell back to uniform",
        }

    ref = s[stat_mask]
    s = torch.where(finite, s, ref.median())
    s = winsorize(s, cfg.clip_quantile)
    z = standardize(s) if cfg.standardize else s

    if cfg.beta is not None:
        beta, achieved = float(cfg.beta), ess_fraction(exp_weights(z, cfg.beta))
    elif cfg.ess_floor is not None and cfg.ess_floor < 1.0:
        beta, achieved = solve_beta_for_ess(z, float(cfg.ess_floor), cfg.beta_max)
    else:
        beta, achieved = 0.0, 1.0

    w = exp_weights(z, beta)

    # neutral positions get exactly the mean weight (1.0)
    neutral = (~finite) | sink
    if neutral.any():
        w = torch.where(neutral, torch.ones_like(w), w)
        w = w / w.mean()

    if cfg.shuffle_within_sequence:
        g = torch.Generator().manual_seed(int(cfg.seed))
        idx = torch.randperm(w.numel(), generator=g)
        w = w[idx]

    return {
        "weights": w,
        "beta": beta,
        "achieved_ess_fraction": achieved,
        "stats": concentration_stats(w),
        "z_stats": {
            "mean": float(z.mean()),
            "std": float(z.std(unbiased=False)),
            "min": float(z.min()),
            "max": float(z.max()),
        },
    }


def random_weights_like(w: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """
    Negative control: same *distribution* of weights, no alignment to the signal.

    Draws a permutation of the empirical weights, which preserves the exact marginal
    distribution (and therefore the concentration stats) while destroying any
    correlation with position (v2 §5.2).
    """
    g = torch.Generator().manual_seed(int(seed))
    flat = w.detach().flatten().clone()
    return flat[torch.randperm(flat.numel(), generator=g)].reshape(w.shape)


def per_sequence_weights(
    scores_per_seq: Sequence[torch.Tensor],
    positions_per_seq: Sequence[torch.Tensor],
    cfg: WeightConfig,
    valid_per_seq: Optional[Sequence[torch.Tensor]] = None,
) -> List[torch.Tensor]:
    """Independent normalisation inside each sequence (the `normalize=per_sequence` mode)."""
    out = []
    for i, (s, p) in enumerate(zip(scores_per_seq, positions_per_seq)):
        v = valid_per_seq[i] if valid_per_seq is not None else None
        out.append(build_weights(s, p, v, cfg)["weights"])
    return out
