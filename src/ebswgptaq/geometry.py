"""
Activation-geometry diagnostics (hypothesis H2 of the pivot).

Question: do high-JSD positions occupy a *different* activation subspace from
low-JSD positions, or are they merely the same directions with larger magnitude?

If the answer is "same subspace", then column weighting is just a rescaling of
directions that uniform calibration already sees, and the geometry-based story
collapses. That is exactly the kind of pre-registered falsification the pivot asks
for, so these diagnostics are reported for several layers rather than the best one.

All routines take [d, N] activation matrices (columns are tokens) and return plain
Python numbers so they can be JSON-serialised straight into the results file.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import torch

from .quant_core import spectral_summary


def weighted_covariance(X: torch.Tensor, w: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Covariance-style Gram matrix of X [d, N], optionally token-weighted."""
    Xf = X.detach().to(torch.float32)
    if w is not None:
        wv = torch.as_tensor(w, dtype=torch.float32, device=Xf.device).flatten().clamp_min(0)
        Xf = Xf * wv.sqrt().unsqueeze(0)
    n = max(Xf.shape[1], 1)
    return (Xf @ Xf.transpose(0, 1)) / n


def top_subspace(H: torch.Tensor, k: int = 16, seed: int = 0) -> torch.Tensor:
    """Top-k principal directions (orthonormal, [d, k]) of a PSD matrix."""
    d = H.shape[0]
    k = max(1, min(k, d))
    Hs = H.detach().to(torch.float32).cpu()
    if d <= 1024:
        evals, evecs = torch.linalg.eigh(Hs)
        return evecs[:, -k:].flip(1)
    torch.manual_seed(int(seed))
    _U, _S, V = torch.svd_lowrank(Hs, q=min(k + 8, d), niter=4)
    return V[:, -k:].flip(1)


def principal_angles(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Principal angles (radians, ascending) between the column spaces of A and B.

    Computed from the singular values of Q_A^T Q_B, which is numerically stable and
    avoids forming the projector matrices.
    """
    Qa = torch.linalg.qr(A.to(torch.float32).cpu(), mode="reduced").Q
    Qb = torch.linalg.qr(B.to(torch.float32).cpu(), mode="reduced").Q
    s = torch.linalg.svdvals(Qa.transpose(0, 1) @ Qb).clamp(-1.0, 1.0)
    return torch.acos(s).sort().values


def subspace_overlap(A: torch.Tensor, B: torch.Tensor) -> Dict[str, float]:
    """
    Mean cos^2 of principal angles: 1.0 = identical subspace, ~k/d = chance.

    `chance` is the expected overlap for two random k-dim subspaces in d dims,
    so the comparison to report is overlap vs chance, not overlap vs 1.
    """
    d = A.shape[0]
    k = min(A.shape[1], B.shape[1])
    angles = principal_angles(A, B)
    cos2 = torch.cos(angles).pow(2)
    return {
        "k": int(k),
        "d": int(d),
        "mean_cos2": float(cos2.mean()),
        "min_cos2": float(cos2.min()),
        "max_cos2": float(cos2.max()),
        "mean_angle_deg": float(math.degrees(float(angles.mean()))),
        "max_angle_deg": float(math.degrees(float(angles.max()))),
        "chance_cos2": float(min(1.0, k / max(d, 1))),
    }


def split_by_threshold(scores: torch.Tensor, threshold: float, valid: Optional[torch.Tensor] = None):
    """Boolean masks for high / low score groups, honouring a validity mask."""
    s = scores.detach().to(torch.float32).flatten()
    ok = torch.isfinite(s)
    if valid is not None:
        ok &= valid.flatten().bool()
    high = ok & (s >= threshold)
    low = ok & (s < threshold)
    return high, low


def matched_random_split(n: int, n_high: int, seed: int = 0) -> torch.Tensor:
    """Random subset of size n_high -- the matched-size control for H2."""
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g)
    mask = torch.zeros(n, dtype=torch.bool)
    mask[perm[:n_high]] = True
    return mask


def geometry_report(
    X: torch.Tensor,
    high_mask: torch.Tensor,
    low_mask: torch.Tensor,
    k: int = 16,
    max_dim: int = 2560,
    seed: int = 0,
) -> Dict[str, object]:
    """
    Full H2 diagnostic block for one (layer, module, probe) triple.

    Reports the spectrum of each group, the high/low subspace overlap, and the
    overlap of the high group with a matched random subset of the same size (the
    control that decides whether any difference is real).
    """
    Xf = X.detach().to(torch.float32)
    hi = Xf[:, high_mask.bool()]
    lo = Xf[:, low_mask.bool()]
    n_hi, n_lo = hi.shape[1], lo.shape[1]
    if n_hi < k + 4 or n_lo < k + 4:
        return {"error": f"too few columns (high={n_hi}, low={n_lo}) for k={k}"}

    H_hi = weighted_covariance(hi)
    H_lo = weighted_covariance(lo)
    A = top_subspace(H_hi, k, seed)
    B = top_subspace(H_lo, k, seed)

    rand_mask = matched_random_split(Xf.shape[1], n_hi, seed=seed)
    X_rand = Xf[:, rand_mask]
    A_rand = top_subspace(weighted_covariance(X_rand), k, seed=seed)

    return {
        "n_high": int(n_hi),
        "n_low": int(n_lo),
        "k": int(k),
        "high_vs_low": subspace_overlap(A, B),
        "high_vs_random": subspace_overlap(A, A_rand),
        "low_vs_random": subspace_overlap(B, A_rand),
        "spectrum_high": spectral_summary(H_hi, max_dim=max_dim),
        "spectrum_low": spectral_summary(H_lo, max_dim=max_dim),
        "norm_high_median": float(hi.norm(dim=0).median()),
        "norm_low_median": float(lo.norm(dim=0).median()),
    }


def aggregate_geometry(reports: List[Dict]) -> Dict[str, object]:
    """Aggregate across layers/modules; never cherry-pick a single favourable layer."""
    ok = [r for r in reports if "high_vs_low" in r]
    if not ok:
        return {"n_reports": 0}
    mean_ov = sum(r["high_vs_low"]["mean_cos2"] for r in ok) / len(ok)
    mean_ctrl = sum(r["high_vs_random"]["mean_cos2"] for r in ok) / len(ok)
    return {
        "n_reports": len(ok),
        "mean_overlap_high_vs_low": mean_ov,
        "mean_overlap_high_vs_random": mean_ctrl,
        "mean_gap_over_control": mean_ov - mean_ctrl,
        "frac_layers_below_control": sum(
            1 for r in ok if r["high_vs_low"]["mean_cos2"] < r["high_vs_random"]["mean_cos2"]
        ) / len(ok),
        "mean_angle_deg": sum(r["high_vs_low"]["mean_angle_deg"] for r in ok) / len(ok),
    }
