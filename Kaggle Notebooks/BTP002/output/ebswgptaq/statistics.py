"""
Statistics for the proxy gate: paired uncertainty and rank agreement.

All comparisons in this project are *paired*: the same held-out sequences, the same
tokens, the same decoding settings. So the unit of resampling is the sequence (or
problem), never the individual token -- token-level bootstrapping would produce
absurdly narrow intervals because tokens are strongly dependent within a sequence.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch


def paired_bootstrap_ci(
    deltas: Sequence[float],
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Percentile bootstrap CI for the mean of paired differences.

    `deltas[i]` = metric(method) - metric(baseline) for sequence/problem i.
    A CI that excludes 0 is the evidence bar for the specificity gate.
    """
    d = torch.as_tensor(list(deltas), dtype=torch.float32).flatten()
    n = d.numel()
    if n == 0:
        return {}
    g = torch.Generator().manual_seed(int(seed))
    idx = torch.randint(0, n, (n_resamples, n), generator=g)
    means = d[idx].mean(dim=1)
    lo = torch.quantile(means, alpha / 2)
    hi = torch.quantile(means, 1 - alpha / 2)
    return {
        "n": n,
        "mean": float(d.mean()),
        "std": float(d.std(unbiased=True)) if n > 1 else 0.0,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "alpha": alpha,
        "n_resamples": n_resamples,
        "excludes_zero": bool(float(lo) > 0 or float(hi) < 0),
        "p_direction": float((means > 0).float().mean()),
    }


def mcnemar_exact(baseline_correct: Sequence[bool], method_correct: Sequence[bool]
                  ) -> Dict[str, float]:
    """
    Exact McNemar test on paired per-item correctness.

    b = baseline right & method wrong, c = baseline wrong & method right.
    Two-sided exact binomial p-value on min(b, c) with n = b + c.
    """
    b = sum(1 for x, y in zip(baseline_correct, method_correct) if x and not y)
    c = sum(1 for x, y in zip(baseline_correct, method_correct) if (not x) and y)
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "n": 0, "p_value": 1.0, "statistic": 0.0}
    k = min(b, c)
    # exact two-sided binomial with p=0.5
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    p = min(1.0, 2.0 * tail)
    return {
        "b": b, "c": c, "n": n,
        "p_value": float(p),
        "statistic": float(abs(b - c)),
        "delta_accuracy": (c - b) / max(len(list(method_correct)), 1),
    }


def rank_correlation(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    """Spearman rho and Kendall tau between two per-position score vectors."""
    try:
        from scipy.stats import kendalltau, spearmanr
    except Exception as exc:  # pragma: no cover
        return {"error": f"scipy unavailable: {exc}"}

    a = torch.as_tensor(list(x), dtype=torch.float64).flatten().numpy()
    b = torch.as_tensor(list(y), dtype=torch.float64).flatten().numpy()
    if a.size < 3 or b.size < 3:
        return {"spearman": None, "kendall": None, "n": int(a.size)}
    rho, p_rho = spearmanr(a, b)
    tau, p_tau = kendalltau(a, b)
    return {
        "n": int(a.size),
        "spearman": float(rho),
        "spearman_p": float(p_rho),
        "kendall": float(tau),
        "kendall_p": float(p_tau),
    }


def topk_overlap(x: Sequence[float], y: Sequence[float], k: int) -> Dict[str, float]:
    """Jaccard overlap of the top-k position sets under two scores."""
    a = torch.as_tensor(list(x), dtype=torch.float32).flatten()
    b = torch.as_tensor(list(y), dtype=torch.float32).flatten()
    k = min(k, a.numel(), b.numel())
    if k == 0:
        return {"k": 0, "overlap": 0.0}
    sa = set(torch.topk(a, k).indices.tolist())
    sb = set(torch.topk(b, k).indices.tolist())
    return {
        "k": k,
        "overlap": len(sa & sb) / k,
        "jaccard": len(sa & sb) / max(len(sa | sb), 1),
        "chance": k / max(a.numel(), 1),
    }


def summarise_deltas(per_seq_method: Dict[str, Sequence[float]],
                     baseline_key: str,
                     n_resamples: int = 10000,
                     seed: int = 0) -> Dict[str, Dict[str, float]]:
    """Paired bootstrap of every variant against one baseline, keyed by variant."""
    base = list(per_seq_method[baseline_key])
    out: Dict[str, Dict[str, float]] = {}
    for key, values in per_seq_method.items():
        if key == baseline_key:
            continue
        values = list(values)
        if len(values) != len(base):
            out[key] = {"error": f"length mismatch {len(values)} vs {len(base)}"}
            continue
        deltas = [v - b for v, b in zip(values, base)]
        out[key] = paired_bootstrap_ci(deltas, n_resamples=n_resamples, seed=seed)
    return out
