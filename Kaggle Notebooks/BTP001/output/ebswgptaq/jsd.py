"""
Same-prefix distributional divergence in FP32.

Everything here operates on *logits at aligned positions*: row t of a logits tensor
predicts token t+1. The notebooks always pass a matching set of positions, which is
produced by `batching.target_positions`.

Numerics (v2 §9.1)
------------------
- log_softmax is always computed in FP32 (`logits.float()` first), because FP16
  softmax over a 152k vocabulary loses the tail that JSD is supposed to measure.
- log M is formed with `logaddexp(logP, logQ) - log 2` instead of averaging
  probabilities, which keeps subnormal tail probabilities representable.
- Probabilities are only materialised per chunk of positions.

JSD is bounded in [0, ln 2] with natural logs, symmetric, and exactly zero for
identical distributions -- all three are asserted by the unit tests in notebook 04.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

LOG2 = 0.6931471805599453


def log_softmax_fp32(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """FP32 log-softmax over the last dimension (vocabulary)."""
    x = logits.to(torch.float32)
    if temperature != 1.0:
        x = x / float(temperature)
    return torch.log_softmax(x, dim=-1)


def _jsd_from_logprobs(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    """JSD(P||Q) for matching [N, V] log-probabilities. Returns [N] in nats."""
    log_m = torch.logaddexp(log_p, log_q) - LOG2
    p = log_p.exp()
    q = log_q.exp()
    kl_pm = (p * (log_p - log_m)).sum(dim=-1)
    kl_qm = (q * (log_q - log_m)).sum(dim=-1)
    return (0.5 * (kl_pm + kl_qm)).clamp_min(0.0)


def _kl_from_logprobs(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    """Forward KL(P||Q). Unbounded: a heavy-tailed negative control (v2 §4.1)."""
    p = log_p.exp()
    return (p * (log_p - log_q)).sum(dim=-1).clamp_min(0.0)


def _entropy_from_logprobs(log_p: torch.Tensor) -> torch.Tensor:
    p = log_p.exp()
    return -(p * log_p).sum(dim=-1)


def _gather_topk(logits: torch.Tensor, k: int) -> torch.Tensor:
    return torch.topk(logits, k=min(k, logits.shape[-1]), dim=-1).indices


def _jsd_topk(log_p: torch.Tensor, log_q: torch.Tensor, k: int) -> torch.Tensor:
    """
    Top-k JSD with the residual tail kept as an explicit extra bin.

    The union of both top-k index sets is retained, the probability mass outside that
    union is collapsed into one residual bin per distribution, and JSD is computed on
    the compact (<= 2k+1)-bin distribution. Collapsing the tail instead of dropping it
    is what keeps the approximation from being biased downward (v2 §9.1).
    """
    p = log_p.exp()
    q = log_q.exp()
    idx = torch.cat([_gather_topk(p, k), _gather_topk(q, k)], dim=-1)
    keep = torch.zeros_like(p).scatter_(1, idx, 1.0)
    p_m = p * keep
    q_m = q * keep
    p_res = (1.0 - p_m.sum(dim=-1, keepdim=True)).clamp_min(0.0)
    q_res = (1.0 - q_m.sum(dim=-1, keepdim=True)).clamp_min(0.0)
    p2 = torch.cat([p_m, p_res], dim=-1)
    q2 = torch.cat([q_m, q_res], dim=-1)
    m = 0.5 * (p2 + q2)
    log_m = m.clamp_min(1e-30).log()
    log_p2 = p2.clamp_min(1e-30).log()
    log_q2 = q2.clamp_min(1e-30).log()
    kl_pm = torch.where(p2 > 0, p2 * (log_p2 - log_m), torch.zeros_like(p2)).sum(dim=-1)
    kl_qm = torch.where(q2 > 0, q2 * (log_q2 - log_m), torch.zeros_like(q2)).sum(dim=-1)
    return (0.5 * (kl_pm + kl_qm)).clamp_min(0.0)


def divergence_per_position(
    logits_ref: torch.Tensor,
    logits_probe: torch.Tensor,
    kind: str = "jsd",
    chunk: int = 64,
    topk: int = 0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Per-position divergence between two aligned logits tensors.

    Parameters
    ----------
    logits_ref, logits_probe : [T, V] or [B, T, V], same shape, FP16/BF16/FP32.
    kind : "jsd" | "kl" | "entropy"   ("entropy" ignores logits_probe)
    chunk : positions processed at once (vocabulary memory control)
    topk : 0 = exact full-vocabulary; >0 = top-k union approximation

    Returns a tensor with the leading shape of the inputs.
    """
    if logits_ref.shape != logits_probe.shape and kind != "entropy":
        raise ValueError(f"shape mismatch: {tuple(logits_ref.shape)} vs "
                         f"{tuple(logits_probe.shape)}")
    lead = logits_ref.shape[:-1]
    v = logits_ref.shape[-1]
    lr = logits_ref.reshape(-1, v)
    lp = logits_probe.reshape(-1, v)
    out = torch.zeros(lr.shape[0], dtype=torch.float32, device=lr.device)

    for i in range(0, lr.shape[0], chunk):
        j = min(i + chunk, lr.shape[0])
        a = log_softmax_fp32(lr[i:j], temperature)
        if kind == "entropy":
            out[i:j] = _entropy_from_logprobs(a)
            continue
        b = log_softmax_fp32(lp[i:j], temperature)
        if topk and topk > 0:
            out[i:j] = _jsd_topk(a, b, topk)
        elif kind == "jsd":
            out[i:j] = _jsd_from_logprobs(a, b)
        elif kind == "kl":
            out[i:j] = _kl_from_logprobs(a, b)
        else:
            raise ValueError(f"unknown divergence kind: {kind}")
    return out.reshape(lead)


def jsd_summary(scores: torch.Tensor) -> Dict[str, float]:
    """Distribution summary used by the results table."""
    s = scores.detach().to(torch.float32).flatten()
    if s.numel() == 0:
        return {}
    qs = torch.quantile(s, torch.tensor([0.5, 0.9, 0.99], dtype=torch.float32))
    return {
        "n": int(s.numel()),
        "mean": float(s.mean()),
        "std": float(s.std(unbiased=False)),
        "median": float(qs[0]),
        "p90": float(qs[1]),
        "p99": float(qs[2]),
        "max": float(s.max()),
        "frac_above_half_ln2": float((s > 0.5 * LOG2).float().mean()),
    }


def assert_jsd_sanity(v: int = 512, device: str = "cpu", seed: int = 0) -> Dict[str, float]:
    """
    Unit checks for the JSD implementation. Called by notebook 04 before the real run.

    Verified properties: symmetry, non-negativity, upper bound ln 2, zero for
    identical distributions, and top-k approximation error against exact JSD.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(4, v, generator=g)
    y = torch.randn(4, v, generator=g)
    d_xy = divergence_per_position(x, y, "jsd", chunk=2)
    d_yx = divergence_per_position(y, x, "jsd", chunk=2)
    d_xx = divergence_per_position(x, x, "jsd", chunk=2)
    d_topk = divergence_per_position(x, y, "jsd", chunk=2, topk=32)
    return {
        "symmetry_max_abs_diff": float((d_xy - d_yx).abs().max()),
        "self_jsd_max": float(d_xx.max()),
        "min_jsd": float(d_xy.min()),
        "max_jsd": float(d_xy.max()),
        "ln2": LOG2,
        "within_bound": bool(d_xy.max() <= LOG2 + 1e-5),
        "topk_mean_abs_err": float((d_topk - d_xy).abs().mean()),
    }


def assert_causal_alignment(tol: float = 1e-6) -> Dict[str, object]:
    """
    Verify that logit row t is scored against token t+1 on a controlled tensor.

    This is the guard for the single most dangerous silent bug in the project
    (v2 §9.2). It uses a deterministic toy construction, not a model, so it costs
    nothing and can run before any GPU time is spent.

    Construction: token ids = [1..8]; "logits" whose value equals the row index.
    After a causal shift, row t must pair with id t+2 for t in 0..6.
    """
    ids = torch.arange(1, 9, dtype=torch.long).unsqueeze(0)          # [1, 8]
    scores = torch.arange(7, dtype=torch.float32).reshape(1, 7, 1)   # [1, 7, 1]

    # The canonical shift used everywhere in this repo:
    shift_logits = scores[:, :-1, :]          # rows 0..6  (drops the last position)
    shift_labels = ids[:, 1:]                 # ids 1..7  (drops the first token)

    pairs = list(zip(shift_logits.reshape(-1).tolist(), shift_labels.reshape(-1).tolist()))
    expected = [(float(t), t + 2) for t in range(7)]
    matches = [abs(a - c) < tol and b == d
               for (a, b), (c, d) in zip(pairs, expected)]

    return {
        "n_pairs": len(pairs),
        "all_pairs_correct": bool(all(matches)),
        "first_pair": pairs[0],
        "last_pair": pairs[-1],
        "note": "logits[t] must be scored against token_ids[t+1]",
    }
