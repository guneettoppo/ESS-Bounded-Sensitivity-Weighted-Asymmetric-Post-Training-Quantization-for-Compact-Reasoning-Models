"""
Calibration batches with an explicit, testable position contract.

The single most common silent bug in this literature is a one-position misalignment
between logits, input ids and activation columns (v2 §9.2). This module fixes one
canonical definition and every other module consumes it:

    A CalibBatch has exactly one sequence (B = 1) of length T.
    `valid_mask[t]`  is True iff token t is a real (non-padding) token.
    Activation column t of every linear module corresponds to token t.
    Logit row t predicts token t+1 (causal LM convention).

Therefore:
    calibration columns  = positions where valid_mask is True
    scoreable positions  = positions where valid_mask[t] and valid_mask[t+1]
                           (a next-token target exists)

Keeping B = 1 is intentional: it makes the (sequence, position) -> column mapping
trivial to verify and costs nothing at 2048 tokens on a T4.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch


@dataclass
class CalibBatch:
    """One calibration sequence and everything aligned to it."""

    input_ids: torch.Tensor                 # [1, T] int64
    attention_mask: torch.Tensor            # [1, T] int64
    valid_mask: torch.Tensor                # [T] bool
    weights: Optional[torch.Tensor] = None  # [T] float32, None => uniform
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return int(self.input_ids.shape[1])

    @property
    def n_valid(self) -> int:
        return int(self.valid_mask.sum())

    def to(self, device) -> "CalibBatch":
        return CalibBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            valid_mask=self.valid_mask,
            weights=self.weights,
            meta=self.meta,
        )

    def scoreable_positions(self) -> torch.Tensor:
        """Indices t where both t and t+1 are real tokens (a target exists)."""
        valid = self.valid_mask
        if valid.numel() < 2:
            return torch.zeros(0, dtype=torch.long)
        pair = valid[:-1] & valid[1:]
        return torch.nonzero(pair, as_tuple=False).flatten()


def target_positions(valid_mask: torch.Tensor) -> torch.Tensor:
    """Position indices whose *next* token is a real token."""
    if valid_mask.numel() < 2:
        return torch.zeros(0, dtype=torch.long)
    return torch.nonzero(valid_mask[:-1] & valid_mask[1:], as_tuple=False).flatten()


def batches_from_records(
    records: Sequence[Dict],
    max_seq_len: int,
    weights_per_record: Optional[Sequence[torch.Tensor]] = None,
    device: Optional[torch.device] = None,
    drop_incomplete_target: bool = True,
) -> List[CalibBatch]:
    """
    Build CalibBatches from JSONL records that already carry `token_ids`.

    `records[i]["token_ids"]` must be the exact ids used for the forward pass.
    Passing ids (rather than re-tokenising) is what makes notebook 01 -> 06
    bit-reproducible.
    """
    batches: List[CalibBatch] = []
    for idx, rec in enumerate(records):
        ids = rec.get("token_ids")
        if ids is None:
            raise KeyError(
                f"Record {idx} has no 'token_ids'. Notebook 01 stores them; "
                "re-tokenising here would silently change the calibration set."
            )
        ids = list(ids)[:max_seq_len]
        if drop_incomplete_target and len(ids) < 2:
            continue
        t = len(ids)
        input_ids = torch.tensor([ids], dtype=torch.long)
        mask = torch.ones(t, dtype=torch.long)
        valid = torch.ones(t, dtype=torch.bool)

        w = None
        if weights_per_record is not None:
            w = torch.as_tensor(weights_per_record[idx], dtype=torch.float32).flatten()
            if w.numel() != t:
                raise ValueError(
                    f"Weight vector for record {idx} has length {w.numel()} "
                    f"but the sequence has {t} tokens."
                )

        batch = CalibBatch(
            input_ids=input_ids,
            attention_mask=mask,
            valid_mask=valid,
            weights=w,
            meta={
                "record_index": idx,
                "prompt_id": rec.get("prompt_id"),
                "source": rec.get("source"),
                "length": t,
            },
        )
        if device is not None:
            batch = batch.to(device)
        batches.append(batch)
    return batches


def total_columns(batches: Sequence[CalibBatch]) -> int:
    return sum(b.n_valid for b in batches)


def describe_batches(batches: Sequence[CalibBatch]) -> Dict[str, object]:
    lengths = [b.length for b in batches]
    n_valid = [b.n_valid for b in batches]
    has_w = all(b.weights is not None for b in batches)
    return {
        "n_sequences": len(batches),
        "total_tokens": int(sum(n_valid)),
        "length_min": min(lengths) if lengths else 0,
        "length_max": max(lengths) if lengths else 0,
        "length_mean": round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
        "weights_present": has_w,
        "sources": sorted({b.meta.get("source") for b in batches if b.meta.get("source")}),
    }


def concat_weights(batches: Sequence[CalibBatch]) -> Optional[torch.Tensor]:
    """Stack the per-batch weight vectors into one global pool-ordered vector."""
    if not all(b.weights is not None for b in batches):
        return None
    return torch.cat([b.weights.flatten() for b in batches], dim=0)


def calibration_sufficiency(batches: Sequence[CalibBatch], max_in_features: int) -> Dict[str, object]:
    """
    Is there enough calibration data for the Hessian to be well conditioned?

    `H` is (d_in x d_in) built from N token columns. If N < d_in the Gram matrix is
    structurally rank-deficient, and the ridge damping is the only thing making the
    solve possible -- at which point the weighting is not doing what you think.
    As a rule of thumb N >= 2*d_in is comfortable, N >= d_in is marginal.

    Called loudly by notebooks 02 and 06 because short generation traces are the most
    likely way this project silently produces garbage.
    """
    n = total_columns(batches)
    d = int(max_in_features)
    ratio = n / max(d, 1)
    if ratio >= 2.0:
        verdict, action = "ok", "proceed"
    elif ratio >= 1.0:
        verdict, action = "marginal", ("increase n_reasoning or gen_max_new_tokens "
                                       "before trusting the result")
    else:
        verdict, action = "rank-deficient", ("STOP: fewer tokens than input channels. "
                                             "Increase n_reasoning / gen_max_new_tokens "
                                             "or the Hessian will be dominated by damping")
    return {
        "total_tokens": int(n),
        "max_in_features": d,
        "tokens_per_channel": round(ratio, 2),
        "verdict": verdict,
        "action": action,
    }
