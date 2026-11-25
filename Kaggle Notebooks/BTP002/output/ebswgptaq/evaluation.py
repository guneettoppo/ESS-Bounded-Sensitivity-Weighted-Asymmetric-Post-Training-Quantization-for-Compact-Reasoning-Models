"""
Held-out evaluation: teacher-forced fidelity, strata, and (later) generation metrics.

The proxy gate is decided by teacher-forced fidelity on identical prefixes, because
that isolates the quantiser's effect on the output distribution from decoding noise.
Generation-based accuracy is a *downstream* pilot and only runs after the gate
(v2 §5.4, §6).

Memory note
-----------
`teacher_forced_fidelity` expects both the FP16 reference and the quantized model to
be resident. For a 1.5B model in FP16 that is ~6 GB, which fits a T4. For 4B+ use the
`swap` mode (not implemented by default) or evaluate in two passes while caching only
the positions you need.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .batching import CalibBatch, target_positions
from .jsd import divergence_per_position, log_softmax_fp32


# ======================================================================================
# Teacher-forced fidelity
# ======================================================================================


@torch.no_grad()
def teacher_forced_fidelity(
    ref_wrapper,
    q_wrapper,
    batches: Sequence[CalibBatch],
    chunk_positions: int = 64,
    topk: int = 0,
    keep_per_position: bool = True,
) -> Dict[str, object]:
    """
    Per-sequence KL(FP16 || quantized), JSD, and teacher-forced NLL for both models.

    Returns
    -------
    dict with
        per_sequence : list of {mean_kl, mean_jsd, mean_entropy, nll_ref, nll_q, n}
        arrays       : concatenated per-position scores (optional, for strata)
        aggregate    : mean/summary values used in the results table
    """
    per_sequence: List[Dict[str, float]] = []
    all_kl: List[torch.Tensor] = []
    all_jsd: List[torch.Tensor] = []
    all_ent: List[torch.Tensor] = []
    all_seq_id: List[torch.Tensor] = []
    total_nll_ref = 0.0
    total_nll_q = 0.0
    total_n = 0

    for si, batch in enumerate(batches):
        pos = target_positions(batch.valid_mask)
        if pos.numel() == 0:
            continue
        pos = pos.to(batch.input_ids.device)

        logits_ref, _ = ref_wrapper.forward_logits(batch.input_ids, batch.attention_mask)
        logits_q, _ = q_wrapper.forward_logits(batch.input_ids, batch.attention_mask)
        lr = logits_ref[0].index_select(0, pos)          # [P, V]
        lq = logits_q[0].index_select(0, pos)

        kl = divergence_per_position(lr, lq, "kl", chunk=chunk_positions)
        jsd = divergence_per_position(lr, lq, "jsd", chunk=chunk_positions, topk=topk)
        ent = divergence_per_position(lr, lr, "entropy", chunk=chunk_positions)

        labels = batch.input_ids[0].index_select(0, pos + 1)
        lp_ref = log_softmax_fp32(lr).gather(1, labels.unsqueeze(1)).squeeze(1)
        lp_q = log_softmax_fp32(lq).gather(1, labels.unsqueeze(1)).squeeze(1)

        per_sequence.append({
            "seq_index": si,
            "n_positions": int(pos.numel()),
            "mean_kl": float(kl.mean()),
            "mean_jsd": float(jsd.mean()),
            "mean_entropy": float(ent.mean()),
            "nll_ref": float(-lp_ref.mean()),
            "nll_q": float(-lp_q.mean()),
            "ppl_ref": float(torch.exp(-lp_ref.mean())),
            "ppl_q": float(torch.exp(-lp_q.mean())),
        })
        total_nll_ref += float(-lp_ref.sum())
        total_nll_q += float(-lp_q.sum())
        total_n += int(pos.numel())

        if keep_per_position:
            all_kl.append(kl.cpu())
            all_jsd.append(jsd.cpu())
            all_ent.append(ent.cpu())
            all_seq_id.append(torch.full((pos.numel(),), si, dtype=torch.long))

    if not per_sequence:
        return {"per_sequence": [], "aggregate": {"n_sequences": 0}}

    n = len(per_sequence)
    aggregate = {
        "n_sequences": n,
        "n_positions": total_n,
        "mean_kl": sum(r["mean_kl"] for r in per_sequence) / n,
        "mean_jsd": sum(r["mean_jsd"] for r in per_sequence) / n,
        "mean_entropy": sum(r["mean_entropy"] for r in per_sequence) / n,
        "nll_ref": total_nll_ref / max(total_n, 1),
        "nll_q": total_nll_q / max(total_n, 1),
        "ppl_ref": math.exp(min(total_nll_ref / max(total_n, 1), 60)),
        "ppl_q": math.exp(min(total_nll_q / max(total_n, 1), 60)),
        "nll_delta": (total_nll_q - total_nll_ref) / max(total_n, 1),
    }

    out: Dict[str, object] = {"per_sequence": per_sequence, "aggregate": aggregate}
    if keep_per_position:
        out["arrays"] = {
            "kl": torch.cat(all_kl),
            "jsd": torch.cat(all_jsd),
            "entropy": torch.cat(all_ent),
            "seq_id": torch.cat(all_seq_id),
        }
    return out


def strata_report(
    arrays: Dict[str, torch.Tensor],
    thresholds: Dict[str, float],
    prefix: str = "",
) -> Dict[str, object]:
    """
    Break fidelity down into high-entropy fork positions, commitment positions and
    the rest, using thresholds frozen on *calibration* scores (no leakage).

    `thresholds` must come from the calibration split, e.g.
        {"entropy_p90": 4.31, "jsd_p90": 0.021}
    """
    ent = arrays["entropy"]
    jsd = arrays["jsd"]
    kl = arrays["kl"]

    report: Dict[str, object] = {}
    if "entropy_p90" in thresholds:
        hi = ent >= thresholds["entropy_p90"]
        report["high_entropy"] = {
            "n": int(hi.sum()),
            "mean_kl": float(kl[hi].mean()) if hi.any() else None,
            "mean_jsd": float(jsd[hi].mean()) if hi.any() else None,
        }
        report["low_entropy"] = {
            "n": int((~hi).sum()),
            "mean_kl": float(kl[~hi].mean()) if (~hi).any() else None,
            "mean_jsd": float(jsd[~hi].mean()) if (~hi).any() else None,
        }
        if hi.any() and (~hi).any():
            report["kl_ratio_high_over_low"] = (
                float(kl[hi].mean()) / max(float(kl[~hi].mean()), 1e-12)
            )
    if "jsd_p90" in thresholds:
        hi = jsd >= thresholds["jsd_p90"]
        report["high_jsd"] = {"n": int(hi.sum()), "mean_kl": float(kl[hi].mean()) if hi.any() else None}
        report["low_jsd"] = {"n": int((~hi).sum()), "mean_kl": float(kl[~hi].mean()) if (~hi).any() else None}
    report["overall"] = {
        "n": int(ent.numel()),
        "mean_kl": float(kl.mean()),
        "mean_jsd": float(jsd.mean()),
        "mean_entropy": float(ent.mean()),
    }
    if prefix:
        report = {f"{prefix}{k}": v for k, v in report.items()}
    return report


# ======================================================================================
# Generation-based pilot metrics (notebook 09)
# ======================================================================================

_BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def extract_boxed(text: str) -> Optional[str]:
    matches = _BOXED.findall(text or "")
    return matches[-1].strip() if matches else None


def extract_final_answer(text: str) -> Optional[str]:
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed
    nums = _NUMBER.findall(text or "")
    return nums[-1] if nums else None


def normalise_answer(ans: Optional[str]) -> str:
    if ans is None:
        return ""
    a = ans.strip().lower()
    a = a.replace("$", "").replace("\\%", "%").replace(",", "")
    a = re.sub(r"\s+", "", a)
    a = a.rstrip(".")
    return a


def answers_match(pred: Optional[str], gold: Optional[str]) -> bool:
    """Cheap numeric/symbolic match. Upgrade to math-verify for the full phase."""
    p, g = normalise_answer(pred), normalise_answer(gold)
    if not p or not g:
        return False
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except Exception:
        return p.replace(" ", "") == g.replace(" ", "")


def answer_appeared_early(trace: str, gold: Optional[str], tail_tokens: int = 120) -> bool:
    """
    Overthinking detector: the correct answer appears in the trace, but the model
    keeps going and does not commit to it (v1 §3.4 failure taxonomy, category a).
    """
    g = normalise_answer(gold if gold is not None else extract_final_answer(trace))
    if not g:
        return False
    text = normalise_answer(trace)
    first = text.find(g)
    if first < 0:
        return False
    # is the answer present well before the end of the trace?
    return first < max(0, len(text) - tail_tokens)


@torch.no_grad()
def generate_and_score(
    wrapper,
    records: Sequence[Dict],
    max_new_tokens: int = 4096,
    do_sample: bool = False,
    temperature: float = 0.6,
    top_p: float = 0.95,
    template: str = "plain",
    limit: Optional[int] = None,
) -> Dict[str, object]:
    """
    Generate one answer per problem and compute Acc / mean CoT length / TPCA.

    TPCA (tokens per correct answer) is the deployment-relevant metric: it charges a
    method for both the accuracy it loses and the extra tokens it spends.
    """
    from .calibration_data import render_prompt

    tokenizer = wrapper.tokenizer
    rows: List[Dict] = []
    items = list(records)[:limit] if limit else list(records)

    for item in items:
        rendered = item.get("rendered_prompt") or render_prompt(item["prompt"], template)
        enc = tokenizer(rendered, return_tensors="pt").to(wrapper.device)
        plen = int(enc["input_ids"].shape[1])
        out = wrapper.generate(
            enc["input_ids"], attention_mask=enc["attention_mask"],
            max_new_tokens=int(max_new_tokens), do_sample=bool(do_sample),
            temperature=float(temperature), top_p=float(top_p),
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        new_ids = out[0][plen:].tolist()
        text = tokenizer.decode(new_ids, skip_special_tokens=True)

        gold = item.get("answer")
        pred = extract_final_answer(text)
        correct = answers_match(pred, gold)
        rows.append({
            "prompt_id": item.get("prompt_id"),
            "n_tokens": len(new_ids),
            "hit_max_new_tokens": len(new_ids) >= int(max_new_tokens),
            "pred": pred,
            "gold": gold,
            "correct": bool(correct),
            "overthinking": bool(answer_appeared_early(text, gold)) and not correct,
            "trace_head": text[:400],
        })

    n = len(rows)
    n_correct = sum(1 for r in rows if r["correct"])
    total_tokens = sum(r["n_tokens"] for r in rows)
    return {
        "n": n,
        "accuracy": n_correct / max(n, 1),
        "n_correct": n_correct,
        "mean_cot_tokens": total_tokens / max(n, 1),
        "tpca": total_tokens / max(n_correct, 1),
        "truncation_rate": sum(1 for r in rows if r["hit_max_new_tokens"]) / max(n, 1),
        "overthinking_rate": sum(1 for r in rows if r["overthinking"]) / max(n, 1),
        "rows": rows,
        "per_item_correct": [bool(r["correct"]) for r in rows],
        "per_item_tokens": [int(r["n_tokens"]) for r in rows],
    }
