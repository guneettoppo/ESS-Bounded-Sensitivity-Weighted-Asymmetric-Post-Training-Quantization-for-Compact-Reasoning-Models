"""
Self-generated reasoning traces for the calibration corpus (Milestone 1).

Why self-generated
------------------
Generic web text is distributionally far from a long chain of thought, and
quantising on it leaves the high-entropy fork positions under-represented. Okoshi et
al. and Zheng et al. both use self-generated trajectories; the pivot adopts that
without claiming it as novel.

Determinism
-----------
The primary corpus uses greedy decoding so the corpus can be regenerated bit-exactly
from (prompt_ids, model revision, seed). Prompts, full traces, token ids and every
decoding parameter are stored per record, because an unreproducible calibration set
invalidates every downstream comparison.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import torch

from .config import CalibConfig


def _gen_kwargs(cfg: CalibConfig, max_new_tokens: Optional[int], tokenizer) -> Dict:
    kwargs = {
        "max_new_tokens": int(max_new_tokens or cfg.gen_max_new_tokens),
        "do_sample": bool(cfg.gen_do_sample),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if cfg.gen_do_sample:
        kwargs.update({"temperature": float(cfg.gen_temperature), "top_p": float(cfg.gen_top_p)})
    return kwargs


@torch.no_grad()
def generate_traces(
    wrapped,
    prompts: Sequence[Dict],
    cfg: CalibConfig,
    template: str = "plain",
    max_new_tokens: Optional[int] = None,
    verbose: bool = True,
) -> List[Dict]:
    """
    Generate one trace per prompt and return JSONL-ready records.

    Each record carries the exact token_ids of prompt+trace, so the calibration
    batches never re-tokenise.
    """
    from .calibration_data import render_prompt

    tokenizer = wrapped.tokenizer
    records: List[Dict] = []
    t0 = time.time()

    for i, item in enumerate(prompts):
        question = item["prompt"]
        rendered = render_prompt(question, template)
        enc = tokenizer(rendered, return_tensors="pt").to(wrapped.device)
        prompt_len = int(enc["input_ids"].shape[1])

        out = wrapped.generate(enc["input_ids"], attention_mask=enc["attention_mask"],
                               **_gen_kwargs(cfg, max_new_tokens, tokenizer))
        full_ids = out[0].tolist()
        new_ids = full_ids[prompt_len:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True)

        hit_limit = len(new_ids) >= int(max_new_tokens or cfg.gen_max_new_tokens)
        records.append({
            "prompt_id": item["prompt_id"],
            "source": item.get("source", "unknown"),
            "source_split": item.get("source_split", "unknown"),
            "prompt": question,
            "rendered_prompt": rendered,
            "template": template,
            "trace": text,
            "token_ids": full_ids,
            "prompt_len": prompt_len,
            "n_new_tokens": len(new_ids),
            "hit_max_new_tokens": bool(hit_limit),
            "do_sample": bool(cfg.gen_do_sample),
            "temperature": float(cfg.gen_temperature) if cfg.gen_do_sample else None,
            "top_p": float(cfg.gen_top_p) if cfg.gen_do_sample else None,
            "seed": int(cfg.seed),
            "model_id": wrapped.cfg.model_id,
            "model_revision": wrapped.cfg.revision,
            "tokenizer_name": getattr(tokenizer, "name_or_path", None),
        })
        if verbose and (i + 1) % 4 == 0:
            print(f"  {i+1}/{len(prompts)} traces  "
                  f"({records[-1]['n_new_tokens']} new tokens, "
                  f"{time.time()-t0:.0f}s elapsed)")
    return records


@torch.no_grad()
def tokenize_anchors(wrapped, texts: Sequence[str], cfg: CalibConfig) -> List[Dict]:
    """General-language anchors: tokenised, truncated, no generation."""
    tokenizer = wrapped.tokenizer
    records: List[Dict] = []
    for i, text in enumerate(texts):
        ids = tokenizer(text, truncation=True, max_length=cfg.max_seq_len,
                        add_special_tokens=True)["input_ids"]
        if len(ids) < 8:
            continue
        records.append({
            "prompt_id": f"anchor:{i:03d}",
            "source": cfg.general_dataset,
            "source_split": cfg.general_split,
            "prompt": None,
            "rendered_prompt": None,
            "template": None,
            "trace": text,
            "token_ids": ids,
            "prompt_len": 0,
            "n_new_tokens": len(ids),
            "hit_max_new_tokens": False,
            "do_sample": False,
            "seed": int(cfg.seed),
            "model_id": wrapped.cfg.model_id,
            "model_revision": wrapped.cfg.revision,
        })
    return records


def filter_records(records: Sequence[Dict], min_tokens: int = 64,
                   max_tokens: Optional[int] = None) -> List[Dict]:
    """Drop degenerate (too short) traces; keep everything else for inspection."""
    out = []
    for r in records:
        n = len(r.get("token_ids") or [])
        if n < min_tokens:
            continue
        if max_tokens is not None and n > max_tokens:
            continue
        out.append(r)
    return out


def build_mixed_corpus(
    reasoning_records: Sequence[Dict],
    anchor_records: Sequence[Dict],
    cfg: CalibConfig,
    max_seq_len: Optional[int] = None,
) -> List[Dict]:
    """
    Assemble the mixed corpus and truncate every sequence to the calibration length.

    Truncation is from the LEFT for reasoning traces (keep the reasoning body, drop
    the tail of the problem statement only if it is enormous) and from the right for
    anchors. In practice most traces are shorter than the window, so this rarely
    matters -- but it is recorded per record either way.
    """
    max_len = int(max_seq_len or cfg.max_seq_len)
    corpus: List[Dict] = []

    for rec in list(reasoning_records) + list(anchor_records):
        ids = list(rec["token_ids"])
        original = len(ids)
        if original > max_len:
            if rec["prompt_len"] and rec["prompt_len"] < max_len // 2:
                # keep the prompt head and the most recent reasoning tokens
                head = ids[: rec["prompt_len"]]
                tail = ids[-(max_len - len(head)):]
                ids = head + tail
                mode = "head+tail"
            else:
                ids = ids[:max_len]
                mode = "right"
        else:
            mode = "none"
        out = dict(rec)
        out["token_ids"] = ids
        out["length_before_truncation"] = original
        out["truncation_mode"] = mode
        corpus.append(out)
    return corpus


def corpus_stats(corpus: Sequence[Dict]) -> Dict[str, object]:
    lens = [len(r["token_ids"]) for r in corpus]
    n_reason = sum(1 for r in corpus if r.get("prompt") is not None)
    return {
        "n_sequences": len(corpus),
        "n_reasoning": n_reason,
        "n_general": len(corpus) - n_reason,
        "reasoning_fraction": round(n_reason / max(len(corpus), 1), 3),
        "total_tokens": int(sum(lens)),
        "length_min": min(lens) if lens else 0,
        "length_max": max(lens) if lens else 0,
        "length_mean": round(sum(lens) / max(len(lens), 1), 1),
        "n_truncated": sum(1 for r in corpus if r.get("truncation_mode") not in (None, "none")),
        "n_hit_max_new_tokens": sum(1 for r in corpus if r.get("hit_max_new_tokens")),
    }
