"""
Prompt sources, the mixed calibration corpus, and the contamination audit.

Rules enforced here (v2 §5.1, §9.4)
----------------------------------
1. Calibration prompts come from **training** splits only, and evaluation prompt ids
   are disjoint by construction.
2. The corpus is stored as records carrying `token_ids`, so every later stage uses
   byte-identical tokenisation.
3. An n-gram overlap audit is run against the evaluation benchmarks and written to
   disk. A non-zero overlap is reported, never silently ignored.
"""
from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional, Sequence, Tuple

from .config import CalibConfig

PROMPT_TEMPLATES = {
    "plain": "{question}",
    "instruct": "Solve the following math problem step by step.\n\n{question}",
    "chat": ("<|im_start|>user\nSolve the following problem step by step. "
             "Put the final answer in \\boxed{{}}.\n\n{question}<|im_end|>\n<|im_start|>assistant\n"),
}


def stable_id(text: str, prefix: str = "") -> str:
    return prefix + hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def load_prompt_source(cfg: CalibConfig, limit: Optional[int] = None) -> List[Dict]:
    """
    Load training-split reasoning prompts.

    Uses `datasets` (available on Kaggle with internet enabled). For an offline run,
    substitute a local JSONL via `load_local_prompts`.
    """
    from datasets import load_dataset

    ds = load_dataset(cfg.prompt_dataset, cfg.prompt_config, split=cfg.prompt_split)
    rows: List[Dict] = []
    for i, ex in enumerate(ds):
        if limit is not None and i >= limit:
            break
        q = ex.get(cfg.prompt_field)
        if not q:
            continue
        q = str(q).strip()
        rows.append({
            "prompt_id": stable_id(q, prefix=cfg.prompt_split + ":"),
            "prompt": q,
            "source": cfg.prompt_dataset,
            "source_split": cfg.prompt_split,
        })
    return rows


def load_local_prompts(path: str, field: str = "question") -> List[Dict]:
    """Offline fallback: JSONL with one {field, ...} object per line."""
    import json

    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = str(obj.get(field, "")).strip()
            if q:
                rows.append({
                    "prompt_id": obj.get("prompt_id") or stable_id(q, "local:"),
                    "prompt": q,
                    "source": "local",
                    "source_split": "local",
                })
    return rows


def load_general_anchors(cfg: CalibConfig, limit: Optional[int] = None) -> List[str]:
    """C4 / WikiText-2 raw text used as general-language anchors (25% of the corpus)."""
    from datasets import load_dataset

    ds = load_dataset(cfg.general_dataset, cfg.general_config, split=cfg.general_split)
    texts: List[str] = []
    buf: List[str] = []
    for ex in ds:
        t = str(ex.get(cfg.general_field, "")).strip()
        if not t:
            continue
        buf.append(t)
        joined = "\n".join(buf)
        if len(joined) > 4000:
            texts.append(joined)
            buf = []
            if limit is not None and len(texts) >= limit:
                break
    if buf and (limit is None or len(texts) < limit):
        texts.append("\n".join(buf))
    return texts[:limit] if limit else texts


def sample_prompts(prompts: Sequence[Dict], n: int, seed: int = 0,
                   exclude_ids: Optional[set] = None) -> List[Dict]:
    """Deterministic, reproducible sample with optional id exclusion."""
    import random

    pool = [p for p in prompts if not exclude_ids or p["prompt_id"] not in exclude_ids]
    rng = random.Random(seed)
    if n >= len(pool):
        return list(pool)
    return rng.sample(pool, n)


def render_prompt(question: str, template: str = "plain") -> str:
    tpl = PROMPT_TEMPLATES.get(template, PROMPT_TEMPLATES["plain"])
    return tpl.format(question=question)


def normalise_for_overlap(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def ngrams(text: str, n: int) -> List[str]:
    words = normalise_for_overlap(text).split(" ")
    if len(words) < n:
        return []
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


def leak_audit(calib_texts: Sequence[str], eval_texts: Sequence[str], n: int = 13
               ) -> Dict[str, object]:
    """
    n-gram overlap between calibration prompts/traces and evaluation prompts.

    Returns the fraction of calibration n-grams found in the evaluation set, plus
    example collisions so they can be inspected rather than trusted.
    """
    calib_ng = set()
    for t in calib_texts:
        calib_ng.update(ngrams(t, n))
    eval_ng = set()
    for t in eval_texts:
        eval_ng.update(ngrams(t, n))

    if not calib_ng:
        return {"n": n, "calib_ngrams": 0, "eval_ngrams": len(eval_ng),
                "overlap_count": 0, "overlap_frac": 0.0, "examples": []}

    inter = calib_ng & eval_ng
    examples = sorted(inter)[:10]
    return {
        "n": n,
        "calib_ngrams": len(calib_ng),
        "eval_ngrams": len(eval_ng),
        "overlap_count": len(inter),
        "overlap_frac": len(inter) / len(calib_ng),
        "examples": examples,
        "clean": len(inter) == 0,
    }


def build_calibration_plan(cfg: CalibConfig, eval_prompt_ids: Optional[set] = None
                           ) -> Dict[str, object]:
    """
    Decide *which* prompts and anchors to use, before any generation happens.

    Returning a plan (instead of generating inside this module) keeps the expensive
    step in notebook 01 and makes the plan inspectable/printable.
    """
    prompts = load_prompt_source(cfg, limit=max(64, cfg.n_reasoning * 4))
    chosen = sample_prompts(prompts, cfg.n_reasoning, seed=cfg.seed,
                            exclude_ids=eval_prompt_ids)
    anchors = load_general_anchors(cfg, limit=cfg.n_general)
    return {
        "reasoning_prompts": chosen,
        "general_anchors": anchors,
        "prompt_source": cfg.prompt_dataset,
        "prompt_split": cfg.prompt_split,
        "seed": cfg.seed,
        "n_reasoning": len(chosen),
        "n_general": len(anchors),
        "prompt_ids": [p["prompt_id"] for p in chosen],
    }


CALIB_EVAL_PROTOCOL = {
    "mix": "75% self-generated reasoning traces, 25% general anchors (v1 §2.2)",
    "reasoning_source": "training splits only (GSM8K/MATH train)",
    "general_source": "WikiText-2 raw train",
    "decoding": "greedy (deterministic) for the primary corpus",
    "leak_audit": "13-gram overlap vs every evaluation set, reported in results",
}
