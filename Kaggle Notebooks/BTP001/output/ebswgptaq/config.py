"""
Typed configuration objects for the EB-SW-GPTAQ proxy study.

Every notebook loads exactly one YAML file per component so that no experiment
parameter is hard-coded in a script (Milestone 0 requirement, v2 §7).
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml

# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Which model, in which precision, on which device."""

    model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    revision: Optional[str] = None          # pin a commit hash for reproducibility
    dtype: str = "float16"                  # "float16" | "bfloat16" | "float32"
    fp32_lm_head: bool = True               # T4 has no native BF16 -> keep head in FP32
    device: str = "cuda"
    attn_implementation: str = "sdpa"       # "sdpa" | "eager"
    trust_remote_code: bool = False
    max_seq_len: int = 2048
    exclude_name_patterns: Tuple[str, ...] = ("lm_head",)   # never quantized


# --------------------------------------------------------------------------------------
# Calibration corpus
# --------------------------------------------------------------------------------------


@dataclass
class CalibConfig:
    """Mixed calibration corpus: self-generated reasoning traces + general anchors."""

    n_reasoning: int = 20                   # pilot size; v2 §5.1 allows 16-32
    n_general: int = 7
    n_holdout: int = 8                      # disjoint held-out prompts for the proxy gate
    max_seq_len: int = 2048
    seed: int = 0
    min_trace_tokens: int = 64              # drop degenerate/truncated traces

    prompt_dataset: str = "openai/gsm8k"
    prompt_config: str = "main"
    prompt_split: str = "train"
    prompt_field: str = "question"

    general_dataset: str = "wikitext"
    general_config: str = "wikitext-2-raw-v1"
    general_split: str = "train"
    general_field: str = "text"

    gen_max_new_tokens: int = 1280
    gen_do_sample: bool = False             # deterministic primary corpus (v2 §5.1)
    gen_temperature: float = 0.6
    gen_top_p: float = 0.95

    ngram_leak_n: int = 13                  # contamination audit granularity


# --------------------------------------------------------------------------------------
# Quantization
# --------------------------------------------------------------------------------------


@dataclass
class QuantConfig:
    """One quantization run. Everything a reader needs to reproduce it."""

    bits: int = 4
    group_size: int = 128
    sym: bool = False                       # False => asymmetric grid with zero-point
    damp_percent: float = 0.01
    act_order: bool = False                 # Hessian-diagonal channel permutation
    true_sequential: bool = False           # exact greedy OBC instead of lazy blocks
    blocksize: int = 128
    working_dtype: str = "float32"          # dtype of the solver's working copy of W

    # Hessian / cross-covariance accumulation
    hessian_dtype: str = "float32"          # "float32" | "float64"
    min_tokens_per_layer: int = 256         # skip layers with too few calibration columns

    # asymmetric reconstruction (GPTAQ)
    asym: bool = True                       # False => weighted symmetric GPTQ
    drift_correction: bool = True           # True => capture X~ from the FP16 model
    #                                        False => X~ := X (plain weighted GPTQ)

    # grid fitting
    mse_clip: bool = False
    n_clip_grid: int = 20

    # module selection
    exclude_name_patterns: Tuple[str, ...] = ("lm_head",)
    min_in_features: int = 16

    # diagnostics
    diag_max_dim: int = 2560                # full eigendecomposition below this size
    diag_topk: int = 64                     # randomised sketch size above it
    diag_every_n: int = 2                   # only measure 1 layer in N (n=8960 is costly)


# --------------------------------------------------------------------------------------
# Token weighting (the signal under test)
# --------------------------------------------------------------------------------------


@dataclass
class WeightConfig:
    """
    Map per-position sensitivity scores to calibration weights.

    signal:
        "none"    -> all weights exactly 1.0 (uniform GPTQ/GPTAQ baseline)
        "jsd"     -> same-prefix JSD between FP16 and a quantized probe
        "entropy" -> FP16 output entropy (open-loop secondary signal)
        "kl"      -> same-prefix forward KL (heavy-tailed negative control)
        "random"  -> random weights with the same distributional shape
    """

    signal: str = "jsd"
    probe: str = "rtn"                      # which probe produced the JSD scores
    clip_quantile: float = 0.99             # upper winsorization, 0.0 disables
    standardize: bool = True                # z-score after clipping
    normalize: str = "global"               # "global" | "per_sequence"

    # Either a fixed inverse temperature beta, or a target ESS floor (bisection).
    beta: Optional[float] = None
    ess_floor: Optional[float] = 0.5
    beta_max: float = 50.0

    sink_mask: bool = True                  # neutralise BOS/attention-sink positions
    sink_positions: Tuple[int, ...] = (0,)
    shuffle_within_sequence: bool = False   # specificity control (breaks alignment)
    seed: int = 0


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------


@dataclass
class EvalConfig:
    n_holdout: int = 64                     # v2 §5.4 / v1 §3.4 use 64 dev problems
    max_seq_len: int = 2048
    seed: int = 0
    chunk_positions: int = 64               # positions per JSD/KL chunk (vocab memory)
    topk: int = 0                           # 0 => exact full-vocabulary JSD
    high_entropy_frac: float = 0.10         # top-10% fork positions
    bootstrap: int = 10000


# --------------------------------------------------------------------------------------
# YAML plumbing
# --------------------------------------------------------------------------------------


def _coerce(value: Any) -> Any:
    """Turn YAML lists into tuples where the dataclass declares a Tuple field."""
    if isinstance(value, list):
        return tuple(value)
    return value


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def from_dict(cls, data: Dict[str, Any]):
    """Build a dataclass from a dict, ignoring unknown keys, coercing lists to tuples."""
    if data is None:
        return cls()
    fields = {f.name: f for f in dataclasses.fields(cls)}
    kwargs = {}
    for key, value in data.items():
        if key in fields:
            kwargs[key] = _coerce(value)
    return cls(**kwargs)


def load_config(path: str, cls):
    """Load `path` into dataclass `cls`. Returns cls() if path is None."""
    if path is None:
        return cls()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            f"Expected one of the files shipped in the repo's configs/ directory."
        )
    return from_dict(cls, load_yaml(path))


def config_to_dict(cfg) -> Dict[str, Any]:
    return dataclasses.asdict(cfg)
