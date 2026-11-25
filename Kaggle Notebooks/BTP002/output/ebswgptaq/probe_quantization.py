"""
Preliminary quantized probes used to define the JSD signal.

The probe answers one question: "where does quantization actually perturb the next
token distribution?" It must therefore be *cheap and fixed* -- not tuned, not
strong. The pivot's probe-swap analysis (v2 §5.3) exists precisely because the
signal might be an artefact of the probe, so the probe is a parameter, never a
constant.

Two probes:
    rtn  - round-to-nearest, no data, ~seconds. Zero Hessian, zero activation passes.
    gptq - uniform-weight GPTQ on the same calibration set (drift correction off for
           cost, documented as a deviation from full GPTAQ).

Both are produced by the same solver used for the variants, so the probe cannot
accidentally be "better" because of a different quantiser implementation.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import torch

from .config import QuantConfig
from .logging_utils import log_event, save_json
from . import quant_core


def probe_config(bits: int = 4, group_size: int = 128, sym: bool = False,
                 method: str = "rtn") -> QuantConfig:
    """Fixed probe hyper-parameters. Deliberately plain: no act_order, no clipping."""
    return QuantConfig(
        bits=bits,
        group_size=group_size,
        sym=sym,
        damp_percent=0.01,
        act_order=False,
        true_sequential=False,
        asym=False,                # symmetric reconstruction for the probe
        drift_correction=False,    # one pass per layer -> cheap
        mse_clip=False,
        hessian_dtype="float32",
    )


def build_probe(wrapped, batches: Optional[Sequence] = None, method: str = "rtn",
                bits: int = 4, group_size: int = 128, sym: bool = False,
                verbose: bool = True, tag: Optional[str] = None) -> Dict[str, object]:
    """
    Build and apply a probe model in place. Returns metadata + results dict.

    `batches` is required for method="gptq" and ignored for "rtn".
    """
    cfg = probe_config(bits=bits, group_size=group_size, sym=sym, method=method)
    t0 = time.time()

    if method == "rtn":
        results = quant_core.quantize_model_rtn(wrapped, cfg, verbose=verbose)
    elif method in ("gptq", "gptaq"):
        if batches is None:
            raise ValueError("method='gptq' needs calibration batches")
        cfg.asym = method == "gptaq"
        results = quant_core.quantize_model(wrapped, batches, cfg, verbose=verbose)
    else:
        raise ValueError(f"unknown probe method: {method}")

    tag = tag or f"probe_{method}_w{bits}"
    summary = quant_core.summarize_results(results)
    meta = {
        "tag": tag,
        "method": method,
        "bits": bits,
        "group_size": group_size,
        "sym": sym,
        "drift_correction": cfg.drift_correction,
        "asym": cfg.asym,
        "n_modules": summary["n_modules"],
        "rel_fro_err_by_module": summary["rel_fro_err_by_module"],
        "wall_clock_s": round(time.time() - t0, 1),
        "summary": summary,
    }
    if verbose:
        print(f"probe built: {tag}  modules={summary['n_modules']}  "
              f"{meta['wall_clock_s']}s")
    log_event("build_probe", **{k: v for k, v in meta.items() if k != "summary"})
    return {"results": results, "meta": meta}


def save_probe(results: Dict, meta: Dict, rel_dir: str) -> str:
    """Persist a probe so notebooks 04/08 do not have to re-quantise."""
    from .logging_utils import out

    path = out(f"{rel_dir}/probe.pt")
    quant_core.save_quant_state(results, path, meta=meta)
    save_json(meta, f"{rel_dir}/probe_meta.json")
    return path


def load_probe(wrapped, rel_dir: str, apply: bool = True) -> Dict:
    """Load and (optionally) apply a saved probe to the current model."""
    from .logging_utils import require

    path = require(f"{rel_dir}/probe.pt", produced_by="03_build_probe.ipynb")
    return quant_core.load_quant_state(wrapped, path, apply=apply)


def describe_probe(meta: Dict) -> str:
    return (f"probe={meta['method']} W{meta['bits']}A16 g{meta['group_size']} "
            f"sym={meta['sym']} drift={meta['drift_correction']} "
            f"modules={meta['n_modules']}")


# ======================================================================================
# Milestone 3: same-prefix score extraction
# ======================================================================================


@torch.no_grad()
def extract_position_scores(
    ref_wrapper,
    probe_wrapper,
    batches,
    chunk: int = 64,
    topk: int = 0,
    with_topk_validation: bool = False,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Per-position JSD / forward-KL / entropy between the FP16 reference and a probe.

    Memory-safe by construction: the backbone forward is done once per sequence to get
    hidden states ([T, d], a few MB), and the vocabulary projection happens one chunk
    of positions at a time, so a [T, V] float32 logits tensor is never materialised.

    Alignment contract (v2 §9.1-9.2):
        only positions t with a valid target t+1 are scored, and the returned score
        vector for a sequence has exactly one entry per scored position, with the
        matching position indices in `positions`. Nothing is silently zero-padded.

    Returns
    -------
    dict with per-sequence lists of tensors plus a `summary` block.
    """
    from .batching import target_positions
    from .jsd import divergence_per_position, jsd_summary

    seq_jsd: List[torch.Tensor] = []
    seq_ent: List[torch.Tensor] = []
    seq_kl: List[torch.Tensor] = []
    seq_pos: List[torch.Tensor] = []
    seq_ids: List[torch.Tensor] = []
    topk_err: List[float] = []

    for bi, batch in enumerate(batches):
        pos = target_positions(batch.valid_mask)
        if pos.numel() == 0:
            continue
        h_ref = ref_wrapper.forward_hidden(batch.input_ids, batch.attention_mask)[0]
        h_probe = probe_wrapper.forward_hidden(batch.input_ids, batch.attention_mask)[0]

        jsd = torch.zeros(pos.numel(), dtype=torch.float32, device=h_ref.device)
        ent = torch.zeros_like(jsd)
        kl = torch.zeros_like(jsd)

        for i in range(0, pos.numel(), chunk):
            idx = pos[i:i + chunk]
            lr = ref_wrapper._project(h_ref.index_select(0, idx))
            lq = probe_wrapper._project(h_probe.index_select(0, idx))
            jsd[i:i + idx.numel()] = divergence_per_position(lr, lq, "jsd", chunk=chunk)
            ent[i:i + idx.numel()] = divergence_per_position(lr, lr, "entropy", chunk=chunk)
            kl[i:i + idx.numel()] = divergence_per_position(lr, lq, "kl", chunk=chunk)
            if with_topk_validation and topk and bi < 1:
                approx = divergence_per_position(lr, lq, "jsd", chunk=chunk, topk=topk)
                topk_err.append(float(
                    (approx - jsd[i:i + idx.numel()]).abs().mean()
                    / jsd[i:i + idx.numel()].mean().clamp_min(1e-12)
                ))

        seq_jsd.append(jsd.cpu())
        seq_ent.append(ent.cpu())
        seq_kl.append(kl.cpu())
        seq_pos.append(pos.cpu())
        seq_ids.append(torch.full((pos.numel(),), bi, dtype=torch.long))
        if verbose and (bi + 1) % 4 == 0:
            print(f"    scored {bi+1}/{len(batches)} sequences")

    if not seq_jsd:
        return {"jsd": [], "entropy": [], "kl": [], "summary": {"n_positions": 0}}

    all_jsd = torch.cat(seq_jsd)
    summary = {
        "n_sequences": len(seq_jsd),
        "n_positions": int(all_jsd.numel()),
        "jsd": jsd_summary(all_jsd),
        "entropy": jsd_summary(torch.cat(seq_ent)),
        "kl": jsd_summary(torch.cat(seq_kl)),
        "topk": int(topk),
        "topk_mean_rel_err": (sum(topk_err) / len(topk_err)) if topk_err else None,
        "chunk": int(chunk),
    }
    return {
        "jsd": seq_jsd, "entropy": seq_ent, "kl": seq_kl,
        "positions": seq_pos, "seq_id": seq_ids, "summary": summary,
    }
