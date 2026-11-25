"""
Integer-code export and packing for real low-bit kernels (thin adapter).

The solver deliberately never touches a kernel. It returns integer codes plus group
parameters, and this module is the only place that knows about GPTQ's on-disk
layout. That keeps the research code honest: everything measured through the
proxy gate is fake-quant, and turning a checkpoint into real INT3/INT4 inference is
a separate, verifiable step.

Two conventions are translated here, and both are easy to get wrong:

1. Zero representation.
   Our quantiser stores the *minimum* of each group (`zero` is a float offset):
        dequant = q * scale + zero
   GPTQ stores an integer zero-point z with
        dequant = (q - z) * scale
   so z = round(-zero / scale). If you forget this, asymmetric checkpoints come out
   with a constant shift and accuracy looks catastrophic for no reason.

2. Bit packing.
   `qweight` is int32, big-endian by bit order, with `32 // bits` consecutive input
   channels packed per word: word r of column c holds channels
   [r*words .. r*words + words - 1], low bits first.

`self_test()` unpacks the packed tensors and compares against the solver's own
dequantisation, so a layout mistake fails here instead of silently in a kernel.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import torch

from .logging_utils import save_json


def words_per_int(bits: int) -> int:
    if 32 % bits != 0:
        raise ValueError(f"bits={bits} does not divide 32; kernel packing needs 2,4,8,16")
    return 32 // bits


PACKABLE_BITS = (2, 4, 8, 16)


def pack_codes_along_in(q_int: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Pack integer codes [out, in] into GPTQ's qweight layout [in/words, out] int32.

    Consecutive input channels share a word; the first channel occupies the low bits.
    """
    w = words_per_int(bits)
    out_f, in_f = q_int.shape
    if in_f % w != 0:
        raise ValueError(f"in_features={in_f} is not a multiple of {w}")

    vals = q_int.to(torch.int64).transpose(0, 1).contiguous()   # [in, out]
    packed = torch.zeros(in_f // w, out_f, dtype=torch.int32)
    mask = (1 << bits) - 1
    for i in range(w):
        chunk = (vals[i::w] & mask).to(torch.int32)
        packed |= chunk << (bits * i)
    return packed


def unpack_codes_along_in(packed: torch.Tensor, bits: int, out_f: int) -> torch.Tensor:
    """Inverse of pack_codes_along_in. Returns [out, in] int64 codes."""
    w = words_per_int(bits)
    mask = (1 << bits) - 1
    rows = packed.shape[0]
    in_f = rows * w
    vals = torch.zeros(in_f, out_f, dtype=torch.int64)
    p = packed.to(torch.int64)
    for i in range(w):
        vals[i::w] = (p >> (bits * i)) & mask
    return vals.transpose(0, 1).contiguous()


def to_zero_point_form(zeros: torch.Tensor, scales: torch.Tensor, bits: int,
                       sym: bool) -> torch.Tensor:
    """Convert our float-min `zero` to GPTQ's integer zero-point."""
    if sym:
        return torch.full_like(zeros, (1 << (bits - 1)))
    qmax = float(2 ** bits - 1)
    z = torch.round(-zeros.to(torch.float32) / scales.to(torch.float32).clamp_min(1e-12))
    return z.clamp(0, qmax).to(torch.int32)


def export_module(result, bits: int, sym: bool) -> Dict[str, torch.Tensor]:
    """
    One QuantizedLinear -> GPTQ-style tensors plus our own dequant reference.

    For bit widths that do not divide 32 (3, 5, 6, 7) the integer codes are exported
    unpacked as int8 under `q_int`, because GPTQ's strided int32 layout has no exact
    3-bit form: real 3-bit kernels use a different stride/padding convention that has
    to be validated against the specific loader. Fake-quant evaluation (which is what
    the proxy gate uses) needs nothing more than `q_int` + group parameters.
    """
    z_int = to_zero_point_form(result.zeros, result.scales, bits, sym)
    if sym:
        # GPTQ stores symmetric codes offset by 2^(b-1) with a zero-point of the same
        # value, so that dequant is (q_stored - z)*scale for both grid types.
        offset = 1 << (bits - 1)
        codes_store = (result.q_int.to(torch.int32) + offset)
        z_int = torch.full_like(result.zeros, float(offset)).to(torch.int32)
    else:
        codes_store = result.q_int.to(torch.int32)

    out: Dict[str, torch.Tensor] = {
        "scales": result.scales.to(torch.float16).contiguous(),      # [out, G]
        "g_idx": result.g_idx.to(torch.int32),
        "W_hat": result.W_hat.detach().to(torch.float16),            # fake-quant reference
        "q_int": codes_store.to(torch.int8).cpu(),                   # always available
    }
    if bits in PACKABLE_BITS:
        out["qweight"] = pack_codes_along_in(codes_store, bits)       # [in/w, out]
        out["qzeros"] = pack_codes_along_in(z_int.transpose(0, 1).contiguous(), bits)
        out["layout"] = torch.tensor([0], dtype=torch.int8)           # 0 = packed
    else:
        out["qzeros_int"] = z_int.cpu()
        out["layout"] = torch.tensor([1], dtype=torch.int8)           # 1 = unpacked int8
    return out


def self_test(result, bits: int, sym: bool) -> Dict[str, float]:
    """
    Round-trip: pack -> unpack -> dequantise, and compare with the solver's W_hat.

    A large error here means a layout/sign bug, not a modelling problem.
    """
    exp = export_module(result, bits, sym)
    out_f, in_f = result.q_int.shape

    if "qweight" in exp:
        codes = unpack_codes_along_in(exp["qweight"], bits, out_f)
        code_match = float((codes == exp["q_int"].to(torch.int64)).float().mean())
    else:
        codes = exp["q_int"].to(torch.int64)
        code_match = 1.0

    # rebuild dequantised weights from the *exported* representation
    g = result.g_idx.long()
    sc = result.scales.index_select(1, g)
    z_src = exp.get("qzeros_int", None)
    if z_src is None:
        z_int = to_zero_point_form(result.zeros, result.scales, bits, sym)
    else:
        z_int = z_src
    z = z_int.index_select(1, g)
    W_rt = (codes.to(torch.float32) - z.to(torch.float32)) * sc
    denom = result.W_hat.abs().mean().clamp_min(1e-8)
    return {
        "code_roundtrip_exact_frac": code_match,
        "layout": "packed" if "qweight" in exp else "unpacked-int8",
        "dequant_max_abs_err": float((W_rt - result.W_hat.to(torch.float32)).abs().max()),
        "dequant_rel_err": float(((W_rt - result.W_hat.to(torch.float32)).abs().mean()
                                 / denom)),
    }


def save_packed_checkpoint(results: Dict, out_dir: str, bits: int, sym: bool,
                           meta: Optional[Dict] = None) -> str:
    """Write one safetensors/torch file per module plus a config.json describing it."""
    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        "format": "ebswgptaq-fakequant-v1",
        "bits": bits,
        "sym": sym,
        "note": ("Fake-quant checkpoint. Weights are stored as integer codes plus group "
                 "parameters; W_hat is the dequantised reference. Load with "
                 "ebswgptaq.quant_core.load_quant_state for proxy evaluation, or run "
                 "ebswgptaq.quantizer_adapter.export_for_kernel for real kernels."),
        "modules": {},
    }
    tensors: Dict[str, torch.Tensor] = {}
    for name, res in results.items():
        exp = export_module(res, bits, sym)
        safe = name.replace(".", "__")
        for key, value in exp.items():
            tensors[f"{safe}.{key}"] = value.cpu()
        manifest["modules"][name] = {"prefix": safe, "meta": res.meta}

    path = os.path.join(out_dir, "model.safetensors")
    try:
        from safetensors.torch import save_file

        save_file(tensors, path)
    except Exception:
        path = os.path.join(out_dir, "model.pt")
        torch.save(tensors, path)
    if meta:
        manifest["meta"] = meta
    save_json(manifest, os.path.join(out_dir, "quant_config.json"))
    return out_dir


def export_for_kernel(results: Dict, out_dir: str, base_model_id: str, bits: int,
                      sym: bool, desc_act: bool = False) -> str:
    """
    Placeholder for the real-kernel step (later phase).

    Deliberately raises: producing a loadable Marlin/GPTQModel checkpoint requires
    verified qzeros/qweight conventions for the exact library version in use, and
    an unverified export would produce a checkpoint that silently mis-dequantises.
    The saved fake-quant tensors are already in the shape the converter needs.

    When this phase is implemented:
      1. save_packed_checkpoint(...) here
      2. copy the base model's config.json / tokenizer files
      3. convert with GPTQModel: `GPTQModel.load(base_model_id, model_save_path)`
         or `python -m gptqmodel.export --model <dir> --format marlin`
      4. verify: dequantise the converted checkpoint and compare against `W_hat`
         with max_abs_err < 1e-3 (fp16 tolerance) before reporting any speed number
      5. only then report VRAM and tokens/sec (v2 §9.5: never mix compression
         metrics with throughput claims)
    """
    raise NotImplementedError(
        "Real packed kernels are a later phase (see LATER_PHASES.md, notebook 12). "
        "Use save_packed_checkpoint() to export the integer codes now. "
        f"Base model for the final export: {base_model_id}"
    )
