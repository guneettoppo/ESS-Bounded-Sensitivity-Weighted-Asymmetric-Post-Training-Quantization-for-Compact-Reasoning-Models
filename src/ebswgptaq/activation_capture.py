"""
Capture per-module input activations for geometry analysis (Milestone 4).

Returns a list with one [d_in, T] tensor per calibration sequence rather than one
giant matrix, because a 1.5B model with 32k calibration tokens produces ~1.2 GB per
large module and there is no reason to hold it all at once.

The activation returned by a forward hook is the module's *input*, i.e. exactly the
matrix X that the solver accumulates into the Hessian, so geometry diagnostics and
the solver see the same columns in the same order.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from .batching import CalibBatch


def capture_layer_inputs(
    wrapped,
    batches: Sequence[CalibBatch],
    layer_idx: int,
    module_filter: Optional[Sequence[str]] = None,
    store_dtype: torch.dtype = torch.float16,
) -> Dict[str, List[torch.Tensor]]:
    """
    For every linear module of one decoder layer, capture [d_in, T] inputs per batch.

    Returns {module_name: [tensor_batch0, tensor_batch1, ...]} on CPU.
    """
    mods = wrapped.layer_linears(layer_idx)
    if module_filter:
        mods = {n: m for n, m in mods.items() if any(f in n for f in module_filter)}

    out: Dict[str, List[torch.Tensor]] = {name: [] for name in mods}
    handles = []

    def make(name):
        def hook(_mod, inp, _out):
            x = inp[0]
            if x.dim() == 3:
                x = x[0]
            out[name].append(x.detach().to(store_dtype).T.contiguous().cpu())
        return hook

    for name, mod in mods.items():
        handles.append(mod.register_forward_hook(make(name)))
    try:
        with torch.no_grad():
            for batch in batches:
                wrapped.forward_hidden(batch.input_ids, batch.attention_mask)
    finally:
        for h in handles:
            h.remove()
    return out


def concat_columns(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """[d, T1] + [d, T2] + ... -> [d, sum(T)]."""
    if not tensors:
        raise ValueError("no tensors to concatenate")
    return torch.cat(list(tensors), dim=1)


def select_positions(X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Column subset selected by a boolean mask over columns."""
    m = mask.detach().cpu().flatten().bool()
    if m.numel() != X.shape[1]:
        raise ValueError(f"mask length {m.numel()} != {X.shape[1]} columns")
    return X[:, m]


def target_masks_per_batch(batches) -> List[torch.Tensor]:
    """
    One boolean mask per batch, True at positions that have a valid next-token target.

    Captured activations have one column per *token*, while sensitivity scores exist
    only for positions with a target, so any analysis that joins the two (geometry,
    weighting) must select columns through these masks. Getting this wrong shifts
    every score by one column and is invisible in the output.
    """
    from .batching import target_positions

    masks = []
    for b in batches:
        m = torch.zeros(b.length, dtype=torch.bool)
        pos = target_positions(b.valid_mask)
        if pos.numel():
            m[pos] = True
        masks.append(m)
    return masks


def concat_target_columns(tensors: Sequence[torch.Tensor], masks: Sequence[torch.Tensor],
                          dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Concatenate per-batch [d, T] tensors keeping only scored columns -> [d, P]."""
    if len(tensors) != len(masks):
        raise ValueError(f"{len(tensors)} tensors vs {len(masks)} masks")
    parts = [t.to(dtype)[:, m] for t, m in zip(tensors, masks)]
    return torch.cat(parts, dim=1)


def column_alignment_check(wrapped, batch: CalibBatch, layer_idx: int) -> Dict[str, object]:
    """
    Empirical alignment probe: column count must equal the number of valid tokens,
    and column t must be the activation that produced logit row t.

    The cheap invariant verified here is that the captured matrix has exactly
    `batch.n_valid` columns for a single-sequence batch, which catches double-counting
    (e.g. a hook firing twice or a [B,T,d] reshape that transposes batch and time).
    """
    caps = capture_layer_inputs(wrapped, [batch], layer_idx)
    first = next(iter(caps.values()))
    d = first[0].shape[0]
    return {
        "layer": layer_idx,
        "modules_captured": len(caps),
        "d_in": int(d),
        "columns": int(first[0].shape[1]),
        "valid_tokens": int(batch.n_valid),
        "aligned": bool(first[0].shape[1] == batch.n_valid),
    }
