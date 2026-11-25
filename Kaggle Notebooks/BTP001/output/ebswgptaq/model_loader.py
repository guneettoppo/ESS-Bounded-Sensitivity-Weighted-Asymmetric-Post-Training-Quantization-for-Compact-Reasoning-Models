"""
Model loading and the forward-pass wrapper used by every notebook.

Why a wrapper
-------------
T4 has no native BF16 tensor cores and Qwen-family logits can exceed the FP16
range (|logit| > 65504 is unusual but the *pre-softmax* accumulation can overflow
in fp16 for some Qwen2 checkpoints). Two guards are applied here:

1. `fp32_lm_head=True` keeps `lm_head` in FP32 and computes logits as
   `lm_head(hidden.float())`, so the vocabulary projection never overflows.
2. `safe_logits()` asserts finite values and reports the offending batch instead
   of letting NaNs propagate silently into JSD/KL (where they become `nan`
   scores that quietly destroy the weighting).

The wrapper also owns the layer/linear enumeration that the quantizer walks, so
there is exactly one definition of "which modules get quantized".
"""
from __future__ import annotations

import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .config import ModelConfig


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class ModelWrapper:
    """Thin, explicit wrapper around a HF causal LM."""

    def __init__(self, model: nn.Module, tokenizer, cfg: ModelConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = next(model.parameters()).device
        self._lm_head_fp32 = None
        self._decoder_layers = None

    # ------------------------------------------------------------------ basics

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @property
    def n_layers(self) -> int:
        return len(self.decoder_layers())

    def _backbone_attr(self) -> str:
        for attr in ("model", "transformer", "decoder"):
            if hasattr(self.model, attr):
                return attr
        raise AttributeError(
            "Cannot find the transformer backbone on this model. "
            "Expected .model / .transformer / .decoder."
        )

    def transformer(self) -> nn.Module:
        """The backbone without the LM head."""
        return getattr(self.model, self._backbone_attr())

    def decoder_layers(self) -> List[nn.Module]:
        if self._decoder_layers is not None:
            return self._decoder_layers
        backbone = self.transformer()
        for attr in ("layers", "h", "block", "blocks"):
            if hasattr(backbone, attr):
                self._decoder_layers = list(getattr(backbone, attr))
                return self._decoder_layers
        raise AttributeError("Cannot find transformer blocks (layers/h/block).")

    def linear_modules(self, prefix: str = "") -> "Dict[str, nn.Linear]":
        """
        Every nn.Linear in the whole model, keyed by a name that is resolvable via
        `model.get_submodule(name)` (so it is prefixed with the backbone attribute).
        The LM head and too-small layers are excluded.
        """
        out: Dict[str, nn.Linear] = {}
        for name, mod in self.model.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            if any(pat in name for pat in self.cfg.exclude_name_patterns if pat):
                continue
            out[f"{prefix}{name}"] = mod
        return out

    def layer_linears(self, layer_idx: int) -> Dict[str, nn.Linear]:
        prefix = self._layer_prefix(layer_idx)
        found: Dict[str, nn.Linear] = {}
        for name, mod in self.decoder_layers()[layer_idx].named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            full = f"{prefix}.{name}" if name else prefix
            if any(pat in full for pat in self.cfg.exclude_name_patterns if pat):
                continue
            found[full] = mod
        return found

    def _layer_prefix(self, layer_idx: int) -> str:
        backbone_attr = self._backbone_attr()
        backbone = self.transformer()
        for attr in ("layers", "h", "block", "blocks"):
            if hasattr(backbone, attr):
                return f"{backbone_attr}.{attr}.{layer_idx}"
        return f"{backbone_attr}.{layer_idx}"

    # -------------------------------------------------------------- lm head

    def lm_head_weight(self) -> torch.Tensor:
        head = self.model.get_output_embeddings()
        if self.cfg.fp32_lm_head:
            if self._lm_head_fp32 is None:
                self._lm_head_fp32 = head.weight.detach().to(torch.float32)
            return self._lm_head_fp32
        return head.weight.detach()

    # -------------------------------------------------------------- forward

    @torch.no_grad()
    def forward_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        need_hidden: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        """
        Run the backbone and produce FP32 logits.

        Returns (logits [B, T, V] float32, hidden_states or None).

        The head is applied manually so that (a) FP16 overflow cannot happen in
        the vocabulary projection on a T4, and (b) activations can be captured
        from the backbone alone without materialising logits.
        """
        out = self.transformer()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=need_hidden,
            return_dict=True,
        )
        hidden = out.last_hidden_state
        logits = self._project(hidden)
        hs = getattr(out, "hidden_states", None)
        return logits, hs

    def _project(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Hidden states -> FP32 logits, chunked over positions.

        Hidden states are [.., d]; the output is [.., V]. Chunking means a
        [T=2048, V=151936] float32 tensor (1.2 GB) never exists at once.
        """
        weight = self.lm_head_weight()
        lead = hidden.shape[:-1]
        flat = hidden.reshape(-1, hidden.shape[-1])
        chunks = []
        step = 256
        for i in range(0, flat.shape[0], step):
            piece = flat[i:i + step].to(torch.float32)
            chunks.append(piece @ weight.to(piece.device).T)
        logits = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
        return logits.reshape(*lead, logits.shape[-1])

    @torch.no_grad()
    def forward_hidden(self, input_ids: torch.Tensor,
                       attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Backbone forward *without* the vocabulary projection.

        Activation capture and divergence extraction use this: projecting 32k
        calibration tokens through a 152k-way head costs more FLOPs than the whole
        transformer pass and produces a tensor nobody needs.
        """
        out = self.transformer()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return out.last_hidden_state

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model.generate(input_ids=input_ids, **kwargs)

    # -------------------------------------------------------------- weights

    def set_weight(self, name: str, value: torch.Tensor) -> None:
        mod = self.model.get_submodule(name)
        mod.weight.data.copy_(value.to(mod.weight.dtype).to(mod.weight.device))

    def get_weight(self, name: str) -> torch.Tensor:
        return self.model.get_submodule(name).weight.detach()

    def named_linears(self) -> Dict[str, nn.Linear]:
        return self.linear_modules()


def safe_logits(logits: torch.Tensor, tag: str = "") -> torch.Tensor:
    """Fail loudly on non-finite logits instead of propagating NaNs."""
    if not torch.isfinite(logits).all():
        n_bad = int((~torch.isfinite(logits)).sum())
        raise FloatingPointError(
            f"Non-finite logits detected {tag}: {n_bad} values. "
            "If this is a Qwen checkpoint running in fp16 on a T4, keep "
            "fp32_lm_head=True (the default) and consider attn_implementation='eager'."
        )
    return logits


def load_model(
    cfg: ModelConfig,
    load_in_dtype: Optional[torch.dtype] = None,
    verbose: bool = True,
) -> ModelWrapper:
    """Load tokenizer + causal LM in the requested dtype."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = load_in_dtype or DTYPES.get(cfg.dtype, torch.float16)
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_id, revision=cfg.revision, trust_remote_code=cfg.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    common = dict(
        revision=cfg.revision,
        device_map=cfg.device if cfg.device != "cpu" else "cpu",
        attn_implementation=cfg.attn_implementation,
        trust_remote_code=cfg.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype=dtype, **common)
    except TypeError:
        # transformers 5.x renamed torch_dtype -> dtype
        model = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=dtype, **common)
    model.eval()
    model.config.use_cache = False

    wrapper = ModelWrapper(model, tokenizer, cfg)
    if verbose:
        n_par = sum(p.numel() for p in model.parameters())
        print(f"Loaded {cfg.model_id}  dtype={dtype}  params={n_par/1e9:.3f}B  "
              f"layers={wrapper.n_layers}  in {time.time()-t0:.1f}s")
        print(f"  revision={getattr(model.config, '_name_or_path', cfg.model_id)} "
              f"lm_head_dtype={wrapper.lm_head_weight().dtype}")
        if torch.cuda.is_available():
            print(f"  cuda_allocated={torch.cuda.memory_allocated()/2**30:.2f} GB")
    return wrapper


def model_fingerprint(cfg: ModelConfig, tokenizer) -> Dict[str, object]:
    """Model/tokenizer revision record for the results table (v2 §11)."""
    return {
        "model_id": cfg.model_id,
        "requested_revision": cfg.revision,
        "dtype": cfg.dtype,
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "tokenizer_name": getattr(tokenizer, "name_or_path", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
    }
