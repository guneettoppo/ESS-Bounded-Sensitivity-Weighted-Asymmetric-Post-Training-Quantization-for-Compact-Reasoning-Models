"""
Integration test for the model-level driver using a mock transformer.

The orchestration in `quant_core.quantize_model` is where the fragile parts live:
forward hooks, [1,T,d] -> [d,T] column order, swapping FP16 reference weights back
in, and the ordering guarantee that layer l sees layers < l already quantised.

None of that is exercised by a per-layer unit test, and none of it needs a real
checkpoint. This file builds a 3-block toy "Qwen-shaped" model with the same
attribute layout (model.layers[i].<proj>) and runs the full driver on it.

Run: .venv/Scripts/python.exe tests/test_pipeline.py
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from ebswgptaq.batching import CalibBatch, batches_from_records
from ebswgptaq.config import ModelConfig, QuantConfig
from ebswgptaq.model_loader import ModelWrapper
from ebswgptaq import quant_core
from ebswgptaq.evaluation import teacher_forced_fidelity

torch.manual_seed(0)
FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILS.append(name)


D, VOCAB, NLAYERS = 32, 64, 3


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(D, D)
        self.o_proj = nn.Linear(D, D)
        self.up_proj = nn.Linear(D, 2 * D)
        self.down_proj = nn.Linear(2 * D, D)

    def forward(self, x):
        h = self.o_proj(torch.tanh(self.q_proj(x)))
        return x + self.down_proj(torch.nn.functional.gelu(self.up_proj(h)))


class Backbone(nn.Module):
    """Qwen-shaped backbone: embedding lives on the backbone, not the LM."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, D)
        self.layers = nn.ModuleList([Block() for _ in range(NLAYERS)])

    def forward(self, input_ids=None, attention_mask=None, use_cache=False,
                output_hidden_states=False, return_dict=True, **kw):
        x = self.embed_tokens(input_ids)
        for block in self.layers:
            x = block(x)
        return SimpleNamespace(last_hidden_state=x, hidden_states=None)


class ToyCausalLM(nn.Module):
    """Same public surface as a HF causal LM, but tiny and deterministic."""

    def __init__(self):
        super().__init__()
        self.model = Backbone()
        self.lm_head = nn.Linear(D, VOCAB, bias=False)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids=None, **kw):
        return self.model(input_ids=input_ids)


def make_batches(n_seq=4, length=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    records = []
    for i in range(n_seq):
        records.append({
            "prompt_id": f"p{i}",
            "source": "synthetic",
            "token_ids": torch.randint(0, VOCAB, (length,), generator=g).tolist(),
        })
    return batches_from_records(records, max_seq_len=length)


class Unused:
    pass


def build_wrapper(dtype=torch.float32):
    model = ToyCausalLM().to(dtype)
    # copy params through so our generic module loader can enumerate them
    wrapper = ModelWrapper(model, tokenizer=None, cfg=ModelConfig(model_id="toy"))
    wrapper.device = torch.device("cpu")
    return wrapper


print("=" * 70)
print("Integration test: model-level driver on a toy transformer")
print("=" * 70)

batches = make_batches()
print(f"batches: {len(batches)} x {batches[0].length} tokens, "
      f"{sum(b.n_valid for b in batches)} calibration columns")

wrapper = build_wrapper()
linears = wrapper.linear_modules()
check("module enumeration", len(linears) == NLAYERS * 4,
      f"{len(linears)} linears: {sorted(linears)[:2]}...")
check("lm_head excluded", all("lm_head" not in n for n in linears))
check("names resolvable", all(wrapper.model.get_submodule(n) is m
                              for n, m in linears.items()))

# ---------------------------------------------------------------- activation capture
print("\nactivation capture and column alignment")
cap = quant_core.capture_inputs(wrapper, batches[0],
                                wrapper.layer_linears(0))
for name, x in cap.items():
    check(f"capture {name.split('.')[-1]} shape [d_in, T]",
          x.shape == (wrapper.model.get_submodule(name).in_features, batches[0].length),
          str(tuple(x.shape)))

# the captured input of layer 0's q_proj must equal the embedding output
# (capture stores fp16 for memory, so compare at fp16 tolerance)
manual = wrapper.transformer().embed_tokens(batches[0].input_ids)[0]
check("captured columns equal the true activations",
      torch.allclose(cap["model.layers.0.q_proj"].float(), manual.T.float(), atol=5e-3),
      f"max_err={(cap['model.layers.0.q_proj'].float() - manual.T.float()).abs().max():.2e}")

# ---------------------------------------------------------------- drift + solve
print("\nquantize_model with drift correction (2 passes per layer)")
before = {n: m.weight.detach().clone() for n, m in wrapper.linear_modules().items()}
cfg = QuantConfig(bits=4, group_size=16, asym=True, drift_correction=True,
                  act_order=False, damp_percent=0.01, diag_max_dim=64)
results = quant_core.quantize_model(wrapper, batches, cfg, verbose=False)
check("every layer quantized", len(results) == NLAYERS * 4,
      f"{len(results)} modules")
changed = [n for n, m in wrapper.linear_modules().items()
           if not torch.allclose(m.weight.detach(), before[n])]
check("weights were modified in place", len(changed) == len(results))
finite = all(torch.isfinite(r.W_hat).all() for r in results.values())
check("all fake-quant weights finite", bool(finite))
rel = [r.meta["rel_fro_err"] for r in results.values()]
check("reconstruction error is in a sane range (0 < err < 0.5)",
      all(0 < x < 0.5 for x in rel), f"mean={sum(rel)/len(rel):.4f}")

# objective must beat RTN on the same layer, using the solver's own statistics
print("\nobjective ordering vs RTN on the same layer")
w0 = quant_core.quantize_model_rtn(build_wrapper(), QuantConfig(bits=4, group_size=16),
                                   verbose=False, max_layers=NLAYERS)
for name in list(results)[:4]:
    r = results[name]
    rr = w0[name]
    # recompute the layer statistics for a fair comparison
    l_idx = int(name.split(".")[2])
    acc = quant_core.WeightedHessian(r.q_int.shape[1], "float32", "cpu", track_cross=True)
    for b in batches:
        x = quant_core.capture_inputs(wrapper, b, {name: wrapper.model.get_submodule(name)})[name]
        acc.add(x, None, b.weights)
    o_sol = quant_core.reconstruction_objective(r.W_hat, before[name], acc.H, acc.Cw)
    o_rtn = quant_core.reconstruction_objective(rr.W_hat, before[name], acc.H, acc.Cw)
    check(f"obj solver < rtn  ({name.split('.')[-1]})", o_sol <= o_rtn,
          f"sol={o_sol:.4e} rtn={o_rtn:.4e}")

# ---------------------------------------------------------------- no drift
print("\nquantize_model without drift correction (1 pass per layer)")
w2 = build_wrapper()
res2 = quant_core.quantize_model(w2, batches, QuantConfig(bits=4, group_size=16,
                                                         asym=True, drift_correction=False),
                                 verbose=False)
check("single-pass mode completes", len(res2) == NLAYERS * 4)
check("single-pass mode is finite",
      all(torch.isfinite(r.W_hat).all() for r in res2.values()))

# ---------------------------------------------------------------- weighted run
print("\nquantize_model with token weights (ESS-bounded)")
from ebswgptaq.score_weights import build_weights
from ebswgptaq.config import WeightConfig

g = torch.Generator().manual_seed(3)
for b in batches:
    scores = torch.randn(b.length, generator=g) * 2
    b.weights = build_weights(scores, torch.arange(b.length),
                              None, WeightConfig(signal="jsd", ess_floor=0.5))["weights"]
w3 = build_wrapper()
res3 = quant_core.quantize_model(w3, batches, QuantConfig(bits=4, group_size=16,
                                                         asym=True, drift_correction=True),
                                 verbose=False)
check("weighted run completes", len(res3) == NLAYERS * 4)
check("weighted run finite", all(torch.isfinite(r.W_hat).all() for r in res3.values()))

# ---------------------------------------------------------------- fidelity
print("\nteacher_forced_fidelity between FP16 and the quantized model")
ref = build_wrapper()


def clone_wrapper(src):
    """A second model with *identical* weights (so KL=0 is the expected value)."""
    w = build_wrapper()
    w.model.load_state_dict(src.model.state_dict())
    return w


q = clone_wrapper(ref)
for name, r in res3.items():
    q.set_weight(name, r.W_hat)
out = teacher_forced_fidelity(ref, q, batches, chunk_positions=16)
agg = out["aggregate"]
print(f"       n_seq={agg['n_sequences']} mean_kl={agg['mean_kl']:.4e} "
      f"mean_jsd={agg['mean_jsd']:.4e} nll_delta={agg['nll_delta']:.4e}")
check("fidelity returns one row per sequence", len(out["per_sequence"]) == len(batches))
check("quantized model diverges from FP16 (KL > 0)", agg["mean_kl"] > 0)
check("KL is not absurd", agg["mean_kl"] < 5.0, f"{agg['mean_kl']:.3e}")
check("nll delta is finite and small", abs(agg["nll_delta"]) < 1.0,
      f"{agg['nll_delta']:.3e}")
print("       note: the *sign* of nll_delta is not meaningful on a random toy model "
      "evaluated on its own calibration data; only finiteness and magnitude are checked.")
check("per-position arrays exist", "arrays" in out and out["arrays"]["kl"].numel() > 0)

# identical models must give zero divergence
out_zero = teacher_forced_fidelity(ref, clone_wrapper(ref), batches, chunk_positions=16)
check("identical models give KL ~ 0", out_zero["aggregate"]["mean_kl"] < 1e-9,
      f"{out_zero['aggregate']['mean_kl']:.2e}")

print("\n" + "=" * 70)
if FAILS:
    print(f"{len(FAILS)} FAILURES: {FAILS}")
    sys.exit(1)
print("ALL PIPELINE TESTS PASSED")
