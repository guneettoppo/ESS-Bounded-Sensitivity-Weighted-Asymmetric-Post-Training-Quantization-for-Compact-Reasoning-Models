"""Notebook specs 00-04 (environment, traces, baseline, probes, JSD)."""
from nb_common import md, code, epilogue

NB00_TITLE = "00_environment_check"

NB00 = [
    md("""\
# 00 · Environment, reproducibility and library self-check

**What this notebook does.** Records a full reproduction manifest (Python, torch,
CUDA, transformers, GPU model, revisions), then runs the library's *CPU-only* unit
checks so you find a broken install or a broken formula in 60 seconds instead of
40 minutes into a quantization run.

**Why this comes first.** Every later notebook writes numbers into a results table.
If the environment is not recorded, those numbers are unreproducible and the whole
study is unpublishable — that is Milestone 0 in the pivot document.

**Kaggle settings.**

| Setting | Value | Why |
|---|---|---|
| Accelerator | GPU T4 x1 (or P100) | 16 GB is enough for a 1.5B FP16 model + FP32 Hessians |
| Internet | **On** | model + dataset download |
| Persistence | Save Version -> *Save & Run All* | so later notebooks can attach this output |

**Runtime.** Under 2 minutes. No GPU work.

**Produces.** `env/env_report.json`, `logs/run_log.jsonl`
"""),
    code("""\
# Check the packages this project needs. Kaggle preinstalls all of them; this prints
# the exact versions that go into the reproduction manifest.
REQUIRED = ["torch", "transformers", "datasets", "accelerate", "safetensors", "yaml", "scipy"]
missing = []
print("package versions")
print("-" * 46)
for name in REQUIRED:
    try:
        mod = __import__(name)
        print(f"  {name:14s} {getattr(mod, '__version__', 'ok')}")
    except Exception:
        missing.append(name)
        print(f"  {name:14s} MISSING")

if missing:
    print("\\nInstall the missing packages, then re-run this cell:")
    print("  !pip -q install " + " ".join(missing))
else:
    print("\\nAll required packages are present.")
"""),
    code("""\
# Hardware reality check. The whole study is budgeted for a single 16 GB T4, so the
# numbers below decide how big a calibration corpus and model you can afford.
import os, json, torch

print("torch", torch.__version__, "| cuda runtime", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        native_bf16 = p.major >= 8
        print(f"  GPU {i}: {p.name} | {p.total_memory/2**30:.1f} GB | "
              f"sm_{p.major}{p.minor} | native bf16 = {native_bf16}")
    free, total = torch.cuda.mem_get_info()
    print(f"  free now: {free/2**30:.2f} GB of {total/2**30:.2f} GB")
    if not (p.major >= 8):
        print("  -> No native BF16: run the model in FP16 with fp32_lm_head=True "
              "(the default in configs/model.yaml).")

try:
    pages = os.sysconf("SC_PHYS_PAGES")
    page_size = os.sysconf("SC_PAGE_SIZE")
    print(f"\\nCPU RAM: {pages * page_size / 2**30:.1f} GB")
except Exception:
    pass
try:
    import shutil
    du = shutil.disk_usage("/kaggle/working" if os.path.isdir("/kaggle/working") else ".")
    print(f"Working disk: {du.free/2**30:.1f} GB free of {du.total/2**30:.1f} GB")
except Exception:
    pass
"""),
    code("""\
# CPU-only correctness checks. These are the same assertions the repo's test suite
# runs; they cover the mathematics every later notebook depends on.
import torch
from ebswgptaq import jsd, quant_core, score_weights, quantizer_adapter
from ebswgptaq.config import QuantConfig, WeightConfig

torch.manual_seed(0)
checks = {}

print("1) JSD invariants (symmetry, bounds, self-distance, top-k convergence)")
s = jsd.assert_jsd_sanity(v=512)
for k, v in s.items():
    print(f"     {k:24s} {v}")
checks["jsd_symmetry"] = s["symmetry_max_abs_diff"] < 1e-6
checks["jsd_self_zero"] = s["self_jsd_max"] < 1e-9
checks["jsd_bounded"] = bool(s["within_bound"])

print("\\n2) causal alignment (logits[t] scored against token_ids[t+1])")
ca = jsd.assert_causal_alignment()
print(f"     {ca}")
checks["causal_alignment"] = ca["all_pairs_correct"]

print("\\n3) group quantizer round-trip")
W = torch.randn(128, 512) * 0.02
q = quant_core.GroupQuantizer(bits=4, group_size=128, sym=False)
sc, zr, gi = q.fit(W)
q_int, dq = q.quantize(W, sc, zr, gi)
rel = float((dq - W).norm() / W.norm())
print(f"     q_int range [{int(q_int.min())}, {int(q_int.max())}] | "
      f"groups={sc.shape[1]} | rel_fro_err={rel:.4f}")
checks["quantizer_roundtrip"] = 0.0 < rel < 0.2

print("\\n4) token weights and Kish ESS")
w = score_weights.build_weights(torch.randn(1000) * 2, torch.arange(1000), None,
                                WeightConfig(signal="jsd", ess_floor=0.5))["weights"]
st = score_weights.concentration_stats(w)
print(f"     ess_frac={st['ess_fraction']:.3f} max={st['max']:.2f} "
      f"mass_top1%={st['mass_top1pct']:.3f}")
checks["ess_bounded"] = abs(st["ess_fraction"] - 0.5) < 0.03
checks["weights_mean_one"] = abs(st["mean"] - 1.0) < 1e-4

print("\\n5) integer-code packing round-trip")
r = quant_core.quantize_linear("t", W, W.abs().T @ W.abs(), None,
                               QuantConfig(bits=4, group_size=64, asym=False))
rt = quantizer_adapter.self_test(r, 4, False)
print(f"     {rt}")
checks["packing_roundtrip"] = rt["code_roundtrip_exact_frac"] > 0.999

print("\\n" + "-" * 60)
for name, ok in checks.items():
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
print("-" * 60)
assert all(checks.values()), f"failed checks: {[k for k, v in checks.items() if not v]}"
print("Environment is fit to run the pipeline.")
"""),
    code("""\
# Record the manifest. This file is the reproduction record for the whole study.
from ebswgptaq import logging_utils as lu

manifest = lu.environment_manifest({
    "notebook": "00_environment_check",
    "artifact_root": lu.write_root(),
    "config_dir": CONFIG_DIR,
    "library_version": __import__("ebswgptaq").__version__,
    "self_checks_passed": True,
})
lu.save_json(manifest, "env/env_report.json")
lu.log_event("00_environment_check", gpu=manifest.get("torch", {}).get("devices"))
print(json.dumps({k: manifest[k] for k in ("timestamp", "python", "platform")}, indent=2))
print("\\nwrote", lu.out("env/env_report.json"))
"""),
    md("""\
## Handoff

Nothing downstream depends on this notebook, but re-run it after changing the
environment so the manifest always matches the run that produced the results.

Next: **01_generate_traces.ipynb**.
"""),
]


NB01_TITLE = "01_generate_traces"

NB01 = [
    md("""\
# 01 · Mixed calibration corpus (self-generated traces + general anchors)

**What this notebook does.** Builds the calibration corpus every later notebook
shares, and a disjoint held-out corpus for the proxy gate.

| Part | Size (pilot default) | Source |
|---|---|---|
| Reasoning traces | 20 sequences | GSM8K **train** prompts, greedy self-generation by the FP16 model |
| General anchors | 7 sequences | WikiText-2 raw train |
| Held-out prompts | 8 sequences | GSM8K **train** prompts, disjoint ids, never used for calibration |

Ratio is 20:7 ≈ 74:26, i.e. the 75/25 mix from the study plan.

**Sizing matters more than it looks.** The solver builds a `d_in × d_in` Hessian from
`N` token columns. For `down_proj` in a 1.5B model `d_in = 8960`, so the corpus needs
roughly **18k+ tokens** for the Gram matrix to be well conditioned. These defaults
(20 traces × ~1400 tokens + 7 anchors) land around 30k. Notebook 02 asserts this and
refuses to proceed on a rank-deficient corpus — do not lower `n_reasoning` or
`gen_max_new_tokens` without re-checking.

**Three rules this notebook enforces, and why**

1. **Training splits only.** Evaluation prompts never enter the calibration set.
   The n-gram audit at the end is reported, not assumed.
2. **Greedy decoding for the primary corpus.** A sampled corpus cannot be
   regenerated bit-exactly, and an unreproducible calibration set invalidates every
   comparison downstream.
3. **Store `token_ids`, not text.** Notebooks 02-08 use these exact ids. Nothing
   re-tokenises, so the calibration columns are identical across variants.

**Kaggle settings.** GPU T4 x1, Internet **on**. **Runtime** ≈ 30-45 min
(20 calibration + 8 held-out traces, up to 1280 new tokens each, greedy).

**Produces**
`calib/calib_corpus.jsonl`, `calib/holdout_corpus.jsonl`, `calib/corpus_manifest.json`,
`calib/leak_audit.json`
"""),
    code("""\
# Prerequisites: none. This is the first stage that touches the model.
from ebswgptaq import logging_utils as lu
from ebswgptaq.config import ModelConfig, CalibConfig, load_config

print("artifact root:", lu.write_root())
print("configs      :", CONFIG_DIR)

mcfg = load_config(f"{CONFIG_DIR}/model.yaml", ModelConfig)
ccfg = load_config(f"{CONFIG_DIR}/calibration.yaml", CalibConfig)
lu.set_seed(ccfg.seed)
print("\\nmodel :", mcfg.model_id, "| dtype", mcfg.dtype, "| fp32_lm_head", mcfg.fp32_lm_head)
print("corpus:", ccfg.n_reasoning, "reasoning +", ccfg.n_general, "anchor +",
      ccfg.n_holdout, "holdout sequences, max_len", ccfg.max_seq_len)
print("decode:", "greedy" if not ccfg.gen_do_sample else "sampled",
      "| max_new_tokens", ccfg.gen_max_new_tokens)
"""),
    code("""\
# Load the model in FP16 and prove the forward pass is finite before generating
# anything. On a T4 this is also where an FP16 overflow in the 152k-way head shows up.
import torch
from ebswgptaq import model_loader as ml

wrapped = ml.load_model(mcfg)

enc = wrapped.tokenizer("What is 2 + 2? Answer:", return_tensors="pt").to(wrapped.device)
logits, _ = wrapped.forward_logits(enc["input_ids"], enc["attention_mask"])
ml.safe_logits(logits, tag="smoke-test")           # raises on NaN/Inf
print("logits:", tuple(logits.shape), logits.dtype, "| max |logit| =", float(logits.abs().max()))
print("next token:", repr(wrapped.tokenizer.decode(logits[0, -1].argmax())))
print(lu.gpu_memory_report())
"""),
    code("""\
# Pick prompts. Calibration and held-out ids are disjoint by construction
# (the held-out sample excludes every calibration id).
from ebswgptaq import calibration_data as cd

pool = cd.load_prompt_source(ccfg, limit=400)
print(f"prompt pool: {len(pool)} from {ccfg.prompt_dataset}/{ccfg.prompt_split}")

calib_prompts = cd.sample_prompts(pool, ccfg.n_reasoning, seed=ccfg.seed)
calib_ids = {p["prompt_id"] for p in calib_prompts}
holdout_prompts = cd.sample_prompts(pool, ccfg.n_holdout, seed=ccfg.seed + 977,
                                    exclude_ids=calib_ids)

overlap = {p["prompt_id"] for p in holdout_prompts} & calib_ids
assert not overlap, f"calibration/holdout leakage: {overlap}"
print(f"calibration prompts: {len(calib_prompts)}")
print(f"held-out prompts   : {len(holdout_prompts)}  (overlap: {len(overlap)})")
print("\\nexample calibration prompt:")
print(" ", calib_prompts[0]["prompt"][:200].replace("\\n", " "))
"""),
    md("""\
### Generate the reasoning traces

This is the slow part. Each trace is generated by the FP16 model itself, greedily,
from a training-split prompt — so the calibration activations sit on exactly the
distribution the quantised model will be asked about later.

`hit_max_new_tokens` is recorded per trace: a trace that ran to the cap is a
truncated thought, and you should report how many of those are in the corpus rather
than quietly including them.
"""),
    code("""\
import time
from ebswgptaq import trace_generation as tg

t0 = time.time()
calib_records = tg.generate_traces(wrapped, calib_prompts, ccfg, template="instruct")
print(f"\\ncalibration traces: {len(calib_records)} in {time.time()-t0:.0f}s")

t1 = time.time()
holdout_records = tg.generate_traces(wrapped, holdout_prompts, ccfg, template="instruct",
                                     verbose=False)
print(f"held-out traces   : {len(holdout_records)} in {time.time()-t1:.0f}s")

for name, recs in (("calibration", calib_records), ("held-out", holdout_records)):
    n_tok = [r["n_new_tokens"] for r in recs]
    n_cap = sum(r["hit_max_new_tokens"] for r in recs)
    print(f"  {name:12s} mean_new_tokens={sum(n_tok)/len(n_tok):7.1f}  "
          f"at_cap={n_cap}/{len(recs)}")
"""),
    code("""\
# General-language anchors: tokenised, not generated. Without these the calibration
# set is 100% math reasoning and general perplexity/formatting would be unprotected.
anchors = cd.load_general_anchors(ccfg, limit=ccfg.n_general)
print(f"anchor texts: {len(anchors)}")
anchor_records = tg.tokenize_anchors(wrapped, anchors, ccfg)
print(f"anchor sequences: {len(anchor_records)}  "
      f"lengths={[len(r['token_ids']) for r in anchor_records]}")
"""),
    code("""\
# Assemble, truncate to the calibration window, and report the corpus you actually got.
calib_corpus = tg.build_mixed_corpus(calib_records, anchor_records, ccfg)
holdout_corpus = tg.build_mixed_corpus(holdout_records, [], ccfg)

print("CALIBRATION CORPUS")
for k, v in tg.corpus_stats(calib_corpus).items():
    print(f"  {k:26s} {v}")
print("\\nHELD-OUT CORPUS")
for k, v in tg.corpus_stats(holdout_corpus).items():
    print(f"  {k:26s} {v}")

# sanity: no sequence may exceed the window the quantizer will use
assert all(len(r["token_ids"]) <= ccfg.max_seq_len for r in calib_corpus + holdout_corpus)
"""),
    code("""\
# Contamination audit against the GSM8K *test* split. Reported, never assumed.
# (The full later-phase audit covers MATH-500, MGSM, AIME and LiveCodeBench too.)
from datasets import load_dataset

gsm_test = load_dataset("openai/gsm8k", "main", split="test")
eval_texts = [str(x["question"]) for x in gsm_test]

calib_texts = []
for r in calib_corpus:
    if r.get("prompt"):
        calib_texts.append(r["prompt"])
    calib_texts.append(r["trace"][:2000])

audit = cd.leak_audit(calib_texts, eval_texts, n=ccfg.ngram_leak_n)
print(f"{ccfg.ngram_leak_n}-gram overlap audit vs GSM8K test ({len(eval_texts)} prompts)")
for k, v in audit.items():
    if k != "examples":
        print(f"  {k:16s} {v}")
if audit.get("examples"):
    print("  collisions:")
    for ex in audit["examples"]:
        print("   -", ex[:120])
"""),
    code("""\
# Persist. From here on, every notebook reads these files and never re-tokenises.
lu.write_jsonl(calib_corpus, "calib/calib_corpus.jsonl")
lu.write_jsonl(holdout_corpus, "calib/holdout_corpus.jsonl")

manifest = {
    "model": ml.model_fingerprint(mcfg, wrapped.tokenizer),
    "calibration": {**tg.corpus_stats(calib_corpus),
                    "protocol": cd.CALIB_EVAL_PROTOCOL,
                    "decoding": {"do_sample": ccfg.gen_do_sample,
                                 "temperature": ccfg.gen_temperature,
                                 "top_p": ccfg.gen_top_p,
                                 "max_new_tokens": ccfg.gen_max_new_tokens},
                    "seed": ccfg.seed,
                    "max_seq_len": ccfg.max_seq_len},
    "holdout": tg.corpus_stats(holdout_corpus),
    "calib_prompt_ids": [p["prompt_id"] for p in calib_prompts],
    "holdout_prompt_ids": [p["prompt_id"] for p in holdout_prompts],
    "leak_audit": audit,
    "template": "instruct",
}
lu.save_json(manifest, "calib/corpus_manifest.json")
lu.save_json(audit, "calib/leak_audit.json")
lu.log_event("01_generate_traces", n_calib=len(calib_corpus),
             n_holdout=len(holdout_corpus),
             total_tokens=manifest["calibration"]["total_tokens"])
print("wrote calib/calib_corpus.jsonl, calib/holdout_corpus.jsonl, calib/corpus_manifest.json")
print("\\ntotal calibration tokens:", manifest["calibration"]["total_tokens"])
print("reasoning fraction       :", manifest["calibration"]["reasoning_fraction"])
"""),
    code(epilogue(
        ["calib/calib_corpus.jsonl", "calib/holdout_corpus.jsonl",
         "calib/corpus_manifest.json", "calib/leak_audit.json"],
        "calib/calib_corpus.jsonl     -> notebooks 02, 03, 04, 05, 06, 07, 08, 09\n"
        "calib/holdout_corpus.jsonl   -> notebooks 02, 07, 09\n"
        "calib/corpus_manifest.json   -> every results table")),
]


NB02_TITLE = "02_uniform_baseline"

NB02 = [
    md("""\
# 02 · Uniform baseline: W4A16 first, then W3A16

**What this notebook does.** Runs the *unweighted* asymmetric second-order quantiser
(`w_t = 1`, full GPTAQ-style reconstruction) at 4 bits — the pipeline-validation
regime — and then at 3 bits, the regime where compact reasoners are reported to
collapse.

**Why W4 before W3.** The pivot is explicit: *"start with W4A16 for pipeline
validation; attempt W3A16 only after the baseline has demonstrable headroom."* If the
W4 checkpoint already loses accuracy, you have a bug, not a research finding. This
notebook therefore measures teacher-forced fidelity after each run and prints a
verdict before you spend 3-bit GPU time.

**Prerequisites**

| Artifact | From |
|---|---|
| `calib/calib_corpus.jsonl` | 01_generate_traces.ipynb |
| `calib/holdout_corpus.jsonl` | 01_generate_traces.ipynb |

**Kaggle settings.** GPU T4 x1. **Runtime** ≈ 25-45 min for W4, similar for W3 —
dominated by the 28 sequential layer solves, not by data loading. These are estimates
from FLOP counts, not measurements: use `DRY_RUN = True` below to time four layers
before committing to the full run.
Note that two 1.5B FP16 models are resident during the fidelity check (≈6 GB).

**Produces** `quant/uniform_w4_*/`, `quant/uniform_w3_*/`, `eval/baseline_w4.json`,
`eval/baseline_w3.json`
"""),
    code("""\
# Prerequisites: the calibration corpus from notebook 01.
from ebswgptaq import logging_utils as lu

calib_path = lu.require("calib/calib_corpus.jsonl", produced_by="01_generate_traces.ipynb")
holdout_path = lu.require("calib/holdout_corpus.jsonl", produced_by="01_generate_traces.ipynb")

import json, time, torch
from ebswgptaq.config import ModelConfig, CalibConfig, QuantConfig, load_config
from ebswgptaq import model_loader as ml, batching as bt, quant_core as qc

mcfg = load_config(f"{CONFIG_DIR}/model.yaml", ModelConfig)
ccfg = load_config(f"{CONFIG_DIR}/calibration.yaml", CalibConfig)
lu.set_seed(ccfg.seed)

calib = lu.read_jsonl(calib_path)
holdout = lu.read_jsonl(holdout_path)
calib_batches = bt.batches_from_records(calib, ccfg.max_seq_len)
holdout_batches = bt.batches_from_records(holdout, ccfg.max_seq_len)
print("calibration batches:", json.dumps(bt.describe_batches(calib_batches), indent=2))
print("holdout batches    :", json.dumps(bt.describe_batches(holdout_batches), indent=2))
print("\\nuniform baseline = every weight exactly 1.0 ->",
      all(b.weights is None for b in calib_batches))
"""),
    md("""\
### Engineering check before the expensive run

The pivot's engineering gate is: *"uniform weights reproduce the unweighted GPTAQ
objective and outputs within numerical tolerance."*

Concretely, passing `w=None` and passing `w=ones` must produce identical statistics
and identical quantised weights. If they differ, the weighting path has a bug that
would corrupt every later comparison — so it is cheaper to catch it on one layer now.
"""),
    code("""\
# Engineering gate: w=None must be identical to w=ones, exactly.
import torch
from ebswgptaq import score_weights as sw

DRY_RUN = False          # True -> quantize only the first 4 layers (a ~2 min timing run)
MAX_LAYERS = 4 if DRY_RUN else None
print("DRY_RUN =", DRY_RUN, "| max_layers =", MAX_LAYERS)

wrapped = ml.load_model(mcfg, verbose=False)

# Calibration sufficiency: H is (d_in x d_in) built from N token columns, so N < d_in
# means a structurally rank-deficient Hessian and a result dominated by the ridge.
max_d = int(max(m.in_features for m in wrapped.linear_modules().values()))
suff = bt.calibration_sufficiency(calib_batches, max_d)
print("\\ncalibration sufficiency:", json.dumps(suff, indent=2))
assert suff["verdict"] != "rank-deficient", suff["action"]

name, mod = next(iter(wrapped.layer_linears(0).items()))
W = mod.weight.detach()

acc_none = qc.WeightedHessian(W.shape[1], "float32", track_cross=False)
acc_ones = qc.WeightedHessian(W.shape[1], "float32", track_cross=False)
for b in calib_batches[:3]:
    x = qc.capture_inputs(wrapped, b, {name: mod})[name]
    acc_none.add(x, None, None)
    acc_ones.add(x, None, torch.ones(x.shape[1]))

print("H identical (w=None vs w=ones):", torch.allclose(acc_none.H, acc_ones.H, atol=1e-4))
print("max abs H difference          :", float((acc_none.H - acc_ones.H).abs().max()))

qcfg_test = load_config(f"{CONFIG_DIR}/quant_w4.yaml", QuantConfig)
r_none = qc.quantize_linear(name, W, acc_none.H, None, qcfg_test)
r_ones = qc.quantize_linear(name, W, acc_ones.H, None, qcfg_test)
print("weights identical             :", torch.equal(r_none.W_hat, r_ones.W_hat))
assert torch.equal(r_none.W_hat, r_ones.W_hat), "uniform-path mismatch: fix before proceeding"
print("\\nENGINEERING GATE PASSED")
del acc_none, acc_ones, r_none, r_ones, wrapped
torch.cuda.empty_cache()
print(lu.gpu_memory_report())
"""),
    md("""\
### Run W4A16 (uniform GPTAQ)

Watching the per-layer output: `cond` is the condition number of the Hessian
`X D Xᵀ` after damping, and `obj` is the layer's reconstruction objective. A
condition number in the 1e3-1e6 range for attention projections is normal; 1e12+
means the ridge damping is too small for this layer.
"""),
    code("""\
def tag_for(qcfg, variant="uniform"):
    return (f"{variant}_w{qcfg.bits}_g{qcfg.group_size}"
            f"_asym{int(qcfg.asym)}_drift{int(qcfg.drift_correction)}")


qcfg_w4 = load_config(f"{CONFIG_DIR}/quant_w4.yaml", QuantConfig)
tag_w4 = tag_for(qcfg_w4)
print("config:", tag_w4)
print(f"  bits={qcfg_w4.bits} group={qcfg_w4.group_size} sym={qcfg_w4.sym} "
      f"asym={qcfg_w4.asym} drift={qcfg_w4.drift_correction} "
      f"act_order={qcfg_w4.act_order} true_sequential={qcfg_w4.true_sequential}")

wrapped = ml.load_model(mcfg, verbose=False)
t0 = time.time()
results_w4 = qc.quantize_model(wrapped, calib_batches, qcfg_w4, verbose=True,
                               max_layers=MAX_LAYERS)
elapsed_w4 = time.time() - t0
print(f"\\nW4 quantization wall clock: {elapsed_w4/60:.1f} min")
if MAX_LAYERS:
    print(f"DRY RUN: {MAX_LAYERS} layers only -> extrapolate x"
          f"{wrapped.n_layers/MAX_LAYERS:.1f} for the full model")
print(json.dumps(qc.summarize_results(results_w4), indent=2))
print(lu.gpu_memory_report())
"""),
    code("""\
# Held-out teacher-forced fidelity. This is the number the whole study is judged on,
# so it is measured immediately -- a broken W4 run should not consume W3 GPU time.
ref = ml.load_model(mcfg, verbose=False)     # FP16 reference, second copy
from ebswgptaq import evaluation as ev

fid_w4 = ev.teacher_forced_fidelity(ref, wrapped, holdout_batches,
                                    chunk_positions=64, keep_per_position=True)
agg = fid_w4["aggregate"]
print("W4A16 fidelity on held-out sequences")
for k, v in agg.items():
    print(f"  {k:14s} {v}")
print("\\nreading: mean_kl is KL(FP16 || quantized) in nats/token on identical prefixes;")
print("         ppl_q is the quantized model's teacher-forced perplexity.")
"""),
    code("""\
# Persist the W4 checkpoint and its metrics.
qc.save_quant_state(results_w4, lu.out(f"quant/{tag_w4}/state.pt"),
                    meta={"tag": tag_w4, "config": qcfg_w4.__dict__,
                          "wall_clock_s": round(elapsed_w4, 1)})
w4_metrics = {
    "tag": tag_w4,
    "config": qcfg_w4.__dict__,
    "calibration": bt.describe_batches(calib_batches),
    "calibration_sufficiency": suff,
    "max_in_features": max_d,
    "holdout": bt.describe_batches(holdout_batches),
    "summary": qc.summarize_results(results_w4),
    "fidelity": agg,
    "per_sequence": fid_w4["per_sequence"],
    "wall_clock_s": round(elapsed_w4, 1),
    "peak_vram_gb": lu.gpu_memory_report().get("total_gb"),
    "dry_run": bool(DRY_RUN),
}
lu.save_json(w4_metrics, f"eval/baseline_w4.json")
lu.log_event("02_uniform_baseline", tag=tag_w4, mean_kl=agg["mean_kl"],
             nll_delta=agg["nll_delta"])
print("wrote", lu.out(f"eval/baseline_w4.json"))

HEADROOM = agg["nll_delta"] > 0.005
print(f"\\nW4 headroom check: nll_delta={agg['nll_delta']:.4f} -> "
      f"{'OK to attempt W3' if HEADROOM else 'W4 looks lossless (or broken); inspect before W3'}")
"""),
    md("""\
### Run W3A16 — the regime under study

Same corpus, same group size, same solver; only the bit-width changes. v2 says to
attempt W3 only once the W4 baseline has demonstrable headroom. If W4 looked
lossless, that is itself informative: the model may be too easy, and you should
record it rather than push forward.

Expect `mean_kl` to jump by roughly an order of magnitude and `ppl_q` to climb. That
gap is the headroom the weighted variants (notebook 06) have to close.
"""),
    code("""\
RUN_W3 = True     # set False to stop after W4

if RUN_W3:
    # the W4 model and its W_hat tensors are still resident (~6 GB); free them first.
    # globals().pop keeps this cell re-runnable.
    for _name in ("wrapped", "results_w4"):
        globals().pop(_name, None)
    torch.cuda.empty_cache()
    print(lu.gpu_memory_report())

    qcfg_w3 = load_config(f"{CONFIG_DIR}/quant_w3.yaml", QuantConfig)
    tag_w3 = tag_for(qcfg_w3)
    print("config:", tag_w3)

    wrapped_w3 = ml.load_model(mcfg, verbose=False)
    t0 = time.time()
    results_w3 = qc.quantize_model(wrapped_w3, calib_batches, qcfg_w3, verbose=True,
                                   max_layers=MAX_LAYERS)
    elapsed_w3 = time.time() - t0
    print(f"\\nW3 quantization wall clock: {elapsed_w3/60:.1f} min")

    fid_w3 = ev.teacher_forced_fidelity(ref, wrapped_w3, holdout_batches,
                                        chunk_positions=64, keep_per_position=True)
    agg_w3 = fid_w3["aggregate"]
    print("\\nW3A16 fidelity on held-out sequences")
    for k, v in agg_w3.items():
        print(f"  {k:14s} {v}")

    qc.save_quant_state(results_w3, lu.out(f"quant/{tag_w3}/state.pt"),
                        meta={"tag": tag_w3, "config": qcfg_w3.__dict__,
                              "wall_clock_s": round(elapsed_w3, 1)})
    lu.save_json({
        "tag": tag_w3, "config": qcfg_w3.__dict__,
        "calibration": bt.describe_batches(calib_batches),
        "calibration_sufficiency": suff,
        "max_in_features": max_d,
        "holdout": bt.describe_batches(holdout_batches),
        "summary": qc.summarize_results(results_w3),
        "fidelity": agg_w3, "per_sequence": fid_w3["per_sequence"],
        "wall_clock_s": round(elapsed_w3, 1),
        "dry_run": bool(DRY_RUN),
    }, "eval/baseline_w3.json")
    lu.log_event("02_uniform_baseline_w3", tag=tag_w3, mean_kl=agg_w3["mean_kl"])

    print("\\n--- W4 vs W3 ---")
    print(f"  {'metric':14s} {'W4':>12s} {'W3':>12s} {'ratio':>8s}")
    for k in ("mean_kl", "mean_jsd", "nll_q", "ppl_q"):
        r4, r3 = agg[k], agg_w3[k]
        print(f"  {k:14s} {r4:12.5f} {r3:12.5f} {r3/max(r4,1e-9):8.2f}x")

    for _name in ("results_w3", "wrapped_w3"):
        globals().pop(_name, None)
    torch.cuda.empty_cache()
    print(lu.gpu_memory_report())
else:
    tag_w3 = None
    print("skipped W3")
"""),
    code(epilogue(
        ["quant/<tag_w4>/state.pt", "eval/baseline_w4.json",
         "quant/<tag_w3>/state.pt", "eval/baseline_w3.json"],
        "quant/uniform_w4_*/state.pt -> notebooks 06 (baseline row), 07 (reference)\n"
        "quant/uniform_w3_*/state.pt -> notebooks 06, 07\n"
        "eval/baseline_w{4,3}.json    -> notebooks 07, 09")),
]


NB03_TITLE = "03_build_probe"

NB03 = [
    md("""\
# 03 · Preliminary quantized probes (RTN and GPTQ)

**What this notebook does.** Builds the cheap, *fixed* quantized models whose output
distribution is compared against FP16 to define the JSD signal.

**Why the probe must be cheap and boring.** The probe answers one question: *where
does quantization perturb the next-token distribution?* It is not the method and it
must not be tuned. If the probe were strong, the signal would measure the probe's
quality rather than quantization sensitivity. Two probes are built because
notebook 08 has to show whether the signal survives swapping them (hypothesis H5).

| Probe | Cost | What it is |
|---|---|---|
| RTN | seconds, no data | round-to-nearest, group-wise, 4-bit |
| GPTQ | ~10 min | uniform-weight GPTQ on the same calibration set (drift off for cost) |

**Prerequisites**

| Artifact | From |
|---|---|
| `calib/calib_corpus.jsonl` | 01_generate_traces.ipynb |

**Kaggle settings.** GPU T4 x1. **Runtime** ≈ 12 min total.
**Produces** `probes/probe_rtn_w4/`, `probes/probe_gptq_w4/`
"""),
    code("""\
from ebswgptaq import logging_utils as lu
calib_path = lu.require("calib/calib_corpus.jsonl", produced_by="01_generate_traces.ipynb")

import json, time, torch
from ebswgptaq.config import ModelConfig, CalibConfig, load_config
from ebswgptaq import model_loader as ml, batching as bt, probe_quantization as pq, evaluation as ev

mcfg = load_config(f"{CONFIG_DIR}/model.yaml", ModelConfig)
ccfg = load_config(f"{CONFIG_DIR}/calibration.yaml", CalibConfig)
lu.set_seed(ccfg.seed)
calib = lu.read_jsonl(calib_path)
batches = bt.batches_from_records(calib, ccfg.max_seq_len)
print(bt.describe_batches(batches))

PROBE_BITS = 4          # probe bit-width; fixed, never tuned
PROBE_GROUP = 128
ref = ml.load_model(mcfg, verbose=False)
"""),
    md("""\
### Probe A — RTN

No Hessian, no calibration data, no compensation: just the grid. This is the probe
with zero degrees of freedom, so any signal derived from it cannot be an artefact of
a solver choice.
"""),
    code("""\
w_rtn = ml.load_model(mcfg, verbose=False)
out_rtn = pq.build_probe(w_rtn, None, method="rtn", bits=PROBE_BITS,
                         group_size=PROBE_GROUP, verbose=True, tag="probe_rtn_w4")
pq.save_probe(out_rtn["results"], out_rtn["meta"], "probes/probe_rtn_w4")
print(json.dumps(out_rtn["meta"], indent=2)[:1200])

fid_rtn = ev.teacher_forced_fidelity(ref, w_rtn, batches, chunk_positions=64,
                                     keep_per_position=False)
print("\\nRTN probe fidelity on the calibration corpus")
for k, v in fid_rtn["aggregate"].items():
    print(f"  {k:14s} {v}")
lu.save_json({"meta": out_rtn["meta"], "fidelity": fid_rtn["aggregate"]},
             "probes/probe_rtn_w4/fidelity.json")
del w_rtn
torch.cuda.empty_cache()
"""),
    md("""\
### Probe B — GPTQ (uniform weights)

Same solver as the real experiments but with `w = 1` and no asymmetric term, so the
probe differs from the RTN probe *only* in whether second-order information was used.

Drift correction is off here: it doubles the cost and the probe is not the method.
That deviation is recorded in the probe metadata and discussed in notebook 08.
"""),
    code("""\
w_gptq = ml.load_model(mcfg, verbose=False)
t0 = time.time()
out_gptq = pq.build_probe(w_gptq, batches, method="gptq", bits=PROBE_BITS,
                         group_size=PROBE_GROUP, verbose=False, tag="probe_gptq_w4")
print(f"GPTQ probe built in {(time.time()-t0)/60:.1f} min")
pq.save_probe(out_gptq["results"], out_gptq["meta"], "probes/probe_gptq_w4")
print(json.dumps({k: v for k, v in out_gptq["meta"].items() if k != "summary"}, indent=2))

fid_gptq = ev.teacher_forced_fidelity(ref, w_gptq, batches, chunk_positions=64,
                                      keep_per_position=False)
print("\\nGPTQ probe fidelity on the calibration corpus")
for k, v in fid_gptq["aggregate"].items():
    print(f"  {k:14s} {v}")
lu.save_json({"meta": out_gptq["meta"], "fidelity": fid_gptq["aggregate"]},
             "probes/probe_gptq_w4/fidelity.json")

print("\\n--- probe comparison (calibration corpus) ---")
print(f"  {'metric':14s} {'RTN':>12s} {'GPTQ':>12s}")
for k in ("mean_kl", "mean_jsd", "nll_q", "ppl_q"):
    print(f"  {k:14s} {fid_rtn['aggregate'][k]:12.5f} {fid_gptq['aggregate'][k]:12.5f}")
"""),
    code(epilogue(
        ["probes/probe_rtn_w4/probe.pt", "probes/probe_rtn_w4/probe_meta.json",
         "probes/probe_rtn_w4/fidelity.json",
         "probes/probe_gptq_w4/probe.pt", "probes/probe_gptq_w4/probe_meta.json",
         "probes/probe_gptq_w4/fidelity.json"],
        "probes/probe_rtn_w4/*  -> notebooks 04 (signal), 08 (probe swap)\n"
        "probes/probe_gptq_w4/* -> notebook 08")),
]


NB04_TITLE = "04_compute_jsd"

NB04 = [
    md("""\
# 04 · Same-prefix JSD, forward KL and entropy per token position

**What this notebook does.** Runs the FP16 model and the probe over the *identical*
teacher-forced sequences and records, for every position with a valid next token:

* `jsd` — symmetric bounded divergence, the signal under test (hypothesis H1)
* `kl` — unbounded forward KL, the heavy-tailed negative control
* `entropy` — FP16 output entropy, the open-loop secondary signal

**The alignment contract, again.** Logit row `t` predicts token `t+1`. Only positions
where token `t+1` is a real token are scored, and the saved arrays carry the position
indices so notebooks 05-08 can map a score back to an activation column without
guessing. Notebook 04 asserts this before spending GPU time.

**Prerequisites**

| Artifact | From |
|---|---|
| `calib/calib_corpus.jsonl` | 01_generate_traces.ipynb |
| `probes/probe_rtn_w4/probe.pt` | 03_build_probe.ipynb |

**Kaggle settings.** GPU T4 x1. **Runtime** ≈ 8-15 min (two models resident, exact
full-vocabulary divergence over ~30k positions).

**Produces** `scores/jsd_rtn_calib.pt`, `scores/scores_rtn_summary.json`
"""),
    code("""\
from ebswgptaq import logging_utils as lu

calib_path = lu.require("calib/calib_corpus.jsonl", produced_by="01_generate_traces.ipynb")
PROBE = "rtn"                       # "rtn" or "gptq"  (notebook 08 uses both)
probe_dir = f"probes/probe_{PROBE}_w4"
lu.require(f"{probe_dir}/probe.pt", produced_by="03_build_probe.ipynb")

import json, time, torch
from ebswgptaq.config import ModelConfig, CalibConfig, load_config
from ebswgptaq import model_loader as ml, batching as bt, probe_quantization as pq
from ebswgptaq import jsd as js

mcfg = load_config(f"{CONFIG_DIR}/model.yaml", ModelConfig)
ccfg = load_config(f"{CONFIG_DIR}/calibration.yaml", CalibConfig)
lu.set_seed(ccfg.seed)
print("probe under test:", probe_dir)
"""),
    md("""\
### Unit test before the GPU run

Three properties must hold or the scores are meaningless: logits row `t` pairs with
token `t+1`; JSD is symmetric and bounded by ln 2; JSD is exactly zero for identical
distributions. Each of these catches a different classic bug (off-by-one, sign,
softmax over the wrong axis).
"""),
    code("""\
print("causal alignment:", js.assert_causal_alignment())
sanity = js.assert_jsd_sanity(v=1024)
for k, v in sanity.items():
    print(f"  {k:24s} {v}")
assert sanity["symmetry_max_abs_diff"] < 1e-6
assert sanity["self_jsd_max"] < 1e-9
assert sanity["within_bound"]
print("\\nJSD implementation verified")
"""),
    code("""\
# Load the reference and the probe, then score the calibration corpus position by
# position. Exact full-vocabulary divergence: no top-k approximation is used for the
# primary scores (the approximation is validated separately on a subset).
ref = ml.load_model(mcfg, verbose=False)
probe = ml.load_model(mcfg, verbose=False)
from ebswgptaq import quant_core as qc
qc.load_quant_state(probe, lu.find(f"{probe_dir}/probe.pt"), apply=True)

calib = lu.read_jsonl(calib_path)
batches = bt.batches_from_records(calib, ccfg.max_seq_len)
print(bt.describe_batches(batches))
print(lu.gpu_memory_report())
"""),
    code("""\
t0 = time.time()
scores = pq.extract_position_scores(ref, probe, batches, chunk=64, topk=256,
                                    with_topk_validation=True, verbose=True)
print(f"\\nscoring wall clock: {time.time()-t0:.1f}s")
print(json.dumps(scores["summary"], indent=2))
print("\\nnote: the primary scores are exact full-vocabulary JSD. The k=256 value above")
print("      is only the approximation error measured on one sequence, reported so the")
print("      approximation can be justified if you ever need it for speed.")
"""),
    md("""\
### Sanity-check the signal before trusting it

Two things must be true for the signal to be usable:

1. **High entropy must correlate with high JSD.** If the model's own uncertainty says
   nothing about where quantization hurts, the open-loop signal (notebook 06's
   `entropy` variant) is dead on arrival — and that is a publishable finding, not a
   failure.
2. **KL must be far more concentrated than JSD.** That heavy tail is exactly the
   failure mode of naive reweighting reported by Shin (94.9% of weight mass on 1% of
   tokens). If KL is not the most concentrated of the three, check the plumbing.
"""),
    code("""\
import torch
all_jsd = torch.cat(scores["jsd"])
all_kl = torch.cat(scores["kl"])
all_ent = torch.cat(scores["entropy"])

def top1_mass(x):
    s = torch.sort(x.flatten(), descending=True).values
    k = max(1, int(round(0.01 * s.numel())))
    return float(s[:k].sum() / s.sum().clamp_min(1e-12))

def ess_of(x):
    w = (x / x.mean().clamp_min(1e-12)).clamp_min(0)
    return float(w.sum() ** 2 / w.pow(2).sum() / w.numel())

print(f"{'signal':10s} {'mean':>10s} {'p90':>10s} {'p99':>10s} {'max':>10s} "
      f"{'mass@1%':>9s} {'ESS/N':>8s}")
for name, x in (("jsd", all_jsd), ("kl", all_kl), ("entropy", all_ent)):
    q = torch.quantile(x, torch.tensor([0.9, 0.99]))
    print(f"{name:10s} {x.mean():10.4f} {q[0]:10.4f} {q[1]:10.4f} {x.max():10.4f} "
          f"{top1_mass(x):9.3f} {ess_of(x):8.3f}")

try:
    from scipy.stats import spearmanr
    rho = spearmanr(all_ent.numpy(), all_jsd.numpy()).statistic
    print(f"\\nspearman(entropy, jsd) = {rho:.4f}")
    print("  (a weak correlation means the open-loop entropy signal is not a proxy "
          "for quantization sensitivity)")
except Exception as e:
    print("scipy unavailable:", e)

print(f"\\ntop-k validation (k={scores['summary']['topk']}): "
      f"mean relative error = {scores['summary']['topk_mean_rel_err']}")
"""),
    code("""\
# Persist every per-position array plus the thresholds notebooks 05 and 07 need.
# Thresholds are computed on CALIBRATION scores only and reused on held-out data --
# that is what keeps the stratification leakage-free.
q90_ent = float(torch.quantile(all_ent, torch.tensor(0.90)))
q90_jsd = float(torch.quantile(all_jsd, torch.tensor(0.90)))
q99_jsd = float(torch.quantile(all_jsd, torch.tensor(0.99)))

payload = {
    "probe": PROBE,
    "jsd": scores["jsd"], "kl": scores["kl"], "entropy": scores["entropy"],
    "positions": scores["positions"], "seq_id": scores["seq_id"],
    "records": [{"prompt_id": r.get("prompt_id"), "source": r.get("source"),
                 "n_tokens": len(r["token_ids"])} for r in calib],
    "thresholds": {"entropy_p90": q90_ent, "jsd_p90": q90_jsd, "jsd_p99": q99_jsd},
    "summary": scores["summary"],
    "probe_meta": lu.load_json(f"{probe_dir}/probe_meta.json",
                               produced_by="03_build_probe.ipynb"),
}
torch.save(payload, lu.out(f"scores/jsd_{PROBE}_calib.pt"))
lu.save_json({k: v for k, v in payload.items()
              if k not in ("jsd", "kl", "entropy", "positions", "seq_id")},
             f"scores/scores_{PROBE}_summary.json")
lu.log_event("04_compute_jsd", probe=PROBE, n_positions=int(all_jsd.numel()),
             jsd_mean=float(all_jsd.mean()))

print("wrote", lu.out(f"scores/jsd_{PROBE}_calib.pt"))
print("thresholds (calibration-only):", payload["thresholds"])
"""),
    code(epilogue(
        ["scores/jsd_rtn_calib.pt", "scores/scores_rtn_summary.json"],
        "scores/jsd_<probe>_calib.pt  -> notebooks 05 (geometry), 06 (weights),\n"
        "                                07 (strata), 08 (probe swap)\n"
        "thresholds inside it         -> notebooks 07, 09 (leakage-free strata)")),
]
