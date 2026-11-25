"""Notebook specs 05-09 (geometry, weighted variants, proxy eval, probe swap, pilot)."""
from nb_common import md, code, epilogue

NB05_TITLE = "05_analyze_geometry"

NB05 = [
    md("""\
# 05 · Activation geometry of high-JSD vs low-JSD positions

**The question (hypothesis H2).** Do positions whose output distribution quantization
perturbs most sit in a *different activation subspace*, or are they the same
directions at a larger scale?

This matters because the whole method is a per-column reweighting of the Gram matrix
`X D Xᵀ`. If high-JSD positions span the same subspace as everything else, weighting
them just rescales directions uniform calibration already sees — and the geometric
story is dead. The pivot asks for exactly this falsification, *before* any weighted
quantization run, which is why this notebook comes before notebook 06.

**The control that decides it.** Each high-JSD group is compared against a *matched
random subset of the same size*, drawn from the same layer. Subspace overlap with the
low-JSD group that is not above the random control means "no distinct geometry".

Reported per (layer, module): top-k subspace overlap (mean cos² of principal angles),
principal angles, effective rank and spectral summary of both groups, and median
activation norms.

**Prerequisites**

| Artifact | From |
|---|---|
| `calib/calib_corpus.jsonl` | 01_generate_traces.ipynb |
| `scores/jsd_rtn_calib.pt` | 04_compute_jsd.ipynb |

**Kaggle settings.** GPU T4 x1. **Runtime** ≈ 10-15 min.
**Produces** `geometry/geometry_rtn.json`, `geometry/geometry_aggregate.json`
"""),
    code("""\
from ebswgptaq import logging_utils as lu
lu.require("calib/calib_corpus.jsonl", produced_by="01_generate_traces.ipynb")
scores_path = lu.require("scores/jsd_rtn_calib.pt", produced_by="04_compute_jsd.ipynb")

import json, time, torch
from ebswgptaq.config import ModelConfig, CalibConfig, load_config
from ebswgptaq import model_loader as ml, batching as bt, activation_capture as ac, geometry as gm

mcfg = load_config(f"{CONFIG_DIR}/model.yaml", ModelConfig)
ccfg = load_config(f"{CONFIG_DIR}/calibration.yaml", CalibConfig)
lu.set_seed(ccfg.seed)

scores = torch.load(scores_path, map_location="cpu", weights_only=False)
calib = lu.read_jsonl("calib/calib_corpus.jsonl")
batches = bt.batches_from_records(calib, ccfg.max_seq_len)

thr = scores["thresholds"]["jsd_p90"]
print("calibration-only threshold jsd_p90 =", round(thr, 6))
print("scored positions:", scores["summary"]["n_positions"])

all_jsd = torch.cat(scores["jsd"])
print(f"positions above threshold: {int((all_jsd >= thr).sum())} "
      f"({float((all_jsd >= thr).float().mean())*100:.1f}%)")
"""),
    md("""\
### Which layers to inspect — decided *before* looking at results

Early / middle / late blocks, and four module types (two attention projections, two
MLP projections). Choosing layers after seeing results is the classic way to
manufacture a geometry story, so the list is fixed here and the aggregated result is
reported alongside the per-layer table.
"""),
    code("""\
wrapped = ml.load_model(mcfg, verbose=False)
n_layers = wrapped.n_layers
LAYERS = sorted({n_layers // 4, n_layers // 2, (3 * n_layers) // 4})
MODULE_FILTER = ["q_proj", "o_proj", "gate_proj", "down_proj"]
K_SUBSPACE = 16
print(f"model: {n_layers} layers | inspecting {LAYERS} | modules {MODULE_FILTER} | k={K_SUBSPACE}")
"""),
    code("""\
# Column alignment note: activations have one column per TOKEN, scores exist only for
# positions with a valid next-token target. concat_target_columns applies the same
# mask that produced the scores, so column i of X always matches score i.
masks = ac.target_masks_per_batch(batches)
n_cols = sum(int(m.sum()) for m in masks)
n_scores = int(torch.cat(scores["jsd"]).numel())
print(f"scored positions: {n_scores} | selected activation columns: {n_cols}")
assert n_cols == n_scores, "alignment mismatch: captured columns != scored positions"

high_all = all_jsd >= thr
low_all = ~high_all
print(f"high group: {int(high_all.sum())} | low group: {int(low_all.sum())}")
"""),
    code("""\
reports = []
t0 = time.time()
for layer_idx in LAYERS:
    print(f"\\nlayer {layer_idx}")
    caps = ac.capture_layer_inputs(wrapped, batches, layer_idx, module_filter=MODULE_FILTER)
    for name, tensors in caps.items():
        X = ac.concat_target_columns(tensors, masks, dtype=torch.float32)
        rep = gm.geometry_report(X, high_all, low_all, k=K_SUBSPACE, seed=ccfg.seed)
        rep.update({"layer": layer_idx, "module": name.split(".")[-1], "name": name})
        reports.append(rep)
        if "error" in rep:
            print(f"  {rep['module']:10s} SKIPPED ({rep['error']})")
            continue
        print(f"  {rep['module']:10s} overlap high-vs-low={rep['high_vs_low']['mean_cos2']:.4f} "
              f"| high-vs-random={rep['high_vs_random']['mean_cos2']:.4f} "
              f"| angle={rep['high_vs_low']['mean_angle_deg']:.1f}deg "
              f"| eff_rank high/low="
              f"{rep['spectrum_high'].get('eff_rank', float('nan')):.1f}/"
              f"{rep['spectrum_low'].get('eff_rank', float('nan')):.1f}")
    del caps
print(f"\\ngeometry sweep: {time.time()-t0:.1f}s")
"""),
    md("""\
### Reading the result

* `overlap high-vs-low` ≈ `overlap high-vs-random` → **no distinct subspace**. The
  high-JSD positions are not geometrically special; H2 fails, and the honest report is
  that the signal is a *scalar* importance rather than a *subspace* phenomenon.
* `overlap high-vs-low` clearly **below** `high-vs-random` → the high-JSD positions
  occupy different directions. This is the signal geometry H2 predicts.
* A large `norm_high_median / norm_low_median` ratio would mean the effect is mostly
  magnitude, which is the "just rescaling" explanation you must rule out.

The aggregate block below is what goes in the paper — not the best-looking layer.
"""),
    code("""\
agg = gm.aggregate_geometry(reports)
print(json.dumps(agg, indent=2))

print("\\nper-layer table")
print(f"  {'layer':>5s} {'module':10s} {'hi-vs-lo':>9s} {'hi-vs-rnd':>10s} "
      f"{'gap':>8s} {'angle':>7s} {'norm ratio':>10s}")
for r in reports:
    if "error" in r:
        continue
    gap = r["high_vs_low"]["mean_cos2"] - r["high_vs_random"]["mean_cos2"]
    nr = r["norm_high_median"] / max(r["norm_low_median"], 1e-12)
    print(f"  {r['layer']:5d} {r['module']:10s} {r['high_vs_low']['mean_cos2']:9.4f} "
          f"{r['high_vs_random']['mean_cos2']:10.4f} {gap:8.4f} "
          f"{r['high_vs_low']['mean_angle_deg']:7.1f} {nr:10.3f}")
"""),
    code("""\
lu.save_json({"thresholds": scores["thresholds"], "k": K_SUBSPACE,
              "layers": LAYERS, "modules": MODULE_FILTER, "reports": reports},
             "geometry/geometry_rtn.json")
lu.save_json({"aggregate": agg, "thresholds": scores["thresholds"]},
             "geometry/geometry_aggregate.json")
lu.log_event("05_analyze_geometry", **{k: v for k, v in agg.items()
                                       if isinstance(v, (int, float))})

verdict = ("distinct subspace" if agg.get("mean_gap_over_control", 0) > 0.05
           else "NO distinct subspace beyond the random control")
print("\\nH2 (geometry) verdict:", verdict)
print("Report this verdict as-is. A negative result here is a legitimate finding.")
"""),
    code(epilogue(
        ["geometry/geometry_rtn.json", "geometry/geometry_aggregate.json"],
        "geometry/geometry_aggregate.json -> results table, notebook 06 discussion")),
]


NB06_TITLE = "06_run_weighted_variants"

NB06 = [
    md("""\
# 06 · The four proxy variants (the decisive experiment)

**What this notebook does.** Quantises the model four times with the *only* thing
changed being the token weight vector `w`:

| Variant | Weight vector | Purpose |
|---|---|---|
| `uniform` | `w_t = 1` | main baseline (GPTAQ-class) |
| `jsd` | `exp(β z_t)`, z from same-prefix JSD | proposed signal |
| `random` | permutation of the `jsd` weights | controls for *nonuniform* weighting |
| `shuffled` | `jsd` weights permuted *within each sequence* | preserves the distribution, breaks position alignment |

If `jsd` beats `uniform` but does not beat `random`/`shuffled`, the gain is a
regularisation side-effect of nonuniform weights, not evidence for the signal. That is
the specificity gate (hypothesis H4) and it is the only reason the controls exist.

**Prerequisites**

| Artifact | From |
|---|---|
| `calib/calib_corpus.jsonl` | 01_generate_traces.ipynb |
| `scores/jsd_rtn_calib.pt` | 04_compute_jsd.ipynb |
| `eval/baseline_w3.json` | 02_uniform_baseline.ipynb |

**Kaggle settings.** GPU T4 x1. **Runtime** ≈ 25-40 min **per variant** (four variants
≈ 2-2.5 h), dominated by 28 sequential layer solves. Set `DRY_RUN = True` below to time
four layers first. Fits one session, but plan the weekly GPU budget.

**Produces** `quant/weighted_<variant>_w{bits}_*/`, `weights/<variant>_weights.pt`
"""),
    code("""\
from ebswgptaq import logging_utils as lu
lu.require("calib/calib_corpus.jsonl", produced_by="01_generate_traces.ipynb")
scores_path = lu.require("scores/jsd_rtn_calib.pt", produced_by="04_compute_jsd.ipynb")
lu.require("eval/baseline_w3.json", produced_by="02_uniform_baseline.ipynb")

import json, time, torch
from ebswgptaq.config import ModelConfig, CalibConfig, QuantConfig, WeightConfig, load_config
from ebswgptaq import model_loader as ml, batching as bt, quant_core as qc
from ebswgptaq import score_weights as sw

mcfg = load_config(f"{CONFIG_DIR}/model.yaml", ModelConfig)
ccfg = load_config(f"{CONFIG_DIR}/calibration.yaml", CalibConfig)
lu.set_seed(ccfg.seed)

# Which bit-width to run the experiment at. W3 is the regime under study; W4 is the
# validation regime. Keep group size and solver settings identical across variants.
QUANT_CONFIG_FILE = f"{CONFIG_DIR}/quant_w3.yaml"
qcfg = load_config(QUANT_CONFIG_FILE, QuantConfig)

scores = torch.load(scores_path, map_location="cpu", weights_only=False)
calib = lu.read_jsonl("calib/calib_corpus.jsonl")
batches = bt.batches_from_records(calib, ccfg.max_seq_len)

print("quantizer:", json.dumps({k: v for k, v in qcfg.__dict__.items()
                                if k in ("bits","group_size","sym","asym","drift_correction",
                                         "act_order","true_sequential","damp_percent")}))
print("batches  :", bt.describe_batches(batches))

# Same sufficiency guard as notebook 02. If the corpus is too small the Hessian is
# rank-deficient and every variant is measuring the ridge, not the weighting.
baseline_metrics = lu.load_json("eval/baseline_w3.json",
                                produced_by="02_uniform_baseline.ipynb")
max_d = int(baseline_metrics.get("max_in_features") or 8960)
suff = bt.calibration_sufficiency(batches, max_d)
print("\\ncalibration sufficiency:", json.dumps(suff, indent=2))
assert suff["verdict"] != "rank-deficient", suff["action"]
"""),
    md("""\
### Step 1 — turn scores into weights, per variant

Two details that decide whether the comparison is fair:

1. **Scores exist only for positions with a target** (`t` where `t+1` is real), while
   the Hessian accumulates over *all* tokens. Unscored positions get the neutral weight
   `1.0`; they are never silently dropped, which would change the effective dataset
   between variants.
2. **Every weight vector is normalised to mean 1** over the pool, so the ridge damping
   keeps the same meaning across variants (v2 §9.3).
"""),
    code("""\
SIGNAL_SOURCE = {"jsd": "jsd", "entropy": "entropy", "kl": "kl"}

def weights_for_variant(spec, seed=0):
    \"\"\"Return one mean-1 weight vector per calibration sequence.\"\"\"
    wcfg = WeightConfig(
        signal=spec["signal"],
        probe=spec.get("probe", "rtn"),
        clip_quantile=spec.get("clip_quantile", 0.99),
        standardize=spec.get("standardize", True),
        normalize=spec.get("normalize", "global"),
        beta=spec.get("beta"),
        ess_floor=spec.get("ess_floor"),
        sink_mask=spec.get("sink_mask", True),
        shuffle_within_sequence=False,
        seed=seed,
    )
    out = []
    for i, b in enumerate(batches):
        if wcfg.signal == "none":
            out.append(torch.ones(b.length))
            continue
        key = SIGNAL_SOURCE.get(wcfg.signal, "jsd")
        s = scores[key][i] if wcfg.signal != "random" else scores["jsd"][i]
        # expand scored positions back to a full-length vector, NaN elsewhere
        full = torch.full((b.length,), float("nan"))
        pos = bt.target_positions(b.valid_mask)
        full[pos] = s.to(torch.float32)
        w = sw.build_weights(full, torch.arange(b.length), b.valid_mask, wcfg)["weights"]
        if wcfg.signal == "random":
            w = sw.random_weights_like(w, seed=seed + i)
        out.append(w)
    return out


experiments = __import__("ebswgptaq.config", fromlist=["load_yaml"]).load_yaml(
    f"{CONFIG_DIR}/experiments.yaml")
VARIANTS = ["uniform", "jsd", "random", "shuffled"]     # v2 §10 step 7
EXTRA_VARIANTS = ["entropy", "kl", "jsd_ess50"]          # run later if budget allows

print("variants to run:", VARIANTS)
for name in VARIANTS + EXTRA_VARIANTS:
    print(f"  {name:10s} {experiments['variants'][name]}")
"""),
    code("""\
# Build all weight vectors once and inspect their distributions *before* quantising.
# If the controls do not have the same concentration as the signal, the comparison is
# meaningless -- so this table is the honesty check for the whole notebook.
variant_weights = {}
for name in VARIANTS + EXTRA_VARIANTS:
    spec = experiments["variants"][name]
    variant_weights[name] = weights_for_variant(spec, seed=ccfg.seed)

print(f"  {'variant':10s} {'ESS/N':>8s} {'max':>8s} {'median':>8s} {'mass@1%':>9s} {'mass@10%':>9s}")
for name, wl in variant_weights.items():
    flat = torch.cat(wl)
    st = sw.concentration_stats(flat)
    print(f"  {name:10s} {st['ess_fraction']:8.3f} {st['max']:8.3f} {st['median']:8.4f} "
          f"{st['mass_top1pct']:9.4f} {st['mass_top10pct']:9.4f}")

# the controls must reproduce the signal's concentration, not its alignment
sig = sw.concentration_stats(torch.cat(variant_weights["jsd"]))
rnd = sw.concentration_stats(torch.cat(variant_weights["random"]))
shf = sw.concentration_stats(torch.cat(variant_weights["shuffled"]))
print(f"\\ncontrol check: ESS jsd={sig['ess_fraction']:.3f} random={rnd['ess_fraction']:.3f} "
      f"shuffled={shf['ess_fraction']:.3f}")
assert abs(sig["ess_fraction"] - rnd["ess_fraction"]) < 0.05, \\
    "random control has a different concentration: fix before quantising"
"""),
    code("""\
# Correlation between weight and score is the *alignment* the controls are supposed to
# destroy: Spearman near 0 for random/shuffled, clearly positive for jsd.
try:
    from scipy.stats import spearmanr

    ref_scores = torch.cat(scores["jsd"]).numpy()
    assert ref_scores.size == sum(int(bt.target_positions(b.valid_mask).numel())
                                 for b in batches), "score/batch count mismatch"
    for name in ("jsd", "random", "shuffled", "entropy"):
        parts = []
        for i, b in enumerate(batches):
            pos = bt.target_positions(b.valid_mask)
            parts.append(variant_weights[name][i][pos])
        rho = spearmanr(ref_scores, torch.cat(parts).numpy()).statistic
        print(f"  spearman(weight, JSD) {name:10s} = {rho:+.4f}")
    print("\\n  expected: jsd ~ +0.7 or higher, entropy lower but positive,")
    print("            random/shuffled ~ 0. Anything else means the control is broken.")
except Exception as e:
    print("scipy unavailable:", e)
"""),
    md("""\
### Step 2 — quantise each variant

Identical corpus, identical quantizer settings, identical seeds. The only difference
is `b.weights`, which the accumulator turns into `H_w = X D Xᵀ` and `C_w = (X̃−X) D Xᵀ`.

Watch `rel_fro_err` and the objective between variants: they should move a little, not
wildly. A large objective improvement that does not show up in held-out fidelity
(notebook 07) is the classic sign that the objective was optimised for the calibration
set rather than for the model.
"""),
    code("""\
def tag_for(qcfg, variant):
    return f"weighted_{variant}_w{qcfg.bits}_g{qcfg.group_size}_asym{int(qcfg.asym)}"


RUN = VARIANTS          # extend to VARIANTS + EXTRA_VARIANTS for the full sweep
DRY_RUN = False         # True -> quantize only the first 4 layers, to time the pipeline
MAX_LAYERS = 4 if DRY_RUN else None
print("RUN =", RUN, "| DRY_RUN =", DRY_RUN)
summary_rows = []
wall = {}

for name in RUN:
    print("=" * 78)
    print(f"variant: {name}")
    print("=" * 78)
    for i, b in enumerate(batches):
        b.weights = variant_weights[name][i]

    wrapped = ml.load_model(mcfg, verbose=False)
    t0 = time.time()
    results = qc.quantize_model(wrapped, batches, qcfg, verbose=False,
                                max_layers=MAX_LAYERS)
    elapsed = time.time() - t0
    wall[name] = elapsed

    tag = tag_for(qcfg, name)
    qc.save_quant_state(results, lu.out(f"quant/{tag}/state.pt"),
                        meta={"tag": tag, "variant": name, "config": qcfg.__dict__,
                              "weight_config": experiments["variants"][name],
                              "wall_clock_s": round(elapsed, 1)})
    torch.save({"weights": variant_weights[name],
                "variant": name,
                "stats": sw.concentration_stats(torch.cat(variant_weights[name]))},
               lu.out(f"weights/{name}_weights.pt"))

    s = qc.summarize_results(results)
    row = {"variant": name, "tag": tag, "wall_clock_s": round(elapsed, 1),
           "total_objective": s["total_objective"], "median_cond": s["median_cond"],
           "weight_stats": sw.concentration_stats(torch.cat(variant_weights[name]))}
    summary_rows.append(row)
    print(f"  {name}: {elapsed/60:.1f} min  objective={s['total_objective']:.4e}  "
          f"ESS/N={row['weight_stats']['ess_fraction']:.3f}")
    del wrapped, results
    torch.cuda.empty_cache()

lu.save_json(summary_rows, "quant/variant_summary.json")
print("\\nwrote", lu.out("quant/variant_summary.json"))
"""),
    code("""\
print(f"  {'variant':10s} {'minutes':>8s} {'objective':>14s} {'ESS/N':>8s} {'mass@1%':>9s}")
base_obj = next(r for r in summary_rows if r["variant"] == "uniform")["total_objective"]
for r in summary_rows:
    rel = 100 * (r["total_objective"] - base_obj) / max(abs(base_obj), 1e-12)
    print(f"  {r['variant']:10s} {r['wall_clock_s']/60:8.1f} {r['total_objective']:14.4e} "
          f"{r['weight_stats']['ess_fraction']:8.3f} {r['weight_stats']['mass_top1pct']:9.4f}"
          f"   ({rel:+.1f}% vs uniform)")

print("\\nNote: a lower calibration objective does NOT imply better held-out fidelity.")
print("That question is decided in notebook 07, on data no variant was calibrated on.")
"""),
    code(epilogue(
        ["quant/weighted_uniform_w*/state.pt", "quant/weighted_jsd_w*/state.pt",
         "quant/weighted_random_w*/state.pt", "quant/weighted_shuffled_w*/state.pt",
         "weights/*_weights.pt", "quant/variant_summary.json"],
        "quant/weighted_<variant>_*/state.pt -> notebooks 07, 09\n"
        "weights/<variant>_weights.pt        -> notebooks 07, 08\n"
        "quant/variant_summary.json          -> results table")),
]


NB07_TITLE = "07_evaluate_proxy"

NB07 = [
    md("""\
# 07 · Proxy evaluation — held-out teacher-forced fidelity and the go/no-go decision

**What this notebook does.** Loads each variant checkpoint and measures fidelity
against FP16 on the **held-out** corpus (prompts no variant ever saw), then applies the
pivot's pre-registered gates.

**Metrics**

| Metric | Meaning |
|---|---|
| `mean_kl` | KL(FP16 ‖ variant) in nats/token on identical prefixes — the primary number |
| `mean_jsd` | same, but bounded and symmetric (robust to single-position blowups) |
| `nll_q` / `ppl_q` | teacher-forced negative log-likelihood / perplexity of the variant |
| `nll_delta` | how much worse the variant is than FP16 |
| stratified KL | high-entropy forks vs everything else, using calibration-only thresholds |

**Gates (pre-registered in `configs/experiments.yaml`)**

* **Specificity**: `jsd` must beat `uniform`, `random` *and* `shuffled` on held-out
  `mean_kl`, with a paired 95% bootstrap CI excluding zero.
* **Robustness**: the ranking must survive the probe swap (notebook 08).
* **Go**: only if both pass do you spend GPU time on downstream benchmarks.

**Prerequisites**

| Artifact | From |
|---|---|
| `calib/holdout_corpus.jsonl` | 01_generate_traces.ipynb |
| `quant/weighted_*/state.pt` | 06_run_weighted_variants.ipynb |
| `scores/jsd_rtn_calib.pt` | 04_compute_jsd.ipynb (thresholds only) |

**Kaggle settings.** GPU T4 x1. **Runtime** ≈ 5-8 min per variant (one variant loaded
at a time, plus the FP16 reference).
**Produces** `eval/proxy_<variant>.json`, `eval/proxy_summary.json`,
`eval/specificity_gate.json`
"""),
    code("""\
from ebswgptaq import logging_utils as lu
lu.require("calib/holdout_corpus.jsonl", produced_by="01_generate_traces.ipynb")
lu.require("quant/variant_summary.json", produced_by="06_run_weighted_variants.ipynb")
scores_path = lu.require("scores/jsd_rtn_calib.pt", produced_by="04_compute_jsd.ipynb")

import glob, json, time, torch
from ebswgptaq.config import ModelConfig, CalibConfig, load_config
from ebswgptaq import model_loader as ml, batching as bt, quant_core as qc
from ebswgptaq import evaluation as ev, statistics as st

mcfg = load_config(f"{CONFIG_DIR}/model.yaml", ModelConfig)
ccfg = load_config(f"{CONFIG_DIR}/calibration.yaml", CalibConfig)
lu.set_seed(ccfg.seed)

holdout = lu.read_jsonl("calib/holdout_corpus.jsonl")
holdout_batches = bt.batches_from_records(holdout, ccfg.max_seq_len)
print(bt.describe_batches(holdout_batches))

scores = torch.load(scores_path, map_location="cpu", weights_only=False)
thresholds = scores["thresholds"]
print("calibration-only strata thresholds:", thresholds)
"""),
    code("""\
variant_summary = lu.load_json("quant/variant_summary.json",
                               produced_by="06_run_weighted_variants.ipynb")
tags = {row["variant"]: row["tag"] for row in variant_summary}
print("variants found:", json.dumps(tags, indent=2))

ref = ml.load_model(mcfg, verbose=False)
results_by_variant = {}

for variant, tag in tags.items():
    print("=" * 78)
    print(f"{variant}  ({tag})")
    print("=" * 78)
    state_path = lu.require(f"quant/{tag}/state.pt",
                            produced_by="06_run_weighted_variants.ipynb")
    model = ml.load_model(mcfg, verbose=False)
    qc.load_quant_state(model, state_path, apply=True)

    t0 = time.time()
    fid = ev.teacher_forced_fidelity(ref, model, holdout_batches,
                                     chunk_positions=64, keep_per_position=True)
    agg = fid["aggregate"]
    strata = ev.strata_report(fid["arrays"], thresholds)
    out = {"variant": variant, "tag": tag, "aggregate": agg, "strata": strata,
           "per_sequence": fid["per_sequence"], "wall_clock_s": round(time.time()-t0, 1)}
    lu.save_json(out, f"eval/proxy_{variant}.json")
    results_by_variant[variant] = out

    print(f"  mean_kl={agg['mean_kl']:.6f}  mean_jsd={agg['mean_jsd']:.6f}  "
          f"ppl_q={agg['ppl_q']:.4f}  nll_delta={agg['nll_delta']:+.5f}")
    if "kl_ratio_high_over_low" in strata:
        print(f"  high-entropy KL / low-entropy KL = "
              f"{strata['kl_ratio_high_over_low']:.3f}")
    del model
    torch.cuda.empty_cache()
"""),
    md("""\
### Statistical comparison

Everything is paired: the same held-out sequences, the same tokens. The resampling
unit is the *sequence*, not the token — token-level bootstrap would give absurdly
narrow intervals because tokens within a sequence are strongly dependent.

A CI that excludes zero is the bar. A point estimate that looks better but has a CI
straddling zero is not evidence.
"""),
    code("""\
per_seq = {v: [row["mean_kl"] for row in r["per_sequence"]]
           for v, r in results_by_variant.items()}
baseline = "uniform" if "uniform" in per_seq else next(iter(per_seq))

print(f"paired bootstrap of mean_kl vs '{baseline}'  (negative delta = better)\")
print(f"  {'variant':12s} {'delta':>12s} {'ci_lo':>12s} {'ci_hi':>12s} {'excludes 0':>11s}")
boot = {}
for variant, values in per_seq.items():
    if variant == baseline:
        continue
    deltas = [v - b for v, b in zip(values, per_seq[baseline])]
    ci = st.paired_bootstrap_ci(deltas, n_resamples=10000, seed=ccfg.seed)
    boot[variant] = ci
    print(f"  {variant:12s} {ci['mean']:12.6f} {ci['ci_lo']:12.6f} {ci['ci_hi']:12.6f} "
          f"{str(ci['excludes_zero']):>11s}")

lu.save_json(boot, "eval/proxy_bootstrap.json")
"""),
    md("""\
### The gates

Fill this in honestly. If `jsd` does not beat `random` and `shuffled`, the specificity
gate fails and the correct action is to stop and write up the negative result — which
the pivot explicitly treats as a legitimate outcome.
"""),
    code("""\
def metric(variant, key="mean_kl"):
    return results_by_variant[variant]["aggregate"][key]

gates = {}
have = set(results_by_variant)

if {"jsd", "uniform"} <= have:
    jsd_kl = metric("jsd")
    uni_kl = metric("uniform")
    gaps = {"vs_uniform": (uni_kl - jsd_kl) / uni_kl}
    sig_ok = jsd_kl < uni_kl and boot.get("jsd", {}).get("excludes_zero", False) \\
        and boot.get("jsd", {}).get("mean", 0) < 0
    for ctrl in ("random", "shuffled"):
        if ctrl in have:
            gaps[f"vs_{ctrl}"] = (metric(ctrl) - jsd_kl) / metric(ctrl)
    ctrl_ok = all(jsd_kl < metric(c) for c in ("random", "shuffled") if c in have)
    gates["specificity"] = {
        "jsd_beats_uniform": bool(jsd_kl < uni_kl),
        "jsd_beats_controls": bool(ctrl_ok),
        "paired_ci_excludes_zero": bool(boot.get("jsd", {}).get("excludes_zero", False)),
        "relative_gaps": {k: round(v, 5) for k, v in gaps.items()},
        "pass": bool(sig_ok and ctrl_ok),
    }

print(json.dumps(gates, indent=2))

if gates.get("specificity", {}).get("pass"):
    print("\\nSPECIFICITY GATE: PASS -> proceed to probe swap (08), then the "
          "downstream pilot (09).")
else:
    print("\\nSPECIFICITY GATE: FAIL / INCONCLUSIVE")
    print("  The honest action is to record this and NOT claim the method works.")
    print("  Before concluding, check: (a) is the W3 baseline actually collapsing?")
    print("  (b) did you run the probe swap (notebook 08)? (c) is the holdout large "
          "enough to resolve the effect?")

lu.save_json(gates, "eval/specificity_gate.json")
"""),
    code("""\
print(f"  {'variant':12s} {'mean_kl':>11s} {'mean_jsd':>11s} {'ppl_q':>9s} "
      f"{'nll_delta':>11s} {'hi-ent KL':>10s}")
for v, r in results_by_variant.items():
    a = r["aggregate"]
    hi = r["strata"].get("high_entropy", {}).get("mean_kl")
    hi_s = f"{hi:.5f}" if hi is not None else "n/a"
    print(f"  {v:12s} {a['mean_kl']:11.6f} {a['mean_jsd']:11.6f} {a['ppl_q']:9.4f} "
          f"{a['nll_delta']:+11.5f} {hi_s:>10s}")
"""),
    code(epilogue(
        ["eval/proxy_uniform.json", "eval/proxy_jsd.json", "eval/proxy_random.json",
         "eval/proxy_shuffled.json", "eval/proxy_bootstrap.json",
         "eval/specificity_gate.json"],
        "eval/specificity_gate.json -> the go/no-go decision recorded in the results table\n"
        "eval/proxy_*.json          -> notebooks 08, 09 and the paper figures")),
]


NB08_TITLE = "08_probe_swap"

NB08 = [
    md("""\
# 08 · Probe-swap robustness (hypothesis H5)

**The question.** Is the JSD signal a property of *quantization sensitivity*, or a
property of one particular probe model?

If the RTN probe and a GPTQ probe rank token positions completely differently, then any
gain attributed to "the signal" is really "the signal from this probe", and the claim
has to be narrowed accordingly. That is the point of this notebook — characterising the
dependence is a result either way.

**Measurements**

| Metric | What it tells you |
|---|---|
| Spearman / Kendall between RTN-JSD and GPTQ-JSD | global rank agreement |
| Top-k overlap (k = 1%, 5%) | whether the *positions that get reweighted* agree |
| Optional: quantise the `jsd` variant with GPTQ-probe weights | does the benefit transfer? |

**Prerequisites**

| Artifact | From |
|---|---|
| `probes/probe_rtn_w4/`, `probes/probe_gptq_w4/` | 03_build_probe.ipynb |
| `scores/jsd_rtn_calib.pt` | 04_compute_jsd.ipynb |
| `quant/variant_summary.json` | 06_run_weighted_variants.ipynb |

**Kaggle settings.** GPU T4 x1. **Runtime** ≈ 10 min for the GPTQ-probe scoring,
+15 min if you run the optional transfer quantisation.
**Produces** `scores/jsd_gptq_calib.pt`, `eval/probe_swap.json`
"""),
    code("""\
from ebswgptaq import logging_utils as lu
lu.require("scores/jsd_rtn_calib.pt", produced_by="04_compute_jsd.ipynb")
lu.require("probes/probe_gptq_w4/probe.pt", produced_by="03_build_probe.ipynb")

import json, time, torch
from ebswgptaq.config import ModelConfig, CalibConfig, load_config
from ebswgptaq import model_loader as ml, batching as bt, probe_quantization as pq
from ebswgptaq import statistics as st

mcfg = load_config(f"{CONFIG_DIR}/model.yaml", ModelConfig)
ccfg = load_config(f"{CONFIG_DIR}/calibration.yaml", CalibConfig)
lu.set_seed(ccfg.seed)

calib = lu.read_jsonl("calib/calib_corpus.jsonl")
batches = bt.batches_from_records(calib, ccfg.max_seq_len)
scores_rtn = torch.load(lu.find("scores/jsd_rtn_calib.pt"), map_location="cpu",
                        weights_only=False)
print("RTN scores:", scores_rtn["summary"]["n_positions"], "positions")
"""),
    code("""\
# Score the same positions with the GPTQ probe. The probe is applied to a freshly
# loaded FP16 model so the two score sets come from identical references.
ref = ml.load_model(mcfg, verbose=False)
probe = ml.load_model(mcfg, verbose=False)
from ebswgptaq import quant_core as qc
qc.load_quant_state(probe, lu.find("probes/probe_gptq_w4/probe.pt"), apply=True)

t0 = time.time()
scores_gptq = pq.extract_position_scores(ref, probe, batches, chunk=64, topk=0,
                                         verbose=True)
print(f"scoring wall clock: {time.time()-t0:.1f}s")
print(json.dumps(scores_gptq["summary"]["jsd"], indent=2))
"""),
    code("""\
payload = {
    "probe": "gptq",
    "jsd": scores_gptq["jsd"], "kl": scores_gptq["kl"],
    "entropy": scores_gptq["entropy"],
    "positions": scores_gptq["positions"], "seq_id": scores_gptq["seq_id"],
    "thresholds": scores_rtn["thresholds"],     # keep calibration thresholds frozen
    "summary": scores_gptq["summary"],
}
torch.save(payload, lu.out("scores/jsd_gptq_calib.pt"))
lu.save_json({k: v for k, v in payload.items()
              if k not in ("jsd", "kl", "entropy", "positions", "seq_id")},
             "scores/scores_gptq_summary.json")
print("wrote", lu.out("scores/jsd_gptq_calib.pt"))
"""),
    md("""\
### Rank agreement between probes

Read this carefully. High Spearman but low top-k overlap is a common and important
pattern: the two probes agree on the broad ordering of "which positions are boring",
but disagree on *which few positions are the most sensitive*. Since the weighting only
matters at the top of the distribution, top-k overlap is the metric to trust.
"""),
    code("""\
a = torch.cat(scores_rtn["jsd"])
b = torch.cat(scores_gptq["jsd"])
assert a.numel() == b.numel(), "score length mismatch between probes"

rc = st.rank_correlation(a.tolist(), b.tolist())
print("global rank agreement")
for k, v in rc.items():
    print(f"  {k:12s} {v}")

ov = {}
n = a.numel()
for frac in (0.01, 0.05, 0.10):
    k = max(1, int(round(frac * n)))
    o = st.topk_overlap(a.tolist(), b.tolist(), k)
    ov[f"top{int(frac*100)}pct"] = o
    print(f"\\ntop-{int(frac*100)}% overlap (k={k})")
    for kk, vv in o.items():
        print(f"  {kk:10s} {vv}")

summary = {"rank_correlation": rc, "topk_overlap": ov,
           "rtn_jsd_mean": float(a.mean()), "gptq_jsd_mean": float(b.mean())}
lu.save_json(summary, "eval/probe_swap.json")
"""),
    md("""\
### Optional — does the benefit transfer across probes?

This re-quantises the `jsd` variant using weights derived from the **GPTQ** probe. If
the held-out fidelity gain survives, the signal is robust to the probe choice. If it
does not, the claim must be narrowed to "with an RTN probe".

Cost is one more full quantization of the model — run it only if the specificity gate
in notebook 07 passed, otherwise there is nothing to transfer.
"""),
    code("""\
RUN_TRANSFER = False      # set True after notebook 07's gate passes

if RUN_TRANSFER:
    from ebswgptaq.config import QuantConfig, WeightConfig, load_config
    from ebswgptaq import quant_core as qc2, score_weights as sw, evaluation as ev

    qcfg = load_config(f"{CONFIG_DIR}/quant_w3.yaml", QuantConfig)
    wcfg = WeightConfig(signal="jsd", probe="gptq", clip_quantile=0.99,
                        standardize=True, beta=1.0, sink_mask=True, seed=ccfg.seed)

    wl = []
    for i, b in enumerate(batches):
        full = torch.full((b.length,), float("nan"))
        pos = bt.target_positions(b.valid_mask)
        full[pos] = scores_gptq["jsd"][i].to(torch.float32)
        wl.append(sw.build_weights(full, torch.arange(b.length), b.valid_mask, wcfg)["weights"])
        b.weights = wl[-1]
    print("gptq-probe weight ESS/N:", sw.concentration_stats(torch.cat(wl))["ess_fraction"])

    model = ml.load_model(mcfg, verbose=False)
    t0 = time.time()
    results = qc2.quantize_model(model, batches, qcfg, verbose=False)
    tag = f"weighted_jsd_gptqprobe_w{qcfg.bits}_g{qcfg.group_size}"
    qc2.save_quant_state(results, lu.out(f"quant/{tag}/state.pt"),
                         meta={"tag": tag, "variant": "jsd_gptqprobe"})
    print(f"quantized in {(time.time()-t0)/60:.1f} min -> {tag}")

    holdout = lu.read_jsonl("calib/holdout_corpus.jsonl")
    hb = bt.batches_from_records(holdout, ccfg.max_seq_len)
    fid = ev.teacher_forced_fidelity(ref, model, hb, chunk_positions=64,
                                     keep_per_position=False)
    print("held-out fidelity:", json.dumps(fid["aggregate"], indent=2))
    lu.save_json({"variant": "jsd_gptqprobe", "tag": tag,
                  "aggregate": fid["aggregate"]}, "eval/proxy_jsd_gptqprobe.json")
else:
    print("transfer run skipped (RUN_TRANSFER=False)")
"""),
    code("""\
print("INTERPRETATION")
print("-" * 70)
rho = rc.get("spearman")
print(f"Spearman(RTN-JSD, GPTQ-JSD) = {rho}")
if rho is not None and rho > 0.7:
    print("  -> the ranking is largely probe-independent; the signal is a property of")
    print("     quantization sensitivity, not of one probe. Report the RTN result and")
    print("     note the GPTQ agreement as robustness evidence.")
elif rho is not None and rho > 0.4:
    print("  -> moderate agreement. Report both probes and narrow the claim to")
    print("     'with a preliminary probe of type X'; check top-k overlap before")
    print("     claiming the same positions are targeted.")
else:
    print("  -> weak agreement: the signal is probe-specific. That is a legitimate")
    print("     finding; state it explicitly rather than reporting only the better probe.")
print()
print("check top-1% overlap above: even with high Spearman, agreement on the few")
print("most-sensitive positions may be poor, and those are the ones that get weight.")
"""),
    code(epilogue(
        ["scores/jsd_gptq_calib.pt", "scores/scores_gptq_summary.json",
         "eval/probe_swap.json"],
        "eval/probe_swap.json     -> robustness section of the results table\n"
        "scores/jsd_gptq_calib.pt -> optional transfer run / later ablations")),
]


NB09_TITLE = "09_downstream_pilot"

NB09 = [
    md("""\
# 09 · Downstream pilot — generation-based reasoning metrics

**Only run this if the specificity gate in notebook 07 passed.** The pivot is explicit:
no broad benchmark campaign until the proxy gate clears. This notebook is a *pilot*,
not the evaluation suite — it exists to check that a proxy improvement is visible in
actual reasoning, on a small fixed subset, before committing GPU hours.

**Metrics (the four from the study plan)**

| Metric | Definition |
|---|---|
| Acc (pass@1) | exact match against the gold `\\boxed{}` answer |
| Mean CoT length | mean generated tokens per problem |
| **TPCA** | total generated tokens ÷ number of correctly solved problems |
| Truncation rate | fraction hitting `max_new_tokens` |
| Over-thinking rate | correct answer appears in the trace but the final answer is wrong |

TPCA is the deployment-relevant one: it charges a method for both the accuracy it
loses and the extra thinking it spends recovering it.

**Prerequisites**

| Artifact | From |
|---|---|
| `quant/weighted_*/state.pt` | 06_run_weighted_variants.ipynb |
| `eval/specificity_gate.json` | 07_evaluate_proxy.ipynb |
| `calib/holdout_corpus.jsonl` | 01_generate_traces.ipynb |

**Kaggle settings.** GPU T4 x1. **Runtime** ≈ 25-40 min for 20 problems × 2 variants at
2048 new tokens. Raise `N_PROBLEMS` only after the pilot is stable.

**Produces** `eval/downstream_<variant>.json`, `eval/downstream_summary.json`
"""),
    code("""\
from ebswgptaq import logging_utils as lu
gate = lu.require("eval/specificity_gate.json", produced_by="07_evaluate_proxy.ipynb")
lu.require("quant/variant_summary.json", produced_by="06_run_weighted_variants.ipynb")

import json, time, torch
gate_data = lu.load_json(gate)
print("specificity gate:", json.dumps(gate_data, indent=2)[:800])
if not gate_data.get("specificity", {}).get("pass", False):
    print("\\nWARNING: the specificity gate did not pass. This pilot is exploratory only;")
    print("         do not present its result as confirming the method.")
"""),
    code("""\
from ebswgptaq.config import ModelConfig, CalibConfig, load_config
from ebswgptaq import model_loader as ml, quant_core as qc, evaluation as ev

mcfg = load_config(f"{CONFIG_DIR}/model.yaml", ModelConfig)
ccfg = load_config(f"{CONFIG_DIR}/calibration.yaml", CalibConfig)
lu.set_seed(ccfg.seed)

N_PROBLEMS = 20          # pilot size; keep fixed across variants
MAX_NEW = 2048           # tokens per generation (full runs use 4096-16384)
VARIANTS_TO_RUN = ["uniform", "jsd"]

summary = lu.load_json("quant/variant_summary.json",
                       produced_by="06_run_weighted_variants.ipynb")
tags = {row["variant"]: row["tag"] for row in summary}
print("running:", VARIANTS_TO_RUN, "| tags:", {v: tags.get(v) for v in VARIANTS_TO_RUN})
"""),
    code("""\
# Fixed evaluation subset, taken from the GSM8K *test* split (never a calibration
# source). Deterministic sample so every variant answers the same problems.
from datasets import load_dataset
import random

gsm_test = load_dataset("openai/gsm8k", "main", split="test")
rng = random.Random(ccfg.seed)
idx = rng.sample(range(len(gsm_test)), N_PROBLEMS)
problems = [{"prompt_id": f"gsm8k-test:{i}",
             "prompt": gsm_test[i]["question"],
             "answer": gsm_test[i]["answer"].split("####")[-1].strip()}
            for i in sorted(idx)]
print(f"{len(problems)} problems from GSM8K test")
print("\\nexample:")
print(" ", problems[0]["prompt"][:180].replace("\\n", " "))
print("  gold:", problems[0]["answer"])
"""),
    code("""\
results = {}
for variant in VARIANTS_TO_RUN:
    tag = tags[variant]
    path = lu.require(f"quant/{tag}/state.pt", produced_by="06_run_weighted_variants.ipynb")
    model = ml.load_model(mcfg, verbose=False)
    qc.load_quant_state(model, path, apply=True)

    t0 = time.time()
    out = ev.generate_and_score(model, problems, max_new_tokens=MAX_NEW,
                                do_sample=False, template="instruct")
    elapsed = time.time() - t0
    out["variant"] = variant
    out["tag"] = tag
    out["wall_clock_s"] = round(elapsed, 1)
    results[variant] = out
    lu.save_json({k: v for k, v in out.items() if k != "rows"} | {"n_rows": len(out["rows"])},
                 f"eval/downstream_{variant}.json")
    print(f"\\n{variant}: acc={out['accuracy']:.3f} ({out['n_correct']}/{out['n']})  "
          f"mean_tokens={out['mean_cot_tokens']:.0f}  TPCA={out['tpca']:.0f}  "
          f"trunc={out['truncation_rate']:.3f}  overthink={out['overthinking_rate']:.3f}  "
          f"[{elapsed/60:.1f} min]")
    del model
    torch.cuda.empty_cache()
"""),
    code("""\
print(f"  {'variant':10s} {'acc':>7s} {'mean_tok':>9s} {'TPCA':>8s} {'trunc':>7s} {'overthink':>10s}")
for v, r in results.items():
    print(f"  {v:10s} {r['accuracy']:7.3f} {r['mean_cot_tokens']:9.0f} {r['tpca']:8.0f} "
          f"{r['truncation_rate']:7.3f} {r['overthinking_rate']:10.3f}")

if "uniform" in results and "jsd" in results:
    u, j = results["uniform"], results["jsd"]
    print("\\npaired comparison (same problems):")
    from ebswgptaq import statistics as st
    tok_delta = [a - b for a, b in zip(j["per_item_tokens"], u["per_item_tokens"])]
    ci = st.paired_bootstrap_ci(tok_delta, n_resamples=10000, seed=ccfg.seed)
    print(f"  token delta (jsd - uniform): {ci['mean']:+.1f} tokens/problem "
          f"[{ci['ci_lo']:+.1f}, {ci['ci_hi']:+.1f}]")
    mc = st.mcnemar_exact(u["per_item_correct"], j["per_item_correct"])
    print(f"  McNemar on correctness: b={mc['b']} c={mc['c']} p={mc['p_value']:.4f}")

lu.save_json({v: {k: r[k] for k in ("accuracy", "n_correct", "mean_cot_tokens", "tpca",
                                    "truncation_rate", "overthinking_rate",
                                    "wall_clock_s")} for v, r in results.items()},
             "eval/downstream_summary.json")
print("\\nwrote", lu.out("eval/downstream_summary.json"))
"""),
    md("""\
### What this pilot does and does not establish

With 20 problems, one greedy sample per problem and a 2048-token cap, this cannot
establish an accuracy difference — the confidence interval on Acc is enormous. It *can*
show a token-cost difference and it can reveal a broken checkpoint.

For the real numbers you need the later phase: full GSM8K/MATH-500, k samples per
problem with `maj@k`, MGSM for the cross-lingual claim, and the four-category failure
taxonomy. See `LATER_PHASES.md`.
"""),
    code(epilogue(
        ["eval/downstream_uniform.json", "eval/downstream_jsd.json",
         "eval/downstream_summary.json"],
        "eval/downstream_summary.json -> pilot row of the results table\n"
        "Full benchmark sweep                 -> LATER_PHASES.md, notebooks 10-16")),
]
