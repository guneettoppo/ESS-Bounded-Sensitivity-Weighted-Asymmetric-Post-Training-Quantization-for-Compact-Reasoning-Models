# Later phases — notebooks 10-16, ablations, benchmarks, kernels

Everything in this file is **deliberately not built yet**. The proxy gate
(notebooks 00-09) decides whether any of it is worth the GPU time. This file exists so
nothing is forgotten, and so each piece has a concrete interface before it is written.

Mapped against the source documents:

* `EB_SW_GPTAQ_Research_Pivot_v2_...pdf` §7 Milestone 7 (downstream pilot and expansion)
* `EB-SW-GPTAQ Complete Research Plan (3).pdf` §3.3-3.6 (benchmarks, diagnostics,
  ablations, 12-week schedule)

---

## 0. Gate that unlocks this file

```
specificity gate (notebook 07)
  jsd beats uniform AND random AND shuffled on held-out mean_kl,
  paired 95% bootstrap CI on the delta excludes zero
        AND
robustness gate (notebook 08)
  ranking survives RTN -> GPTQ probe swap (or the dependence is characterised)

  PASS  -> notebook 10 onward, in order
  FAIL  -> stop. Write the negative result (it is publishable), then optionally run
           notebook 14 alone to show the effect is absent downstream too.
```

Do not start notebook 10 with `RUN_TRANSFER=False` still set in notebook 08, and do not
run 13 without 07.

---

## 1. Notebook 10 — calibration-data ablation (A1)

**Question.** How much of any gain comes from domain-matched calibration data rather
than from the weighting?

| Variant | Corpus |
|---|---|
| `c4_only` | 22 general anchors, no reasoning traces |
| `cot_only` | 22 self-generated traces, no anchors |
| `mixed` | the 16/6 corpus from notebook 01 (already measured) |

* **Needs:** 01, 06, 07 (the `jsd` and `uniform` rows for comparison)
* **Produces:** `quant/ablation_a1_<combo>_<variant>/`, `eval/ablation_a1.json`
* **Implementation:** re-run `quantize_model` with three different record lists. No new
  library code — `build_mixed_corpus` already accepts any two lists.
* **Cost:** 6 quantization runs ≈ 1.5 h
* **Acceptance:** the mixed corpus is not the *only* corpus on which `jsd` works; if it
  is, say so and narrow the claim to "with reasoning-matched calibration".

---

## 2. Notebook 11 — ESS / temperature sweep (A2, A5)

**Question.** Is there a U-shape between weight concentration and held-out fidelity?
Where does it turn? Does sink masking matter?

| Axis | Values |
|---|---|
| A2 ESS floor | 1.00 (uniform), 0.90, 0.75, 0.50, 0.25, unbounded (fixed β high) |
| A5 sink mask | on / off (2 runs at ESS 0.50) |

* **Needs:** 01, 04
* **Produces:** `eval/ablation_a2.json`, `eval/ablation_a5.json`, and the U-curve figure
* **New library code needed:** none — `solve_beta_for_ess` already covers every floor;
  `WeightConfig.sink_mask` toggles A5.
* **Cost:** 8 runs ≈ 2 h
* **Acceptance:** report the full curve, not the best point. If ESS ≈ 0.5 is not
  special, that is a real (and useful) finding — v1's ESS bound would then be a
  cosmetic control, exactly the risk the pivot flags under "ESS is cosmetic".

**Reference point to reproduce:** Shin's reported collapse is ESS/N = 2.7 % with 94.9 %
of mass on 1 % of tokens. `concentration_stats` reports both numbers directly, so the
unbounded variant can be checked against that claim.

---

## 3. Notebook 12 — real packed kernels (correctness, then speed)

**Question.** Do the gains survive real INT3/INT4 storage, and what is the actual VRAM
and throughput?

This is the only notebook that may report latency, and only after the checkpoint is
verified to dequantise correctly.

* **Needs:** the winning variant from 07, plus `save_packed_checkpoint`
* **Steps (also in `quantizer_adapter.export_for_kernel` docstring):**
  1. `save_packed_checkpoint(results, out_dir, bits, sym)` — integer codes + group params
  2. copy the base model's `config.json` and tokenizer files
  3. convert with GPTQModel / vLLM's Marlin exporter **for the exact library version
     you install**
  4. **verify**: dequantise the converted checkpoint and compare with `W_hat`;
     `max_abs_err < 1e-3` in FP16, or stop — a silently shifted zero-point looks like a
     catastrophic accuracy loss
  5. only then report peak VRAM and tokens/sec
* **Produces:** `packed/<variant>/`, `eval/packed_runtime.json`
* **Acceptance:** the fidelity measured through the packed checkpoint matches the
  fake-quant fidelity within noise. If it does not, the export is wrong, not the method.

**Known blocker to record in the write-up:** GPTQ's int32 strided layout has no exact
3-bit form (`32 / 3` is not an integer). The adapter therefore exports 3-bit codes
unpacked (`layout: "unpacked-int8"`). Real 3-bit inference needs a kernel-specific
stride convention; document whichever one you adopt.

---

## 4. Notebook 13 — full reasoning benchmarks (MATH-500 first)

Order matters: one benchmark, fully done, before four benchmarks half-done.

| Step | Benchmark | n | Decoding | Metrics |
|---|---|---|---|---|
| 13a | MATH-500 | 500 | temp 0.6, top_p 0.95, 16k new tokens | Acc, mean CoT length, TPCA, truncation, over-thinking |
| 13b | GSM8K test | 1319 | 4k new tokens | same |
| 13c | MGSM (hi/bn, zh, de, ja) | 250 each | 4k | same — **this is the H3 cross-lingual claim** |
| 13d | AIME 2024+2025 | 60, k=4 or 8 | 16k | pass@1, maj@k |
| 13e | LiveCodeBench | recent subset | 4k | syntax integrity + pass rate |
| 13f | WikiText-2 perplexity | — | teacher-forced | general-language sanity |

* **Needs:** 06 (checkpoints), 07 (gate)
* **New library code needed:**
  * `evaluation.load_benchmark(name)` → uniform `{prompt_id, prompt, answer, meta}` records
  * `evaluation.failure_taxonomy(rows)` → the four categories from v1 §3.4:
    over-thinking/commitment failure, infinite loop/truncation, genuine arithmetic
    error, formatting/degenerate output
  * `evaluation.math_verify_match(pred, gold)` → symbolic equivalence via `math-verify`
    or SymPy, replacing the regex matcher (keep the regex one as a fallback)
* **Cost:** 13a ≈ 6 h for two variants at 500 problems × 16k tokens. Budget it week by
  week; the free T4 quota is 30 h/week.
* **Acceptance:** paired bootstrap CIs for Acc and TPCA on every benchmark pair, plus
  McNemar on per-problem correctness. A result without an interval does not go in the
  table.

**Warning that belongs in the paper:** the English logit-penalty baseline of Lotfi et
al. is a competitor only on 13a/13b. On 13c it cannot fire at all (the keyword list is
English), which is the point of H3 — do not present its MGSM result as a fair fight
without saying so.

---

## 5. Notebook 14 — mechanistic diagnostics on the full suite

| Diagnostic | What it shows | Library support |
|---|---|---|
| Holdout KL stratified into fork / commitment / execution positions | *where* the repair happens | `strata_report` (extend: commitment positions = tokens in `</think>`, `\boxed{`, EOS) |
| Failure taxonomy over every incorrect trace | over-thinking vs arithmetic | to be written (notebook 13) |
| Hessian condition number vs held-out KL, per module | the stability mechanism | `spectral_summary` + `spectral_summary` outputs already stored in `meta` |
| Real Pareto frontier: accuracy vs GPU-seconds per solved problem | deployment relevance | needs notebook 12 timings |

* **Needs:** 07, 12, 13
* **Produces:** `eval/diagnostics.json`, paper figures 1-4

---

## 6. Notebook 15 — scale-up (optional, needs more than a T4)

| Model | Role | Hardware | Note |
|---|---|---|---|
| Qwen3-1.7B (thinking) | second architecture family | T4 16 GB | fits, same pipeline |
| Qwen3-4B (thinking) | the regime Shin reported | T4 16 GB | **tight**: 8 GB of weights + 8 GB of FP16 references will not both fit; run with `drift_correction: false` or quantise in two shards |
| DeepSeek-R1-Distill-Qwen-7B / Qwen3-8B | scale invariance for a main-track submission | A100 40 GB or 2×T4 | out of free-tier budget |

* **Needs:** 07 (a repeatable effect on the primary model first)
* **Acceptance:** the direction of the effect and the sign of the bootstrap CI match the
  1.5B result. If they do not, report that — a scale-dependent effect is a finding.

Memory maths to avoid a failed run: `quantize_model` holds (a) the live model, (b) FP16
reference weights in CPU RAM, (c) one layer's `H`, `C_w`, `H⁻¹` in FP32 (3 × d²ᵢ × 4 B;
for d=8960 that is ~1 GB). At 4B this exceeds a 16 GB T4 with `drift_correction: true`.

---

## 7. Notebook 16 — ablations that need the benchmark harness (A3, A4, A6)

| Axis | Variants | Question |
|---|---|---|
| A3 signal | entropy / JSD / forward-KL / random | is the two-pass JSD probe necessary, or does single-pass entropy suffice? |
| A4 asymmetry | `asym: false` vs `asym: true` | does weighted Hessian interact with drift correction? |
| A6 rotations | QuaRot/Hadamard + GPTQ vs QuaRot/Hadamard + ours | does the method compose additively with outlier rotation? |

* **A3:** partly covered in notebook 06 (`entropy`, `kl` variants are defined but not in
  the default `RUN` list). Set `RUN = VARIANTS + EXTRA_VARIANTS`.
* **A4:** one config flag (`QuantConfig.asym`); 2 runs.
* **A6:** needs a QuaRot/Hadamard implementation — not in this repo. Either add
  `src/ebswgptaq/rotations.py` or use the official QuaRot code and re-apply our solver to
  the rotated weights. Decide before writing it; the interface is "rotate, then quantise
  with the same weight vector".
* **Cost:** 7 runs ≈ 2 h (excluding A6 if it needs a separate rotation pass).

---

## 8. Statistics and manuscript (mirrors v1 Phase 5)

| Deliverable | Content | Status |
|---|---|---|
| Paired bootstrap 95% CIs | Acc, TPCA, mean_kl per benchmark × variant | `statistics.paired_bootstrap_ci` ready |
| McNemar | per-problem correctness, every pair | `statistics.mcnemar_exact` ready |
| Figure 1 | Hessian condition number vs held-out KL | data already in `meta["hessian"]` |
| Figure 2 | the ESS U-curve (A2) | notebook 11 |
| Figure 3 | failure taxonomy breakdown | notebook 13 |
| Figure 4 | Pareto: accuracy vs GPU-seconds/solved | notebook 12 + 13 |
| Table 1 | results with intervals | all of the above |
| Appendix | leak audit, environment manifests, hyper-parameters, reproduction commands | `env/`, `calib/leak_audit.json` already written |

Every number in the paper must come from a JSON file in `artifacts/`. If it does not
exist on disk, it does not exist.

---

## 9. Compute budget (T4 hours)

| Notebook | Hours |
|---|---|
| 00-05 | 1.5 |
| 06 | 0.8-1.2 (4 variants) |
| 07 | 0.5 |
| 08 | 0.4 |
| 09 | 0.7 |
| 10 | 1.5 |
| 11 | 2.0 |
| 12 | 1.0 |
| 13 (MATH-500 + GSM8K, two variants) | 12-15 |
| 14 | 1.0 |
| 16 | 2.0 |
| **Total to a submittable result on one model** | **≈ 25-28 h** |

Free Kaggle T4 quota is 30 h/week, so this fits in one or two weeks of wall-clock work.
The 130 h figure in the v1 plan assumes the full four-model, four-benchmark sweep.

---

## 10. Things that must not be forgotten when writing up

- [ ] State that token-importance weighting of GPTQ statistics is **not novel** (RSQ).
      The claim is the signal + the mechanism evidence + the controls.
- [ ] Report the probe-swap result even if it is bad.
- [ ] Report the geometry result even if high-JSD positions are *not* geometrically
      distinct — that outcome means the signal is a scalar importance, and it changes
      the story.
- [ ] Keep fake-quant fidelity and real-kernel runtime in separate tables.
- [ ] Record the calibration corpus provenance (training splits, greedy decoding,
      prompt ids) and the 13-gram audit in the appendix.
- [ ] If `random` or `shuffled` matches `jsd`, say so plainly. The negative result is
      the honest outcome and the pivot explicitly plans for it.
- [ ] Cite the exact model revision and tokenizer revision of every checkpoint.
