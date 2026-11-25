# EB-SW-GPTAQ — BTP codebase

Ten self-contained Kaggle notebooks implementing the **proxy-gate study** from
`EB_SW_GPTAQ_Research_Pivot_v2_and_Implementation_Plan_EKA.pdf`, plus the library they
are generated from.

The research question, in one line:

> Do token positions whose output distribution quantization perturbs most (same-prefix
> JSD against a cheap probe) sit in activation directions that uniform calibration
> under-weights — and does up-weighting those columns during asymmetric
> second-order PTQ improve held-out preservation of the FP16 model?

Nothing here claims novelty for token weighting: RSQ already scales features by token
importance before computing GPTQ statistics. The distinct thing under test is the
**signal** (same-prefix JSD), the **mechanism analysis** (activation subspaces), and the
**specificity controls** (random and within-sequence-shuffled weightings).

---

## Status

| Item | State |
|---|---|
| Notebooks 00-09 (proxy gate) | written, all code cells compile |
| Library (16 modules) | written, unit- and integration-tested on CPU |
| Solver | verified against the dense objective identity, and against RTN on every variant |
| Real packed INT3/INT4 kernels | **not implemented** — deliberate, see `LATER_PHASES.md` |
| Downstream benchmarks (MATH-500, MGSM, AIME, LiveCodeBench) | **not implemented** — after the gate |
| Run on a real GPU | **not yet** — no GPU in the authoring environment |

Test results that back the table above (`.venv/Scripts/python.exe tests/test_core.py`):

```
T1 objective identity    direct=5089.492  formula+const=5089.492    (exact)
T2 accumulator           H and C_w match the dense computation to 0.0e+00
T3 fine-grid solve       distance to the unconstrained optimum 0.0014 vs 0.0195 to W
T4 solver vs RTN         -10.1% objective (uniform GPTQ) ... -14.4% (weighted GPTAQ)
T5 option paths          act_order / true_sequential / mse_clip / sym grid all beat RTN
T7 JSD                   symmetry 0.0, self-distance 2.6e-10, bounded by ln2
                         top-k convergence: 6.5% -> 0.8% -> 0.01% relative error
T8 ESS bisection         targets 0.90/0.50/0.25 hit exactly; monotone in beta
T9 packing               3-bit and 4-bit round-trip exact, dequant max err < 1e-2
T10 causal alignment     logits[t] pairs with token_ids[t+1]
```

and `tests/test_pipeline.py` (a mock transformer with the same module layout as Qwen):
hooks capture the true activation columns, weights are modified in place, the solver
beats RTN per module, weighted and unweighted runs both complete, and two identical
models give KL = 0.0.

---

## Layout

```
BTP code/
├── README.md                  this file
├── CODE_INDEX.md              every file, every function, what uses it
├── LATER_PHASES.md            notebooks 10-16, ablations, benchmarks, real kernels
├── requirements-kaggle.txt
├── configs/                   model, calibration, quant (w4/w3), probes, experiments
├── src/ebswgptaq/             canonical library (16 modules)
├── notebooks/                 generated, self-contained, one per stage
├── scripts/
│   ├── nb_common.py           notebook scaffolding + embedded-library cell
│   ├── nb_00_04.py            notebook specs 00-04
│   ├── nb_05_09.py            notebook specs 05-09
│   └── build_notebooks.py     writes notebooks/*.ipynb, --check compiles every cell
└── tests/
    ├── test_core.py           solver/JSD/ESS/packing/causal-alignment unit tests
    └── test_pipeline.py       model-level driver integration test (mock transformer)
```

The notebooks are **generated** from `src/`. Edit the library, then:

```bash
python scripts/build_notebooks.py --check
```

Never hand-edit the library cell inside a notebook — it will be overwritten.

---

## The notebook chain

Each notebook is self-contained: cell 0 reconstructs the package and configs on disk.
Artifacts are the only thing that crosses a notebook boundary.

| # | Notebook | Needs | Produces | T4 runtime* |
|---|---|---|---|---|
| 00 | `00_environment_check` | — | `env/env_report.json` | <2 min |
| 01 | `01_generate_traces` | — | `calib/calib_corpus.jsonl`, `calib/holdout_corpus.jsonl`, `calib/corpus_manifest.json`, `calib/leak_audit.json` | 30-45 min |
| 02 | `02_uniform_baseline` | 01 | `quant/uniform_w4_*`, `quant/uniform_w3_*`, `eval/baseline_w{4,3}.json` | 50-90 min (W4 + W3) |
| 03 | `03_build_probe` | 01 | `probes/probe_rtn_w4/`, `probes/probe_gptq_w4/` | ~15 min |
| 04 | `04_compute_jsd` | 01, 03 | `scores/jsd_rtn_calib.pt`, `scores/scores_rtn_summary.json` | 10-20 min |
| 05 | `05_analyze_geometry` | 01, 04 | `geometry/geometry_rtn.json`, `geometry/geometry_aggregate.json` | 10-15 min |
| 06 | `06_run_weighted_variants` | 01, 02, 04 | `quant/weighted_<variant>_*`, `weights/*.pt`, `quant/variant_summary.json` | 25-40 min **per variant** |
| 07 | `07_evaluate_proxy` | 01, 04, 06 | `eval/proxy_*.json`, `eval/specificity_gate.json` | 5-8 min per variant |
| 08 | `08_probe_swap` | 03, 04, 06 | `scores/jsd_gptq_calib.pt`, `eval/probe_swap.json` | ~12 min (+20 optional) |
| 09 | `09_downstream_pilot` | 06, 07 | `eval/downstream_*.json` | 25-40 min |

\* **These are estimates from FLOP and bandwidth counts, not measurements** — nothing
here has been run on a GPU yet. Set `DRY_RUN = True` in notebooks 02 and 06 to time four
layers (~2 min) and extrapolate before committing a session.

Total for the full gate at pilot scale: **≈ 6-8 GPU-hours**, inside the free Kaggle
quota (30 h/week on a T4).

---

## Running on Kaggle

Kaggle gives every notebook a fresh VM, so `/kaggle/working` does **not** leak into the
next notebook by itself. Two working options:

**Option A — attach the previous notebook's output (recommended)**

1. Open notebook 00, set Accelerator = `GPU T4 x1`, Internet = `On`.
2. `Save Version` → *Save & Run All*. The artifacts become that version's output.
3. Open notebook 01 → **Add Input** → *Notebook Output* → notebook 00.
4. Repeat down the chain.

`logging_utils.find()` searches `/kaggle/working/artifacts` first, then every
`/kaggle/input/*/artifacts`, so the artifacts appear without any path editing. If
something is missing you get an error naming the notebook you must run first, not a
silent empty directory.

**Option B — zip and re-attach as a dataset**

```python
from ebswgptaq import logging_utils as lu
lu.zip_artifacts("artifacts_04.zip")
```

Download it from the notebook's *Output* tab, create a private Kaggle dataset from the
zip, and attach that to the next notebook. Useful when you want to move artifacts off
Kaggle (or hand them to a supervisor) without re-running anything.

Every notebook ends with a manifest cell that lists exactly what it wrote and what the
next notebook needs.

---

## Validate the chain cheaply before spending GPU hours

Nothing here has been executed on a real GPU. Before committing a session, run a
60-minute smoke pass that exercises every code path at 1/10 scale:

| Notebook | Change | Cost |
|---|---|---|
| 01 | in the config cell: `ccfg.n_reasoning = 3`, `ccfg.n_holdout = 2`, `ccfg.gen_max_new_tokens = 256`, `ccfg.n_general = 2` | ~5 min |
| 02 | `DRY_RUN = True` (4 layers), `RUN_W3 = False` | ~4 min |
| 03 | `PROBE_BITS = 4`, RTN only (comment out the GPTQ block) | ~2 min |
| 04 | leave as is | ~3 min |
| 05 | `LAYERS = [0]`, `K_SUBSPACE = 8` | ~2 min |
| 06 | `DRY_RUN = True`, `RUN = ["uniform", "jsd"]` | ~6 min |
| 07-09 | leave as is (`N_PROBLEMS = 5` in 09) | ~15 min |

The point is to hit every line of code once — hook capture, weight swap, Cholesky,
greedy loop, JSON IO, `require()`, plots — not to produce numbers. **A dry run's fidelity
numbers are meaningless** (most layers stay FP16), and notebook 02 will report the
calibration corpus as `rank-deficient` and assert out — override that assert for the
smoke pass only.

After the smoke pass, restore the configs (`git checkout configs/`) and run for real.

---

## Known risks on Kaggle

| Risk | Symptom | Mitigation already in the code |
|---|---|---|
| Calibration corpus too small | Hessian rank-deficient, all variants identical | `batching.calibration_sufficiency` asserts in notebooks 02 and 06, naming the fix |
| OOM on a 16 GB T4 | CUDA out of memory during W3 | reference weights live in CPU RAM; `del` + `empty_cache()` between stages; the 4B model needs `drift_correction: false` |
| FP16 overflow in the lm_head | `nan` scores, weights collapse to uniform | `fp32_lm_head=True` by default; `safe_logits()` raises instead of propagating |
| `transformers` API drift | `TypeError: from_pretrained() got an unexpected keyword` | `load_model` retries with `dtype=` if `torch_dtype=` is rejected |
| Slow layer diagnostics | 10+ min of pure overhead on the 8960×8960 module | `diag_every_n: 2` and power iteration instead of eigendecomposition above `diag_max_dim` |
| Artifacts not visible in the next notebook | `FileNotFoundError` naming a notebook | `require()` message lists every searched root; attach the previous version's output |
| Kaggle 12 h session cap | run truncated mid-variant | per-variant checkpoints are written as each variant finishes, so a re-run skips completed ones |
| GPU quota exhaustion (30 h/week) | notebook killed at start | full gate ≈ 6-8 h; dry run first |

The one thing no code can protect you from: a `transformers` version whose
`Qwen2DecoderLayer` changes signature. We only call `model.model(...)` and module
forward hooks, never a layer directly, which is the most version-stable surface
available — but if `forward_hidden` ever throws, that is where to look.

---

## Decisions that will bite you if you change them

1. **One sequence per batch (`B = 1`).** The whole alignment story depends on
   `column t of every linear's input ⇔ token t`. Batching would require an extra
   index mapping and an untested code path. At 2048 tokens the overhead is negligible.
2. **Scores exist only where a next-token target exists; Hessians accumulate over all
   tokens.** Unscored positions get weight `1.0`, never zero and never dropped —
   dropping them would change the effective calibration set between variants.
3. **Grid parameters are fitted on the original `W`**, then the asymmetric optimum is
   absorbed into the greedy column loop. Fitting the grid after shifting `W` distorts
   per-group scales on outlier channels.
4. **Reference weights and produced weights live in CPU RAM.** That is what keeps a
   1.5B model + drift correction inside 16 GB. It costs ~17 s of PCIe traffic per run.
5. **`fp32_lm_head=True` on T4.** The 152k-way projection in FP16 is where Qwen-family
   checkpoints overflow; `safe_logits()` raises instead of letting NaNs reach the JSD
   scores, where they would silently destroy the weights.
6. **Damping is `damp_percent × mean(diag(H))`**, matching GPTQ convention, not an
   absolute value. Changing it changes the effective objective between layers.

---

## What you may and may not claim

**Supported by what this code measures**

- Whether JSD-weighted asymmetric reconstruction improves held-out teacher-forced
  fidelity over uniform, random and shuffled weightings, with paired intervals.
- Whether high-JSD positions have a distinct activation subspace (with a random control).
- Whether the signal survives an RTN → GPTQ probe swap.
- Per-module reconstruction objectives, Hessian condition numbers, weight concentration
  (ESS/N, top-1% mass) for every variant.

**Not supported yet**

- Any speed, VRAM or throughput claim — the checkpoints are fake-quant. Report
  compression and task metrics separately from runtime (`LATER_PHASES.md`, notebook 12).
- Any multilingual claim — MGSM is not run until the gate passes (notebook 13).
- Any novelty claim over RSQ. The honest framing is "a different signal, tested against
  the strongest controls we could build".

---

## Tests

```bash
# CPU only, no model download, ~60 s total
.venv/Scripts/python.exe tests/test_core.py        # the mathematics
.venv/Scripts/python.exe tests/test_pipeline.py    # the orchestration
.venv/Scripts/python.exe tests/test_artifacts.py   # the notebook-to-notebook contract
```

| Test file | What it proves |
|---|---|
| `test_core.py` | objective identity, accumulator, fine-grid convergence, solver vs RTN, option paths, JSD, ESS bisection, 3/4-bit packing, causal alignment |
| `test_pipeline.py` | forward hooks capture the true activation columns, weight swapping works, drift and no-drift runs both complete, fidelity is zero for identical models |
| `test_artifacts.py` | `require()` names the missing notebook, artifacts are discoverable from an attached output, JSON/JSONL round-trip, zip layout re-attaches cleanly |

The `.venv/` here is a CPU-only test environment (torch + numpy + scipy + pyyaml,
~700 MB). Recreate it with:

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/Scripts/python.exe torch \
    --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/Scripts/python.exe numpy pyyaml scipy
```

Neither the tests nor the notebooks need this venv on Kaggle — Kaggle already has
everything.

The notebooks also re-run the cheap subset of these checks at the point of use:
notebook 00 (everything), 02 (uniform-path equivalence), 04 (causal alignment + JSD
bounds), 06 (control concentration).
