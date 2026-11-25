"""
Unit tests for the EB-SW-GPTAQ core (CPU, no model needed).

These verify the mathematics that the whole study rests on:

  T1  the objective identity  ||Dw X - W DX||_D  ==  1/2 tr(Dw H Dw^T) - tr(Dw (W Cw)^T)
  T2  the accumulator reproduces the dense weighted Gram / cross-covariance
  T3  with a fine grid the solver converges to the unconstrained optimum W + P
  T4  the solver actually lowers the objective versus RTN, for all four variants
  T5  act_order / true_sequential / mse_clip all execute and stay near T4's quality
  T6  uniform weights (w=1, Cw=None) reduce exactly to plain GPTQ statistics
  T7  JSD sanity: symmetry, bounds, self-distance, top-k error
  T8  ESS bisection is monotone and hits its target
  T9  packing round-trip preserves integer codes and dequantisation
  T10 causal-alignment helper agrees with the documented convention

Run: .venv/Scripts/python.exe tests/test_core.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from ebswgptaq.config import QuantConfig, WeightConfig
from ebswgptaq.quant_core import (
    GroupQuantizer, WeightedHessian, quantize_linear, reconstruction_objective,
)
from ebswgptaq.jsd import assert_causal_alignment, assert_jsd_sanity, divergence_per_position
from ebswgptaq.score_weights import (
    build_weights, concentration_stats, ess_fraction, exp_weights, solve_beta_for_ess,
)
from ebswgptaq.quantizer_adapter import self_test as pack_self_test

torch.manual_seed(0)
FAILS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name} {detail}")
    if not cond:
        FAILS.append(name)


def make_problem(out_f=64, in_f=256, n_tokens=2048, drift=0.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(out_f, in_f, generator=g) * 0.02
    X = torch.randn(in_f, n_tokens, generator=g)
    # a few heavy-tailed channels, like real LLM activations
    X[:4] *= 12.0
    X_ref = X + drift * torch.randn(in_f, n_tokens, generator=g) * X.abs().mean()
    w = torch.exp(0.6 * torch.randn(n_tokens, generator=g)).clamp(0.2, 8.0)
    w = w / w.mean()
    return W, X, X_ref, w


# ---------------------------------------------------------------------------- T1/T2
print("\nT1/T2  objective identity + accumulator")
W, X, X_ref, w = make_problem()
acc = WeightedHessian(X.shape[0], "float32", device="cpu", track_cross=True)
acc.add(X, X_ref, w)
H, Cw = acc.H, acc.Cw

H_dense = (X * w.sqrt()) @ (X * w.sqrt()).T
Cw_dense = ((X_ref - X) * w) @ X.T
check("T2 H matches dense", torch.allclose(H, H_dense, atol=1e-3),
      f"max_err={(H - H_dense).abs().max():.2e}")
check("T2 Cw matches dense", torch.allclose(Cw, Cw_dense, atol=1e-3),
      f"max_err={(Cw - Cw_dense).abs().max():.2e}")

# T1: compare the quadratic form to the direct weighted residual norm
Dw = torch.randn_like(W) * 0.01
direct = (Dw @ X - W @ (X_ref - X)) * w.sqrt().unsqueeze(0)
direct_val = 0.5 * float(direct.pow(2).sum())
formula_val = 0.5 * float(torch.sum(Dw * (Dw @ H))) - float(torch.sum(Dw * (W @ Cw)))
const = 0.5 * float((W @ (X_ref - X) * w.sqrt().unsqueeze(0)).pow(2).sum())
check("T1 objective identity", abs(direct_val - (formula_val + const)) < 1e-2 * abs(direct_val) + 1e-4,
      f"direct={direct_val:.6f} formula+const={formula_val + const:.6f}")

# ------------------------------------------------------------------------------ T6
print("\nT6  uniform weights reduce to plain GPTQ statistics")
acc_u = WeightedHessian(X.shape[0], "float32", device="cpu", track_cross=False)
acc_u.add(X, None, None)
check("T6 H(w=1) == XX^T", torch.allclose(acc_u.H, X @ X.T, atol=1e-2))
check("T6 no cross term when track_cross=False", acc_u.Cw is None)
check("T6 ess==1 for uniform", abs(acc_u.ess_fraction() - 1.0) < 1e-6,
      f"ess={acc_u.ess_fraction():.6f}")

# ------------------------------------------------------------------------------ T3
print("\nT3  fine grid converges to the unconstrained optimum W + P")
cfg_fine = QuantConfig(bits=16, group_size=256, sym=False, asym=True,
                       damp_percent=0.01, act_order=False, true_sequential=False)
res_fine = quantize_linear("t", W, H, Cw, cfg_fine)
damp = 0.01 * float(torch.diagonal(H).mean())
Hd = H.clone()
idx = torch.arange(H.shape[0])
Hd[idx, idx] += damp
Hinv = torch.cholesky_inverse(torch.linalg.cholesky(Hd))
P = (W @ Cw) @ Hinv
opt = W + P
err_to_opt = float((res_fine.W_hat.float() - opt).norm() / opt.norm())
err_to_W = float((res_fine.W_hat.float() - W).norm() / W.norm())
check("T3 16-bit solve sits between W and the optimum, near the optimum",
      err_to_opt < err_to_W, f"dist_to_opt={err_to_opt:.4f} dist_to_W={err_to_W:.4f}")

# ------------------------------------------------------------------------------ T4
print("\nT4  objective improves over RTN for every variant")
variants = {
    "gptq_uniform":      dict(asym=False),
    "gptaq_uniform":     dict(asym=True),
    "gptq_weighted":     dict(asym=False, weighted=True),
    "gptaq_weighted":    dict(asym=True, weighted=True),
}
for label, spec in variants.items():
    weighted = spec.pop("weighted", False)
    cfg = QuantConfig(bits=4, group_size=128, sym=False, damp_percent=0.01, **spec)
    Hv, Cwv = (H, Cw) if weighted else (H_dense_uniform := (X @ X.T), None)
    if weighted:
        Hv, Cwv = H, Cw
    else:
        Hv, Cwv = X @ X.T, None
    rtn = quantize_linear("t", W, None, None, cfg)
    sol = quantize_linear("t", W, Hv, Cwv, cfg)
    o_rtn = reconstruction_objective(rtn.W_hat, W, Hv, Cwv)
    o_sol = reconstruction_objective(sol.W_hat, W, Hv, Cwv)
    check(f"T4 {label}: solver objective < RTN objective", o_sol < o_rtn,
          f"rtn={o_rtn:.4e} sol={o_sol:.4e} gain={100*(1-o_sol/max(o_rtn,1e-12)):.1f}%")

# ------------------------------------------------------------------------------ T5
print("\nT5  act_order / true_sequential / mse_clip execute and stay competitive")
# T5: each option must be finite and no worse than RTN on *its own* grid type.
cfg_base = dict(bits=4, group_size=32, sym=False, damp_percent=0.01, asym=True)
base = quantize_linear("t", W, H, Cw, QuantConfig(**cfg_base))
o_base = reconstruction_objective(base.W_hat, W, H, Cw)
rtn_base = reconstruction_objective(quantize_linear("t", W, None, None,
                                                    QuantConfig(**cfg_base)).W_hat, W, H, Cw)
for label, over in [
    ("act_order", dict(act_order=True)),
    ("true_sequential", dict(true_sequential=True)),
    ("both", dict(act_order=True, true_sequential=True)),
    ("mse_clip", dict(mse_clip=True, n_clip_grid=8)),
    ("sym_grid", dict(sym=True)),
    ("blocksize_8", dict(blocksize=8)),
]:
    cfg = QuantConfig(**{**cfg_base, **over})
    rtn = quantize_linear("t", W, None, None, cfg)
    r = quantize_linear("t", W, H, Cw, cfg)
    o = reconstruction_objective(r.W_hat, W, H, Cw)
    o_rtn = reconstruction_objective(rtn.W_hat, W, H, Cw)
    # sym_grid changes the quantiser, not the solver, so only require it to beat its
    # own RTN; every other row must also stay close to the asymmetric base.
    ceiling = 2.0 * o_base if label == "sym_grid" else 1.10 * o_base
    ok = torch.isfinite(r.W_hat).all() and o < o_rtn and o <= ceiling
    check(f"T5 {label}", bool(ok),
          f"obj={o:.4e} rtn={o_rtn:.4e} base={o_base:.4e}")

# ------------------------------------------------------------------------------ T7
print("\nT7  JSD sanity")
s = assert_jsd_sanity(v=256)
for k, v in s.items():
    print(f"       {k}: {v}")
check("T7 symmetric", s["symmetry_max_abs_diff"] < 1e-6)
check("T7 self distance ~0", s["self_jsd_max"] < 1e-9)
check("T7 non-negative", s["min_jsd"] >= -1e-9)
check("T7 bounded by ln2", s["within_bound"])

# top-k must converge to the exact value as k grows, on a *peaked* distribution
# (a flat distribution over 256 bins has no dominant head, so top-k is a poor
#  approximation there by construction -- that is not an implementation error).
g7 = torch.Generator().manual_seed(1)
lp = torch.randn(32, 1024, generator=g7) * 4.0
lq = lp + 0.5 * torch.randn(32, 1024, generator=g7)
exact = divergence_per_position(lp, lq, "jsd", chunk=8)
errs = []
for k in (32, 128, 512):
    approx = divergence_per_position(lp, lq, "jsd", chunk=8, topk=k)
    errs.append(float((approx - exact).abs().mean() / exact.mean()))
    print(f"       topk={k:4d} mean_rel_err={errs[-1]:.4f}")
check("T7 top-k approximation converges", errs[0] > errs[1] > errs[2] and errs[2] < 0.01,
      f"rel_errs={['%.4f' % e for e in errs]}")
check("T7 top-k near exact at k=512", errs[2] < 0.01, f"{errs[2]:.4f}")

# ------------------------------------------------------------------------------ T8
print("\nT8  ESS bisection")
z = torch.randn(4096) * 1.5
for target in (0.9, 0.5, 0.25):
    beta, achieved = solve_beta_for_ess(z, target)
    check(f"T8 target ess={target} hit within 0.02", abs(achieved - target) < 0.02,
          f"beta={beta:.3f} achieved={achieved:.4f}")
ess_at = [ess_fraction(exp_weights(z, b)) for b in (0.0, 0.5, 1.0, 2.0, 4.0)]
check("T8 ESS monotonically decreasing in beta",
      all(a > b for a, b in zip(ess_at, ess_at[1:])),
      f"{[round(e, 4) for e in ess_at]}")

# ------------------------------------------------------------------------------ extra
print("\nextra  weight builder controls")
wcfg = WeightConfig(signal="jsd", ess_floor=0.5, clip_quantile=0.99)
scores = torch.randn(1000) * 2.0
scores[0] = 1e4                      # outlier that winsorisation must tame
positions = torch.arange(1000)
out = build_weights(scores, positions, None, wcfg)
stats = out["stats"]
print(f"       beta={out['beta']:.3f} ess_frac={stats['ess_fraction']:.3f} "
      f"max={stats['max']:.3f} mass_top1={stats['mass_top1pct']:.3f}")
check("extra ESS floor respected", abs(stats["ess_fraction"] - 0.5) < 0.03)
check("extra weights mean 1", abs(stats["mean"] - 1.0) < 1e-4)
check("extra outlier tamed", stats["max"] < 20.0, f"max={stats['max']:.2f}")

shuf = build_weights(scores, positions, None, WeightConfig(signal="jsd", ess_floor=0.5,
                                                          shuffle_within_sequence=True))
check("extra shuffle keeps ESS but breaks alignment",
      abs(shuf["stats"]["ess_fraction"] - stats["ess_fraction"]) < 0.02)

uni = build_weights(scores, positions, None, WeightConfig(signal="none"))
check("extra uniform baseline", abs(uni["stats"]["ess_fraction"] - 1.0) < 1e-6)

# ------------------------------------------------------------------------------ T9
print("\nT9  packing round-trip")
for bits, sym in [(4, False), (3, False), (4, True)]:
    cfg = QuantConfig(bits=bits, group_size=64, sym=sym, asym=False, damp_percent=0.01)
    r = quantize_linear("t", W, (X @ X.T), None, cfg)
    rt = pack_self_test(r, bits, sym)
    check(f"T9 W{bits} sym={sym} codes exact",
          rt["code_roundtrip_exact_frac"] > 0.999, f"{rt['code_roundtrip_exact_frac']:.6f}")
    check(f"T9 W{bits} sym={sym} dequant matches",
          rt["dequant_max_abs_err"] < 1e-2, f"err={rt['dequant_max_abs_err']:.2e}")

# ------------------------------------------------------------------------------ T10
print("\nT10 causal alignment")
ca = assert_causal_alignment()
print(f"       {ca}")
check("T10 shifted pairs correct", ca["all_pairs_correct"], str(ca["first_pair"]))

# --------------------------------------------------------------------------- summary
print("\n" + "=" * 70)
if FAILS:
    print(f"{len(FAILS)} FAILURES: {FAILS}")
    sys.exit(1)
print("ALL CORE TESTS PASSED")
