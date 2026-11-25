"""
Group-wise fake quantizer and the weighted asymmetric second-order solver.

This is the method under test. Everything else in the repository exists to
measure whether it helps.

What the solver optimises
-------------------------
For a linear layer with pretrained weight W in R^{out x in}, calibration inputs
X in R^{in x N} (from the *partially quantised* model), reference inputs
X~ in R^{in x N} (from the FP16 model), and a positive diagonal token weight
matrix D = diag(w_1..w_N) with mean(w) = 1:

    min_{W_hat in Q_b}  1/2 * || (W_hat X - W X~) D^{1/2} ||_F^2
                        + lam/2 * || W_hat - W ||_F^2

With DW = W_hat - W and DX = X~ - X this is exactly

    1/2 tr(DW H_w DW^T) - tr(DW (W C_w)^T) + const
    H_w = X D X^T + lam*I          (accumulated in FP32/FP64)
    C_w = (X~ - X) D X^T           (raw cross-covariance, no ridge inside)

The unconstrained optimum is DW* = W C_w H_w^{-1}. Because the quantiser still
constrains W_hat to the grid, we do not "shift then quantise". Instead we follow
GPTAQ: fit the grid (scale/zero) on the ORIGINAL W, then absorb the asymmetric
optimum into the greedy column-by-column OBC update. The identity used is
elementary but worth writing down, because it is what makes the code short:

    fixing column q to its quantised value, the optimal values of the remaining
    columns F are

        W[:, F] <- (W + P)[:, F] - ((W + P)[:, q] - q_hat) * Hinv[q, F] / Hinv[q, q]

    with P = W C_w Hinv. i.e. pre-shifting the *running target* by P and then
    applying the standard GPTQ rule is mathematically exact for the quadratic.
    (Verified in notebook 06 by the direct-objective unit test.)

Where the "weighted" and "asymmetric" parts live
------------------------------------------------
- weighting   : inside H_w and C_w (the accumulator applies sqrt(w) / w).
- asymmetry   : P, and it is switched off when Cw is None.

`asym=False` therefore gives weighted symmetric GPTQ (the EB-SW-GPTQ ablation) and
`Cw=None` gives a uniform-weight GPTAQ-class baseline.

Outputs
-------
`QuantizedLinear` carries the actual integer codes (q_int), the group parameters
(scales, zeros, g_idx) and the dequantised W_hat. `quantizer_adapter.py` turns the
first four into a packed-checkpoint layout without touching the solver.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .config import QuantConfig


# ======================================================================================
# Group index bookkeeping
# ======================================================================================


def group_index(d_in: int, group_size: int) -> torch.Tensor:
    """Group id per input channel (GPTQ g_idx convention: groups along `in`)."""
    return (torch.arange(d_in, dtype=torch.long) // group_size).to(torch.int32)


def n_groups(d_in: int, group_size: int) -> int:
    return int(math.ceil(d_in / group_size))


# ======================================================================================
# Quantizer
# ======================================================================================


class GroupQuantizer:
    """
    Per-output-row, per-input-group uniform (or symmetric) fake quantiser.

    Kept deliberately free of any kernel/storage assumption: it returns integer
    codes plus group params, and a dequantised tensor. Real packing happens later.
    """

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        sym: bool = False,
        mse_clip: bool = False,
        n_clip_grid: int = 20,
        eps: float = 1e-8,
    ):
        if not 2 <= bits <= 16:
            raise ValueError(f"bits must be in [2, 16], got {bits}")
        self.bits = int(bits)
        self.group_size = int(group_size)
        self.sym = bool(sym)
        self.mse_clip = bool(mse_clip)
        self.n_clip_grid = int(n_clip_grid)
        self.eps = float(eps)

        if self.sym:
            self.qmax = float(2 ** (self.bits - 1) - 1)
            self.qmin = -self.qmax
        else:
            self.qmax = float(2 ** self.bits - 1)
            self.qmin = 0.0
        self.code_dtype = torch.int16 if self.bits > 8 else torch.int8

    # ------------------------------------------------------------------ fitting

    def fit(self, W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fit group scale/zero on W [out, in]. Grid is fitted on the original W."""
        Wf = W.detach().to(torch.float32)
        out_f, in_f = Wf.shape
        g = self.group_size

        pad = (-in_f) % g
        if pad:
            # Repeating the last column cannot change any group's min/max.
            Wf = torch.cat([Wf, Wf[:, -1:].expand(out_f, pad)], dim=1)
        groups = Wf.reshape(out_f, -1, g)

        if self.sym:
            amax = groups.abs().amax(dim=-1)
            if self.mse_clip:
                scale = self._mse_clip_scale(groups, amax, sym=True)
            else:
                scale = amax / self.qmax
            scale = scale.clamp_min(self.eps)
            zero = torch.zeros_like(scale)
        else:
            lo = groups.amin(dim=-1)
            hi = groups.amax(dim=-1)
            if self.mse_clip:
                scale, zero = self._mse_clip_scale_asym(groups, hi)
            else:
                scale = ((hi - lo) / self.qmax).clamp_min(self.eps)
                zero = lo

        g_idx = group_index(in_f, g)
        return scale, zero, g_idx

    def _mse_clip_scale(self, groups: torch.Tensor, amax: torch.Tensor, sym: bool = True):
        """Pick the per-group clipping ratio in (0,1] minimising reconstruction MSE."""
        ratios = torch.linspace(1.0, 0.5, self.n_clip_grid, device=groups.device)
        best_err = None
        best_idx = torch.zeros_like(amax, dtype=torch.long)
        for i, r in enumerate(ratios):
            scale = (amax * r / self.qmax).clamp_min(self.eps)
            q = torch.round(groups / scale.unsqueeze(-1)).clamp(self.qmin, self.qmax)
            err = (groups - q * scale.unsqueeze(-1)).pow(2).mean(dim=-1)
            if best_err is None:
                best_err = err
            else:
                better = err < best_err
                best_err = torch.where(better, err, best_err)
                best_idx = torch.where(better, torch.full_like(best_idx, i), best_idx)
        return (amax * ratios[best_idx] / self.qmax).clamp_min(self.eps)

    def _mse_clip_scale_asym(self, groups: torch.Tensor, hi: torch.Tensor):
        lo = groups.amin(dim=-1)
        ratios = torch.linspace(1.0, 0.5, self.n_clip_grid, device=groups.device)
        best_err = None
        best_idx = torch.zeros_like(hi, dtype=torch.long)
        for i, r in enumerate(ratios):
            span = ((hi - lo) * r).clamp_min(self.eps)
            scale = span / self.qmax
            q = torch.round((groups - lo.unsqueeze(-1)) / scale.unsqueeze(-1)).clamp(
                self.qmin, self.qmax
            )
            err = (groups - (q * scale.unsqueeze(-1) + lo.unsqueeze(-1))).pow(2).mean(dim=-1)
            if best_err is None:
                best_err = err
            else:
                better = err < best_err
                best_err = torch.where(better, err, best_err)
                best_idx = torch.where(better, torch.full_like(best_idx, i), best_idx)
        scale = (((hi - lo).clamp_min(self.eps)) * ratios[best_idx] / self.qmax)
        return scale.clamp_min(self.eps), lo

    # ---------------------------------------------------------------- applying

    def expand_params(self, scale, zero, g_idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """[out, G] group params -> [out, in] per-channel params."""
        return scale.index_select(1, g_idx.long()), zero.index_select(1, g_idx.long())

    def quantize_column(self, w: torch.Tensor, scale_col: torch.Tensor,
                        zero_col: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantise a single column w [out] with its per-row group params [out]."""
        if self.sym:
            q = torch.round(w / scale_col).clamp(self.qmin, self.qmax)
            dq = q * scale_col
        else:
            q = torch.round((w - zero_col) / scale_col).clamp(self.qmin, self.qmax)
            dq = q * scale_col + zero_col
        return q.to(self.code_dtype), dq

    def quantize(self, W: torch.Tensor, scale, zero, g_idx
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantise a whole matrix. Returns (q_int [out,in], W_hat [out,in] fp32)."""
        Wf = W.detach().to(torch.float32)
        sc, zr = self.expand_params(scale, zero, g_idx)
        if self.sym:
            q = torch.round(Wf / sc).clamp(self.qmin, self.qmax)
            dq = q * sc
        else:
            q = torch.round((Wf - zr) / sc).clamp(self.qmin, self.qmax)
            dq = q * sc + zr
        return q.to(self.code_dtype), dq

    def dequantize(self, q_int: torch.Tensor, scale, zero, g_idx) -> torch.Tensor:
        sc, zr = self.expand_params(scale, zero, g_idx)
        q = q_int.to(torch.float32)
        return q * sc if self.sym else q * sc + zr


# ======================================================================================
# Streaming second-order statistics
# ======================================================================================


class WeightedHessian:
    """
    Streaming accumulator for

        H_w = sum_t w_t x_t x_t^T                      (in x in)
        C_w = sum_t w_t (x~_t - x_t) x_t^T             (in x in, optional)

    Accumulated in FP32 by default; FP64 is available but note that T4 FP64 is
    ~1/32 of FP32 throughput, so use FP64 only for small diagnostic runs.
    """

    def __init__(self, d_in: int, dtype: str = "float32", device=None,
                 track_cross: bool = True):
        self.d_in = int(d_in)
        self.dtype = {"float32": torch.float32, "float64": torch.float64}[dtype]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.H = torch.zeros(d_in, d_in, dtype=self.dtype, device=self.device)
        self.Cw = (torch.zeros(d_in, d_in, dtype=self.dtype, device=self.device)
                   if track_cross else None)
        self.n_tokens = 0
        self.sum_w = 0.0
        self.sum_w2 = 0.0

    def add(self, X: torch.Tensor, X_ref: Optional[torch.Tensor] = None,
            w: Optional[torch.Tensor] = None) -> None:
        """X, X_ref: [d_in, T]. w: [T] positive token weights (mean-normalised)."""
        Xf = X.detach().to(device=self.device, dtype=self.dtype)
        if Xf.shape[0] != self.d_in:
            raise ValueError(f"expected {self.d_in} input channels, got {Xf.shape[0]}")
        t = Xf.shape[1]

        if w is None:
            wv = torch.ones(t, dtype=self.dtype, device=self.device)
        else:
            wv = torch.as_tensor(w, device=self.device, dtype=self.dtype).flatten()
            if wv.numel() != t:
                raise ValueError(f"weight length {wv.numel()} != {t} tokens")
            wv = wv.clamp_min(0.0)

        # H_w: scale columns by sqrt(w) then take the Gram matrix (v2 §9.3).
        Xw = Xf * wv.sqrt().unsqueeze(0)
        self.H.add_(Xw @ Xw.transpose(0, 1))

        # C_w uses the *unweighted* X on the right and w (not sqrt(w)) on the left.
        if self.Cw is not None and X_ref is not None:
            Xr = X_ref.detach().to(device=self.device, dtype=self.dtype)
            if Xr.shape != Xf.shape:
                raise ValueError(f"reference activation shape {tuple(Xr.shape)} "
                                 f"!= quantized shape {tuple(Xf.shape)}")
            diff = (Xr - Xf) * wv.unsqueeze(0)
            self.Cw.add_(diff @ Xf.transpose(0, 1))

        self.n_tokens += t
        self.sum_w += float(wv.sum())
        self.sum_w2 += float(wv.pow(2).sum())

    def ess_fraction(self) -> float:
        """Kish ESS / N over the accumulated pool."""
        if self.sum_w2 <= 0:
            return 0.0
        return float(self.sum_w ** 2 / self.sum_w2 / max(self.n_tokens, 1))

    def finalize(self) -> Dict[str, object]:
        return {
            "H": self.H,
            "Cw": self.Cw,
            "n_tokens": self.n_tokens,
            "ess_fraction": self.ess_fraction(),
            "dtype": str(self.dtype).replace("torch.", ""),
        }


# ======================================================================================
# Spectral diagnostics
# ======================================================================================


def _power_lambda_max(M: torch.Tensor, iters: int = 20) -> float:
    """Largest eigenvalue of a symmetric PSD matrix by power iteration (cheap: d²/iter)."""
    d = M.shape[0]
    v = torch.ones(d, dtype=torch.float32)
    v = v / v.norm()
    lam = 0.0
    for _ in range(iters):
        w = M @ v
        nrm = float(w.norm())
        if nrm < 1e-30:
            return 0.0
        v = w / nrm
        lam = nrm
    return lam


def spectral_summary(H: torch.Tensor, max_dim: int = 2560, topk: int = 64,
                     eps: float = 1e-12) -> Dict[str, float]:
    """
    Condition number / effective rank of a symmetric PSD matrix.

    Below `max_dim` this is an exact eigendecomposition. Above it, a full
    decomposition costs more than the rest of the layer's quantization (n=8960 is
    ~2.4e11 flops, i.e. ~2 minutes of CPU per call), so we use power iteration for
    lambda_max plus a q=`topk` randomised sketch for the effective rank, and report
    the condition number as None rather than inventing one.

    The *damped* condition number -- the one that actually matters for the solve -- is
    estimated after inversion and stored as `cond_damped_est`.
    """
    d = H.shape[0]
    out: Dict[str, float] = {"dim": d, "trace": float(torch.diagonal(H).sum())}
    Hs = H.detach().to(torch.float32).cpu()
    if d <= max_dim:
        ev = torch.linalg.eigvalsh(Hs).clamp_min(0.0)
        lam = ev.flip(0)                                # descending
        total = float(lam.sum())
        probs = (lam / total).clamp_min(eps) if total > 0 else lam
        ent = float(-(probs * probs.log()).sum())
        out.update({
            "lambda_max": float(lam[0]),
            "lambda_min": float(lam[-1]),
            "cond": float(lam[0] / max(float(lam[-1]), eps)),
            "eff_rank": float(torch.exp(torch.tensor(ent))),
            "eff_rank_norm": float(torch.exp(torch.tensor(ent)) / d),
            "n_below_1e-6_relmax": int((lam < 1e-6 * float(lam[0])).sum()),
            "partial": False,
        })
    else:
        lam_max = _power_lambda_max(Hs, iters=20)
        q = max(8, min(int(topk), d))
        try:
            _U, S, _V = torch.svd_lowrank(Hs, q=q, niter=2)
            lam_top = (S ** 2).clamp_min(0.0)
            share = lam_top / lam_top.sum().clamp_min(eps)
            eff = float(torch.exp(-(share.clamp_min(eps).log() * share).sum()))
            tail_mass = max(float(torch.diagonal(Hs).sum()) - float(lam_top.sum()), 0.0)
        except Exception:
            eff, tail_mass = float("nan"), float("nan")
        out.update({
            "lambda_max": float(lam_max),
            "lambda_min": None,
            "cond": None,
            "eff_rank_topk": eff,
            "eff_rank_topk_norm": eff / d if eff == eff else None,
            "residual_trace": tail_mass,
            "partial": True,
            "topk": q,
        })
    return out


# ======================================================================================
# Inverse
# ======================================================================================


def inverse_cholesky(H: torch.Tensor, jitters: Sequence[float] = (0.0, 1e-7, 1e-5, 1e-3, 1e-1)
                     ) -> Tuple[torch.Tensor, float]:
    """
    H^{-1} via Cholesky with escalating jitter, then an eigendecomposition fallback.

    Returns (Hinv, jitter_used). jitter_used == -1.0 means the eigen fallback ran.
    """
    d = H.shape[0]
    eye = torch.eye(d, dtype=H.dtype, device=H.device)
    for j in jitters:
        try:
            A = H if j == 0.0 else H + j * eye
            L = torch.linalg.cholesky(A)
            return torch.cholesky_inverse(L), float(j)
        except Exception:
            continue
    evals, evecs = torch.linalg.eigh(H)
    evals = evals.clamp_min(1e-8)
    return (evecs * (1.0 / evals)) @ evecs.transpose(0, 1), -1.0


# ======================================================================================
# Objective (used by the uniform-equivalence unit test)
# ======================================================================================


def reconstruction_objective(W_hat: torch.Tensor, W: torch.Tensor, H: torch.Tensor,
                             Cw: Optional[torch.Tensor] = None) -> float:
    """
    Evaluate 1/2 tr(DW H DW^T) - tr(DW (W Cw)^T) for DW = W_hat - W.

    H must be the *unridged* weighted Gram and Cw the raw weighted cross-covariance.
    This is the same quantity the solver minimises, so it is the ground truth for
    the "does the solver actually improve the objective" check.
    """
    DW = (W_hat.detach().to(torch.float32) - W.detach().to(torch.float32))
    Hf = H.detach().to(torch.float32)
    val = 0.5 * torch.sum(DW * (DW @ Hf))
    if Cw is not None:
        C = W.detach().to(torch.float32) @ Cw.detach().to(torch.float32)
        val = val - torch.sum(DW * C)
    return float(val)


# ======================================================================================
# The solver
# ======================================================================================


@dataclass
class QuantizedLinear:
    name: str
    q_int: torch.Tensor            # [out, in] integer codes (int8 / int16)
    scales: torch.Tensor           # [out, n_groups] fp32
    zeros: torch.Tensor            # [out, n_groups] fp32
    g_idx: torch.Tensor            # [in] int32
    W_hat: torch.Tensor            # dequantised, in the module's dtype
    meta: Dict[str, object] = field(default_factory=dict)

    def state_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "q_int": self.q_int.cpu(),
            "scales": self.scales.cpu(),
            "zeros": self.zeros.cpu(),
            "g_idx": self.g_idx.cpu(),
            "W_hat": self.W_hat.detach().cpu(),
            "meta": self.meta,
        }


def quantize_linear(
    name: str,
    W: torch.Tensor,
    H: Optional[torch.Tensor],
    Cw: Optional[torch.Tensor],
    cfg: QuantConfig,
    verbose: bool = False,
    compute_diagnostics: bool = True,
) -> Optional[QuantizedLinear]:
    """
    Quantise one linear layer.

    H is the FP32/FP64 weighted Gram (X D X^T). Pass H=None for RTN (no
    second-order solve). Cw is the weighted cross-covariance; pass None to disable
    the asymmetric term (weighted symmetric GPTQ).
    """
    t0 = time.time()
    W_true = W.detach().to(torch.float32).clone()          # pristine original
    out_f, in_f = W_true.shape
    if in_f < cfg.min_in_features:
        return None

    quant = GroupQuantizer(bits=cfg.bits, group_size=cfg.group_size, sym=cfg.sym,
                           mse_clip=cfg.mse_clip, n_clip_grid=cfg.n_clip_grid)

    # ---- 1. fit the grid on the ORIGINAL W (v1 §2.5: do not fit on the shifted matrix)
    scales, zeros, g_idx = quant.fit(W_true)

    # ---- 2. RTN shortcut: no Hessian, no sequential compensation
    if H is None:
        q_int, W_hat = quant.quantize(W_true, scales, zeros, g_idx)
        meta = {
            "method": "rtn", "bits": cfg.bits, "group_size": cfg.group_size,
            "sym": cfg.sym, "n_tokens": 0,
            "rel_fro_err": float((W_hat - W_true).norm() / W_true.norm().clamp_min(1e-12)),
            "time_s": round(time.time() - t0, 3),
        }
        return QuantizedLinear(name, q_int, scales, zeros, g_idx, W_hat.to(W.dtype), meta)

    # ---- 3. spectral diagnostics on the undamped Hessian (sampled: see diag_every_n)
    diag = (spectral_summary(H, cfg.diag_max_dim, cfg.diag_topk)
            if compute_diagnostics else {"skipped": True, "dim": in_f})

    Hd = H.detach().to(torch.float32).clone()
    dead = torch.diagonal(Hd) == 0
    n_dead = int(dead.sum())
    if n_dead:
        Hd[dead, dead] = 1.0
        W_solve = W_true.clone()
        W_solve[:, dead] = 0.0
    else:
        W_solve = W_true

    # ---- 4. adaptive ridge damping (v1 §2.5 step iv)
    trace_mean = float(torch.diagonal(Hd).mean())
    damp = float(cfg.damp_percent) * trace_mean
    if damp <= 0:
        damp = 1e-8 * max(1.0, float(torch.diagonal(Hd).abs().max()))
    idx = torch.arange(in_f, device=Hd.device)
    Hd[idx, idx] += damp
    Hinv, jitter_used = inverse_cholesky(Hd)

    # damped condition number -- the one that actually governs the solve. lambda_max(H)
    # by power iteration, lambda_max(H^-1) by power iteration: both are O(d²) per step,
    # so this is essentially free even at d=8960.
    if compute_diagnostics:
        try:
            lam_max_h = _power_lambda_max(Hd.detach().float().cpu(), iters=15)
            lam_max_hinv = _power_lambda_max(Hinv.detach().float().cpu(), iters=15)
            diag["cond_damped_est"] = float(lam_max_h * lam_max_hinv)
        except Exception:
            diag["cond_damped_est"] = None

    # ---- 5. asymmetric pre-shift P = W Cw Hinv  (exact quadratic optimum)
    use_asym = bool(cfg.asym) and (Cw is not None)
    if use_asym:
        P = (W_solve @ Cw.detach().to(torch.float32)) @ Hinv
    else:
        P = None

    # ---- 6. optional activation-order permutation (Hessian diagonal)
    if cfg.act_order:
        perm = torch.argsort(torch.diagonal(Hd), descending=True)
    else:
        perm = torch.arange(in_f, device=Hd.device)
    invperm = torch.argsort(perm)

    Hp = Hinv.index_select(0, perm).index_select(1, perm).contiguous()
    Wp = W_solve.index_select(1, perm).contiguous()
    Tmat = Wp if P is None else (Wp + P.index_select(1, perm))

    # per-(permuted)column group params, so the inner loop is a pure index
    groups_in_order = g_idx.long().index_select(0, perm)
    sc_cols = scales.index_select(1, groups_in_order)      # [out, in]
    zr_cols = zeros.index_select(1, groups_in_order)       # [out, in]

    Q = torch.zeros(out_f, in_f, dtype=torch.float32, device=W.device)
    Qint = torch.zeros(out_f, in_f, dtype=quant.code_dtype, device=W.device)

    # ---- 7. greedy OBC column updates
    if cfg.true_sequential:
        for j in range(in_f):
            w = Tmat[:, j]
            q_int, dq = quant.quantize_column(w, sc_cols[:, j], zr_cols[:, j])
            Q[:, j] = dq
            Qint[:, j] = q_int
            d = Hp[j, j]
            err = (w - dq) / d
            if j + 1 < in_f:
                Tmat[:, j + 1:] -= torch.outer(err, Hp[j, j + 1:])
    else:
        bs = max(1, int(cfg.blocksize))
        for i1 in range(0, in_f, bs):
            i2 = min(i1 + bs, in_f)
            cnt = i2 - i1
            W1 = Tmat[:, i1:i2].clone()
            Q1 = torch.zeros(out_f, cnt, dtype=torch.float32, device=W.device)
            Qi1 = torch.zeros(out_f, cnt, dtype=quant.code_dtype, device=W.device)
            Err1 = torch.zeros(out_f, cnt, dtype=torch.float32, device=W.device)
            H1 = Hp[i1:i2, i1:i2]
            for i in range(cnt):
                j = i1 + i
                w = W1[:, i]
                q_int, dq = quant.quantize_column(w, sc_cols[:, j], zr_cols[:, j])
                Q1[:, i] = dq
                Qi1[:, i] = q_int
                d = H1[i, i]
                err = (w - dq) / d
                W1[:, i:] -= torch.outer(err, H1[i, i:])
                Err1[:, i] = err
            Q[:, i1:i2] = Q1
            Qint[:, i1:i2] = Qi1
            if i2 < in_f:
                Tmat[:, i2:] -= Err1 @ Hp[i1:i2, i2:]

    # ---- 8. undo the permutation
    if cfg.act_order:
        Q = Q.index_select(1, invperm)
        Qint = Qint.index_select(1, invperm)

    W_hat = Q.to(W.dtype)
    obj = reconstruction_objective(Q, W_solve, H, Cw if use_asym else None)

    meta = {
        "method": "gptaq" if use_asym else "gptq",
        "bits": cfg.bits, "group_size": cfg.group_size, "sym": cfg.sym,
        "act_order": cfg.act_order, "true_sequential": cfg.true_sequential,
        "asym": use_asym, "damp": damp, "jitter_used": jitter_used,
        "n_dead_columns": n_dead, "n_tokens": H.shape[0],
        "objective": obj,
        "objective_per_token": obj / max(in_f, 1),
        "rel_fro_err": float((Q - W_solve).norm() / W_solve.norm().clamp_min(1e-12)),
        "max_abs_err": float((Q - W_solve).abs().max()),
        "hessian": diag,
        "time_s": round(time.time() - t0, 3),
    }
    if verbose:
        cond = diag.get("cond") or diag.get("cond_damped_est")
        cond_s = f"{cond:.2e}" if isinstance(cond, (int, float)) else "n/a"
        print(f"    {name:52s} {out_f:5d}x{in_f:<5d} "
              f"cond={cond_s} obj={obj:.3e} {meta['time_s']}s")
    return QuantizedLinear(name, Qint, scales, zeros, g_idx, W_hat, meta)


# ======================================================================================
# Activation capture
# ======================================================================================


def capture_inputs(wrapped, batch, modules: Dict[str, nn.Linear],
                   to_cpu_dtype=torch.float16) -> Dict[str, torch.Tensor]:
    """
    One forward pass, returning the input activation of each module as [in, T] on CPU.

    Hooks are removed immediately; nothing is retained on the GPU. The [1, T, in]
    activation is transposed to [in, T] so column t is exactly token t (B = 1).
    """
    buf: Dict[str, torch.Tensor] = {}
    handles = []

    def make(name):
        def hook(_mod, inp, _out):
            x = inp[0]
            if x.dim() == 3:
                x = x[0]
            buf[name] = x.detach().to(to_cpu_dtype).T.contiguous().cpu()
        return hook

    for name, mod in modules.items():
        handles.append(mod.register_forward_hook(make(name)))
    try:
        with torch.no_grad():
            wrapped.forward_hidden(batch.input_ids, batch.attention_mask)
    finally:
        for h in handles:
            h.remove()
    return buf


def capture_fp16_reference(wrapped, batch, modules, ref_cpu: Dict[str, torch.Tensor]
                           ) -> Dict[str, torch.Tensor]:
    """Forward pass with the original FP16 weights swapped in, capturing inputs."""
    swap_to_ref(wrapped, ref_cpu, len(wrapped.decoder_layers()))
    try:
        return capture_inputs(wrapped, batch, modules)
    finally:
        swap_to_current(wrapped)


# ======================================================================================
# Weight swapping (reference vs. partially quantized state)
# ======================================================================================


def swap_to_ref(wrapped, ref_cpu: Dict[str, torch.Tensor], upto: int) -> None:
    """Restore the original FP16 weights for layers 0..upto-1."""
    for layer_idx in range(min(upto, len(wrapped.decoder_layers()))):
        for name, mod in wrapped.layer_linears(layer_idx).items():
            if name in ref_cpu:
                mod.weight.data.copy_(ref_cpu[name].to(mod.weight.device))


def swap_to_current(wrapped, cur_cpu: Optional[Dict[str, torch.Tensor]] = None) -> None:
    """Copy back the quantized weights we have already produced."""
    if not cur_cpu:
        return
    for name, tens in cur_cpu.items():
        mod = wrapped.model.get_submodule(name)
        mod.weight.data.copy_(tens.to(mod.weight.device))


# ======================================================================================
# Model-level driver
# ======================================================================================


def quantize_model(
    wrapped,
    batches: Sequence,
    cfg: QuantConfig,
    verbose: bool = True,
    progress: Optional[Callable[[int, int, str], None]] = None,
    max_layers: Optional[int] = None,
) -> Dict[str, QuantizedLinear]:
    """
    Quantise every linear module of the backbone, layer by layer, in place.

    Two forward passes per layer when `cfg.drift_correction` is True:
      (a) with the original FP16 weights  -> X~  (the reference input)
      (b) with the layers < l already quantised -> X (the drifted input)
    That is the honest GPTAQ setting. Setting drift_correction=False collapses
    X~ := X, which is plain weighted GPTQ (the `asym=0`-style ablation of drift).

    Memory: reference weights and produced weights live in CPU RAM, so the GPU only
    ever holds the live model plus one layer's Hessian/cross-covariance.
    """
    n_layers = len(wrapped.decoder_layers())
    n_layers = min(n_layers, max_layers) if max_layers else n_layers

    ref_cpu: Dict[str, torch.Tensor] = {}
    if cfg.drift_correction:
        for l in range(n_layers):
            for name, mod in wrapped.layer_linears(l).items():
                ref_cpu[name] = mod.weight.detach().to(torch.float16).cpu()
        if verbose:
            total = sum(t.numel() for t in ref_cpu.values()) * 2 / 2 ** 30
            print(f"  cached FP16 reference weights: {total:.2f} GB in CPU RAM")

    results: Dict[str, QuantizedLinear] = {}
    cur_cpu: Dict[str, torch.Tensor] = {}
    t_start = time.time()

    for l in range(n_layers):
        linears = wrapped.layer_linears(l)
        names = [n for n, m in linears.items() if m.in_features >= cfg.min_in_features]
        mods = {n: linears[n] for n in names}
        if not names:
            continue

        accs = {
            n: WeightedHessian(mods[n].in_features, cfg.hessian_dtype,
                               track_cross=cfg.asym)
            for n in names
        }

        # (a) reference activations, if drift correction is on and the layer is not l=0
        need_ref = cfg.drift_correction and l > 0
        if need_ref:
            ref_buf: List[Dict[str, torch.Tensor]] = []
            swap_to_ref(wrapped, ref_cpu, l + 1)
            try:
                for b in batches:
                    ref_buf.append(capture_inputs(wrapped, b, mods))
            finally:
                swap_to_current(wrapped, cur_cpu)

        # (b) drifted activations under the current (partially quantized) model
        for bi, b in enumerate(batches):
            x = capture_inputs(wrapped, b, mods)
            w = b.weights
            for n in names:
                xr = ref_buf[bi][n] if need_ref else None
                accs[n].add(x[n], xr, w)

        if verbose:
            print(f"  layer {l:2d}/{n_layers-1}  tokens={accs[names[0]].n_tokens}  "
                  f"ess={accs[names[0]].ess_fraction():.3f}")

        # (c) solve each module
        diag_this_layer = (l % max(1, int(cfg.diag_every_n))) == 0
        for n in names:
            res = quantize_linear(n, mods[n].weight.detach(), accs[n].H, accs[n].Cw,
                                  cfg, verbose=verbose,
                                  compute_diagnostics=diag_this_layer)
            if res is None:
                continue
            wrapped.set_weight(n, res.W_hat)
            cur_cpu[n] = res.W_hat.detach().to(torch.float16).cpu()
            results[n] = res

        del accs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if progress is not None:
            progress(l, n_layers, f"layer {l} done")

    if verbose:
        print(f"  quantized {len(results)} modules in {time.time()-t_start:.1f}s")
    return results


def quantize_model_rtn(wrapped, cfg: QuantConfig, verbose: bool = True,
                       max_layers: Optional[int] = None) -> Dict[str, QuantizedLinear]:
    """Round-to-nearest baseline: same grid, no second-order solve."""
    n_layers = min(len(wrapped.decoder_layers()), max_layers or 10 ** 9)
    results: Dict[str, QuantizedLinear] = {}
    for l in range(n_layers):
        for name, mod in wrapped.layer_linears(l).items():
            res = quantize_linear(name, mod.weight.detach(), None, None, cfg, verbose=False)
            if res is None:
                continue
            wrapped.set_weight(name, res.W_hat)
            results[name] = res
    if verbose:
        print(f"  RTN applied to {len(results)} modules")
    return results


# ======================================================================================
# Checkpoint helpers
# ======================================================================================


def save_quant_state(results: Dict[str, QuantizedLinear], path: str, meta: Dict = None) -> str:
    payload = {name: r.state_dict() for name, r in results.items()}
    payload["__meta__"] = meta or {}
    torch.save(payload, path)
    return path


def load_quant_state(wrapped, path: str, apply: bool = True) -> Dict[str, QuantizedLinear]:
    """Re-apply a saved fake-quant checkpoint to a freshly loaded FP16 model."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload.pop("__meta__", None)
    results: Dict[str, QuantizedLinear] = {}
    for name, state in payload.items():
        res = QuantizedLinear(
            name=name,
            q_int=state["q_int"],
            scales=state["scales"],
            zeros=state["zeros"],
            g_idx=state["g_idx"],
            W_hat=state["W_hat"],
            meta=state.get("meta", {}),
        )
        results[name] = res
        if apply:
            wrapped.set_weight(name, res.W_hat)
    return results


def summarize_results(results: Dict[str, QuantizedLinear]) -> Dict[str, object]:
    by_module: Dict[str, List[float]] = {}
    for name, r in results.items():
        key = name.split(".")[-1]
        by_module.setdefault(key, []).append(float(r.meta.get("rel_fro_err", 0.0)))
    total_obj = sum(float(r.meta.get("objective", 0.0)) for r in results.values())
    conds = sorted(c for c in (r.meta.get("hessian", {}).get("cond")
                               for r in results.values()) if c is not None)
    median_cond = conds[len(conds) // 2] if conds else None
    return {
        "n_modules": len(results),
        "rel_fro_err_by_module": {k: round(sum(v) / len(v), 6) for k, v in by_module.items()},
        "total_objective": total_obj,
        "median_cond": median_cond,
        "max_cond": max(conds) if conds else None,
        "total_time_s": round(sum(float(r.meta.get("time_s", 0)) for r in results.values()), 1),
    }
