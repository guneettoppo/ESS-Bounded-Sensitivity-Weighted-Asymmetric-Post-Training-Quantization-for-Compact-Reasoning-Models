"""
EB-SW-GPTAQ — research library for the BTP proxy-gate study.

Modules
-------
config              : dataclass configs + YAML loading
logging_utils       : artifact paths, JSON/JSONL IO, environment manifest, seeding
model_loader        : FP16 loader with FP32 lm_head guard, ModelWrapper
batching            : CalibBatch construction, causal/valid-position bookkeeping
calibration_data    : training-split prompt sources, mixed-corpus assembly, leak audit
trace_generation    : self-generated CoT traces (deterministic primary decoding)
quant_core          : group quantizer + weighted GPTQ/GPTAQ solver (the method)
probe_quantization  : RTN / GPTQ probe checkpoints used for the JSD signal
jsd                 : FP32 log-softmax, exact + top-k JSD, KL, strata
score_weights       : score -> weights (temperature, ESS bisection, controls)
activation_capture  : hooks that return per-module calibration inputs
geometry            : covariance spectra, effective rank, principal angles, overlap
evaluation          : held-out teacher-forced KL/JSD, perplexity, generation scoring
statistics          : paired bootstrap CI, McNemar, rank correlation
quantizer_adapter   : integer-code export / packing stub for real kernels later
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
