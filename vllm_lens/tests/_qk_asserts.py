"""Shared assertions for attention-pattern tests.

Two kinds of comparison, deliberately kept apart:

- :func:`assert_attention_close` — the **fp32 ground-truth check** against
  HuggingFace eager attention.  With both engines in true fp32
  (``VLLM_FLOAT32_MATMUL_PRECISION=highest`` and ``TRITON_F32_DEFAULT=ieee``
  on the vLLM side, ``allow_tf32=False`` for HF) the only remaining
  differences are accumulation order, so a tight absolute tolerance on the
  probabilities is a near-exact test.
- :func:`assert_attention_tv` — a loose **bf16 canary** for the production
  dtype (used by the opt-in multi-architecture sweep).  Two valid bf16
  computations through different kernels legitimately disagree on
  near-tied rows, so this bounds per-row total variation
  (``0.5 * Σ|got - want|`` ∈ [0, 1]; a mis-attended row scores ≈ 1) rather
  than anything elementwise.  It is a sanity net, not the oracle.

Exact plumbing (hook firing, per-request slicing, TP/PP merge, wire
round-trip) is tested bit-for-bit elsewhere (``test_qk_exact.py``); neither
helper here is meant to stand in for that.
"""

from __future__ import annotations

import torch

# fp32-vs-fp32 with TF32 disabled on both sides (VLLM_FLOAT32_MATMUL_PRECISION
# =highest, TRITON_F32_DEFAULT=ieee, torch allow_tf32=False).  Measured on
# H100, Qwen2.5-0.5B, layers 0/1/2/11, 5- and 168-token prompts: max |Δp|
# 3.5e-6 … 2.8e-5 (accumulation order only).  Without TRITON_F32_DEFAULT=ieee
# the Triton fp32 attention kernel uses TF32 and max |Δp| jumps to ~3e-3 —
# so this bound also guards the test setup itself.
FP32_ATOL = 1e-4

# bf16 canary: measured on H100 (FlashInfer) vs HF eager bf16 across
# Qwen2.5-0.5B / 1.5B layers 2–27: mean row TV 0.015–0.03, max row TV ≤ 0.27
# (a diffuse early-layer row).  A wrong row scores TV ≈ 1.
MEAN_TV_MAX = 0.05
ROW_TV_MAX = 0.4


def row_total_variation(got: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
    """Per-row total variation distance, shape (..., rows)."""
    return 0.5 * (got - want).abs().sum(-1)


def assert_attention_close(
    got: torch.Tensor,
    want: torch.Tensor,
    *,
    label: str = "",
    atol: float = FP32_ATOL,
) -> None:
    """fp32 ground truth: every probability within ``atol`` of HF's."""
    assert got.shape == want.shape, f"{label} shape: {got.shape} vs {want.shape}"
    diff = (got.float() - want.float()).abs()
    max_diff = diff.max().item()
    assert torch.isfinite(got).all(), f"{label} non-finite weights"
    # Always report the measured error (visible with ``pytest -s``) so the
    # tolerance can be audited against real numbers, not just pass/fail.
    print(
        f"[qk fp32] {label}: max |Δp| = {max_diff:.3e}, mean = {diff.mean().item():.3e}"
    )
    assert max_diff <= atol, (
        f"{label} max |Δp| = {max_diff:.3e} > {atol} "
        f"(mean |Δp| = {diff.mean().item():.3e}, "
        f"max row TV = {row_total_variation(got, want).max().item():.3e})"
    )


def assert_attention_tv(
    got: torch.Tensor,
    want: torch.Tensor,
    *,
    label: str = "",
    mean_tv_max: float = MEAN_TV_MAX,
    row_tv_max: float = ROW_TV_MAX,
) -> None:
    """bf16 canary: bounded per-row total variation (see module docstring)."""
    assert got.shape == want.shape, f"{label} shape: {got.shape} vs {want.shape}"
    tv = row_total_variation(got, want)
    mean_tv = tv.mean().item()
    max_tv = tv.max().item()
    assert mean_tv < mean_tv_max, f"{label} mean row TV {mean_tv:.4f} ≥ {mean_tv_max}"
    assert max_tv < row_tv_max, f"{label} max row TV {max_tv:.4f} ≥ {row_tv_max}"
