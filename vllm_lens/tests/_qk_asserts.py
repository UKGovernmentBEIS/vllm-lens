"""Shared assertions for attention-pattern parity tests.

Metric design (deliberately length-independent — see PR discussion):

- Mean-abs-diff over a probability row shrinks as 1/n with sequence
  length (two disjoint one-hot rows differ by only 2/n), so it becomes
  vacuous for long prompts.  We assert on **per-row total variation**
  (``0.5 * Σ|got - want|``, in [0, 1]) instead: a row attending to
  entirely the wrong position scores TV ≈ 1 regardless of length.
- Diffuse (near-uniform) rows — ubiquitous in layer 0 — are genuinely
  noisy between two valid bf16 computations, both in TV and argmax.  So
  the strict argmax check is restricted to **confident rows**, where the
  reference's top-1 probability exceeds its runner-up by a margin; on
  those rows the reconstruction must pick the same position.
"""

from __future__ import annotations

import torch

# Calibrated from recorded sweep numbers (PR #34 discussion): correct
# reconstructions measured mean row TV ≈ 0.10 on the noisiest case
# (near-uniform layer-0 rows, where bf16 logit noise between two *valid*
# computations legitimately moves mass) and ≲ 0.02 elsewhere, while a
# mis-attended row scores TV ≈ 1.  The sharp check for "attends to the
# wrong place" is the confident-row argmax below, which is immune to
# diffuse-row noise.
MEAN_TV_MAX = 0.15
ROW_TV_MAX = 0.6
CONFIDENT_MARGIN = 0.05
CONFIDENT_AGREE_MIN = 0.98


def row_total_variation(got: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
    """Per-row total variation distance, shape (..., rows)."""
    return 0.5 * (got - want).abs().sum(-1)


def assert_attention_matches(
    got: torch.Tensor,
    want: torch.Tensor,
    *,
    label: str = "",
    mean_tv_max: float = MEAN_TV_MAX,
    row_tv_max: float = ROW_TV_MAX,
) -> None:
    """Assert two (num_heads, q_len, kv_len) attention tensors agree.

    Bounds mean and worst-case per-row total variation, and requires
    argmax agreement on rows where the reference is confident.
    """
    assert got.shape == want.shape, f"{label} shape: {got.shape} vs {want.shape}"

    tv = row_total_variation(got, want)
    mean_tv = tv.mean().item()
    max_tv = tv.max().item()
    assert mean_tv < mean_tv_max, f"{label} mean row TV {mean_tv:.4f}"
    assert max_tv < row_tv_max, f"{label} max row TV {max_tv:.4f}"

    top2 = want.topk(min(2, want.shape[-1]), dim=-1).values
    if top2.shape[-1] < 2:
        return
    confident = (top2[..., 0] - top2[..., 1]) > CONFIDENT_MARGIN
    if confident.sum().item() == 0:
        return
    agree = (got.argmax(-1) == want.argmax(-1))[confident].float().mean().item()
    assert agree >= CONFIDENT_AGREE_MIN, (
        f"{label} argmax agreement on {int(confident.sum())} confident rows: "
        f"{agree:.2%}"
    )
