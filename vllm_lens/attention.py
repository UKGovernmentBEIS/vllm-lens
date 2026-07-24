"""Offline reconstruction of attention patterns from captured Q/K.

vLLM's fused attention kernels (FlashAttention et al.) never materialize
the attention-weight matrix, so it cannot be captured directly.  What
*can* be captured — via ``extra_args={"output_qk": ...}`` — are the
post-RoPE query and key tensors at the input of each attention layer,
together with the exact parameters vLLM hands the kernel (scale, logit
soft-cap, sliding window, ALiBi slopes, attention sinks).  This module
replays the kernel's computation from those:

    weights = softmax(mask(soft_cap(Q @ K^T * scale)) + alibi)

The replay is exact for standard softmax attention (up to floating-point
accumulation order).  Multi-head latent attention (MLA) is rejected at
capture time — its Q/K live in a compressed latent space.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from jaxtyping import Float

__all__ = ["attention_patterns", "compute_attention_weights"]


def compute_attention_weights(
    q: Float[torch.Tensor, "q_len num_heads head_dim"],  # type: ignore[reportUndefinedVariable]
    k: Float[torch.Tensor, "kv_len num_kv_heads head_dim"],  # type: ignore[reportUndefinedVariable]
    *,
    scale: float,
    causal: bool = True,
    q_start: int | None = None,
    sliding_window: tuple[int, int] = (-1, -1),
    logits_soft_cap: float = 0.0,
    alibi_slopes: torch.Tensor | Sequence[float] | None = None,
    sinks: torch.Tensor | Sequence[float] | None = None,
    dtype: torch.dtype = torch.float32,
) -> Float[torch.Tensor, "num_heads q_len kv_len"]:  # type: ignore[reportUndefinedVariable]
    """Reconstruct attention weights from captured Q/K.

    Args:
        q: Queries ``(q_len, num_heads, head_dim)``.  May cover fewer
            positions than ``k`` (e.g. a single decode step).
        k: Keys ``(kv_len, num_kv_heads, head_dim)``.  Grouped-query
            attention is handled by repeating KV heads.
        scale: Softmax scale — pass the captured ``scale`` metadata
            (vLLM has already resolved model-specific conventions such
            as Granite's ``attention_multiplier``).
        causal: Apply a causal mask (decoder attention).
        q_start: Absolute position of ``q[0]`` within the sequence the
            keys cover.  Defaults to ``kv_len - q_len`` (queries are the
            final positions).
        sliding_window: vLLM-normalized ``(left, right)`` window sizes;
            ``-1`` means unbounded.  Pass the captured ``sliding_window``
            metadata.
        logits_soft_cap: Gemma-style tanh cap; ``0`` disables.
        alibi_slopes: Per-query-head ALiBi slopes, or ``None``.
        sinks: Per-query-head attention-sink logits, or ``None``.  When
            present, the sink participates in the softmax denominator
            (rows then sum to less than 1), matching the kernel.
        dtype: Computation dtype for the logits/softmax (fp32 default,
            matching kernel accumulation).

    Returns:
        Attention weights ``(num_heads, q_len, kv_len)``.  Fully-masked
        rows are all-zero rather than NaN.
    """
    if q.ndim != 3 or k.ndim != 3:
        raise ValueError(
            f"expected 3-D q and k, got {tuple(q.shape)}, {tuple(k.shape)}"
        )
    q_len, num_heads, head_dim = q.shape
    kv_len, num_kv_heads, k_head_dim = k.shape
    if head_dim != k_head_dim:
        raise ValueError(f"head_dim mismatch: q {head_dim} vs k {k_head_dim}")
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads ({num_heads}) not divisible by num_kv_heads ({num_kv_heads})"
        )
    if q_start is None:
        q_start = kv_len - q_len
    if q_start < 0:
        raise ValueError(f"q_len ({q_len}) exceeds kv_len ({kv_len})")

    qf = q.to(dtype)
    kf = k.to(dtype)
    if num_kv_heads < num_heads:
        kf = kf.repeat_interleave(num_heads // num_kv_heads, dim=1)

    logits = torch.einsum("qhd,khd->hqk", qf, kf) * scale

    if logits_soft_cap and logits_soft_cap > 0:
        logits = logits_soft_cap * torch.tanh(logits / logits_soft_cap)

    # Relative distance (key position j) - (absolute query position i).
    q_pos = torch.arange(q_start, q_start + q_len, device=logits.device)
    k_pos = torch.arange(kv_len, device=logits.device)
    rel = k_pos.unsqueeze(0) - q_pos.unsqueeze(1)  # (q_len, kv_len)

    if alibi_slopes is not None:
        slopes = torch.as_tensor(alibi_slopes, dtype=dtype, device=logits.device)
        if slopes.numel() != num_heads:
            raise ValueError(
                f"alibi_slopes has {slopes.numel()} entries, expected {num_heads}"
            )
        # FlashAttention's ALiBi bias is slope * -(i - j) on the causal
        # region; masked positions are excluded below anyway.
        logits = logits + slopes.view(-1, 1, 1) * rel.to(dtype)

    left, right = sliding_window
    mask = torch.zeros(q_len, kv_len, dtype=torch.bool, device=logits.device)
    if causal:
        mask |= rel > (right if right >= 0 else 0)
    elif right >= 0:
        mask |= rel > right
    if left >= 0:
        mask |= rel < -left
    logits = logits.masked_fill(mask.unsqueeze(0), float("-inf"))

    if sinks is not None:
        sink = torch.as_tensor(sinks, dtype=dtype, device=logits.device)
        if sink.numel() != num_heads:
            raise ValueError(f"sinks has {sink.numel()} entries, expected {num_heads}")
        sink_col = sink.view(-1, 1, 1).expand(num_heads, q_len, 1)
        weights = torch.softmax(torch.cat([logits, sink_col], dim=-1), dim=-1)
        weights = weights[..., :-1]
    else:
        weights = torch.softmax(logits, dim=-1)

    # Fully-masked rows softmax to NaN; zero them instead.
    return torch.nan_to_num(weights, nan=0.0)


def attention_patterns(
    activations: Mapping[str, Any],
    layer: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> Float[torch.Tensor, "num_heads seq_len seq_len"]:  # type: ignore[reportUndefinedVariable]
    """Reconstruct one layer's full attention pattern from captured activations.

    Convenience wrapper over :func:`compute_attention_weights` for the
    ``output.activations`` dict produced by ``extra_args={"output_qk": ...}``
    (keys ``attn_q``, ``attn_k``, ``qk_layers``, ``qk_meta``).

    Args:
        activations: The activations mapping from a capture-enabled request.
        layer: Global decoder-layer index to reconstruct (must be one of
            the captured ``qk_layers``).
        dtype: Computation dtype.

    Returns:
        Attention weights ``(num_heads, seq_len, seq_len)`` over every
        captured position (prompt + generated tokens).
    """
    try:
        layers: list[int] = list(activations["qk_layers"])
        q_all = activations["attn_q"]
        k_all = activations["attn_k"]
        meta_list = activations["qk_meta"]
    except KeyError as e:
        raise KeyError(
            f"activations dict has no {e.args[0]!r} — was the request made "
            'with extra_args={"output_qk": ...}?'
        ) from None
    if layer not in layers:
        raise ValueError(f"layer {layer} not captured (captured layers: {layers})")
    i = layers.index(layer)
    meta = meta_list[i]

    if meta.get("use_alibi_sqrt"):
        raise NotImplementedError(
            "use_alibi_sqrt models are not supported by the offline replay"
        )

    sw = meta.get("sliding_window") or (-1, -1)
    return compute_attention_weights(
        q_all[i],
        k_all[i],
        scale=float(meta["scale"]),
        causal=True,
        q_start=0,
        sliding_window=(int(sw[0]), int(sw[1])),
        logits_soft_cap=float(meta.get("logits_soft_cap") or 0.0),
        alibi_slopes=meta.get("alibi_slopes"),
        sinks=meta.get("sinks"),
        dtype=dtype,
    )
