"""CPU-only unit tests for ``vllm_lens.attention.compute_attention_weights``.

Each variant is checked against a naive per-element reference
implementation (and SDPA for the plain-causal case).  No GPU or engine
required.
"""

import math

import pytest
import torch

from vllm_lens.attention import attention_patterns, compute_attention_weights


def _naive_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    scale: float,
    causal: bool = True,
    q_start: int | None = None,
    sliding_window: tuple[int, int] = (-1, -1),
    logits_soft_cap: float = 0.0,
    alibi_slopes: list[float] | None = None,
    sinks: list[float] | None = None,
) -> torch.Tensor:
    """Triple-loop reference: unvectorized, but obviously correct."""
    q_len, num_heads, _ = q.shape
    kv_len, num_kv_heads, _ = k.shape
    group = num_heads // num_kv_heads
    if q_start is None:
        q_start = kv_len - q_len
    left, right = sliding_window

    out = torch.zeros(num_heads, q_len, kv_len)
    for h in range(num_heads):
        for i in range(q_len):
            i_abs = q_start + i
            logits = []
            for j in range(kv_len):
                rel = j - i_abs
                if causal and rel > max(right, 0):
                    logits.append(float("-inf"))
                    continue
                if not causal and right >= 0 and rel > right:
                    logits.append(float("-inf"))
                    continue
                if left >= 0 and rel < -left:
                    logits.append(float("-inf"))
                    continue
                x = float(q[i, h] @ k[j, h // group]) * scale
                if logits_soft_cap > 0:
                    x = logits_soft_cap * math.tanh(x / logits_soft_cap)
                if alibi_slopes is not None:
                    x += alibi_slopes[h] * rel
                logits.append(x)
            row = torch.tensor(logits)
            if sinks is not None:
                row = torch.cat([row, torch.tensor([sinks[h]])])
            probs = torch.softmax(row, dim=-1)
            if sinks is not None:
                probs = probs[:-1]
            out[h, i] = torch.nan_to_num(probs, nan=0.0)
    return out


def _rand_qk(q_len=6, kv_len=6, heads=4, kv_heads=4, dim=8, seed=0):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(q_len, heads, dim, generator=gen)
    k = torch.randn(kv_len, kv_heads, dim, generator=gen)
    return q, k


def _assert_matches_reference(q, k, **kwargs):
    got = compute_attention_weights(q, k, **kwargs)
    want = _naive_reference(q, k, **kwargs)
    assert got.shape == want.shape
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-4)
    return got


class TestComputeAttentionWeights:
    def test_plain_causal_matches_reference_and_sdpa(self):
        q, k = _rand_qk()
        scale = 1 / math.sqrt(q.shape[-1])
        got = _assert_matches_reference(q, k, scale=scale)

        # Cross-check against PyTorch SDPA-derived weights.
        qt = q.permute(1, 0, 2)
        kt = k.permute(1, 0, 2)
        logits = qt @ kt.transpose(-1, -2) * scale
        mask = torch.triu(torch.ones(6, 6, dtype=torch.bool), diagonal=1)
        sdpa_weights = torch.softmax(logits.masked_fill(mask, float("-inf")), dim=-1)
        torch.testing.assert_close(got, sdpa_weights, atol=1e-5, rtol=1e-4)

        # Rows are proper distributions.
        torch.testing.assert_close(got.sum(-1), torch.ones(4, 6), atol=1e-5, rtol=1e-5)

    def test_gqa_repeats_kv_heads(self):
        q, k = _rand_qk(heads=8, kv_heads=2)
        _assert_matches_reference(q, k, scale=0.3)

    def test_sliding_window(self):
        q, k = _rand_qk(q_len=8, kv_len=8)
        got = _assert_matches_reference(q, k, scale=0.3, sliding_window=(3, 0))
        # Position 7 must not attend further back than position 4.
        assert got[:, 7, :4].abs().max().item() == 0.0
        assert got[:, 7, 4:8].sum(-1).allclose(torch.ones(4), atol=1e-5)

    def test_logits_soft_cap(self):
        q, k = _rand_qk()
        # Large scale so capping actually changes the result.
        _assert_matches_reference(q, k, scale=5.0, logits_soft_cap=10.0)
        capped = compute_attention_weights(q, k, scale=5.0, logits_soft_cap=10.0)
        uncapped = compute_attention_weights(q, k, scale=5.0)
        assert not torch.allclose(capped, uncapped)

    def test_alibi_bias(self):
        q, k = _rand_qk()
        _assert_matches_reference(
            q, k, scale=0.3, alibi_slopes=[0.5, 0.25, 0.125, 0.0625]
        )

    def test_sinks_reduce_row_sums(self):
        q, k = _rand_qk()
        got = _assert_matches_reference(q, k, scale=0.3, sinks=[1.0, 2.0, 0.5, 0.0])
        assert (got.sum(-1) < 1.0).all()

    def test_q_start_decode_row_equivalence(self):
        """A single decode-step q must reproduce the full matrix's last row."""
        q, k = _rand_qk(q_len=6, kv_len=6)
        full = compute_attention_weights(q, k, scale=0.3)
        last = compute_attention_weights(q[-1:], k, scale=0.3)  # q_start defaults to 5
        torch.testing.assert_close(full[:, -1:, :], last)
        explicit = compute_attention_weights(q[-1:], k, scale=0.3, q_start=5)
        torch.testing.assert_close(last, explicit)

    def test_fully_masked_rows_are_zero(self):
        # Queries placed far past every key with a zero-width window:
        # every row is fully masked and must come back as zeros, not NaN.
        q, k = _rand_qk(q_len=2, kv_len=2)
        got = compute_attention_weights(
            q, k, scale=0.3, q_start=10, sliding_window=(0, 0)
        )
        assert not got.isnan().any()
        assert got.abs().max().item() == 0.0

    def test_shape_validation(self):
        q, k = _rand_qk(heads=3, kv_heads=2)
        with pytest.raises(ValueError, match="not divisible"):
            compute_attention_weights(q, k, scale=1.0)


class TestAttentionPatterns:
    def _fake_activations(self, **meta_overrides):
        q, k = _rand_qk(q_len=5, kv_len=5, heads=4, kv_heads=2)
        meta = {
            "scale": 0.25,
            "sliding_window": (-1, -1),
            "logits_soft_cap": 0.0,
            "alibi_slopes": None,
            "sinks": None,
            "use_alibi_sqrt": False,
        }
        meta.update(meta_overrides)
        return {
            "attn_q": q.unsqueeze(0),
            "attn_k": k.unsqueeze(0),
            "qk_layers": [7],
            "qk_meta": [meta],
        }

    def test_reconstructs_from_activations_dict(self):
        acts = self._fake_activations()
        got = attention_patterns(acts, 7)
        want = compute_attention_weights(
            acts["attn_q"][0], acts["attn_k"][0], scale=0.25
        )
        torch.testing.assert_close(got, want)

    def test_missing_layer_raises(self):
        with pytest.raises(ValueError, match="not captured"):
            attention_patterns(self._fake_activations(), 3)

    def test_missing_keys_raise(self):
        with pytest.raises(KeyError, match="output_qk"):
            attention_patterns({"residual_stream": torch.zeros(1)}, 0)

    def test_alibi_sqrt_rejected(self):
        with pytest.raises(NotImplementedError):
            attention_patterns(self._fake_activations(use_alibi_sqrt=True), 7)
