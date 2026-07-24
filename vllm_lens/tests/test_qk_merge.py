"""CPU-only unit tests for the Q/K multi-rank merge in _activations_plugin.

Builds fake per-rank payloads in the exact shape ``_build_qk_payload``
emits and checks TP head-concatenation order, KV-head replication
dedupe, and PP layer concatenation — including the merged per-head
metadata (ALiBi slopes / sinks follow the Q-head order).
"""

import pickle

import torch

from vllm_lens._activations_plugin import _merge_captured_qk, _merge_qk_parts


def _mk_part(
    *,
    tp_rank: int = 0,
    pp_rank: int = 0,
    tp_size: int = 1,
    layers: list[int] | None = None,
    n_q_heads: int = 2,
    n_kv_heads: int = 1,
    total_kv: int | None = None,
    pos: int = 3,
    head_size: int = 4,
    fill: float | None = None,
    alibi: bool = False,
) -> dict:
    layers = layers if layers is not None else [0, 1]
    n_layers = len(layers)
    value = float(tp_rank if fill is None else fill)
    q = torch.full((n_layers, pos, n_q_heads, head_size), value)
    k = torch.full((n_layers, pos, n_kv_heads, head_size), value)
    meta = [
        {
            "scale": 0.5,
            "sliding_window": [-1, -1],
            "logits_soft_cap": 0.0,
            "alibi_slopes": [float(tp_rank * n_q_heads + h) for h in range(n_q_heads)]
            if alibi
            else None,
            "sinks": None,
            "use_alibi_sqrt": False,
            "num_heads_local": n_q_heads,
            "num_kv_heads_local": n_kv_heads,
            "head_size": head_size,
            "num_kv_heads_total": total_kv if total_kv is not None else n_kv_heads,
        }
        for _ in layers
    ]
    return {
        "q": q,
        "k": k,
        "layers": list(layers),
        "meta": meta,
        "tp_rank": tp_rank,
        "pp_rank": pp_rank,
        "tp_size": tp_size,
    }


class TestMergeQKParts:
    def test_single_rank_passthrough(self):
        part = _mk_part()
        out = _merge_qk_parts([part])
        assert torch.equal(out["attn_q"], part["q"])
        assert torch.equal(out["attn_k"], part["k"])
        assert out["qk_layers"] == [0, 1]
        assert len(out["qk_meta"]) == 2

    def test_tp_concat_order(self):
        # TP=2, 2 distinct kv heads total (1 per rank): no dedupe, heads
        # concatenated in tp_rank order even if parts arrive shuffled.
        parts = [
            _mk_part(tp_rank=1, tp_size=2, total_kv=2, alibi=True),
            _mk_part(tp_rank=0, tp_size=2, total_kv=2, alibi=True),
        ]
        out = _merge_qk_parts(parts)
        assert out["attn_q"].shape == (2, 3, 4, 4)  # 2+2 query heads
        assert out["attn_k"].shape == (2, 3, 2, 4)  # 1+1 kv heads
        # Rank 0's heads (value 0.0) come first, rank 1's (value 1.0) after.
        assert out["attn_q"][0, 0, 0, 0].item() == 0.0
        assert out["attn_q"][0, 0, 2, 0].item() == 1.0
        assert out["attn_k"][0, 0, 0, 0].item() == 0.0
        assert out["attn_k"][0, 0, 1, 0].item() == 1.0
        # Per-query-head meta follows the Q order: rank0 heads 0,1 then
        # rank1 heads 2,3.
        assert out["qk_meta"][0]["alibi_slopes"] == [0.0, 1.0, 2.0, 3.0]
        assert out["qk_meta"][0]["num_heads_local"] == 4
        assert out["qk_meta"][0]["num_kv_heads_local"] == 2

    def test_kv_replication_dedupe(self):
        # TP=4 with only 2 kv heads total: each kv head is replicated on 2
        # consecutive ranks (stride 2) — keep ranks 0 and 2 only.
        parts = [
            _mk_part(tp_rank=r, tp_size=4, n_q_heads=1, n_kv_heads=1, total_kv=2)
            for r in range(4)
        ]
        out = _merge_qk_parts(parts)
        assert out["attn_q"].shape == (2, 3, 4, 4)  # all 4 ranks' q heads
        assert out["attn_k"].shape == (2, 3, 2, 4)  # ranks 0 and 2 only
        assert out["attn_k"][0, 0, 0, 0].item() == 0.0
        assert out["attn_k"][0, 0, 1, 0].item() == 2.0

    def test_pp_layer_concat(self):
        parts = [
            _mk_part(pp_rank=1, layers=[2, 3], fill=1.0),
            _mk_part(pp_rank=0, layers=[0, 1], fill=0.0),
        ]
        out = _merge_qk_parts(parts)
        assert out["qk_layers"] == [0, 1, 2, 3]
        assert out["attn_q"].shape[0] == 4
        # Lower pp_rank's layers come first regardless of input order.
        assert out["attn_q"][0, 0, 0, 0].item() == 0.0
        assert out["attn_q"][2, 0, 0, 0].item() == 1.0
        assert len(out["qk_meta"]) == 4

    def test_tp_and_pp_combined(self):
        parts = [
            _mk_part(tp_rank=t, pp_rank=p, tp_size=2, total_kv=2, layers=layers)
            for p, layers in ((0, [0]), (1, [1]))
            for t in (0, 1)
        ]
        out = _merge_qk_parts(parts)
        assert out["qk_layers"] == [0, 1]
        assert out["attn_q"].shape == (2, 3, 4, 4)
        assert out["attn_k"].shape == (2, 3, 2, 4)


class TestMergeCapturedQK:
    def test_decodes_pickled_rank_payloads(self):
        payload = pickle.dumps({"qk": _mk_part()})
        out = _merge_captured_qk([payload, None])
        assert out is not None
        assert out["attn_q"].shape == (2, 3, 2, 4)

    def test_all_none_returns_none(self):
        assert _merge_captured_qk([None, None]) is None
        assert _merge_captured_qk(None) is None
