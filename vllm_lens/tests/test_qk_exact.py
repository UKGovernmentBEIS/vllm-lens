"""Exact (bit-for-bit) tests of the Q/K capture plumbing.

The HF-parity tests answer "are these the right tensors?".  These answer
"did we deliver *exactly* what the attention kernel saw?" — with no
tolerance at all.  A plain ``torch`` pre-hook is installed on the very same
``Attention`` modules (``_debug_raw_qk_*`` RPCs) recording each forward's
whole-batch Q/K, with no per-request slicing, merging or serialization.
The production capture — after ``query_start_loc`` slicing, per-chunk
accumulation, TP head-shard concat + KV-replication dedupe, PP layer concat
and the activation wire format — must ``torch.equal`` it.

Single-request engines are used so the raw whole-batch capture *is* the
request; per-request demultiplexing of a mixed batch is covered by the fp32
HF test (``test_qk_match.py::test_batched_requests_demultiplex``).
"""

import gc
import json
import pickle

import pytest
import torch
from vllm import LLM, SamplingParams

from vllm_lens._helpers._activation_store import ActivationStore
from vllm_lens._helpers._serialize import (
    decode_activations,
    serialize_activations,
    serialize_activations_binary,
)

from .conftest import LAYER_IDX, MODEL_NAME, NUM_LAYERS, PROMPT

_LONG_PROMPT = " ".join(
    ["Alan Turing described a universal computing machine in 1936."] * 40
)


def _engine(**kwargs) -> LLM:
    return LLM(model=MODEL_NAME, dtype="auto", gpu_memory_utilization=0.3, **kwargs)


def _teardown(llm: LLM) -> None:
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def _capture_with_raw(llm: LLM, prompt: str, layers, max_tokens: int):
    """Run one request with production Q/K capture and raw reference hooks."""
    layer_list = list(range(NUM_LAYERS)) if layers is True else list(layers)
    llm.collective_rpc("_debug_raw_qk_install", args=(layer_list,))
    try:
        sp = SamplingParams(
            temperature=0.0, max_tokens=max_tokens, extra_args={"output_qk": layers}
        )
        out = llm.generate([prompt], sp)[0]
        raws = [pickle.loads(b) for b in llm.collective_rpc("_debug_raw_qk_pop")]
    finally:
        llm.collective_rpc("_debug_raw_qk_remove")
    return out, raws


def _expected_from_raw(
    raws, acts
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Merge raw per-rank captures the way the plugin is specified to.

    TP: concat Q along heads in tp_rank order; K from every ``stride``-th
    rank (vLLM replicates each KV head across ``tp_size // total_kv``
    consecutive ranks) — and the replicas themselves must be identical.
    PP: each layer lives on exactly one stage.
    """
    by_pp: dict[int, list] = {}
    for r in raws:
        by_pp.setdefault(r["pp_rank"], []).append(r)
    q_exp: dict[int, torch.Tensor] = {}
    k_exp: dict[int, torch.Tensor] = {}
    for pp_rank in sorted(by_pp):
        group = sorted(by_pp[pp_rank], key=lambda r: r["tp_rank"])
        tp_size = len(group)
        for li, layer in enumerate(acts["qk_layers"]):
            parts = [r["layers"][layer] for r in group if layer in r["layers"]]
            if not parts:
                continue  # layer on another stage
            assert len(parts) == tp_size, f"layer {layer}: {len(parts)}/{tp_size} ranks"
            total_kv = int(acts["qk_meta"][li]["num_kv_heads_total"])
            stride = max(1, tp_size // total_kv)
            for start in range(0, tp_size, stride):
                for j in range(start + 1, start + stride):
                    assert torch.equal(parts[start]["k"], parts[j]["k"]), (
                        f"layer {layer}: KV replica on tp_rank {j} differs from {start}"
                    )
            q_exp[layer] = torch.cat([p["q"] for p in parts], dim=1)
            k_exp[layer] = torch.cat(
                [parts[i]["k"] for i in range(0, tp_size, stride)], dim=1
            )
    return q_exp, k_exp


def _assert_exact(out, raws) -> None:
    acts = out.activations
    n_prompt = len(out.prompt_token_ids)
    n_gen = len(out.outputs[0].token_ids)
    n = acts["attn_q"].shape[1]
    assert n == n_prompt + n_gen - 1
    q_exp, k_exp = _expected_from_raw(raws, acts)
    assert set(q_exp) == set(acts["qk_layers"])
    for li, layer in enumerate(acts["qk_layers"]):
        raw_n = q_exp[layer].shape[0]
        # The kernel may see one extra position (the final sampled token's
        # forward) that the capture trims; nothing else may differ.
        assert raw_n - n in (0, 1), f"layer {layer}: raw {raw_n} vs captured {n}"
        assert torch.equal(acts["attn_q"][li], q_exp[layer][:n]), f"attn_q L{layer}"
        assert torch.equal(acts["attn_k"][li], k_exp[layer][:n]), f"attn_k L{layer}"
        assert acts["attn_q"].dtype == q_exp[layer].dtype


# -- single GPU -------------------------------------------------------------


@pytest.fixture(scope="module")
def llm():
    e = _engine()
    yield e
    _teardown(e)


def test_prefill_and_decode_exact(llm):
    out, raws = _capture_with_raw(
        llm, PROMPT, [LAYER_IDX, NUM_LAYERS - 1], max_tokens=4
    )
    assert len(raws) == 1
    _assert_exact(out, raws)


def test_all_layers_exact(llm):
    out, raws = _capture_with_raw(llm, PROMPT, True, max_tokens=1)
    assert out.activations["qk_layers"] == list(range(NUM_LAYERS))
    _assert_exact(out, raws)


def test_wire_roundtrip_exact(llm):
    """Base64 and binary wire formats reproduce the tensors bit-for-bit."""
    out, _ = _capture_with_raw(llm, PROMPT, [LAYER_IDX], max_tokens=2)
    acts = out.activations
    b64 = json.loads(json.dumps(serialize_activations(acts)))
    dec = decode_activations({"activations": b64})
    store = ActivationStore()
    binary = json.loads(json.dumps(serialize_activations_binary(acts, store)))
    dec_bin = decode_activations(
        {"activations": binary},
        fetch_bytes=lambda h: store.get(h)[0],  # type: ignore[index]
    )
    for d in (dec, dec_bin):
        assert torch.equal(d["attn_q"], acts["attn_q"])
        assert torch.equal(d["attn_k"], acts["attn_k"])
        assert d["attn_q"].dtype == acts["attn_q"].dtype
        assert d["qk_layers"] == acts["qk_layers"] and d["qk_meta"] == acts["qk_meta"]


def test_chunked_prefill_exact():
    """Per-chunk accumulation reassembles exactly what the kernel saw."""
    e = _engine(enable_chunked_prefill=True, max_num_batched_tokens=64)
    try:
        out, raws = _capture_with_raw(e, _LONG_PROMPT, [LAYER_IDX], max_tokens=2)
        assert len(out.prompt_token_ids) > 128, "prompt must span several chunks"
        _assert_exact(out, raws)
    finally:
        _teardown(e)


# -- parallel ---------------------------------------------------------------

_needs_2 = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")


@_needs_2
def test_tp2_exact():
    e = _engine(tensor_parallel_size=2)
    try:
        out, raws = _capture_with_raw(e, PROMPT, [LAYER_IDX], max_tokens=3)
        assert sorted(r["tp_rank"] for r in raws) == [0, 1]
        _assert_exact(out, raws)
    finally:
        _teardown(e)


@_needs_2
def test_pp2_exact():
    e = _engine(pipeline_parallel_size=2)
    try:
        out, raws = _capture_with_raw(e, PROMPT, True, max_tokens=3)
        assert sorted(r["pp_rank"] for r in raws) == [0, 1]
        assert out.activations["qk_layers"] == list(range(NUM_LAYERS))
        _assert_exact(out, raws)
    finally:
        _teardown(e)
