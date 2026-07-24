"""Q/K capture under tensor and pipeline parallelism.

TP=2: every rank captures its own head shard; the plugin concatenates
along the head dimension in tp_rank order (Qwen2.5-0.5B has 2 kv heads,
so TP=2 gives 1 kv head per rank with no replication).  PP=2: each stage
captures its own layers, concatenated along the layer dimension.
"""

import gc

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from vllm_lens.attention import attention_patterns

from .conftest import LAYER_IDX, MODEL_NAME, NUM_LAYERS, PROMPT

pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires at least 2 GPUs"
)

_NUM_Q_HEADS = 14
_NUM_KV_HEADS = 2
_HEAD_SIZE = 64


@pytest.fixture(scope="module")
def hf_attention():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype="auto",
        device_map="cuda",
        attn_implementation="eager",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def get(layer: int) -> tuple[torch.Tensor, int]:
        token_ids = tokenizer(PROMPT).input_ids
        ids = torch.tensor([token_ids], device="cuda")
        with torch.no_grad():
            out = model(ids, output_attentions=True, use_cache=False)
        return out.attentions[layer][0].float().cpu(), len(token_ids)

    yield get
    del model
    gc.collect()
    torch.cuda.empty_cache()


def _run_engine(weights_layer: int, **engine_kwargs):
    llm = LLM(
        model=MODEL_NAME,
        dtype="auto",
        gpu_memory_utilization=0.3,
        **engine_kwargs,
    )
    try:
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            extra_args={"output_qk": True},
        )
        output = llm.generate([PROMPT], sampling_params)[0]
        acts = output.activations
        weights = attention_patterns(acts, weights_layer)
        return acts, weights
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()


def _assert_close(got: torch.Tensor, want: torch.Tensor):
    assert got.shape == want.shape, f"{got.shape} vs {want.shape}"
    mean_abs_diff = (got - want).abs().mean().item()
    assert mean_abs_diff < 1e-2, f"Mean abs diff too large: {mean_abs_diff:.6f}"


def test_tp2_merges_head_shards(hf_attention):
    hf_weights, n = hf_attention(LAYER_IDX)
    acts, weights = _run_engine(LAYER_IDX, tensor_parallel_size=2)

    # Global head counts after the TP merge.
    assert acts["attn_q"].shape[2] == _NUM_Q_HEADS
    assert acts["attn_k"].shape[2] == _NUM_KV_HEADS
    assert acts["attn_q"].shape[0] == NUM_LAYERS
    meta = acts["qk_meta"][LAYER_IDX]
    assert meta["num_heads_local"] == _NUM_Q_HEADS
    assert meta["num_kv_heads_local"] == _NUM_KV_HEADS

    # Head order must be correct, not just head count — a wrong concat
    # order would scramble per-head patterns and break the HF match.
    _assert_close(weights[:, :n, :n], hf_weights)


def test_pp2_merges_layer_shards(hf_attention):
    # LAYER_IDX (=2) lives on stage 0; NUM_LAYERS-4 on stage 1.
    late_layer = NUM_LAYERS - 4
    hf_weights, n = hf_attention(late_layer)
    acts, weights = _run_engine(late_layer, pipeline_parallel_size=2)

    assert acts["qk_layers"] == list(range(NUM_LAYERS))
    assert acts["attn_q"].shape[0] == NUM_LAYERS
    _assert_close(weights[:, :n, :n], hf_weights)
