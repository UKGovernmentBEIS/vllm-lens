"""Q/K capture under chunked prefill.

With ``max_num_batched_tokens`` far below the prompt length, the prefill
runs in several chunks and the qk hook fires once per chunk.  The
accumulated capture must cover every prompt position exactly once, and
the reconstructed pattern must still match HF (this is what proves the
per-chunk K accumulation reconstructs the full prefix — vllm-lens forces
``skip_reading_prefix_cache`` for capture requests, so no positions are
served from the prefix cache).
"""

import gc
import os

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from vllm_lens.attention import attention_patterns

from .conftest import LAYER_IDX, MODEL_NAME

_LONG_PROMPT = (
    "The history of artificial intelligence began in antiquity, with myths, "
    "stories and rumors of artificial beings endowed with intelligence or "
    "consciousness by master craftsmen. The seeds of modern AI were planted "
    "by philosophers who attempted to describe the process of human thinking "
    "as the mechanical manipulation of symbols. This work culminated in the "
    "invention of the programmable digital computer in the 1940s, a machine "
    "based on the abstract essence of mathematical reasoning. This device and "
    "the ideas behind it inspired a handful of scientists to begin seriously "
    "discussing the possibility of building an electronic brain. "
) * 3


@pytest.fixture(scope="module")
def llm_chunked():
    # True fp32 relies on vLLM's default VLLM_FLOAT32_MATMUL_PRECISION=highest
    # and on vllm-lens defaulting TRITON_F32_DEFAULT=ieee for fp32 models (see
    # _patched_create_engine_config) — deliberately not pinned here so the
    # tight tolerance also verifies that default.
    os.environ.pop("TRITON_F32_DEFAULT", None)
    llm = LLM(
        model=MODEL_NAME,
        dtype="float32",
        gpu_memory_utilization=0.3,
        enable_chunked_prefill=True,
        max_num_batched_tokens=64,
    )
    yield llm
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def test_chunked_prefill_covers_full_prompt(llm_chunked):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    n_prompt = len(tokenizer(_LONG_PROMPT).input_ids)
    assert n_prompt > 128, "prompt must span multiple 64-token chunks"

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=4,
        extra_args={"output_qk": [LAYER_IDX]},
    )
    output = llm_chunked.generate([_LONG_PROMPT], sampling_params)[0]
    acts = output.activations

    n_gen = len(output.outputs[0].token_ids)
    expected = n_prompt + n_gen - 1
    assert acts["attn_q"].shape[1] == expected
    assert acts["attn_k"].shape[1] == expected


def test_chunked_prefill_matches_hf(llm_chunked):
    torch.backends.cuda.matmul.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        device_map="cuda",
        attn_implementation="eager",
    ).eval()
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        token_ids = tokenizer(_LONG_PROMPT).input_ids
        n = len(token_ids)

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            extra_args={"output_qk": [LAYER_IDX]},
        )
        output = llm_chunked.generate([_LONG_PROMPT], sampling_params)[0]
        weights = attention_patterns(output.activations, LAYER_IDX)

        ids = torch.tensor([token_ids], device="cuda")
        with torch.no_grad():
            out = model(ids, output_attentions=True, use_cache=False)
        hf_weights = out.attentions[LAYER_IDX][0].float().cpu()

        from ._qk_asserts import assert_attention_close

        assert_attention_close(weights[:, :n, :n], hf_weights, label="chunked prefill")
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
