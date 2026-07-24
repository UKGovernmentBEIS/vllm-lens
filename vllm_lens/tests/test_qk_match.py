"""Q/K capture ground truth: reconstructed attention matches HF transformers.

Captures post-RoPE Q/K via ``extra_args={"output_qk": ...}`` from a real
vLLM engine and checks that ``vllm_lens.attention.attention_patterns``
reproduces HuggingFace's real attention weights (eager attention with
``output_attentions=True``) for the same token ids.
"""

import gc
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from vllm_lens._worker_ext import HiddenStatesExtension
from vllm_lens.attention import attention_patterns

from .conftest import LAYER_IDX, MODEL_NAME, NUM_LAYERS, PROMPT

_NUM_Q_HEADS = 14  # Qwen2.5-0.5B
_NUM_KV_HEADS = 2
_HEAD_SIZE = 64


@pytest.fixture(scope="module")
def llm_model():
    llm = LLM(
        model=MODEL_NAME,
        dtype="auto",
        gpu_memory_utilization=0.3,
    )
    yield llm
    del llm
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def hf_eager():
    """HF model with eager attention — required for output_attentions."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype="auto",
        device_map="cuda",
        attn_implementation="eager",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    yield model, tokenizer
    del model
    gc.collect()
    torch.cuda.empty_cache()


def _hf_attention(model, token_ids: list[int], layer: int) -> torch.Tensor:
    """HF ground-truth attention weights (num_heads, seq, seq)."""
    ids = torch.tensor([token_ids], device="cuda")
    with torch.no_grad():
        out = model(ids, output_attentions=True, use_cache=False)
    return out.attentions[layer][0].float().cpu()


def _capture(llm: LLM, prompt: str, output_qk, max_tokens: int = 1):
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        extra_args={"output_qk": output_qk},
    )
    outputs = llm.generate([prompt], sampling_params)
    return outputs[0]


def _assert_attention_close(got: torch.Tensor, want: torch.Tensor):
    assert got.shape == want.shape, f"{got.shape} vs {want.shape}"
    # HF eager casts attention probs back to bf16 (≈0.4% quantization on
    # values near 1) while our replay stays fp32, so use the same 1e-2
    # mean-abs-diff convention as the residual-stream tests.
    mean_abs_diff = (got - want).abs().mean().item()
    assert mean_abs_diff < 1e-2, f"Mean abs diff too large: {mean_abs_diff:.6f}"
    # Rows should mostly agree on the most-attended source position.
    got_argmax = got.argmax(dim=-1)
    want_argmax = want.argmax(dim=-1)
    agreement = (got_argmax == want_argmax).float().mean().item()
    assert agreement > 0.9, f"Row-argmax agreement too low: {agreement:.2%}"


class TestMatchesTransformers:
    def test_prefill_pattern_matches_hf(self, llm_model, hf_eager):
        model, tokenizer = hf_eager
        token_ids = tokenizer(PROMPT).input_ids
        n = len(token_ids)

        output = _capture(llm_model, PROMPT, output_qk=[LAYER_IDX], max_tokens=1)
        acts = output.activations
        weights = attention_patterns(acts, LAYER_IDX)

        hf_weights = _hf_attention(model, token_ids, LAYER_IDX)
        _assert_attention_close(weights[:, :n, :n], hf_weights)

    def test_decode_rows_match_hf(self, llm_model, hf_eager):
        model, tokenizer = hf_eager
        prompt_ids = tokenizer(PROMPT).input_ids

        output = _capture(llm_model, PROMPT, output_qk=[LAYER_IDX], max_tokens=8)
        gen_ids = list(output.outputs[0].token_ids)
        weights = attention_patterns(output.activations, LAYER_IDX)

        # Captured positions = prompt + all-but-last generated token.
        all_ids = prompt_ids + gen_ids[:-1]
        assert weights.shape[1] == len(all_ids)

        hf_weights = _hf_attention(model, all_ids, LAYER_IDX)
        _assert_attention_close(weights, hf_weights)

    def test_shapes_and_meta(self, llm_model, hf_eager):
        _, tokenizer = hf_eager
        n = len(tokenizer(PROMPT).input_ids)

        output = _capture(llm_model, PROMPT, output_qk=[LAYER_IDX], max_tokens=1)
        acts = output.activations
        assert acts["attn_q"].shape == (1, n, _NUM_Q_HEADS, _HEAD_SIZE)
        assert acts["attn_k"].shape == (1, n, _NUM_KV_HEADS, _HEAD_SIZE)
        assert acts["qk_layers"] == [LAYER_IDX]
        meta = acts["qk_meta"][0]
        assert meta["scale"] == pytest.approx(_HEAD_SIZE**-0.5)
        assert meta["logits_soft_cap"] == 0.0
        assert tuple(meta["sliding_window"]) == (-1, -1)

    def test_true_captures_all_layers(self, llm_model):
        output = _capture(llm_model, PROMPT, output_qk=True, max_tokens=1)
        acts = output.activations
        assert acts["qk_layers"] == list(range(NUM_LAYERS))
        assert acts["attn_q"].shape[0] == NUM_LAYERS

    def test_combined_with_residual_stream(self, llm_model, hf_eager):
        model, tokenizer = hf_eager
        n = len(tokenizer(PROMPT).input_ids)
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            extra_args={
                "output_residual_stream": [LAYER_IDX],
                "output_qk": [LAYER_IDX],
            },
        )
        output = llm_model.generate([PROMPT], sampling_params)[0]
        acts = output.activations
        assert "residual_stream" in acts and "attn_q" in acts

        weights = attention_patterns(acts, LAYER_IDX)
        hf_weights = _hf_attention(model, tokenizer(PROMPT).input_ids, LAYER_IDX)
        _assert_attention_close(weights[:, :n, :n], hf_weights)

    def test_no_leaked_state(self, llm_model):
        _capture(llm_model, PROMPT, output_qk=[LAYER_IDX], max_tokens=2)
        counts = llm_model.collective_rpc("_debug_captured_qk_count")
        assert all(c == 0 for c in counts)


class TestMLARefusal:
    def test_install_qk_hooks_rejects_mla(self):
        from vllm.model_executor.layers.attention.mla_attention import MLAAttention

        worker = HiddenStatesExtension()
        worker.compilation_config = SimpleNamespace(  # type: ignore[attr-defined]
            static_forward_context={
                "model.layers.0.self_attn.attn": object.__new__(MLAAttention)
            }
        )
        with pytest.raises(RuntimeError, match="MLA"):
            worker.install_qk_hooks()
