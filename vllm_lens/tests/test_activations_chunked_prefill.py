import gc

import pytest
import torch
from vllm import LLM, SamplingParams

from .conftest import LAYER_IDX, MODEL_NAME, PROMPT

LONG_PROMPT = "The quick brown fox jumps over the lazy dog. " * 20


@pytest.fixture(scope="module")
def chunked_llm():
    llm = LLM(
        model=MODEL_NAME,
        dtype="auto",
        gpu_memory_utilization=0.3,
        max_num_batched_tokens=64,
        enable_chunked_prefill=True,
    )
    yield llm
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def _get_hf_acts(model, tokenizer, prompt: str) -> torch.Tensor:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    num_tokens = inputs.input_ids.shape[1]
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    return out.hidden_states[LAYER_IDX + 1][0, :num_tokens].float()


def _get_vllm_acts(llm: LLM, prompts: list[str], max_tokens: int) -> list[torch.Tensor]:
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        extra_args={"output_residual_stream": [LAYER_IDX]},
    )
    outputs = llm.generate(prompts, sampling_params)
    return [out.activations["residual_stream"] for out in outputs]  # type: ignore[reportAttributeAccessIssue]


def _assert_close(vllm_acts: torch.Tensor, hf_acts: torch.Tensor):
    assert vllm_acts.shape == hf_acts.shape, (
        f"Shape mismatch: vLLM {vllm_acts.shape} vs HF {hf_acts.shape}"
    )
    mean_abs_diff = (vllm_acts - hf_acts.to(vllm_acts.device)).abs().mean().item()
    assert mean_abs_diff < 1e-2, f"Mean abs diff too large: {mean_abs_diff:.6f}"


class TestChunkedPrefillMatchesTransformers:
    def test_long_prompt_activations_match(self, chunked_llm, hf_model):
        """Activations from a chunked long prompt match HuggingFace reference."""
        model, tokenizer = hf_model
        num_tokens = tokenizer(LONG_PROMPT, return_tensors="pt").input_ids.shape[1]
        assert num_tokens > 64, (
            f"Prompt must exceed chunk size; got {num_tokens} tokens"
        )

        hf_acts = _get_hf_acts(model, tokenizer, LONG_PROMPT)
        streams = _get_vllm_acts(chunked_llm, [LONG_PROMPT], max_tokens=1)
        vllm_acts = streams[0][0, :num_tokens].float()

        _assert_close(vllm_acts, hf_acts)

    def test_activation_shape_covers_full_prompt(self, chunked_llm, hf_model):
        """Activation token dim covers the full prompt, not just one chunk."""
        _, tokenizer = hf_model
        num_tokens = tokenizer(LONG_PROMPT, return_tensors="pt").input_ids.shape[1]

        streams = _get_vllm_acts(chunked_llm, [LONG_PROMPT], max_tokens=1)
        # streams[0] shape: (n_layers, total_tokens, hidden_dim)
        assert streams[0].shape[1] >= num_tokens, (
            f"Activation token dim {streams[0].shape[1]} < prompt tokens {num_tokens}. "
            "Chunked prefill may not be concatenating all chunks."
        )

    def test_batch_mixed_short_and_long(self, chunked_llm, hf_model):
        """Batch with short (<64 token) and long (~200 token) prompts both match HF."""
        model, tokenizer = hf_model
        prompts = [PROMPT, LONG_PROMPT]

        streams = _get_vllm_acts(chunked_llm, prompts, max_tokens=1)

        for i, prompt in enumerate(prompts):
            num_tokens = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
            hf_acts = _get_hf_acts(model, tokenizer, prompt)
            vllm_acts = streams[i][0, :num_tokens].float()
            _assert_close(vllm_acts, hf_acts)
