import gc

import pytest
import torch
from vllm import LLM, SamplingParams

from .conftest import LAYER_IDX, MODEL_NAME, PROMPT, PROMPTS


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


class TestOfflineMatchesTransformers:
    def test_single_prompt_1_token(self, llm_model, hf_model):
        model, tokenizer = hf_model
        num_tokens = tokenizer(PROMPT, return_tensors="pt").input_ids.shape[1]

        hf_acts = _get_hf_acts(model, tokenizer, PROMPT)
        streams = _get_vllm_acts(llm_model, [PROMPT], max_tokens=1)
        vllm_acts = streams[0][0, :num_tokens].float()

        _assert_close(vllm_acts, hf_acts)

    def test_single_prompt_10_tokens(self, llm_model, hf_model):
        model, tokenizer = hf_model
        num_tokens = tokenizer(PROMPT, return_tensors="pt").input_ids.shape[1]

        hf_acts = _get_hf_acts(model, tokenizer, PROMPT)
        streams = _get_vllm_acts(llm_model, [PROMPT], max_tokens=10)
        vllm_acts = streams[0][0, :num_tokens].float()

        _assert_close(vllm_acts, hf_acts)

    def test_batch_prompts_10_tokens(self, llm_model, hf_model):
        model, tokenizer = hf_model

        streams = _get_vllm_acts(llm_model, PROMPTS, max_tokens=10)

        for i, prompt in enumerate(PROMPTS):
            num_tokens = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
            hf_acts = _get_hf_acts(model, tokenizer, prompt)
            vllm_acts = streams[i][0, :num_tokens].float()
            _assert_close(vllm_acts, hf_acts)

    def test_activation_shape_matches_token_count(self, llm_model):
        """Activation seq dim must equal prompt + generated - 1."""
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=10,
            extra_args={"output_residual_stream": [LAYER_IDX]},
        )
        outputs = llm_model.generate(PROMPTS, sampling_params)
        for output in outputs:
            n_prompt = len(output.prompt_token_ids)
            n_gen = len(output.outputs[0].token_ids)
            expected = n_prompt + n_gen - 1
            rs = output.activations["residual_stream"]
            assert rs.shape[1] == expected, (
                f"Expected seq_len={expected} (prompt={n_prompt}, gen={n_gen}), "
                f"got {rs.shape[1]}"
            )
