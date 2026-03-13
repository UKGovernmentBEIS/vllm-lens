import asyncio
import gc

import pytest
import torch
from syrupy.assertion import SnapshotAssertion
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

from .conftest import LAYER_IDX, MODEL_NAME, PROMPT, PROMPTS


@pytest.fixture(scope="module")
async def vllm_model():
    engine_args = AsyncEngineArgs(
        model=MODEL_NAME,
        dtype="auto",
        gpu_memory_utilization=0.3,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    yield engine
    engine.shutdown()
    gc.collect()
    torch.cuda.empty_cache()


def _get_hf_acts(model, tokenizer, prompt: str) -> torch.Tensor:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    num_tokens = inputs.input_ids.shape[1]
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    return out.hidden_states[LAYER_IDX + 1][0, :num_tokens].float()


async def _get_vllm_acts(
    engine, prompt: str, request_id: str, max_tokens: int
) -> torch.Tensor:
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        extra_args={"output_residual_stream": [LAYER_IDX]},
    )
    final = None
    async for output in engine.generate(prompt, sampling_params, request_id=request_id):
        final = output
    assert final is not None
    stream = final.activations["residual_stream"]
    return stream


def _assert_close(vllm_acts: torch.Tensor, hf_acts: torch.Tensor):
    assert vllm_acts.shape == hf_acts.shape, (
        f"Shape mismatch: vLLM {vllm_acts.shape} vs HF {hf_acts.shape}"
    )
    mean_abs_diff = (vllm_acts - hf_acts.to(vllm_acts.device)).abs().mean().item()
    assert mean_abs_diff < 1e-2, f"Mean abs diff too large: {mean_abs_diff:.6f}"


class TestMatchesTransformers:
    async def test_single_prompt_1_token(self, vllm_model, hf_model):
        model, tokenizer = hf_model
        num_tokens = tokenizer(PROMPT, return_tensors="pt").input_ids.shape[1]

        hf_acts = _get_hf_acts(model, tokenizer, PROMPT)
        stream = await _get_vllm_acts(vllm_model, PROMPT, "single-1tok", max_tokens=1)
        vllm_acts = stream[0, :num_tokens].float()

        _assert_close(vllm_acts, hf_acts)

    async def test_single_prompt_10_tokens(self, vllm_model, hf_model):
        model, tokenizer = hf_model
        num_tokens = tokenizer(PROMPT, return_tensors="pt").input_ids.shape[1]

        hf_acts = _get_hf_acts(model, tokenizer, PROMPT)
        stream = await _get_vllm_acts(vllm_model, PROMPT, "single-10tok", max_tokens=10)
        vllm_acts = stream[0, :num_tokens].float()

        _assert_close(vllm_acts, hf_acts)

    async def test_batch_prompts_10_tokens(self, vllm_model, hf_model):
        model, tokenizer = hf_model

        vllm_streams = await asyncio.gather(
            *(
                _get_vllm_acts(vllm_model, p, f"batch-{i}", max_tokens=10)
                for i, p in enumerate(PROMPTS)
            )
        )

        for i, prompt in enumerate(PROMPTS):
            num_tokens = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
            hf_acts = _get_hf_acts(model, tokenizer, prompt)
            vllm_acts = vllm_streams[i][0, :num_tokens].float()
            _assert_close(vllm_acts, hf_acts)


class TestSnapshotRegression:
    """Snapshot regression: assert extracted activations don't change."""

    async def test_activations_snapshot(self, vllm_model, snapshot: SnapshotAssertion):
        stream = await _get_vllm_acts(vllm_model, PROMPT, "snap", max_tokens=1)
        acts = stream.cpu().float()
        assert {
            "shape": list(acts.shape),
            "values": acts.round(decimals=4).tolist(),
        } == snapshot
