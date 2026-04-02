import asyncio
import gc

from dotenv import load_dotenv

load_dotenv()
import pytest
import torch
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

from .conftest import LAYER_IDX, MODEL_NAME, PROMPTS

TEST_PROMPTS = PROMPTS[:5]


@pytest.fixture(scope="module")
async def vllm_model():
    engine_args = AsyncEngineArgs(
        model="RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic",
        dtype="auto",
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
        max_model_len=1024,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    yield engine
    engine.shutdown()
    gc.collect()
    torch.cuda.empty_cache()


async def _get_activations(
    engine: AsyncLLMEngine, prompt: str, request_id: str
) -> torch.Tensor:
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=1,
        extra_args={"output_residual_stream": [LAYER_IDX]},
    )
    final = None
    async for output in engine.generate(prompt, sampling_params, request_id=request_id):
        final = output
    assert final is not None
    return final.activations["residual_stream"]


async def test_concurrent_vs_sequential(vllm_model):

    concurrent = await asyncio.gather(
        *(
            _get_activations(vllm_model, p, f"concurrent-{i}")
            for i, p in enumerate(PROMPTS)
        )
    )

    sequential = []
    for i, p in enumerate(PROMPTS):
        acts = await _get_activations(vllm_model, p, f"sequential-{i}")
        sequential.append(acts)

    comparisons = []
    max_diffs = []
    for i, prompt in enumerate(PROMPTS):
        equal = torch.equal(concurrent[i], sequential[i])
        comparisons.append(equal)
        diff = (concurrent[i] - sequential[i]).abs().max().item()
        max_diffs.append(diff)

    num_equal = sum(comparisons)
    assert num_equal == len(PROMPTS), (
        f"Not all are equal {comparisons}, max diffs: {max_diffs}"
    )
