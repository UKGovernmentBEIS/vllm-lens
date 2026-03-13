"""Cross-API activation consistency tests.

Verifies that activation extraction with steering vectors produces identical
results across async vLLM (AsyncLLMEngine), sync vLLM (LLM), and the Inspect
provider (VLLMLensAPI via get_model + vLLM server).
"""

import gc
import signal
import socket
import subprocess
import sys
import time

import httpx
import pytest
import torch
from inspect_ai.model import ChatMessageUser, GenerateConfig, get_model
from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, LLM, SamplingParams

from vllm_lens import SteeringVector

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
NUM_LAYERS = 24
HIDDEN_DIM = 896
CAPTURE_LAYER = 16
STEER_LAYER = 8
MAX_TOKENS = 5
GPU_MEM_UTIL = 0.3

PROMPTS = [
    "The future of artificial intelligence is",
    "Once upon a time there was a",
    "The quick brown fox jumped over",
    "Scientists recently discovered that the",
    "Deep learning has revolutionized the",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_deterministic_steering_vector() -> list[SteeringVector]:
    """Create a deterministic steering vector for cross-API comparison.

    Uses a fixed seed so all three API paths receive identical steering data.
    """
    rng = torch.Generator().manual_seed(42)
    activations = torch.randn(1, HIDDEN_DIM, generator=rng)
    return [
        SteeringVector(
            activations=activations,
            layer_indices=[STEER_LAYER],
            scale=1.0,
        )
    ]


def _clone_steering(steering: list[SteeringVector]) -> list[SteeringVector]:
    """Clone steering vectors so each generation gets a fresh copy.

    The plugin pops ``apply_steering_vectors`` from ``extra_args``, so each
    call needs its own copy.
    """
    return [
        sv.model_copy(update={"activations": sv.activations.clone()}) for sv in steering
    ]


def _assert_activations_close(
    a: torch.Tensor,
    b: torch.Tensor,
    label: str,
    atol: float = 1e-4,
) -> None:
    """Assert two activation tensors are nearly identical."""
    assert a.shape == b.shape, f"{label}: shape mismatch {a.shape} vs {b.shape}"
    a_f = a.float()
    b_f = b.float()
    max_abs_diff = (a_f - b_f).abs().max().item()
    mean_abs_diff = (a_f - b_f).abs().mean().item()
    assert mean_abs_diff < atol, (
        f"{label}: mean_abs_diff={mean_abs_diff:.8f} exceeds {atol} "
        f"(max_abs_diff={max_abs_diff:.8f})"
    )


def _chat_token_ids(tokenizer, prompt: str) -> list[int]:
    """Apply the chat template to a user prompt and return token IDs.

    This produces the same tokens that the vLLM server generates when
    receiving a ChatMessageUser via chat completions.
    """
    messages = [{"role": "user", "content": prompt}]
    result = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    # Some tokenizers return a BatchEncoding; extract input_ids.
    if not isinstance(result, list):
        return result["input_ids"]
    return result


def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_server(base_url: str, timeout: float = 120.0) -> None:
    """Poll the server's /health endpoint until it responds 200 or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{base_url}/health")
                if resp.status_code == 200:
                    return
        except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException):
            pass
        time.sleep(2.0)
    raise TimeoutError(
        f"vLLM server at {base_url} did not become healthy within {timeout}s"
    )


# ---------------------------------------------------------------------------
# Path runners
# ---------------------------------------------------------------------------


async def _run_async_path(
    token_ids_map: dict[str, list[int]],
    steering: list[SteeringVector],
) -> dict[str, torch.Tensor]:
    """Run all prompts through AsyncLLMEngine and return activations."""
    engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(
            model=MODEL_NAME,
            dtype="auto",
            gpu_memory_utilization=GPU_MEM_UTIL,
        )
    )

    results: dict[str, torch.Tensor] = {}
    for i, prompt in enumerate(PROMPTS):
        sp = SamplingParams(
            temperature=0.0,
            max_tokens=MAX_TOKENS,
            extra_args={
                "output_residual_stream": [CAPTURE_LAYER],
                "apply_steering_vectors": _clone_steering(steering),
            },
        )
        final = None
        async for output in engine.generate(
            {"prompt_token_ids": token_ids_map[prompt]},
            sp,
            request_id=f"async-{i}",
        ):
            final = output
        assert final is not None
        results[prompt] = final.activations["residual_stream"]  # type: ignore[reportAttributeAccessIssue]

    engine.shutdown()
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return results


def _run_sync_path(
    token_ids_map: dict[str, list[int]],
    steering: list[SteeringVector],
) -> dict[str, torch.Tensor]:
    """Run all prompts through LLM and return activations."""
    llm = LLM(
        model=MODEL_NAME,
        dtype="auto",
        gpu_memory_utilization=GPU_MEM_UTIL,
    )

    results: dict[str, torch.Tensor] = {}
    for i, prompt in enumerate(PROMPTS):
        sp = SamplingParams(
            temperature=0.0,
            max_tokens=MAX_TOKENS,
            extra_args={
                "output_residual_stream": [CAPTURE_LAYER],
                "apply_steering_vectors": _clone_steering(steering),
            },
        )
        outputs = llm.generate([{"prompt_token_ids": token_ids_map[prompt]}], sp)
        results[prompt] = outputs[0].activations["residual_stream"]  # type: ignore[reportAttributeAccessIssue]

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return results


async def _run_inspect_path(
    steering: list[SteeringVector],
) -> dict[str, torch.Tensor]:
    """Start vLLM server, run prompts via Inspect provider, return activations."""
    port = _find_free_port()
    base_url = f"http://localhost:{port}"

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            MODEL_NAME,
            "--port",
            str(port),
            "--gpu-memory-utilization",
            str(GPU_MEM_UTIL),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    try:
        _wait_for_server(base_url, timeout=120.0)

        model = get_model(
            f"vllm-lens/{MODEL_NAME}",
            base_url=f"{base_url}/v1",
        )

        results: dict[str, torch.Tensor] = {}
        for prompt in PROMPTS:
            config = GenerateConfig(
                temperature=0.0,
                max_tokens=MAX_TOKENS,
                extra_body={
                    "extra_args": {
                        "output_residual_stream": [CAPTURE_LAYER],
                        "apply_steering_vectors": _clone_steering(steering),
                    },
                },
            )
            output = await model.generate(
                [ChatMessageUser(content=prompt)],
                config=config,
            )
            assert output.metadata is not None, "No metadata in Inspect output"
            assert "activations" in output.metadata, (
                f"No activations in metadata. Keys: {list(output.metadata.keys())}"
            )
            results[prompt] = output.metadata["activations"]["residual_stream"]

        del model
        return results
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Orchestrator fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
async def all_results():
    """Run all three API paths sequentially and return their activation dicts.

    Execution order: async -> sync -> inspect.
    Each engine is fully shut down before the next starts.

    Returns a tuple of (results_dict, prompt_token_counts) where results_dict
    maps API name to prompt→tensor and prompt_token_counts maps prompt to
    the number of prompt tokens (excluding generated tokens).  Comparisons
    should use only prompt tokens because generated tokens can diverge across
    processes due to CUDA non-determinism in Flash Attention.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    token_ids_map = {prompt: _chat_token_ids(tokenizer, prompt) for prompt in PROMPTS}
    prompt_token_counts = {prompt: len(ids) for prompt, ids in token_ids_map.items()}
    steering = _make_deterministic_steering_vector()

    async_acts = await _run_async_path(token_ids_map, steering)
    sync_acts = _run_sync_path(token_ids_map, steering)
    inspect_acts = await _run_inspect_path(steering)

    yield (
        {
            "async": async_acts,
            "sync": sync_acts,
            "inspect": inspect_acts,
        },
        prompt_token_counts,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCrossAPIActivations:
    """Verify activation extraction matches across async, sync, and Inspect paths.

    Comparisons involving the Inspect (HTTP) path use only prompt-token
    activations.  Generated tokens can diverge across separate processes
    due to CUDA non-determinism in Flash Attention — even with temperature=0,
    tiny floating-point differences can flip argmax when logits are close,
    causing cascading divergence.  Prompt tokens are processed in a single
    deterministic prefill step and should match exactly.
    """

    def test_async_vs_sync(self, all_results):
        """Async and sync paths (same process) produce identical activations."""
        results, _ = all_results
        for prompt in PROMPTS:
            _assert_activations_close(
                results["async"][prompt],
                results["sync"][prompt],
                label=f"async-vs-sync [{prompt[:30]}...]",
            )

    def test_async_vs_inspect(self, all_results):
        """Async and Inspect prompt-token activations match."""
        results, prompt_counts = all_results
        for prompt in PROMPTS:
            n = prompt_counts[prompt]
            _assert_activations_close(
                results["async"][prompt][:, :n],
                results["inspect"][prompt][:, :n],
                label=f"async-vs-inspect [{prompt[:30]}...]",
            )

    def test_sync_vs_inspect(self, all_results):
        """Sync and Inspect prompt-token activations match."""
        results, prompt_counts = all_results
        for prompt in PROMPTS:
            n = prompt_counts[prompt]
            _assert_activations_close(
                results["sync"][prompt][:, :n],
                results["inspect"][prompt][:, :n],
                label=f"sync-vs-inspect [{prompt[:30]}...]",
            )

    def test_activation_shapes(self, all_results):
        """All activations should have shape (1, num_tokens, HIDDEN_DIM)."""
        results, _ = all_results
        for api_name, acts in results.items():
            for prompt, tensor in acts.items():
                assert tensor.dim() == 3, (
                    f"{api_name} [{prompt[:30]}...]: "
                    f"expected 3D tensor, got {tensor.dim()}D"
                )
                assert tensor.shape[0] == 1, (
                    f"{api_name} [{prompt[:30]}...]: "
                    f"expected 1 layer, got {tensor.shape[0]}"
                )
                assert tensor.shape[2] == HIDDEN_DIM, (
                    f"{api_name} [{prompt[:30]}...]: "
                    f"expected hidden_dim={HIDDEN_DIM}, got {tensor.shape[2]}"
                )

    def test_activations_not_zero(self, all_results):
        """Activations should not be all zeros (sanity check)."""
        results, _ = all_results
        for api_name, acts in results.items():
            for prompt, tensor in acts.items():
                assert tensor.abs().max().item() > 0.01, (
                    f"{api_name} [{prompt[:30]}...]: activations are all zeros"
                )
