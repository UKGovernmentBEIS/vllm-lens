"""Tests for activation steering via ``apply_steering_vectors``."""

import gc

import pytest
import torch
from vllm import AsyncEngineArgs, AsyncLLMEngine, RequestOutput, SamplingParams

from vllm_lens import SteeringVector

from .conftest import LAYER_IDX, MODEL_NAME, PROMPT


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


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


async def _generate(
    engine,
    prompt: str,
    request_id: str,
    max_tokens: int = 10,
    extra_args: dict | None = None,
) -> RequestOutput:
    """Run a single generation and return the final RequestOutput."""
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        extra_args=extra_args or {},
    )
    final = None
    async for output in engine.generate(prompt, sp, request_id=request_id):
        final = output
    assert final is not None
    return final


def _make_steering_vector(
    hidden_dim: int,
    layer_indices: list[int],
    scale: float = 1.0,
    norm_match: bool = False,
    position_indices: list[int] | None = None,
    n_positions: int | None = None,
) -> list[SteeringVector]:
    """Build an ``apply_steering_vectors`` list with a single config.

    If *n_positions* is given, creates a 3D tensor ``(n_layers, n_pos, hidden_dim)``.
    Otherwise creates 2D ``(n_layers, hidden_dim)``.
    """
    n_layers = len(layer_indices)
    if n_positions is not None:
        activations = torch.randn(n_layers, n_positions, hidden_dim)
    else:
        activations = torch.randn(n_layers, hidden_dim)
    return [
        SteeringVector(
            activations=activations,
            layer_indices=layer_indices,
            scale=scale,
            norm_match=norm_match,
            position_indices=position_indices,
        )
    ]


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestSteering:
    async def test_steering_changes_output(self, vllm_model):
        """Applying a steering vector should change the generated text."""
        baseline = await _generate(vllm_model, PROMPT, "steer-baseline", max_tokens=20)
        baseline_text = baseline.outputs[0].text

        # Get hidden_dim from the model via a capture run
        capture = await _generate(
            vllm_model,
            PROMPT,
            "steer-dim-probe",
            max_tokens=1,
            extra_args={"output_residual_stream": [LAYER_IDX]},
        )
        hidden_dim = capture.activations["residual_stream"].shape[-1]  # type: ignore[reportAttributeAccessIssue]

        vectors = _make_steering_vector(hidden_dim, [LAYER_IDX], scale=10.0)
        steered = await _generate(
            vllm_model,
            PROMPT,
            "steer-changed",
            max_tokens=20,
            extra_args={"apply_steering_vectors": vectors},
        )
        steered_text = steered.outputs[0].text

        assert steered_text != baseline_text, (
            "Steering with scale=10.0 should change the output text"
        )

    async def test_coefficient_zero_matches_baseline(self, vllm_model):
        """scale=0 should produce the same output as no steering."""
        baseline = await _generate(vllm_model, PROMPT, "zero-baseline", max_tokens=20)
        baseline_text = baseline.outputs[0].text

        # Probe hidden_dim so the steering vector shape matches the model.
        capture = await _generate(
            vllm_model,
            PROMPT,
            "zero-dim-probe",
            max_tokens=1,
            extra_args={"output_residual_stream": [LAYER_IDX]},
        )
        hidden_dim = capture.activations["residual_stream"].shape[-1]  # type: ignore[reportAttributeAccessIssue]

        vectors = _make_steering_vector(hidden_dim, [LAYER_IDX], scale=0.0)
        steered = await _generate(
            vllm_model,
            PROMPT,
            "zero-steered",
            max_tokens=20,
            extra_args={"apply_steering_vectors": vectors},
        )
        assert steered.outputs[0].text == baseline_text

    async def test_steering_with_capture(self, vllm_model):
        """Steering + capture: captured activations should reflect the
        steered hidden states (not the original)."""
        # First capture without steering
        unsteered = await _generate(
            vllm_model,
            PROMPT,
            "cap-unsteer",
            max_tokens=1,
            extra_args={"output_residual_stream": [LAYER_IDX]},
        )
        unsteered_acts = unsteered.activations["residual_stream"]  # type: ignore[reportAttributeAccessIssue]
        hidden_dim = unsteered_acts.shape[-1]

        # Now steer AND capture
        vectors = _make_steering_vector(hidden_dim, [LAYER_IDX], scale=5.0)
        steered = await _generate(
            vllm_model,
            PROMPT,
            "cap-steered",
            max_tokens=1,
            extra_args={
                "output_residual_stream": [LAYER_IDX],
                "apply_steering_vectors": vectors,
            },
        )
        steered_acts = steered.activations["residual_stream"]  # type: ignore[reportAttributeAccessIssue]

        # The captured activations should differ because steering was applied.
        diff = (steered_acts.float() - unsteered_acts.float()).abs().max().item()
        assert diff > 0.1, (
            f"Expected captured activations to differ after steering, "
            f"but max diff = {diff:.6f}"
        )

    async def test_multiple_steering_vectors(self, vllm_model):
        """Multiple steering configs in a single request should all apply."""
        baseline = await _generate(vllm_model, PROMPT, "multi-baseline", max_tokens=20)
        baseline_text = baseline.outputs[0].text

        capture = await _generate(
            vllm_model,
            PROMPT,
            "multi-dim",
            max_tokens=1,
            extra_args={"output_residual_stream": [LAYER_IDX]},
        )
        hidden_dim = capture.activations["residual_stream"].shape[-1]  # type: ignore[reportAttributeAccessIssue]

        # Two steering vectors at different layers
        vectors = [
            SteeringVector(
                activations=torch.randn(1, hidden_dim),
                layer_indices=[LAYER_IDX],
                scale=5.0,
            ),
            SteeringVector(
                activations=torch.randn(1, hidden_dim),
                layer_indices=[0],
                scale=5.0,
            ),
        ]
        steered = await _generate(
            vllm_model,
            PROMPT,
            "multi-steered",
            max_tokens=20,
            extra_args={"apply_steering_vectors": vectors},
        )
        assert steered.outputs[0].text != baseline_text

    async def test_3d_positional_steering(self, vllm_model):
        """3D activations with position_indices should only affect
        those positions."""
        # Capture baseline
        baseline = await _generate(
            vllm_model,
            PROMPT,
            "pos-baseline",
            max_tokens=1,
            extra_args={"output_residual_stream": [LAYER_IDX]},
        )
        hidden_dim = baseline.activations["residual_stream"].shape[-1]  # type: ignore[reportAttributeAccessIssue]

        # Steer only at position 0
        vectors = _make_steering_vector(
            hidden_dim,
            [LAYER_IDX],
            scale=10.0,
            position_indices=[0],
            n_positions=1,
        )
        steered = await _generate(
            vllm_model,
            PROMPT,
            "pos-steered",
            max_tokens=1,
            extra_args={
                "output_residual_stream": [LAYER_IDX],
                "apply_steering_vectors": vectors,
            },
        )

        baseline_acts = baseline.activations["residual_stream"][0].float()  # type: ignore[reportAttributeAccessIssue]
        steered_acts = steered.activations["residual_stream"][0].float()  # type: ignore[reportAttributeAccessIssue]

        # Position 0 should be significantly different
        diff_pos0 = (steered_acts[0] - baseline_acts[0]).abs().max().item()
        assert diff_pos0 > 0.1, f"Position 0 should differ, but max diff = {diff_pos0}"

        # Later positions should be less affected (only indirect effects
        # through model computation, not direct steering).
        # We can't assert they're identical since the steering at pos 0
        # propagates through attention, but the direct delta should only
        # be at pos 0.
