"""Tests for activation capture and steering with pipeline parallelism (PP=2)."""

import gc

import pytest
import torch
from vllm import LLM, SamplingParams

from vllm_lens import SteeringVector

from .conftest import LAYER_IDX, MODEL_NAME, NUM_LAYERS, PROMPT, PROMPTS

# Skip entire module if fewer than 2 GPUs available.
pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Pipeline parallelism tests require at least 2 GPUs",
)


@pytest.fixture(scope="module")
def llm_pp2():
    llm = LLM(
        model=MODEL_NAME,
        dtype="auto",
        gpu_memory_utilization=0.3,
        pipeline_parallel_size=2,
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


# ------------------------------------------------------------------
# Activation capture tests
# ------------------------------------------------------------------


class TestPPActivationCapture:
    def test_all_layers_captured(self, llm_pp2):
        """With output_residual_stream=True, all layers should be present."""
        sp = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            extra_args={"output_residual_stream": True},
        )
        outputs = llm_pp2.generate([PROMPT], sp)
        stream = outputs[0].activations["residual_stream"]
        assert stream.shape[0] == NUM_LAYERS, (
            f"Expected {NUM_LAYERS} layers, got {stream.shape[0]}"
        )

    def test_cross_pp_rank_layer_selection(self, llm_pp2):
        """Selecting layers from both PP ranks should work.

        With PP=2 on 24 layers: rank 0 has layers 0-11, rank 1 has 12-23.
        Requesting [2, 20] captures one layer from each rank.
        """
        sp = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            extra_args={"output_residual_stream": [2, 20]},
        )
        outputs = llm_pp2.generate([PROMPT], sp)
        stream = outputs[0].activations["residual_stream"]
        assert stream.shape[0] == 2, f"Expected 2 layers, got {stream.shape[0]}"

    def test_single_layer_from_second_rank(self, llm_pp2):
        """Capturing a layer only on the second PP rank should work."""
        sp = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            extra_args={"output_residual_stream": [20]},
        )
        outputs = llm_pp2.generate([PROMPT], sp)
        stream = outputs[0].activations["residual_stream"]
        assert stream.shape[0] == 1

    def test_batch_prompts_all_layers(self, llm_pp2):
        """Multiple prompts should each get full activations."""
        sp = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            extra_args={"output_residual_stream": True},
        )
        outputs = llm_pp2.generate(PROMPTS, sp)
        for i, out in enumerate(outputs):
            stream = out.activations["residual_stream"]
            assert stream.shape[0] == NUM_LAYERS, (
                f"Prompt {i}: expected {NUM_LAYERS} layers, got {stream.shape[0]}"
            )

    def test_activations_match_hf(self, llm_pp2, hf_model):
        """PP=2 activations for a specific layer should match HuggingFace."""
        model, tokenizer = hf_model
        hf_acts = _get_hf_acts(model, tokenizer, PROMPT)
        num_tokens = tokenizer(PROMPT, return_tensors="pt").input_ids.shape[1]

        sp = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            extra_args={"output_residual_stream": [LAYER_IDX]},
        )
        outputs = llm_pp2.generate([PROMPT], sp)
        vllm_acts = outputs[0].activations["residual_stream"][0, :num_tokens].float()

        assert vllm_acts.shape == hf_acts.shape, (
            f"Shape mismatch: vLLM {vllm_acts.shape} vs HF {hf_acts.shape}"
        )
        mean_abs_diff = (vllm_acts - hf_acts.to(vllm_acts.device)).abs().mean().item()
        assert mean_abs_diff < 1e-2, f"Mean abs diff too large: {mean_abs_diff:.6f}"


# ------------------------------------------------------------------
# Steering tests with PP
# ------------------------------------------------------------------


class TestPPSteering:
    def test_steering_on_second_pp_rank(self, llm_pp2):
        """Steering at a layer held by PP rank 1 should change output."""
        sp_baseline = SamplingParams(temperature=0.0, max_tokens=20)
        baseline = llm_pp2.generate([PROMPT], sp_baseline)
        baseline_text = baseline[0].outputs[0].text

        # Probe hidden dim
        sp_probe = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            extra_args={"output_residual_stream": [20]},
        )
        probe = llm_pp2.generate([PROMPT], sp_probe)
        hidden_dim = probe[0].activations["residual_stream"].shape[-1]

        # Steer at layer 20 (on PP rank 1) with high scale
        vectors = [
            SteeringVector(
                activations=torch.randn(1, hidden_dim),
                layer_indices=[20],
                scale=10.0,
            )
        ]
        sp_steered = SamplingParams(
            temperature=0.0,
            max_tokens=20,
            extra_args={"apply_steering_vectors": vectors},
        )
        steered = llm_pp2.generate([PROMPT], sp_steered)
        assert steered[0].outputs[0].text != baseline_text, (
            "Steering on PP rank 1 should change the output"
        )

    def test_steering_and_capture_across_ranks(self, llm_pp2):
        """Steer at layer on rank 1, capture from both ranks."""
        # Capture baseline from both ranks
        sp_baseline = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            extra_args={"output_residual_stream": [2, 20]},
        )
        baseline = llm_pp2.generate([PROMPT], sp_baseline)
        baseline_acts = baseline[0].activations["residual_stream"]
        hidden_dim = baseline_acts.shape[-1]

        # Steer at layer 20 (rank 1) and capture from both ranks
        vectors = [
            SteeringVector(
                activations=torch.randn(1, hidden_dim),
                layer_indices=[20],
                scale=5.0,
            )
        ]
        sp_steered = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            extra_args={
                "output_residual_stream": [2, 20],
                "apply_steering_vectors": vectors,
            },
        )
        steered = llm_pp2.generate([PROMPT], sp_steered)
        steered_acts = steered[0].activations["residual_stream"]

        assert steered_acts.shape == baseline_acts.shape
        # Layer 20 (index 1 in the captured tensor) should differ
        diff_layer20 = (
            (steered_acts[1].float() - baseline_acts[1].float()).abs().max().item()
        )
        assert diff_layer20 > 0.1, (
            f"Layer 20 should differ after steering, but max diff = {diff_layer20}"
        )
