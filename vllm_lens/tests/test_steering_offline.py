"""Regression tests for steering via the offline entry points (issue #28).

``LLM.chat`` submits requests to the engine directly rather than routing
through ``LLM.generate``, so it needs its own patch; and the offline path
must accept both live ``SteeringVector`` objects and the JSON-string wire
format documented for the HTTP API (``vllm_xargs``).  Before the fix:

- ``LLM.chat`` + live vectors    -> msgpack TypeError at serialization
- ``LLM.chat`` + JSON string     -> silently unsteered generation
- ``LLM.generate`` + JSON string -> collective_rpc failure on the worker
"""

import gc
import json

import pytest
import torch
from vllm import LLM, RequestOutput, SamplingParams

from vllm_lens import SteeringVector

from .conftest import LAYER_IDX, MODEL_NAME, PROMPT

MAX_TOKENS = 20


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


def _generate(
    llm: LLM,
    extra_args: dict | None = None,
    max_tokens: int = MAX_TOKENS,
) -> RequestOutput:
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        extra_args=extra_args or {},
    )
    return llm.generate([PROMPT], sp)[0]


def _chat(
    llm: LLM,
    extra_args: dict | None = None,
    max_tokens: int = MAX_TOKENS,
) -> RequestOutput:
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        extra_args=extra_args or {},
    )
    return llm.chat([[{"role": "user", "content": PROMPT}]], sp)[0]


@pytest.fixture(scope="module")
def steering_vector(llm_model) -> SteeringVector:
    """A fixed random steering vector sized to the model's hidden dim."""
    probe = _generate(
        llm_model,
        extra_args={"output_residual_stream": [LAYER_IDX]},
        max_tokens=1,
    )
    hidden_dim = probe.activations["residual_stream"].shape[-1]  # type: ignore[reportAttributeAccessIssue]
    generator = torch.Generator().manual_seed(0)
    return SteeringVector(
        activations=torch.randn(1, hidden_dim, generator=generator),
        layer_indices=[LAYER_IDX],
        scale=10.0,
    )


def _wire_format(sv: SteeringVector) -> str:
    """The JSON-string form the HTTP API accepts via ``vllm_xargs``."""
    return json.dumps([sv.model_dump(mode="json")])


class TestOfflineSteeringEntryPoints:
    def test_generate_accepts_json_wire_format(self, llm_model, steering_vector):
        """``LLM.generate`` must decode the JSON-string wire format.

        Regression: the sync path used to pickle the raw string, so the
        worker crashed with ``'str' object has no attribute
        'layer_indices'`` inside collective_rpc.
        """
        baseline = _generate(llm_model).outputs[0].text
        live = (
            _generate(
                llm_model,
                extra_args={"apply_steering_vectors": [steering_vector]},
            )
            .outputs[0]
            .text
        )
        wire = (
            _generate(
                llm_model,
                extra_args={"apply_steering_vectors": _wire_format(steering_vector)},
            )
            .outputs[0]
            .text
        )

        assert live != baseline, "live-object steering should change the output"
        # Same vector, same greedy decode -> the two wire forms must be
        # equivalent, not merely "different from baseline".
        assert wire == live, (
            "JSON wire format should produce the same steered output as "
            "live SteeringVector objects"
        )

    def test_chat_accepts_live_objects(self, llm_model, steering_vector):
        """``LLM.chat`` with live ``SteeringVector`` objects must steer.

        Regression: chat was never patched, so live tensors hit vLLM's
        msgpack encoder and raised ``TypeError: ... is not serializable``.
        """
        baseline = _chat(llm_model).outputs[0].text
        steered = (
            _chat(
                llm_model,
                extra_args={"apply_steering_vectors": [steering_vector]},
            )
            .outputs[0]
            .text
        )

        assert steered != baseline, (
            "steering via LLM.chat with live SteeringVector objects "
            "should change the output"
        )

    def test_chat_accepts_json_wire_format(self, llm_model, steering_vector):
        """``LLM.chat`` with the JSON-string wire format must steer.

        Regression: the JSON string survived serialization but was never
        consumed, so generation ran *silently unsteered* — the dangerous
        failure mode for research use.
        """
        baseline = _chat(llm_model).outputs[0].text
        live = (
            _chat(
                llm_model,
                extra_args={"apply_steering_vectors": [steering_vector]},
            )
            .outputs[0]
            .text
        )
        wire = (
            _chat(
                llm_model,
                extra_args={"apply_steering_vectors": _wire_format(steering_vector)},
            )
            .outputs[0]
            .text
        )

        assert wire != baseline, (
            "steering via LLM.chat with the JSON wire format should "
            "change the output (silent no-op regression)"
        )
        assert wire == live, (
            "JSON wire format should produce the same steered output as "
            "live SteeringVector objects"
        )

    def test_chat_activation_capture(self, llm_model):
        """``output_residual_stream`` must attach activations on chat outputs.

        ``LLM.chat`` shares the prepare/finalize logic with the patched
        ``LLM.generate``, so capture should work on both entry points.
        """
        output = _chat(
            llm_model,
            extra_args={"output_residual_stream": [LAYER_IDX]},
            max_tokens=5,
        )
        activations = getattr(output, "activations", None)
        assert activations is not None, (
            "LLM.chat should attach captured activations to the output"
        )
        rs = activations["residual_stream"]
        n_prompt = len(output.prompt_token_ids)
        n_gen = len(output.outputs[0].token_ids)
        assert rs.shape[0] == 1  # one captured layer
        assert rs.shape[1] == n_prompt + n_gen - 1
