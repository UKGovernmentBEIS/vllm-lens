"""Small-model HTTP checks of values, interventions, transports and cleanup.

Run with VLLM_TEST_MODEL=Qwen/Qwen2.5-0.5B-Instruct. Unlike the original
Llama-specific server tests, these infer hidden size from captured states.
"""

import pytest
import torch

from vllm_lens import Hook, SteeringVector
from vllm_lens.client import VLLMLensClient

LAYER = 2
PROMPT = "The capital of France is"


@pytest.fixture
def client(vllm_server):
    client = VLLMLensClient(vllm_server)
    client.clear_hooks()
    yield client
    client.clear_hooks()


@pytest.mark.parametrize("endpoint", ["generate", "chat"])
@pytest.mark.parametrize("transport", ["base64", "binary"])
def test_capture_and_interventions(client, endpoint, transport):
    def generate(**kwargs):
        prompt = (
            PROMPT if endpoint == "generate" else [{"role": "user", "content": PROMPT}]
        )
        return getattr(client, endpoint)(
            prompt,
            max_tokens=1,
            temperature=0.0,
            capture_layers=[LAYER],
            activations_transport=transport,
            **kwargs,
        )

    baseline = generate().activations["residual_stream"]
    assert baseline.shape[0] == 1 and baseline.shape[1] > 1
    assert torch.isfinite(baseline).all() and baseline.abs().max() > 0.01

    def capture(ctx, h):
        ctx.saved.setdefault("parts", []).append(h.detach().cpu())

    captured = generate(hooks=[Hook(fn=capture, layer_indices=[LAYER])])
    hook_values = torch.cat(captured.hook_results["0"]["parts"], dim=0)
    native = captured.activations["residual_stream"][0]
    # vLLM may execute one surplus forward after stopping. Native capture is
    # trimmed; generic hook results intentionally retain every invocation.
    assert hook_values.shape[0] >= native.shape[0]
    torch.testing.assert_close(hook_values[: native.shape[0]], native, rtol=0, atol=0)

    vector = SteeringVector(
        activations=torch.ones(1, baseline.shape[-1]),
        layer_indices=[LAYER],
        scale=0.5,
    )
    steered = generate(steering_vectors=[vector]).activations["residual_stream"]
    # Same prompt, same capture/injection layer: directly check sign and scale.
    # Allow bf16 rounding, but a silent no-op (delta=0) must fail.
    error = (steered.float() - baseline.float() - 0.5).abs().mean()
    assert error < 0.02, f"incorrect steering delta: mean error {error}"

    def zero(ctx, h):
        return torch.zeros_like(h)

    modified = generate(hooks=[Hook(fn=zero, layer_indices=[LAYER])])
    assert modified.activations["residual_stream"].abs().max() < 0.02

    # Repeated identical prompts also exercise prefix-cache bypass. A previous
    # intervention must not contaminate the next request or its cached prefix.
    restored = generate()
    torch.testing.assert_close(
        restored.activations["residual_stream"], baseline, rtol=0.02, atol=0.02
    )
    assert not restored.hook_results


def test_persistent_hook_cleanup(client):
    def capture(ctx, h):
        ctx.saved["seen"] = True

    client.register_hooks([Hook(fn=capture, layer_indices=[LAYER])])
    client.generate(PROMPT, max_tokens=1)
    results = client.collect_hook_results()
    assert results and any(hooks["0"]["seen"] for hooks in results.values())
    client.clear_hook_results()
    assert not client.collect_hook_results()
    client.clear_hooks()
    client.generate(PROMPT, max_tokens=1)
    assert not client.collect_hook_results()


@pytest.mark.parametrize("endpoint", ["generate", "chat"])
@pytest.mark.parametrize("transport", ["base64", "binary"])
def test_capture_positions(client, endpoint, transport):
    def generate(**kwargs):
        prompt = (
            PROMPT if endpoint == "generate" else [{"role": "user", "content": PROMPT}]
        )
        return getattr(client, endpoint)(
            prompt,
            max_tokens=4,
            temperature=0.0,
            capture_layers=[LAYER],
            activations_transport=transport,
            **kwargs,
        )

    baseline = generate()
    reference = baseline.activations["residual_stream"][0]
    n_prompt = baseline.raw["usage"]["prompt_tokens"]
    n_rows = reference.shape[0]  # prompt tokens plus the generated ones fed back
    assert n_rows >= n_prompt

    last = generate(capture_positions="last_prompt_token").activations
    assert last["residual_stream"].shape == (1, 1, reference.shape[-1])
    assert last["residual_stream_positions"] == [n_prompt - 1]
    torch.testing.assert_close(
        last["residual_stream"][0, 0], reference[n_prompt - 1], rtol=0.02, atol=0.02
    )

    positions = [0, n_prompt - 1, n_prompt + 1, 10_000]
    expected = [pos for pos in positions if pos < n_rows]
    explicit = generate(capture_positions=positions).activations
    assert explicit["residual_stream_positions"] == expected
    torch.testing.assert_close(
        explicit["residual_stream"][0], reference[expected], rtol=0.02, atol=0.02
    )

    # Rejected when the request is submitted, so the server returns an error.
    with pytest.raises(RuntimeError):
        generate(capture_positions=[-1])
