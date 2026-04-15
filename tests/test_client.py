"""Tests for VLLMLensClient."""

import torch
from vllm_lens import Hook, SteeringVector
from vllm_lens.client import VLLMLensClient



def _make_client(vllm_server: str) -> VLLMLensClient:
    return VLLMLensClient(vllm_server)


# ---------------------------------------------------------------------------
# Basic generate
# ---------------------------------------------------------------------------


def test_generate_returns_text(vllm_server):
    client = _make_client(vllm_server)
    output = client.generate("Hello", max_tokens=5)
    assert isinstance(output.text, str)
    assert len(output.text) > 0


def test_generate_with_logprobs(vllm_server):
    client = _make_client(vllm_server)
    output = client.generate("Hello", max_tokens=1, logprobs=5)
    assert output.logprobs is not None
    assert "top_logprobs" in output.logprobs


def test_model_auto_detected(vllm_server):
    client = VLLMLensClient(vllm_server)
    assert "Llama" in client.model or "llama" in client.model


# ---------------------------------------------------------------------------
# Activation capture
# ---------------------------------------------------------------------------


def test_capture_layers(vllm_server):
    client = _make_client(vllm_server)
    output = client.generate("Hello", max_tokens=1, capture_layers=[15])
    assert output.activations is not None
    rs = output.activations["residual_stream"]
    assert rs.shape[0] == 1  # single layer
    assert rs.shape[2] == 4096


# ---------------------------------------------------------------------------
# ctx.get_parameter
# ---------------------------------------------------------------------------


def test_get_parameter_returns_lm_head(vllm_server):
    """ctx.get_parameter('lm_head.weight') should return the full unembedding matrix."""
    client = _make_client(vllm_server)

    def check_param(ctx, h):
        weight = ctx.get_parameter("lm_head.weight")
        ctx.saved["shape"] = list(weight.shape)
        ctx.saved["dtype"] = str(weight.dtype)
        return None

    hook = Hook(fn=check_param, layer_indices=[31])  # last layer
    output = client.generate("Hello", max_tokens=1, hooks=[hook])
    assert output.hook_results is not None
    shape = output.hook_results["0"]["shape"]
    # Llama 8B vocab size is 128256, hidden dim 4096
    assert shape[0] == 128256, f"Expected full vocab size, got {shape[0]}"
    assert shape[1] == 4096, f"Expected hidden dim 4096, got {shape[1]}"


def test_prefetch_parameter(vllm_server):
    """Prefetched parameters should be accessible from any layer."""
    client = _make_client(vllm_server)
    client.clear_hooks()

    def check_prefetched(ctx, h):
        weight = ctx.get_parameter("lm_head.weight")
        ctx.saved["shape"] = list(weight.shape)
        return None

    # Hook on layer 0 (early layer) but access lm_head (last PP stage).
    # Without prefetch this would fail on PP>1; with prefetch it works.
    hook = Hook(fn=check_prefetched, layer_indices=[0])
    client.register_hooks([hook], prefetch_params=["lm_head.weight"])
    client.generate("Hello", max_tokens=1)

    results = client.collect_hook_results()
    assert len(results) >= 1
    for req_id, hook_data in results.items():
        shape = hook_data["0"]["shape"]
        assert shape[0] == 128256
        assert shape[1] == 4096

    client.clear_hooks()


# ---------------------------------------------------------------------------
# Per-request hooks
# ---------------------------------------------------------------------------


def test_per_request_hook_capture(vllm_server):
    client = _make_client(vllm_server)

    def capture(ctx, h):
        ctx.saved["mean"] = h.mean().item()
        return None

    hook = Hook(fn=capture, layer_indices=[15])
    output = client.generate("Hello", max_tokens=1, hooks=[hook])
    assert output.hook_results is not None
    assert "0" in output.hook_results
    assert "mean" in output.hook_results["0"]


def test_per_request_hook_modification(vllm_server):
    client = _make_client(vllm_server)

    baseline = client.generate("The capital of France is", max_tokens=20)

    def zero_out(ctx, h):
        return torch.zeros_like(h)

    hook = Hook(fn=zero_out, layer_indices=[15])
    modified = client.generate("The capital of France is", max_tokens=20, hooks=[hook])
    assert modified.text != baseline.text


# ---------------------------------------------------------------------------
# Steering vectors
# ---------------------------------------------------------------------------


def test_steering_via_client(vllm_server):
    client = _make_client(vllm_server)

    baseline = client.generate("I think the best dessert is", max_tokens=20)

    sv = SteeringVector(
        activations=torch.randn(1, 4096),
        layer_indices=[15],
        scale=15.0,
        norm_match=True,
    )
    steered = client.generate(
        "I think the best dessert is", max_tokens=20, steering_vectors=[sv]
    )
    assert steered.text != baseline.text


# ---------------------------------------------------------------------------
# Persistent hooks
# ---------------------------------------------------------------------------


def test_persistent_lifecycle(vllm_server):
    client = _make_client(vllm_server)
    client.clear_hooks()  # clean slate

    def tag(ctx, h):
        ctx.saved["seen"] = True
        return None

    hook = Hook(fn=tag, layer_indices=[15])
    client.register_hooks([hook])

    client.generate("Hello", max_tokens=1)
    client.generate("World", max_tokens=1)

    results = client.collect_hook_results()
    assert len(results) >= 2

    for req_id, hook_data in results.items():
        assert "0" in hook_data
        assert hook_data["0"]["seen"] is True

    client.clear_hooks()

    empty = client.collect_hook_results()
    assert len(empty) == 0


def test_persistent_collect_non_destructive(vllm_server):
    client = _make_client(vllm_server)
    client.clear_hooks()

    def tag(ctx, h):
        ctx.saved["x"] = 1
        return None

    client.register_hooks([Hook(fn=tag, layer_indices=[15])])
    client.generate("Hello", max_tokens=1)

    first = client.collect_hook_results()
    second = client.collect_hook_results()
    assert len(first) > 0
    assert len(first) == len(second)

    client.clear_hooks()
