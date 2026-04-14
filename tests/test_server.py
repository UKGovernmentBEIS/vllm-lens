"""Integration tests for vllm-lens features against a running vLLM server.

These test activation extraction, steering vectors, and the generic hook
system over the OpenAI-compatible HTTP API.
"""

import json

import requests
import torch
from vllm_lens import Hook, deserialize_hook_results, deserialize_tensor, SteeringVector

from .conftest import MODEL


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------


def test_activation_extraction(vllm_server):
    resp = requests.post(
        f"{vllm_server}/v1/completions",
        json={
            "model": MODEL,
            "prompt": "The future of AI is",
            "max_tokens": 10,
            "temperature": 0.0,
            "vllm_xargs": {"output_residual_stream": "[15]"},
        },
    ).json()

    assert "error" not in resp, resp
    rs = deserialize_tensor(resp["activations"]["residual_stream"])
    assert rs.ndim == 3
    assert rs.shape[2] == 4096  # Llama 8B hidden dim


# ---------------------------------------------------------------------------
# Steering vectors
# ---------------------------------------------------------------------------


def test_steering_changes_output(vllm_server):
    prompt = "I think the best dessert is"

    baseline = requests.post(
        f"{vllm_server}/v1/completions",
        json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": 20,
            "temperature": 0.0,
        },
    ).json()

    sv = SteeringVector(
        activations=torch.randn(1, 4096),
        layer_indices=[15],
        scale=15.0,
        norm_match=True,
    )
    steered = requests.post(
        f"{vllm_server}/v1/completions",
        json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": 20,
            "temperature": 0.0,
            "vllm_xargs": {
                "apply_steering_vectors": json.dumps([sv.model_dump()]),
            },
        },
    ).json()

    assert steered["choices"][0]["text"] != baseline["choices"][0]["text"]


def test_steering_scale_zero_matches_baseline(vllm_server):
    prompt = "The capital of France is"

    baseline = requests.post(
        f"{vllm_server}/v1/completions",
        json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": 20,
            "temperature": 0.0,
        },
    ).json()

    sv = SteeringVector(
        activations=torch.randn(1, 4096),
        layer_indices=[15],
        scale=0.0,
    )
    steered = requests.post(
        f"{vllm_server}/v1/completions",
        json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": 20,
            "temperature": 0.0,
            "vllm_xargs": {
                "apply_steering_vectors": json.dumps([sv.model_dump()]),
            },
        },
    ).json()

    assert steered["choices"][0]["text"] == baseline["choices"][0]["text"]


# ---------------------------------------------------------------------------
# Generic hooks
# ---------------------------------------------------------------------------


def test_hook_capture(vllm_server):
    def capture_norm(ctx, h):
        ctx.saved[f"norm_L{ctx.layer_idx}"] = h.norm(dim=-1).cpu()
        return None

    hook = Hook(fn=capture_norm, layer_indices=[15, 16])
    resp = requests.post(
        f"{vllm_server}/v1/completions",
        json={
            "model": MODEL,
            "prompt": "Hello world",
            "max_tokens": 5,
            "temperature": 0.0,
            "vllm_xargs": {
                "apply_hooks": json.dumps([hook.model_dump()]),
            },
        },
    ).json()

    assert "error" not in resp, resp
    assert "hook_results" in resp
    results = deserialize_hook_results(resp["hook_results"])
    hook_0 = results["0"]
    assert "norm_L15" in hook_0
    assert "norm_L16" in hook_0
    assert isinstance(hook_0["norm_L15"], torch.Tensor)


def test_hook_modification_changes_output(vllm_server):
    prompt = "The meaning of life is"

    baseline = requests.post(
        f"{vllm_server}/v1/completions",
        json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": 20,
            "temperature": 0.0,
        },
    ).json()

    def zero_residual(ctx, h):
        return torch.zeros_like(h)

    hook = Hook(fn=zero_residual, layer_indices=[15])
    modified = requests.post(
        f"{vllm_server}/v1/completions",
        json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": 20,
            "temperature": 0.0,
            "vllm_xargs": {
                "apply_hooks": json.dumps([hook.model_dump()]),
            },
        },
    ).json()

    assert "error" not in modified, modified
    assert modified["choices"][0]["text"] != baseline["choices"][0]["text"]


def test_hook_none_preserves_output(vllm_server):
    prompt = "The capital of France is"

    baseline = requests.post(
        f"{vllm_server}/v1/completions",
        json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": 20,
            "temperature": 0.0,
        },
    ).json()

    def noop_hook(ctx, h):
        return None

    hook = Hook(fn=noop_hook, layer_indices=[15])
    hooked = requests.post(
        f"{vllm_server}/v1/completions",
        json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": 20,
            "temperature": 0.0,
            "vllm_xargs": {
                "apply_hooks": json.dumps([hook.model_dump()]),
            },
        },
    ).json()

    assert "error" not in hooked, hooked
    assert hooked["choices"][0]["text"] == baseline["choices"][0]["text"]


def test_hook_matches_native_activation_extraction(vllm_server):
    """A hook capturing hidden states must produce identical activations
    to the native ``output_residual_stream`` system."""
    layer = 15

    def capture_accumulate(ctx, h):
        if "parts" not in ctx.saved:
            ctx.saved["parts"] = []
        ctx.saved["parts"].append(h.cpu())
        return None

    hook = Hook(fn=capture_accumulate, layer_indices=[layer])

    # Request both native capture and hook capture in the same call.
    resp = requests.post(
        f"{vllm_server}/v1/completions",
        json={
            "model": MODEL,
            "prompt": "The future of AI is",
            "max_tokens": 10,
            "temperature": 0.0,
            "vllm_xargs": {
                "output_residual_stream": f"[{layer}]",
                "apply_hooks": json.dumps([hook.model_dump()]),
            },
        },
    ).json()

    assert "error" not in resp, resp

    native_all = deserialize_tensor(resp["activations"]["residual_stream"])
    native_layer = native_all[0]  # (seq_len, hidden_dim) — single layer requested

    hook_results = deserialize_hook_results(resp["hook_results"])
    parts = hook_results["0"]["parts"]
    hook_acts = torch.cat(parts, dim=0)

    min_len = min(native_layer.shape[0], hook_acts.shape[0])
    n = native_layer[:min_len].float()
    h = hook_acts[:min_len].float()

    mean_diff = (n - h).abs().mean().item()
    assert mean_diff < 1e-4, f"Mean abs diff {mean_diff} too large"


# ---------------------------------------------------------------------------
# Pre-hooks
# ---------------------------------------------------------------------------


def test_pre_hook_modification_changes_output(vllm_server):
    """A pre-hook that corrupts layer 0 input should change the output."""
    prompt = "The capital of France is"

    baseline = requests.post(
        f"{vllm_server}/v1/completions",
        json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": 20,
            "temperature": 0.0,
        },
    ).json()

    def corrupt_input(ctx, h):
        gen = torch.Generator(device=h.device).manual_seed(42)
        noise = torch.randn(h.shape, generator=gen, device=h.device, dtype=h.dtype)
        return h + noise * 100.0

    hook = Hook(fn=corrupt_input, layer_indices=[0], pre=True)
    corrupted = requests.post(
        f"{vllm_server}/v1/completions",
        json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": 20,
            "temperature": 0.0,
            "vllm_xargs": {
                "apply_hooks": json.dumps([hook.model_dump()]),
            },
        },
    ).json()

    assert "error" not in corrupted, corrupted
    assert corrupted["choices"][0]["text"] != baseline["choices"][0]["text"]


def test_pre_hook_none_preserves_output(vllm_server):
    """A pre-hook returning None should not change the output."""
    prompt = "The capital of France is"

    baseline = requests.post(
        f"{vllm_server}/v1/completions",
        json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": 20,
            "temperature": 0.0,
        },
    ).json()

    def noop(ctx, h):
        return None

    hook = Hook(fn=noop, layer_indices=[0], pre=True)
    hooked = requests.post(
        f"{vllm_server}/v1/completions",
        json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": 20,
            "temperature": 0.0,
            "vllm_xargs": {
                "apply_hooks": json.dumps([hook.model_dump()]),
            },
        },
    ).json()

    assert "error" not in hooked, hooked
    assert hooked["choices"][0]["text"] == baseline["choices"][0]["text"]
