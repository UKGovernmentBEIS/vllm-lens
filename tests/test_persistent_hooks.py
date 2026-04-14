"""Integration tests for persistent (Garçon-style) hook management.

Tests the /v1/hooks/register, /v1/hooks/collect, and /v1/hooks/clear
endpoints for user-controlled hook lifecycle.
"""

import json

import requests
import torch
from vllm_lens import Hook, deserialize_hook_results

from .conftest import MODEL


def _completions(base_url: str, prompt: str, max_tokens: int = 5) -> dict:
    return requests.post(f"{base_url}/v1/completions", json={
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).json()


def test_register_collect_clear(vllm_server):
    """Full lifecycle: register hooks, run requests, collect, clear."""
    def capture_mean(ctx, h):
        ctx.saved[f"mean_L{ctx.layer_idx}"] = h.mean(dim=-1).cpu()
        return None

    hook = Hook(fn=capture_mean, layer_indices=[15])

    # Register
    resp = requests.post(f"{vllm_server}/v1/hooks/register", json={
        "hooks": [hook.model_dump()],
    }).json()
    assert resp["status"] == "ok"
    assert resp["count"] == 1

    # Run two requests — hooks fire, results stay server-side
    r1 = _completions(vllm_server, "Hello world")
    assert "error" not in r1, r1
    assert "hook_results" not in r1  # not auto-collected

    r2 = _completions(vllm_server, "Goodbye world")
    assert "error" not in r2, r2

    # Collect all results
    collect_resp = requests.post(f"{vllm_server}/v1/hooks/collect").json()
    results = collect_resp["results"]
    assert len(results) >= 2, f"Expected >= 2 requests, got {len(results)}"

    # Each request should have hook 0 with mean_L15
    for req_id, hook_data in results.items():
        deserialized = deserialize_hook_results(hook_data)
        assert "0" in deserialized, f"Missing hook 0 for {req_id}"
        assert "mean_L15" in deserialized["0"], f"Missing mean_L15 for {req_id}"
        assert isinstance(deserialized["0"]["mean_L15"], torch.Tensor)

    # Clear
    clear_resp = requests.post(f"{vllm_server}/v1/hooks/clear").json()
    assert clear_resp["status"] == "ok"

    # Collect again — should be empty
    empty = requests.post(f"{vllm_server}/v1/hooks/collect").json()
    assert len(empty["results"]) == 0


def test_persistent_hooks_coexist_with_per_request(vllm_server):
    """Persistent hooks and per-request hooks can be active simultaneously."""
    def persistent_hook(ctx, h):
        ctx.saved["persistent"] = True
        return None

    def per_request_hook(ctx, h):
        ctx.saved["per_request"] = True
        return None

    persistent = Hook(fn=persistent_hook, layer_indices=[15])
    per_request = Hook(fn=per_request_hook, layer_indices=[15])

    # Register persistent hook
    requests.post(f"{vllm_server}/v1/hooks/register", json={
        "hooks": [persistent.model_dump()],
    })

    # Run with per-request hook too
    resp = requests.post(f"{vllm_server}/v1/completions", json={
        "model": MODEL,
        "prompt": "Test both",
        "max_tokens": 5,
        "temperature": 0.0,
        "vllm_xargs": {
            "apply_hooks": json.dumps([per_request.model_dump()]),
        },
    }).json()

    assert "error" not in resp, resp
    # Per-request hook results come back immediately
    assert "hook_results" in resp
    per_req_results = deserialize_hook_results(resp["hook_results"])
    assert per_req_results["0"]["per_request"] is True

    # Persistent hook results are server-side
    collect_resp = requests.post(f"{vllm_server}/v1/hooks/collect").json()
    results = collect_resp["results"]
    assert len(results) >= 1
    for req_id, hook_data in results.items():
        deserialized = deserialize_hook_results(hook_data)
        # Persistent hook is the last in the list (after per-request hooks)
        found_persistent = any(
            "persistent" in hd for hd in deserialized.values()
        )
        assert found_persistent, f"Missing persistent hook data for {req_id}"

    # Cleanup
    requests.post(f"{vllm_server}/v1/hooks/clear")


def test_persistent_modification_changes_output(vllm_server):
    """A persistent modifying hook should affect all requests."""
    prompt = "The capital of France is"

    # Baseline without hooks
    baseline = _completions(vllm_server, prompt, max_tokens=20)
    baseline_text = baseline["choices"][0]["text"]

    # Register a hook that zeros hidden states
    def zero_hook(ctx, h):
        return torch.zeros_like(h)

    hook = Hook(fn=zero_hook, layer_indices=[15])
    requests.post(f"{vllm_server}/v1/hooks/register", json={
        "hooks": [hook.model_dump()],
    })

    # Same prompt should now produce different output
    hooked = _completions(vllm_server, prompt, max_tokens=20)
    assert hooked["choices"][0]["text"] != baseline_text

    # Cleanup and verify baseline is restored
    requests.post(f"{vllm_server}/v1/hooks/clear")
    restored = _completions(vllm_server, prompt, max_tokens=20)
    assert restored["choices"][0]["text"] == baseline_text
