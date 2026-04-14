"""Benchmark: session-based (persistent) vs per-request activation extraction.

Compares wall-clock time for extracting activations from N prompts using:
1. Per-request: output_residual_stream in each request (results in response)
2. Persistent hook: register once, run all, collect once at the end

Usage:
    # Against a running server:
    python examples/bench_session_vs_per_request.py --base-url http://localhost:8000

    # With options:
    python examples/bench_session_vs_per_request.py --n-prompts 50 --layer 15 --max-tokens 10
"""

from __future__ import annotations

import argparse
import json
import time

import requests
import torch
from vllm_lens import Hook, deserialize_hook_results, deserialize_tensor

MODEL = "meta-llama/Llama-3.1-8B-Instruct"

PROMPTS = [
    "The future of artificial intelligence is",
    "In the beginning, there was",
    "The most important scientific discovery of the 21st century is",
    "When I think about the meaning of life, I",
    "The relationship between mathematics and physics",
    "A good leader should always",
    "The history of computing begins with",
    "One of the greatest challenges facing humanity is",
    "The beauty of nature can be seen in",
    "If I could travel anywhere in time, I would",
    "The development of language shaped human civilization by",
    "Music has the power to",
    "The ocean covers most of our planet and",
    "Education is the foundation of",
    "The stars in the night sky remind us that",
    "Technology has transformed the way we",
    "The art of storytelling has been",
    "Climate change is affecting our world by",
    "The human brain is capable of",
    "Philosophy helps us understand",
    "The invention of the printing press",
    "Democracy requires active participation from",
    "The theory of evolution explains",
    "Space exploration has revealed",
    "The power of imagination allows us to",
    "Ancient civilizations contributed to modern society by",
    "The internet has fundamentally changed",
    "Creativity is essential for",
    "The pursuit of knowledge drives",
    "The balance between work and life is",
    "Renewable energy sources offer",
    "The complexity of ecosystems shows",
    "Cultural diversity enriches",
    "The role of empathy in society is",
    "Scientific method requires",
    "The evolution of transportation has",
    "Global cooperation is needed to",
    "The importance of critical thinking",
    "Artificial neural networks are inspired by",
    "The future of medicine will be shaped by",
    "Literature reflects the human experience through",
    "The quantum world behaves differently because",
    "Sustainable development means",
    "The human immune system protects us by",
    "Innovation often comes from",
    "The study of history teaches us",
    "Biodiversity is important because",
    "The social contract between citizens and",
    "Mathematical patterns appear throughout",
    "The exploration of consciousness remains",
]


def bench_per_request(
    base_url: str, prompts: list[str], layers: list[int], max_tokens: int
) -> tuple[float, list[torch.Tensor]]:
    """Extract activations per-request via output_residual_stream."""
    activations = []
    layers_str = json.dumps(layers)
    t0 = time.perf_counter()
    for prompt in prompts:
        resp = requests.post(f"{base_url}/v1/completions", json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "vllm_xargs": {"output_residual_stream": layers_str},
        }).json()
        assert "error" not in resp, resp
        rs = deserialize_tensor(resp["activations"]["residual_stream"])
        activations.append(rs)
    elapsed = time.perf_counter() - t0
    return elapsed, activations


def bench_persistent(
    base_url: str, prompts: list[str], layers: list[int], max_tokens: int
) -> tuple[float, list[torch.Tensor]]:
    """Extract activations via persistent hook + bulk collection."""

    def capture(ctx, h):
        if "parts" not in ctx.saved:
            ctx.saved["parts"] = []
        ctx.saved["parts"].append(h.cpu())
        return None

    hook = Hook(fn=capture, layer_indices=layers)

    t0 = time.perf_counter()

    # Register once
    resp = requests.post(f"{base_url}/v1/hooks/register", json={
        "hooks": [hook.model_dump()],
    }).json()
    assert resp["status"] == "ok"

    # Run all prompts (no activation data in responses)
    for prompt in prompts:
        resp = requests.post(f"{base_url}/v1/completions", json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }).json()
        assert "error" not in resp, resp

    # Collect all at once
    collect_resp = requests.post(f"{base_url}/v1/hooks/collect").json()
    results = collect_resp["results"]

    # Deserialize
    activations = []
    for req_id in sorted(results.keys()):
        hook_data = deserialize_hook_results(results[req_id])
        parts = hook_data["0"]["parts"]
        activations.append(torch.cat(parts, dim=0))

    elapsed = time.perf_counter() - t0

    # Cleanup
    requests.post(f"{base_url}/v1/hooks/clear")

    return elapsed, activations


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark session vs per-request activation extraction"
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--n-prompts", type=int, default=50)
    parser.add_argument("--layer", type=int, default=15,
                        help="Layer index to capture, or -1 for all 32 layers")
    parser.add_argument("--max-tokens", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warmup requests before benchmarking")
    args = parser.parse_args()

    prompts = PROMPTS[:args.n_prompts]
    if len(prompts) < args.n_prompts:
        # Cycle if we need more
        prompts = (prompts * (args.n_prompts // len(prompts) + 1))[:args.n_prompts]

    layer = args.layer
    all_layers = list(range(32)) if layer == -1 else [layer]
    layer_desc = "all 32" if layer == -1 else str(layer)
    print(f"Config: {args.n_prompts} prompts, layer {layer_desc}, "
          f"max_tokens={args.max_tokens}")
    print(f"Server: {args.base_url}\n")

    # Warmup
    print(f"Warming up with {args.warmup} requests...")
    for p in prompts[:args.warmup]:
        requests.post(f"{args.base_url}/v1/completions", json={
            "model": MODEL, "prompt": p, "max_tokens": 1, "temperature": 0.0,
        })

    # Benchmark per-request
    print("\n--- Per-request activation extraction ---")
    t_per, acts_per = bench_per_request(
        args.base_url, prompts, all_layers, args.max_tokens
    )
    print(f"  Time: {t_per:.2f}s ({t_per/len(prompts)*1000:.1f}ms/prompt)")
    print(f"  Collected {len(acts_per)} activation tensors")

    # Benchmark persistent
    print("\n--- Persistent (session) activation extraction ---")
    t_persistent, acts_persistent = bench_persistent(
        args.base_url, prompts, all_layers, args.max_tokens
    )
    print(f"  Time: {t_persistent:.2f}s ({t_persistent/len(prompts)*1000:.1f}ms/prompt)")
    print(f"  Collected {len(acts_persistent)} activation tensors")

    # Compare
    print(f"\n--- Comparison ---")
    speedup = t_per / t_persistent if t_persistent > 0 else float("inf")
    print(f"  Per-request:  {t_per:.2f}s")
    print(f"  Persistent:   {t_persistent:.2f}s")
    print(f"  Speedup:      {speedup:.2f}x")

    # Note: correctness (activation match) is verified by the test suite
    # (test_persistent_capture_matches_native). The persistent path returns
    # results keyed by internal request IDs with no guaranteed ordering,
    # so per-prompt comparison requires extra bookkeeping.
    print(f"\n  Shapes: per-request {acts_per[0].shape}, "
          f"persistent {acts_persistent[0].shape}")


if __name__ == "__main__":
    main()
