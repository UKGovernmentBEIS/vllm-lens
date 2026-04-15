"""Logit lens: see what the model "thinks" at each layer.

Projects hidden states from each layer through the final layer norm
and unembedding matrix (lm_head) to reveal how predictions evolve
across layers. Early layers often predict generic tokens, with the
final answer crystallizing in later layers.

Reference: https://www.lesswrong.com/posts/AcKRB8wDds238WkfB

Usage:
    python -m vllm_lens._examples.logit_lens \\
        --base-url http://localhost:8000 \\
        --prompt "The Eiffel Tower is in the city of"

Requires a running vLLM server with vllm-lens installed.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch

from vllm_lens import Hook, deserialize_hook_results

from ._utils import N_LAYERS, completions, find_norm


def run_logit_lens(
    base_url: str,
    prompt: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """Run the logit lens on a prompt.

    Uses a hook that projects hidden states through the model's own
    lm_head on the GPU, avoiding large activation transfers.
    """
    print(f"Prompt: {prompt!r}\n")

    all_layers = list(range(N_LAYERS))

    def project_hook(ctx, h):
        """Project hidden states through norm + unembed weight, save top-k."""
        weight = ctx.get_parameter("lm_head.weight")
        norm = find_norm(ctx.model)
        with torch.no_grad():
            normed = norm(h) if norm is not None else h
            logits = normed.float() @ weight.float().T  # (seq_len, vocab_size)
            topk = logits.topk(top_k, dim=-1)
        if "top_ids" not in ctx.saved:
            ctx.saved["top_ids"] = []
            ctx.saved["top_logits"] = []
        ctx.saved["top_ids"].append(topk.indices.cpu())
        ctx.saved["top_logits"].append(topk.values.float().cpu())
        return None

    hook = Hook(fn=project_hook, layer_indices=all_layers)
    resp = completions(
        base_url,
        prompt,
        vllm_xargs={
            "apply_hooks": json.dumps([hook.model_dump()]),
        },
    )
    assert "error" not in resp, resp

    tokens = resp["choices"][0]["logprobs"]["tokens"][:-1]
    generated = resp["choices"][0]["text"]
    print(f"Generated: {generated!r}")
    print(f"Tokens ({len(tokens)}): {tokens}\n")

    hook_results = deserialize_hook_results(resp["hook_results"])
    top_ids = hook_results["0"]["top_ids"]
    top_logits = hook_results["0"]["top_logits"]

    return {
        "tokens": tokens,
        "generated": generated,
        "top_ids": top_ids,
        "top_logits": top_logits,
        "n_layers": N_LAYERS,
    }


def print_logit_lens(results: dict[str, Any], focus_position: int = -1) -> None:
    """Print the logit lens for a specific token position."""
    tokens = results["tokens"]
    top_ids = results["top_ids"]
    top_logits = results["top_logits"]
    n_layers = results["n_layers"]

    if focus_position < 0:
        focus_position = len(tokens) + focus_position

    print(
        f"Logit lens at position {focus_position} (token: {tokens[focus_position]!r})"
    )
    print("  Predicting the token AFTER this position.\n")
    print(f"{'Layer':>6s}  {'Top-1 ID':>8s}  {'Logit':>8s}  {'Top-5 IDs'}")
    print("-" * 60)

    for layer_idx in range(n_layers):
        ids = top_ids[layer_idx]
        logits = top_logits[layer_idx]

        pos_ids = ids[focus_position]
        pos_logits = logits[focus_position]

        top1_id = pos_ids[0].item()
        top1_logit = pos_logits[0].item()
        top5 = [str(pos_ids[k].item()) for k in range(pos_ids.shape[0])]

        print(
            f"L{layer_idx:02d}     {top1_id:>8d}  {top1_logit:>8.2f}  {', '.join(top5)}"
        )

    print(
        "\nNote: Token IDs shown (no tokenizer access over HTTP). "
        "Load results and decode locally:\n"
        "  results = torch.load('logit_lens_results.pt')\n"
        "  tokenizer.decode([id]) for each ID"
    )


def main():
    parser = argparse.ArgumentParser(description="Logit lens via vllm-lens")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--prompt", default="The Eiffel Tower is in the city of")
    parser.add_argument(
        "--position",
        type=int,
        default=-1,
        help="Token position to focus on (-1 = last)",
    )
    args = parser.parse_args()

    results = run_logit_lens(args.base_url, args.prompt)
    print_logit_lens(results, focus_position=args.position)

    torch.save(results, "logit_lens_results.pt")
    print("\nResults saved to logit_lens_results.pt")


if __name__ == "__main__":
    main()
