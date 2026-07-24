"""Attention patterns: reconstruct where the model attends, from a vLLM server.

vLLM's fused attention kernels never materialize attention weights, so
they can't be captured directly.  Instead, ``extra_args={"output_qk": ...}``
captures each layer's post-RoPE query/key tensors plus the exact kernel
parameters (scale, sliding window, logit soft-cap, ...), and
``vllm_lens.attention.attention_patterns`` replays the softmax offline —
giving the true attention matrix for any decoder layer.

Usage:
    python examples/attention_patterns.py \\
        --base-url http://localhost:8000 \\
        --prompt "The Eiffel Tower is in the city of" \\
        --layer 12

Requires a running vLLM server with vllm-lens installed.
"""

from __future__ import annotations

import argparse

from vllm_lens.attention import attention_patterns
from vllm_lens.client import VLLMLensClient


def show_attention(
    client: VLLMLensClient,
    prompt: str,
    layer: int,
    top_k: int = 4,
    max_tokens: int = 24,
) -> None:
    print(f"Prompt: {prompt!r}\n")

    output = client.generate(
        prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        capture_qk=[layer],
    )
    weights = attention_patterns(output.activations, layer)
    num_heads, seq_len, _ = weights.shape

    # Token strings for display, via the served model's tokenizer.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(client.model)
    token_ids = tokenizer(prompt).input_ids
    generated = output.text
    all_tokens = tokenizer.convert_ids_to_tokens(
        token_ids + tokenizer(generated, add_special_tokens=False).input_ids
    )[:seq_len]

    # Average over heads: for each of the last few positions, show the
    # source tokens it attends to most.
    mean_attn = weights.mean(dim=0)
    print(f"Layer {layer}, {num_heads} heads, {seq_len} positions")
    print(f"Top-{top_k} attended tokens for the last 8 positions (head-averaged):\n")
    for pos in range(max(1, seq_len - 8), seq_len):
        row = mean_attn[pos]
        top = row.topk(min(top_k, pos + 1))
        srcs = ", ".join(
            f"{all_tokens[j]!r}:{row[j]:.2f}"
            for j in top.indices.tolist()
            if j < len(all_tokens)
        )
        tok = all_tokens[pos] if pos < len(all_tokens) else "?"
        print(f"  {pos:4d} {tok!r:>14} <- {srcs}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--prompt", default="The Eiffel Tower is in the city of")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=24)
    args = parser.parse_args()

    client = VLLMLensClient(base_url=args.base_url, api_key=args.api_key)
    show_attention(
        client,
        args.prompt,
        layer=args.layer,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
