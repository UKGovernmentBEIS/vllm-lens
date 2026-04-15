"""Causal tracing (activation patching) à la ROME (Meng et al., 2022).

Locates where factual knowledge is stored in a transformer by:
1. Running a clean forward pass and saving all hidden states.
2. Corrupting the subject token embeddings (via a pre-hook on layer 0).
3. For each (layer, token position), restoring the clean hidden state
   and measuring how much the correct-answer probability recovers.

The output is a (n_layers, n_tokens) heatmap of causal importance.

Usage:
    python -m vllm_lens._examples.causal_tracing \\
        --base-url http://localhost:8000 \\
        --prompt "The Eiffel Tower is in" \\
        --subject "Eiffel Tower" \\
        --answer " Paris"

Requires a running vLLM server with vllm-lens installed.
"""

from __future__ import annotations

import argparse
from typing import Any

import torch
from vllm_lens import Hook

from ..client import GenerateOutput, VLLMLensClient
from ._utils import N_LAYERS


def get_answer_logprob(output: GenerateOutput, answer_token: str) -> float:
    """Extract the log-probability of a specific token from the output."""
    if output.logprobs is None:
        return -100.0
    top = output.logprobs["top_logprobs"][0]
    if answer_token in top:
        return top[answer_token]
    # Not in top-k — return a very low logprob.
    return -100.0


def find_subject_positions(
    client: VLLMLensClient, prompt: str, subject: str
) -> tuple[list[int], list[str]]:
    """Find which token positions correspond to the subject string.

    Returns ``(subject_positions, all_tokens)`` where positions are
    0-indexed into the full token list (including BOS).
    """
    output = client.generate(prompt, max_tokens=1, logprobs=1, echo=True)
    resp = output.raw

    tokens = resp["choices"][0]["logprobs"]["tokens"]
    # tokens[0] is typically BOS (<|begin_of_text|>), rest map to the prompt.
    # Reconstruct the prompt from non-BOS tokens to find character offsets.
    char_start = prompt.find(subject)
    if char_start == -1:
        raise ValueError(f"Subject {subject!r} not found in prompt {prompt!r}")
    char_end = char_start + len(subject)

    # Build character position map, skipping special tokens that don't
    # appear in the raw prompt string.
    subject_positions = []
    char_pos = 0
    for i, tok in enumerate(tokens):
        # Special tokens (BOS, etc.) don't consume prompt characters.
        if tok.startswith("<|") and tok.endswith("|>"):
            continue
        tok_start = char_pos
        tok_end = char_pos + len(tok)
        if tok_end > char_start and tok_start < char_end:
            subject_positions.append(i)
        char_pos = tok_end

    return subject_positions, tokens


def run_causal_trace(
    client: VLLMLensClient,
    prompt: str,
    subject: str,
    answer_token: str,
    noise_scale: float = 3.0,
) -> dict[str, Any]:
    """Run the full causal tracing experiment.

    Returns a dict with:
    - clean_logprob: log P(answer) on clean input
    - corrupted_logprob: log P(answer) with noised subject
    - patch_logprobs: (n_layers, n_tokens) tensor of log P(answer)
      when restoring clean state at each (layer, position)
    - tokens: list of token strings
    - subject_positions: which tokens are the subject
    """
    print(f"Prompt: {prompt!r}")
    print(f"Subject: {subject!r}")
    print(f"Expected answer token: {answer_token!r}")

    # --- Step 1: Clean run — capture all hidden states ---
    print("\n[1/3] Clean run with activation capture...")
    all_layers = list(range(N_LAYERS))

    def capture_all(ctx, h):
        # Key by layer index for correct ordering with PP.
        ctx.saved[f"L{ctx.layer_idx}"] = h.cpu()
        return None

    capture_hook = Hook(fn=capture_all, layer_indices=all_layers)
    clean_output = client.generate(prompt, max_tokens=1, hooks=[capture_hook], logprobs=20)

    clean_logprob = get_answer_logprob(clean_output, answer_token)
    print(f"  Clean logprob({answer_token!r}): {clean_logprob:.4f}")

    assert clean_output.hook_results is not None
    saved = clean_output.hook_results["0"]
    n_tokens = saved["L0"].shape[0]
    print(f"  Captured {n_tokens} token positions across {N_LAYERS} layers")

    # Get tokens and subject positions.
    subject_positions, all_tokens = find_subject_positions(client, prompt, subject)
    tokens = all_tokens  # include generated token for display context
    print(f"  Subject positions: {subject_positions}")
    print(f"  Tokens: {tokens}")

    # --- Step 2: Corrupted run — noise on subject tokens ---
    print("\n[2/3] Corrupted run (noise on subject tokens)...")

    # We use a pre-hook on layer 0 to add noise to the subject token
    # embeddings. The noise is deterministic (seeded) so all TP ranks agree.
    noise_seed = 42

    def corrupt_subject(ctx, h):
        """Add Gaussian noise to subject token positions."""
        gen = torch.Generator(device=h.device).manual_seed(noise_seed)
        for pos in subject_positions:
            if pos < h.shape[0]:
                noise = torch.randn(
                    h.shape[-1], generator=gen, device=h.device, dtype=h.dtype
                )
                h = h.clone() if pos == subject_positions[0] else h
                h[pos] = h[pos] + noise * noise_scale
        return h

    corrupt_hook = Hook(fn=corrupt_subject, layer_indices=[0], pre=True)
    corrupted_output = client.generate(
        prompt, max_tokens=1, hooks=[corrupt_hook], logprobs=20,
    )

    corrupted_logprob = get_answer_logprob(corrupted_output, answer_token)
    print(f"  Corrupted logprob({answer_token!r}): {corrupted_logprob:.4f}")
    print(f"  Generated: {corrupted_output.text!r}")

    # --- Step 3: Patch runs — restore clean state at each (layer, pos) ---
    print(f"\n[3/3] Patching {N_LAYERS} layers × {n_tokens} positions...")

    patch_logprobs = torch.zeros(N_LAYERS, n_tokens)

    for layer_idx in range(N_LAYERS):
        clean_layer_acts = saved[f"L{layer_idx}"]  # (n_tokens, hidden_dim)

        for token_pos in range(n_tokens):
            # Create a hook that: (1) corrupts subject at layer 0 pre-hook,
            # (2) restores clean hidden state at (layer_idx, token_pos) post-hook.
            clean_vec = clean_layer_acts[token_pos]  # (hidden_dim,)

            def make_patch_hook(target_pos, target_vec):
                def patch(ctx, h):
                    h = h.clone()
                    if target_pos < h.shape[0]:
                        h[target_pos] = target_vec.to(h.device, h.dtype)
                    return h

                return patch

            patch_hook = Hook(
                fn=make_patch_hook(token_pos, clean_vec),
                layer_indices=[layer_idx],
            )

            patched_output = client.generate(
                prompt,
                max_tokens=1,
                hooks=[corrupt_hook, patch_hook],
                logprobs=20,
            )
            lp = get_answer_logprob(patched_output, answer_token)
            patch_logprobs[layer_idx, token_pos] = lp

        print(
            f"  Layer {layer_idx:2d}/{N_LAYERS}: "
            f"max recovery = {patch_logprobs[layer_idx].max():.4f}"
        )

    return {
        "clean_logprob": clean_logprob,
        "corrupted_logprob": corrupted_logprob,
        "patch_logprobs": patch_logprobs,
        "tokens": tokens,
        "subject_positions": subject_positions,
    }


def print_heatmap(results: dict[str, Any]) -> None:
    """Print a simple ASCII heatmap of the causal trace."""
    clean = results["clean_logprob"]
    corrupted = results["corrupted_logprob"]
    patch = results["patch_logprobs"]
    tokens = results["tokens"]

    # Normalize: 0 = corrupted, 1 = clean
    recovery = (patch - corrupted) / (clean - corrupted + 1e-8)
    recovery = recovery.clamp(0, 1)

    chars = " ░▒▓█"
    print("\nCausal trace heatmap (rows=layers, cols=tokens)")
    print(f"  Clean logprob: {clean:.4f}, Corrupted: {corrupted:.4f}")
    print()

    # Header
    header = "     " + "".join(f"{t[:6]:>7s}" for t in tokens)
    print(header)

    for layer_idx in range(recovery.shape[0]):
        row = f"L{layer_idx:02d}  "
        for pos in range(recovery.shape[1]):
            v = recovery[layer_idx, pos].item()
            ci = min(int(v * len(chars)), len(chars) - 1)
            row += f"  {chars[ci]}    "
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Causal tracing via vllm-lens")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--prompt", default="The Eiffel Tower is in the city of")
    parser.add_argument("--subject", default="Eiffel Tower")
    parser.add_argument("--answer", default=" Paris")
    parser.add_argument("--noise-scale", type=float, default=3.0)
    args = parser.parse_args()

    client = VLLMLensClient(args.base_url)
    results = run_causal_trace(
        client,
        args.prompt,
        args.subject,
        args.answer,
        args.noise_scale,
    )

    print_heatmap(results)

    # Save results
    torch.save(results, "causal_trace_results.pt")
    print("\nResults saved to causal_trace_results.pt")


if __name__ == "__main__":
    main()
