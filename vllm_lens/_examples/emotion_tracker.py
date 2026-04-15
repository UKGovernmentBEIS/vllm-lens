"""Emotion tracker: monitor emotion activations during generation.

Extracts emotion direction vectors from contrastive stories (following
Anthropic's "Emotion Concepts" methodology), then tracks projections
onto those directions token-by-token as the model generates a response.

Methodology (Sofroniew et al., 2026):
1. Extract activations from emotion stories, mean-pool from token 50+
2. Compute per-emotion mean, subtract grand mean across all emotions
3. Project out top PCA components of neutral transcripts (50% variance)
4. Monitor: dot product of per-token hidden states with emotion vectors

Usage:
    python -m vllm_lens._examples.emotion_tracker \\
        --base-url http://localhost:8000 \\
        --emotions anxiety amusement curiosity confidence excitement

Requires: pip install datasets matplotlib

Reference: https://transformer-circuits.pub/2026/emotions/index.html
"""

from __future__ import annotations

import argparse
from typing import Any

import torch
from datasets import load_dataset

from ..client import VLLMLensClient
from ._utils import N_LAYERS

# Layer ~2/3 through the model (following the paper).
PROBE_LAYER = int(N_LAYERS * 2 / 3)

DEFAULT_EMOTIONS = ["anxious", "amused", "desperate", "proud", "defiant"]
DEFAULT_PROMPT = "Hi, what's on your mind? Write a poem about it."
STORIES_PER_EMOTION = 1000
MIN_TOKEN_OFFSET = 50  # skip first 50 tokens per story


def load_stories(
    emotions: list[str],
    n_per_emotion: int = STORIES_PER_EMOTION,
) -> tuple[dict[str, list[str]], list[str]]:
    """Load emotion stories and neutral stories from HuggingFace.

    Returns (emotion_stories, neutral_stories).
    """
    print("Loading stories from HuggingFace...")
    ds = load_dataset(
        "ryancodrai/emotion-probes", data_files="expression/stories.parquet"
    )
    neutral_ds = load_dataset(
        "ryancodrai/emotion-probes",
        data_files="expression/neutral_stories.parquet",
    )

    stories_df = ds["train"]
    neutral_df = neutral_ds["train"]

    emotion_stories: dict[str, list[str]] = {}
    for emotion in emotions:
        matching = [row["story"] for row in stories_df if row["emotion"] == emotion][
            :n_per_emotion
        ]
        if not matching:
            print(f"  WARNING: No stories found for emotion '{emotion}'")
        else:
            emotion_stories[emotion] = matching
            print(f"  {emotion}: {len(matching)} stories")

    neutral_stories = [row["story"] for row in neutral_df][:n_per_emotion]
    print(f"  neutral: {len(neutral_stories)} stories")

    return emotion_stories, neutral_stories


def extract_mean_activations(
    client: VLLMLensClient,
    texts: list[str],
    layer: int,
    label: str = "",
) -> torch.Tensor:
    """Extract mean-pooled activations (from token 50+) for a list of texts.

    Returns tensor of shape (n_texts, hidden_dim).
    """
    from vllm_lens import Hook

    def capture_mean_from_50(ctx, h):
        # Mean-pool from token 50 onwards (skip preamble).
        start = min(MIN_TOKEN_OFFSET, h.shape[0] - 1)
        ctx.saved[f"mean_L{ctx.layer_idx}"] = h[start:].mean(dim=0).cpu()
        return None

    hook = Hook(fn=capture_mean_from_50, layer_indices=[layer])

    client.register_hooks([hook])
    for i, text in enumerate(texts):
        client.generate(text, max_tokens=1)
        if (i + 1) % 10 == 0:
            print(f"    {label} {i + 1}/{len(texts)}")

    results = client.collect_hook_results()
    client.clear_hooks()

    acts = []
    for req_id in sorted(results.keys()):
        hook_data = results[req_id]["0"]
        acts.append(hook_data[f"mean_L{layer}"])

    return torch.stack(acts)  # (n_texts, hidden_dim)


def compute_emotion_vectors(
    emotion_acts: dict[str, torch.Tensor],
    neutral_acts: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute emotion vectors following the Anthropic methodology.

    1. Per-emotion mean, subtract grand mean across emotions.
    2. PCA on neutral activations, project out top components (50% variance).
    """
    # Step 1: Per-emotion mean, subtract grand mean.
    emotion_means = {e: acts.mean(dim=0) for e, acts in emotion_acts.items()}
    grand_mean = torch.stack(list(emotion_means.values())).mean(dim=0)
    raw_vectors = {e: m - grand_mean for e, m in emotion_means.items()}

    # Step 2: PCA on neutral activations, project out confounds.
    neutral_centered = neutral_acts - neutral_acts.mean(dim=0)
    U, S, Vt = torch.linalg.svd(neutral_centered.float(), full_matrices=False)

    # Find K components explaining 50% of variance.
    variance = S**2
    cumvar = variance.cumsum(dim=0) / variance.sum()
    k = int((cumvar < 0.5).sum().item()) + 1
    print(f"  Projecting out {k} neutral PCA components (50% variance)")

    # Project out these components from emotion vectors.
    components = Vt[:k]  # (k, hidden_dim)
    clean_vectors = {}
    for e, v in raw_vectors.items():
        v_float = v.float()
        projections = (v_float @ components.T) @ components
        clean_vectors[e] = v_float - projections

    return clean_vectors


def track_emotions(
    client: VLLMLensClient,
    prompt: str,
    emotion_vectors: dict[str, torch.Tensor],
    layer: int,
    max_tokens: int = 100,
) -> dict[str, Any]:
    """Generate a response and track emotion projections token-by-token."""
    from vllm_lens import Hook

    emotions = list(emotion_vectors.keys())
    # Stack vectors for efficient batch dot product.
    V = torch.stack(
        [emotion_vectors[e] for e in emotions]
    ).float()  # (n_emotions, hidden_dim)

    def project_emotions(ctx, h):
        V_dev = V.to(h.device)
        projections = h.float() @ V_dev.T
        n = sum(1 for k in ctx.saved if k.startswith("p_"))
        ctx.saved[f"p_{n}"] = projections.cpu().tolist()
        return None

    hook = Hook(fn=project_emotions, layer_indices=[layer])
    client.prefetch_params(["lm_head.weight"])  # in case PP
    output = client.chat(
        [{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        hooks=[hook],
        logprobs=True,
        top_logprobs=1,
    )

    assert output.hook_results is not None
    saved = output.hook_results["0"]
    parts = [
        saved[k]
        for k in sorted(saved, key=lambda x: int(x.split("_")[1]))
        if k.startswith("p_")
    ]
    # Each part is a nested list [[e1, e2, ...], ...] per forward pass.
    all_rows = []
    for p in parts:
        all_rows.extend(p)
    projections = torch.tensor(all_rows)

    # Get token strings from chat logprobs. Take last N to align with response.
    tokens: list[str] = []
    if output.logprobs:
        if "content" in output.logprobs:
            tokens = [lp["token"] for lp in output.logprobs["content"]]
        elif "tokens" in output.logprobs:
            tokens = output.logprobs["tokens"]

    # Align: take last tokens.length projections (skip prompt prefix).
    if tokens:
        offset = max(0, projections.shape[0] - len(tokens))
        projections = projections[offset:]
    n = min(len(tokens), projections.shape[0])
    tokens = tokens[:n]
    projections = projections[:n]

    return {
        "tokens": tokens,
        "projections": projections,  # (n_tokens, n_emotions)
        "emotions": emotions,
        "text": output.text,
        "prompt": prompt,
    }


def compute_display_scale(projections: torch.Tensor) -> float:
    """99th percentile of absolute values, for [-1, 1] scaling.

    Skips the first token (BOS) which often has outlier activations.
    """
    vals = projections[1:] if projections.shape[0] > 1 else projections
    abs_vals = vals.abs()
    return float(torch.quantile(abs_vals.float(), 0.99).item())


def plot_matplotlib(results: dict[str, Any], output_path: str) -> None:
    """Generate a matplotlib line plot of emotion activations."""
    import matplotlib.pyplot as plt

    tokens = results["tokens"]
    projections = results["projections"]  # (n_tokens, n_emotions)
    emotions = results["emotions"]

    scale = compute_display_scale(projections)
    scaled = projections.float() / (scale + 1e-8)

    fig, ax = plt.subplots(figsize=(max(14, len(tokens) * 0.4), 6))

    colors = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261"]
    for i, emotion in enumerate(emotions):
        ax.plot(
            range(len(tokens)),
            scaled[:, i].numpy(),
            label=emotion,
            color=colors[i % len(colors)],
            linewidth=2,
            alpha=0.8,
        )

    ax.set_xlabel("Token", fontsize=12)
    ax.set_ylabel("Emotion activation (scaled)", fontsize=12)
    ax.set_title(
        f"Emotion activations during generation\nPrompt: {results['prompt'][:80]}...",
        fontsize=13,
    )
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(
        [t.strip() or t for t in tokens],
        rotation=75,
        ha="right",
        fontsize=7,
    )
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")


STEERING_LAYERS = list(range(16, 25))  # middle layers for steering


def _build_serialized_steering(
    emotion_vectors: dict[str, torch.Tensor], layer: int
) -> dict[str, Any]:
    """Build serialized SteeringVector per emotion for embedding in HTML.

    Steers across multiple middle layers for stronger effect.
    """

    from vllm_lens import SteeringVector

    result = {}
    n_layers = len(STEERING_LAYERS)
    for emotion, vec in emotion_vectors.items():
        # Repeat the vector across steering layers.
        sv = SteeringVector(
            activations=vec.unsqueeze(0).expand(n_layers, -1),
            layer_indices=STEERING_LAYERS,
            scale=1.0,  # JS will multiply by slider value
            norm_match=False,  # raw addition for stronger effect
        )
        result[emotion] = sv.model_dump()
    return result


def _build_serialized_hook(emotion_vectors: dict[str, torch.Tensor], layer: int) -> str:
    """Build a JSON-serialized hook for embedding in HTML.

    The hook projects hidden states onto emotion directions. The
    serialized form can be sent via vllm_xargs from JavaScript.
    """
    import json

    from vllm_lens import Hook

    emotions = list(emotion_vectors.keys())
    V = torch.stack([emotion_vectors[e] for e in emotions]).float()

    def project_emotions(ctx, h):
        V_dev = V.to(h.device)
        projections = h.float() @ V_dev.T  # (seq_len, n_emotions)
        # Store as plain nested lists so they serialize as JSON arrays
        # (not base64 tensor blobs) — needed for JS to parse.
        n = sum(1 for k in ctx.saved if k.startswith("p_"))
        ctx.saved[f"p_{n}"] = projections.cpu().tolist()
        return None

    hook = Hook(fn=project_emotions, layer_indices=[layer])
    return json.dumps([hook.model_dump()])


def generate_html(
    emotion_vectors: dict[str, torch.Tensor],
    layer: int,
    base_url: str,
    model: str,
    prompt: str,
    output_path: str,
) -> None:
    """Generate interactive HTML with live chat and sparkline visualization."""
    import json
    from pathlib import Path

    emotions = list(emotion_vectors.keys())
    hook_json = _build_serialized_hook(emotion_vectors, layer)

    template_path = Path(__file__).parent / "emotion_tracker.html"
    template = template_path.read_text()

    replacements = {
        "$BASE_URL": json.dumps(base_url),
        "$MODEL": json.dumps(model),
        "$EMOTIONS": json.dumps(emotions),
        "$LAYER": json.dumps(layer),
        "$HOOK_JSON": hook_json,
        "$STEERING_VECTORS": json.dumps(
            _build_serialized_steering(emotion_vectors, layer)
        ),
        "$INITIAL_PROMPT": json.dumps(prompt),
    }
    html = template
    for key, value in replacements.items():
        html = html.replace(key, value)

    with open(output_path, "w") as f:
        f.write(html)
    print(f"Interactive HTML saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Emotion tracker via vllm-lens")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--emotions",
        nargs="+",
        default=DEFAULT_EMOTIONS,
        help="Emotions to track",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--layer", type=int, default=PROBE_LAYER)
    parser.add_argument(
        "--stories-per-emotion",
        type=int,
        default=STORIES_PER_EMOTION,
    )
    parser.add_argument("--output", default="emotion_tracker")
    parser.add_argument(
        "--html-base-url",
        default="",
        help="Base URL for API calls in the HTML (empty = relative/same origin)",
    )
    parser.add_argument(
        "--load",
        default=None,
        help="Load saved vectors from .pt file (skip extraction)",
    )
    parser.add_argument(
        "--html-only",
        action="store_true",
        help="Regenerate HTML from saved .pt file (implies --load)",
    )
    args = parser.parse_args()

    client = VLLMLensClient(args.base_url)

    load_path = args.load
    if args.html_only and not load_path:
        load_path = f"{args.output}.pt"

    if load_path:
        print(f"Loading saved data from {load_path}...")
        saved = torch.load(load_path, weights_only=False)
        emotion_vectors = saved["emotion_vectors"]
        print(f"  Loaded {len(emotion_vectors)} emotion vectors")
    else:
        print(f"Emotions: {args.emotions}")
        print(f"Layer: {args.layer}\n")

        emotion_stories, neutral_stories = load_stories(
            args.emotions, args.stories_per_emotion
        )

        print("\nExtracting emotion activations...")
        emotion_acts = {}
        for emotion, stories in emotion_stories.items():
            print(f"  {emotion}:")
            emotion_acts[emotion] = extract_mean_activations(
                client, stories, args.layer, label=emotion
            )

        print("  neutral:")
        neutral_acts = extract_mean_activations(
            client, neutral_stories, args.layer, label="neutral"
        )

        print("\nComputing emotion vectors...")
        emotion_vectors = compute_emotion_vectors(emotion_acts, neutral_acts)
        for e, v in emotion_vectors.items():
            print(f"  {e}: norm={v.norm():.2f}")

        torch.save(
            {"emotion_vectors": emotion_vectors},
            f"{args.output}.pt",
        )
        print(f"Vectors saved to {args.output}.pt")

    print("\nGenerating HTML...")
    generate_html(
        emotion_vectors,
        args.layer,
        args.html_base_url,
        client.model,
        args.prompt,
        f"{args.output}.html",
    )


if __name__ == "__main__":
    main()
