"""Render the Jacobian-lens token grid — the tokens a model has "in mind".

For each prompt position, show the **top-k** Jacobian-lens tokens (rank 1 at top,
shaded by probability), i.e. the concepts the model is disposed toward but hasn't
emitted. Pass one or more layers with --layers to see how they evolve with depth
(one subplot per layer).

Reproduces the README figure. Runs under HF (needs all-layer residuals + the
fitted lens); the readouts are identical to what vllm-lens captures during vLLM
inference (verified bit-for-bit). Requires: transformers, torch, matplotlib.

    # top-k per position across a few layers (default):
    python make_token_grid.py --model Qwen/Qwen3-1.7B --lens qwen3-1.7b-lens.pt \
        --prompt "The Eiffel Tower is located in the city of" --out token_grid.png
    # a single layer:
    python make_token_grid.py --model ... --lens ... --layers 22 --k 8 --out one.png
"""

from __future__ import annotations

import argparse

import matplotlib
import torch
import transformers

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INK, MUTED = "#0b0b0b", "#52514e"
BLUES = LinearSegmentedColormap.from_list("bb", ["#fcfcfb", "#2a78d6", "#0e2747"])


def load(model_name, lens_path):
    hf = (
        transformers.AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32
        )
        .cuda()
        .eval()
    )
    tok = transformers.AutoTokenizer.from_pretrained(model_name)
    ck = torch.load(lens_path, map_location="cpu", weights_only=True)
    jac = {int(k): v.float().cuda() for k, v in ck["J"].items()}
    return hf, tok, jac


@torch.no_grad()
def all_hidden(hf, tok, prompt):
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    prompt_toks = [tok.decode([t]) for t in ids[0]]
    hs = hf(ids, output_hidden_states=True).hidden_states  # embed + per-block
    return prompt_toks, hs


@torch.no_grad()
def topk_at_layer(hf, tok, jac, hs, layer, k):
    """Top-k tokens + probs at `layer` for every position."""
    h = hs[layer + 1][0].float()
    logits = hf.lm_head(hf.model.norm((h @ jac[layer].T).to(hf.lm_head.weight.dtype)))
    probs, idx = logits.softmax(-1).topk(k, dim=-1)  # [seq, k]
    words = [[tok.decode([int(i)]).strip() or "·" for i in row] for row in idx]
    return words, probs.tolist()


def draw_layer(ax, prompt_toks, words, probs, layer, k):
    seq = len(prompt_toks)
    for c in range(seq):
        for r in range(k):
            word = words[c][r]
            prob = probs[c][r]
            y = k - 1 - r
            ax.add_patch(
                plt.Rectangle(
                    (c, y),
                    1,
                    1,
                    facecolor=BLUES(min(prob * 2.0, 1.0)),
                    edgecolor="#fcfcfb",
                    linewidth=1.5,
                )
            )
            surfaced = word.lower() != prompt_toks[c].strip().lower() and any(
                ch.isalpha() for ch in word
            )
            ax.text(
                c + 0.5,
                y + 0.5,
                word[:10],
                ha="center",
                va="center",
                fontsize=8 if len(word) > 6 else 9,
                color="#ffffff" if prob > 0.3 else INK,
                fontweight="bold" if (surfaced and r == 0) else "normal",
            )
    ax.set_xlim(0, seq)
    ax.set_ylim(0, k)
    ax.set_yticks([k - 0.5 - i for i in range(k)])
    ax.set_yticklabels([f"top-{i + 1}" for i in range(k)], fontsize=8, color=MUTED)
    ax.set_ylabel(f"layer {layer}", fontsize=10, color=INK)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--lens", required=True)
    ap.add_argument("--prompt", default="The Eiffel Tower is located in the city of")
    ap.add_argument(
        "--layers",
        default=None,
        help="comma-separated, e.g. '14,20,24'; default spreads across depth",
    )
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--out", default="token_grid.png")
    args = ap.parse_args()

    hf, tok, jac = load(args.model, args.lens)
    fitted = sorted(jac)
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        n = hf.config.num_hidden_layers
        want = [int(0.55 * n), int(0.72 * n), int(0.88 * n)]
        layers = [min(fitted, key=lambda f: abs(f - w)) for w in want]

    prompt_toks, hs = all_hidden(hf, tok, args.prompt)
    seq = len(prompt_toks)
    fig, axes = plt.subplots(
        len(layers),
        1,
        figsize=(max(7, seq * 1.15), (0.5 * args.k + 0.7) * len(layers) + 1.2),
        squeeze=False,
    )
    for ax, layer in zip(axes[:, 0], layers):
        words, probs = topk_at_layer(hf, tok, jac, hs, layer, args.k)
        draw_layer(ax, prompt_toks, words, probs, layer, args.k)
    bottom = axes[-1, 0]
    bottom.set_xticks([i + 0.5 for i in range(seq)])
    bottom.set_xticklabels(
        [t.strip() or "·" for t in prompt_toks],
        rotation=40,
        ha="right",
        fontsize=10,
        color=INK,
    )
    bottom.set_xlabel("prompt position", color=MUTED, fontsize=9)
    for ax in axes[:-1, 0]:
        ax.set_xticks([])
    fig.suptitle(
        f"Top-{args.k} Jacobian-lens tokens per position — {args.model.split('/')[-1]}\n{args.prompt!r}",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
