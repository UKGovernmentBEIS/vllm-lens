"""Render the Jacobian-lens token grid — the tokens a model has "in mind".

Two modes:
  * default: top-1 J-lens token at every (layer, position)  ["token grid by layer"]
  * --layer L: top-k J-lens tokens per position at a single layer L  [parametrised]

Reproduces the figure in the README. Runs under HF (needs all-layer residuals +
the fitted lens); the readouts are identical to what vllm-lens captures during
vLLM inference (verified bit-for-bit). Requires: transformers, torch, matplotlib.

    python make_token_grid.py --model Qwen/Qwen3-1.7B --lens qwen3-1.7b-lens.pt \
        --prompt "The Eiffel Tower is located in the city of" --out token_grid_by_layer.png
    # single-layer top-k view:
    python make_token_grid.py --model ... --lens ... --layer 22 --k 6 --out topk.png
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
        transformers.AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        .cuda()
        .eval()
    )
    tok = transformers.AutoTokenizer.from_pretrained(model_name)
    ck = torch.load(lens_path, map_location="cpu", weights_only=True)
    J = {int(k): v.float().cuda() for k, v in ck["J"].items()}
    return hf, tok, J


@torch.no_grad()
def jlens(hf, tok, J, prompt, layer, k):
    """Top-k tokens + probs at `layer` for every position."""
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    prompt_toks = [tok.decode([t]) for t in ids[0]]
    hs = hf(ids, output_hidden_states=True).hidden_states
    h = hs[layer + 1][0].float()
    logits = hf.lm_head(hf.model.norm((h @ J[layer].T).to(hf.lm_head.weight.dtype)))
    p, idx = logits.softmax(-1).topk(k, dim=-1)  # [seq, k]
    words = [[tok.decode([int(i)]).strip() or "·" for i in row] for row in idx]
    return prompt_toks, words, p.tolist()


def _cell(ax, x, y, word, prob, prompt_tok, bold_top=False):
    ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor=BLUES(min(prob * 2.0, 1.0)),
                               edgecolor="#fcfcfb", linewidth=1.5))
    surfaced = word.lower() != prompt_tok.strip().lower() and any(c.isalpha() for c in word)
    ax.text(x + 0.5, y + 0.5, word[:10], ha="center", va="center",
            fontsize=8 if len(word) > 6 else 9,
            color="#ffffff" if prob > 0.3 else INK,
            fontweight="bold" if (surfaced and bold_top) else "normal")


def grid_by_layer(hf, tok, J, prompt, layer_step, out):
    layers = [l for l in sorted(J) if l % layer_step == 0]
    prompt_toks = None
    rows = {}
    for layer in layers:
        prompt_toks, words, probs = jlens(hf, tok, J, prompt, layer, k=1)
        rows[layer] = ([w[0] for w in words], [p[0] for p in probs])
    seq = len(prompt_toks)
    fig, ax = plt.subplots(figsize=(max(7, seq * 1.05), 0.42 * len(layers) + 1.4))
    for r, layer in enumerate(layers):
        words, probs = rows[layer]
        for c in range(seq):
            _cell(ax, c, r, words[c], probs[c], prompt_toks[c], bold_top=True)
    ax.set_xlim(0, seq); ax.set_ylim(0, len(layers))
    ax.set_xticks([i + 0.5 for i in range(seq)])
    ax.set_xticklabels([t.strip() or "·" for t in prompt_toks], rotation=40, ha="right", fontsize=9, color=INK)
    ax.set_yticks([i + 0.5 for i in range(len(layers))])
    ax.set_yticklabels([f"L{l}" for l in layers], fontsize=8, color=MUTED)
    ax.set_xlabel("prompt position", color=MUTED, fontsize=9)
    ax.set_title(f"Jacobian-lens top-1 token per (layer, position)\n{prompt!r}", fontsize=11, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    print("wrote", out)


def grid_topk(hf, tok, J, prompt, layer, k, out):
    prompt_toks, words, probs = jlens(hf, tok, J, prompt, layer, k)
    seq = len(prompt_toks)
    fig, ax = plt.subplots(figsize=(max(7, seq * 1.15), 0.5 * k + 1.6))
    for c in range(seq):
        for r in range(k):
            _cell(ax, c, k - 1 - r, words[c][r], probs[c][r], prompt_toks[c], bold_top=(r == 0))
    ax.set_xlim(0, seq); ax.set_ylim(0, k)
    ax.set_xticks([i + 0.5 for i in range(seq)])
    ax.set_xticklabels([t.strip() or "·" for t in prompt_toks], rotation=40, ha="right", fontsize=10, color=INK)
    ax.set_yticks([k - 0.5 - i for i in range(k)])
    ax.set_yticklabels([f"top-{i + 1}" for i in range(k)], fontsize=8, color=MUTED)
    ax.set_xlabel("prompt position", color=MUTED, fontsize=9)
    ax.set_title(f"Top-{k} Jacobian-lens tokens per position @ layer {layer}\n{prompt!r}", fontsize=11, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--lens", required=True)
    ap.add_argument("--prompt", default="The Eiffel Tower is located in the city of")
    ap.add_argument("--layer", type=int, default=None, help="single layer -> top-k per position")
    ap.add_argument("--layer-step", type=int, default=2, help="grid-by-layer: show every Nth layer")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--out", default="token_grid.png")
    args = ap.parse_args()

    hf, tok, J = load(args.model, args.lens)
    if args.layer is not None:
        grid_topk(hf, tok, J, args.prompt, args.layer, args.k, args.out)
    else:
        grid_by_layer(hf, tok, J, args.prompt, args.layer_step, args.out)


if __name__ == "__main__":
    main()
