"""Jacobian lens / J-space: read out what a model is "disposed to say".

Like the logit lens, but first transports each layer's residual into the
final-layer basis with a pre-fitted average input-output Jacobian
``J_l = E[dh_final/dh_l]`` before the norm + unembedding::

    jacobian_lens_l(h) = unembed( J_l @ h )

Concepts the model will emit surface in the mid/late layers, before the output.

Reference: Anthropic, "Verbalizable Representations Form a Global Workspace in
Language Models" (https://transformer-circuits.pub/2026/workspace); reference
implementation: https://github.com/anthropics/jacobian-lens (Apache-2.0), from
which the fitting estimator below is adapted.

Workflow — fit once (PyTorch), serve, then read out live in vLLM:

    # (1) compute J_l for a model (needs the backward pass; HF, once per model):
    python examples/jacobian_lens.py fit --model Qwen/Qwen3-1.7B --out lens.pt

    # (2) start a vllm-lens server (V1 runner so hooks work):
    VLLM_USE_V2_MODEL_RUNNER=0 vllm serve Qwen/Qwen3-1.7B

    # (3) read the lens out live — prints the concept grid, and with --grid-out
    #     saves the top-k "tokens in mind" figure (one subplot per layer):
    python examples/jacobian_lens.py run --lens lens.pt \\
        --prompt "The Eiffel Tower is located in the city of" \\
        --layers 14,20,24 --grid-out token_grid.png

`run` also accepts a pre-fitted lens from the Hub (e.g. Neuronpedia). The lens is
sent to the worker in the hook closure, so pass a modest set of `--layers` for
large models. `--grid-out` needs matplotlib.
"""

from __future__ import annotations

import argparse
import math

import torch

# ────────────────────────────── lens I/O ──────────────────────────────


def save_lens(jacobians: dict[int, torch.Tensor], d_model: int, out: str) -> None:
    torch.save(
        {
            "J": {lyr: J.to(torch.float16) for lyr, J in jacobians.items()},
            "source_layers": sorted(jacobians),
            "d_model": d_model,
        },
        out,
    )


def load_lens(path: str) -> tuple[dict[int, torch.Tensor], list[int]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if "J" not in ckpt:
        raise ValueError(f"{path} is not a Jacobian-lens file (keys {sorted(ckpt)!r})")
    jacobians = {int(k): v.float() for k, v in ckpt["J"].items()}
    return jacobians, sorted(jacobians)


# ─────────────────── (1) fit J_l in PyTorch (backward) ───────────────────
# Adapted from github.com/anthropics/jacobian-lens (Apache-2.0). For each output
# dim, inject a one-hot cotangent at every valid target position and backprop;
# the gradient at source position p is sum_{p'>=p} dh_final[p']/dh_l[p], meaned
# over source positions p, averaged over prompts.

SKIP_FIRST = 16  # early positions are attention sinks with atypical statistics


def _hf_decoder(hf):
    """Return (text_module, layers, n_layers, d_model) across common layouts."""
    for path in ("model", "model.language_model", "language_model"):
        mod = hf
        try:
            for part in path.split("."):
                mod = getattr(mod, part)
        except AttributeError:
            continue
        if hasattr(mod, "layers"):
            cfg = hf.config.get_text_config()
            return mod, mod.layers, cfg.num_hidden_layers, cfg.hidden_size
    raise ValueError(f"could not locate decoder layers on {type(hf).__name__}")


def fit_lens(model_name, *, n_prompts=64, dim_batch=32, max_seq_len=128):
    """Compute the average-Jacobian lens for every layer of an HF model."""
    import transformers
    from datasets import load_dataset

    hf = (
        transformers.AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16
        )
        .cuda()
        .eval()
    )
    for p in hf.parameters():
        p.requires_grad_(False)
    tok = transformers.AutoTokenizer.from_pretrained(model_name)
    text_module, layers, n_layers, d_model = _hf_decoder(hf)
    source_layers = list(range(n_layers - 1))
    target_layer = n_layers - 1

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    corpus = [r for r in ds["text"] if len(r) > 500 and not r.strip().startswith("=")]
    corpus = corpus[:n_prompts]

    jac_sum = {lyr: torch.zeros(d_model, d_model) for lyr in source_layers}
    n_done = 0
    for pi, prompt in enumerate(corpus):
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=max_seq_len)
        ids = ids.input_ids.cuda()
        seq_len = ids.shape[1]
        if seq_len <= SKIP_FIRST + 1:
            continue
        valid = torch.arange(SKIP_FIRST, seq_len - 1, device=ids.device)
        acts: dict[int, torch.Tensor] = {}
        handles = []
        min_src = min(source_layers)

        def make_hook(idx):
            def hook(_m, _in, out):
                t = out if torch.is_tensor(out) else out[0]
                if idx == min_src:
                    t.requires_grad_(True)  # root the graph at the earliest source
                acts[idx] = t

            return hook

        for idx in {*source_layers, target_layer}:
            handles.append(layers[idx].register_forward_hook(make_hook(idx)))
        try:
            with torch.enable_grad():
                text_module(input_ids=ids.expand(dim_batch, -1), use_cache=False)
                target = acts[target_layer]
                srcs = [acts[lyr] for lyr in source_layers]
                cot = torch.zeros_like(target)
                b = torch.arange(dim_batch, device=target.device)
                n_passes = math.ceil(d_model / dim_batch)
                for pass_i, start in enumerate(range(0, d_model, dim_batch)):
                    n = min(dim_batch, d_model - start)
                    cot.zero_()
                    cot[b[:n, None], valid[None, :], start + b[:n, None]] = 1.0
                    grads = torch.autograd.grad(
                        target,
                        srcs,
                        grad_outputs=cot,
                        retain_graph=(pass_i < n_passes - 1),
                    )
                    for lyr, g in zip(source_layers, grads):
                        rows = g[:n][:, valid, :].float().mean(dim=1)
                        jac_sum[lyr][start : start + n] += rows.cpu()
                    del grads
        finally:
            for h in handles:
                h.remove()
        n_done += 1
        print(f"  fit prompt {pi + 1}/{len(corpus)} (seq_len={seq_len})")

    if n_done == 0:
        raise ValueError("no prompts were long enough to fit on")
    jacobians = {lyr: jac_sum[lyr] / n_done for lyr in source_layers}
    print(f"fit done over {n_done} prompts")
    return jacobians, d_model


# ─────────────────── (2) run J_l live in vLLM (forward) ───────────────────


def _find_norm(model):
    """Final layer norm of a vLLM model (Qwen/Llama/Gemma layouts). Defined here
    (not imported from _utils) so it pickles by value into the worker-side hook."""
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    lm = getattr(model, "language_model", None)
    if lm is not None and hasattr(getattr(lm, "model", None), "norm"):
        return lm.model.norm
    return None


def run_jlens(client, prompt, jacobians, layers, *, k=6, use_jacobian=True):
    """Read out the top-k J-lens tokens at each (layer, position) via a hook on
    the live vLLM worker. Returns (prompt_tokens, {layer: (ids[seq,k], probs)})."""
    from vllm_lens import Hook

    def project_hook(ctx, h):
        weight = ctx.get_parameter("lm_head.weight")
        norm = _find_norm(ctx.model)
        with torch.no_grad():
            transported = h.float()
            if use_jacobian and ctx.layer_idx in jacobians:
                J = jacobians[ctx.layer_idx].to(h.device).float()
                transported = transported @ J.T
            normed = norm(transported) if norm is not None else transported
            logits = normed.float() @ weight.float().T
            probs, idx = logits.softmax(-1).topk(k, dim=-1)
        ctx.saved[f"ids_{ctx.layer_idx}"] = idx.cpu()
        ctx.saved[f"probs_{ctx.layer_idx}"] = probs.float().cpu()
        return None

    hook = Hook(fn=project_hook, layer_indices=layers)
    client.prefetch_params(["lm_head.weight"])
    output = client.generate(prompt, max_tokens=1, hooks=[hook], logprobs=1, echo=True)
    tokens = output.logprobs["tokens"][:-1] if output.logprobs else []
    assert output.hook_results is not None, "no hook results returned"
    saved = output.hook_results["0"]
    results = {lyr: (saved[f"ids_{lyr}"], saved[f"probs_{lyr}"]) for lyr in layers}
    return tokens, results


def print_grid(tokens, results, layers, tok, layer_step=1):
    """Text: top-1 J-lens token at each (layer, position)."""
    shown = [lyr for i, lyr in enumerate(layers) if i % layer_step == 0]
    header = "layer\\pos | " + " | ".join(f"{t!r:>12}" for t in tokens)
    print("\nTop-1 Jacobian-lens token per (layer, position):\n")
    print(header)
    print("-" * len(header))
    for lyr in shown:
        ids = results[lyr][0][:, 0]
        cells = " | ".join(
            f"{tok.decode([int(ids[p])])!r:>12}" for p in range(len(tokens))
        )
        print(f"L{lyr:>2}      | {cells}")


def save_token_grid(tokens, results, layers, tok, out, k=6):
    """Figure: top-k J-lens tokens per position, one subplot per layer."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    ink, muted = "#0b0b0b", "#52514e"
    blues = LinearSegmentedColormap.from_list("bb", ["#fcfcfb", "#2a78d6", "#0e2747"])
    seq = len(tokens)
    fig, axes = plt.subplots(
        len(layers),
        1,
        figsize=(max(7, seq * 1.15), (0.5 * k + 0.7) * len(layers) + 1.2),
        squeeze=False,
    )
    for ax, lyr in zip(axes[:, 0], layers):
        ids, probs = results[lyr]
        for c in range(seq):
            for r in range(k):
                word = tok.decode([int(ids[c, r])]).strip() or "·"
                prob = float(probs[c, r])
                y = k - 1 - r
                ax.add_patch(
                    plt.Rectangle(
                        (c, y),
                        1,
                        1,
                        facecolor=blues(min(prob * 2.0, 1.0)),
                        edgecolor="#fcfcfb",
                        linewidth=1.5,
                    )
                )
                surfaced = word.lower() != tokens[c].strip().lower() and any(
                    ch.isalpha() for ch in word
                )
                ax.text(
                    c + 0.5,
                    y + 0.5,
                    word[:10],
                    ha="center",
                    va="center",
                    fontsize=8 if len(word) > 6 else 9,
                    color="#ffffff" if prob > 0.3 else ink,
                    fontweight="bold" if (surfaced and r == 0) else "normal",
                )
        ax.set_xlim(0, seq)
        ax.set_ylim(0, k)
        ax.set_yticks([k - 0.5 - i for i in range(k)])
        ax.set_yticklabels([f"top-{i + 1}" for i in range(k)], fontsize=8, color=muted)
        ax.set_ylabel(f"layer {lyr}", fontsize=10, color=ink)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    bottom = axes[-1, 0]
    bottom.set_xticks([i + 0.5 for i in range(seq)])
    bottom.set_xticklabels(
        [t.strip() or "·" for t in tokens],
        rotation=40,
        ha="right",
        fontsize=10,
        color=ink,
    )
    bottom.set_xlabel("prompt position", color=muted, fontsize=9)
    for ax in axes[:-1, 0]:
        ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    print(f"wrote {out}")


def _default_layers(source_layers, n_layers):
    """A spread across depth (keeps the shipped lens small on large models)."""
    want = [int(0.55 * n_layers), int(0.72 * n_layers), int(0.88 * n_layers)]
    return sorted({min(source_layers, key=lambda f: abs(f - w)) for w in want})


def main():
    ap = argparse.ArgumentParser(description="Jacobian lens / J-space via vllm-lens")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="compute J_l for a model in PyTorch (backward)")
    f.add_argument("--model", default="Qwen/Qwen3-1.7B")
    f.add_argument("--out", default="lens.pt")
    f.add_argument("--n-prompts", type=int, default=64)
    f.add_argument("--dim-batch", type=int, default=32)
    f.add_argument("--max-seq-len", type=int, default=128)

    r = sub.add_parser("run", help="read a fitted lens out live in vLLM")
    r.add_argument("--base-url", default="http://localhost:8000")
    r.add_argument("--lens", required=True)
    r.add_argument("--prompt", default="The Eiffel Tower is located in the city of")
    r.add_argument(
        "--layers",
        default=None,
        help="comma-separated layers to read out; default spreads across depth",
    )
    r.add_argument("--k", type=int, default=6)
    r.add_argument(
        "--grid-out",
        default=None,
        help="save the top-k token grid PNG here (needs matplotlib)",
    )
    r.add_argument(
        "--baseline", action="store_true", help="skip J_l (logit-lens baseline)"
    )

    args = ap.parse_args()

    if args.cmd == "fit":
        jacobians, d_model = fit_lens(
            args.model,
            n_prompts=args.n_prompts,
            dim_batch=args.dim_batch,
            max_seq_len=args.max_seq_len,
        )
        save_lens(jacobians, d_model, args.out)
        print(f"saved lens -> {args.out} ({len(jacobians)} layers, d_model={d_model})")
        return

    from transformers import AutoTokenizer

    from vllm_lens.client import VLLMLensClient

    jacobians, source_layers = load_lens(args.lens)
    client = VLLMLensClient(args.base_url)
    from _utils import get_num_layers

    n_layers = get_num_layers(client.model)
    source_layers = [lyr for lyr in source_layers if lyr < n_layers]
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        layers = _default_layers(source_layers, n_layers)

    tok = AutoTokenizer.from_pretrained(client.model)
    tokens, results = run_jlens(
        client, args.prompt, jacobians, layers, k=args.k, use_jacobian=not args.baseline
    )
    print(f"Prompt: {args.prompt!r}")
    print_grid(tokens, results, layers, tok)
    if args.grid_out:
        save_token_grid(tokens, results, layers, tok, args.grid_out, k=args.k)


if __name__ == "__main__":
    main()
