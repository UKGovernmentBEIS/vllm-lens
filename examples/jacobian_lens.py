"""Jacobian lens / J-space: read out what a model is "disposed to say".

Like the logit lens, but first transports each layer's residual into the
final-layer basis with a pre-fitted average input-output Jacobian
``J_l = E[dh_final/dh_l]`` before the norm + unembedding::

    jacobian_lens_l(h) = unembed( J_l @ h )

Concepts the model will emit surface in the mid/late layers, before the output.

Reference: Anthropic, "Verbalizable Representations Form a Global Workspace in
Language Models" (https://transformer-circuits.pub/2026/workspace); reference
implementation: https://github.com/anthropics/jacobian-lens (Apache-2.0).

This script is the READOUT + VISUALIZATION half of the flow, and runs in the
vllm-lens env. Fitting the lens is done ONCE by ``jacobian_lens_fit.py`` (in the
separate prime-rl fit env — see ``jacobian_lens_fit_env.sh``); it scales from
small dense models up to large MoE (e.g. GLM-4.5-Air) across GPUs/nodes and
writes a ``lens.pt`` that this script reads back.

Workflow — fit once, serve, then read out live in vLLM:

    # (1) fit J_l for a model (prime-rl fit env; single- or multi-node):
    #     see jacobian_lens_fit.py / jacobian_lens_fit_env.sh
    uv run --no-sync torchrun --nproc-per-node=8 \\
        examples/jacobian_lens_fit.py --model Qwen/Qwen3-1.7B --out lens.pt

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

import torch

# ────────────────────────────── lens I/O ──────────────────────────────


def load_lens(path: str) -> tuple[dict[int, torch.Tensor], list[int]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if "J" not in ckpt:
        raise ValueError(f"{path} is not a Jacobian-lens file (keys {sorted(ckpt)!r})")
    jacobians = {int(k): v.float() for k, v in ckpt["J"].items()}
    return jacobians, sorted(jacobians)


# ─────────────────── run J_l live in vLLM (forward) ───────────────────


def _norm_params(model_name):
    """(rms_norm_eps, vocab_size) from a served model's text config."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    return getattr(cfg, "rms_norm_eps", 1e-6), getattr(cfg, "vocab_size", None)


def _resolve_norm_name(model):
    """Dotted param name of the final RMSNorm weight, relative to the top-level
    vLLM ``*ForCausalLM``.

    Runs **on the worker** against the served model. Unwraps multimodal /
    composite wrappers via ``get_language_model()`` and descends the standard
    ``.model.norm`` (or bare ``.norm``) tree, so it covers nested/multimodal
    layouts without a per-model attribute map. ``PPMissingLayer`` placeholders
    keep the same submodule names on every pipeline stage, so this resolves to
    the same dotted name on all ranks even though the weight tensor itself lives
    only on the last PP stage.
    """
    parts: list[str] = []
    m = model
    get_lm = getattr(m, "get_language_model", None)
    if callable(get_lm):
        try:
            lm = get_lm()
        except (AttributeError, NotImplementedError):
            lm = None
        if lm is not None and lm is not m:
            # Locate the (possibly nested) attribute path holding the text LM.
            for name, mod in m.named_modules():
                if mod is lm:
                    parts.append(name)
                    m = lm
                    break
    inner = getattr(m, "model", None)
    if inner is not None and hasattr(inner, "norm"):
        parts += ["model", "norm", "weight"]
    elif hasattr(m, "norm"):
        parts += ["norm", "weight"]
    else:
        raise ValueError(f"could not locate final norm on {type(model).__name__}")
    return ".".join(parts)


def _probe_norm_name(client, prompt):
    """Resolve the served model's final-norm weight name via a one-shot probe.

    The HTTP client can't introspect the worker's module tree, and the PP-safe
    prefetch needs the dotted name up front — so we fire a tiny hook that walks
    ``ctx.model`` on the worker (:func:`_resolve_norm_name`) and returns the
    name. Works under TP/PP/EP since the module names are identical on every
    rank; the probe only reads structure, never a (possibly missing) weight.
    """
    from vllm_lens import Hook

    def probe(ctx, h):
        ctx.saved["norm_name"] = _resolve_norm_name(ctx.model)
        return None

    out = client.generate(
        prompt, max_tokens=1, hooks=[Hook(fn=probe, layer_indices=[0])]
    )
    assert out.hook_results is not None, "norm-name probe returned no hook results"
    return out.hook_results["0"]["norm_name"]


def run_jlens(
    client,
    prompt,
    jacobians,
    layers,
    *,
    k=6,
    use_jacobian=True,
    norm_weight=None,
):
    """Read out the top-k J-lens tokens at each (layer, position) via a hook on
    the live vLLM worker. Returns (prompt_tokens, {layer: (ids[seq,k], probs)}).

    Correct under tensor/pipeline/expert parallelism: the final norm and lm_head
    live only on the *last* PP stage (``PPMissingLayer`` elsewhere), so we
    prefetch **both** their weights to every rank (``prefetch_params``
    PP-broadcasts + TP-gathers) and apply the norm **manually** — a layer's hook
    fires on whichever stage owns it, where those modules can't be called. We
    also slice off the ``ParallelLMHead`` vocab padding. The manual norm is the
    standard RMSNorm (Qwen/Llama/GLM/DeepSeek); a variant norm (e.g. Gemma's
    ``1 + w`` gain, or a logit soft-cap) would need that variant here.

    ``norm_weight`` is the dotted name of the final-norm weight relative to the
    served ``*ForCausalLM``. When ``None`` it is auto-detected from the served
    model's structure (:func:`_probe_norm_name`), so no hardcoded default is
    baked in; pass an explicit name to override for exotic layouts.
    """
    from vllm_lens import Hook

    rms_eps, vocab_size = _norm_params(client.model)
    if norm_weight is None:
        norm_weight = _probe_norm_name(client, prompt)

    def project_hook(ctx, h):
        lm_w = ctx.get_parameter("lm_head.weight")  # [vocab_padded, d], TP-gathered
        norm_w = ctx.get_parameter(norm_weight)  # [d], prefetched / PP-broadcast
        with torch.no_grad():
            x = h.float()
            if use_jacobian and ctx.layer_idx in jacobians:
                J = jacobians[ctx.layer_idx].to(x.device).float()
                x = x @ J.T
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + rms_eps)
            normed = x * norm_w.float()
            logits = normed @ lm_w.float().T
            if vocab_size is not None:
                logits = logits[..., :vocab_size]  # drop ParallelLMHead padding
            probs, idx = logits.softmax(-1).topk(k, dim=-1)
        ctx.saved[f"ids_{ctx.layer_idx}"] = idx.cpu()
        ctx.saved[f"probs_{ctx.layer_idx}"] = probs.float().cpu()
        return None

    hook = Hook(fn=project_hook, layer_indices=layers)
    # Prefetch to every rank so hooks on any PP stage can unembed.
    client.prefetch_params(["lm_head.weight", norm_weight])
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
    ap = argparse.ArgumentParser(
        description="Jacobian lens / J-space readout + visualization via vllm-lens. "
        "Fit the lens first with jacobian_lens_fit.py (prime-rl fit env)."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

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
    r.add_argument(
        "--norm-weight",
        default=None,
        help="dotted name of the final-norm weight (default: auto-detect from "
        "the served model's structure)",
    )

    args = ap.parse_args()

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

    # The lens ships to the worker inside the hook closure, so only send the J_l
    # for the layers we actually read out (a full lens is d_model^2 per layer —
    # ~5.8 GB for GLM-5.2). --baseline needs no lens at all.
    if args.baseline:
        jacobians = {}
    else:
        jacobians = {lyr: jacobians[lyr] for lyr in layers if lyr in jacobians}

    tok = AutoTokenizer.from_pretrained(client.model)
    tokens, results = run_jlens(
        client,
        args.prompt,
        jacobians,
        layers,
        k=args.k,
        use_jacobian=not args.baseline,
        norm_weight=args.norm_weight,
    )
    print(f"Prompt: {args.prompt!r}")
    print_grid(tokens, results, layers, tok)
    if args.grid_out:
        save_token_grid(tokens, results, layers, tok, args.grid_out, k=args.k)


if __name__ == "__main__":
    main()
