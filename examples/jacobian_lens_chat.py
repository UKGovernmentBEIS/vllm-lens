"""Live Jacobian-lens (J-space) chat visualizer.

Generates a self-contained HTML page: chat with a served model and watch, for
each turn, the top-k tokens the model is *disposed to say* at every position —
the Jacobian-lens readout ``unembed(J_l @ h_l)`` — with a dropdown to switch the
readout layer instantly.

The readout is the same one ``jacobian_lens.py`` applies (transport by ``J_l`` ->
RMSNorm -> unembed -> top-k), run in a hook on the vLLM worker. Here the hook is
fully self-contained: it resolves the final-norm weight and decodes the top-k
ids to strings on the worker, so the browser renders them directly and no
tokenizer is needed in the page. The fitted lens ships inside the hook, so use a
modest set of layers.

Fit a lens first with ``jacobian_lens_fit.py``, serve the model, then generate:

    VLLM_USE_V2_MODEL_RUNNER=0 vllm serve Qwen/Qwen3-1.7B
    python examples/jacobian_lens_chat.py --lens lens.pt \\
        --base-url http://localhost:8000 --out jacobian_lens_chat
    # then open jacobian_lens_chat.html

The server must be running when you generate (to read the model name/config).
Single node (no pipeline parallelism) is assumed for the served demo.

Reference: Anthropic, "Verbalizable Representations Form a Global Workspace in
Language Models" (https://transformer-circuits.pub/2026/workspace).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vllm_lens.client import VLLMLensClient

from jacobian_lens import _norm_params, load_lens


def _build_serialized_hook(jacobians, layers, rms_eps, vocab_size, k, model_name):
    """JSON-serialize a self-contained hook that reads out top-k J-space tokens.

    Mirrors ``jacobian_lens.run_jlens``'s ``project_hook`` (transport by ``J_l``,
    manual RMSNorm, unembed, strip ParallelLMHead padding, top-k). The hook is
    defined here (``__main__``) so cloudpickle serializes it by value — it must
    therefore reference nothing from the ``jacobian_lens`` module (which the
    worker cannot import), so the final-norm name is resolved *inside* the hook
    and the top-k ids are decoded via a cached tokenizer. Per forward pass it
    stores ``t_{layer}_{pass}`` (top-k strings) and ``p_{layer}_{pass}`` (probs).
    """
    from vllm_lens import Hook

    def jspace_hook(ctx, h):
        import torch
        from transformers import AutoTokenizer

        cache = jspace_hook.__dict__
        tok = cache.get("tok")
        if tok is None:
            tok = AutoTokenizer.from_pretrained(model_name)
            cache["tok"] = tok

        # Resolve the final RMSNorm weight name on the worker (standard
        # get_language_model -> .model.norm / .norm tree). Cached across passes.
        norm_name = cache.get("norm_name")
        if norm_name is None:
            m, parts = ctx.model, []
            get_lm = getattr(m, "get_language_model", None)
            lm = None
            if callable(get_lm):
                try:
                    lm = get_lm()
                except (AttributeError, NotImplementedError):
                    lm = None
            if lm is not None and lm is not m:
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
            norm_name = ".".join(parts)
            cache["norm_name"] = norm_name

        lm_w = ctx.get_parameter("lm_head.weight")  # [vocab_padded, d]
        norm_w = ctx.get_parameter(norm_name)  # [d]
        with torch.no_grad():
            x = h.float()
            if ctx.layer_idx in jacobians:
                J = jacobians[ctx.layer_idx].to(x.device).float()
                x = x @ J.T
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + rms_eps)
            logits = (x * norm_w.float()) @ lm_w.float().T
            if vocab_size is not None:
                logits = logits[..., :vocab_size]  # drop ParallelLMHead padding
            probs, idx = logits.softmax(-1).topk(k, dim=-1)  # [seq, k]

        rows = idx.cpu().tolist()
        toks = [[tok.decode([i]) for i in row] for row in rows]
        n = sum(1 for key in ctx.saved if key.startswith(f"t_{ctx.layer_idx}_"))
        ctx.saved[f"t_{ctx.layer_idx}_{n}"] = toks
        ctx.saved[f"p_{ctx.layer_idx}_{n}"] = probs.float().cpu().tolist()
        return None

    hook = Hook(fn=jspace_hook, layer_indices=list(layers))
    return json.dumps([hook.model_dump()])


def generate_html(jacobians, layers, k, base_url, model, prompt, out, rms_eps, vocab_size):
    """Render the self-contained visualizer HTML from the template."""
    hook_json = _build_serialized_hook(jacobians, layers, rms_eps, vocab_size, k, model)
    template = (Path(__file__).parent / "jacobian_lens_chat.html").read_text()
    replacements = {
        "$BASE_URL": json.dumps(base_url),
        "$MODEL": json.dumps(model),
        "$LAYERS": json.dumps(list(layers)),
        "$TOPK": json.dumps(k),
        "$INITIAL_PROMPT": json.dumps(prompt),
        "$HOOK_JSON": hook_json,  # already a JSON array; inserted raw
    }
    html = template
    for key, value in replacements.items():
        html = html.replace(key, value)
    Path(out).write_text(html)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description="Live J-space chat visualizer")
    ap.add_argument("--lens", required=True, help="fitted lens .pt (jacobian_lens_fit.py)")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument(
        "--layers",
        default=None,
        help="comma-separated layers to capture (default: all fitted). Ships "
        "inside the hook, so keep it modest on large models.",
    )
    ap.add_argument("--k", type=int, default=6, help="top-k tokens per position.")
    ap.add_argument("--prompt", default="Tell me a short story about a lighthouse.")
    ap.add_argument(
        "--out", default="jacobian_lens_chat", help="output basename (.html appended)."
    )
    ap.add_argument(
        "--html-base-url",
        default="",
        help="base URL the HTML uses for API calls (empty = same origin as the page).",
    )
    args = ap.parse_args()

    jacobians, _fitted = load_lens(args.lens)
    if args.layers:
        want = [int(x) for x in args.layers.split(",")]
        jacobians = {lyr: jacobians[lyr] for lyr in want if lyr in jacobians}
    layers = sorted(jacobians)
    if not layers:
        raise SystemExit("no layers to capture (check --lens / --layers)")

    client = VLLMLensClient(args.base_url)
    rms_eps, vocab_size = _norm_params(client.model)

    # fp16 keeps the lens embedded in the hook smaller (cast back to float in-hook).
    jacobians = {lyr: v.half() for lyr, v in jacobians.items()}

    generate_html(
        jacobians,
        layers,
        args.k,
        args.html_base_url,
        client.model,
        args.prompt,
        f"{args.out}.html",
        rms_eps,
        vocab_size,
    )


if __name__ == "__main__":
    main()
