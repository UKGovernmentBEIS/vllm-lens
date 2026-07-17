"""Live, interactive Jacobian-lens (J-space) visualizer.

A chat page: type a message and watch — live, as you type — the top-k tokens the
model is *disposed to say* at each position (the J-lens readout
``unembed(J_l @ h)``), with a dropdown to switch the readout layer and a box to
set how many top tokens to show. "Create response" generates the assistant turn
(proper chat template) and the readout follows the reply.

Rather than send the lens on every message, this **registers** the readout hook
on the server once (the fitted lens is uploaded a single time), then the page
fires a small request as you type / on each turn and pulls the readout from
``/v1/hooks/collect`` (correlated by request id). Re-run this generator if the
server restarts (it re-registers the hook and rewrites the page).

The readout is the same projection ``jacobian_lens.py`` applies (transport by
``J_l`` -> RMSNorm -> unembed -> top-k). The hook is self-contained: it resolves
the final-norm name and decodes the top-k ids to strings on the worker, so the
page renders strings directly.

    VLLM_USE_V2_MODEL_RUNNER=0 vllm serve Qwen/Qwen3-1.7B --port 8000
    python examples/jacobian_lens_chat.py --lens lens.pt --base-url http://localhost:8000
    # open jacobian_lens_chat.html (set --html-base-url to your forwarded port)

Single node assumed (get_parameter gathers locally; no prefetch).

Reference: Anthropic, "Verbalizable Representations Form a Global Workspace in
Language Models" (https://transformer-circuits.pub/2026/workspace).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vllm_lens import Hook
from vllm_lens.client import VLLMLensClient

from jacobian_lens import _norm_params, load_lens


def _build_hook(jacobians, layers, rms_eps, vocab_size, k, model_name) -> Hook:
    """A self-contained readout hook (registered on the server, run per request).

    Cloudpickle serializes it by value (defined in ``__main__``), so it must not
    reference the ``jacobian_lens`` module: it resolves the final-norm name and
    decodes the top-k ids to strings on the worker itself. Per forward pass it
    stores ``t_{layer}_{pass}`` (top-k strings) and ``p_{layer}_{pass}`` (probs).
    """

    def jspace_hook(ctx, h):
        import torch
        from transformers import AutoTokenizer

        cache = jspace_hook.__dict__
        tok = cache.get("tok")
        if tok is None:
            tok = AutoTokenizer.from_pretrained(model_name)
            cache["tok"] = tok

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

        lm_w = ctx.get_parameter("lm_head.weight")
        norm_w = ctx.get_parameter(norm_name)
        with torch.no_grad():
            x = h.float()
            if ctx.layer_idx in jacobians:
                J = jacobians[ctx.layer_idx].to(x.device).float()
                x = x @ J.T
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + rms_eps)
            logits = (x * norm_w.float()) @ lm_w.float().T
            if vocab_size is not None:
                logits = logits[..., :vocab_size]
            probs, idx = logits.softmax(-1).topk(k, dim=-1)

        rows = idx.cpu().tolist()
        toks = [[tok.decode([i]) for i in row] for row in rows]
        n = sum(1 for key in ctx.saved if key.startswith(f"t_{ctx.layer_idx}_"))
        ctx.saved[f"t_{ctx.layer_idx}_{n}"] = toks
        ctx.saved[f"p_{ctx.layer_idx}_{n}"] = probs.float().cpu().tolist()
        return None

    return Hook(fn=jspace_hook, layer_indices=list(layers))


def _write_html(layers, k, html_base_url, model, prompt, out):
    import json

    template = (Path(__file__).parent / "jacobian_lens_chat.html").read_text()
    replacements = {
        "$BASE_URL": json.dumps(html_base_url),
        "$MODEL": json.dumps(model),
        "$LAYERS": json.dumps(list(layers)),
        "$TOPK": json.dumps(k),
        "$INITIAL_PROMPT": json.dumps(prompt),
    }
    html = template
    for key, value in replacements.items():
        html = html.replace(key, value)
    Path(out).write_text(html)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description="Interactive J-space visualizer")
    ap.add_argument(
        "--lens", required=True, help="fitted lens .pt (jacobian_lens_fit.py)"
    )
    ap.add_argument(
        "--base-url", default="http://localhost:8000", help="server URL for setup."
    )
    ap.add_argument(
        "--layers",
        default=None,
        help="comma-separated layers to capture (default: all fitted).",
    )
    ap.add_argument(
        "--k",
        type=int,
        default=25,
        help="top-k tokens captured per position; the page's K control displays "
        "up to this many (registered once, so K changes are instant/client-side).",
    )
    ap.add_argument("--prompt", default="Tell me about the Eiffel Tower.")
    ap.add_argument(
        "--out", default="jacobian_lens_chat", help="output basename (.html appended)."
    )
    ap.add_argument(
        "--html-base-url",
        default=None,
        help="URL the PAGE calls at runtime (default: same as --base-url). Set to "
        "your forwarded port, e.g. http://localhost:8123, or '' for same origin.",
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
    jacobians = {lyr: v.half() for lyr, v in jacobians.items()}

    # Register the readout hook on the server once (uploads the lens a single
    # time). It then applies to every request; the page reads results back via
    # /v1/hooks/collect. clear with client.clear_hooks() when done.
    hook = _build_hook(jacobians, layers, rms_eps, vocab_size, args.k, client.model)
    print(
        f"registering J-lens hook (layers {layers}, top-{args.k}) on {args.base_url} ..."
    )
    client.clear_hooks()  # replace any hook from a previous run (register appends)
    client.register_hooks([hook])
    print("registered.")

    html_base = args.base_url if args.html_base_url is None else args.html_base_url
    _write_html(
        layers, args.k, html_base, client.model, args.prompt, f"{args.out}.html"
    )
    print(
        "Open the HTML and type. Note: the hook is registered on the server and "
        "applies to ALL requests until cleared (client.clear_hooks() / POST "
        "/v1/hooks/clear); re-run this script if the server restarts."
    )


if __name__ == "__main__":
    main()
