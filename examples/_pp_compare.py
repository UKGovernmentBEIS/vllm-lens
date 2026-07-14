"""Parallel-equivalence harness for the Jacobian-lens run path.

Captures, per layer and position, the top-k token ids AND logit values from a
vllm-lens server, so two runs under different TP/PP/EP configs can be compared
rigorously: exact top-1 agreement, and — where top-1 flips — whether the flip
is a near-tie (fp non-determinism, expected) or a real disagreement (a bug).

Usage:
    python examples/_pp_compare.py save --base-url http://localhost:8000 \
        --lens qwen3-0.6b-lens.pt --tag tp1 --out /tmp/jl_tp1.pt
    python examples/_pp_compare.py cmp /tmp/jl_tp1.pt /tmp/jl_tp2.pt
"""

from __future__ import annotations

import argparse
import sys

import torch

sys.path.insert(0, "examples")
import cloudpickle  # noqa: E402
import jacobian_lens  # noqa: E402
from jacobian_lens import load_lens  # noqa: E402

# The hook closure references module-level helpers in this module; force
# cloudpickle to serialize them by value so the server needn't import it.
cloudpickle.register_pickle_by_value(jacobian_lens)


def save(args):
    from vllm_lens import Hook
    from vllm_lens.client import VLLMLensClient
    from transformers import AutoConfig
    from _utils import get_num_layers

    jacobians, source_layers = load_lens(args.lens)
    client = VLLMLensClient(args.base_url)
    n = get_num_layers(client.model)
    source_layers = [lyr for lyr in source_layers if lyr < n]
    top_k = args.top_k
    use_jacobian = not args.baseline
    norm_weight = "model.norm.weight"
    cfg = AutoConfig.from_pretrained(client.model)
    cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    rms_eps = getattr(cfg, "rms_norm_eps", 1e-6)
    vocab_size = getattr(cfg, "vocab_size", None)

    def project_hook(ctx, h):
        lm_w = ctx.get_parameter("lm_head.weight")
        norm_w = ctx.get_parameter(norm_weight)
        with torch.no_grad():
            x = h.float()
            if use_jacobian and ctx.layer_idx in jacobians:
                J = jacobians[ctx.layer_idx].to(x.device).float()
                x = x @ J.T
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + rms_eps)
            normed = x * norm_w.float()
            logits = normed @ lm_w.float().T
            if vocab_size is not None:
                logits = logits[..., :vocab_size]
            topk = logits.topk(top_k, dim=-1)
        ctx.saved[f"ids_{ctx.layer_idx}"] = topk.indices.cpu()
        ctx.saved[f"vals_{ctx.layer_idx}"] = topk.values.float().cpu()
        return None

    hook = Hook(fn=project_hook, layer_indices=source_layers)
    client.prefetch_params(["lm_head.weight", norm_weight])
    output = client.generate(args.prompt, max_tokens=1, hooks=[hook],
                             logprobs=5, echo=True)
    tokens = output.logprobs["tokens"][:-1] if output.logprobs else []
    saved = output.hook_results["0"]
    blob = {
        "tag": args.tag,
        "tokens": tokens,
        "source_layers": source_layers,
        "top_ids": {i: saved[f"ids_{i}"] for i in source_layers},
        "top_vals": {i: saved[f"vals_{i}"] for i in source_layers},
        "baseline": args.baseline,
        "model": client.model,
    }
    torch.save(blob, args.out)
    print(f"saved {args.tag} -> {args.out}  "
          f"({len(source_layers)} layers, {len(tokens)} tokens, "
          f"baseline={args.baseline})")


def cmp(args):
    a = torch.load(args.a, weights_only=False)
    b = torch.load(args.b, weights_only=False)
    print(f"A={a['tag']}  B={b['tag']}  model={a['model']}")
    assert a["source_layers"] == b["source_layers"], "layer sets differ"
    assert a["tokens"] == b["tokens"], f"tokens differ:\n{a['tokens']}\n{b['tokens']}"
    layers = a["source_layers"]
    total = top1_bad = 0
    max_logit_drift = 0.0
    flips = []  # (layer, pos, gapA, gapB) at top-1 disagreements
    for lyr in layers:
        ia, ib = a["top_ids"][lyr], b["top_ids"][lyr]
        va, vb = a["top_vals"][lyr], b["top_vals"][lyr]
        total += ia.shape[0]
        # logit drift on the shared rank-0 value
        max_logit_drift = max(max_logit_drift, float((va[:, 0] - vb[:, 0]).abs().max()))
        flip_mask = ia[:, 0] != ib[:, 0]
        for p in flip_mask.nonzero(as_tuple=True)[0].tolist():
            top1_bad += 1
            gapA = float(va[p, 0] - va[p, 1])  # top1-top2 gap in A
            gapB = float(vb[p, 0] - vb[p, 1])
            flips.append((lyr, p, gapA, gapB))
    print(f"\ntop-1 agreement: {total - top1_bad}/{total} "
          f"({100 * (total - top1_bad) / total:.1f}%)")
    print(f"max |logit(top1_A) - logit(top1_B)| drift: {max_logit_drift:.4f}")
    # A flip is explainable by fp non-determinism if the (smaller) top1-top2 gap
    # is below the observed cross-config logit drift: noise of that magnitude can
    # reorder the top-2. A flip whose gap EXCEEDS the drift is a real
    # disagreement and would signal a readout bug.
    thresh = max(max_logit_drift, 0.05)
    unexplained = [(lyr, p, gA, gB) for lyr, p, gA, gB in flips
                   if min(gA, gB) >= thresh]
    if flips:
        print(f"\ntop-1 flips (explainable if min(gap) < drift={thresh:.3f}):")
        for lyr, p, gA, gB in sorted(flips, key=lambda x: min(x[2], x[3]))[:20]:
            tag = "REAL-DISAGREEMENT" if min(gA, gB) >= thresh else "noise"
            print(f"  L{lyr:>2} pos{p:>2}: gapA={gA:.4f} gapB={gB:.4f}  {tag}")
        print(f"\n{len(flips) - len(unexplained)}/{len(flips)} flips explained by "
              f"fp noise; {len(unexplained)} unexplained")
    verdict = ("IDENTICAL top-1" if top1_bad == 0 else
               "EQUIVALENT (all flips within fp-noise drift)"
               if not unexplained else
               "DIVERGENT (flips exceed drift => real disagreement)")
    print(f"\nVERDICT: {verdict}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("save")
    s.add_argument("--base-url", default="http://localhost:8000")
    s.add_argument("--lens", required=True)
    s.add_argument("--prompt", default="The Eiffel Tower is located in the city of")
    s.add_argument("--tag", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--top-k", type=int, default=10)
    s.add_argument("--baseline", action="store_true")
    s.set_defaults(func=save)
    c = sub.add_parser("cmp")
    c.add_argument("a")
    c.add_argument("b")
    c.set_defaults(func=cmp)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
