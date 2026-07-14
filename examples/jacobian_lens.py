"""Jacobian lens / J-space: read out what a model is "disposed to say".

Like the logit lens, but first transports each layer's residual into the
final-layer basis with the average input-output Jacobian
``J_l = E[dh_final/dh_l]`` before the norm + unembedding::

    jacobian_lens_l(h) = unembed( J_l @ h )

Concepts the model will emit surface in the mid/late layers, before the output
(e.g. "...Eiffel Tower...city of" reads "Paris" well before the final token).

Reference: Anthropic, "Verbalizable Representations Form a Global Workspace in
Language Models" (https://transformer-circuits.pub/2026/workspace); reference
implementation: https://github.com/anthropics/jacobian-lens (Apache-2.0), from
which the estimator below is adapted.

Two steps — fit once, then run:

    # (1) compute J_l for a model in PyTorch (needs the backward pass; run under
    #     HF, once per model — the lens is prompt-independent and cached):
    python examples/jacobian_lens.py fit \\
        --model Qwen/Qwen3-0.6B --out qwen3-0.6b-lens.pt

    # (2) run it live in vLLM (forward-only; the only addition over logit_lens
    #     is J_l @ h in the hook). Needs a running vllm-lens server with
    #     VLLM_USE_V2_MODEL_RUNNER=0:
    python examples/jacobian_lens.py run \\
        --base-url http://localhost:8000 --lens qwen3-0.6b-lens.pt \\
        --prompt "The Eiffel Tower is located in the city of"

You can also `run --lens` a pre-fitted lens from the Hub (e.g. Neuronpedia).
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
# Adapted from github.com/anthropics/jacobian-lens (Apache-2.0). Estimator: for
# each output dim, inject a one-hot cotangent at every valid target position and
# backprop; the gradient at source position p is sum_{p'>=p} dh_final[p']/dh_l[p],
# meaned over source positions p, averaged over prompts.

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


def fit_lens(
    model_name: str,
    *,
    n_prompts: int = 64,
    dim_batch: int = 32,
    max_seq_len: int = 128,
) -> tuple[dict[int, torch.Tensor], int]:
    """Compute the average-Jacobian lens for every layer of an HF model."""
    import transformers
    from datasets import load_dataset

    hf = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16
    ).cuda()
    hf.eval()
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
                target = acts[target_layer]  # [dim_batch, seq, d]
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
                        rows = g[:n][:, valid, :].float().mean(dim=1)  # [n, d]
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


def run_jacobian_lens(
    client, prompt, jacobians, source_layers, *, top_k=5, use_jacobian=True
):
    from vllm_lens import Hook

    print(f"Prompt: {prompt!r}")

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
            topk = logits.topk(top_k, dim=-1)
        ctx.saved[f"ids_{ctx.layer_idx}"] = topk.indices.cpu()
        return None

    hook = Hook(fn=project_hook, layer_indices=source_layers)
    client.prefetch_params(["lm_head.weight"])
    output = client.generate(prompt, max_tokens=1, hooks=[hook], logprobs=5, echo=True)
    tokens = output.logprobs["tokens"][:-1] if output.logprobs else []
    assert output.hook_results is not None, "no hook results returned"
    saved = output.hook_results["0"]
    return {
        "tokens": tokens,
        "top_ids": {i: saved[f"ids_{i}"] for i in source_layers},
        "source_layers": source_layers,
    }


def print_grid(results, model_name, layer_step=4):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    tokens = results["tokens"]
    layers = [lyr for lyr in results["source_layers"] if lyr % layer_step == 0]
    header = "layer\\pos | " + " | ".join(f"{t!r:>12}" for t in tokens)
    print("\nEach cell = Jacobian-lens top-1 token at (layer, position).\n")
    print(header)
    print("-" * len(header))
    for layer in layers:
        top1 = results["top_ids"][layer][:, 0]
        cells = " | ".join(
            f"{tok.decode([int(top1[p])])!r:>12}" for p in range(len(tokens))
        )
        print(f"L{layer:>2}      | {cells}")


# ──────────────────────────────── CLI ────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="Jacobian lens / J-space via vllm-lens")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="compute J_l for a model in PyTorch (backward)")
    f.add_argument("--model", default="Qwen/Qwen3-0.6B")
    f.add_argument("--out", default="lens.pt")
    f.add_argument("--n-prompts", type=int, default=64)
    f.add_argument("--dim-batch", type=int, default=32)
    f.add_argument("--max-seq-len", type=int, default=128)

    r = sub.add_parser("run", help="read out a fitted lens live in vLLM")
    r.add_argument("--base-url", default="http://localhost:8000")
    r.add_argument("--lens", required=True)
    r.add_argument("--prompt", default="The Eiffel Tower is located in the city of")
    r.add_argument("--layer-step", type=int, default=4)
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

    from vllm_lens.client import VLLMLensClient

    jacobians, source_layers = load_lens(args.lens)
    client = VLLMLensClient(args.base_url)
    from _utils import get_num_layers

    source_layers = [lyr for lyr in source_layers if lyr < get_num_layers(client.model)]
    results = run_jacobian_lens(
        client, args.prompt, jacobians, source_layers, use_jacobian=not args.baseline
    )
    print_grid(results, client.model, layer_step=args.layer_step)


if __name__ == "__main__":
    main()
