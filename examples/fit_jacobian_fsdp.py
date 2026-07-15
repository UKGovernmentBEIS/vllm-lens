"""FSDP Jacobian-lens fitter — shard the model across GPUs/nodes so J_l can be
fit for arbitrarily large models (the single-GPU `jacobian_lens.py fit` needs the
whole model + backward graph on one device).

Reuses the mt-finetuning scaffolding *pattern*: launched with `accelerate launch`
under an FSDP config (FULL_SHARD + transformer wrap + CPU-RAM-efficient loading),
but replaces the training loop with the Jacobian estimator — forward, inject
one-hot cotangents at the final decoder layer, `autograd.grad` back to the source
layers, accumulate E[dh_final/dh_l], no optimizer.

Under FSDP each rank holds a *param shard* but computes the *same* forward on the
same data (activations are replicated), so every rank arrives at the identical
Jacobian — no cross-rank reduction is needed; rank 0 saves. (Throughput could be
scaled later with a hybrid mesh that data-parallelises over prompts.)

Launch (single node, 8 GPUs):
    accelerate launch --config_file examples/accelerate_configs/fsdp.yaml \
        --num_processes 8 examples/fit_jacobian_fsdp.py \
        --model zai-org/GLM-4.5-Air --layers 25,33,40 --out glm45air-lens.pt

Two nodes (run on each; NODE_RANK=0 on the head):
    accelerate launch --config_file examples/accelerate_configs/fsdp.yaml \
        --num_machines 2 --machine_rank $NODE_RANK --main_process_ip $HEAD_IP \
        --num_processes 16 examples/fit_jacobian_fsdp.py --model ... --layers ...

The output .pt is byte-compatible with `jacobian_lens.py` / `_pp_compare.py`
(keys: J, source_layers, d_model).
"""

from __future__ import annotations

import argparse

import torch

SKIP_FIRST = 16  # early positions are attention sinks with atypical statistics


def _decoder(model):
    """(decoder_module, layers, n_layers, d_model) for an unwrapped HF causal LM."""
    cfg = model.config.get_text_config()
    dec = model.get_decoder() if hasattr(model, "get_decoder") else None
    if dec is not None and hasattr(dec, "layers"):
        return dec, dec.layers, cfg.num_hidden_layers, cfg.hidden_size
    for path in ("model", "model.language_model", "language_model"):
        mod = model
        try:
            for part in path.split("."):
                mod = getattr(mod, part)
        except AttributeError:
            continue
        if hasattr(mod, "layers"):
            return mod, mod.layers, cfg.num_hidden_layers, cfg.hidden_size
    raise ValueError(f"could not locate decoder layers on {type(model).__name__}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="lens.pt")
    ap.add_argument(
        "--layers",
        default=None,
        help="comma-separated source layers to fit (default: every layer). Fewer "
        "layers = far cheaper (one backward sweep per fitted layer set).",
    )
    ap.add_argument("--n-prompts", type=int, default=32)
    ap.add_argument(
        "--dim-batch",
        type=int,
        default=32,
        help="output dims done per forward+backward. Larger = fewer forwards "
        "(d_model/dim_batch per prompt) but more activation memory.",
    )
    ap.add_argument("--max-seq-len", type=int, default=128)
    args = ap.parse_args()

    from accelerate import Accelerator
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    accelerator = Accelerator()

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Resolve layer geometry on the unwrapped model BEFORE FSDP wrapping.
    _, _, n_layers, d_model = _decoder(model)
    target_layer = n_layers - 1
    if args.layers:
        source_layers = sorted(int(x) for x in args.layers.split(","))
    else:
        source_layers = list(range(n_layers - 1))
    assert all(0 <= s < target_layer for s in source_layers), (
        f"source layers must be in [0, {target_layer}); got {source_layers}"
    )
    min_src = min(source_layers)
    accelerator.print(
        f"model={args.model} n_layers={n_layers} d_model={d_model} "
        f"fitting {len(source_layers)} layers {source_layers} (target=L{target_layer})"
    )

    model = accelerator.prepare(model)
    model.eval()
    dec_layers = _decoder(accelerator.unwrap_model(model))[1]

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    corpus = [r for r in ds["text"] if len(r) > 500 and not r.strip().startswith("=")]
    corpus = corpus[: args.n_prompts]

    jac_sum = {lyr: torch.zeros(d_model, d_model) for lyr in source_layers}
    n_done = 0
    device = accelerator.device

    acts: dict[int, torch.Tensor] = {}

    def make_hook(idx):
        def hook(_m, _in, out):
            t = out if torch.is_tensor(out) else out[0]
            if idx == min_src:
                t.requires_grad_(True)  # root the graph at the earliest source
            acts[idx] = t

        return hook

    handles = [
        dec_layers[idx].register_forward_hook(make_hook(idx))
        for idx in {*source_layers, target_layer}
    ]

    try:
        for pi, prompt in enumerate(corpus):
            ids = tok(
                prompt, return_tensors="pt", truncation=True, max_length=args.max_seq_len
            ).input_ids.to(device)
            seq_len = ids.shape[1]
            if seq_len <= SKIP_FIRST + 1:
                continue
            valid = torch.arange(SKIP_FIRST, seq_len - 1, device=device)
            batched = ids.expand(args.dim_batch, -1)
            b = torch.arange(args.dim_batch, device=device)

            # Fresh forward + a single .backward() per cotangent block. FSDP1's
            # backward hooks don't support autograd.grad() (leaf-node error) nor
            # multiple backwards over one retained graph (params reshard after the
            # first), so we re-forward per block and use plain .backward(), reading
            # the source-activation grads via retain_grad().
            for start in range(0, d_model, args.dim_batch):
                n = min(args.dim_batch, d_model - start)
                acts.clear()
                with torch.enable_grad():
                    model(input_ids=batched)
                target = acts[target_layer]
                srcs = [acts[lyr] for lyr in source_layers]
                for s in srcs:
                    s.retain_grad()
                cot = torch.zeros_like(target)
                cot[b[:n, None], valid[None, :], start + b[:n, None]] = 1.0
                torch.autograd.backward(target, grad_tensors=cot)
                for lyr, s in zip(source_layers, srcs):
                    rows = s.grad[:n][:, valid, :].float().mean(dim=1)  # [n, d]
                    jac_sum[lyr][start : start + n] += rows.detach().cpu()
            n_done += 1
            accelerator.print(f"  fit prompt {pi + 1}/{len(corpus)} (seq_len={seq_len})")
    finally:
        for h in handles:
            h.remove()

    accelerator.wait_for_everyone()
    if n_done == 0:
        raise ValueError("no prompts were long enough to fit on")
    if accelerator.is_main_process:
        jacobians = {lyr: (jac_sum[lyr] / n_done).to(torch.float16) for lyr in source_layers}
        torch.save(
            {"J": jacobians, "source_layers": source_layers, "d_model": d_model},
            args.out,
        )
        accelerator.print(
            f"fit done over {n_done} prompts -> {args.out} "
            f"({len(jacobians)} layers, d_model={d_model})"
        )


if __name__ == "__main__":
    main()
