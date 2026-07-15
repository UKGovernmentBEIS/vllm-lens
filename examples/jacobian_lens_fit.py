"""Jacobian-lens fitter for arbitrary-size models (FSDP2 + expert-parallel).

Fits the average Jacobian lens ``J_l = E[dh_final/dh_l]`` for a set of decoder
source layers l, where ``h_l`` is the residual-stream output of layer l and
``h_final`` is the output of the LAST decoder layer. The output ``.pt`` is
byte-compatible with the vllm-lens reader/visualizer ``jacobian_lens.py`` (keys:
``J``, ``source_layers``, ``d_model``) — fit here, then serve + read out there.

This is the ONE fitter for the Jacobian lens. It runs on prime-rl's FSDP2 +
expert-parallel (EP) model stack, so it scales from a 1.7B dense model on a
single GPU up to large MoE models (validated on GLM-4.5-Air, 110B) across many
GPUs / nodes — the same models vllm-lens can then serve and read the lens out on.

Why prime-rl: the naive HF+FSDP path OOMs on large MoE by materializing per-param
grads. This fitter reuses prime-rl's ``setup_model`` / parallel-dims / EP /
torchtitan infra (the exact stack the SFT/RL trainers use) and does NO weight
updates — it only captures activations and backprops activation->activation.

Two environments (see ``jacobian_lens_fit_env.sh``):
  * FIT here runs in the *prime-rl* env (torch 2.11/cu128 + torchtitan). This
    script ``import``s prime-rl internals; it will not run in the vllm-lens env.
  * READ OUT + VISUALIZE run in the *vllm-lens* env (vLLM) via
    ``jacobian_lens.py run`` against a served model.

Method (per prompt):
  1. Forward hooks capture the residual-stream output of each source layer and of
     the last decoder layer. Under EP the residual stream is FULL-WIDTH and
     replicated on every rank (EP shards expert *params*, not activations), so the
     captured activations are complete on rank 0.
  2. Params are frozen; the earliest captured source activation is rooted with
     ``requires_grad_(True)``. Backward then traverses only activation->activation
     (never per-param grads), so no FSDP2 grad reduce-scatter fires and peak
     memory stays ~one layer's worth.
  3. For each block of ``dim_batch`` output dims of ``h_final`` we re-run the
     forward on ``dim_batch`` identical copies of the prompt, build a one-hot
     cotangent selecting one output dim per batch row over the valid token
     positions, and call ``torch.autograd.backward(h_final, grad_tensors=cot)``
     (``autograd.grad`` errors under sharded/EP autograd). Each source's retained
     ``.grad``, meaned over valid source positions, gives that block of rows of
     ``J_l``, accumulated over prompts.

Every rank runs the same forward on the same data, so ``J_l`` is identical across
ranks and only rank 0 saves.

Launch — single node (8 GPUs on one host):

    cd /path/to/prime-rl        # the fit env from jacobian_lens_fit_env.sh
    unset VIRTUAL_ENV
    uv run --no-sync torchrun --nproc-per-node=8 \\
        /path/to/vllm-lens/examples/jacobian_lens_fit.py \\
        --model zai-org/GLM-4.5-Air --layers 25,33,40 --ep 8 \\
        --out glm45air-lens.pt

Launch — multi-node (torchrun; run on EACH node, N_GPUS per node):

    # node 0 (rendezvous host) and node 1, differing only in --node-rank:
    uv run --no-sync torchrun \\
        --nnodes=2 --node-rank=0 --nproc-per-node=8 \\
        --rdzv-id=jaclens --rdzv-backend=c10d \\
        --rdzv-endpoint=$HEAD_IP:29500 \\
        /path/to/vllm-lens/examples/jacobian_lens_fit.py \\
        --model zai-org/GLM-4.5-Air --layers 25,33,40 --ep 16 --out glm45air-lens.pt
    # node 1: identical, but --node-rank=1

Launch — multi-node under SLURM (srun sets the rendezvous env for you):

    srun --nodes=2 --ntasks-per-node=8 --gpus-per-node=8 \\
        torchrun --nnodes=$SLURM_NNODES --node-rank=$SLURM_NODEID \\
            --nproc-per-node=8 --rdzv-backend=c10d \\
            --rdzv-endpoint=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1):29500 \\
            examples/jacobian_lens_fit.py --model ... --layers ... --ep 16 --out ...

``--ep`` is the expert-parallelism degree for MoE models (defaults to ``auto`` =
``min(world_size, 8)``); it must divide the world size and is a no-op for dense
models. Fit only the ``--layers`` you plan to read out — cost scales with the
number of source layers (one backward sweep per prompt covers all of them, but
each fitted layer adds a ``d_model x d_model`` matrix to ship + store).
"""

# Patch ring_flash_attn compat before torch imports (prime-rl requirement).
import prime_rl._compat  # noqa: F401

import argparse
import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist

from prime_rl.configs.trainer import ModelConfig, TokenizerConfig
from prime_rl.trainer.model import setup_model, setup_tokenizer
from prime_rl.trainer.parallel_dims import get_parallel_dims, resolve_ep
from prime_rl.trainer.utils import setup_torch_distributed
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.process import set_proc_title
from prime_rl.utils.vlm import get_language_model


def _parse_args():
    ap = argparse.ArgumentParser(
        description="Fit a Jacobian lens (J_l = E[dh_final/dh_l]) on prime-rl's "
        "FSDP2 + EP stack, for models of any size. Launch under torchrun."
    )
    ap.add_argument("--model", required=True, help="HF model name or local path.")
    ap.add_argument(
        "--layers",
        default=None,
        help="comma-separated source layers to fit (default: every layer in "
        "[0, n_layers-1)). Fewer layers = far cheaper; fit only what you read out.",
    )
    ap.add_argument(
        "--n-prompts",
        type=int,
        default=32,
        help="number of wikitext prompts to average the Jacobian over.",
    )
    ap.add_argument(
        "--dim-batch",
        type=int,
        default=32,
        help="output dims of h_final per forward+backward. Larger => fewer "
        "forwards (d_model/dim_batch per prompt) but more activation memory.",
    )
    ap.add_argument(
        "--max-seq-len",
        type=int,
        default=128,
        help="max prompt length (tokenizer truncation).",
    )
    ap.add_argument(
        "--skip-first",
        type=int,
        default=16,
        help="skip the first N token positions (attention sinks / atypical stats).",
    )
    ap.add_argument(
        "--ep",
        default="auto",
        help="expert-parallelism degree for MoE (int, or 'auto' = "
        "min(world_size, 8)). No-op for dense models.",
    )
    ap.add_argument(
        "--out",
        default="jacobian-lens.pt",
        help="output path for the fitted lens (torch.save: J, source_layers, d_model).",
    )
    args = ap.parse_args()
    args.ep = args.ep if args.ep == "auto" else int(args.ep)
    return args


def _build_model_config(args) -> ModelConfig:
    """Build prime-rl's ``ModelConfig`` programmatically (no TOML).

    ``impl='custom'`` is required for EP; ``compile``, ``ac`` and
    ``ac_offloading`` are disabled because torch.compile and activation
    checkpointing / offloading all break forward-hook capture and the
    intermediate-activation backward this fitter relies on; and the fused lm-head
    (chunked-logprob path) needs labels we never provide, so we use the vanilla
    lm-head (h_final is captured pre-norm by the hook — lm_head output is unused
    anyway).

    ``ac_offloading`` must be passed as ``None`` at construction: it defaults to
    enabled, and a ModelConfig validator re-enables ``ac`` whenever
    ``ac_offloading`` is set — so leaving it on would silently turn AC back on.
    """
    return ModelConfig(
        name=args.model,
        impl="custom",
        ep=args.ep,
        ep_comm_backend="torch",
        attn="sdpa",
        compile=None,
        ac=None,
        ac_offloading=None,
        fused_lm_head_token_chunk_size="disabled",
    )


def _resolve_geometry(model: torch.nn.Module):
    """(decoder_layers, n_layers, d_model) on a (possibly FSDP2/EP-sharded) model.

    ``get_language_model`` unwraps multimodal/composite wrappers to the text
    decoder stack, so no per-model attribute map is needed.
    """
    layers = get_language_model(model).layers
    text_cfg = model.config.get_text_config()
    return layers, len(layers), text_cfg.hidden_size


def fit(args):
    world = get_world()
    logger = setup_logger("info", tag="jacobian-lens-fit")
    logger.info(f"Starting Jacobian-lens fitter in {world}")

    # --- Distributed + parallel dims (reused verbatim from the SFT trainer) ---
    setup_torch_distributed(timeout=timedelta(seconds=600))
    torch.set_float32_matmul_precision("high")
    # The cuDNN SDPA backend fails to load in some environments; disable it so
    # SDPA falls back to flash / mem-efficient / math kernels, which handle the
    # partial-graph (intermediate-activation) backward this fitter relies on.
    torch.backends.cuda.enable_cudnn_sdp(False)

    model_config = _build_model_config(args)
    resolve_ep(model_config)
    parallel_dims = get_parallel_dims(model_config, args.max_seq_len)
    logger.info(f"Parallel dims: {parallel_dims} (ep={model_config.ep})")

    # --- Model + tokenizer ---
    logger.info(f"Initializing model ({model_config.name})")
    model = setup_model(
        model_config,
        parallel_dims,
        loading_from_checkpoint_later=False,
        fused_cross_entropy=False,
    )
    model.eval()
    # Freeze all params: backward then traverses only activation->activation,
    # never building per-param grads (the source of the naive-path OOM). FSDP2
    # reshards after forward regardless of requires_grad, so peak memory stays
    # ~one layer.
    for p in model.parameters():
        p.requires_grad_(False)

    tokenizer = setup_tokenizer(TokenizerConfig(name=args.model))

    device = torch.device("cuda", world.local_rank)
    torch.cuda.synchronize()

    # DTensor.numel() reports the global logical size; count the true local shard.
    def _local_numel(p):
        return p.to_local().numel() if hasattr(p, "to_local") else p.numel()

    n_local = sum(_local_numel(p) for p in model.parameters())
    n_global = sum(p.numel() for p in model.parameters())
    print(
        f"[MEM rank{world.rank}] local_shard={n_local / 1e9:.3f}B "
        f"global={n_global / 1e9:.3f}B "
        f"alloc={torch.cuda.memory_allocated() / 1e9:.2f}GB "
        f"reserved={torch.cuda.memory_reserved() / 1e9:.2f}GB",
        file=sys.stderr,
        flush=True,
    )

    layers, n_layers, d_model = _resolve_geometry(model)
    target_layer = n_layers - 1
    if args.layers is not None:
        source_layers = sorted({int(x) for x in args.layers.split(",")})
    else:
        source_layers = list(range(n_layers - 1))
    assert all(0 <= s < target_layer for s in source_layers), (
        f"source layers must be in [0, {target_layer}); got {source_layers}"
    )
    min_src = min(source_layers)
    logger.info(
        f"n_layers={n_layers} d_model={d_model} fitting {len(source_layers)} layers "
        f"{source_layers} (target=L{target_layer}) dim_batch={args.dim_batch}"
    )

    # --- Forward hooks to capture residual-stream activations ---
    acts: dict[int, torch.Tensor] = {}

    def make_hook(idx: int):
        def hook(_m, _in, out):
            t = out if torch.is_tensor(out) else out[0]
            if idx == min_src and not t.requires_grad:
                t.requires_grad_(True)  # root the graph at the earliest source layer
            acts[idx] = t

        return hook

    handles = [
        layers[idx].register_forward_hook(make_hook(idx))
        for idx in {*source_layers, target_layer}
    ]

    # --- Prompts (wikitext, tokenized directly; no dataloader/renderer) ---
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    corpus = [r for r in ds["text"] if len(r) > 500 and not r.strip().startswith("=")]
    corpus = corpus[: args.n_prompts]

    jac_sum = {lyr: torch.zeros(d_model, d_model) for lyr in source_layers}
    n_done = 0
    skip_first = args.skip_first

    try:
        for pi, prompt in enumerate(corpus):
            ids = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_seq_len,
            ).input_ids.to(device)
            seq_len = ids.shape[1]
            if seq_len <= skip_first + 1:
                continue
            valid = torch.arange(skip_first, seq_len - 1, device=device)
            batched = ids.expand(args.dim_batch, -1)
            position_ids = (
                torch.arange(seq_len, device=device)
                .unsqueeze(0)
                .repeat(args.dim_batch, 1)
            )
            b = torch.arange(args.dim_batch, device=device)

            for start in range(0, d_model, args.dim_batch):
                n = min(args.dim_batch, d_model - start)
                model.zero_grad(set_to_none=True)
                acts.clear()
                with torch.enable_grad():
                    # logits_to_keep=1 => lm_head runs on 1 position only (h_final
                    # is captured pre-norm by the hook, so lm_head output is unused).
                    model(
                        input_ids=batched, position_ids=position_ids, logits_to_keep=1
                    )
                target = acts[target_layer]
                srcs = [acts[lyr] for lyr in source_layers]
                for s in srcs:
                    s.retain_grad()
                cot = torch.zeros_like(target)
                cot[b[:n, None], valid[None, :], start + b[:n, None]] = 1.0
                torch.autograd.backward(target, grad_tensors=cot)
                for lyr, s in zip(source_layers, srcs):
                    rows = s.grad[:n][:, valid, :].float().mean(dim=1)  # [n, d_model]
                    jac_sum[lyr][start : start + n] += rows.detach().cpu()
            n_done += 1
            logger.info(f"  fit prompt {pi + 1}/{len(corpus)} (seq_len={seq_len})")
    finally:
        for h in handles:
            h.remove()

    if dist.is_initialized():
        dist.barrier()
    if n_done == 0:
        raise ValueError("no prompts were long enough to fit on")

    if world.is_master:
        jacobians = {
            int(lyr): (jac_sum[lyr] / n_done).to(torch.float16) for lyr in source_layers
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        torch.save(
            {
                "J": jacobians,
                "source_layers": sorted(jacobians),
                "d_model": int(d_model),
            },
            args.out,
        )
        logger.success(
            f"fit done over {n_done} prompts -> {args.out} "
            f"({len(jacobians)} layers, d_model={d_model})"
        )

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def main():
    set_proc_title("JacobianLensFit")
    fit(_parse_args())


if __name__ == "__main__":
    main()
