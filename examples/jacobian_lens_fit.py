"""Fit a Jacobian lens: ``J_l = E[dh_final/dh_l]`` averaged over a text corpus.

For each source layer ``l``, ``J_l`` linearly transports that layer's
residual-stream vector ``h_l`` into the final layer's basis, so that
``unembed(J_l @ h_l)`` reads out what the model is disposed to say from layer
``l``. This script does the one-time fit and writes ``--out`` with keys ``J``,
``source_layers`` and ``d_model``; ``jacobian_lens.py`` reads that file back to
apply and visualize the lens.

Reference: Anthropic, "Verbalizable Representations Form a Global Workspace in
Language Models" (https://transformer-circuits.pub/2026/workspace).

How it works: the model's parameters are frozen; forward hooks capture the
residual stream of each source layer and of the final layer, then a backward
pass from ``h_final`` yields ``dh_final/dh_l`` as a batched vector-Jacobian
product (``dim_batch`` output dimensions per backward, ``d_model/dim_batch``
backwards per prompt), averaged over token positions and prompts. Because it
needs a backward pass, it runs on prime-rl's FSDP2 + expert-parallel stack
rather than the vLLM serving env — build that env with
``jacobian_lens_fit_env.sh``. It scales from a small dense model on one GPU to
large MoE models across many GPUs / nodes.

Usage -- single node (8 GPUs):

    cd /path/to/prime-rl && unset VIRTUAL_ENV        # the fit env
    uv run --no-sync torchrun --nproc-per-node=8 \\
        /path/to/examples/jacobian_lens_fit.py \\
        --model Qwen/Qwen3-1.7B --out lens.pt

Usage -- multi-node (run on every node; differ only in ``--node-rank``):

    uv run --no-sync torchrun \\
        --nnodes=2 --node-rank=$NODE --nproc-per-node=8 \\
        --rdzv-backend=c10d --rdzv-endpoint=$HEAD_IP:29500 \\
        /path/to/examples/jacobian_lens_fit.py \\
        --model zai-org/GLM-4.5-Air --layers 25,33,40 --ep 16 --out lens.pt

Key options: ``--layers`` selects source layers (default: all; fit only the
layers you will read out, since each adds a ``d_model x d_model`` matrix).
``--ep`` sets the expert-parallelism degree for MoE models (must divide the
world size; no-op for dense). ``--n-prompts`` is the number of web-text contexts
to average over (default 1000). Only rank 0 writes the lens.
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
        default=1000,
        help="number of web-text contexts to average the Jacobian over. The "
        "reference lens uses 1000x128 tokens (~100 is usable, little gain beyond).",
    )
    ap.add_argument(
        "--dim-batch",
        type=int,
        default=32,
        help="output dims of h_final per backward. Larger => fewer backwards "
        "(d_model/dim_batch per prompt) but more grad memory per backward.",
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
        "--fp8",
        default="auto",
        choices=["auto", "on", "off"],
        help="fit an FP8-quantized checkpoint (prime-rl's blockwise FP8 linear, "
        "whose autograd.Function backprops grad_x so activation->activation flows "
        "through fp8 layers). 'auto' (default) enables it iff the HF config's "
        "quantization_config.quant_method == 'fp8'. Requires SM90 (Hopper).",
    )
    ap.add_argument(
        "--out",
        default="jacobian-lens.pt",
        help="output path for the fitted lens (torch.save: J, source_layers, d_model).",
    )
    ap.add_argument(
        "--dist-timeout",
        type=int,
        default=600,
        help="process-group (NCCL) collective timeout in seconds. The first load "
        "of an HF checkpoint whose weights need format conversion to prime-rl's "
        "layout is a one-time single-rank step while other ranks wait at a "
        "barrier; for very large models (e.g. 753B) that conversion can exceed "
        "the default, tripping the collective watchdog. Raise this for the "
        "cache-building run (the converted 'prime/' dir is reused afterwards).",
    )
    ap.add_argument(
        "--prompts-file",
        default=None,
        help="JSONL file (one JSON-encoded string per line) of prompts to fit on, "
        "instead of streaming FineWeb. Use for fully offline / reproducible runs. "
        "When omitted, rank 0 streams FineWeb once and caches it to "
        "'<out>.prompts.jsonl' (all ranks read that — 16 ranks streaming at once "
        "trips HF Hub rate limits).",
    )
    ap.add_argument(
        "--per-node-convert",
        action="store_true",
        help="run the one-time HF->prime weight conversion on EACH node's local "
        "rank 0 (writing a node-local 'prime/'), instead of only global rank 0. "
        "Set this when --model is on node-local storage (e.g. /opt/dlami/nvme/...) "
        "not shared across nodes, so every node produces its own converted cache.",
    )
    args = ap.parse_args()
    args.ep = args.ep if args.ep == "auto" else int(args.ep)
    return args


def _detect_fp8(model_name: str) -> bool:
    """True iff the HF checkpoint is FP8-quantized.

    Reads only the raw ``config.json`` (via ``get_config_dict``, which does NOT
    instantiate the model-specific config class), so it works for custom
    ``model_type``s without ``trust_remote_code``. ``quantization_config`` is a
    standard top-level field.
    """
    from transformers import PretrainedConfig

    cfg_dict, _ = PretrainedConfig.get_config_dict(model_name)
    quant = cfg_dict.get("quantization_config") or {}
    return quant.get("quant_method") == "fp8"


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

    ``fp8`` enables prime-rl's blockwise FP8 linear (``replace_linear_with_fp8_
    blockwise_linear``) whose autograd.Function backprops grad_x, so the
    activation->activation backward flows through fp8 layers. ``--fp8 auto``
    turns it on exactly when the checkpoint declares ``quant_method == 'fp8'``.

    When fp8 is enabled we also force ``optimization_dtype='bfloat16'``.
    ``Float8BlockwiseLinear`` stores its weight in bf16 and casts to fp8 per
    matmul (it does NOT keep fp8 weights), and prime-rl's FSDP mixed-precision
    policy computes in bf16 regardless — so bf16 storage matches the compute
    dtype. It is also required for memory: the default float32 storage
    materializes a 753B model at 4 bytes/param (~188 GB per GPU at ep=16),
    OOMing in ``model.to_empty`` before any weights load; bf16 halves that.
    """
    if args.fp8 == "auto":
        fp8 = _detect_fp8(args.model)
    else:
        fp8 = args.fp8 == "on"
    return ModelConfig(
        name=args.model,
        impl="custom",
        ep=args.ep,
        ep_comm_backend="torch",
        attn="sdpa",
        optimization_dtype="bfloat16" if fp8 else "float32",
        compile=None,
        ac=None,
        ac_offloading=None,
        fp8=fp8,
        per_node_convert=args.per_node_convert,
        fused_lm_head_token_chunk_size="disabled",
    )


def _needs_packed_batching(model: torch.nn.Module) -> bool:
    """True for models whose forward requires a PACKED, batch-1 varlen layout.

    DSA (DeepSeek Sparse Attention, e.g. GLM-5.2 ``model_type == 'glm_moe_dsa'``)
    flattens ``position_ids`` to one sequence and builds per-token sparse-window
    bounds ``ks = arange(S) - flat_position_ids``, ``ke = arange(1, S+1)``; its
    ``compute_sparse_indices`` strips the batch dim (``hidden_states[0]``),
    assuming batch=1. Feeding it the batched-VJP ``[dim_batch, seq]`` layout
    crashes ("Tensors must have same number of dimensions: got 4 and 3").

    For these models we instead PACK ``dim_batch`` identical copies into one flat
    sequence with ``position_ids`` resetting to 0 at each segment boundary, which
    the ks/ke formula turns into block-diagonal causal attention (each segment
    attends only within itself => independent identical copies). Regular-attention
    models are unaffected and keep the ``[dim_batch, seq]`` batched-VJP path.
    """
    try:
        model_type = model.config.get_text_config().model_type
    except Exception:
        model_type = getattr(model.config, "model_type", None)
    return model_type == "glm_moe_dsa"


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
    setup_torch_distributed(timeout=timedelta(seconds=args.dist_timeout))
    torch.set_float32_matmul_precision("high")
    # The cuDNN SDPA backend fails to load in some environments; disable it so
    # SDPA falls back to flash / mem-efficient / math kernels, which handle the
    # partial-graph (intermediate-activation) backward this fitter relies on.
    torch.backends.cuda.enable_cudnn_sdp(False)

    model_config = _build_model_config(args)
    resolve_ep(model_config)
    parallel_dims = get_parallel_dims(model_config, args.max_seq_len)
    logger.info(
        f"Parallel dims: {parallel_dims} (ep={model_config.ep}, fp8={model_config.fp8})"
    )

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
    packed = _needs_packed_batching(model)
    logger.info(
        f"n_layers={n_layers} d_model={d_model} fitting {len(source_layers)} layers "
        f"{source_layers} (target=L{target_layer}) dim_batch={args.dim_batch} "
        f"layout={'packed' if packed else 'batched'}"
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

    # --- Prompts: generic web-text matching the reference lens's
    # "pretraining-like corpus" (FineWeb sample-10BT). Multinode-safe: only rank 0
    # hits the HF Hub (16 ranks streaming at once => HTTP 429), writing the chosen
    # prompts to a shared JSONL cache; every rank then reads that file, so all
    # ranks fit the identical corpus in a deterministic order. --prompts-file
    # skips the fetch entirely (fully offline / reproducible).
    import json
    from pathlib import Path

    if args.prompts_file:
        prompts_path = Path(args.prompts_file)
    else:
        prompts_path = Path(f"{args.out}.prompts.jsonl")
        if world.is_master and not prompts_path.exists():
            from datasets import load_dataset

            ds = load_dataset(
                "HuggingFaceFW/fineweb",
                name="sample-10BT",
                split="train",
                streaming=True,
            )
            n = 0
            tmp = Path(f"{prompts_path}.tmp")
            with open(tmp, "w") as f:
                for row in ds:
                    if len(row["text"]) > 500:
                        f.write(json.dumps(row["text"]) + "\n")
                        n += 1
                        if n >= args.n_prompts:
                            break
            os.replace(tmp, prompts_path)
            logger.info(f"cached {n} FineWeb prompts -> {prompts_path}")
        if dist.is_initialized():
            dist.barrier()  # ranks wait for rank 0 to materialize the corpus

    with open(prompts_path) as f:
        corpus = [json.loads(line) for line in f if line.strip()][: args.n_prompts]
    logger.info(f"loaded {len(corpus)} prompts from {prompts_path}")

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
            b = torch.arange(args.dim_batch, device=device)

            # Two input layouts produce the SAME per-block Jacobian rows and the
            # SAME number of backwards (d_model/dim_batch); they differ only in how
            # the dim_batch identical prompt copies are arranged so both regular
            # and DSA (packed varlen) attention accept them.
            #   * batched (regular attn): a real batch [dim_batch, seq]; row i's
            #     one-hot cotangent selects output dim start+i.
            #   * packed (DSA): ONE flat sequence [1, dim_batch*seq] with
            #     position_ids resetting to 0 per segment => block-diagonal causal
            #     attention (each segment independent+identical). Segment i's valid
            #     positions (global i*seq + valid) seed output dim start+i.
            if packed:
                input_ids = ids.repeat(1, args.dim_batch)  # [1, dim_batch*seq]
                position_ids = (
                    torch.arange(seq_len, device=device)
                    .repeat(args.dim_batch)
                    .unsqueeze(0)
                )  # [1, dim_batch*seq], resets to 0 per segment
            else:
                input_ids = ids.expand(args.dim_batch, -1)  # [dim_batch, seq]
                position_ids = (
                    torch.arange(seq_len, device=device)
                    .unsqueeze(0)
                    .repeat(args.dim_batch, 1)
                )

            # ONE forward per prompt, then N = d_model/dim_batch backwards that
            # reuse the retained graph (each covers a different block of output
            # dims). The forward is identical across blocks, so re-running it per
            # block (the old path) wasted ~dim_batch full forwards. `inputs=srcs`
            # restricts backprop to the captured activations (no param grads), and
            # retain_graph is dropped on the last block so the graph is freed.
            model.zero_grad(set_to_none=True)
            acts.clear()
            with torch.enable_grad():
                # logits_to_keep=1 => lm_head runs on 1 position only (h_final is
                # captured pre-norm by the hook, so lm_head output is unused).
                model(input_ids=input_ids, position_ids=position_ids, logits_to_keep=1)
            target = acts[target_layer]
            srcs = [acts[lyr] for lyr in source_layers]
            for s in srcs:
                s.retain_grad()
            starts = list(range(0, d_model, args.dim_batch))
            for bi, start in enumerate(starts):
                n = min(args.dim_batch, d_model - start)
                for s in srcs:
                    s.grad = None  # backward accumulates into .grad; reset per block
                cot = torch.zeros_like(target)
                if packed:
                    # segment i (i<n): global positions i*seq + valid, dim start+i.
                    seg = b[:n, None]  # [n, 1]
                    cot[0, seg * seq_len + valid[None, :], start + seg] = 1.0
                else:
                    cot[b[:n, None], valid[None, :], start + b[:n, None]] = 1.0
                torch.autograd.backward(
                    target,
                    grad_tensors=cot,
                    retain_graph=(bi < len(starts) - 1),
                    inputs=srcs,
                )
                for lyr, s in zip(source_layers, srcs):
                    if packed:
                        # [1, dim_batch*seq, d] -> [dim_batch, seq, d]; segment i's
                        # grad is that segment's copy (block-diagonal => no leak).
                        g = s.grad[0].reshape(args.dim_batch, seq_len, d_model)
                        rows = g[:n][:, valid, :].float().mean(dim=1)  # [n, d_model]
                    else:
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
