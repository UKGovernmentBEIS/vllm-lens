"""TransformerLens activation extraction benchmark."""

import time
from typing import Annotated

import torch
import typer
from datasets import load_dataset
from dotenv import load_dotenv
from transformer_lens import HookedTransformer
from utils.types import BenchmarkConfig, BenchmarkResult

app = typer.Typer()


def log(msg: str) -> None:
    """Flush-safe logging so output appears in Slurm logs immediately."""
    print(msg, flush=True)


def log_gpu() -> None:
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        log(
            f"  [GPU] allocated={alloc:.2f}GB reserved={reserved:.2f}GB total={total:.2f}GB"
        )


def load_model(model: str) -> HookedTransformer:
    return HookedTransformer.from_pretrained(model, dtype=torch.bfloat16)


def extract_activations(
    model: HookedTransformer,
    prompts: list[str],
    layer: int,
    batch_size: int,
    max_new_tokens: int,
) -> list[torch.Tensor]:
    hook_name = f"blocks.{layer}.hook_resid_post"
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    model.eval()
    all_acts: list[torch.Tensor] = []

    log(
        f"Starting extraction: {len(prompts)} prompts, batch_size={batch_size}, "
        f"n_batches={n_batches}, max_new_tokens={max_new_tokens}"
    )
    log_gpu()

    for batch_idx, i in enumerate(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i : i + batch_size]
        log(
            f"\n--- Batch {batch_idx + 1}/{n_batches} ({len(batch_prompts)} prompts) ---"
        )

        log("  Tokenizing...")
        tokens = model.to_tokens(batch_prompts, prepend_bos=True)
        log(f"  Token shape: {tokens.shape}")
        log_gpu()

        with torch.no_grad():
            # Pass 1: generate full sequences
            log(f"  Generating (max_new_tokens={max_new_tokens})...")
            t_gen = time.perf_counter()
            try:
                full_tokens = model.generate(
                    tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=1.0,
                    verbose=False,
                )
            except torch.cuda.OutOfMemoryError:
                log("  CUDA OOM during generate()!")
                log_gpu()
                raise
            except Exception as e:
                log(f"  ERROR during generate(): {type(e).__name__}: {e}")
                raise
            gen_time = time.perf_counter() - t_gen
            log(
                f"  Generate done in {gen_time:.1f}s, output shape: {full_tokens.shape}"
            )
            log_gpu()

            # Pass 2: extract activations over the complete sequences
            log(f"  Running run_with_cache (hook={hook_name})...")
            t_cache = time.perf_counter()
            try:
                _, cache = model.run_with_cache(full_tokens, names_filter=hook_name)
            except torch.cuda.OutOfMemoryError:
                log("  CUDA OOM during run_with_cache()!")
                log_gpu()
                raise
            except Exception as e:
                log(f"  ERROR during run_with_cache(): {type(e).__name__}: {e}")
                raise
            cache_time = time.perf_counter() - t_cache
            act = cache[hook_name]
            log(f"  Cache done in {cache_time:.1f}s, activation shape: {act.shape}")
            log_gpu()

            log("  Moving to CPU...")
            all_acts.append(act.cpu())
            del cache, act, full_tokens
            torch.cuda.empty_cache()
            log_gpu()

    return all_acts


@app.command()
def main(
    config_file: Annotated[str, typer.Option(help="Path to JSON BenchmarkConfig file")],
) -> None:
    """Benchmark TransformerLens activation extraction."""
    from pathlib import Path

    load_dotenv()
    cfg = BenchmarkConfig.model_validate_json(Path(config_file).read_text())

    if cfg.tensor_parallelism > 1 or cfg.pipeline_parallelism > 1:
        raise typer.BadParameter(
            f"TransformerLens does not support parallelism (got tp={cfg.tensor_parallelism}, pp={cfg.pipeline_parallelism})"
        )

    ds = load_dataset(cfg.dataset, split="train")
    prompts = [row["instruction"] for row in ds.select(range(cfg.samples))]

    log(f"Model: {cfg.model}, samples: {cfg.samples}, layer: {cfg.layer}")
    log(f"max_new_tokens: {cfg.max_new_tokens}")

    t0 = time.perf_counter()
    tl = load_model(cfg.model)
    startup_time = time.perf_counter() - t0
    log(f"Model loaded in {startup_time:.1f}s")
    log_gpu()

    t1 = time.perf_counter()
    for batch_size in (8,):
        try:
            torch.cuda.empty_cache()
            log(f"\nTrying batch_size={batch_size}")
            all_acts = extract_activations(
                tl, prompts, cfg.layer, batch_size, cfg.max_new_tokens
            )
            break
        except torch.cuda.OutOfMemoryError:
            if batch_size == 2:
                raise
            log(f"OOM at batch_size={batch_size}, retrying with {batch_size // 2}")
            torch.cuda.empty_cache()
            t1 = time.perf_counter()
    run_time = time.perf_counter() - t1

    # Compute activation shape metadata (after timing).
    # Each tensor has shape (batch, seq_len, d_model).
    n_activation_vectors = sum(a.shape[0] for a in all_acts)
    total_len = sum(a.shape[0] * a.shape[1] for a in all_acts)
    average_len = total_len / n_activation_vectors
    d_model = all_acts[0].shape[-1]

    result = BenchmarkResult(
        lib_name="TransformerLens",
        model=cfg.model,
        n_samples=cfg.samples,
        startup_time=startup_time,
        run_time=run_time,
        batch_size=batch_size,
        n_activation_vectors=n_activation_vectors,
        average_len=average_len,
        d_model=d_model,
    )
    out_path = result.save()
    log(f"{result}  → {out_path}")


if __name__ == "__main__":
    app()
