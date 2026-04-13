"""NNSight + Transformers activation extraction benchmark."""

import time
from functools import reduce
from pathlib import Path
from typing import Annotated

import torch
import typer
from datasets import load_dataset
from dotenv import load_dotenv
from nnsight import LanguageModel
from tqdm import tqdm
from utils.types import BenchmarkConfig, BenchmarkResult

LIB_NAME = "nnsight-transformers"

app = typer.Typer()


def load_model(model: str) -> LanguageModel:
    return LanguageModel(model, device_map="auto", torch_dtype=torch.bfloat16)


def _resolve_layer(nns, layer_prefix: str, layer: int):
    """Traverse dotted layer_prefix to get the target layer module."""
    parent = reduce(getattr, layer_prefix.split("."), nns)
    return parent[layer]


def extract_activations(
    nns: LanguageModel,
    prompts: list[str],
    layer: int,
    layer_prefix: str,
    batch_size: int,
    max_new_tokens: int,
) -> list:
    # Use generator.all() to capture activations at every generation step.
    # all() applies the intervention recursively across all iterations
    # without a Python-level for-loop, which is faster than iter[:].
    # See: https://nnsight.net/features/4_multiple_token/
    target_layer = _resolve_layer(nns, layer_prefix, layer)
    collected: list[torch.Tensor] = []
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    for batch_start in tqdm(
        range(0, len(prompts), batch_size),
        total=n_batches,
        desc="nnsight-transformers",
    ):
        batch = prompts[batch_start : batch_start + batch_size]
        with nns.generate(max_new_tokens=max_new_tokens) as generator:
            all_acts = list().save()  # type: ignore[attr-defined]
            for prompt in batch:
                with generator.invoke(prompt):
                    with generator.all():
                        all_acts.append(target_layer.output[0].cpu())
        collected.extend(list(all_acts))
    return collected


@app.command()
def main(
    config_file: Annotated[str, typer.Option(help="Path to JSON BenchmarkConfig file")],
) -> None:
    """Benchmark NNSight + Transformers activation extraction."""
    load_dotenv()
    cfg = BenchmarkConfig.model_validate_json(Path(config_file).read_text())

    if cfg.tensor_parallelism > 1 or cfg.pipeline_parallelism > 1:
        raise typer.BadParameter(
            f"NNSight + Transformers does not support parallelism "
            f"(got tp={cfg.tensor_parallelism}, pp={cfg.pipeline_parallelism})"
        )

    ds = load_dataset(cfg.dataset, split="train")
    prompts = [row["instruction"] for row in ds.select(range(cfg.samples))]

    t0 = time.perf_counter()
    nns = load_model(cfg.model)
    startup_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    # Search through batch sizes to find the max we can use without OOM
    acts: list[torch.Tensor] = []
    batch_size = 2
    for batch_size in (128, 64, 32, 16, 8, 4, 2):
        try:
            torch.cuda.empty_cache()
            acts = extract_activations(
                nns,
                prompts,
                cfg.layer,
                cfg.layer_prefix,
                batch_size,
                cfg.max_new_tokens,
            )
            break
        except torch.cuda.OutOfMemoryError:
            if batch_size == 2:
                raise
            typer.echo(
                f"OOM at batch_size={batch_size}, retrying with {batch_size // 2}"
            )
            torch.cuda.empty_cache()
            # Reset startup counter (as previous batch size OOM'd and we're starting again)
            t1 = time.perf_counter()
    run_time = time.perf_counter() - t1

    # Compute activation shape metadata (after timing).
    # Each tensor has shape (batch, seq_len, d_model).
    n_activation_vectors = sum(a.shape[0] for a in acts)
    total_len = sum(a.shape[0] * a.shape[1] for a in acts)
    average_len = total_len / n_activation_vectors
    d_model = acts[0].shape[-1]

    result = BenchmarkResult(
        lib_name=cfg.lib_name or LIB_NAME,
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
    typer.echo(f"{result}  → {out_path}")


if __name__ == "__main__":
    app()
