"""TransformerLens activation extraction benchmark."""

import time
from typing import Annotated

import torch
import typer
from transformer_lens import HookedTransformer
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
from utils.types import BenchmarkConfig, BenchmarkResult

app = typer.Typer()


def load_model(model: str) -> HookedTransformer:
    return HookedTransformer.from_pretrained(model, dtype=torch.bfloat16)


def extract_activations(
    model: HookedTransformer,
    prompts: list[str],
    layer: int,
    batch_size: int,
) -> None:
    hook_name = f"blocks.{layer}.hook_resid_post"
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    model.eval()
    for i in tqdm(
        range(0, len(prompts), batch_size), total=n_batches, desc="transformer-lens"
    ):
        tokens = model.to_tokens(prompts[i : i + batch_size], prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)


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

    batch_size = cfg.batch_size or 16
    ds = load_dataset(cfg.dataset, split="train")
    prompts = [row["instruction"] for row in ds.select(range(cfg.samples))]

    t0 = time.perf_counter()
    tl = load_model(cfg.model)
    startup_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    extract_activations(tl, prompts, cfg.layer, batch_size)
    run_time = time.perf_counter() - t1

    result = BenchmarkResult(
        lib_name="TransformerLens",
        model=cfg.model,
        n_samples=cfg.samples,
        startup_time=startup_time,
        run_time=run_time,
        batch_size=batch_size,
    )
    out_path = result.save()
    typer.echo(f"{result}  → {out_path}")


if __name__ == "__main__":
    app()
