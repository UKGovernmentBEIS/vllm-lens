"""NNSight (HuggingFace transformers backend) activation extraction benchmark.

Uses nnsight LanguageModel with generate() + tracer.all() to capture residual
stream activations at every generated token step.
"""

import time
from typing import Annotated

import torch
import typer
from datasets import load_dataset
from dotenv import load_dotenv
from nnsight import LanguageModel
from tqdm import tqdm

from utils.types import BenchmarkConfig, BenchmarkResult

LIB_NAME = "nnsight-hf"

app = typer.Typer()


def load_model(model: str) -> LanguageModel:
    return LanguageModel(model, device_map="auto", dtype=torch.bfloat16)


def extract_activations(
    lm: LanguageModel, prompts: list[str], layer: int, batch_size: int
) -> None:
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    for i in tqdm(
        range(0, len(prompts), batch_size), total=n_batches, desc="nnsight-hf"
    ):
        batch = prompts[i : i + batch_size]
        # tracer.all() applies the save at every generation step (prefill + each
        # new token), exercising the full activation-capture overhead.
        with lm.generate(batch, max_new_tokens=1024, do_sample=False) as tracer:
            with tracer.all():
                lm.model.decoder.layers[layer].output[0].save()
        torch.cuda.empty_cache()


@app.command()
def main(
    config_file: Annotated[str, typer.Option(help="Path to JSON BenchmarkConfig file")],
) -> None:
    """Benchmark NNSight HuggingFace activation extraction with generation."""
    from pathlib import Path

    load_dotenv()
    cfg = BenchmarkConfig.model_validate_json(Path(config_file).read_text())

    if cfg.tensor_parallelism > 1 or cfg.pipeline_parallelism > 1:
        raise typer.BadParameter(
            f"nnterp does not support parallelism (got tp={cfg.tensor_parallelism}, pp={cfg.pipeline_parallelism})"
        )

    batch_size = cfg.batch_size or 16
    ds = load_dataset(cfg.dataset, split="train")
    prompts = [row["instruction"] for row in ds.select(range(cfg.samples))]

    t0 = time.perf_counter()
    lm = load_model(cfg.model)
    startup_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    extract_activations(lm, prompts, cfg.layer, batch_size)
    run_time = time.perf_counter() - t1

    result = BenchmarkResult(
        lib_name=LIB_NAME,
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
