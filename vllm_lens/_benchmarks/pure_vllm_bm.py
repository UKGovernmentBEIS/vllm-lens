"""Pure vLLM generation benchmark (no activation extraction)."""

import time
from pathlib import Path
from typing import Annotated

import typer
from datasets import load_dataset
from dotenv import load_dotenv
from vllm import LLM, SamplingParams

from utils.types import BenchmarkConfig, BenchmarkResult

LIB_NAME = "pure-vllm"

app = typer.Typer()


def load_model(
    model: str,
    tensor_parallelism: int,
    pipeline_parallelism: int = 1,
    distributed_executor_backend: str | None = None,
    trust_remote_code: bool = False,
    enforce_eager: bool = False,
) -> LLM:
    kwargs = dict(
        model=model,
        dtype="auto",
        max_model_len=2048,
        tensor_parallel_size=tensor_parallelism,
        pipeline_parallel_size=pipeline_parallelism,
        gpu_memory_utilization=0.90,
    )
    if distributed_executor_backend:
        kwargs["distributed_executor_backend"] = distributed_executor_backend
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    if enforce_eager:
        kwargs["enforce_eager"] = True
    return LLM(**kwargs)


def generate(llm: LLM, prompts: list[str], max_new_tokens: int) -> list:
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=max_new_tokens,
    )
    return llm.generate(prompts, sampling_params)


@app.command()
def main(
    config_file: Annotated[str, typer.Option(help="Path to JSON BenchmarkConfig file")],
) -> None:
    """Benchmark pure vLLM generation without activation extraction."""
    load_dotenv()
    cfg = BenchmarkConfig.model_validate_json(Path(config_file).read_text())

    ds = load_dataset(cfg.dataset, split="train")
    prompts = [row["instruction"] for row in ds.select(range(cfg.samples))]

    t0 = time.perf_counter()
    llm = load_model(
        cfg.model,
        cfg.tensor_parallelism,
        cfg.pipeline_parallelism,
        cfg.distributed_executor_backend,
        cfg.trust_remote_code,
        cfg.enforce_eager,
    )
    startup_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    generate(llm, prompts, cfg.max_new_tokens)
    run_time = time.perf_counter() - t1

    result = BenchmarkResult(
        lib_name=cfg.lib_name or LIB_NAME,
        model=cfg.model,
        n_samples=cfg.samples,
        startup_time=startup_time,
        run_time=run_time,
        tensor_parallelism=cfg.tensor_parallelism,
        pipeline_parallelism=cfg.pipeline_parallelism,
        n_activation_vectors=0,
        average_len=0,
        d_model=0,
    )
    out_path = result.save()
    typer.echo(f"{result}  → {out_path}")


if __name__ == "__main__":
    app()
