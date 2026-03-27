"""vllm-lens activation extraction benchmark."""

import time
from pathlib import Path
from typing import Annotated

import typer
from datasets import load_dataset
from dotenv import load_dotenv
from vllm import LLM, SamplingParams

from utils.types import BenchmarkConfig, BenchmarkResult

LIB_NAME = "vllm-lens"

app = typer.Typer()


def load_model(
    model: str,
    tensor_parallelism: int,
    pipeline_parallelism: int = 1,
    distributed_executor_backend: str | None = None,
    trust_remote_code: bool = False,
) -> LLM:
    kwargs = dict(
        model=model,
        dtype="auto",
        max_model_len=2048,
        tensor_parallel_size=tensor_parallelism,
        pipeline_parallel_size=pipeline_parallelism,
        gpu_memory_utilization=0.90,
    )
    if pipeline_parallelism == 1:
        kwargs["quantization"] = "fp8"
    if distributed_executor_backend:
        kwargs["distributed_executor_backend"] = distributed_executor_backend
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    return LLM(**kwargs)


def extract_activations(
    llm: LLM, prompts: list[str], layer: int, max_new_tokens: int
) -> list:
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        extra_args={"output_residual_stream": [layer]},
    )
    return llm.generate(prompts, sampling_params)


@app.command()
def main(
    config_file: Annotated[str, typer.Option(help="Path to JSON BenchmarkConfig file")],
) -> None:
    """Benchmark vllm-lens activation extraction."""
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
    )
    startup_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    outputs = extract_activations(llm, prompts, cfg.layer, cfg.max_new_tokens)
    run_time = time.perf_counter() - t1

    # Compute activation shape metadata (after timing).
    acts = [out.activations["residual_stream"] for out in outputs]
    n_activation_vectors = len(acts)
    average_len = sum(a.shape[1] for a in acts) / len(acts)
    d_model = acts[0].shape[-1]

    result = BenchmarkResult(
        lib_name=cfg.lib_name or LIB_NAME,
        model=cfg.model,
        n_samples=cfg.samples,
        startup_time=startup_time,
        run_time=run_time,
        tensor_parallelism=cfg.tensor_parallelism,
        pipeline_parallelism=cfg.pipeline_parallelism,
        n_activation_vectors=n_activation_vectors,
        average_len=average_len,
        d_model=d_model,
    )
    out_path = result.save()
    typer.echo(f"{result}  → {out_path}")


if __name__ == "__main__":
    app()
