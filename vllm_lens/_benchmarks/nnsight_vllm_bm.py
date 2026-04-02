"""NNSight + vLLM activation extraction benchmark."""

import time
from functools import reduce
from pathlib import Path
from typing import Annotated

import ray
import torch
import typer
from datasets import load_dataset
from dotenv import load_dotenv
from nnsight.modeling.vllm import VLLM
from utils.types import BenchmarkConfig, BenchmarkResult

LIB_NAME = "nnsight-vllm"

app = typer.Typer()


def load_model(
    model: str,
    tensor_parallelism: int,
    use_ray: bool,
    trust_remote_code: bool = False,
):
    kwargs = dict(
        dtype="auto",
        max_model_len=2048,
        tensor_parallel_size=tensor_parallelism,
        dispatch=True,
    )
    if use_ray:
        kwargs["distributed_executor_backend"] = "ray"
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    return VLLM(model, **kwargs)


def _resolve_layer(nns, layer_prefix: str, layer: int):
    """Traverse dotted layer_prefix to get the target layer module."""
    parent = reduce(getattr, layer_prefix.split("."), nns)
    return parent[layer]


CHUNK_SIZE = 100  # prompts per trace to avoid msgspec 4GB encode limit


def extract_activations(
    nns,
    prompts: list[str],
    layer: int,
    layer_prefix: str,
    max_new_tokens: int,
) -> list:
    # Process prompts in chunks to avoid exceeding msgspec's 4GB encoding
    # limit when accumulated activations get too large.
    # Each chunk runs a separate nns.trace() with tracer.all() to capture
    # activations at every generation step.
    # See: https://nnsight.net/notebooks/features/vllm_support/
    target_layer = _resolve_layer(nns, layer_prefix, layer)
    all_acts = []
    for start in range(0, len(prompts), CHUNK_SIZE):
        chunk = prompts[start : start + CHUNK_SIZE]
        with nns.trace(temperature=1.0, max_tokens=max_new_tokens) as tracer:
            activations = [list() for _ in range(len(chunk))].save()
            for i, prompt in enumerate(chunk):
                with tracer.invoke(prompt):
                    with tracer.all():
                        activations[i].append(target_layer.output[0].cpu())
        # Each activations[i] is a list of [d_model] tensors — stack into [seq_len, d_model]
        all_acts.extend(torch.stack(list(pa)) for pa in activations)
    return all_acts


@app.command()
def main(
    config_file: Annotated[str, typer.Option(help="Path to JSON BenchmarkConfig file")],
) -> None:
    """Benchmark NNSight + vLLM activation extraction."""
    load_dotenv()
    cfg = BenchmarkConfig.model_validate_json(Path(config_file).read_text())

    if cfg.use_ray:
        ray.init(address="auto")

    ds = load_dataset(cfg.dataset, split="train")
    prompts = [row["instruction"] for row in ds.select(range(cfg.samples))]

    t0 = time.perf_counter()
    nns = load_model(
        cfg.model, cfg.tensor_parallelism, cfg.use_ray, cfg.trust_remote_code
    )
    startup_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    acts = extract_activations(
        nns, prompts, cfg.layer, cfg.layer_prefix, cfg.max_new_tokens
    )
    run_time = time.perf_counter() - t1

    # Compute activation shape metadata (after timing).
    # Each activation has shape [seq_len, d_model] from stacked per-step tensors.
    n_activation_vectors = len(acts)
    average_len = sum(a.shape[0] for a in acts) / len(acts)
    d_model = acts[0].shape[-1]

    result = BenchmarkResult(
        lib_name=cfg.lib_name or LIB_NAME,
        model=cfg.model,
        n_samples=cfg.samples,
        startup_time=startup_time,
        run_time=run_time,
        tensor_parallelism=cfg.tensor_parallelism,
        n_activation_vectors=n_activation_vectors,
        average_len=average_len,
        d_model=d_model,
    )
    out_path = result.save()
    typer.echo(f"{result}  → {out_path}")


if __name__ == "__main__":
    app()
