"""NNSight + vLLM activation extraction benchmark."""

import time
from functools import reduce
from pathlib import Path
from typing import Annotated

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


def extract_activations(
    nns,
    prompts: list[str],
    layer: int,
    layer_prefix: str,
    max_new_tokens: int,
) -> list:
    # Use list().save() — an nnsight-managed list at trace scope — so the graph
    # tracks appends properly across the vLLM engine subprocess boundary.
    # A plain Python list doesn't work because tracer.invoke() body runs in
    # the engine subprocess.
    target_layer = _resolve_layer(nns, layer_prefix, layer)
    with nns.trace() as tracer:
        activations = list().save()
        for prompt in prompts:
            with tracer.invoke(prompt, temperature=1.0, max_tokens=max_new_tokens):
                activations.append(target_layer.output[0].cpu())
    return [a for a in activations]


@app.command()
def main(
    config_file: Annotated[str, typer.Option(help="Path to JSON BenchmarkConfig file")],
) -> None:
    """Benchmark NNSight + vLLM activation extraction."""
    load_dotenv()
    cfg = BenchmarkConfig.model_validate_json(Path(config_file).read_text())

    if cfg.use_ray:
        import ray

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
    n_activation_vectors = len(acts)
    d_model = acts[0].shape[-1]
    # nnsight + vLLM returns last-token hidden state per prompt (shape [d_model]).
    # If shape has a sequence dimension, compute average_len; otherwise it's 1.
    if acts[0].ndim >= 2:
        average_len = sum(a.shape[-2] for a in acts) / len(acts)
    else:
        average_len = 1.0

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
