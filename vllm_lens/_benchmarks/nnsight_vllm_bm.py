"""NNSight + vLLM activation extraction benchmark (async mode).

Uses ``nnsight.modeling.vllm.VLLM(mode="async")`` + ``tracer.cache()`` and
fires every prompt concurrently via ``asyncio.gather`` so vLLM's scheduler
can form multi-request batches. This replaces the older sync path
(``tracer.all()`` + 100-prompt trace batching), which is materially slower
and also hits msgspec's 4 GB encode limit on 1000×1024 workloads because
every prompt's activations ride back on a single engine output socket.

Per-request zstd transport (landed on nnsight's ``refactor/transform``
branch) avoids the 4 GB cap and keeps submissions concurrent end-to-end.
"""

import asyncio
import time
from functools import reduce
from pathlib import Path
from typing import Annotated

import nnsight as nns
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
        gpu_memory_utilization=0.90,
        dispatch=True,
        mode="async",
    )
    if use_ray:
        kwargs["distributed_executor_backend"] = "ray"
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    return VLLM(model, **kwargs)


def _resolve_layer(model, layer_prefix: str, layer: int):
    parent = reduce(getattr, layer_prefix.split("."), model)
    return parent[layer]


async def _run_one(model, prompt: str, target_layer, max_new_tokens: int):
    """Submit one async trace and drain its backend stream."""
    with model.trace(prompt, temperature=1.0, max_tokens=max_new_tokens) as tracer:
        c = tracer.cache(modules=[target_layer])
        nns.save(c)

    # AsyncVLLMBackend.__call__() returns the async generator; the
    # bare ``tracer.backend`` is the backend object itself, not iterable.
    final_output = None
    async for output in tracer.backend():
        if output.finished:
            final_output = output

    saves = getattr(final_output, "saves", {}) if final_output else {}
    return saves.get("c", None)


async def _run_all(model, prompts: list[str], target_layer, max_new_tokens: int):
    tasks = [
        asyncio.create_task(_run_one(model, p, target_layer, max_new_tokens))
        for p in prompts
    ]
    return await asyncio.gather(*tasks)


def _cache_tensor(cache_dict):
    """Return the concatenated hidden-state tensor for a CacheDict.

    ``tracer.cache()`` stores an ``Entry`` (or list of Entry) per module;
    for decoder layers each output is ``(hidden_states, ...)`` with
    ``hidden_states`` shaped ``[T_step, hidden]``. We concat along dim 0.
    """
    if cache_dict is None or len(cache_dict) == 0:
        return None

    key = next(iter(cache_dict.keys()))
    entry = cache_dict[key]
    entries = entry if isinstance(entry, list) else [entry]

    pieces: list[torch.Tensor] = []
    for e in entries:
        out = e.output
        if isinstance(out, (tuple, list)):
            out = out[0]
        if isinstance(out, torch.Tensor):
            pieces.append(out)

    if not pieces:
        return None
    return torch.cat(pieces, dim=0)


def extract_activations(
    model,
    prompts: list[str],
    layer: int,
    layer_prefix: str,
    max_new_tokens: int,
) -> list[torch.Tensor]:
    target_layer = _resolve_layer(model, layer_prefix, layer)
    caches = asyncio.run(_run_all(model, prompts, target_layer, max_new_tokens))
    tensors = [t for t in (_cache_tensor(c) for c in caches) if t is not None]
    return tensors


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
    model = load_model(
        cfg.model, cfg.tensor_parallelism, cfg.use_ray, cfg.trust_remote_code
    )
    startup_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    acts = extract_activations(
        model, prompts, cfg.layer, cfg.layer_prefix, cfg.max_new_tokens
    )
    run_time = time.perf_counter() - t1

    n_activation_vectors = len(acts)
    average_len = sum(a.shape[0] for a in acts) / len(acts) if acts else 0.0
    d_model = acts[0].shape[-1] if acts else 0

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
