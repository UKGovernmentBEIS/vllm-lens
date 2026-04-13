import os
import subprocess
from pathlib import Path
from typing import Annotated

import sifter
import typer
from dotenv import load_dotenv
from isambard_container_tools._helpers.pre_download import (
    pre_download_datasets,
    pre_download_models,
)
from utils.types import BenchmarkConfig, BenchmarkRun

_HERE_resolved = Path(__file__).parent

# Convert Lustre path to PROJECTDIR-relative path so it's accessible inside the
# Singularity container (which only has $PROJECTDIR bound, not /lus/lfs1aip2/...).
_PROJECTDIR = os.environ.get("PROJECTDIR", "")
if _PROJECTDIR:
    try:
        _rel = _HERE_resolved.relative_to(Path(_PROJECTDIR).resolve())
        HERE = Path(_PROJECTDIR) / _rel
    except ValueError:
        HERE = _HERE_resolved
else:
    HERE = _HERE_resolved

SLURM_TEMPLATE = HERE / "ray.slurm"
LOGS_DIR = HERE / "logs"


def _resolve_container(container_name: str) -> str:
    """Resolve a container name like 'vllm-lens-0.18.0' to a .sif path via sifter."""
    last_dash = container_name.rfind("-")
    prefix = container_name[: last_dash + 1]
    version = container_name[last_dash + 1 :]
    return sifter.find_latest_container(prefix=prefix, version=version)


# ─── Per-benchmark configuration ──────────────────────────────────────────────
BENCHMARKS: list[BenchmarkRun] = [
    BenchmarkRun(
        name="vllm-lens",
        script="vllm_lens_bm.py",
        n_gpus=1,
        tensor_parallelism=1,
        pipeline_parallelism=1,
        lib_name="vllm-lens",
        container_env="HF_HUB_OFFLINE=1",
        packages=["python-dotenv"],
    ),
    BenchmarkRun(
        name="vllm-lens-tp4",
        script="vllm_lens_bm.py",
        n_gpus=4,
        tensor_parallelism=4,
        pipeline_parallelism=1,
        lib_name="vllm-lens-tp4",
        container_env="HF_HUB_OFFLINE=1",
        packages=["python-dotenv"],
    ),
    BenchmarkRun(
        name="pure-vllm",
        script="pure_vllm_bm.py",
        n_gpus=1,
        tensor_parallelism=1,
        pipeline_parallelism=1,
        enforce_eager=True,
        container_name="vllm-0.18.0",
        container_env="HF_HUB_OFFLINE=1",
        packages=["python-dotenv", "datasets"],
    ),
    BenchmarkRun(
        name="pure-vllm-tp4",
        script="pure_vllm_bm.py",
        n_gpus=4,
        tensor_parallelism=4,
        pipeline_parallelism=1,
        enforce_eager=True,
        container_name="vllm-0.18.0",
        container_env="HF_HUB_OFFLINE=1",
        packages=["python-dotenv", "datasets"],
    ),
    BenchmarkRun(
        name="nnsight-vllm",
        script="nnsight_vllm_bm.py",
        n_gpus=1,
        tensor_parallelism=1,
        batch_size=512,
        container_name="vllm-0.15.1",
        container_env="HF_HUB_OFFLINE=1",
        packages=["python-dotenv", "datasets", "nnsight"],
    ),
    BenchmarkRun(
        name="nnsight-vllm-tp4",
        script="nnsight_vllm_bm.py",
        n_gpus=4,
        tensor_parallelism=4,
        batch_size=64,
        use_ray=True,
        lib_name="nnsight-vllm-tp4",
        container_name="vllm-0.15.1",
        container_env="HF_HUB_OFFLINE=1",
        packages=["python-dotenv", "datasets", "nnsight", "ray"],
    ),
    # TP4 NNsight OOMs with 1 node
    BenchmarkRun(
        name="nnsight-vllm-tp4-2-nodes",
        script="nnsight_vllm_bm.py",
        n_gpus=2,
        tensor_parallelism=4,
        n_nodes=2,
        batch_size=64,
        use_ray=True,
        lib_name="nnsight-vllm-tp4",
        container_name="vllm-0.15.1",
        container_env="HF_HUB_OFFLINE=1",
        packages=["python-dotenv", "datasets", "nnsight", "ray"],
    ),
    BenchmarkRun(
        name="transformer-lens",
        script="transformer_lens_bm.py",
        n_gpus=1,
        time="04:00:00",
        exclusive=True,
        container_name="vllm-0.18.0",
        container_env="HF_HUB_OFFLINE=1",
        packages=["python-dotenv", "transformer-lens", "tqdm"],
    ),
    BenchmarkRun(
        name="hf-transformers",
        script="hf_transformers_bm.py",
        n_gpus=1,
        exclusive=True,
        container_name="vllm-0.18.0",
        container_env="HF_HUB_OFFLINE=1",
        packages=["python-dotenv", "datasets", "tqdm"],
    ),
    BenchmarkRun(
        name="nnsight-vllm-llama405b-tp16",
        script="nnsight_vllm_bm.py",
        model="meta-llama/Meta-Llama-3.1-405B-Instruct",
        n_gpus=4,
        n_nodes=4,
        time="04:00:00",
        tensor_parallelism=16,
        use_ray=True,
        layer_prefix="model.layers",
        lib_name="nnsight-vllm-llama405b-tp16",
        container_name="vllm-0.15.1",
        container_env="HF_HUB_OFFLINE=1",
        packages=["python-dotenv", "datasets", "nnsight", "ray"],
    ),
    BenchmarkRun(
        name="vllm-lens-llama405b-tp16",
        script="vllm_lens_bm.py",
        model="meta-llama/Meta-Llama-3.1-405B-Instruct",
        n_gpus=4,
        n_nodes=4,
        time="04:00:00",
        tensor_parallelism=16,
        pipeline_parallelism=1,
        distributed_executor_backend="ray",
        use_ray=True,
        layer_prefix="model.layers",
        lib_name="vllm-lens-llama405b-tp16",
        container_env="HF_HUB_OFFLINE=1",
        packages=["python-dotenv", "ray"],
    ),
    BenchmarkRun(
        name="pure-vllm-llama405b-tp16",
        script="pure_vllm_bm.py",
        model="meta-llama/Meta-Llama-3.1-405B-Instruct",
        n_gpus=4,
        n_nodes=4,
        time="04:00:00",
        tensor_parallelism=16,
        pipeline_parallelism=1,
        distributed_executor_backend="ray",
        enforce_eager=True,
        use_ray=True,
        layer_prefix="model.layers",
        container_name="vllm-0.18.0",
        container_env="HF_HUB_OFFLINE=1",
        packages=["python-dotenv", "datasets", "ray"],
    ),
    BenchmarkRun(
        name="vllm-lens-glm5-tp16",
        script="vllm_lens_bm.py",
        model="zai-org/GLM-5-FP8",
        n_gpus=4,
        n_nodes=4,
        time="04:00:00",
        tensor_parallelism=16,
        pipeline_parallelism=1,
        distributed_executor_backend="ray",
        use_ray=True,
        trust_remote_code=True,
        layer_prefix="model.layers",
        lib_name="vllm-lens-glm5-tp16",
        container_env="HF_HUB_OFFLINE=1;DG_JIT_CACHE_DIR=/tmp/deepgemm_jit",
        packages=["python-dotenv", "ray"],
    ),
    BenchmarkRun(
        name="pure-vllm-glm5-tp16",
        script="pure_vllm_bm.py",
        model="zai-org/GLM-5-FP8",
        n_gpus=4,
        n_nodes=4,
        time="04:00:00",
        tensor_parallelism=16,
        pipeline_parallelism=1,
        distributed_executor_backend="ray",
        enforce_eager=True,
        use_ray=True,
        trust_remote_code=True,
        layer_prefix="model.layers",
        lib_name="pure-vllm-glm5-tp16",
        container_name="vllm-0.18.0",
        container_env="HF_HUB_OFFLINE=1;DG_JIT_CACHE_DIR=/tmp/deepgemm_jit",
        packages=["python-dotenv", "datasets", "ray"],
    ),
]

app = typer.Typer()


def _build_config(
    bench: BenchmarkRun, samples: int, layer: int, dataset: str
) -> BenchmarkConfig:
    """Build a BenchmarkConfig from a BenchmarkRun and CLI args."""
    return BenchmarkConfig(
        name=bench.name,
        model=bench.model,
        samples=samples,
        layer=layer,
        dataset=dataset,
        tensor_parallelism=bench.tensor_parallelism,
        pipeline_parallelism=bench.pipeline_parallelism,
        distributed_executor_backend=bench.distributed_executor_backend,
        trust_remote_code=bench.trust_remote_code,
        layer_prefix=bench.layer_prefix,
        lib_name=bench.lib_name or bench.name,
        use_ray=bench.use_ray,
        max_new_tokens=bench.max_new_tokens,
        enforce_eager=bench.enforce_eager,
        batch_size=bench.batch_size,
    )


def submit_job(
    bench: BenchmarkRun,
    samples: int,
    layer: int,
    dataset: str,
    dry_run: bool = False,
) -> str | None:
    """Build environment, write sbatch command, and submit."""
    config = _build_config(bench, samples, layer, dataset)
    config_json = config.model_dump_json()

    env = os.environ.copy()
    env["CONTAINER"] = _resolve_container(bench.container_name)
    env["WORK_DIR"] = str(HERE)
    env["USER_SCRIPT"] = str(HERE / bench.script)
    env["BENCHMARK_CONFIG"] = config_json
    env["PACKAGES"] = " ".join(bench.packages)
    if bench.container_env:
        env["CONTAINER_ENV"] = bench.container_env

    cmd = [
        "sbatch",
        "--export=ALL",
        f"--job-name=speed-{bench.name}",
        f"--nodes={bench.n_nodes}",
        f"--gpus-per-node={bench.n_gpus}",
        f"--time={bench.time}",
    ]
    if bench.exclusive:
        cmd.append("--exclusive")
    cmd.append(str(SLURM_TEMPLATE))

    if dry_run:
        typer.echo(f"  [DRY RUN] {bench.name}")
        typer.echo(f"    cmd: {' '.join(cmd)}")
        typer.echo(f"    USER_SCRIPT: {env['USER_SCRIPT']}")
        typer.echo(f"    BENCHMARK_CONFIG: {env['BENCHMARK_CONFIG']}")
        typer.echo(f"    PACKAGES: {env['PACKAGES']}")
        return "dry-run"

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        typer.echo(f"  [ERROR] {bench.name}: {result.stderr.strip()}", err=True)
        return None

    job_id = result.stdout.strip().split()[-1]
    return job_id


@app.command()
def main(
    benchmarks: Annotated[
        list[str] | None,
        typer.Option("--benchmarks", "-b", help="Names to run (default: all)"),
    ] = None,
    samples: Annotated[int, typer.Option(help="Number of Alpaca samples")] = 1000,
    layer: Annotated[
        int, typer.Option(help="Layer index for activation extraction")
    ] = 10,
    dataset: Annotated[
        str, typer.Option(help="HuggingFace dataset path")
    ] = "tatsu-lab/alpaca",
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print sbatch commands without submitting")
    ] = False,
) -> None:
    """Submit activation-extraction benchmarks as Slurm jobs."""
    load_dotenv()
    LOGS_DIR.mkdir(exist_ok=True)
    pre_download_datasets([{"path": dataset, "split": "train"}])

    selected = [
        b for b in BENCHMARKS if benchmarks is None or b.name in (benchmarks or [])
    ]
    if not selected:
        typer.echo(
            f"No matching benchmarks. Available: {[b.name for b in BENCHMARKS]}",
            err=True,
        )
        raise typer.Exit(1)

    for bench in selected:
        pre_download_models([bench.model])
        job_id = submit_job(bench, samples, layer, dataset, dry_run=dry_run)
        if job_id and job_id != "dry-run":
            total_gpus = bench.n_gpus * bench.n_nodes
            node_info = f" × {bench.n_nodes} nodes" if bench.n_nodes > 1 else ""
            typer.echo(
                f"  {bench.name:35s}  job={job_id}  ({total_gpus} GPU{'s' if total_gpus > 1 else ''}{node_info})"
            )

    typer.echo(f"\nLogs → {LOGS_DIR}/speed_<name>_<jobid>.log")
    typer.echo(f"Results → {HERE}/output/<name>-results.json")


if __name__ == "__main__":
    app()
