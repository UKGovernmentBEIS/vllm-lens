"""Shared types for benchmark scripts."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class BenchmarkRun(BaseModel):
    """Configuration for submitting a benchmark as a Slurm job.

    Used by run_all.py to define each benchmark and submit it via sbatch.
    """

    name: str
    script: str
    n_gpus: int
    packages: list[str] = []
    model: str = "facebook/opt-30b"
    n_nodes: int = 1
    time: str = "04:00:00"
    container_name: str = "vllm-lens-0.18.0"
    container_env: str = ""
    # ── Fields that flow into BenchmarkConfig for the script ──
    tensor_parallelism: int = 1
    pipeline_parallelism: int = 1
    distributed_executor_backend: str | None = None
    trust_remote_code: bool = False
    layer_prefix: str = "model.decoder.layers"
    use_ray: bool = False
    lib_name: str = ""
    max_new_tokens: int = 1024


class BenchmarkConfig(BaseModel):
    """Configuration passed to individual benchmark scripts as JSON via --config."""

    name: str
    model: str = "facebook/opt-30b"
    samples: int = 1000
    layer: int = 10
    dataset: str = "tatsu-lab/alpaca"
    tensor_parallelism: int = 1
    pipeline_parallelism: int = 1
    distributed_executor_backend: str | None = None
    trust_remote_code: bool = False
    layer_prefix: str = "model.decoder.layers"
    lib_name: str = ""
    use_ray: bool = False
    max_new_tokens: int = 1024


class Benchmark(BaseModel):
    """Base benchmark configuration."""

    lib_name: str
    model: str
    n_samples: int
    batch_size: int | None = None
    tensor_parallelism: int = 1
    pipeline_parallelism: int = 1

    def _model_slug(self) -> str:
        """Extract short model name: 'facebook/opt-30b' -> 'opt-30b'."""
        return self.model.strip("/").rsplit("/", 1)[-1].lower()

    def run_name(self) -> str:
        """Build a unique run name from hyperparams."""
        parts = [self.lib_name, self._model_slug()]
        if self.batch_size is not None:
            parts.append(f"b{self.batch_size}")
        parts.append(f"s{self.n_samples}")
        parts.append(f"tp{self.tensor_parallelism}")
        parts.append(f"pp{self.pipeline_parallelism}")
        return "_".join(parts)


class BenchmarkResult(Benchmark):
    """Result of a single benchmark run."""

    startup_time: float
    run_time: float
    n_activation_vectors: int | None = None
    average_len: float | None = None
    d_model: int | None = None

    def results_filename(self) -> Path:
        """Return the full output path for this result."""
        output_dir = Path(__file__).resolve().parent.parent / "output"
        return output_dir / (self.run_name() + ".json")

    def save(self, output_dir: Path | None = None) -> Path:
        """Write result JSON to *output_dir* and return the path."""
        if output_dir is None:
            path = self.results_filename()
        else:
            path = output_dir / (self.run_name() + ".json")
        path.parent.mkdir(exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))
        return path
