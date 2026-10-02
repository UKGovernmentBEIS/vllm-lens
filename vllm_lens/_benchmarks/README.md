# vLLM-Lens Benchmarks

Note these are designed to run on the [Isambard cluster](https://docs.isambard.ac.uk/specs/#system-specifications-isambard-ai-phase-2), so they won't work out-of-the-box unless you have access to it.


Install the public benchmark dependencies with `uv sync --group benchmarking`.
The launcher uses Hugging Face's public `datasets` and `huggingface_hub` APIs
for dataset/model downloads. No private repositories are required.

Supply your Apptainer images explicitly in a directory accessible to the compute
nodes, named `<container_name>.sif` (for example `vllm-0.18.0.sif`). The benchmark
configuration in `run_all.py` specifies which images each comparison needs;
these historical comparison versions are independent of the library's current
validated vLLM pin.

```bash
uv run python vllm_lens/_benchmarks/run_all.py \
  --container-dir /path/to/images --benchmarks pure-vllm --dry-run
```

`VLLM_BENCHMARK_CONTAINER_DIR` can also supply the image directory. Dry runs
validate image paths and print submissions without downloading models/datasets
or submitting jobs. Real runs prefetch weights/data before submitting through
Slurm; gated models still require your Hugging Face access.
