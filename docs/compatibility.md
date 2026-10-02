# vLLM compatibility

vllm-lens integrates with private vLLM runner and serving APIs. A successful
installation or server startup is not sufficient evidence of compatibility:
capture and interventions must also produce the expected values.

We validate one vLLM release at a time, starting with 0.30.0. After a release
passes local GPU validation, pin the package to that exact version. Test newer
stable releases as they become available and advance the pin only when they
pass; if a candidate fails, retain the last validated pin and investigate.

## Matrix and evidence

| vLLM | Python | PyTorch | Runner | Status |
| --- | --- | --- | --- | --- |
| 0.30.0 | 3.12 | 2.13.0 / CUDA 13.0 | V1, eager | GPU checks run; bfloat16 HF parity fails; not validated |

This row is a **test target, not a declaration that the new suite has passed**.
GPU compatibility tests run locally. Update a row to validated only after
attaching a passing local run for the relevant repository SHA,
including the exact Python/PyTorch/vLLM versions and GPU/driver from its
`environment.json`. CPU CI results alone do not validate inference.

The [local compatibility script](../scripts/check_compatibility.py) saves
environment metadata, per-suite logs and JUnit reports in `--output-dir`.
Attach the reports to the PR or release and include the environment summary in
the release notes. Keep previous validation results when testing a new stable
vLLM release. Do not label an untested replacement supported.

The package's existing `vllm>=0.16.0` dependency remains temporarily unchanged
while the first validation is pending; it is not a tested support guarantee.
Once 0.30.0 passes both suites, change it to `vllm==0.30.0` and update the
development lock and dependency sources to match. The existing lock uses
0.19.0, but that is not an additional support target. Never advance the pin
based only on successful installation, CPU checks or a planned GPU test.

V2 model-runner mode is unsupported and rejected by the plugin. The plugin
defaults to V1 and forces eager execution; native V2/CUDA-graph support is outside
this matrix work. Model-specific features requiring V2 cannot use the fallback.
Passing the small Qwen suite does not establish support for every model,
quantization mode, attention backend, hardware platform or distributed layout.

## CPU checks on every PR

Serialization, binary transport and registration diagnostics live in
`tests/unit/`, outside the GPU fixtures in `vllm_lens/tests/conftest.py`.
The R-lens rule tests also run without vLLM, model downloads or a CUDA runtime:

```bash
uv venv --python 3.12 .venv-cpu
uv pip install --python .venv-cpu/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-cpu/bin/python -r requirements/cpu-tests.txt
uv pip install --python .venv-cpu/bin/python --no-deps -e .
.venv-cpu/bin/python -m pytest tests/unit examples/tests/test_r_lens_rules.py
```

Layer-discovery unit tests additionally import real vLLM classes. They run in the
compatibility environment alongside the GPU discovery tests, not in the minimal
CPU job. The existing lint, format and type checks still run separately.

## Run GPU tests through Slurm

Check out the PR branch (`ci/vllm-compatibility-matrix`) on the cluster. From the
repository root, submit:

```bash
sbatch scripts/compatibility.slurm
```

The job uses the cluster's default account and partition and requests **one node,
`--gres=gpu:2`, eight CPUs, 32 GiB host memory and two hours**. Slurm queues the job
until those resources are available. A single `srun` task launches the tests
inside the allocation; vLLM spawns its TP/PP workers within that step. The script
does not request an exclusive node or modify other jobs.

The compute environment needs Linux, working NVIDIA drivers, Python 3 and
[uv](https://docs.astral.sh/uv/getting-started/installation/) on `PATH`. The job
provisions Python 3.12 and vLLM **0.30.0** in a fresh environment, then runs smoke
and parallel suites sequentially. It stops at the first failure. Ensure any
site-required modules are available in the batch environment.

The test dependencies include `accelerate>=1.1.0` for the Hugging Face reference
fixtures' CUDA device mapping and `ninja` for FlashInfer's runtime kernel builds.
The launcher and direct suite runner put the test environment's executables on
`PATH`; no shell activation is needed. The suite runner checks `ninja` before
starting a model. A missing build tool is a setup failure, so fix the environment
and rerun before drawing conclusions about vLLM compatibility.

Slurm's `CUDA_VISIBLE_DEVICES` is preserved. Compatibility runs disable `.env`
loading and clear inherited `RAY_ADDRESS` so they cannot override the assigned
GPUs or connect to another job's Ray instance. The batch script uses a private
job temporary directory and a job-derived HTTP port; an occupied port fails
validation rather than reusing the existing server. Only its own temporary
directory is removed at exit. Slurm manages worker cleanup when the job ends.

For a single-GPU smoke run or a different explicit candidate:

```bash
sbatch --gres=gpu:1 scripts/compatibility.slurm --suite smoke
sbatch scripts/compatibility.slurm --vllm-version X.Y.Z
# Run just the two-GPU tests:
sbatch scripts/compatibility.slurm --suite parallel
```

Resource overrides such as `--time=04:00:00` go **before** the script path;
launcher arguments such as `--suite smoke` go **after** it. Both suites require
two GPUs on the same node. A 16 GiB GPU is a sensible minimum starting point for
the small ungated `Qwen/Qwen2.5-0.5B-Instruct` model and its Hugging Face reference.
The first run needs network access on the compute node and several GB of disk
space for packages and model weights; later runs reuse download caches.

`sbatch` prints the job ID. Monitor it with `squeue -j JOB_ID`, read the combined
setup/test output in `slurm-vllm-lens-compat-JOB_ID.out`, and use `scancel JOB_ID`
to cancel that job if needed. Environments and reports are kept in the checkout,
so it must be on storage accessible from the compute node.

Git metadata must also be accessible from the compute node. A shared worktree
whose `.git` points to a node-local directory is insufficient; use a complete
shared clone or constrain the job to the node holding the checkout. For the
node-local `/home/ubuntu/vllm-lens` checkout on aft-0:

```bash
sbatch --nodelist=aft-0 scripts/compatibility.slurm
```

The suite runner rejects an unavailable Git revision and writes failed setup
evidence before checking CUDA or starting models.

Inside an existing GPU allocation, use `srun python3 scripts/run_compatibility.py`.
On a standalone GPU machine without Slurm, `python3 scripts/run_compatibility.py`
also works. When Slurm is detected, the launcher and test harness require a job
step and a device mask before touching GPUs; an `salloc` shell on a login node
alone is insufficient.

A smoke-only pass does not validate TP/PP or justify advancing the package pin.
The job log records the environment and results paths. Reports are saved under
`compatibility-results/<version>/<timestamp>-<unique-id>/{smoke,parallel}/`,
including `environment.json` (with Slurm job/step IDs and the device mask),
per-suite logs and JUnit XML. Previous runs are preserved. Attach the complete results directory to the PR after testing.

The launcher uses `uv pip` with an explicit candidate override, so it can test a
new release even after the package has an exact vLLM pin. It does not change
`pyproject.toml`, `uv.lock`, or an existing environment. Avoid `uv sync` in this
test environment: that restores the development lock and CUDA index overrides.

For debugging, rerun an individual suite with the environment path printed by
the launcher and a new output directory:

```bash
srun /path/to/test-environment/bin/python scripts/check_compatibility.py \
  --expected-vllm 0.30.0 --suite smoke --output-dir compatibility-results/retry-smoke
```

The script checks the installed vLLM version and visible GPUs before running,
requires successful HTTP patch registration, and starts a fresh HTTP server.
It refuses to reuse a server already listening on `VLLM_TEST_PORT` (default
8100), which could otherwise test a different checkout or dependency set.
Each suite runs in a separate process to isolate existing GPU teardown behavior.
Any failure, missing JUnit report, empty selection or skipped test fails the run.

Coverage:

| Suite | Checks |
| --- | --- |
| Smoke | Layer discovery; offline capture against Hugging Face; mixed-length batching; chunked prefill; offline generate/chat steering and hooks; async steering; HTTP completion/chat capture and interventions via base64 and binary transport; repeated-prefix isolation; persistent-hook cleanup |
| Parallel | TP=2 / PP=2 batched capture; PP capture/reference checks; norm-matched steering on TP=2 / PP=2 |

The HTTP tests check hook/native capture equality and the numerical sign/scale
of an injected vector at the capture layer. Zeroing hooks must change the
captured state; the following request must restore the baseline. This catches
silent no-ops that a successful completion or shape-only check would miss.

## Local validation before a release

CI runs CPU checks only. Run GPU compatibility tests in a Slurm allocation
using the commands above; there are no scheduled or manually dispatched
GPU jobs in GitHub Actions.

Before a release, run the smoke and parallel suites for the proposed pinned
version at the release commit, using a fresh environment. Run suites
sequentially to avoid GPU memory contention. Review failures and attach the
reports before updating this matrix and the release notes. This is a maintainer
release checklist; the PyPI publication workflow does not enforce GPU evidence.

Until GPU validation is available, keep the target pending and the compatibility
PR in draft. Do not mark issue #39 complete based on CPU results alone.

## Local results, 2026-10-02

Slurm job **138.0**, on aft-0 with two H100 80GB GPUs and driver 595.58.03,
tested the clean implementation revision
`62bf8eba3915fd7f7129d38e5ab055f32b3c6dcf`. The environment used Python 3.12.3,
vLLM 0.30.0, PyTorch 2.13.0 / CUDA 13.0, Transformers 5.18.0,
Accelerate 1.15.0 and Ninja 1.13.2. All six smoke groups were run separately
to inventory failures, followed by the complete standard parallel suite.
The normal launcher still stops at its first failed group.

| Group | Passed | Failed |
| --- | ---: | ---: |
| Discovery | 17 | 0 |
| Offline capture / HF parity | 4 | 1 |
| Chunked prefill | 3 | 0 |
| Offline interventions | 6 | 0 |
| Async interventions | 6 | 0 |
| HTTP hooks, transports and cleanup | 5 | 0 |
| TP/PP capture | 2 | 0 |
| Pipeline capture and steering | 7 | 0 |
| TP/PP norm-matched steering | 2 | 0 |
| **Total** | **52** | **1** |

There were no errors or skipped tests. All 101 CPU regressions also passed
without vLLM/CUDA, with Ruff 0.15.3 lint/format and Pyright passing.

Two setup dependencies were required: Ninja for FlashInfer builds and
Accelerate for reference-model CUDA placement. GPU checks also found a real
post-hook replacement bug: adding a bfloat16 delta to the MLP half of a fused
residual output left values as large as 1.97 when a hook requested zero.
Replacement now writes the requested stream directly and zeros that request's
residual slice, preserving other requests and original tensors. The existing
offline and HTTP checks pass with their original tolerances. CPU regressions
cover outliers, three dtypes, chained hooks, preceding steering and isolation.

The remaining failure is `test_batch_prompts_10_tokens`: bfloat16 vLLM versus
bfloat16 Hugging Face has mean absolute error **0.010466**, above the unchanged
**0.01** limit, for “In the beginning there was nothing but”. Native capture
and independent hook capture are bit-identical; the same discrepancy occurs
when the prompt runs individually. vLLM's fused residual/normalization path
and Hugging Face use different rounding sequences.

Separate ten-prompt diagnostics in Slurm job 137 found worst mean errors of
**0.000681** for float32 vLLM versus float32 HF, and **0.010051** for bfloat16
vLLM versus float32 HF. These are precision diagnostics. The acceptance tests
still compare bfloat16 on both sides, and all correctness thresholds remain
unchanged. A float32 reference alone does not resolve bfloat16 validation.

Full logs, JUnit, package/hardware/revision/Slurm metadata and CPU results are
retained at
`/home/ubuntu/vllm-lens/compatibility-results/0.30.0/diagnostic-job-138/`.
Precision diagnostics are under `hook-fix-job-137/`; combined job outputs are
`slurm-vllm-lens-compat-{137,138}.out`. A compact evidence archive is
`compatibility-results/0.30.0/evidence-62bf8eb.tar.gz`.

The earlier full inventory in shared-storage job 82 had 46 passed and 7 failed;
its six zeroing-hook failures motivated the fix. Its Git metadata was on a
different node, so those reports could not record a revision themselves. The
final job 138 reports record the tested SHA and a clean working tree correctly.

**0.30.0 remains unvalidated and unpinned.** The package's existing vLLM range
and development lock remain temporary; PR #41 stays draft and issue #39 stays
open until the numerical requirement is resolved and both suites pass. PyPI
listed no stable release newer than 0.30.0 at the 2026-10-01 lookup.

## Diagnosing integration drift

Failure to install a completion/chat response patch or the custom HTTP routes
now logs the failed component, vLLM version and original exception. Offline-only
environments can still operate when serving dependencies are absent.

For serving validation, set `VLLM_LENS_STRICT_COMPATIBILITY=1`: a missing HTTP
integration raises during plugin registration instead of only warning. The
compatibility script sets this automatically. `VLLM_LENS_DISABLE=1` still makes
the plugin a complete no-op, including in strict mode.
