# GPU compatibility handoff

This is a temporary handoff requested by the maintainer on 2026-10-01 for an
agent running on the GPU cluster. **Delete this file from the repository when
the work is done, and commit/push its deletion.** Keep lasting instructions and
validation results in `docs/compatibility.md` and the PR description instead.

## Objective and working branch

Continue issue [#39](https://github.com/UKGovernmentBEIS/vllm-lens/issues/39)
and draft PR [#41](https://github.com/UKGovernmentBEIS/vllm-lens/pull/41):
diagnose the failing vLLM 0.30.0 GPU compatibility run, fix the setup or actual
integration as appropriate, and run the complete smoke and TP/PP suites.
The maintainer has authorized implementation, test runs through Slurm, commits,
pushes, and updates to this PR. Continue the work rather than just proposing fixes.

- Repository: `UKGovernmentBEIS/vllm-lens`.
- Branch: `ci/vllm-compatibility-matrix`.
- Cluster checkout: `/home/ubuntu/vllm-lens` on `aft-0` (verify locally).
- Last implementation commit before this handoff: `acf81bf` (Ninja/PATH fix).
- Check branch, status and any local instructions before editing. Preserve other
  work; do not reset or switch a dirty checkout. The previous agent worked in
  `/tmp/vllm-lens-issue39` on a different host; do not assume that path exists here.
- No GPU tests belong in GitHub Actions. CPU CI remains; GPU validation is local
  to the maintainer's machine/cluster. Do not reintroduce weekly/manual GPU CI.

## Mandatory: use Slurm for GPU work

The maintainer explicitly requires scheduling to avoid interrupting other jobs.
**Never run GPU tests, engine startup, CUDA probes or kernel compilation directly
on the host outside a Slurm job step.** Reading logs and editing files is fine.
Do not bypass scheduler guards or manually set a mask to access unallocated GPUs.

Use the default/unspecified account and partition. This cluster requires `gres`.
From the repository root:

```bash
sbatch scripts/compatibility.slurm
# A focused single-GPU smoke run:
sbatch --gres=gpu:1 scripts/compatibility.slurm --suite smoke
# The two-GPU suite alone:
sbatch scripts/compatibility.slurm --suite parallel
```

The batch script requests one node, one task, `--gres=gpu:2`, eight CPUs, 32 GiB
RAM and two hours. It invokes the launcher with `srun`. Submission resource
overrides go BEFORE the script path; launcher arguments go AFTER it. Let Slurm
queue the job. Check whether the user's current run is still active before
submitting another; do not edit files underneath an active test run.

Preserve Slurm's `CUDA_VISIBLE_DEVICES`, disable dotenv loading, clear inherited
`RAY_ADDRESS`, and keep the script's private temporary/Ray directory and
job-derived HTTP port. Do not use broad process kills or disturb other jobs.
If cancellation is necessary, identify and cancel only this compatibility job.
An allocation shell alone is insufficient: GPU execution must be within `srun`.

`sbatch` prints the job ID. Use `squeue -j JOB_ID` for progress; the combined
output is `slurm-vllm-lens-compat-JOB_ID.out` in the submission directory.
Each pytest subprocess has a 30-minute timeout. Pytest captures output, so a
quiet log alone does not establish a hang. Inspect the final traceback.

## Current failure: inspect this first

The first run confirmed two visible CUDA GPUs but discovery failed because
FlashInfer could not launch `ninja` while profiling its sampler during engine
startup. Commit `acf81bf` adds `ninja>=1.11` to the compatibility dependencies,
prepends the environment's `bin` to child-process PATH, and checks Ninja before
model startup. Direct suite invocation also sets the executable search path.

The maintainer pulled that fix and reran. Latest environment:

```text
/home/ubuntu/vllm-lens/.venv-compatibility/vllm-0.30.0-tywuquxz
```

Latest result directory:

```text
/home/ubuntu/vllm-lens/compatibility-results/0.30.0/20261001T203357Z-d14s5fcb/smoke
```

Ninja now prints `1.13.2.git.kitware.jobserver-pipe-1`, and the torch GPU check
passed. The printed assertion source includes “Need 2 visible CUDA GPUs”; that
is not an error when execution continues to discovery.

Final output supplied by the maintainer:

```text
# All 15 layer-discovery unit tests passed.
vllm_lens/tests/test_layer_discovery.py::test_registry_discovery_and_capture ERROR [ 94%]
vllm_lens/tests/test_layer_discovery.py::test_layer_path_override PASSED [100%]
16 passed, 14 warnings, 1 error in 196.18s (0:03:16)
```

**The immediate blocker is another missing test dependency: `accelerate`.**
The traceback is in setup of `test_registry_discovery_and_capture`, before its
capture/reference comparison runs:

```text
vllm_lens/tests/conftest.py:37: in hf_model
    model = AutoModelForCausalLM.from_pretrained(...)
transformers/modeling_utils.py:4193: in from_pretrained
    device_map = check_and_set_device_map(device_map)
# device_map = {'': device(type='cuda', index=0)}
transformers/integrations/accelerate.py:137:
ValueError: Using a `device_map`, `tp_plan`, `torch.device` context manager or
setting `torch.set_default_device(device)` requires `accelerate`.
You can install it with `pip install accelerate`
```

First add `accelerate` to `requirements/compatibility-tests.txt` with a version
constraint appropriate to the installed Transformers, so fresh environments
include it. This dependency fix has NOT been made by the handing-off agent.
Inspect the fixture and dependency resolution, then rerun through Slurm. For a
focused retry you can install it into the existing compatibility environment,
but also commit the reproducible requirements fix. The passing layer-path test
shows progress beyond the previous Ninja failure; it does not establish full
capture parity or compatibility.

Read `discovery.log` and `environment.json` in the directory above and the
relevant Slurm output for complete evidence. Check job state before starting
another run. Distinguish environment/build-tool failures from vLLM API or
numerical compatibility failures. Do not disable a failing backend or weaken
correctness checks just to obtain a pass.

Older evidence, if needed:
`compatibility-results/0.30.0/20261001T202352Z-yc6fqg7a/smoke/` (job 58.0).
That run had 15 passed, 1 failed, 1 error and ended in
`FileNotFoundError: [Errno 2] No such file or directory: 'ninja'`.

## Implementation and test expectations

- `scripts/run_compatibility.py`: stdlib bootstrap; requires Linux, uv, NVIDIA
  driver and Slurm job/step on this cluster. Creates a unique Python 3.12 venv
  under `.venv-compatibility/`, installs the explicit vLLM override and
  `requirements/compatibility-tests.txt`, checks GPUs, then runs smoke and parallel
  sequentially. Defaults to 0.30.0. Never clears previous environments/results.
- `scripts/check_compatibility.py`: checks exact installed version and GPUs,
  selects V1/eager mode and strict plugin registration, runs separate pytest
  processes, saves hardware/package/revision/Slurm metadata in `environment.json`,
  and retains per-suite `.log` and JUnit `.xml` files. Failures, skips, empty
  selections and missing reports must fail validation.
- Smoke covers discovery, capture against Hugging Face, mixed batches, chunked
  prefill, offline/async steering and hooks, and HTTP completion/chat capture,
  binary/base64 transport, intervention effects and cleanup.
- Parallel covers TP=2 and PP=2 capture and steering. One smoke pass does not
  validate parallel execution. Inspect all reports, not just a process exit code.
- The plugin forces V1/eager. V2/native CUDA-graph support is out of scope.
- Read `docs/compatibility.md` for exact commands and evidence requirements.
  Focused retries may reuse an environment, but use Slurm and a fresh results
  directory. For direct `check_compatibility.py` retries, use its venv Python
  within an allocated `srun` step and preserve the batch script's isolation.
- Do not `uv sync` the compatibility environment: that restores the older dev
  lock and CUDA source overrides. Fix setup in repository files so a fresh
  `sbatch scripts/compatibility.slurm` reproduces the working environment.
- CPU regression tests live in `tests/unit` and
  `examples/tests/test_r_lens_rules.py`; setup requirements are documented.
  `tests/unit/test_compatibility_setup.py` and `test_compatibility_slurm.py` cover
  launcher failures, allocation guards, executable lookup and batch handoff.

Previous-host verification at `acf81bf`: 89 CPU tests passed in a clean Python
3.12 environment without vLLM (torch 2.14.1+cpu); Ruff lint/format and Pyright
passed. GPU compatibility has not passed. These environments and tool paths
may not exist on this host. Run appropriate CPU checks for changes; run lint,
format and type checks before the final push.

## Version policy and completion

The maintainer wants to start with **0.30.0**, then test newer stable releases
explicitly and keep the exact pin at the latest version that passes. Do not use
an unbounded `latest` target or infer that untested versions work. At the last
lookup 0.30.0 was the latest stable; verify availability before proposing newer
candidates. If there are no newer stable releases, finish with 0.30.0.

Currently `pyproject.toml` still says `vllm>=0.16.0` and `uv.lock` uses 0.19.0.
This is temporary while validation is pending, not a compatibility guarantee.
Once 0.30.0 passes BOTH full suites, pin to `vllm==0.30.0`, update the development
lock/dependency sources to match, and validate the final configuration. Test
available newer stable candidates sequentially with `--vllm-version X.Y.Z`;
retain the last validated pin if a newer candidate fails. Avoid unrelated
dependency churn. Never pin an unvalidated target as supported.

Commit/push fixes to the existing PR branch and update PR #41's description
with the actual problem, final behavior, checks and limitations. Keep it draft
until full GPU evidence exists. Retain results tied to the tested revision,
including exact packages, GPU/driver, Slurm IDs, logs and JUnit; summarize or
attach evidence to the PR without committing environments, caches or large logs.
Report setup failures honestly and do not mark #39 complete based on CPU tests.
Do not merge or publish a release as part of this handoff.

When finished, move any useful lasting notes into the normal documentation,
**delete `GPU_COMPATIBILITY_HANDOFF.md`, commit and push the deletion**, and tell
the maintainer what passed, which version is pinned, and where evidence lives.
