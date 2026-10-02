# GPU compatibility handoff — 2026-10-02

The maintainer is reinstalling the repository and starting an agent directly
on the GPU machine after a cluster outage. Continue the work below.
**All GPU work must go through Slurm.** This is the maintainer's explicit
requirement, even when your shell is already on a GPU node.

This file is temporary. When the work is finished, preserve lasting results in
`docs/compatibility.md` and the PR, then **delete this file and commit/push its
deletion**.

## Objective and checkout

Finish GPU validation for issue [#39](https://github.com/UKGovernmentBEIS/vllm-lens/issues/39)
and draft PR [#41](https://github.com/UKGovernmentBEIS/vllm-lens/pull/41).
Diagnose and fix actual failures, retain evidence, and update the existing PR.
The maintainer has authorized implementation, Slurm test runs, commits, pushes
and PR updates. Do not stop at a plan when you can continue the work.

- Repository: `https://github.com/UKGovernmentBEIS/vllm-lens`.
- Branch: **`ci/vllm-compatibility-matrix`**; do not work on main accidentally.
- Latest implementation commit before this handoff: **`1b30b30`**.
- That commit incorporates main at `a71e386`, including Q/K capture and the
  FP32 Triton precision fix. GitHub reported no merge conflicts, and both CI
  jobs passed: [run 37003508660](https://github.com/UKGovernmentBEIS/vllm-lens/actions/runs/37003508660).
- **128 CPU tests passed** without vLLM/CUDA. Ruff lint/format, Pyright,
  actionlint and Slurm shell syntax checks passed.
- Previous cluster checkout: `/home/ubuntu/vllm-lens`, node `aft-0`. Verify the
  current hostname, checkout, branch, status and local instructions. The machine
  may have been rebuilt; old environments and evidence may no longer exist.
- The previous non-GPU agent used `/tmp/vllm-lens-issue39` on another host.
  Do not assume that worktree or its tool environments exist here.
- Preserve unrelated changes. Use a complete clone whose Git metadata is
  accessible on the compute node; a shared worktree pointing at another node's
  local `.git` directory cannot record valid evidence.

## Mandatory Slurm execution

**Never run inference, GPU tests, CUDA probes or CUDA kernel compilation outside
a Slurm job step. Never bypass scheduler checks or set a device mask to access
unallocated GPUs.** Reading logs, editing code and CPU-only checks can run in
your ordinary shell. If Slurm is unavailable after the outage, report that
blocker; do not fall back to running GPU work directly.

Use the default/unspecified account and partition. The cluster requires
`--gres`; the supplied script requests one node, one task, two GPUs, eight CPUs,
32 GiB host memory and two hours. From the repository root:

```bash
sbatch scripts/compatibility.slurm
```

If the checkout is node-local on aft-0 and that is still the correct node:

```bash
sbatch --nodelist=aft-0 scripts/compatibility.slurm
```

Check node availability first. Do not blindly use the old node name on a rebuilt
cluster. If the checkout is shared, allow Slurm to select an appropriate node.
Other useful submissions:

```bash
sbatch --gres=gpu:1 scripts/compatibility.slurm --suite smoke
sbatch scripts/compatibility.slurm --suite parallel
sbatch scripts/compatibility.slurm --vllm-version X.Y.Z
```

Resource options go BEFORE the script path; Python launcher options go AFTER it.
The script launches setup and tests through `srun`. For focused debugging in an
existing allocation, still use `srun`; an allocation shell alone is insufficient.
Preserve the batch script's isolation when writing diagnostic job scripts:

- Keep Slurm's `CUDA_VISIBLE_DEVICES`; do not source `.env` over it.
- Set `PYTHON_DOTENV_DISABLED=1`, clear inherited `RAY_ADDRESS`, and use private
  temporary/Ray directories and a job-specific HTTP port.
- Keep all TP/PP workers within the allocated step. Do not use broad process
  kills, cancel other jobs, claim an exclusive node unnecessarily or clear
  shared caches. Cancel only a confirmed compatibility job if needed.
- Check for an existing compatibility job before submitting another, and do not
  modify code underneath an active test run. Commit the tested code so its
  revision and clean/dirty status in the reports are meaningful.

Monitor the ID returned by `sbatch` with `squeue -j JOB_ID`. Combined output is
`slurm-vllm-lens-compat-JOB_ID.out` in the submission directory. Each pytest group
has a 30-minute timeout. Captured pytest output may remain quiet until a test
finishes; inspect logs/job state before assuming a hang.

## What passed, and what remains unknown

GPU evidence described below is recorded in the PR and documentation. It has
not been rechecked on this rebuilt machine. Recover original files if available;
otherwise preserve the distinction between recorded results and fresh validation.

1. At clean revision `62bf8eba3915fd7f7129d38e5ab055f32b3c6dcf`, Slurm job
   **138.0** ran all six then-existing smoke groups separately and the complete
   standard parallel suite: **52 passed, 1 failed, no errors/skips**. All hook
   checks and all **11 TP/PP tests passed**.
2. The sole failure was BF16 vLLM versus BF16 Hugging Face on one batched prompt:
   mean absolute error **0.010466**, above the unchanged **0.01** limit. Native
   capture and independent hook capture were bit-identical, and the discrepancy
   also occurred for individual requests.
3. Separate ten-prompt diagnostics in Slurm job **137**, revision
   `7ea64dbcc647eca49c0ad87d11fd8e7171ac9f6e`, gave worst mean error **0.000681
   for FP32/FP32**, within tolerance. BF16 vLLM/FP32 HF gave 0.010051.
4. The maintainer explicitly accepts BF16 reference rounding discrepancies
   **provided FP32 parity passes**. They want to verify the computations, not
   gate on vLLM's reduced-precision rounding. Do not reopen the BF16 tolerance
   failure as a blocker or loosen the FP32 correctness threshold.
5. Commit `168de9a` changed the reference tests to execute BOTH engines in FP32,
   while retaining BF16 functional tests, and included async HF parity in smoke.
   Full validation was submitted as **Slurm job 147**. Its completion evidence
   is **unconfirmed following the outage**. Do not call it passing or still
   running without checking recovered reports.
6. Commit `1b30b30` subsequently merged main's Q/K capture and FP32 Triton fix.
   Earlier GPU evidence predates this merge. A complete run of the updated
   smoke and parallel suites is the immediate remaining task.

Old evidence locations, which may have been lost during reinstall:

```text
/home/ubuntu/vllm-lens/compatibility-results/0.30.0/diagnostic-job-138/
/home/ubuntu/vllm-lens/compatibility-results/0.30.0/hook-fix-job-137/
/home/ubuntu/vllm-lens/compatibility-results/0.30.0/evidence-62bf8eb.tar.gz
/home/ubuntu/vllm-lens/slurm-vllm-lens-compat-137.out
/home/ubuntu/vllm-lens/slurm-vllm-lens-compat-138.out
/home/ubuntu/vllm-lens/slurm-vllm-lens-compat-147.out
```

Recorded hardware/software: two H100 80GB GPUs, driver 595.58.03, Python 3.12.3,
vLLM 0.30.0, PyTorch 2.13.0 / CUDA 13.0, Transformers 5.18.0, Accelerate 1.15.0,
Ninja 1.13.2. Record the actual rebuilt environment; do not assume it matches.

## Setup and correctness details

Read `docs/compatibility.md` and the three scripts before running.

- `scripts/compatibility.slurm` is the supported scheduler entry point.
- `scripts/run_compatibility.py` needs Linux, Python 3, uv and working NVIDIA
  drivers. Inside the allocation it provisions Python 3.12 in a unique
  `.venv-compatibility/` environment, installs an explicit vLLM override and
  `requirements/compatibility-tests.txt`, checks CUDA visibility, and runs smoke
  then parallel sequentially. Default target: **0.30.0**. It stops on failure.
- Test dependencies already include **Accelerate and Ninja**, and environment
  executables are exposed on PATH for FlashInfer subprocesses. Earlier missing
  dependency failures have been fixed; do not repeat those old diagnoses without
  inspecting the new traceback. Transformers is fixed at **5.18.0** across
  candidates; PyTorch follows vLLM's requirements.
- Do not `uv sync` a compatibility environment: it would restore the older
  development lock/CUDA index overrides. Use the launcher. Fresh runs reuse
  package/model download caches without clearing prior environments or results.
- `scripts/check_compatibility.py` checks the exact vLLM version, visible GPUs,
  Ninja and accessible Git revision. It enables strict plugin registration and
  V1/eager mode. V2/native CUDA graphs are outside this work's scope.
- Reports are under
  `compatibility-results/<version>/<timestamp>-<unique-id>/{smoke,parallel}/`:
  `environment.json`, per-group logs and JUnit XML. Missing reports, empty
  selections, errors, failures or skips must fail validation. Inspect the full
  report inventory, not only a process exit code. Keep prior results.
- FP32 reference parity means both models execute in FP32. Casting BF16 outputs
  afterward does not qualify. The shared reference fixture uses `float32`;
  native-dtype fixtures still exercise capture, batching, hooks, steering,
  HTTP and TP/PP. The reference mean absolute error limit remains 0.01.
- Main's plugin now defaults `TRITON_F32_DEFAULT=ieee` for FP32 models while
  respecting explicit overrides. Avoid an inherited TF32 override invalidating
  the intended FP32 comparison; record relevant settings if diagnosing precision.
- A real post-hook bug was fixed: applying a BF16 delta to a fused residual
  output left nonzero values when a hook requested zero. Replacement now writes
  the requested stream directly and clears that request's residual slice,
  preserving other requests and original tensors. Do not revert this fix or
  relax the functional tests to accommodate it.
- Main's Q/K GPU parity suites are separate and are **not selected by this
  compatibility runner**. Do not claim a passing residual-stream run validates
  all Q/K models/backends. Keep the coverage description accurate.
- CPU tests are in `tests/unit` plus `examples/tests/test_r_lens_rules.py`.
  Setup is documented, including a CPU-only torch installation. Main's attention
  helper and engine-config tests now live there too. Run relevant CPU checks and
  lint/format/type checks after fixes; GPU tests always go through Slurm.

## Completion and version policy

Start with the full updated **0.30.0 smoke AND parallel suites**. Diagnose any
failure from its traceback and values; fix setup or integration reproducibly in
the repository. Do not skip failures, weaken checks or disable a failing backend
just to get a pass. If the cluster remains unavailable, report the precise
blocker and preserve the pending status.

Once both suites pass, pin `pyproject.toml` to **`vllm==0.30.0`** and update the
development lock/dependency sources consistently, verifying the final setup.
Currently the dependency is still `vllm>=0.16.0` and the lock uses 0.19.0;
these are temporary, not tested support guarantees. Do not claim support or
advance the pin on CPU results or historical diagnostics alone.

The maintainer's longer-term policy is to test available newer stable releases
explicitly and retain the latest passing exact pin when a newer version fails.
At the 2026-10-01 lookup no stable release newer than 0.30.0 was available;
verify availability rather than inventing candidates or using unbounded latest.
Avoid unrelated dependency churn.

Commit/push to the existing PR branch. Update the PR description and compatibility
matrix with the actual tested revision, complete suite counts, exact versions,
hardware and evidence locations. Preserve/attach logs, JUnit and metadata without
committing venvs, weights, caches or large raw logs. Keep historical diagnostics
distinct from final validation. Only mark the PR ready once the pending GPU
validation is substantiated; do not merge or publish a release in this task.

When done, remove this temporary handoff, commit/push its deletion, and tell the
maintainer what passed, which version is pinned and where the evidence is stored.

## Continuation results — 2026-10-02, 12:40 UTC

GPU validation is now complete. Clean commit `5cbffc1` forces spawned workers
in the compatibility runner after job 1's second pipeline engine deadlocked in
CPU tensor allocation. Full Slurm job **5.0** passed **45 smoke + 11 parallel**
tests, with no failures/errors/skips. All 10 report groups were inspected.
128 CPU tests, Ruff 0.15.3 and Pyright 1.1.414 pass.

Fresh evidence is in `compatibility-results/0.30.0/20261002T122444Z-myq9ko1h/`
and `docs/compatibility-evidence/vllm-0.30.0-job-5.tar.gz`. `docs/compatibility.md`
records the exact tested SHA/environment. No newer stable vLLM release was
available on PyPI on 2026-10-02. Job 1's incomplete reports/stacks are retained
in `20261002T120553Z-itnpoyz2/`. No old reports were recovered.

Remaining blocker: coordinated pin/development-lock update. Prepared changes
in `pyproject.toml` pin vLLM 0.30.0 and replace the cu126 torch/torchvision index
with cu130, but are uncommitted until lock regeneration succeeds. `uv lock`
fails fetching existing private benchmarking dependencies because current
GitHub credentials cannot access `UKGovernmentBEIS/sifter` and
`AI-Safety-Institute/hpc-containers`. Authentication for this PR repository
works. Use session-scoped Git credential configuration for uv:

```bash
GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=credential.helper \
  GIT_CONFIG_VALUE_0='!gh auth git-credential' uv lock
```

A temporary resolution preview excluding benchmarking (not the final lock)
confirmed vLLM 0.30.0 / torch 2.13.0+cu130 resolves. Do not copy that preview
lock: it removes unrelated benchmarking dependencies. Obtain private repo
access, regenerate the complete lock, verify development setup and review
unrelated churn before committing the pin. The prepared final PR text is at
`/tmp/vllm-lens-pr-41.md`. Keep the PR draft until setup is verified, then update
it and remove this handoff as required above. No GPU rerun is required merely
for dependency metadata/docs changes; inference code/tests are unchanged from
the passing revision. Any implementation change does require GPU verification.
