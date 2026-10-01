# vLLM compatibility

vllm-lens integrates with private vLLM runner and serving APIs. A successful
installation or server startup is not sufficient evidence of compatibility:
capture and interventions must also produce the expected values.

We target a small matrix: the repository's established development baseline
and a recent stable release. We do not promise every intermediate vLLM version.

## Matrix and evidence

| vLLM | Python | PyTorch | Runner | Status |
| --- | --- | --- | --- | --- |
| 0.19.0 | 3.12 | Selected by vLLM; the development lock uses 2.10.0 | V1, eager | Baseline target; full compatibility run pending |
| 0.30.0 | 3.12 | Selected by vLLM | V1, eager | Candidate target; full compatibility run pending |

These rows are **test targets, not declarations that the new suite has passed**.
No GPU runner was registered when this workflow was added. Update a row to
validated only after attaching a passing run for the relevant repository SHA,
including the exact Python/PyTorch/vLLM versions and GPU/driver from its
`environment.json`. CPU CI results alone do not validate inference.

The [compatibility workflow](../.github/workflows/compatibility.yaml) retains
environment metadata, per-suite logs and JUnit reports as artifacts. Copy the
environment summary and run link into the release notes for durable evidence;
GitHub artifacts expire. Keep the baseline when advancing the candidate to a
new stable vLLM release. Do not silently label an untested replacement supported.

The package's existing `vllm>=0.16.0` dependency remains an installation range,
not a tested support guarantee. Other versions are unvalidated by this matrix.
Do not narrow that range based solely on missing coverage; add an exclusion or
upper bound when a reproducible failure establishes the boundary. Applications
should pin the vLLM version they validated rather than rely on the open range.

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

## Reproduce a GPU matrix entry

Use Linux, Python 3.12 and a CUDA GPU/driver supported by the selected vLLM wheel.
The smoke suite uses the ungated `Qwen/Qwen2.5-0.5B-Instruct` model. Allow room for
both the vLLM engine and a Hugging Face reference model (a 16 GiB GPU is a sensible
starting point). The parallel suite requires two visible GPUs. Model downloads
require network access on the first run.

Create a fresh environment per version. Use `uv pip`, not `uv sync`: the latter
restores the development lock and its CUDA index overrides instead of testing
the selected vLLM release with its own dependencies.

```bash
uv venv --python 3.12 .venv-compatibility
uv pip install --python .venv-compatibility/bin/python 'vllm==0.19.0' -e . -r requirements/compatibility-tests.txt
.venv-compatibility/bin/python scripts/check_compatibility.py \
  --expected-vllm 0.19.0 --suite smoke --output-dir compatibility-results/0.19.0-smoke

# Use a separate, fresh environment to repeat for vllm==0.30.0.
# On a two-GPU machine, also run:
.venv-compatibility/bin/python scripts/check_compatibility.py \
  --expected-vllm 0.19.0 --suite parallel --output-dir compatibility-results/0.19.0-parallel
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

## Runner setup and release cadence

1. Register a dedicated Linux x64 self-hosted GPU runner with an identifying
   label, e.g. `vllm-lens-gpu`. Ensure its driver supports both target releases.
2. Set the repository Actions variable `VLLM_LENS_GPU_RUNNER` to that label.
   Weekly smoke runs are disabled until it is configured; a manual dispatch
   without the variable fails with setup guidance.
3. Dispatch **vLLM compatibility** on the branch/commit to validate, selecting
   `smoke` or `parallel`. Matrix versions run sequentially to avoid contending
   for GPU memory. Only trusted maintainer-selected refs should run on this
   runner; the workflow does not execute on pull-request events.
4. Before a release, require passing smoke and parallel evidence for both
   targets at the release commit. Review failures before publishing and update
   this matrix and release notes with the evidence. This is a maintainer release
   checklist; the PyPI publication workflow does not enforce the GPU evidence.

Until GPU validation is available, keep the rows pending and the compatibility
PR in draft. Do not mark issue #39 complete based on CPU results alone.

## Diagnosing integration drift

Failure to install a completion/chat response patch or the custom HTTP routes
now logs the failed component, vLLM version and original exception. Offline-only
environments can still operate when serving dependencies are absent.

For serving validation, set `VLLM_LENS_STRICT_COMPATIBILITY=1`: a missing HTTP
integration raises during plugin registration instead of only warning. The
compatibility script sets this automatically. `VLLM_LENS_DISABLE=1` still makes
the plugin a complete no-op, including in strict mode.
