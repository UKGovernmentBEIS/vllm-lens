#!/usr/bin/env bash
#
# Build the environment used to FIT a Jacobian lens (examples/jacobian_lens_fit.py).
#
# The Jacobian-lens flow spans TWO environments:
#   * FIT   — this env. examples/jacobian_lens_fit.py imports prime-rl internals
#             (setup_model / parallel_dims / EP / torchtitan; torch 2.11/cu128)
#             and will NOT run in the vllm-lens env.
#   * READ  — the vllm-lens env (vLLM). Serve the model there, then read the lens
#             out + visualize with examples/jacobian_lens.py run.
# The handoff between them is the fitted lens file (e.g. lens.pt), whose format is
# identical in both envs (keys: J, source_layers, d_model).
#
# This script clones prime-rl pinned at a known-good commit and `uv sync`s it.
# After it finishes, run the fitter from the prime-rl checkout, e.g.:
#
#     cd "$PRIME_RL_DIR"
#     unset VIRTUAL_ENV
#     uv run --no-sync torchrun --nproc-per-node=8 \
#         /abs/path/to/vllm-lens/examples/jacobian_lens_fit.py \
#         --model zai-org/GLM-4.5-Air --layers 25,33,40 --ep 8 --out lens.pt
#
# `--no-sync` is required: after prime-rl's editable install, a plain `uv run`
# would re-sync and uninstall the package (see prime-rl's README).
set -euo pipefail

# Pinned prime-rl commit that provides the trainer infra the fitter imports.
PRIME_RL_COMMIT="${PRIME_RL_COMMIT:-f3dd3c1}"
PRIME_RL_REPO="${PRIME_RL_REPO:-https://github.com/AI-Safety-Institute/prime-rl.git}"
PRIME_RL_DIR="${PRIME_RL_DIR:-$HOME/prime-rl-jacobian-fit}"

command -v uv >/dev/null 2>&1 || {
    echo "error: 'uv' not found. Install from https://docs.astral.sh/uv/ first." >&2
    exit 1
}

if [[ ! -d "$PRIME_RL_DIR/.git" ]]; then
    echo ">> cloning prime-rl into $PRIME_RL_DIR"
    git clone "$PRIME_RL_REPO" "$PRIME_RL_DIR"
fi

cd "$PRIME_RL_DIR"
echo ">> checking out pinned commit $PRIME_RL_COMMIT"
git fetch --all --tags
git checkout "$PRIME_RL_COMMIT"

# Build the env (torch 2.11/cu128 + torchtitan, all extras). This can take a
# while and downloads large wheels.
echo ">> uv sync --all-extras"
uv sync --all-extras

echo
echo ">> done. prime-rl fit env ready at: $PRIME_RL_DIR"
echo ">> fit from there with 'uv run --no-sync torchrun ... jacobian_lens_fit.py'"
echo ">> read the resulting lens out in the vllm-lens env: jacobian_lens.py run"
