#!/usr/bin/env bash
#
# Build the environment for fitting a Jacobian lens (examples/jacobian_lens_fit.py).
#
# The fitter needs a backward pass, so it runs on prime-rl's FSDP2 +
# expert-parallel stack (not the vLLM env used to read the lens out). This clones
# prime-rl at a pinned commit and installs it with uv; CUDA / hardware
# requirements come from prime-rl's own dependencies, not this script.
#
# Then run the fitter from the prime-rl checkout:
#     cd "$PRIME_RL_DIR" && unset VIRTUAL_ENV
#     uv run --no-sync torchrun --nproc-per-node=8 \
#         /abs/path/to/examples/jacobian_lens_fit.py --model Qwen/Qwen3-1.7B --out lens.pt
# (--no-sync stops uv from re-syncing and uninstalling prime-rl's editable install.)
set -euo pipefail

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
echo ">> checking out $PRIME_RL_COMMIT"
git fetch --all --tags
git checkout "$PRIME_RL_COMMIT"

echo ">> uv sync --all-extras (downloads large wheels; takes a while)"
uv sync --all-extras

echo ">> done. fit env ready at: $PRIME_RL_DIR"
