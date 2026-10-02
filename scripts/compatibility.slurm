#!/bin/bash
#SBATCH --job-name=vllm-lens-compat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --output=slurm-vllm-lens-compat-%j.out

# Submit from the repository root. Supply partition/account and any site-specific
# resource overrides as sbatch options, before this script's path.
# Arguments after the path are forwarded to run_compatibility.py.
set -euo pipefail

: "${SLURM_JOB_ID:?Submit this script with sbatch; do not run it directly}"
: "${SLURM_SUBMIT_DIR:?Slurm must provide the submission directory}"
cd "$SLURM_SUBMIT_DIR"
if [[ ! -f scripts/run_compatibility.py ]]; then
    echo 'Submit scripts/compatibility.slurm from the repository root.' >&2
    exit 1
fi

# Keep the scheduler's device mask. In particular, do not source a repository
# .env file that might replace CUDA_VISIBLE_DEVICES or redirect Ray to another job.
export PYTHON_DOTENV_DISABLED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
unset RAY_ADDRESS
export VLLM_TEST_PORT="${VLLM_TEST_PORT:-$((20000 + SLURM_JOB_ID % 30000))}"
compat_job_tmp=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/vllm-lens-${SLURM_JOB_ID}.XXXXXX")
export TMPDIR="$compat_job_tmp"
export RAY_TMPDIR="$compat_job_tmp/ray"
trap 'rm -rf -- "$compat_job_tmp"' EXIT

# One task owns the allocated GPUs; vLLM spawns its TP/PP workers inside the step.
# Slurm tracks those workers and reclaims the allocation on cancellation/timeout.
srun --nodes=1 --ntasks=1 --cpu-bind=cores --kill-on-bad-exit=1 \
    python3 scripts/run_compatibility.py "$@"
