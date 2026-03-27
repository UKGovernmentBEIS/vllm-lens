#!/bin/bash
#SBATCH --job-name=ray-run
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#
# Container-based Ray multi-node script that runs a user-provided script
# on the head node inside the container. The user script is responsible for
# starting Ray head and any application logic (e.g., vLLM serve).
#
# Required environment variables (passed via sbatch --export):
#   CONTAINER        - Path to container image
#   WORK_DIR         - Project root directory
#   USER_SCRIPT      - Python script to run on the head node (inside container)
#   BENCHMARK_CONFIG - JSON config string (written to file, passed as --config-file)
#
# Optional:
#   PACKAGES       - Space-separated Python packages to pip install before running
#   CONTAINER_ENV  - Semicolon-separated KEY=VALUE pairs to export into container
#                    (e.g., VLLM_USE_DEEP_GEMM=0;VLLM_ATTENTION_BACKEND=TRITON_MLA)
#   DEBUG_MODE     - Set to "1" to enable verbose logging and per-node resource monitoring

echo "Using backend ray"

set -euo pipefail

# ── Unset login-node vars that break Apptainer ────────────────────────
# https://docs.isambard.ac.uk/service-status/known_issues/#use-of-environment-variables-in-apptainer
unset DBUS_SESSION_BUS_ADDRESS XDG_RUNTIME_DIR LOCALDIR TMPDIR 2>/dev/null || true

# ── Per-job scratch tmp dir ───────────────────────────────────────────
# Isolate XDG dirs per job on Lustre scratch so concurrent jobs don't collide
# on NFS $HOME (e.g. inspect_ai trace rotation causing stale file handles).
JOB_TMP="$SCRATCHDIR/tmp/${SLURM_JOB_ID}"
mkdir -p "$JOB_TMP"
export XDG_DATA_HOME="$JOB_TMP/xdg_data"
export XDG_CACHE_HOME="$JOB_TMP/xdg_cache"
export XDG_STATE_HOME="$JOB_TMP/xdg_state"
export RAY_TMPDIR="$JOB_TMP"

# ── Debug mode ──────────────────────────────────────────────────────
if [ "${DEBUG_MODE:-0}" = "1" ]; then
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=ALL
    export VLLM_LOGGING_LEVEL=DEBUG
    export FI_LOG_LEVEL=info
    export TORCH_DISTRIBUTED_DEBUG=INFO
    export TORCH_FR_BUFFER_SIZE=1000
    # NCCL flight recorder: dumps in-flight collectives on timeout
    export TORCH_NCCL_ENABLE_MONITORING=1
    export TORCH_NCCL_DUMP_ON_TIMEOUT=1
    export TORCH_NCCL_TRACE_CPP_STACK=1
    DEBUG_LOG_DIR="${WORK_DIR:-$PWD}/logs/debug_${SLURM_JOB_ID}"
    mkdir -p "$DEBUG_LOG_DIR"
    export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="$DEBUG_LOG_DIR/nccl_trace"
    echo "[DEBUG] Debug mode enabled — NCCL flight recorder to $DEBUG_LOG_DIR"
fi

# ── Per-model container environment variables ────────────────────────
# Singularity passes through host env by default, so exporting here
# makes these available inside the container without --env flags.
if [ -n "${CONTAINER_ENV:-}" ]; then
    IFS=';' read -ra _ENVS <<< "$CONTAINER_ENV"
    for _env_pair in "${_ENVS[@]}"; do
        export "$_env_pair"
    done
    unset _ENVS _env_pair
fi

# ── RPC timeout allow for large transfers  ────────────────────────────
# export VLLM_RPC_TIMEOUT=60000 # 60 seconds

# ── Multi-node container support (Slingshot NCCL) ────────────────────
module load brics/apptainer-multi-node
# module load brics/nccl

# ── Resolve head node ─────────────────────────────────────────────────
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
WORKER_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tail -n+2)
HEAD_IP=$(dig +short "$HEAD_NODE")

# ── Ray configuration ─────────────────────────────────────────────────
# Give workers more time to register with GCS (default is too short for
# multi-node on Slingshot).
export RAY_raylet_start_wait_time_s=120
export RAY_CGRAPH_get_timeout=900
RAY_PORT=6378
export RAY_ADDRESS="$HEAD_IP:$RAY_PORT"
EXPECTED_NODES=$SLURM_NNODES
GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-4}
EXPECTED_GPUS=$(( EXPECTED_NODES * GPUS_PER_NODE ))

# ── Increase file descriptor limit ───────────────────────────────────
ulimit -n 65536 2>/dev/null || ulimit -n 16384 2>/dev/null || true

# ── Per-node host tmp dir on RAM-backed tmpfs ─────────────────────────
# Isolate from other jobs; used as /tmp inside container
HOST_TMP="/tmp/vllm_${SLURM_JOB_ID}"
mkdir -p "$HOST_TMP"

# ── Singularity wrapper script ───────────────────────────────────────
# Written as an executable script (not a function) so srun can execve() it
# on remote nodes via shared Lustre filesystem.
#
# Build bind mounts:
# HOST_TMP → /tmp inside container so caches are isolated per job
# /lib64 + /usr/lib64 are needed for Triton JIT compilation - GCC's linker scripts
# reference /lib64/libc.so.6 and /usr/lib64/libc_nonshared.a
# Note: brics/apptainer-multi-node sets SINGULARITY_BINDPATH for Slingshot libs
# --contain uses a minimal /dev, so bind the full host /dev for CXI + GDRCopy access
BIND_MOUNTS="$SCRATCHDIR:$SCRATCHDIR,$HOST_TMP:/tmp"
BIND_MOUNTS="$BIND_MOUNTS,/lib64:/lib64:ro,/usr/lib64:/usr/lib64:ro"
BIND_MOUNTS="$BIND_MOUNTS,/etc/ssl:/etc/ssl:ro,/var/lib/ca-certificates:/var/lib/ca-certificates:ro"
BIND_MOUNTS="$BIND_MOUNTS,/dev:/dev"
BIND_MOUNTS="$BIND_MOUNTS,$PROJECTDIR:$PROJECTDIR"

# --contain disables auto-mounting of $HOME and CWD for filesystem isolation
# --pid creates a PID namespace so killing singularity kills all children
# /host/adapt.sh binds required host dependencies (Slingshot/NCCL)
# See: https://docs.isambard.ac.uk/user-documentation/guides/containers/apptainer-multi-node/
# Note: --contain does NOT clean env vars, so host variables from .env
# (HF_HUB_CACHE, SCRATCHDIR, etc.) are still passed through automatically.
#
# JIT cache dirs (Triton, FlashInfer, torchinductor, torch extensions) stay on
# RAM-backed /tmp — they generate many small files during compilation and would
# bottleneck on Lustre MDS. They don't need to persist across jobs.

echo "Using container: $CONTAINER"
SRUN_EXEC="$JOB_TMP/srun_exec.sh"
cat > "$SRUN_EXEC" << WRAPPER
#!/bin/bash
mkdir -p "$HOST_TMP"
exec singularity exec --nv --pid --contain \\
    --bind "$BIND_MOUNTS" \\
    --env TRITON_CACHE_DIR="/tmp/.triton" \\
    --env FLASHINFER_CACHE_DIR="/tmp/.flashinfer" \\
    --env FLASHINFER_DISABLE_VERSION_CHECK=1 \\
    --env TORCH_EXTENSIONS_DIR="/tmp/.torch_ext" \\
    --env TORCH_HOME="/tmp/.torch" \\
    --env TORCHINDUCTOR_CACHE_DIR="/tmp/.torchinductor" \\
    --env SSL_CERT_FILE="/var/lib/ca-certificates/ca-bundle.pem" \\
    --env REQUESTS_CA_BUNDLE="/var/lib/ca-certificates/ca-bundle.pem" \\
    "$CONTAINER" \\
    bash -c '
        source /host/adapt.sh
        # ── Override adapt.sh defaults for GH200 topology ──
        export NCCL_CROSS_NIC=1
        # Use container-built aws-ofi-nccl (matched to our CUDA) over host version
        export LD_LIBRARY_PATH="/opt/aws-ofi-nccl/lib:\$LD_LIBRARY_PATH"
        exec "\$@"
    ' _ "\$@"
WRAPPER
chmod +x "$SRUN_EXEC"

# ── Top-level cleanup ─────────────────────────────────────────────────
cleanup_job() {
    pkill -TERM -P $$ 2>/dev/null || true
    sleep 2
    pkill -KILL -P $$ 2>/dev/null || true
    rm -rf "$JOB_TMP" 2>/dev/null || true
    rm -rf /tmp/vllm_${SLURM_JOB_ID} 2>/dev/null || true
}
trap cleanup_job EXIT TERM INT

# ── Start Ray worker nodes (staggered to avoid GCS overload) ─────────
# Workers use `ray start --block` which retries until the head is reachable,
# so they are started before the user script (which starts Ray head).
for WORKER in $WORKER_NODES; do
    WORKER_IP=$(dig +short "$WORKER")
    echo "Starting Ray worker $WORKER ($WORKER_IP)..."
    srun --nodelist "$WORKER" --nodes=1 --gpus=$GPUS_PER_NODE \
        --ntasks-per-node=1 "$SRUN_EXEC" \
        bash -c "export VLLM_HOST_IP=$WORKER_IP RAY_raylet_start_wait_time_s=$RAY_raylet_start_wait_time_s; \
        ray start --block \
        --address=$HEAD_IP:$RAY_PORT --node-ip-address=$WORKER_IP" &
    sleep 10  # stagger worker starts to avoid GCS registration storm
done

# ── Write benchmark config to a file (avoids shell-quoting issues) ────
CONFIG_FILE="$JOB_TMP/benchmark_config.json"
if [ -n "${BENCHMARK_CONFIG:-}" ]; then
    printf '%s' "$BENCHMARK_CONFIG" > "$CONFIG_FILE"
fi

# ── Run user script on head node (inside container) ──────────────────
# The user script is responsible for starting Ray head and any application
# logic. It runs in the foreground; when it exits, the job is done.
echo "Running user script on $HEAD_NODE ($HEAD_IP)..."
srun --nodelist "$HEAD_NODE" --nodes=1 --gpus=$GPUS_PER_NODE \
    --ntasks-per-node=1 "$SRUN_EXEC" \
    bash -c "
        export VLLM_HOST_IP=$HEAD_IP RAY_raylet_start_wait_time_s=$RAY_raylet_start_wait_time_s
        # Install to Lustre scratch (RAM-backed /tmp is too small for large dep trees)
        export PYTHONUSERBASE=$JOB_TMP/pip_user
        export PATH=\$PYTHONUSERBASE/bin:\$PATH
        if [ -n \"${PACKAGES:-}\" ]; then
            # shellcheck disable=SC2086
            pip install --cache-dir $JOB_TMP/pip_cache $PACKAGES
        fi
        cd $WORK_DIR
        python $USER_SCRIPT --config-file $CONFIG_FILE
    "