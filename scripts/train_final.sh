#!/bin/bash
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --time=0
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}"
geometry="${1:?Specify sphere or disk}"
config="${2:?Specify the training configuration}"
output="${3:?Specify the training output directory}"
case "$geometry" in sphere|disk) ;; *) exit 2 ;; esac
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
if [[ "${4:-}" == --dry-run ]]; then
    export JAX_PLATFORMS=cpu
    exec "$PROJECT_ROOT/.venv/bin/jaqmc" hall train --dry-run --yml "$config"
fi
: "${SLURM_JOB_ID:?Submit training with sbatch}"
export JAX_PLATFORMS=cuda
printf 'Job: %s\nGeometry: %s\nConfig: %s\nOutput: %s\nCPUs: %s\n' \
    "$SLURM_JOB_ID" "$geometry" "$config" "$output" "$SLURM_CPUS_PER_TASK"
# A nonempty directory could make the workflow restore an earlier checkpoint.
if [[ -d "$output" && -n "$(ls -A -- "$output")" ]]; then
    printf 'Fresh training requires an empty output directory: %s\n' "$output" >&2
    exit 1
fi
sha256sum "$config"
sha256sum --check --quiet "$(dirname -- "$config")/code.sha256"
(
    cd -- "$(dirname -- "$config")"
    sha256sum --check --quiet configs.sha256
)
if [[ "${GPUS_PER_JOB:-4}" == 8 ]]; then
    unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
    mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
    coordinator="${nodes[0]}:$((20000 + SLURM_JOB_ID % 20000))"
    export JAX_LOCAL_DEVICE_IDS=0,1,2,3
    srun --kill-on-bad-exit=1 --nodes=2 --ntasks=2 --ntasks-per-node=1 \
        "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/scripts/train_final_distributed.py" \
        "$config" "$coordinator"
else
    "$PROJECT_ROOT/.venv/bin/python" -c 'import jax; devices = jax.devices(); print("Devices:", devices, flush=True); assert len(devices) == 4 and all(d.platform == "gpu" for d in devices), "Four GPUs are required"'
    "$PROJECT_ROOT/.venv/bin/jaqmc" hall train --yml "$config"
fi
"$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/scripts/retrain_final.py" complete "$geometry" "$output"
