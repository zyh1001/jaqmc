#!/bin/bash
#SBATCH --job-name=sphere-mhpo-mad
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${TRAIN_CONFIG:?Set TRAIN_CONFIG to the prepared training YAML file}"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python_bin="${PROJECT_ROOT}/.venv/bin/python"
jaqmc_bin="${PROJECT_ROOT}/.venv/bin/jaqmc"
for argument in "$@"; do
    if [[ "$argument" == --dry-run ]]; then
        exec "$jaqmc_bin" hall train --yml "$TRAIN_CONFIG" "$@"
    fi
done
: "${SLURM_JOB_ID:?Submit the training job with sbatch}"
export JAX_PLATFORMS=cuda
printf 'Job: %s\nConfig: %s\n' "$SLURM_JOB_ID" "$TRAIN_CONFIG"
sha256sum "$TRAIN_CONFIG"
"$python_bin" -c 'import jax; devices = jax.devices(); print("Devices:", devices, flush=True); assert len(devices) == 4 and all(d.platform == "gpu" for d in devices), "Four GPUs are required"'
exec "$jaqmc_bin" hall train --yml "$TRAIN_CONFIG" "$@"
