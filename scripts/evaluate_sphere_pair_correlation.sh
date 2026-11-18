#!/bin/bash
#SBATCH --job-name=sphere-pair
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/home/jqguo/project/jaqmc_merge}}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/../data}"
RESULT_ROOT="${RESULT_ROOT:-/home/jqguo/project/yuhang/runs}"
RUN_ID="${RUN_ID:?Set the result run identifier}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$PROJECT_ROOT"
.venv/bin/python -c 'import jax; d=jax.devices(); print("Devices:", d, flush=True); assert len(d)==2 and all(x.platform=="gpu" for x in d)'
exec .venv/bin/python scripts/sphere_pair_correlation.py run \
    --data-root "$DATA_ROOT" --output "$RESULT_ROOT/$RUN_ID" \
    --batch-size 4096 --steps 10000 --burn-in 1000 "$@"
