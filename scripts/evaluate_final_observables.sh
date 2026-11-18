#!/bin/bash
#SBATCH --job-name=final-observables
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/jqguo/project/jaqmc_merge}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export SLURM_EXPORT_ENV=ALL
cd "$PROJECT_ROOT"
exec .venv/bin/python scripts/final_observables.py worker "$@"
