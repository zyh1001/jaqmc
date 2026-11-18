#!/bin/bash
#SBATCH --job-name=disk_eval
#SBATCH --partition=v100g32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=disk_eval_%j.out
#SBATCH --error=disk_eval_%j.err

set -euo pipefail

# Slurm executes a copy of this script from the job working directory.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
else
    PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
fi
PYTHON_BIN=${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}
export DISK_ANALYSIS_ROOT="${DISK_ANALYSIS_ROOT:-${PROJECT_ROOT}/../jaqmc}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/disk-eval-mpl-${SLURM_JOB_ID:-$$}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" "${PROJECT_ROOT}/analyzing/evaluateDisk.py" "$@"
