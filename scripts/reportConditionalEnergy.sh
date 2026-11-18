#!/bin/bash
#SBATCH --job-name=conditional-report
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=p1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=conditional_report_%j.out
#SBATCH --error=conditional_report_%j.errs

set -euo pipefail
# Slurm executes a copy of this script from the job working directory.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
else
    PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
fi
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/conditional-report-${SLURM_JOB_ID:-local}"
PLOT_PYTHON="${PLOT_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
exec "$PLOT_PYTHON" -u analyzing/reportConditionalEnergy.py "$@"
