#!/bin/bash
#SBATCH --job-name=disk_eval_repair
#SBATCH --partition=v100g32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=disk_eval_repair_%j.out
#SBATCH --error=disk_eval_repair_%j.err

set -euo pipefail

# Slurm executes a copy of this script from the job working directory.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
else
    PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
fi
RUN_DIR="${1:-${DATA_ROOT:-${PROJECT_ROOT}/../data}/disk/n6_Q18_kapa1_8192_final}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/disk-repair-mpl-${SLURM_JOB_ID:-$$}"
cd "${PROJECT_ROOT}"

JAX_PLATFORMS=cpu "${PROJECT_ROOT}/.venv/bin/python" -m pytest --n-cpu-devices=1 \
    tests/app/hall/disk_evaluation_regression_test.py -q
JAX_PLATFORMS=cpu "${PROJECT_ROOT}/.venv/bin/python" -m pytest --n-cpu-devices=2 \
    tests/app/hall/disk_evaluation_regression_test.py \
    tests/analyzing/disk_evaluation_test.py -q
JAX_PLATFORMS=cpu "${PROJECT_ROOT}/.venv/bin/python" analyzing/recoverDiskPairCorrelation.py \
    "${RUN_DIR}/eval/pair_correlation_r3.0"
JAX_PLATFORMS=cuda bash "${PROJECT_ROOT}/scripts/evaluate_disk.sh" wave_scan "${RUN_DIR}" \
    --save-path "${RUN_DIR}/eval/wavefunction_scan_fixed" "${@:2}"
