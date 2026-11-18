#!/bin/bash
#SBATCH --job-name=hall-fubini
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/../data}"
RESULT_ROOT="${RESULT_ROOT:-/home/jqguo/project/yuhang/runs}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
geometry="${1:?Specify disk, sphere, or torus}"
shift
run_id="${RUN_ID:?Set RUN_ID to a unique run identifier}"
output="${RESULT_ROOT}/${run_id}/${geometry}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$PROJECT_ROOT"
"$PYTHON_BIN" -c 'import jax; devices=jax.devices(); print("Devices:", devices, flush=True); assert len(devices) == 2 and all(d.platform == "gpu" for d in devices), "Two GPUs are required"'
if [[ ! -f "$output/fubini_plan.json" ]]; then
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/hall_fubini.py" prepare "$geometry" \
        --data-root "$DATA_ROOT" --output "$output" \
        --batch-size 4096 --steps 10000 --burn-in 1000 "$@"
fi
exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/hall_fubini.py" run --output "$output"
