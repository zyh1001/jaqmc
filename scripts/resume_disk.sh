#!/bin/bash
#SBATCH --job-name=disk-resume
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=v100
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=resume_disk_%j.out
#SBATCH --error=resume_disk_%j.errs

set -euo pipefail
# Slurm executes a copy of this script from the job working directory.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
else
    PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
fi
SOURCE_ROOT="${SOURCE_ROOT:-${DATA_ROOT:-${PROJECT_ROOT}/../data}/disk/n6_Q18_kapa1_8192_final}"
CHECKPOINT="$SOURCE_ROOT/train_ckpt_149999.npz"
CONFIG="$SOURCE_ROOT/train_config.yaml"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export OPENBLAS_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
job_id="${SLURM_JOB_ID:-input_check}"
output="${PROJECT_ROOT}/runs/n6_Q18_kapa1_8192_final_resume_first_order_2000_${job_id}"
args=(--yml "$CONFIG" system.module=jaqmc.app.hall.config:HallDiskConfig
    "workflow.restore_path=$CHECKPOINT" "workflow.save_path=$output"
    train.run.reset_kfac_state_on_restore=true train.run.iterations=152000
    train.run.save_step_interval=1000 train.run.save_time_interval=0)
dry_run=false
extra_args=()
for argument in "$@"; do
    if [[ "$argument" == --dry-run ]]; then
        dry_run=true
    else
        extra_args+=("$argument")
    fi
done
if "$dry_run"; then
    exec "$PROJECT_ROOT/.venv/bin/jaqmc" hall train --dry-run "${args[@]}" "${extra_args[@]}"
fi
: "${SLURM_JOB_ID:?Submit the training job with sbatch}"
printf 'Job: %s\nCheckpoint: %s\nOutput: %s\n' "$SLURM_JOB_ID" "$CHECKPOINT" "$output"
sha256sum "$CHECKPOINT" "$CONFIG"
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu "$PROJECT_ROOT/.venv/bin/pytest" -q \
    tests/utils/runtime_test.py::test_configure_runtime_skips_distributed_init_on_dry_run \
    tests/workflow/vmc_kfac_resume_test.py \
    tests/workflow/disk_jastrow_registration_test.py \
    tests/laplacian/primitives/annotations_test.py
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu bash "$PROJECT_ROOT/scripts/resume_disk.sh" --dry-run "${extra_args[@]}"
exec "$PROJECT_ROOT/.venv/bin/jaqmc" hall train "${args[@]}" "${extra_args[@]}"
