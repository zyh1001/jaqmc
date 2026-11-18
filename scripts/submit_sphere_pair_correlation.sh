#!/bin/bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/../data}"
RESULT_ROOT="${RESULT_ROOT:-/home/jqguo/project/yuhang/runs}"
RUN_ID="${RUN_ID:-sphere_pair_correlation_$(date -u +%Y%m%dT%H%M%SZ)}"
export PROJECT_ROOT DATA_ROOT RESULT_ROOT RUN_ID
unset SBATCH_GET_USER_ENV SLURM_GET_USER_ENV
export SLURM_EXPORT_ENV=ALL
pair_output="$RESULT_ROOT/$RUN_ID"
mkdir -p "$pair_output/logs"
exec 9>"$pair_output/submission.lock"
flock -n 9
if [[ -s "$pair_output/jobs.tsv" ]]; then
    cat "$pair_output/jobs.tsv"
    exit 0
fi
cd "$PROJECT_ROOT"
job_id=$(sbatch --parsable --export=ALL --job-name=sphere-pair \
    --chdir="$PROJECT_ROOT" --output="$pair_output/logs/%j.out" \
    --error="$pair_output/logs/%j.err" \
    "$PROJECT_ROOT/scripts/evaluate_sphere_pair_correlation.sh")
printf 'sphere\t%s\n' "$job_id" > "$pair_output/jobs.tsv.tmp"
mv "$pair_output/jobs.tsv.tmp" "$pair_output/jobs.tsv"
cat "$pair_output/jobs.tsv"
