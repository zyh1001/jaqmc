#!/bin/bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
RESULT_ROOT="${RESULT_ROOT:-/home/jqguo/project/yuhang/runs}"
RUN_ID="${RUN_ID:-hall_observables_$(date -u +%Y%m%dT%H%M%SZ)}"
export PROJECT_ROOT RESULT_ROOT RUN_ID
mkdir -p "$RESULT_ROOT/$RUN_ID/logs"
exec 9>"$RESULT_ROOT/$RUN_ID/submission.lock"
flock -n 9
declare -A submitted=()
if [[ -f "$RESULT_ROOT/$RUN_ID/jobs.tsv" ]]; then
    while IFS=$'\t' read -r previous_geometry previous_job; do
        submitted["$previous_geometry"]="$previous_job"
    done < "$RESULT_ROOT/$RUN_ID/jobs.tsv"
fi
cd "$PROJECT_ROOT"
for geometry in disk sphere torus; do
    if [[ -n "${submitted[$geometry]:-}" ]]; then
        printf '%s\t%s\n' "$geometry" "${submitted[$geometry]}"
        continue
    fi
    job_id=$(sbatch --parsable --export=ALL --chdir="$PROJECT_ROOT" \
        --job-name="hall-$geometry" \
        --output="$RESULT_ROOT/$RUN_ID/logs/${geometry}_%j.out" \
        --error="$RESULT_ROOT/$RUN_ID/logs/${geometry}_%j.err" \
        "$PROJECT_ROOT/scripts/evaluate_hall_observables.sh" "$geometry")
    printf '%s\t%s\n' "$geometry" "$job_id" | tee -a "$RESULT_ROOT/$RUN_ID/jobs.tsv"
done
printf 'Results: %s\n' "$RESULT_ROOT/$RUN_ID"
