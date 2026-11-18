#!/bin/bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/../data}"
RESULT_ROOT="${RESULT_ROOT:-/home/jqguo/project/yuhang/runs}"
RUN_ID="${RUN_ID:-hall_fubini_$(date -u +%Y%m%dT%H%M%SZ)}"
export PROJECT_ROOT DATA_ROOT RESULT_ROOT RUN_ID
mkdir -p "$RESULT_ROOT/$RUN_ID/logs"
exec 9>"$RESULT_ROOT/$RUN_ID/submission.lock"
flock -n 9
declare -A submitted=()
if [[ -f "$RESULT_ROOT/$RUN_ID/jobs.tsv" ]]; then
    while IFS=$'\t' read -r geometry job_id; do
        submitted["$geometry"]="$job_id"
    done < "$RESULT_ROOT/$RUN_ID/jobs.tsv"
fi
cd "$PROJECT_ROOT"
for geometry in disk sphere torus; do
    if [[ -n "${submitted[$geometry]:-}" ]]; then
        printf '%s\t%s\n' "$geometry" "${submitted[$geometry]}"
        continue
    fi
    job_id=$(sbatch --parsable --export=ALL --chdir="$PROJECT_ROOT" \
        --job-name="fubini-$geometry" \
        --output="$RESULT_ROOT/$RUN_ID/logs/${geometry}_%j.out" \
        --error="$RESULT_ROOT/$RUN_ID/logs/${geometry}_%j.err" \
        "$PROJECT_ROOT/scripts/evaluate_hall_fubini.sh" "$geometry")
    printf '%s\t%s\n' "$geometry" "$job_id" | tee -a "$RESULT_ROOT/$RUN_ID/jobs.tsv"
done
printf 'Results: %s\n' "$RESULT_ROOT/$RUN_ID"
