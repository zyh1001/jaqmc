# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Wait for ParaCloud overlaps, download results, update merge, and plot distances."""

import argparse
import json
import os
import shlex
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from hall_evaluation import save_json
from merge_hall_fubini import import_results


def remote_status(remote: str, directory: str) -> dict:
    """Read completed summaries and Slurm job states from the remote run.

    Returns:
        Completion flags, job states, and job identifiers.
    """
    program = f"""
import json,subprocess
from pathlib import Path
p=Path({directory!r})
jobs=dict(line.split() for line in (p/'jobs.tsv').read_text().splitlines())
result=subprocess.run(['sacct','-n','-X','-P','-j',','.join(jobs.values()),
                       '--format=JobIDRaw,State'],check=True,capture_output=True,text=True)
states=dict(line.split('|')[:2] for line in result.stdout.splitlines() if '|' in line)
print(json.dumps({{'jobs':jobs,'states':states,
 'complete':{{g:(p/g/'fubini_summary.json').exists() for g in jobs}}}}))
"""
    command = shlex.join(
        ["/home/jqguo/project/jaqmc_merge/.venv/bin/python", "-B", "-c", program]
    )
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", remote, command],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return json.loads(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default="paracloud")
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--merge-root", type=Path, required=True)
    parser.add_argument("--figures-root", type=Path, required=True)
    parser.add_argument("--plot-python", type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=300)
    args = parser.parse_args()
    if args.interval < 30:
        parser.error("The polling interval must be at least 30 seconds")
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    status_path = run_root / "collection_status.json"
    while True:
        try:
            status = remote_status(args.remote, args.remote_root)
        except (subprocess.SubprocessError, OSError, ValueError) as error:
            status = {"stage": "remote_check_retry", "error": str(error)}
            save_json(status_path, status)
            print(json.dumps(status), flush=True)
            if not args.watch:
                raise
            time.sleep(args.interval)
            continue
        status.update(
            checked_at=datetime.now(UTC).isoformat(), stage="waiting_for_jobs"
        )
        save_json(status_path, status)
        print(json.dumps(status), flush=True)
        if len(status["complete"]) == 3 and all(status["complete"].values()):
            break
        failed = {
            job: state
            for job, state in status["states"].items()
            if state.split()[0]
            in {
                "FAILED",
                "CANCELLED",
                "TIMEOUT",
                "OUT_OF_MEMORY",
                "NODE_FAIL",
                "PREEMPTED",
                "BOOT_FAIL",
                "DEADLINE",
            }
        }
        if failed:
            status.update(stage="job_failed", failed_jobs=failed)
            save_json(status_path, status)
            raise SystemExit(f"Overlap jobs require attention: {failed}")
        if not args.watch:
            return
        time.sleep(args.interval)
    status["stage"] = "downloading"
    save_json(status_path, status)
    subprocess.run(
        [
            "rsync",
            "-rt",
            "--partial",
            "--timeout=120",
            "--backup",
            "--backup-dir=.download_backups/"
            + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
            f"{args.remote}:{args.remote_root.rstrip('/')}/",
            str(run_root) + "/",
        ],
        check=True,
    )
    status["stage"] = "validating_and_importing"
    save_json(status_path, status)
    report = import_results(run_root, args.merge_root.resolve())
    status.update(stage="plotting", merge_report=report)
    save_json(status_path, status)
    figures_root = args.figures_root.resolve()
    plot_python = args.plot_python or figures_root / ".venv/bin/python"
    plot_env = os.environ.copy()
    plot_env.pop("PYTHONPATH", None)
    subprocess.run(
        [
            str(plot_python),
            "-B",
            str(figures_root / "plot_fubini.py"),
            "--merge-file",
            str(args.merge_root.resolve() / "unified.h5"),
            "--output",
            str(figures_root / "img/observables"),
        ],
        check=True,
        env=plot_env,
    )
    status.update(stage="complete", completed_at=datetime.now(UTC).isoformat())
    save_json(status_path, status)


if __name__ == "__main__":
    main()
