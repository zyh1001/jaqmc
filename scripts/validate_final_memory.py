# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Follow four-GPU validation and use eight GPUs only after a confirmed OOM."""

import argparse
import fcntl
import json
import os
import re
import subprocess
import time
from pathlib import Path

import retrain_final as training

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}
PROBE = r"""
import csv
import importlib.util
import json
import math
from pathlib import Path
import numpy as np
root = Path(__RUN_ROOT__)
spec = importlib.util.spec_from_file_location("training", __HELPER__)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
status = module.remote_status(root)
for geometry, value in status.items():
    output = root / geometry
    error = root / "logs" / f"{geometry}_{value['job']}.err"
    text = error.read_text(errors="replace") if error.exists() else ""
    value["oom"] = value["state"] == "OUT_OF_MEMORY" or (
        "RESOURCE_EXHAUSTED" in text and "out of memory" in text.lower())
    value["log_tail"] = text.splitlines()[-5:]
    value["valid"] = False
    table = output / "train_stats.csv"
    with table.open() if table.exists() else open('/dev/null') as stream:
        rows = list(csv.DictReader(stream))
    value["recorded_steps"] = len(rows)
    if value["state"] == "COMPLETED" and value["exit_code"] == "0:0":
        try:
            assert len(rows) == 20, f"Expected 20 rows, found {len(rows)}"
            assert [int(row["step"]) for row in rows] == list(range(20))
            for key in ("total_energy_real", "total_energy_real_var"):
                assert all(math.isfinite(float(row[key])) for row in rows), key
            marker = value["completion"]
            assert marker and marker["step"] == 19, "Missing final completion marker"
            checkpoint = output / marker["checkpoint"]
            assert checkpoint.stat().st_size == marker["size"]
            with np.load(checkpoint) as data:
                assert int(data["step"].item()) == 19
            value["valid"] = True
            value["last_energy"] = float(rows[-1]["total_energy_real"])
            value["last_variance"] = float(rows[-1]["total_energy_real_var"])
        except (AssertionError, KeyError, ValueError, OSError) as error:
            value["validation_error"] = str(error)
print(json.dumps(status))
"""


def probe(run_id: str) -> dict:
    project = os.environ.get("REMOTE_PROJECT", training.REMOTE_PROJECT)
    root = os.environ.get("RESULT_ROOT", training.REMOTE_RESULTS)
    script = PROBE.replace("__RUN_ROOT__", json.dumps(f"{root}/{run_id}"))
    script = script.replace(
        "__HELPER__", json.dumps(f"{project}/scripts/retrain_final.py")
    )
    result = training.ssh(
        os.environ.get("PARACLOUD_HOST", "paracloud"),
        [f"{project}/.venv/bin/python", "-"],
        input=script,
        capture_output=True,
    )
    return json.loads(result.stdout)


def finished(status: dict) -> bool:
    return all(
        value["state"].split()[0].rstrip("+") in TERMINAL for value in status.values()
    )


def advance(run_id: str, local_root: Path, report_path: Path, report: dict) -> None:
    four = probe(run_id)
    report.update(four_gpu=four, controller_pid=os.getpid(), phase="four_gpu")
    if finished(four):
        failures = [
            g for g, value in four.items() if value["oom"] and not value["valid"]
        ]
        final = dict(four)
        if failures:
            fallback_id = run_id + "_8gpu"
            report.update(
                phase="eight_gpu",
                fallback_run=fallback_id,
                fallback_geometries=failures,
            )
            training.write_json(report_path, report)
            if not report.get("fallback_submitted"):
                print(
                    f"Confirmed OOM for {failures}; submitting {fallback_id}",
                    flush=True,
                )
                root = os.environ.get("RESULT_ROOT", training.REMOTE_RESULTS)
                training.submit(
                    os.environ.get("PARACLOUD_HOST", "paracloud"),
                    os.environ.get("REMOTE_PROJECT", training.REMOTE_PROJECT),
                    f"{root}/{fallback_id}",
                    local_root / fallback_id,
                    steps=20,
                    chunk_size=128,
                    gpus=8,
                    geometries=failures,
                    time_limit="01:00:00",
                )
                report["fallback_submitted"] = True
                training.write_json(report_path, report)
            eight = probe(fallback_id)
            report["eight_gpu"] = eight
            final.update(eight)
            done = finished(eight)
        else:
            done = True
        if done:
            report.update(
                phase="finished",
                final=final,
                result="PASSED"
                if all(v["valid"] for v in final.values())
                else "FAILED",
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_id) or args.interval < 5:
        parser.error(
            "Use a valid run identifier and an interval of at least five seconds"
        )
    local_root = Path(
        os.environ.get("LOCAL_RESULT_ROOT", str(training.PROJECT.parent / "data/final"))
    )
    report_path = local_root / args.run_id / "submission/validation_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    with (report_path.parent / "validation_controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.write(str(os.getpid()) + "\n")
        lock.flush()
        previous = None
        while True:
            try:
                advance(args.run_id, local_root, report_path, report)
            except (
                subprocess.CalledProcessError,
                OSError,
                json.JSONDecodeError,
            ) as error:
                report["last_error"] = str(error)
                training.write_json(report_path, report)
                print(f"Validation check will retry: {error}", flush=True)
                time.sleep(args.interval)
                continue
            four = report["four_gpu"]
            training.write_json(report_path, report)
            brief = {
                "phase": report["phase"],
                "four_gpu": {
                    g: {
                        k: v.get(k) for k in ("job", "state", "recorded_steps", "valid")
                    }
                    for g, v in four.items()
                },
            }
            if "eight_gpu" in report:
                brief["eight_gpu"] = report["eight_gpu"]
            current = json.dumps(brief)
            if current != previous:
                print(current, flush=True)
                previous = current
            if report["phase"] == "finished":
                print(
                    f"Validation result: {report['result']}; report: {report_path}",
                    flush=True,
                )
                if report["result"] != "PASSED":
                    raise SystemExit(1)
                return
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
