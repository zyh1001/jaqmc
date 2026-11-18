# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Download completed cluster evaluations, verify them, and update the dataset."""

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

from hall_evaluation import SOURCES, save_json, sha256
from merge_hall_evaluation import import_results


def remote_status(remote_root: str) -> dict:
    """Read completion markers and scheduler states without changing the cluster.

    Returns:
        Completion flags and accounting states for the recorded jobs.
    """
    code = (
        "from pathlib import Path; import json,subprocess; "
        f"root=Path({remote_root!r}); "
        "jobs=(root/'jobs.tsv').read_text().splitlines(); "
        "ids=[line.split()[1].split(';')[0] for line in jobs]; "
        "account=subprocess.run(['sacct','-j',','.join(ids),'-n','-X',"
        "'--format=JobIDRaw,State','--parsable2'],"
        "capture_output=True,text=True,check=True); "
        "print(json.dumps({'complete':{g:(root/g/'completion.json').exists() "
        "for g in ['disk','sphere','torus']},"
        "'accounting':account.stdout}))"
    )
    result = subprocess.run(
        [
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            "paracloud",
            "python3 -c " + shlex.quote(code),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return json.loads(result.stdout)


def collect(remote_root: str, local_root: Path) -> None:
    """Download results and check every recorded output checksum.

    Raises:
        ValueError: If downloaded result checksums differ from the cluster.
    """
    local_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "rsync",
            "-az",
            "--partial",
            "-e",
            "ssh -T -o BatchMode=yes -o ConnectTimeout=15",
            "paracloud:" + remote_root.rstrip("/") + "/",
            str(local_root) + "/",
        ],
        check=True,
        timeout=3600,
    )
    for geometry in SOURCES:
        root = local_root / geometry
        for line in (
            (root / "result_checksums.sha256").read_text(encoding="utf-8").splitlines()
        ):
            expected, relative = line.split("  ", 1)
            if sha256(root / relative) != expected:
                raise ValueError(f"Downloaded checksum mismatch: {geometry}/{relative}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--merge-root", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    args.local_root.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            status = remote_status(args.remote_root)
            save_json(args.local_root / "collection_status.json", status)
            print(json.dumps(status), flush=True)
            if all(status["complete"].values()):
                collect(args.remote_root, args.local_root)
                report = import_results(args.local_root, args.merge_root)
                save_json(args.local_root / "collection_complete.json", report)
                return
            failed = any(
                word in status["accounting"]
                for word in (
                    "FAILED",
                    "CANCELLED",
                    "TIMEOUT",
                    "OUT_OF_MEMORY",
                    "NODE_FAIL",
                    "PREEMPTED",
                )
            )
            if failed:
                print(
                    "A cluster job stopped without complete results. Check job logs.",
                    flush=True,
                )
                sys.exit(1)
        except (subprocess.SubprocessError, OSError, json.JSONDecodeError) as error:
            print(str(error), flush=True)
        if not args.watch:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
