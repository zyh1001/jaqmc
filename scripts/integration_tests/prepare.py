# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Record Git metadata and source hashes before submitting tests."""

import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

root = Path(__file__).resolve().parents[2]
record = {
    "prepared_at": datetime.now(UTC).isoformat(),
    "commit": "unavailable",
    "working_tree": [],
    "version_source": "source_snapshot",
    "source_sha256": {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(
            [
                *(
                    path
                    for folder in ("src", "tests", "scripts")
                    for path in (root / folder).rglob("*")
                    if path.is_file() and path.suffix in (".py", ".sh", ".gz")
                ),
                root / "pyproject.toml",
                root / "uv.lock",
                root / "integration_sources.json",
            ]
        )
    },
}
if (root / ".git").exists() and shutil.which("git"):
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if revision.returncode == 0:
        record["commit"] = revision.stdout.strip()
        record["version_source"] = "git_and_source_snapshot"
        record["working_tree"] = subprocess.check_output(
            ["git", "--no-optional-locks", "status", "--short"], cwd=root, text=True
        ).splitlines()
output = root / "integration_test_results/prepared_revision.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(record, indent=2) + "\n")
print(f"Prepared test revision: {record['commit']}")
print(f"Revision record: {output}")
