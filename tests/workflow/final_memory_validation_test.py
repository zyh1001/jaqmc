# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Verify the conditional GPU fallback and validation of numerical results."""

import copy
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest


@pytest.fixture
def controller(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "memory_validation", scripts / "validate_final_memory.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("oom", [False, True])
def test_only_oom_uses_eight_gpus(
    controller: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    oom: bool,
) -> None:
    four = {
        "sphere": {"state": "COMPLETED", "valid": True, "oom": False},
        "disk": {"state": "FAILED", "valid": False, "oom": oom},
    }
    submitted = []
    eight = {"disk": {"state": "RUNNING", "valid": False, "oom": False}}
    monkeypatch.setattr(
        controller,
        "probe",
        lambda name: copy.deepcopy(eight if name.endswith("_8gpu") else four),
    )
    monkeypatch.setattr(
        controller.training, "submit", lambda *args, **kwargs: submitted.append(kwargs)
    )
    report = {}
    path = tmp_path / "report.json"
    controller.advance("run", tmp_path, path, report)
    if oom:
        assert len(submitted) == 1
        assert submitted[0] == {
            "steps": 20,
            "chunk_size": 128,
            "gpus": 8,
            "geometries": ["disk"],
            "time_limit": "01:00:00",
        }
        assert report["phase"] == "eight_gpu"
        eight["disk"].update(state="COMPLETED", valid=True)
        controller.advance("run", tmp_path, path, report)
        assert len(submitted) == 1
        assert report["result"] == "PASSED"
    else:
        assert not submitted
        assert report["result"] == "FAILED"


@pytest.mark.parametrize("finite", [False, True])
def test_completed_job_requires_finite_twenty_step_output(
    controller: ModuleType, tmp_path: Path, finite: bool
) -> None:
    output = tmp_path / "sphere"
    output.mkdir()
    checkpoint = output / "train_ckpt_000019.npz"
    np.savez(checkpoint, step=19)
    with (output / "train_stats.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "total_energy_real", "total_energy_real_var"])
        for step in range(20):
            writer.writerow([step, -1.0 if finite else float("nan"), 0.01])
    marker = {
        "step": 19,
        "checkpoint": checkpoint.name,
        "size": checkpoint.stat().st_size,
    }
    status = {
        "sphere": {
            "job": "101",
            "state": "COMPLETED",
            "exit_code": "0:0",
            "completion": marker,
        }
    }
    helper = tmp_path / "helper.py"
    helper.write_text("def remote_status(root):\n    return " + repr(status) + "\n")
    script = controller.PROBE.replace("__RUN_ROOT__", json.dumps(str(tmp_path)))
    script = script.replace("__HELPER__", json.dumps(str(helper)))
    result = subprocess.run(
        [sys.executable, "-c", script], text=True, capture_output=True, check=True
    )
    value = json.loads(result.stdout)["sphere"]
    assert value["recorded_steps"] == 20
    assert value["valid"] is finite


def test_four_gpu_success_does_not_submit_eight(
    controller: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status = {
        g: {"state": "COMPLETED", "valid": True, "oom": False}
        for g in ("sphere", "disk")
    }
    monkeypatch.setattr(controller, "probe", lambda name: status)

    def reject_submission(*args: Any, **kwargs: Any) -> None:
        pytest.fail("A successful four-GPU validation must not launch another job")

    monkeypatch.setattr(controller.training, "submit", reject_submission)
    report = {}
    controller.advance("run", tmp_path, tmp_path / "report.json", report)
    assert report["result"] == "PASSED"
