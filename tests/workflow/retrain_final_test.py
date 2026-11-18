# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Check configuration preservation, submission retries, and result verification."""

import copy
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/retrain_final.py"
spec = importlib.util.spec_from_file_location("retrain_final", SCRIPT)
assert spec is not None and spec.loader is not None
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


@pytest.mark.parametrize("geometry", training.GEOMETRIES)
@pytest.mark.parametrize("chunked", [False, True])
def test_training_config_preserves_other_settings(geometry: str, chunked: bool) -> None:
    original = {
        "workflow": {"save_path": "old", "restore_path": None, "batch_size": 8192},
        "system": {"module": "old", "flux": 15 if geometry == "sphere" else 18},
        "wf": {"num_layers": 2, "flux_per_elec": 0, "ndets": 1},
        "estimators": {
            "energy": {"potential": {"interaction_type": "coulomb"}, "kinetic": {}},
            "angular_momentum": {},
        },
        "train": {
            "run": {"iterations": 200000 if geometry == "sphere" else 150000},
            "grads": {"clip_method": "mad", "clip_scale": 100.0},
            "optim": {
                "module": "old",
                "damping": 0.001,
                "learning_rate": {
                    "rate": 0.05,
                    "decay": 1.0,
                    "delay": 2000 if geometry == "sphere" else 200000,
                    "warmup": 0 if geometry == "sphere" else 1000,
                },
            },
        },
    }
    unchanged = copy.deepcopy(original)
    updated = training.normalize_config(
        original,
        geometry,
        "/new/results",
        steps=20 if chunked else None,
        chunk_size=128 if chunked else None,
    )
    assert original == unchanged
    before, after = training.flatten(original), training.flatten(updated)
    allowed = {
        "wf.num_layers",
        "train.grads.clip_method",
        "train.optim.module",
        "train.optim.learning_rate.warmup",
        "workflow.save_path",
    }
    if geometry == "sphere":
        allowed |= {
            "wf.flux_per_elec",
            "wf.dressing.degree",
            "estimators.energy.potential.interaction_type",
        }
    else:
        allowed.add("system.module")
    if chunked:
        allowed |= {
            "train.run.iterations",
            "train.grads.vmap_chunk_size",
            "estimators.energy.kinetic.vmap_chunk_size",
        }
        if geometry == "sphere":
            allowed.add("estimators.angular_momentum.vmap_chunk_size")
        assert updated["train"]["run"]["iterations"] == 20
        assert updated["train"]["grads"]["vmap_chunk_size"] == 128
    assert all(
        before.get(key) == after.get(key)
        for key in (before.keys() | after.keys()) - allowed
    )
    assert updated["wf"]["num_layers"] == 4
    assert updated["train"]["grads"]["clip_method"] == "iqr"
    assert updated["train"]["optim"]["module"] == "jaqmc.optimizer.kfac"
    assert updated["train"]["optim"]["learning_rate"]["warmup"] == 200
    assert updated["workflow"]["restore_path"] is None


@pytest.mark.parametrize("gpus", [4, 8])
def test_submission_retry_preserves_first_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gpus: int
) -> None:
    (tmp_path / "submission").mkdir()
    (tmp_path / "submission/resources.json").write_text(
        json.dumps({"gpus": gpus, "geometries": ["sphere", "disk"], "time_limit": "0"})
    )
    commands = []
    fail_disk = True

    def execute(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[0] == "sbatch" and "--test-only" not in command:
            geometry = command[-3]
            if geometry == "disk" and fail_disk:
                raise subprocess.CalledProcessError(1, command)
            return subprocess.CompletedProcess(
                command, 0, "101\n" if geometry == "sphere" else "102\n", ""
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(training, "run", execute)
    monkeypatch.setattr(training, "preflight", lambda *args: None)
    with pytest.raises(subprocess.CalledProcessError):
        training.remote_submit(tmp_path)
    assert training.read_jobs(tmp_path) == {"sphere": "101"}
    fail_disk = False
    training.remote_submit(tmp_path)
    assert training.read_jobs(tmp_path) == {"sphere": "101", "disk": "102"}
    submissions = [c for c in commands if c[0] == "sbatch" and "--test-only" not in c]
    assert [c[-3] for c in submissions] == ["sphere", "disk", "disk"]
    for command in submissions:
        nodes = gpus // 4
        assert {
            f"--cpus-per-task={12 // nodes}",
            f"--nodes={nodes}",
            f"--ntasks={nodes}",
            "--ntasks-per-node=1",
            "--gres=gpu:4",
            "--time=0",
        } <= set(command)
    count = len(commands)
    training.remote_submit(tmp_path)
    assert len(commands) == count


def test_download_skips_incomplete_and_rejects_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote, local = tmp_path / "remote", tmp_path / "local"
    for directory in ("submission", "logs", "sphere"):
        (remote / directory).mkdir(parents=True)
    checkpoint = remote / "sphere/train_ckpt_199999.npz"
    np.savez(checkpoint, step=199999)
    (remote / "sphere/train_config.yaml").write_text(
        "train:\n  run:\n    iterations: 200000\n"
    )
    monkeypatch.setenv("SLURM_JOB_ID", "101")
    training.complete("sphere", remote / "sphere")
    marker = json.loads((remote / "sphere/training_complete.json").read_text())
    status = {
        "sphere": {"state": "COMPLETED", "exit_code": "0:0", "completion": marker},
        "disk": {"state": "RUNNING", "exit_code": "0:0", "completion": None},
    }

    def sync(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        source = Path(command[-2].split(":", 1)[1])
        shutil.copytree(source, command[-1], dirs_exist_ok=True)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(training, "run", sync)
    with pytest.raises(SystemExit) as error:
        training.download("cluster", str(remote), local, status)
    assert error.value.code == 1
    assert (local / "sphere" / checkpoint.name).read_bytes() == checkpoint.read_bytes()
    assert not (local / "disk").exists()
    checkpoint.write_bytes(b"damaged")
    with pytest.raises(ValueError, match="verification failed"):
        training.download("cluster", str(remote), local, status)


def test_completion_requires_final_training_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "101")
    np.savez(tmp_path / "train_ckpt_199999.npz", step=1000)
    (tmp_path / "train_config.yaml").write_text(
        "train:\n  run:\n    iterations: 200000\n"
    )
    with pytest.raises(ValueError, match="wrong training step"):
        training.complete("sphere", tmp_path)
    assert not (tmp_path / "training_complete.json").exists()


def test_short_completion_uses_configured_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "101")
    np.savez(tmp_path / "train_ckpt_000019.npz", step=19)
    (tmp_path / "train_config.yaml").write_text(
        yaml.safe_dump({"train": {"run": {"iterations": 20}}})
    )
    training.complete("sphere", tmp_path)
    marker = json.loads((tmp_path / "training_complete.json").read_text())
    assert marker["step"] == 19
    assert marker["checkpoint"] == "train_ckpt_000019.npz"


def test_worker_starts_with_absent_output_and_rejects_old_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    (project / ".venv/bin").mkdir(parents=True)
    (project / "scripts").mkdir()
    worker = project / "scripts/train_final.sh"
    shutil.copyfile(SCRIPT.with_name("train_final.sh"), worker)
    python = project / ".venv/bin/python"
    python.write_text("#!/bin/bash\nexit 0\n")
    python.chmod(0o755)
    cli = project / ".venv/bin/jaqmc"
    cli.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$PROJECT_ROOT/called.txt"\n')
    cli.chmod(0o755)
    submission = tmp_path / "run/submission"
    submission.mkdir(parents=True)
    config = submission / "sphere.yaml"
    config.write_text("workflow: {}\n")
    (submission / "configs.sha256").write_text(
        f"{training.digest(config)}  sphere.yaml\n"
    )
    (submission / "code.sha256").write_text(
        f"{training.digest(worker)}  scripts/train_final.sh\n"
    )
    monkeypatch.setenv("PROJECT_ROOT", str(project))
    monkeypatch.setenv("SLURM_JOB_ID", "101")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "12")
    output = tmp_path / "run/sphere"
    command = ["bash", str(worker), "sphere", str(config), str(output)]
    subprocess.run(command, check=True, capture_output=True)
    assert (project / "called.txt").read_text().splitlines() == [
        "hall",
        "train",
        "--yml",
        str(config),
    ]
    output.mkdir()
    (output / "train_ckpt_000001.npz").write_bytes(b"old")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert "Fresh training requires an empty output directory" in result.stderr
