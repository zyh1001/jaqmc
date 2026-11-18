# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Synchronize, submit, inspect, and download the two final Hall training runs."""

import argparse
import copy
import datetime
import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

GEOMETRIES = ("sphere", "disk")
PROJECT = Path(__file__).resolve().parents[1]
REMOTE_PROJECT = "/home/jqguo/project/jaqmc_merge"
REMOTE_RESULTS = "/home/jqguo/project/yuhang/runs"
EXCLUDES = (
    ".git/",
    ".venv/",
    "__pycache__/",
    "*_cache/",
    ".cache/",
    ".backups/",
    "/runs/",
    "/Runs/",
    "/data/",
    "/reports/",
    "/logs/",
    "/checkpoints/",
    "/integration_test_results/",
    "/docs/_build/",
    "*.out",
    "*.err",
    "*.errs",
    "*.npz",
    "*.npy",
    "*.h5",
)


def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    """Run an argument vector without a local shell.

    Returns:
        The completed subprocess result.
    """
    return subprocess.run(args, text=True, check=True, **kwargs)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def normalize_config(
    original: dict[str, Any],
    geometry: str,
    output: str,
    *,
    steps: int | None = None,
    chunk_size: int | None = None,
) -> dict[str, Any]:
    """Apply the requested settings and the known historical field mappings.

    Returns:
        A new configuration preserving the other historical settings.

    Raises:
        ValueError: A historical setting cannot be migrated without changing it.
    """
    config = copy.deepcopy(original)
    config["wf"]["num_layers"] = 4
    config["train"]["grads"]["clip_method"] = "iqr"
    config["train"]["optim"]["module"] = "jaqmc.optimizer.kfac"
    config["train"]["optim"]["learning_rate"]["warmup"] = 200
    config["workflow"]["save_path"] = output
    if config["workflow"]["restore_path"] is not None:
        raise ValueError("Fresh training requires workflow.restore_path=null")
    if geometry == "sphere":
        if config["wf"].pop("flux_per_elec") != 0:
            raise ValueError("This migration requires sphere flux_per_elec=0")
        config["wf"]["dressing"] = {"degree": 0}
        potential = config["estimators"]["energy"]["potential"]
        if potential.pop("interaction_type") != "coulomb":
            raise ValueError("This migration requires the Coulomb interaction")
    else:
        config["system"]["module"] = "jaqmc.app.hall.config:HallDiskConfig"
    if steps is not None:
        config["train"]["run"]["iterations"] = steps
    if chunk_size is not None:
        config["estimators"]["energy"]["kinetic"]["vmap_chunk_size"] = chunk_size
        config["train"]["grads"]["vmap_chunk_size"] = chunk_size
        if geometry == "sphere":
            config["estimators"]["angular_momentum"]["vmap_chunk_size"] = chunk_size
    return config


def flatten(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result = {}
    for key, value in config.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, name))
        else:
            result[name] = value
    return result


def resources(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "submission/resources.json"
    return (
        json.loads(path.read_text())
        if path.exists()
        else {"gpus": 4, "geometries": list(GEOMETRIES), "time_limit": "0"}
    )


def prepare(
    local_run: Path,
    remote_run: str,
    *,
    steps: int | None = None,
    chunk_size: int | None = None,
    gpus: int = 4,
    geometries: list[str] | None = None,
    time_limit: str = "0",
) -> None:
    submission = local_run / "submission"
    submission.mkdir(parents=True, exist_ok=True)
    selected = list(GEOMETRIES) if geometries is None else geometries
    options = dict(gpus=gpus, geometries=selected, time_limit=time_limit)
    options_path = submission / "resources.json"
    if options_path.exists() and json.loads(options_path.read_text()) != options:
        raise ValueError("Existing run resources differ; use a new run identifier")
    write_json(options_path, options)
    sources = {
        "sphere": PROJECT.parent / "data/spherical/mhpo-1det/train/train_config.yaml",
        "disk": PROJECT.parent / "data/disk/n6_Q18_kapa1_8192_final/train_config.yaml",
    }
    changes = {}
    for geometry, source in sources.items():
        if geometry not in selected:
            continue
        original = yaml.safe_load(source.read_text())
        config = normalize_config(
            original,
            geometry,
            f"{remote_run}/{geometry}",
            steps=steps,
            chunk_size=chunk_size,
        )
        original_copy = submission / f"{geometry}_original.yaml"
        target = submission / f"{geometry}.yaml"
        if target.exists():
            if (
                yaml.safe_load(target.read_text()) != config
                or original_copy.read_bytes() != source.read_bytes()
            ):
                raise ValueError(
                    "Existing run configuration differs; use a new run identifier"
                )
        else:
            original_copy.write_bytes(source.read_bytes())
            target.write_text(yaml.safe_dump(config, sort_keys=False))
        before, after = flatten(original), flatten(config)
        changes[geometry] = {
            key: {
                "before": before.get(key, "<absent>"),
                "after": after.get(key, "<absent>"),
            }
            for key in sorted(before.keys() | after.keys())
            if before.get(key) != after.get(key)
        }
    write_json(submission / "config_diff.json", changes)
    # Cover the training source and the scripts used by these two jobs.
    code = sorted((PROJECT / "src").rglob("*.py"))
    code += [
        PROJECT / "scripts" / name
        for name in (
            "retrain_final.sh",
            "train_final.sh",
            "retrain_final.py",
            "train_final_distributed.py",
        )
    ]
    code += [PROJECT / "pyproject.toml", PROJECT / "uv.lock"]
    (submission / "code.sha256").write_text(
        "".join(f"{digest(p)}  {p.relative_to(PROJECT)}\n" for p in code)
    )
    (submission / "configs.sha256").write_text(
        "".join(
            f"{digest(p)}  {p.name}\n"
            for p in sorted(
                submission / name
                for geometry in selected
                for name in (f"{geometry}.yaml", f"{geometry}_original.yaml")
            )
        )
    )


def preflight(project: Path, run_dir: Path) -> None:
    """Resolve both complete configurations and compare every supplied field.

    Raises:
        ValueError: A resolved field differs from the requested configuration.
    """
    for geometry in resources(run_dir)["geometries"]:
        config = run_dir / "submission" / f"{geometry}.yaml"
        command = [
            "bash",
            str(project / "scripts/train_final.sh"),
            geometry,
            str(config),
            str(run_dir / geometry),
            "--dry-run",
        ]
        result = run(
            command,
            cwd=project,
            capture_output=True,
            env={**os.environ, "PROJECT_ROOT": str(project), "JAX_PLATFORMS": "cpu"},
        )
        log = result.stdout + result.stderr
        (run_dir / "submission" / f"{geometry}_preflight.log").write_text(log)
        if "Resolved configurations:" not in log:
            raise ValueError(f"No resolved configuration for {geometry}")
        resolved = yaml.safe_load(log.split("Resolved configurations:", 1)[1])
        expected, actual = (
            flatten(yaml.safe_load(config.read_text())),
            flatten(resolved),
        )
        for key, value in expected.items():
            if key not in actual or actual[key] != value:
                raise ValueError(
                    f"{geometry}: resolved {key} differs: "
                    f"{value!r} != {actual.get(key)!r}"
                )
        (run_dir / "submission" / f"{geometry}_resolved.yaml").write_text(
            yaml.safe_dump(resolved, sort_keys=False)
        )
        print(f"Configuration verified: {geometry}", file=sys.stderr)


def read_jobs(run_dir: Path) -> dict[str, str]:
    path = run_dir / "submission/jobs.tsv"
    return (
        dict(line.split("\t") for line in path.read_text().splitlines())
        if path.exists()
        else {}
    )


def remote_submit(run_dir: Path) -> None:
    with (run_dir / "submission/submit.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        jobs = read_jobs(run_dir)
        options = resources(run_dir)
        if all(geometry in jobs for geometry in options["geometries"]):
            print(json.dumps(jobs))
            return
        for filename, cwd in (
            ("code.sha256", PROJECT),
            ("configs.sha256", run_dir / "submission"),
        ):
            result = run(
                ["sha256sum", "-c", str(run_dir / "submission" / filename)],
                cwd=cwd,
                capture_output=True,
            )
            (run_dir / "submission" / f"{filename}.verified.log").write_text(
                result.stdout
            )
        dependency_check = run(
            [
                str(Path.home() / ".local/bin/uv"),
                "pip",
                "check",
                "--python",
                str(PROJECT / ".venv/bin/python"),
            ],
            capture_output=True,
        )
        (run_dir / "submission/dependencies.log").write_text(
            dependency_check.stdout + dependency_check.stderr
        )
        run(["bash", "-n", str(PROJECT / "scripts/train_final.sh")])
        preflight(PROJECT, run_dir)
        environment = {
            **os.environ,
            "PROJECT_ROOT": str(PROJECT),
            "SLURM_EXPORT_ENV": "ALL",
            "GPUS_PER_JOB": str(options["gpus"]),
        }
        for name in ("SBATCH_GET_USER_ENV", "SLURM_GET_USER_ENV"):
            environment.pop(name, None)
        commands = {}
        nodes = options["gpus"] // 4
        for geometry in options["geometries"]:
            if geometry in jobs:
                continue
            (run_dir / geometry).mkdir(exist_ok=True)
            commands[geometry] = [
                "sbatch",
                "--parsable",
                "--export=ALL",
                "--partition=gpu",
                f"--nodes={nodes}",
                f"--ntasks={nodes}",
                "--ntasks-per-node=1",
                f"--cpus-per-task={12 // nodes}",
                "--gres=gpu:4",
                f"--time={options['time_limit']}",
                f"--job-name=final-{geometry}-{run_dir.name}",
                f"--chdir={PROJECT}",
                f"--output={run_dir}/logs/{geometry}_%j.out",
                f"--error={run_dir}/logs/{geometry}_%j.err",
                str(PROJECT / "scripts/train_final.sh"),
                geometry,
                str(run_dir / "submission" / f"{geometry}.yaml"),
                str(run_dir / geometry),
            ]
            run(
                [commands[geometry][0], "--test-only", *commands[geometry][1:]],
                env=environment,
            )
        for geometry, command in commands.items():
            job = (
                run(command, capture_output=True, env=environment)
                .stdout.strip()
                .split(";")[0]
            )
            if not job.isdigit():
                raise ValueError(f"Invalid Slurm job identifier: {job!r}")
            with (run_dir / "submission/jobs.tsv").open("a") as stream:
                stream.write(f"{geometry}\t{job}\n")
                stream.flush()
                os.fsync(stream.fileno())
            print(f"Submitted {geometry}: {job}", file=sys.stderr)
            jobs[geometry] = job
        print(json.dumps(jobs))


def remote_status(run_dir: Path) -> dict[str, Any]:
    jobs = read_jobs(run_dir)
    states: dict[str, list[str]] = {}
    if jobs:
        accounting = run(
            [
                "sacct",
                "-X",
                "-j",
                ",".join(jobs.values()),
                "--noheader",
                "--parsable2",
                "--format=JobIDRaw,State,ExitCode,Elapsed,AllocCPUS,AllocTRES",
            ],
            capture_output=True,
        )
        for line in accounting.stdout.splitlines():
            fields = line.split("|")
            states[fields[0]] = fields[1:7]
        queued = run(
            ["squeue", "--noheader", "--me", "--format=%i|%T"],
            capture_output=True,
        )
        for line in queued.stdout.splitlines():
            job, state = line.split("|", 1)
            if job in jobs.values():
                states[job] = [state, *states.get(job, ["", "", "", "", ""])[1:]]
    result: dict[str, Any] = {}
    for geometry in resources(run_dir)["geometries"]:
        job_id = jobs.get(geometry, "")
        fields = states.get(
            job_id, ["UNKNOWN" if job_id else "NOT_SUBMITTED", "", "", "", ""]
        )
        marker = run_dir / geometry / "training_complete.json"
        completion = json.loads(marker.read_text()) if marker.exists() else None
        result[geometry] = dict(
            job=job_id,
            state=fields[0],
            exit_code=fields[1],
            elapsed=fields[2],
            cpus=fields[3],
            resources=fields[4],
            completion=completion,
        )
    return result


def complete(geometry: str, output: Path) -> None:
    import numpy as np

    config = yaml.safe_load((output / "train_config.yaml").read_text())
    step = config["train"]["run"]["iterations"] - 1
    checkpoint = output / f"train_ckpt_{step:06d}.npz"
    with np.load(checkpoint) as data:
        if int(data["step"].item()) != step:
            raise ValueError("Final checkpoint has the wrong training step")
    write_json(
        output / "training_complete.json",
        dict(
            geometry=geometry,
            step=step,
            checkpoint=checkpoint.name,
            size=checkpoint.stat().st_size,
            sha256=digest(checkpoint),
            job=os.environ["SLURM_JOB_ID"],
        ),
    )


def ssh(host: str, args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=20",
            host,
            shlex.join(args),
        ],
        **kwargs,
    )


def submit(
    host: str,
    remote_project: str,
    remote_run: str,
    local_run: Path,
    *,
    steps: int | None = None,
    chunk_size: int | None = None,
    gpus: int = 4,
    geometries: list[str] | None = None,
    time_limit: str = "0",
) -> None:
    local_run.mkdir(parents=True, exist_ok=True)
    with (local_run / ".submit.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        prepare(
            local_run,
            remote_run,
            steps=steps,
            chunk_size=chunk_size,
            gpus=gpus,
            geometries=geometries,
            time_limit=time_limit,
        )
        preflight(PROJECT, local_run)
        print("Synchronizing project to ParaCloud...", flush=True)
        ssh(
            host,
            [
                "mkdir",
                "-p",
                remote_project,
                f"{remote_run}/submission",
                f"{remote_run}/logs",
            ],
        )
        sync = [
            "rsync",
            "-az",
            "--itemize-changes",
            "--backup",
            f"--backup-dir={remote_project}/../.jaqmc_merge-sync-backups/{local_run.name}",
        ]
        for pattern in EXCLUDES:
            sync += ["--exclude", pattern]
        sync += [f"{PROJECT}/", f"{host}:{remote_project}/"]
        for dry in (True, False):
            result = run(
                [*sync[:1], *(["--dry-run"] if dry else []), *sync[1:]],
                capture_output=True,
            )
            (
                local_run / "submission" / ("sync_preview.log" if dry else "sync.log")
            ).write_text(result.stdout + result.stderr)
        run(
            [
                "rsync",
                "-az",
                f"{local_run}/submission/",
                f"{host}:{remote_run}/submission/",
            ]
        )
        print("Checking uploaded files and submitting jobs...", flush=True)
        try:
            ssh(
                host,
                [
                    f"{remote_project}/.venv/bin/python",
                    f"{remote_project}/scripts/retrain_final.py",
                    "remote-submit",
                    remote_run,
                ],
            )
        finally:
            run(
                [
                    "rsync",
                    "-az",
                    f"{host}:{remote_run}/submission/",
                    f"{local_run}/submission/",
                ]
            )


def download(
    host: str, remote_run: str, local_run: Path, status: dict[str, Any]
) -> None:
    local_run.mkdir(parents=True, exist_ok=True)
    for directory in ("submission", "logs"):
        run(
            [
                "rsync",
                "-az",
                "--partial",
                f"{host}:{remote_run}/{directory}/",
                f"{local_run}/{directory}/",
            ]
        )
    incomplete = []
    for geometry, value in status.items():
        marker = value["completion"]
        if (
            value["state"] != "COMPLETED"
            or value["exit_code"] != "0:0"
            or marker is None
        ):
            incomplete.append(geometry)
            print(
                f"Not downloaded as completed training: {geometry} ({value['state']})",
                file=sys.stderr,
            )
            continue
        run(
            [
                "rsync",
                "-az",
                "--partial",
                "--append-verify",
                f"{host}:{remote_run}/{geometry}/",
                f"{local_run}/{geometry}/",
            ]
        )
        checkpoint = local_run / geometry / marker["checkpoint"]
        if (
            checkpoint.stat().st_size != marker["size"]
            or digest(checkpoint) != marker["sha256"]
        ):
            raise ValueError(f"Downloaded checkpoint verification failed: {geometry}")
        print(f"Downloaded and verified: {geometry}")
    if incomplete:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "submit",
            "status",
            "download",
            "remote-submit",
            "remote-status",
            "complete",
        ),
    )
    parser.add_argument("identifier", nargs="?")
    parser.add_argument("output", nargs="?")
    parser.add_argument(
        "--steps", type=int, help="Override iterations for a short validation"
    )
    parser.add_argument("--chunk-size", type=int, help="Walkers evaluated per chunk")
    parser.add_argument("--gpus", type=int, choices=(4, 8), default=4)
    parser.add_argument("--geometry", choices=GEOMETRIES, action="append")
    parser.add_argument(
        "--time-limit", default="0", help="Slurm wall time; 0 is unlimited"
    )
    args = parser.parse_args()
    if any(value is not None and value <= 0 for value in (args.steps, args.chunk_size)):
        parser.error("--steps and --chunk-size must be positive")
    if args.command == "complete":
        complete(args.identifier, Path(args.output))
        return
    if args.command.startswith("remote-"):
        directory = Path(args.identifier)
        if args.command == "remote-submit":
            remote_submit(directory)
        else:
            print(json.dumps(remote_status(directory)))
        return
    identifier = args.identifier
    if identifier is None and args.command == "submit":
        identifier = "final_" + datetime.datetime.now(datetime.UTC).strftime(
            "%Y%m%dT%H%M%SZ"
        )
    if not identifier or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", identifier):
        parser.error(
            "Run identifiers may contain letters, digits, underscores, and hyphens"
        )
    host = os.environ.get("PARACLOUD_HOST", "paracloud")
    remote_project = os.environ.get("REMOTE_PROJECT", REMOTE_PROJECT)
    remote_root = os.environ.get("RESULT_ROOT", REMOTE_RESULTS)
    remote_run = f"{remote_root}/{identifier}"
    local_run = (
        Path(os.environ.get("LOCAL_RESULT_ROOT", str(PROJECT.parent / "data/final")))
        / identifier
    )
    remote_python = f"{remote_project}/.venv/bin/python"
    remote_helper = f"{remote_project}/scripts/retrain_final.py"
    print(f"Run: {identifier}\nRemote: {remote_run}\nLocal: {local_run}", flush=True)
    if args.command == "submit":
        submit(
            host,
            remote_project,
            remote_run,
            local_run,
            steps=args.steps,
            chunk_size=args.chunk_size,
            gpus=args.gpus,
            geometries=args.geometry,
            time_limit=args.time_limit,
        )
        return
    status = json.loads(
        ssh(
            host,
            [remote_python, remote_helper, "remote-status", remote_run],
            capture_output=True,
        ).stdout
    )
    for geometry, value in status.items():
        print(
            f"{geometry}: job={value['job']} state={value['state']} "
            f"exit={value['exit_code']} CPUs={value['cpus']} {value['resources']}"
        )
    if args.command == "status":
        return
    download(host, remote_run, local_run, status)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(error.stdout or "", file=sys.stderr)
        print(error.stderr or "", file=sys.stderr)
        raise SystemExit(error.returncode) from error
