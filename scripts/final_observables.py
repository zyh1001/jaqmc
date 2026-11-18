# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Archive final Hall training and manage fixed-network evaluation jobs."""

import argparse
import copy
import fcntl
import json
import os
import shlex
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from hall_evaluation import (
    PROJECT,
    SOURCES,
    build_config,
    prepare_checkpoint,
    run_evaluation,
    save_json,
    sha256,
)

ROOT = Path("/home/ubuntu/work/final_data")
BASELINE = Path("/home/ubuntu/work/data/merge")
TRAIN_RUN = "final_layer4_allnull_8gpu_20261116T144748Z"
LOCAL_TRAIN = Path("/home/ubuntu/work/data/final") / TRAIN_RUN
REMOTE_PROJECT = "/home/jqguo/project/jaqmc_merge"
REMOTE_TRAIN = "/home/jqguo/project/yuhang/runs/" + TRAIN_RUN
REMOTE_RUNS = "/home/jqguo/project/yuhang/runs"
TASKS = [
    ("disk", "regular"),
    ("disk", "scan"),
    ("disk", "fubini"),
    ("sphere", "regular"),
    ("sphere", "pair"),
    ("sphere", "fubini"),
]
FINAL_STEPS = {"disk": 149999, "sphere": 199999}
FAILED = {
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
}


def execute(command: Any, **kwargs) -> subprocess.CompletedProcess[str]:
    """Run a command without interpreting arguments as shell code.

    Returns:
        The verified result or requested metadata.
    """
    return subprocess.run(command, check=True, text=True, **kwargs)


def ssh(command: Any) -> str:
    """Execute one quoted command on the configured cluster.

    Returns:
        The verified result or requested metadata.

    Raises:
        RuntimeError: If the requested input or result fails validation.
    """
    try:
        return execute(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=20",
                "paracloud",
                command,
            ],
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise RuntimeError(error.stderr or error.stdout or str(error)) from error


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def state_path() -> Path:
    return ROOT / "workflow.json"


def state() -> dict[str, Any]:
    return load(state_path())


def checksums(directory: Path) -> dict[str, Any]:
    return {
        str(p.relative_to(directory)): {"sha256": sha256(p), "size": p.stat().st_size}
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def archive() -> None:
    """Verify every downloaded training file before moving its directory.

    Raises:
        FileExistsError: If the requested input or result fails validation.
        ValueError: If the requested input or result fails validation.
    """
    ROOT.mkdir(parents=True, exist_ok=True)
    origins = ROOT / "archive_manifest.json"
    if origins.exists():
        for geo, record in load(origins)["training"].items():
            dest = ROOT / "raw" / geo / "train"
            for name, info in record["files"].items():
                p = dest / name
                if (
                    not p.is_file()
                    or p.stat().st_size != info["size"]
                    or sha256(p) != info["sha256"]
                ):
                    raise ValueError(f"Archived training file changed: {p}")
        print("Training archive verified", flush=True)
        return
    remote_program = (
        "from pathlib import Path\nimport hashlib,json\n"
        f"r=Path({REMOTE_TRAIN!r})\nout={{}}\n"
        'for g in ["disk","sphere"]:\n'
        " d=r/g\n out[g]={}\n"
        ' for p in sorted(d.rglob("*")):\n'
        "  if p.is_file():\n"
        '   with p.open("rb") as f: h=hashlib.file_digest(f,"sha256").hexdigest()\n'
        '   out[g][str(p.relative_to(d))]={"sha256":h,"size":p.stat().st_size}\n'
        "print(json.dumps(out))\n"
    )
    remote = json.loads(
        ssh(REMOTE_PROJECT + "/.venv/bin/python -c " + shlex.quote(remote_program))
    )
    records = {}
    import pandas as pd

    for geo, expected in FINAL_STEPS.items():
        src = LOCAL_TRAIN / geo
        dest = ROOT / "raw" / geo / "train"
        actual = checksums(src)
        if actual != remote[geo]:
            raise ValueError(f"Full training file verification failed: {geo}")
        ckpts = list(src.glob("train_ckpt_*.npz"))
        final = src / f"train_ckpt_{expected:06d}.npz"
        with np.load(final, allow_pickle=False) as z:
            assert int(z["step"]) == expected
            assert len(z["batched_data/data/electrons"]) == 8192
        frame = pd.read_csv(src / "train_stats.csv")
        np.testing.assert_array_equal(frame["step"], np.arange(expected + 1))
        assert np.isfinite(frame["total_energy_real"]).all()
        cfg = yaml.safe_load((src / "train_config.yaml").read_text())
        assert cfg["wf"]["num_layers"] == 4
        assert cfg["train"]["optim"]["learning_rate"]["warmup"] == 200
        if dest.exists():
            raise FileExistsError(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dest)
        for folder in ["logs", "submission"]:
            shutil.copytree(LOCAL_TRAIN / folder, dest / folder)
        for folder in ["evaluation", "fubini"]:
            (dest.parent / folder).mkdir(exist_ok=True)
        records[geo] = dict(
            original_local=str(src),
            local=str(dest),
            remote=REMOTE_TRAIN + "/" + geo,
            files=actual,
            checkpoints=len(ckpts),
            training_rows=len(frame),
        )
    save_json(
        origins, dict(training_run=TRAIN_RUN, verified_at=timestamp(), training=records)
    )
    baseline = BASELINE / "unified.h5"
    torus = ROOT / "raw/torus"
    torus.mkdir(exist_ok=True)
    save_json(
        torus / "source_manifest.json",
        dict(
            source=str(baseline),
            source_sha256=sha256(baseline),
            group="/torus",
            reused_without_recalculation=True,
        ),
    )
    print(
        "Training folders archived with full remote checksum verification", flush=True
    )


def source(geo: str, train_dir: Path) -> Path:
    return train_dir / Path(SOURCES[geo]).name


def scan_config(
    train: dict[str, Any], output: Path, prepared: Any, steps: int, burn_in: int
) -> dict[str, Any]:
    config = build_config(train, "disk", output, prepared, 4096, steps, burn_in, 161803)
    config["estimators"] = {
        "enabled": {
            k: k == "wavefunction_scan"
            for k in [
                "energy",
                "angular_momentum",
                "density",
                "pair_correlation",
                "structure_factor",
                "one_rdm",
                "wavefunction_scan",
            ]
        },
        "wavefunction_scan": dict(
            fixed_electrons=None,
            mobile_electron=0,
            x_min=-8.0,
            x_max=8.0,
            y_min=-8.0,
            y_max=8.0,
            n_x=301,
            n_y=301,
            chunk_size=256,
            selection_samples=8192,
            probe_electron=1,
            radial_direction=[1.0, 0.0],
            radial_min=0.001,
            radial_max=3.0,
            radial_points=200,
        ),
    }
    config["writers"]["console"]["fields"] = "scan=wavefunction_scan:selected_log_abs"
    return config


def validate_scan(directory: Path, config: dict[str, Any], original: Path) -> None:
    from final_dataset import scan_postprocess, verify_parameters

    verify_parameters(
        original,
        directory / f"evaluation_ckpt_{config['run']['iterations'] - 1:06d}.npz",
        4096,
    )
    with np.load(directory / "evaluation_digest.npz") as z:
        digest = dict(z)
    if int(digest["wavefunction_scan:selection_samples"]) != 8192:
        raise ValueError("The scan did not inspect exactly 8192 configurations")
    scan_postprocess(digest, directory / "analysis_outputs")
    save_json(
        directory / "validation.json",
        dict(
            passed=True,
            parameters_unchanged=True,
            actual_batch_size=4096,
            selection_samples=8192,
        ),
    )


def worker(args: argparse.Namespace) -> None:
    """Run one fixed-resource job with an isolated short validation first.

    Raises:
        ValueError: If the requested input or result fails validation.
    """
    import jax

    devices = jax.devices()
    print("JAX devices:", devices, flush=True)
    if len(devices) != 4 or not all(d.platform == "gpu" for d in devices):
        raise ValueError("Exactly four GPU devices are required")
    if int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) != 12:
        raise ValueError("Exactly twelve CPU cores must be allocated")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    save_json(
        output / "resources.json",
        dict(
            gpus=len(devices),
            cpus=12,
            job_id=os.environ["SLURM_JOB_ID"],
            devices=[str(d) for d in devices],
            train_dir=str(args.train_dir),
            geometry=args.geometry,
            kind=args.kind,
        ),
    )
    code = {
        str(p.relative_to(PROJECT)): sha256(p)
        for p in sorted((PROJECT / "src/jaqmc").rglob("*.py"))
    }
    code.update(
        {
            str(p.relative_to(PROJECT)): sha256(p)
            for p in sorted((PROJECT / "scripts").glob("*.py"))
        }
    )
    save_json(output / "code_checksums.json", code)
    original = source(args.geometry, args.train_dir)
    common = dict(
        data_root=Path("/"),
        train_dir=args.train_dir,
        output=output,
        geometry=args.geometry,
        batch_size=4096,
        steps=10000,
        burn_in=1000,
        seed=271828,
        smoke=True,
        dry_run=False,
    )
    if args.kind == "regular":
        import hall_evaluation

        hall_evaluation.run(
            argparse.Namespace(
                **common, trained_only=True, disk_spatial=args.geometry == "disk"
            )
        )
        if args.geometry == "disk":
            from final_dataset import disk_spatial_postprocess

            disk_spatial_postprocess(output / "trained")
    elif args.kind == "pair":
        import sphere_pair_correlation

        common["seed"] = 314159
        sphere_pair_correlation.run(argparse.Namespace(**common))
    elif args.kind == "scan":
        train = yaml.safe_load((args.train_dir / "train_config.yaml").read_text())
        inputs = output / "inputs"
        inputs.mkdir(exist_ok=True)
        prepared = inputs / "train_ckpt_000000.npz"
        provenance = prepare_checkpoint(original, prepared, 4096, 161803)
        save_json(output / "provenance.json", provenance)
        for name, burn in [("smoke", 2), ("evaluation", 1000)]:
            directory = output / name
            config = scan_config(train, directory, prepared, 2, burn)
            run_evaluation(config, directory, True)
            run_evaluation(config, directory)
            validate_scan(directory, config, original)
    else:
        import hall_fubini

        common.update(steps=2000, burn_in=100, seed=62119, checkpoint_steps=None)
        smoke = copy.deepcopy(common)
        smoke.update(
            output=output / "smoke",
            steps=3,
            burn_in=2,
            checkpoint_steps=[FINAL_STEPS[args.geometry]],
        )
        hall_fubini.prepare(argparse.Namespace(**smoke))
        hall_fubini.run(smoke["output"], True)
        hall_fubini.run(smoke["output"], False)
        hall_fubini.prepare(argparse.Namespace(**common))
        hall_fubini.run(output, False)
    # Include postprocessing and every raw result, not the checksum file itself.
    manifest = checksums(output)
    manifest.pop("files.json", None)
    manifest.pop("finished.json", None)
    save_json(output / "files.json", manifest)
    save_json(
        output / "finished.json",
        dict(
            passed=True,
            job_id=os.environ["SLURM_JOB_ID"],
            geometry=args.geometry,
            kind=args.kind,
            files_sha256=sha256(output / "files.json"),
        ),
    )


def submit() -> None:
    """Synchronize code and submit exactly one job for each recorded task.

    Raises:
        ValueError: If the requested input or result fails validation.
    """
    if not (ROOT / "archive_manifest.json").exists():
        raise ValueError("Archive the training folders before submitting")
    if state_path().exists():
        record = state()
    else:
        run_id = "final_observables_" + timestamp()
        record = dict(
            run_id=run_id,
            remote_root=REMOTE_RUNS + "/" + run_id,
            training_run=TRAIN_RUN,
            jobs={},
            created_at=timestamp(),
        )
        save_json(state_path(), record)
    submission = ROOT / "submission" / record["run_id"]
    submission.mkdir(parents=True, exist_ok=True)
    with (submission / "submit.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        record = state()
        remote = record["remote_root"]
        ssh(
            "mkdir -p "
            + shlex.quote(remote + "/logs")
            + " "
            + shlex.quote(remote + "/submission")
        )
        exclusions = [
            ".venv",
            ".git",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            "runs",
            "logs",
            "*.log",
            "*.egg-info",
            ".mypy_cache",
            "data",
        ]
        command = [
            "rsync",
            "-az",
            "--backup",
            "--backup-dir=" + remote + "/submission/code_backup",
        ]
        command.extend("--exclude=" + item for item in exclusions)
        command += [str(PROJECT) + "/", "paracloud:" + REMOTE_PROJECT + "/"]
        with (submission / "sync.log").open("w") as log:
            execute(command, stdout=log, stderr=subprocess.STDOUT)
        code = {
            str(p.relative_to(PROJECT)): sha256(p)
            for folder in ["src", "scripts"]
            for p in (PROJECT / folder).rglob("*")
            if p.is_file() and p.suffix in [".py", ".sh"]
        }
        checksum_file = submission / "code.sha256"
        checksum_file.write_text(
            "".join(h + "  " + name + "\n" for name, h in sorted(code.items()))
        )
        execute(
            ["rsync", "-az", str(checksum_file), "paracloud:" + remote + "/submission/"]
        )
        remote_manifest = shlex.quote(remote + "/submission/code.sha256")
        train_root = shlex.quote(REMOTE_TRAIN)
        verify = (
            f"cd {shlex.quote(REMOTE_PROJECT)} && sha256sum -c {remote_manifest} "
            f"&& bash -n scripts/evaluate_final_observables.sh "
            f"&& .venv/bin/python -m compileall -q scripts "
            "&& .venv/bin/python scripts/final_observables.py preflight "
            f"--train-root {train_root} "
            f"--output {shlex.quote(remote + '/submission/preflight_' + timestamp())}"
        )
        (submission / "remote_preflight.log").write_text(ssh(verify))
        script = REMOTE_PROJECT + "/scripts/evaluate_final_observables.sh"
        # Reject an unsupported allocation before submitting any geometry.
        test = ssh(
            "cd "
            + shlex.quote(REMOTE_PROJECT)
            + "; unset SBATCH_GET_USER_ENV SLURM_GET_USER_ENV; "
            "SLURM_EXPORT_ENV=ALL sbatch --test-only --export=ALL "
            + shlex.quote(script)
            + " worker-unused 2>&1"
        )
        (submission / "resource_test.log").write_text(test)
        for geo, kind in TASKS:
            key = geo + "/" + kind
            if key in record["jobs"]:
                continue
            output = remote + "/" + key
            stdout = shlex.quote(remote + f"/logs/{geo}_{kind}_%j.out")
            stderr = shlex.quote(remote + f"/logs/{geo}_{kind}_%j.err")
            train_dir = shlex.quote(REMOTE_TRAIN + "/" + geo)
            command = (
                f"cd {shlex.quote(REMOTE_PROJECT)}; "
                "unset SBATCH_GET_USER_ENV SLURM_GET_USER_ENV; "
                f"export SLURM_EXPORT_ENV=ALL; sbatch --parsable --export=ALL "
                f"--job-name=final-{geo}-{kind} "
                f"--output={stdout} "
                f"--error={stderr} "
                f"{shlex.quote(script)} --geometry {geo} --kind {kind} "
                f"--train-dir {train_dir} --output {shlex.quote(output)}"
            )
            job_id = ssh(command).strip().split(";")[0]
            if not job_id.isdigit():
                raise ValueError("Invalid submission response: " + job_id)
            record["jobs"][key] = dict(
                job_id=job_id, output=output, submitted_at=timestamp()
            )
            save_json(state_path(), record)
            save_json(submission / "jobs.json", record)
            execute(
                [
                    "rsync",
                    "-az",
                    str(submission / "jobs.json"),
                    "paracloud:" + remote + "/submission/",
                ]
            )
            print(key, job_id, flush=True)
        save_json(
            submission / "original_checksums.json",
            dict(
                merge=checksums(BASELINE),
                figures=checksums(Path("/home/ubuntu/work/data/figures/img")),
            ),
        )


def preflight(args: argparse.Namespace) -> None:
    """Resolve final-state configurations with the installed JaQMC version."""
    import hall_evaluation
    import hall_fubini
    import sphere_pair_correlation

    for geo in ["disk", "sphere"]:
        common = dict(
            geometry=geo,
            data_root=Path("/"),
            train_dir=(
                args.train_root / geo / "train"
                if (args.train_root / geo / "train").is_dir()
                else args.train_root / geo
            ),
            output=args.output / geo / "regular",
            batch_size=4096,
            steps=10000,
            burn_in=1000,
            seed=271828,
            smoke=False,
            dry_run=True,
            trained_only=True,
            disk_spatial=geo == "disk",
        )
        hall_evaluation.run(argparse.Namespace(**common))
        if geo == "sphere":
            common.update(output=args.output / geo / "pair")
            sphere_pair_correlation.run(argparse.Namespace(**common))
        common.update(
            output=args.output / geo / "fubini",
            steps=2000,
            burn_in=100,
            checkpoint_steps=[FINAL_STEPS[geo]],
        )
        hall_fubini.prepare(argparse.Namespace(**common))
        hall_fubini.run(common["output"], True)
    train_dir = args.train_root / "disk"
    if (train_dir / "train").is_dir():
        train_dir = train_dir / "train"
    prepared = args.output / "disk/regular/inputs/train_ckpt_000000.npz"
    train = yaml.safe_load((train_dir / "train_config.yaml").read_text())
    out = args.output / "disk/scan"
    run_evaluation(scan_config(train, out, prepared, 2, 1000), out, True)
    print("All six task configurations resolved", flush=True)


def status() -> dict[str, Any]:
    record = state()
    ids = ",".join(v["job_id"] for v in record["jobs"].values())
    text = ssh(
        "sacct -j "
        + ids
        + " -n -P --format=JobIDRaw,State%40,ExitCode,Elapsed,AllocCPUS,AllocTRES%200"
    )
    by_id = {
        row[0]: row[1:]
        for line in text.splitlines()
        if (row := line.split("|")) and row[0].isdigit()
    }
    result = {}
    for key, info in record["jobs"].items():
        fields = by_id.get(info["job_id"], ["UNKNOWN", "", "", "", ""])
        result[key] = dict(
            **info,
            state=fields[0].split()[0],
            exit_code=fields[1],
            elapsed=fields[2],
            cpus=fields[3],
            allocated=fields[4],
        )
    save_json(ROOT / "status.json", dict(checked_at=timestamp(), jobs=result))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def local_task(key: str) -> Path:
    geo, kind = key.split("/")
    return ROOT / "raw" / geo / ("fubini" if kind == "fubini" else "evaluation/" + kind)


def download(jobs: dict[str, Any] | None = None) -> None:
    """Download only completed tasks and verify every produced file.

    Raises:
        ValueError: If the requested input or result fails validation.
    """
    jobs = status() if jobs is None else jobs
    for key, job in jobs.items():
        complete = job["state"] == "COMPLETED" and job["exit_code"] == "0:0"
        if not complete and job["state"] not in FAILED:
            print("Not completed:", key, job["state"], flush=True)
            continue
        dest = local_task(key)
        dest.mkdir(parents=True, exist_ok=True)
        execute(
            [
                "rsync",
                "-az",
                "--partial",
                "--append-verify",
                "paracloud:" + job["output"] + "/",
                str(dest) + "/",
            ]
        )
        if not complete:
            save_json(
                dest / "download_validation.json",
                dict(
                    passed=False,
                    state=job["state"],
                    job_id=job["job_id"],
                    note="Partial failed-job output retained without resubmission",
                ),
            )
            continue
        finished = load(dest / "finished.json")
        if (
            not finished["passed"]
            or sha256(dest / "files.json") != finished["files_sha256"]
        ):
            raise ValueError("Result manifest mismatch: " + key)
        for name, item in load(dest / "files.json").items():
            p = dest / name
            if p.stat().st_size != item["size"] or sha256(p) != item["sha256"]:
                raise ValueError("Downloaded checksum mismatch: " + str(p))
        save_json(
            dest / "download_validation.json",
            dict(passed=True, job_id=job["job_id"], checked_at=timestamp()),
        )
    record = state()
    for folder in ["logs", "submission"]:
        dest = ROOT / folder / record["run_id"]
        dest.mkdir(parents=True, exist_ok=True)
        execute(
            [
                "rsync",
                "-az",
                "--partial",
                "paracloud:" + record["remote_root"] + "/" + folder + "/",
                str(dest) + "/",
            ]
        )


def verify_originals() -> None:
    """Require unchanged source data and original images before publishing.

    Raises:
        ValueError: If an original file differs from its recorded checksum.
    """
    record = load(ROOT / "submission" / state()["run_id"] / "original_checksums.json")
    for name, folder in [
        ("merge", BASELINE),
        ("figures", BASELINE.parent / "figures/img"),
    ]:
        for relative, item in record[name].items():
            if sha256(folder / relative) != item["sha256"]:
                raise ValueError(f"Original file changed: {folder / relative}")


def finish(include_fubini: bool = True) -> None:
    from final_dataset import build, plot

    verify_originals()
    build(ROOT, BASELINE, include_fubini=include_fubini)
    plot(ROOT, BASELINE, include_fubini=include_fubini)
    verify_originals()
    validation = load(ROOT / "implementation_validation.json")
    validation.update(
        formal_evaluations_complete=include_fubini,
        non_fubini_evaluations_complete=True,
        final_figure_pairs=28 if include_fubini else 24,
        final_dataset_sha256=sha256(ROOT / "merge/unified.h5"),
        completed_at=timestamp(),
    )
    save_json(ROOT / "implementation_validation.json", validation)


def monitor() -> None:
    """Collect successful tasks and finish automatically after all six succeed.

    Raises:
        RuntimeError: If the requested input or result fails validation.
    """
    with (ROOT / ".monitor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                jobs = status()
            except (RuntimeError, subprocess.CalledProcessError) as error:
                print(f"Status query failed: {error}", flush=True)
                time.sleep(60)
                continue
            done = (
                all(
                    v["state"] == "COMPLETED" and v["exit_code"] == "0:0"
                    for v in jobs.values()
                )
                and len(jobs) == 6
            )
            terminal = all(v["state"] in FAILED | {"COMPLETED"} for v in jobs.values())
            if done or (terminal and len(jobs) == 6):
                try:
                    download(jobs)
                except subprocess.CalledProcessError as error:
                    print(f"Download interrupted: {error}", flush=True)
                    time.sleep(60)
                    continue
                if not done:
                    raise RuntimeError(
                        "One or more evaluation jobs failed; "
                        "successful tasks were retained"
                    )
                finish()
                save_json(
                    ROOT / "completion.json",
                    dict(
                        passed=True, completed_at=timestamp(), run_id=state()["run_id"]
                    ),
                )
                return
            time.sleep(60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["archive", "submit", "status", "download", "monitor"]:
        commands.add_parser(name)
    finish_parser = commands.add_parser("finish")
    finish_parser.add_argument("--without-fubini", action="store_true")
    work = commands.add_parser("worker")
    work.add_argument("--geometry", choices=["disk", "sphere"], required=True)
    work.add_argument(
        "--kind", choices=["regular", "pair", "scan", "fubini"], required=True
    )
    work.add_argument("--train-dir", type=Path, required=True)
    work.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("preflight")
    check.add_argument("--train-root", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command in ["worker", "preflight"]:
        globals()[args.command](args)
    elif args.command == "finish":
        finish(include_fubini=not args.without_fubini)
    else:
        try:
            globals()[args.command]()
        except Exception as error:
            if args.command == "monitor":
                save_json(
                    ROOT / "workflow_failure.json",
                    dict(
                        failed_at=timestamp(),
                        error_type=type(error).__name__,
                        error=str(error),
                    ),
                )
            raise


if __name__ == "__main__":
    main()
