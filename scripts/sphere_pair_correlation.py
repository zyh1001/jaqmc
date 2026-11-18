# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Evaluate, collect, and merge repaired spherical pair correlations."""

import argparse
import fcntl
import json
import os
import shlex
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import yaml
from hall_evaluation import (
    PROJECT,
    SOURCES,
    blocking,
    build_config,
    integrated_autocorrelation,
    prepare_checkpoint,
    run_evaluation,
    save_json,
    sha256,
)
from merge_hall_evaluation import make_writer


def bin_geometry(bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Return double-precision angular edges and sphere-area probabilities."""
    from jaqmc.app.hall.estimator.sphere.pair_correlation import angular_bin_geometry

    return angular_bin_geometry(bins)


def inspect_stage(directory: Path, config: dict, source: Path) -> tuple[dict, dict]:
    """Validate exact parameter restoration and reconstruct integer pair counts.

    Returns:
        Numerical arrays and a validation summary.

    Raises:
        ValueError: If parameters, counts, or normalization are invalid.
    """
    steps = config["run"]["iterations"]
    batch = config["workflow"]["batch_size"]
    nelec = sum(config["system"]["nspins"])
    bins = config["estimators"]["pair_correlation"]["bins"]
    checkpoint = directory / f"evaluation_ckpt_{steps - 1:06d}.npz"
    with np.load(source, allow_pickle=False) as original, np.load(checkpoint) as saved:
        keys = {k for k in original.files if k.startswith("params/")}
        if keys != {k for k in saved.files if k.startswith("params/")} or not all(
            np.array_equal(original[k], saved[k]) for k in keys
        ):
            raise ValueError("Wavefunction parameters changed")
        if (
            int(saved["step"]) != steps - 1
            or saved["batched_data/data/electrons"].shape[0] != batch
        ):
            raise ValueError("Unexpected checkpoint step or walker count")
    with h5py.File(directory / "evaluation_stats.h5") as data:
        means = data["pair_correlation:counts_per_configuration"][:].astype(np.float64)
        pair_count = data["pair_correlation:pair_count"][:]
    if means.shape != (steps, bins) or not np.isfinite(means).all():
        raise ValueError("Incomplete or nonfinite count records")
    scaled = means * batch
    counts = np.rint(scaled).astype(np.int64)
    pairs_per_step = batch * nelec * (nelec - 1) // 2
    if (
        np.max(abs(scaled - counts)) > 0.01
        or (counts < 0).any()
        or not (counts.sum(axis=1) == pairs_per_step).all()
        or not (pair_count == nelec * (nelec - 1) // 2).all()
    ):
        raise ValueError("Pair counts are not complete nonnegative integers")
    edges, probability = bin_geometry(bins)
    total = counts.sum(axis=0, dtype=np.int64)
    samples = steps * batch
    value = 2 * total / (samples * nelec**2 * probability)
    factor = nelec / (nelec - 1)
    if not np.isclose(np.dot(value, probability), (nelec - 1) / nelec, atol=1e-12):
        raise ValueError("Pair-correlation integral is not normalized")
    with np.load(directory / "evaluation_digest.npz") as digest:
        np.testing.assert_allclose(
            digest["pair_correlation"], value, rtol=4e-6, atol=1e-8
        )
    arrays = dict(
        theta_edges=edges,
        theta=(edges[:-1] + edges[1:]) / 2,
        bin_probability=probability,
        raw_counts=total,
        counts_by_step=counts,
        value=value,
        g_finite_n=value * factor,
    )
    summary = dict(
        passed=True,
        steps=steps,
        actual_batch_size=batch,
        sample_count=samples,
        electron_number=nelec,
        total_pair_count=int(total.sum()),
        parameters_unchanged=True,
        finite_endpoints=True,
        normalization_integral=float(np.dot(value, probability)),
    )
    if steps >= 1000:
        series = 2 * counts / (batch * nelec**2 * probability)
        result = blocking(series)
        tested = np.stack(
            [blocking(series, b)["standard_error"] for b in (50, 100, 200, 500)]
        )
        positive = np.where(tested > 1e-14, tested, np.nan)
        lower = np.min(np.where(np.isnan(positive), np.inf, positive), axis=0)
        upper = np.max(np.where(np.isnan(positive), 0, positive), axis=0)
        stable = upper <= 1.5 * lower
        arrays.update(
            standard_error=result["standard_error"],
            block_means=result["block_means"],
            g_finite_n_standard_error=result["standard_error"] * factor,
            tested_block_sizes=np.array([50, 100, 200, 500]),
            tested_standard_errors=tested,
            error_stable=stable,
        )
        summary.update(
            block_size=100,
            block_count=steps // 100,
            provisional_error_bins=np.flatnonzero(~stable).tolist(),
            endpoint_value=float(value[-1]),
            endpoint_standard_error=float(result["standard_error"][-1]),
            endpoint_autocorrelation_steps=integrated_autocorrelation(series[:, -1]),
        )
    return arrays, summary


def pair_config(
    train: dict,
    output: Path,
    source: Path,
    args: argparse.Namespace,
    steps: int,
    burn_in: int,
) -> dict:
    """Build a sphere evaluation with only the repaired pair estimator enabled.

    Returns:
        A configuration accepted by the native Hall evaluation workflow.
    """
    config = build_config(
        train, "sphere", output, source, args.batch_size, steps, burn_in, args.seed
    )
    config["estimators"] = {
        "enabled": {
            "energy": False,
            "angular_momentum": False,
            "one_rdm": False,
            "structure_factor": False,
            "density": False,
            "pair_correlation": True,
        },
        "pair_correlation": {"bins": 200, "vmap_chunk_size": 128},
    }
    config["writers"]["console"]["fields"] = "pairs=pair_correlation:pair_count:.0f"
    return config


def run(args: argparse.Namespace) -> None:
    """Evaluate the requested fixed training state and record full provenance."""
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = (
        args.train_dir.resolve() / Path(SOURCES["sphere"]).name
        if getattr(args, "train_dir", None)
        else args.data_root.resolve() / SOURCES["sphere"]
    )
    train = yaml.safe_load((source.parent / "train_config.yaml").read_text())
    inputs = output / "inputs"
    inputs.mkdir(exist_ok=True)
    prepared = inputs / "train_ckpt_000000.npz"
    provenance = prepare_checkpoint(source, prepared, args.batch_size, args.seed)
    evaluation = output / "evaluation"
    config = pair_config(train, evaluation, prepared, args, args.steps, args.burn_in)
    plan = dict(
        geometry="sphere",
        input=provenance,
        config=config,
        source_config_sha256=sha256(source.parent / "train_config.yaml"),
        job_id=os.environ.get("SLURM_JOB_ID"),
        normalization="2*C/(M*N^2*p)",
    )
    save_json(output / "pair_plan.json", plan)
    code = {
        str(p.relative_to(PROJECT)): sha256(p)
        for p in sorted((PROJECT / "src/jaqmc").rglob("*.py"))
    }
    for filename in ("sphere_pair_correlation.py", "hall_evaluation.py"):
        code["scripts/" + filename] = sha256(PROJECT / "scripts" / filename)
    save_json(output / "code_checksums.json", code)
    if args.dry_run:
        run_evaluation(config, evaluation, True)
        return
    smoke_dir = output / "smoke"
    smoke = pair_config(train, smoke_dir, prepared, args, 3, 2)
    if not (smoke_dir / "validation.json").exists():
        run_evaluation(smoke, smoke_dir)
        _, valid = inspect_stage(smoke_dir, smoke, source)
        save_json(smoke_dir / "validation.json", valid)
    run_evaluation(config, evaluation)
    arrays, summary = inspect_stage(evaluation, config, source)
    np.savez_compressed(output / "pair_statistics.npz", **arrays)
    save_json(evaluation / "validation.json", summary)
    save_json(output / "pair_summary.json", summary)
    paths = [
        output / "pair_plan.json",
        output / "code_checksums.json",
        output / "pair_statistics.npz",
        output / "pair_summary.json",
    ]
    for subdir in (inputs, smoke_dir, evaluation):
        paths.extend(p for p in sorted(subdir.rglob("*")) if p.is_file())
    (output / "result_checksums.sha256").write_text(
        "".join(sha256(p) + "  " + str(p.relative_to(output)) + "\n" for p in paths)
    )
    save_json(output / "completion.json", summary)


def import_result(run_root: Path, merge_root: Path) -> dict:
    """Verify downloaded results, back up merge, and replace only sphere pair data.

    Returns:
        The verification report and backup location.

    Raises:
        ValueError: If an input identity or downloaded checksum differs.
    """
    for line in (run_root / "result_checksums.sha256").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        if sha256(run_root / relative) != expected:
            raise ValueError("Downloaded checksum mismatch: " + relative)
    manifest = json.loads((merge_root / "manifest.json").read_text())
    source = Path(manifest["source_root"]) / SOURCES["sphere"]
    plan = json.loads((run_root / "pair_plan.json").read_text())
    if sha256(source) != plan["input"]["source_sha256"]:
        raise ValueError("Source training checkpoint differs")
    if not json.loads((run_root / "completion.json").read_text())["passed"]:
        raise ValueError("Incomplete pair-correlation evaluation")
    arrays, summary = inspect_stage(run_root / "evaluation", plan["config"], source)
    with np.load(run_root / "pair_statistics.npz") as saved:
        for key, value in arrays.items():
            np.testing.assert_array_equal(saved[key], value)
    # This lock is shared with the other additive dataset importer.
    with (merge_root / ".fubini-import.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return publish_pair(run_root, merge_root, arrays, summary, plan)


def publish_pair(
    run_root: Path, merge_root: Path, arrays: dict, summary: dict, plan: dict
) -> dict:
    """Publish validated pair records while retaining previous data and sources.

    Returns:
        Dataset checksum, backup location, and sampling checks.
    """
    manifest = json.loads((merge_root / "manifest.json").read_text())
    schema = json.loads((merge_root / "schema.json").read_text())
    identity = sha256(run_root / "pair_statistics.npz")
    with h5py.File(merge_root / "unified.h5") as h5:
        if h5["sphere/pair_correlation"].attrs.get("statistics_sha256") == identity:
            return json.loads(
                (merge_root / "pair_correlation_validation.json").read_text()
            )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    backup = merge_root.with_name(merge_root.name + "_pair_backup_" + stamp)
    shutil.copytree(
        merge_root, backup, ignore=shutil.ignore_patterns("*.lock", "*.tmp.h5")
    )
    temporary = merge_root / "unified.pair.tmp.h5"
    shutil.copy2(merge_root / "unified.h5", temporary)
    ids = []
    files = [
        "pair_plan.json",
        "pair_summary.json",
        "pair_statistics.npz",
        "code_checksums.json",
        "evaluation/evaluation_stats.h5",
        "evaluation/evaluation_digest.npz",
        "evaluation/evaluation_config.yaml",
        "evaluation/validation.json",
        "result_checksums.sha256",
    ]
    for relative in files:
        path = run_root / relative
        entry = dict(
            file_id=f"F{len(manifest['files']) + 1:04d}",
            geometry="sphere",
            category="repaired_pair_correlation",
            path=os.path.relpath(path, manifest["source_root"]),
            sha256=sha256(path),
            size_bytes=path.stat().st_size,
        )
        manifest["files"].append(entry)
        ids.append(entry["file_id"])
    radius = plan["config"]["system"].get("radius") or np.sqrt(
        plan["config"]["system"]["flux"] / 2
    )
    with h5py.File(temporary, "r+") as h5:
        group = h5["sphere/pair_correlation"]
        if "legacy_weighted" not in group:
            legacy = group.create_group("legacy_weighted")
            h5.copy(group["radial/raw_counts"], legacy, name="raw_counts")
            legacy.attrs.update(
                status="needs_review", note="原始加权统计，端点存在异常，仅供来源追溯"
            )
            schema["fields"]["pair_correlation/legacy_weighted/raw_counts"] = {
                "description": "原始异常加权累计值，仅供来源追溯",
                "dtype": str(legacy["raw_counts"].dtype),
                "unit": "",
                "kind": "array",
            }
        put = make_writer(h5, schema, {"sphere": ids})
        values = {
            "radial/edges": arrays["theta_edges"],
            "radial/coordinate": arrays["theta"],
            "radial/coordinate_name": "两电子夹角，弧度",
            "radial/bin_probability": arrays["bin_probability"],
            "radial/bin_measure": 4 * np.pi * radius**2 * arrays["bin_probability"],
            "radial/raw_counts": arrays["raw_counts"],
            "radial/value": arrays["value"],
            "radial/standard_error": arrays["standard_error"],
            "radial/g_finite_n": arrays["g_finite_n"],
            "radial/g_finite_n_standard_error": arrays["g_finite_n_standard_error"],
            "radial/valid_mask": np.ones(len(arrays["value"]), dtype=bool),
            "radial/normalization": (
                "g=2*C/(M*N^2*p); p=(cos(theta_left)-cos(theta_right))/2"
            ),
            "counts_by_step": arrays["counts_by_step"],
            "block_means": arrays["block_means"],
            "tested_block_sizes": arrays["tested_block_sizes"],
            "tested_standard_errors": arrays["tested_standard_errors"],
            "error_stable": arrays["error_stable"],
            "sample_count": summary["sample_count"],
            "electron_number": summary["electron_number"],
            "uncorrelated_level": (summary["electron_number"] - 1)
            / summary["electron_number"],
            "n_steps": summary["steps"],
            "actual_batch_size": summary["actual_batch_size"],
            "configured_batch_size": summary["actual_batch_size"],
            "evaluation_id": run_root.name,
            "burn_in": plan["config"]["run"]["burn_in"],
            "sampler_steps": plan["config"]["sampler"]["steps"],
            "source_checkpoint_step": 199999,
            "source_checkpoint_sha256": plan["input"]["source_sha256"],
            "seed": plan["config"]["workflow"]["seed"],
            "config_yaml": (run_root / "evaluation/evaluation_config.yaml").read_text(),
        }
        for name, value in values.items():
            put("sphere", "pair_correlation/" + name, value)
        for node in (group, group["radial"]):
            node.attrs.update(
                status="available",
                note="无权重电子对计数，按精确球面分箱面积归一化",
                processing="固定训练参数重新采样，连续时间分块误差",
                source_files=json.dumps(ids),
            )
        group.attrs["statistics_sha256"] = identity
        np.testing.assert_allclose(
            group["radial/value"][:], arrays["value"], rtol=1e-12
        )
    temporary.replace(merge_root / "unified.h5")
    manifest["updated_at"] = stamp
    save_json(merge_root / "manifest.json", manifest)
    save_json(merge_root / "schema.json", schema)
    report = dict(
        summary,
        checked_at=stamp,
        backup=str(backup),
        run_root=str(run_root),
        unified_sha256=sha256(merge_root / "unified.h5"),
    )
    save_json(merge_root / "pair_correlation_validation.json", report)
    validation = json.loads((merge_root / "validation.json").read_text())
    validation.update(
        unified_sha256=report["unified_sha256"],
        pair_correlation_validation="pair_correlation_validation.json",
        source_file_count=len(manifest["files"]),
    )
    validation["known_issues"] = [
        s for s in validation.get("known_issues", []) if "球面对关联" not in s
    ]
    save_json(merge_root / "validation.json", validation)
    readme = merge_root / "README.md"
    with readme.open("a") as stream:
        stream.write(
            "\n球面对关联已按无权重计数和精确球面面积重新计算。"
            "角度分布、两种归一化、逐步计数及分块误差保存在 "
            "`sphere/pair_correlation`。旧加权结果保存在该组的 "
            "`legacy_weighted` 下。\n"
        )
    pending = merge_root / "待补充清单.md"
    text = pending.read_text()
    text = text.replace("用户决定是否修复对关联，", "").replace(
        "球面原有异常对关联结果保留原值并继续标记需要评估。本次没有重新采样对关联。",
        "球面对关联已重新采样并按精确球面面积归一化，原始异常加权结果仅作来源记录保留。",
    )
    pending.write_text(text.replace("| 球面 | ", "| 球面 | 对关联及误差、"))
    return report


def collect(args: argparse.Namespace) -> None:
    """Watch one job, download complete records, verify, and import into merge.

    Raises:
        SystemExit: If the scheduler reports a failed job.
    """
    args.run_root.mkdir(parents=True, exist_ok=True)
    program = f"""
from pathlib import Path
import json,subprocess
p=Path({args.remote_root!r})
job=(p/'jobs.tsv').read_text().split()[1]
s=subprocess.run(['sacct','-n','-X','-P','-j',job,'--format=JobIDRaw,State'],capture_output=True,text=True,check=True)
print(json.dumps({{'job_id':job,'accounting':s.stdout,'complete':(p/'completion.json').exists()}}))
"""
    while True:
        try:
            result = subprocess.run(
                [
                    "ssh",
                    "-T",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=15",
                    "paracloud",
                    "python3 -c " + shlex.quote(program),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=600,
            )
            status = json.loads(result.stdout)
            status["checked_at"] = datetime.now(UTC).isoformat()
            save_json(args.run_root / "collection_status.json", status)
            print(json.dumps(status), flush=True)
            if status["complete"] and "|COMPLETED" in status["accounting"]:
                subprocess.run(
                    [
                        "rsync",
                        "-az",
                        "--partial",
                        "paracloud:" + args.remote_root.rstrip("/") + "/",
                        str(args.run_root) + "/",
                    ],
                    check=True,
                    timeout=3600,
                )
                report = import_result(
                    args.run_root.resolve(), args.merge_root.resolve()
                )
                save_json(args.run_root / "collection_complete.json", report)
                return
            if any(
                s in status["accounting"]
                for s in (
                    "FAILED",
                    "CANCELLED",
                    "TIMEOUT",
                    "OUT_OF_MEMORY",
                    "NODE_FAIL",
                )
            ):
                raise SystemExit("Pair-correlation job failed; inspect cluster logs")
        except (subprocess.SubprocessError, OSError, json.JSONDecodeError) as error:
            print(str(error), flush=True)
        if not args.watch:
            return
        time.sleep(30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate = sub.add_parser("run")
    evaluate.add_argument("--data-root", type=Path, required=True)
    evaluate.add_argument("--train-dir", type=Path)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--batch-size", type=int, default=4096)
    evaluate.add_argument("--steps", type=int, default=10000)
    evaluate.add_argument("--burn-in", type=int, default=1000)
    evaluate.add_argument("--seed", type=int, default=314159)
    evaluate.add_argument("--dry-run", action="store_true")
    merging = sub.add_parser("import")
    watching = sub.add_parser("collect")
    for command in (merging, watching):
        command.add_argument("--run-root", type=Path, required=True)
        command.add_argument("--merge-root", type=Path, required=True)
    watching.add_argument("--remote-root", required=True)
    watching.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    elif args.command == "import":
        print(
            json.dumps(
                import_result(args.run_root.resolve(), args.merge_root.resolve()),
                indent=2,
            )
        )
    else:
        collect(args)


if __name__ == "__main__":
    main()
