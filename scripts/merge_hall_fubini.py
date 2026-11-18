# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Validate completed checkpoint overlaps and add their series to merge."""

import argparse
import fcntl
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
from hall_evaluation import SOURCES, save_json, sha256

STAT_FIELDS = ("log_mean_exp_delta", "log_mean_exp_minus_delta", "phase_mean")
DESCRIPTIONS = {
    "step": "训练检查点步数",
    "distance": "相对于最终训练态的 Fubini–Study 距离",
    "cosine": "归一化波函数重叠的模长",
    "modulus_cosine": "波函数模长的重叠因子",
    "phase_factor": "相位平均的模长",
    "reference_step": "参考训练态的检查点步数",
    "iterations_per_checkpoint": "各检查点的正式采样步数，也是逐步统计的有效长度",
    "burn_in_per_checkpoint": "各检查点的热化步数",
    "total_configurations_per_checkpoint": "各检查点的正式采样构型总数",
}


def read_overlap(directory: Path, iterations: int) -> tuple[tuple, dict]:
    """Recompute overlap factors from the full recorded sampling series.

    Returns:
        Native digest values and validated batch statistics.

    Raises:
        ValueError: If the sampling series is incomplete or nonfinite.
    """
    with h5py.File(directory / "overlap_stats.h5", "r") as handle:
        statistics = {key: handle["fubini:" + key][()] for key in STAT_FIELDS}
    if any(
        values.shape != (iterations,) or not np.isfinite(values).all()
        for values in statistics.values()
    ):
        raise ValueError(f"Incomplete or nonfinite sampling: {directory}")
    log_means = [
        np.logaddexp.reduce(statistics[key].astype(np.float64)) - np.log(iterations)
        for key in STAT_FIELDS[:2]
    ]
    modulus = np.clip(np.exp(-0.5 * sum(log_means)), 0, 1)
    phase = np.clip(abs(statistics["phase_mean"].astype(np.complex128).mean()), 0, 1)
    cosine = np.clip(modulus * phase, 0, 1)
    with np.load(directory / "overlap_digest.npz", allow_pickle=False) as saved:
        native = tuple(
            float(saved[k])
            for k in ("fubini_distance", "cosine", "cos_gamma_modulus", "f_phase")
        )
    if not np.isfinite(native).all():
        raise ValueError(f"Nonfinite distance summary: {directory}")
    np.testing.assert_allclose(
        native[1:], (cosine, modulus, phase), atol=3e-6, rtol=3e-6
    )
    np.testing.assert_allclose(native[0], np.arccos(native[1]), atol=2e-6)
    return native, statistics


def validate_sampling_counts(plan: dict, summary: dict, arrays: dict) -> np.ndarray:
    """Check per-checkpoint counts and add their arrays to the imported series.

    Returns:
        The requested iteration count for each checkpoint.

    Raises:
        ValueError: If counts in the plan and completed summaries disagree.
    """
    iterations = np.array(
        [item.get("iterations", plan["iterations"]) for item in plan["checkpoints"]],
        dtype=np.int64,
    )
    burn_in = np.array(
        [item.get("burn_in", plan["burn_in"]) for item in plan["checkpoints"]],
        dtype=np.int64,
    )
    if np.any(iterations <= 0) or np.any(burn_in < 0):
        raise ValueError("Invalid per-checkpoint sampling specification")
    configurations = iterations * plan["batch_size"]
    common_count = (
        int(configurations[0]) if np.all(configurations == configurations[0]) else None
    )
    if summary["total_configurations_per_checkpoint"] != common_count:
        raise ValueError("Sample count mismatch")
    for key, values in (
        ("iterations_per_checkpoint", iterations),
        ("burn_in_per_checkpoint", burn_in),
        ("total_configurations_per_checkpoint", configurations),
    ):
        if key in arrays:
            np.testing.assert_array_equal(arrays[key], values)
        arrays[key] = values
    records = summary["records"]
    if len(records) != len(plan["checkpoints"]):
        raise ValueError("Incomplete checkpoint summary")
    for index, record in enumerate(records):
        if (
            record.get("iterations", plan["iterations"]) != iterations[index]
            or record.get("burn_in", plan["burn_in"]) != burn_in[index]
            or record.get("total_configurations", configurations[index])
            != configurations[index]
        ):
            raise ValueError("Per-checkpoint sample count mismatch")
    return iterations


def load_geometry(
    directory: Path, geometry: str, source_root: Path, checkpoint: Path | None = None
) -> dict:
    """Check sampling completeness, checkpoint identity, and overlap arithmetic.

    Returns:
        Validated arrays, provenance, and source files.

    Raises:
        ValueError: If input identities, sampling, or summary values disagree.
    """
    plan = json.loads((directory / "fubini_plan.json").read_text())
    summary = json.loads((directory / "fubini_summary.json").read_text())
    final = checkpoint if checkpoint is not None else source_root / SOURCES[geometry]
    digest = sha256(final)
    final_step = int(final.stem.rsplit("_", 1)[1])
    identity = {
        "geometry": geometry,
        "reference_step": final_step,
        "reference_sha256": digest,
    }
    if any(
        record[key] != value
        for record in (plan, summary)
        for key, value in identity.items()
    ):
        raise ValueError(f"Reference checkpoint mismatch: {geometry}")
    steps = np.array([item["step"] for item in plan["checkpoints"]], dtype=np.int64)
    expected = sorted(
        int(p.stem.rsplit("_", 1)[1]) for p in final.parent.glob("train_ckpt_*.npz")
    )
    np.testing.assert_array_equal(steps, expected)
    if (
        plan["sampling_density"] != "abs(psi_checkpoint * psi_final)"
        or plan["iterations"] <= 0
        or plan["batch_size"] <= 0
    ):
        raise ValueError("Unexpected overlap sampling specification")
    with np.load(directory / "fubini_series.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    np.testing.assert_array_equal(arrays["step"], steps)
    np.testing.assert_equal(arrays["reference_step"], final_step)
    iterations = validate_sampling_counts(plan, summary, arrays)
    records = summary["records"]
    statistics = []
    sources = [
        directory / name
        for name in ("fubini_plan.json", "fubini_summary.json", "fubini_series.npz")
    ]
    for index, item in enumerate(plan["checkpoints"]):
        step_dir = directory / f"step_{item['step']:07d}"
        source = final.parent / Path(item["input"]["source_checkpoint"]).name
        if sha256(source) != item["input"]["source_sha256"]:
            raise ValueError(f"Source checkpoint changed: {source}")
        record = records[index]
        digest_path = step_dir / "overlap_digest.npz"
        if record["step"] != item["step"] or record["digest_sha256"] != sha256(
            digest_path
        ):
            raise ValueError(f"Summary digest mismatch: {step_dir}")
        native, batch_stats = read_overlap(step_dir, int(iterations[index]))
        statistics.append(batch_stats)
        keys = ("distance", "cosine", "modulus_cosine", "phase_factor")
        for key, value in zip(keys, native, strict=True):
            np.testing.assert_allclose(arrays[key][index], value, atol=2e-6)
        np.testing.assert_allclose(record["distance"], native[0], atol=2e-6)
        sources.extend(
            [
                digest_path,
                step_dir / "overlap_stats.h5",
                step_dir / "requested_config.yaml",
            ]
        )
    if not np.all((arrays["distance"] >= 0) & (arrays["distance"] <= np.pi / 2 + 1e-6)):
        raise ValueError("Distance outside [0, pi/2]")
    np.testing.assert_allclose(arrays["distance"][steps == final_step], 0, atol=5e-4)
    return {
        "plan": plan,
        "arrays": arrays,
        "statistics": {
            k: np.stack(
                [
                    np.pad(
                        batch[k],
                        (0, int(iterations.max()) - len(batch[k])),
                        constant_values=np.nan,
                    )
                    for batch in statistics
                ]
            )
            for k in STAT_FIELDS
        },
        "sources": sources,
    }


def import_results(run_root: Path, merge_root: Path) -> dict:
    """Back up merge and import all three completed geometries together.

    Returns:
        Validation and backup information.

    """
    with (merge_root / ".fubini-import.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _import_locked(run_root, merge_root)


def _import_locked(run_root: Path, merge_root: Path) -> dict:
    manifest = json.loads((merge_root / "manifest.json").read_text())
    source_root = Path(manifest["source_root"])
    loaded = {g: load_geometry(run_root / g, g, source_root) for g in SOURCES}
    identities = {g: sha256(run_root / g / "fubini_series.npz") for g in SOURCES}
    with h5py.File(merge_root / "unified.h5", "r") as handle:
        for geo, record in loaded.items():
            if (
                handle[f"{geo}/metadata/final_checkpoint_step"][()]
                != record["plan"]["reference_step"]
            ):
                raise ValueError(f"Merged reference step differs: {geo}")
        if all(
            f"{g}/fubini" in handle
            and handle[f"{g}/fubini"].attrs.get("series_sha256") == digest
            for g, digest in identities.items()
        ):
            return json.loads((merge_root / "fubini_validation.json").read_text())
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    backup = merge_root.with_name(merge_root.name + "_fubini_backup_" + stamp)
    shutil.copytree(
        merge_root, backup, ignore=shutil.ignore_patterns("*.lock", "*.tmp.h5")
    )
    schema = json.loads((merge_root / "schema.json").read_text())
    temporary = merge_root / "unified.fubini.tmp.h5"
    shutil.copy2(merge_root / "unified.h5", temporary)
    with h5py.File(temporary, "r+") as handle:
        for geo, record in loaded.items():
            ids = []
            for path in record["sources"]:
                relative = os.path.relpath(path, source_root)
                entry = next(
                    (e for e in manifest["files"] if e["path"] == relative), None
                )
                if entry is None:
                    entry = {"file_id": f"F{len(manifest['files']) + 1:04d}"}
                    manifest["files"].append(entry)
                entry.update(
                    geometry=geo,
                    category="fubini_overlap",
                    path=relative,
                    sha256=sha256(path),
                    size_bytes=path.stat().st_size,
                )
                ids.append(entry["file_id"])
            parent = handle[geo]
            if "fubini" in parent:
                del parent["fubini"]
            group = parent.create_group("fubini")
            group.attrs.update(
                status="available",
                series_sha256=identities[geo],
                source_files=json.dumps(ids),
                run_root=str(run_root),
            )
            values = dict(record["arrays"])
            plan = record["plan"]
            values.update(
                {
                    k: plan[k]
                    for k in (
                        "batch_size",
                        "iterations",
                        "burn_in",
                        "sampler_updates_per_step",
                        "sampling_density",
                        "reference_sha256",
                    )
                }
            )
            values.update({"series/" + k: v for k, v in record["statistics"].items()})
            for name, value in values.items():
                options = (
                    {"compression": "gzip", "shuffle": True} if np.ndim(value) else {}
                )
                dataset = group.create_dataset(
                    name,
                    data=value,
                    dtype=h5py.string_dtype() if isinstance(value, str) else None,
                    **options,
                )
                spec = {
                    "description": DESCRIPTIONS.get(name, name),
                    "dtype": "str" if isinstance(value, str) else str(dataset.dtype),
                    "kind": "array" if dataset.ndim else "scalar",
                    "unit": "rad" if name == "distance" else "",
                }
                dataset.attrs.update(
                    **spec,
                    status="available",
                    source_files=json.dumps(ids),
                    source_fields=json.dumps([name]),
                    note="",
                    processing="固定训练参数，相对于最终训练态的对称密度重叠采样",
                )
                if name.startswith("series/"):
                    dataset.attrs["note"] = (
                        "每行有效长度见 iterations_per_checkpoint，"
                        "末尾不足部分以 NaN 填充"
                    )
                schema["fields"]["fubini/" + name] = spec
            missing = group.create_dataset("standard_error", data=h5py.Empty("float64"))
            spec = {
                "description": "距离标准误差，未计算时留空",
                "dtype": "float64",
                "kind": "array",
                "unit": "rad",
            }
            missing.attrs.update(
                **spec,
                status="not_computed",
                source_files="[]",
                source_fields="[]",
                note="",
                processing="",
            )
            schema["fields"]["fubini/standard_error"] = spec
        handle.attrs["updated_at"] = stamp
        handle.attrs["fubini_extension"] = str(run_root)
    report = {
        "passed": True,
        "checked_at": stamp,
        "backup": str(backup),
        "run_root": str(run_root),
        "series_sha256": identities,
        "checkpoints": {g: len(r["arrays"]["step"]) for g, r in loaded.items()},
        "unified_sha256": sha256(temporary),
    }
    temporary.replace(merge_root / "unified.h5")
    manifest["updated_at"] = stamp
    manifest["fubini_extension"] = str(run_root)
    save_json(merge_root / "manifest.json", manifest)
    save_json(merge_root / "schema.json", schema)
    save_json(merge_root / "fubini_validation.json", report)
    validation = json.loads((merge_root / "validation.json").read_text())
    validation.update(
        fubini_extension=report,
        unified_sha256=report["unified_sha256"],
        source_file_count=len(manifest["files"]),
    )
    save_json(merge_root / "validation.json", validation)
    readme = merge_root / "README.md"
    text = readme.read_text().split("\n## 检查点态距离\n")[0]
    text += (
        "\n## 检查点态距离\n\n三种几何的历史检查点相对于各自最终训练态的距离保存在 "
        "`<几何>/fubini/`。距离单位为弧度，逐步重叠统计保存在 `series/`。"
        "检查点、样本数和统计复算检查见 `fubini_validation.json`。"
        "数据通过项目的 `scripts/merge_hall_fubini.py` 导入，"
        "标准误差未计算，保持为空。\n"
    )
    readme.write_text(text)
    save_json(run_root / "merge_import.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--merge-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            import_results(args.run_root.resolve(), args.merge_root.resolve()), indent=2
        )
    )


if __name__ == "__main__":
    main()
