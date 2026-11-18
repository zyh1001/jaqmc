# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Import validated multigeometry evaluations into the existing HDF5 dataset."""

import argparse
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
from hall_evaluation import SOURCES, save_json, sha256, validate_output


def load_runs(run_root: Path, source_root: Path) -> tuple[dict, list]:
    """Load matching evaluations.

    Returns:
        Validated stage data and source checks.

    Raises:
        ValueError: If a source or stage does not match.
    """
    loaded = {}
    checks = []
    for geometry in SOURCES:
        root = run_root / geometry
        completion = json.loads((root / "completion.json").read_text(encoding="utf-8"))
        provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
        source = source_root / SOURCES[geometry]
        if (
            completion["status"] != "complete"
            or sha256(source) != provenance["source_sha256"]
        ):
            raise ValueError(f"Incomplete or mismatched evaluation: {geometry}")
        stages = {}
        for state in ("trained", "laughlin"):
            directory = root / state
            record = json.loads(
                (directory / "run_manifest.json").read_text(encoding="utf-8")
            )
            if record["status"] != "complete":
                raise ValueError(f"Incomplete evaluation: {geometry}/{state}")
            validate_output(
                directory, record["config"], source if state == "trained" else None
            )
            summary = json.loads(
                (directory / "summary.json").read_text(encoding="utf-8")
            )
            with np.load(directory / "statistics.npz", allow_pickle=False) as f:
                statistics = dict(f)
            with np.load(directory / "evaluation_digest.npz", allow_pickle=False) as f:
                digest = dict(f)
            stages[state] = (record, summary, statistics, digest)
        loaded[geometry] = (provenance, stages)
        checks.append(
            {
                "name": f"{geometry}: complete evaluation with exact source parameters",
                "passed": True,
            }
        )

    return loaded, checks


def register_sources(run_root: Path, source_root: Path, manifest: dict) -> dict:
    """Register result checksums.

    Returns:
        Source identifiers for each geometry.
    """
    source_ids = {}
    for geometry in SOURCES:
        ids = []
        for path in sorted((run_root / geometry).rglob("*")):
            if (
                not path.is_file()
                or "_smoke" in str(path)
                or path.name.endswith(".log")
            ):
                continue
            relative = os.path.relpath(path, source_root)
            existing = next(
                (entry for entry in manifest["files"] if entry["path"] == relative),
                None,
            )
            entry = {
                "file_id": existing["file_id"]
                if existing
                else f"F{len(manifest['files']) + 1:04d}",
                "geometry": geometry,
                "category": "fixed_parameter_evaluation",
                "path": relative,
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
            if existing:
                existing.update(entry)
            else:
                manifest["files"].append(entry)
            ids.append(entry["file_id"])
        source_ids[geometry] = ids

    return source_ids


def register_reference_source(path: Path, source_root: Path, manifest: dict) -> str:
    """Register a supplied energy file.

    Returns:
        The manifest identifier of the supplied file.
    """
    relative = os.path.relpath(path, source_root)
    existing = next(
        (entry for entry in manifest["files"] if entry["path"] == relative), None
    )
    entry = {
        "file_id": existing["file_id"]
        if existing
        else f"F{len(manifest['files']) + 1:04d}",
        "geometry": "all",
        "category": "provided_reference_energies",
        "path": relative,
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }
    if existing:
        existing.update(entry)
    else:
        manifest["files"].append(entry)
    return entry["file_id"]


def write_energy(
    put, geometry: str, directory: Path, config: dict, summary: dict, stats: dict
) -> None:
    """Store energy records and block errors from the same evaluation."""
    # Replace all previous energy series together, including variance fields.
    with h5py.File(directory / "evaluation_stats.h5") as source:
        for key, ds in source.items():
            if key.startswith(("one_rdm", "structure_factor:")):
                continue
            name = key.removeprefix("observable:")
            put(geometry, "energy/series/" + name, ds[:])
            if key in summary["observables"]:
                value = summary["observables"][key]
                name = {
                    "total_energy_real": "total_energy",
                    "energy:kinetic": "kinetic",
                    "energy:potential": "potential",
                }.get(key, name)
                mean = (
                    value["mean"] + 1j * value["imaginary_mean"]
                    if key == "energy:kinetic"
                    else value["mean"]
                )
                put(geometry, "energy/summary/" + name, mean)
                put(
                    geometry,
                    "energy/summary/" + name + "_standard_error",
                    value["standard_error"],
                    status=value["status"],
                )
        put(
            geometry,
            "energy/series/step",
            np.arange(config["run"]["iterations"]),
        )
        put(
            geometry,
            "energy/summary/total_energy",
            np.mean(source["total_energy"][:], dtype=np.complex128),
        )
        if "total_energy_real_var" in source:
            put(
                geometry,
                "energy/summary/local_energy_real_variance",
                np.mean(source["total_energy_real_var"][:], dtype=np.float64),
            )
    energy = summary["observables"]["total_energy_real"]
    put(
        geometry,
        "energy/summary/standard_error",
        energy["standard_error"],
        status=energy["status"],
    )
    put(
        geometry,
        "energy/summary/standard_error_method",
        summary["error_method"],
    )
    put(geometry, "energy/summary/block_size", 100)
    put(
        geometry,
        "energy/summary/block_count",
        config["run"]["iterations"] // 100,
    )
    put(geometry, "energy/block_means", stats["total_energy_real:block_means"])
    put(geometry, "energy/tested_block_sizes", [50, 100, 200, 500])
    put(
        geometry,
        "energy/tested_standard_errors",
        [energy["tested_standard_errors"][str(b)] for b in (50, 100, 200, 500)],
    )


def write_geometry(
    h5,
    put,
    geometry: str,
    provenance: dict,
    stages: dict,
    run_root: Path,
    evaluation_directory: Path | None = None,
) -> None:
    """Store one geometry using the established dataset fields."""
    record, summary, stats, digest = stages["trained"]
    config = record["config"]
    directory = (
        evaluation_directory
        if evaluation_directory is not None
        else run_root / geometry / "trained"
    )
    groups = ["energy", "structure_factor"] + (
        ["one_rdm"] if geometry != "torus" else []
    )
    for observable in groups:
        h5[f"{geometry}/{observable}"].attrs["status"] = "available"
        for key, value in {
            "evaluation_id": run_root.name + "_" + geometry,
            "source_checkpoint_step": int(
                Path(
                    provenance.get("source_checkpoint", SOURCES[geometry])
                ).stem.rsplit("_", 1)[1]
            ),
            "source_checkpoint_sha256": provenance["source_sha256"],
            "configured_batch_size": config["workflow"]["batch_size"],
            "actual_batch_size": config["workflow"]["batch_size"],
            "n_steps": config["run"]["iterations"],
            "burn_in": config["run"]["burn_in"],
            "sampler_steps": config["sampler"]["steps"],
            "seed": config["workflow"]["seed"],
            "config_yaml": (directory / "evaluation_config.yaml").read_text(
                encoding="utf-8"
            ),
        }.items():
            put(geometry, observable + "/" + key, value)
    write_energy(put, geometry, directory, config, summary, stats)
    sf_keys = {
        key.removeprefix("structure_factor:"): value
        for key, value in stats.items()
        if key.startswith("structure_factor:")
    }
    for key, value in sf_keys.items():
        put(geometry, "structure_factor/" + key, value)
    definition = (
        (
            "rho(q)=sum exp(-i q.r); S=<|rho|^2>/N; "
            "connected=block-jackknife corrected density variance/N"
        )
        if geometry != "sphere"
        else (
            "rho(l,m)=sum Y(l,m); S_l=4pi*sum_m<|rho(l,m)|^2>/(N*(2l+1)); "
            "connected subtracts mode mean using block jackknife"
        )
    )
    put(geometry, "structure_factor/definition", definition)
    put(
        geometry,
        "structure_factor/sample_count",
        config["run"]["iterations"] * config["workflow"]["batch_size"],
    )
    if geometry != "torus":
        for target, source in {
            "matrix": "matrix:mean",
            "diagonal_occupations": "diagonal:mean",
            "landau_level_occupations": "landau_level_occupations:mean",
            "occupation_standard_error": ("landau_level_occupations:standard_error"),
            "trace": "trace:mean",
            "matrix_standard_error": "matrix:standard_error",
            "matrix_imaginary_standard_error": ("matrix:imaginary_standard_error"),
        }.items():
            put(geometry, "one_rdm/" + target, stats["one_rdm:" + source])
        for key in (
            "basis_landau_level",
            "basis_angular_momentum",
            "basis_guiding_centre",
        ):
            if "one_rdm:" + key in digest:
                put(geometry, "one_rdm/" + key, digest["one_rdm:" + key])
        with h5py.File(directory / "evaluation_stats.h5") as f:
            put(geometry, "one_rdm/series", f["one_rdm"][:])
        put(geometry, "one_rdm/n_levels", 3)
        put(
            geometry,
            "one_rdm/hermiticity_error",
            summary["one_rdm"]["hermiticity_error"],
        )
        h5[f"{geometry}/one_rdm/natural_occupations"].attrs.update(
            status="not_computed", note="本次不计算自然轨道占据"
        )
    if "laughlin" not in stages:
        return
    _, laughlin, laughlin_stats, _ = stages["laughlin"]
    reference = laughlin["observables"]["total_energy_real"]
    put(geometry, "reference_energies/Laughlin", reference["mean"])
    put(
        geometry,
        "reference_energies/Laughlin_standard_error",
        reference["standard_error"],
        status=reference["status"],
    )
    put(
        geometry,
        "reference_energies/Laughlin_block_means",
        laughlin_stats["total_energy_real:block_means"],
    )
    put(
        geometry,
        "reference_energies/Laughlin_conventions",
        json.dumps(record["config"]["system"]),
    )


def make_writer(h5: h5py.File, schema: dict, source_ids: dict):
    """Create a writer that preserves field types and source metadata.

    Returns:
        A callable for writing one observable field.
    """

    def put(geometry, field, value, note="", status="available", source_files=None):
        parent, name = field.rsplit("/", 1)
        group = h5.require_group(geometry + "/" + parent)
        attrs = dict(group[name].attrs) if name in group else {}
        if name in group:
            del group[name]
        if isinstance(value, str):
            ds = group.create_dataset(name, data=value, dtype=h5py.string_dtype())
            dtype = "str"
        else:
            value = np.asarray(value)
            declared = schema["fields"].get(field, {}).get("dtype")
            if field == "one_rdm/basis_angular_momentum" and value.dtype.kind == "f":
                declared = "float64"
                schema["fields"].setdefault(field, {})["dtype"] = declared
            if declared and declared != "str":
                value = value.astype(declared)
            options = (
                {"compression": "gzip", "compression_opts": 4, "shuffle": True}
                if value.ndim and value.size
                else {}
            )
            ds = group.create_dataset(name, data=value, **options)
            dtype = str(value.dtype)
        ds.attrs.update(attrs)
        ds.attrs.update(
            status=status,
            source_files=json.dumps(
                source_ids[geometry] if source_files is None else source_files
            ),
            processing=(
                "固定参数评估及连续时间分块统计"
                if source_files is None
                else "用户提供的精确对角化能量"
            ),
            note=note,
        )
        schema["fields"].setdefault(
            field,
            {
                "description": note or field,
                "dtype": dtype,
                "unit": "",
                "kind": "array" if ds.shape else "scalar",
            },
        )

    return put


def import_results(
    run_root: Path, merge_root: Path, references: Path | None = None
) -> dict:
    """Back up the dataset and publish only complete, matching evaluations.

    Returns:
        The validation report for the updated dataset.

    Raises:
        ValueError: If a run is incomplete or uses a different trained state.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    manifest = json.loads((merge_root / "manifest.json").read_text(encoding="utf-8"))
    schema = json.loads((merge_root / "schema.json").read_text(encoding="utf-8"))
    source_root = Path(manifest["source_root"])
    loaded, checks = load_runs(run_root, source_root)

    backup = merge_root.with_name(merge_root.name + "_evaluation_backup_" + stamp)
    shutil.copytree(merge_root, backup)
    temporary = merge_root / "unified.evaluation.tmp.h5"
    shutil.copy2(merge_root / "unified.h5", temporary)
    source_ids = register_sources(run_root, source_root, manifest)

    with h5py.File(temporary, "r+") as h5:
        put = make_writer(h5, schema, source_ids)

        for geometry, (provenance, stages) in loaded.items():
            write_geometry(h5, put, geometry, provenance, stages, run_root)
        if references:
            reference_id = register_reference_source(references, source_root, manifest)
            write_references(put, references, reference_id)
        missing_ed = any(
            h5[f"{geometry}/reference_energies/{label}"].shape is None
            or not np.isfinite(h5[f"{geometry}/reference_energies/{label}"][()])
            for geometry in SOURCES
            for label in ("LLL_ED", "2LL_ED", "3LL_ED")
        )
        h5.attrs["evaluation_extension"] = run_root.name
        for geometry in SOURCES:
            matrix = h5[f"{geometry}/one_rdm/matrix"]
            if geometry != "torus" and matrix.shape != (
                (60, 60) if geometry == "disk" else (54, 54)
            ):
                raise ValueError("Unexpected one-body density matrix basis size")
            checks.append(
                {
                    "name": geometry
                    + ": imported finite structure factor and reference energy",
                    "passed": bool(
                        np.isfinite(h5[f"{geometry}/structure_factor/value"][:]).all()
                        and np.isfinite(
                            h5[f"{geometry}/reference_energies/Laughlin"][()]
                        )
                    ),
                }
            )
    if not all(check["passed"] for check in checks):
        raise ValueError("Updated dataset validation failed")
    report = {
        "checked_at": stamp,
        "passed": True,
        "checks": checks,
        "source_root": str(source_root),
        "source_file_count": len(manifest["files"]),
        "baseline_validation": str(backup / "validation.json"),
        "run_root": str(run_root),
        "backup": str(backup),
        "unified_sha256": sha256(temporary),
        "known_issues": ["球面对关联等待用户决定是否修复"]
        + (["精确对角化参考值等待用户提供"] if missing_ed else []),
    }
    manifest["updated_at"] = stamp
    manifest["evaluation_extension"] = str(run_root)
    temporary.replace(merge_root / "unified.h5")
    save_json(merge_root / "manifest.json", manifest)
    save_json(merge_root / "schema.json", schema)
    save_json(merge_root / "validation.json", report)
    write_dataset_documentation(merge_root, run_root, loaded)
    return report


def write_dataset_documentation(merge_root: Path, run_root: Path, loaded: dict) -> None:
    """Describe current data coverage and calculations still awaiting input."""
    rows = []
    names = {"disk": "圆盘", "sphere": "球面", "torus": "环面"}
    ed_values = {}
    with h5py.File(merge_root / "unified.h5") as h5:
        for geometry in SOURCES:
            values = []
            for label in ("LLL_ED", "2LL_ED", "3LL_ED"):
                ds = h5[f"{geometry}/reference_energies/{label}"]
                values.append(float(ds[()]) if ds.shape is not None else float("nan"))
            ed_values[geometry] = values
    missing_ed = any(not np.isfinite(values).all() for values in ed_values.values())
    for geometry, (_, stages) in loaded.items():
        energy = stages["trained"][1]["observables"]["total_energy_real"]
        reference = stages["laughlin"][1]["observables"]["total_energy_real"]
        rows.append(
            f"| {names[geometry]} | {energy['mean']:.10g} "
            f"± {energy['standard_error']:.3g} "
            f"| {reference['mean']:.10g} ± {reference['standard_error']:.3g} | "
            + " | ".join(
                f"{value:.12g}" if np.isfinite(value) else "待提供"
                for value in ed_values[geometry]
            )
            + " |"
        )
    readme = "\n".join(
        [
            "# 三种几何的统一数据集",
            "",
            "数据来自用户指定的圆盘、单行列式球面和环面训练。训练参数保持固定，补充评估结果及连续时间分块误差。",
            "",
            "| 几何 | 训练态总能量 | 朗福林态总能量 | "
            "单能级精确对角化 | 两能级精确对角化 | 三能级精确对角化 |",
            "|---|---|---|---|---|---|",
            *rows,
            "",
            r"能量单位为 $\hbar\omega_c$。三个几何各自沿用训练时的背景电荷和能量零点，"
            "不同几何的原始总能量需要结合各自的能量零点解释。",
            "精确对角化能量由用户提供。环面按用户给出的表达式加上 1.89975557953，"
            "原始表达式及来源保存在对应的能量来源字段中。",
            "",
            "圆盘和球面含三个朗道能级的单体约化密度矩阵及占据误差。"
            "圆盘每层 20 个轨道，球面分别有 16、18、20 个轨道。"
            "本次不计算自然轨道占据。环面原有密度矩阵结果保留原值。",
            "",
            "三个几何均有粒子密度结构因子。连通部分保存分块删一法修正值、直接代入均值的结果和相应误差。球面以角动量编号，环面以倒格子波矢编号。",
            "",
            f"本次评估结果目录：`{run_root}`。",
            "",
            "`manifest.json` 记录来源文件及校验和，`schema.json` 说明字段，"
            "`validation.json` 记录本次验收及备份位置。"
            "逐步原始记录保存在结果目录及对应的数据分组中。",
            "",
            "后续结果通过项目的 `scripts/merge_hall_evaluation.py` 导入。"
            "基础构建程序重建原始数据后，需要再次导入本次补充结果。",
            "",
            "尚需补充的项目见 [待补充清单](待补充清单.md)。",
            "",
        ]
    )
    pending = "\n".join(
        [
            "# 待补充清单",
            "",
            "| 几何 | 已补充 | 待补充或待评估 |",
            "|---|---|---|",
            "| 圆盘 | 结构因子及误差、三个朗道能级占据、"
            "能量及角动量误差、朗福林参考能量 "
            "| 用户提供三种精确对角化参考能量，检查分块误差是否稳定 |",
            "| 球面 | 结构因子及误差、三个朗道能级占据、"
            "能量及角动量误差、朗福林参考能量 "
            "| 用户决定是否修复对关联，提供三种精确对角化参考能量，"
            "检查分块误差是否稳定 |",
            "| 环面 | 结构因子及误差、固定参数的总能量、动能、势能和误差、"
            "朗福林参考能量 "
            "| 用户提供三种精确对角化参考能量，原有密度矩阵误差尚需进一步评估 |",
            "",
            "球面原有异常对关联结果保留原值并继续标记需要评估。本次没有重新采样对关联。",
            "",
            "圆盘和球面的自然轨道占据不属于本次计算项目。能级占据总和不强制归一化，有限基底和统计误差需结合分块记录判断。",
            "",
        ]
    )
    if not missing_ed:
        pending = (
            pending.replace(
                "朗福林参考能量 ", "朗福林参考能量、三种精确对角化参考能量 "
            )
            .replace("用户提供三种精确对角化参考能量，", "")
            .replace("提供三种精确对角化参考能量，", "")
        )
    (merge_root / "README.md").write_text(readme, encoding="utf-8")
    (merge_root / "待补充清单.md").write_text(pending, encoding="utf-8")


def write_references(put, references: Path, source_file_id: str) -> None:
    """Import reference values previously checked by hall_evaluation."""
    payload = json.loads(references.read_text(encoding="utf-8"))
    for geometry, values in payload["reference_energies"].items():
        for label, entry in values.items():
            put(
                geometry,
                "reference_energies/" + label,
                entry["energy"],
                source_files=[source_file_id],
            )
            put(
                geometry,
                "reference_energies/" + label + "_provenance",
                json.dumps(entry),
                source_files=[source_file_id],
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--merge-root", type=Path, required=True)
    parser.add_argument("--references", type=Path)
    args = parser.parse_args()
    result = import_results(
        args.run_root.resolve(), args.merge_root.resolve(), args.references
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
