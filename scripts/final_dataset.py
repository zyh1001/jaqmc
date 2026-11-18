# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Build the final Hall dataset and reproduce its figures from verified results."""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml
from hall_evaluation import save_json, sha256, validate_output, wavefunction_config
from merge_hall_evaluation import make_writer, write_geometry


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_parameters(source: Path, checkpoint: Path, batch: int) -> None:
    """Reject changed network parameters, missing arrays, or wrong walker counts.

    Raises:
        ValueError: If the requested input or result fails validation.
    """
    with (
        np.load(source, allow_pickle=False) as a,
        np.load(checkpoint, allow_pickle=False) as b,
    ):
        keys = {k for k in a.files if k.startswith("params/")}
        if not keys or keys != {k for k in b.files if k.startswith("params/")}:
            raise ValueError("Checkpoint parameter names differ")
        for key in keys:
            np.testing.assert_array_equal(a[key], b[key])
        if b["batched_data/data/electrons"].shape[0] != batch:
            raise ValueError("Incorrect actual walker count")


def scan_postprocess(
    digest: dict[str, Any], output: Path
) -> tuple[dict[str, Any], np.ndarray]:
    """Recover display magnitude and integer phase winding around grid cells.

    Returns:
        The verified result or requested metadata.

    Raises:
        ValueError: If the requested input or result fails validation.
    """
    output.mkdir(parents=True, exist_ok=True)
    d = {
        k.removeprefix("wavefunction_scan:"): v
        for k, v in digest.items()
        if k.startswith("wavefunction_scan:")
    }
    log_abs, phase = np.asarray(d["log_abs"], float), np.asarray(d["phase"], float)
    if log_abs.shape != (len(d["y"]), len(d["x"])) or not np.isfinite(log_abs).all():
        raise ValueError("Invalid conditional-wavefunction grid")
    shift = float(log_abs.max())
    threshold = 1e-4

    def wrap(x: Any) -> None:
        return (x + np.pi) % (2 * np.pi) - np.pi

    winding = (
        wrap(phase[:-1, 1:] - phase[:-1, :-1])
        + wrap(phase[1:, 1:] - phase[:-1, 1:])
        + wrap(phase[1:, :-1] - phase[1:, 1:])
        + wrap(phase[:-1, :-1] - phase[1:, :-1])
    ) / (2 * np.pi)
    charges = np.rint(winding).astype(int)
    iy, ix = np.nonzero(charges != 0)
    x = (np.asarray(d["x"], dtype=float)[:-1] + np.asarray(d["x"], dtype=float)[1:]) / 2
    y = (np.asarray(d["y"], dtype=float)[:-1] + np.asarray(d["y"], dtype=float)[1:]) / 2
    table = np.column_stack([x[ix], y[iy], charges[iy, ix]])
    np.savetxt(
        output / "phase_vortices.csv",
        table,
        delimiter=",",
        header="x,y,charge",
        comments="",
    )
    summary = dict(
        grid_shape_ny_nx=list(log_abs.shape),
        log_shift=shift,
        phase_mask_threshold=threshold,
        vortex_count=len(table),
        net_vortex_charge=int(charges[iy, ix].sum()),
        selection_samples=int(d["selection_samples"]),
        selected_log_abs=float(d["selected_log_abs"]),
    )
    save_json(output / "wavefunction_scan_summary.json", summary)
    return summary, table


def disk_spatial_postprocess(directory: Path) -> None:
    """Recreate the existing radial correlation table from native grid results.

    Raises:
        ValueError: If the requested input or result fails validation.
    """
    with np.load(directory / "evaluation_digest.npz", allow_pickle=False) as z:
        d = dict(z)
    steps = int(d["pair_correlation:n_steps"])
    sample_count = float(d["pair_correlation:sample_count"])
    if sample_count != steps * 4096:
        raise ValueError("Disk pair sample count differs from the requested run")
    x, y = d["pair_correlation:x"].astype(float), d["pair_correlation:y"].astype(float)
    width = float(np.sqrt(d["pair_correlation:bin_area"]))
    maximum = float(max(abs(x[0]), abs(x[-1])) + width / 2)
    bins = round(maximum / width)
    radius = np.hypot(x[:, None], y[None, :])
    index = np.floor(radius / width).astype(int)
    g = d["pair_correlation"].astype(float)
    valid = np.isfinite(g)
    values = []
    cumulative = 0.0
    for i in range(bins):
        mask = index == i
        good = mask & valid
        shell_area = np.pi * ((i + 1) ** 2 - i**2) * width**2

        def average(name: str) -> float:
            if name in ("pair_correlation", "pair_correlation:g_finite_n"):
                return float(np.asarray(d[name])[good].mean()) if good.any() else np.nan
            if name == "pair_correlation:rho2":
                return float(np.asarray(d[name])[mask].sum() * width**2 / shell_area)
            return float(np.asarray(d[name])[mask].mean()) if mask.any() else np.nan

        rho_integral = float(d["pair_correlation:rho2"][mask].sum() * width**2)
        cumulative += rho_integral
        values.append(
            [
                (i + 0.5) * width,
                average("pair_correlation"),
                average("pair_correlation:g_finite_n"),
                average("pair_correlation:rho2"),
                average("pair_correlation:rho1_target"),
                rho_integral,
                cumulative,
                good.sum() * width**2 / shell_area,
            ]
        )
    output = directory / "analysis_outputs"
    output.mkdir(exist_ok=True)
    np.savetxt(
        output / "pair_correlation_radial.txt",
        values,
        header=(
            "r_prime g_angular_average g_finite_n_angular_average "
            "rho2_angular_average rho1_target_angular_average "
            "rho2_shell_integral rho2_cumulative_integral valid_area_fraction"
        ),
    )
    cfg = yaml.safe_load((directory / "evaluation_config.yaml").read_text())
    nsteps = cfg["run"]["iterations"]
    counts = d["density"].astype(float)
    actual = counts.sum() / (nsteps * 4096)
    if not np.isclose(actual, sum(cfg["system"]["nspins"]), atol=0.005):
        raise ValueError(
            f"Disk density integral differs from the electron count: {actual}"
        )
    save_json(
        output / "spatial_validation.json",
        dict(
            passed=True,
            actual_batch_size=4096,
            sample_count=int(sample_count),
            integrated_density=actual,
            undefined_pair_bins=int((~valid).sum()),
            postprocessing_code_sha256=sha256(Path(__file__)),
        ),
    )


def import_builder(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("final_base_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def empty_copy(source: h5py.Group, dest: Any) -> None:
    """Copy the schema and descriptions while clearing all old measurements."""
    for name, node in source.items():
        if isinstance(node, h5py.Group):
            child = dest.create_group(name)
            empty_copy(node, child)
        else:
            child = dest.create_dataset(name, data=h5py.Empty(node.dtype))
        for key in ["description", "unit", "dtype", "kind"]:
            if key in node.attrs:
                child.attrs[key] = node.attrs[key]
        child.attrs.update(
            status="not_computed",
            source_files="[]",
            source_fields="[]",
            note="",
            processing="",
        )


def register(manifest: dict[str, Any], path: Path, geometry: str, category: str) -> str:
    relative = os.path.relpath(path, manifest["source_root"])
    entry = next((v for v in manifest["files"] if v["path"] == relative), None)
    if entry is None:
        entry = dict(file_id=f"F{len(manifest['files']) + 1:04d}", path=relative)
        manifest["files"].append(entry)
    entry.update(
        geometry=geometry,
        category=category,
        sha256=sha256(path),
        size_bytes=path.stat().st_size,
    )
    return entry["file_id"]


def compare_groups(a: h5py.Group, b: h5py.Group) -> None:
    """Check every group, dataset value, and attribute, including null datasets.

    Raises:
        ValueError: If the requested input or result fails validation.
    """
    if set(a.keys()) != set(b.keys()):
        raise ValueError("Group members differ: " + a.name)

    def attrs_equal(x: Any, y: Any) -> None:
        if set(x.attrs) != set(y.attrs):
            raise ValueError("Attributes differ: " + x.name)
        for k in x.attrs:
            np.testing.assert_equal(x.attrs[k], y.attrs[k])

    attrs_equal(a, b)
    for name in a:
        x, y = a[name], b[name]
        attrs_equal(x, y)
        if isinstance(x, h5py.Group):
            compare_groups(x, y)
        else:
            compare_datasets(x, y)


def compare_datasets(x: Any, y: Any) -> None:
    """Compare dataset shapes, types, and values without loading large arrays.

    Raises:
        ValueError: If the requested input or result fails validation.
    """
    if x.shape != y.shape or x.dtype != y.dtype:
        raise ValueError("Dataset shape or dtype differs: " + x.name)
    if x.shape is None:
        return
    if x.ndim == 0 or x.dtype.kind in "OSU":
        np.testing.assert_equal(x[()], y[()])
    else:
        for start in range(0, len(x), 32768):
            np.testing.assert_array_equal(
                x[start : start + 32768], y[start : start + 32768]
            )


def validate_references(root: Path, old: h5py.File) -> None:
    """Require matching physical parameters before reusing reference energies.

    Raises:
        ValueError: If the requested input or result fails validation.
    """
    for geo in ["disk", "sphere"]:
        config = yaml.safe_load(
            (root / "raw" / geo / "train/train_config.yaml").read_text()
        )
        previous = yaml.safe_load(old[f"{geo}/metadata/config_yaml"].asstr()[()])
        keys = set(config["system"]) | set(previous["system"])
        for key in keys - {"module"}:
            if config["system"].get(key) != previous["system"].get(key):
                raise ValueError(
                    f"Physical system differs for reused references: {geo}/{key}"
                )


def validate_regular_input(folder: Path, train_dir: Path, geometry: str) -> None:
    """Require the requested new network and the agreed evaluation sampling.

    Raises:
        ValueError: If the source configuration or sampling differs.
    """
    provenance = read_json(folder / "provenance.json")
    config = read_json(folder / "trained/run_manifest.json")["config"]
    training_config = train_dir / "train_config.yaml"
    train = yaml.safe_load(training_config.read_text(encoding="utf-8"))
    if provenance["source_config_sha256"] != sha256(training_config):
        raise ValueError("Training configuration checksum differs")
    if config["wf"] != wavefunction_config(train, geometry, False):
        raise ValueError("Evaluation wavefunction differs from the trained network")
    if (
        config["workflow"]["batch_size"],
        config["run"]["iterations"],
        config["run"]["burn_in"],
        config["sampler"]["steps"],
    ) != (4096, 10000, 1000, 10):
        raise ValueError("Regular evaluation sampling differs from the agreed plan")


def validate_overlap_parameters(folder: Path, train_dir: Path) -> None:
    """Verify both networks in every completed symmetric-overlap checkpoint.

    Raises:
        ValueError: If parameter names or the walker count differ.
    """
    plan = read_json(folder / "fubini_plan.json")
    reference = train_dir / Path(plan["reference_checkpoint"]).name
    for item in plan["checkpoints"]:
        steps = item.get("iterations", plan["iterations"])
        original = train_dir / Path(item["input"]["source_checkpoint"]).name
        saved = (
            folder / f"step_{item['step']:07d}" / f"overlap_ckpt_{steps - 1:06d}.npz"
        )
        with (
            np.load(original, allow_pickle=False) as current,
            np.load(reference, allow_pickle=False) as final,
            np.load(saved, allow_pickle=False) as result,
        ):
            expected = {
                f"params/{namespace}/{key.removeprefix('params/')}": source[key]
                for namespace, source in (("current", current), ("reference", final))
                for key in source.files
                if key.startswith("params/")
            }
            actual = {key for key in result.files if key.startswith("params/")}
            if not expected or set(expected) != actual:
                raise ValueError("Overlap checkpoint parameter names differ")
            for key, value in expected.items():
                np.testing.assert_array_equal(value, result[key])
            if result["batched_data/data/electrons"].shape[0] != plan["batch_size"]:
                raise ValueError("Incorrect actual overlap walker count")


def import_evaluations(
    root: Path,
    destination: Path,
    workflow: dict[str, Any],
    manifest: dict[str, Any],
    schema: dict[str, Any],
    h5: h5py.File,
    builder: Any,
    include_fubini: bool = True,
) -> None:
    """Import verified final-state measurements and checkpoint overlaps.

    Raises:
        ValueError: If the requested input or result fails validation.
    """
    from merge_hall_fubini import DESCRIPTIONS, load_geometry
    from sphere_pair_correlation import inspect_stage

    for geo in ["disk", "sphere"]:
        folder = root / "raw" / geo / "evaluation/regular"
        trained = folder / "trained"
        provenance = read_json(folder / "provenance.json")
        final = (
            root
            / "raw"
            / geo
            / "train"
            / ("train_ckpt_149999.npz" if geo == "disk" else "train_ckpt_199999.npz")
        )
        if sha256(final) != provenance["source_sha256"]:
            raise ValueError("New evaluation belongs to another network: " + geo)
        validate_regular_input(folder, final.parent, geo)
        record = read_json(trained / "run_manifest.json")
        validate_output(trained, record["config"], final)
        ids = [
            register(manifest, p, geo, "fixed_parameter_evaluation")
            for p in sorted(folder.rglob("*"))
            if p.is_file() and "trained_smoke" not in str(p)
        ]
        summary = read_json(trained / "summary.json")
        with (
            np.load(trained / "statistics.npz") as z,
            np.load(trained / "evaluation_digest.npz") as d,
        ):
            stages = {"trained": (record, summary, dict(z), dict(d))}
        put = make_writer(h5, schema, {geo: ids})
        write_geometry(
            h5,
            put,
            geo,
            provenance,
            stages,
            root / "raw",
            evaluation_directory=trained,
        )
        if geo == "disk":
            builder.current_files = ids
            builder.files = {e["path"]: e for e in manifest["files"]}
            for observable in ["density", "pair_correlation"]:
                cfgtext = (trained / "evaluation_config.yaml").read_text()
                cfg = yaml.safe_load(cfgtext)
                for name, value in dict(
                    actual_batch_size=4096,
                    configured_batch_size=4096,
                    n_steps=10000,
                    burn_in=1000,
                    sampler_steps=10,
                    seed=cfg["workflow"]["seed"],
                    source_checkpoint_step=149999,
                    source_checkpoint_sha256=provenance["source_sha256"],
                    evaluation_id=workflow["run_id"] + "_disk",
                    config_yaml=cfgtext,
                ).items():
                    put(geo, observable + "/" + name, value)
            builder.density(geo, trained, stages["trained"][3])
            builder.pair_correlation(geo, trained, stages["trained"][3])
        else:
            pair = root / "raw/sphere/evaluation/pair"
            plan = read_json(pair / "pair_plan.json")
            if sha256(final) != plan["input"]["source_sha256"]:
                raise ValueError("Sphere pair belongs to another network")
            arrays, summary_pair = inspect_stage(
                pair / "evaluation", plan["config"], final
            )
            pair_ids = [
                register(manifest, p, geo, "sphere_pair_correlation")
                for p in sorted(pair.rglob("*"))
                if p.is_file() and "/smoke/" not in str(p)
            ]
            pair_put = make_writer(h5, schema, {geo: pair_ids})
            import_sphere_pair(h5, pair_put, arrays, summary_pair, plan, pair)
        if not include_fubini:
            h5[f"{geo}/fubini"].attrs.update(
                status="not_computed", note="本轮态距离仍在计算，暂未导入"
            )
            continue
        checkpoint = final
        overlap = load_geometry(
            root / "raw" / geo / "fubini", geo, root, checkpoint=checkpoint
        )
        validate_overlap_parameters(root / "raw" / geo / "fubini", final.parent)
        fubini_ids = [
            register(manifest, p, geo, "fubini_overlap") for p in overlap["sources"]
        ]
        put_overlap = make_writer(h5, schema, {geo: fubini_ids})
        group = h5[f"{geo}/fubini"]
        for name in list(group):
            del group[name]
        values = dict(overlap["arrays"])
        values.update(
            {
                k: overlap["plan"][k]
                for k in [
                    "batch_size",
                    "iterations",
                    "burn_in",
                    "sampler_updates_per_step",
                    "sampling_density",
                    "reference_sha256",
                ]
            }
        )
        values.update({"series/" + k: v for k, v in overlap["statistics"].items()})
        for name, value in values.items():
            put_overlap(geo, "fubini/" + name, value)
            ds = group[name]
            ds.attrs.update(
                description=DESCRIPTIONS.get(name, name),
                unit="rad" if name == "distance" else "",
            )
        group.create_dataset("standard_error", data=h5py.Empty("float64")).attrs[
            "status"
        ] = "not_computed"
        group.attrs.update(status="available", source_files=json.dumps(fubini_ids))


def require_downloaded(root: Path, include_fubini: bool = True) -> None:
    """Require verified downloads for all six independent evaluation tasks.

    Raises:
        ValueError: If the requested input or result fails validation.
    """
    for geo, kind in [
        ("disk", "regular"),
        ("disk", "scan"),
        ("disk", "fubini"),
        ("sphere", "regular"),
        ("sphere", "pair"),
        ("sphere", "fubini"),
    ]:
        if kind == "fubini" and not include_fubini:
            continue
        folder = (
            root
            / "raw"
            / geo
            / ("fubini" if kind == "fubini" else "evaluation/" + kind)
        )
        if not read_json(folder / "download_validation.json")["passed"]:
            raise ValueError("Unverified downloaded result: " + str(folder))


def existing_report(
    destination: Path, identity: str, include_fubini: bool = True
) -> dict[str, Any] | None:
    """Return an already validated publication when its content still matches."""
    report_path = destination / "validation.json"
    if report_path.exists():
        report = read_json(report_path)
        if (
            report.get("workflow_sha256") == identity
            and report.get("include_fubini", True) == include_fubini
            and report.get("passed")
            and sha256(destination / "unified.h5") == report["unified_sha256"]
        ):
            print("Final dataset already validated", flush=True)
            return report
    return None


def refresh_final_groups(base: Any, h5: h5py.File) -> None:
    """Refresh measured groups while preserving copied references and torus."""
    for geo in ["disk", "sphere"]:
        for observable in h5[geo]:
            if observable != "reference_energies":
                base.refresh_group_statuses(h5[f"{geo}/{observable}"])
        h5[geo].attrs["status"] = "available"


def prepare_builder(
    root: Path, baseline: Path, workflow: dict[str, Any], manifest: dict[str, Any]
) -> Any:
    """Prepare and record the existing importer with actual-batch normalization.

    Returns:
        The imported builder module.
    """
    destination = root / "merge"
    # Normalize density using the actual recorded walker count.
    builder_path = destination / "build_dataset.py"
    builder_text = (baseline / "build_dataset.py").read_text()
    builder_text = (
        builder_text.replace("if batch != 8192:", "if batch <= 0:")
        .replace("本次圆盘密度要求实际采样配置数为8192。", "实际采样配置数必须为正。")
        .replace(
            "原始计数按实际8192配置重新归一化\uff1b双精度\uff1b沿用原区间重新拟合边缘",
            "原始计数按记录的实际配置数归一化\uff1b双精度\uff1b沿用原区间拟合边缘",
        )
    )
    builder_path.write_text(builder_text)
    snapshot = root / "submission" / workflow["run_id"] / "collection_code"
    for script in sorted(snapshot.glob("*.py")):
        register(manifest, script, "all", "dataset_assembly_code")
    register(manifest, builder_path, "all", "dataset_assembly_code")
    base = import_builder(builder_path)
    return base


def build(root: Path, baseline: Path, include_fubini: bool = True) -> dict[str, Any]:
    """Publish new sphere/disk data atomically, preserving the entire torus group.

    Returns:
        The verified result or requested metadata.

    Raises:
        ValueError: If the requested input or result fails validation.
    """
    workflow = read_json(root / "workflow.json")
    identity = sha256(root / "workflow.json")
    destination = root / "merge"
    destination.mkdir(exist_ok=True)
    report_path = destination / "validation.json"
    report = existing_report(destination, identity, include_fubini)
    if report is not None:
        return report
    require_downloaded(root, include_fubini)
    old_path = baseline / "unified.h5"
    old_sha = sha256(old_path)
    schema = read_json(baseline / "schema.json")
    manifest = read_json(baseline / "manifest.json")
    original_source = Path(manifest["source_root"])
    for entry in manifest["files"]:
        entry["path"] = os.path.relpath(original_source / entry["path"], root)
        entry["inherited_from_baseline"] = True
    manifest.update(
        source_root=str(root),
        baseline=str(old_path),
        baseline_sha256=old_sha,
        training_run=workflow["training_run"],
    )
    base = prepare_builder(root, baseline, workflow, manifest)
    base.SCHEMA.update(schema["fields"])

    class FinalBuilder(base.Builder):
        def location(self, geometry):
            return root / "raw" / geometry, root / "raw" / geometry / "train"

    temporary = destination / "unified.building.h5"
    with h5py.File(old_path) as old, h5py.File(temporary, "w") as h5:
        validate_references(root, old)
        h5.attrs.update(
            schema_version="1.0",
            created_at=datetime.now(UTC).isoformat(),
            training_run=workflow["training_run"],
            evaluation_run=workflow["run_id"],
            include_fubini=include_fubini,
        )
        old.copy("torus", h5)
        for geo in ["disk", "sphere"]:
            group = h5.create_group(geo)
            empty_copy(old[geo], group)
            del group["reference_energies"]
            old.copy(old[f"{geo}/reference_energies"], group, name="reference_energies")
        save_json(destination / "manifest.json", manifest)
        args = argparse.Namespace(
            source_root=root, output_root=destination, replace=True
        )
        builder = FinalBuilder(args, h5)
        for geo in ["disk", "sphere"]:
            builder.current_files = []
            builder.metadata(geo)
            builder.current_files = []
            builder.training(geo)
        manifest = builder.manifest
        import_evaluations(
            root, destination, workflow, manifest, schema, h5, builder, include_fubini
        )
        scan = root / "raw/disk/evaluation/scan"
        scan_eval = scan / "evaluation"
        original = root / "raw/disk/train/train_ckpt_149999.npz"
        provenance = read_json(scan / "provenance.json")
        if provenance["source_sha256"] != sha256(original):
            raise ValueError("Conditional scan belongs to another network")
        verify_parameters(original, scan_eval / "evaluation_ckpt_000001.npz", 4096)
        scan_ids = [
            register(manifest, p, "disk", "wavefunction_scan")
            for p in scan_eval.rglob("*")
            if p.is_file()
        ]
        scan_put = make_writer(h5, schema, {"disk": scan_ids})
        cfgtext = (scan_eval / "evaluation_config.yaml").read_text()
        for name, value in dict(
            actual_batch_size=4096,
            configured_batch_size=4096,
            n_steps=2,
            burn_in=1000,
            sampler_steps=10,
            seed=161803,
            source_checkpoint_step=149999,
            source_checkpoint_sha256=provenance["source_sha256"],
            evaluation_id=workflow["run_id"] + "_disk_scan",
            config_yaml=cfgtext,
        ).items():
            scan_put("disk", "wavefunction_scan/" + name, value)
        with np.load(scan_eval / "evaluation_digest.npz") as z:
            builder.current_files = scan_ids
            builder.files = {e["path"]: e for e in manifest["files"]}
            builder.wavefunction_scan("disk", scan_eval, dict(z))
        refresh_final_groups(base, h5)
        compare_groups(old["torus"], h5["torus"])
        for geo in ["disk", "sphere"]:
            compare_groups(
                old[f"{geo}/reference_energies"], h5[f"{geo}/reference_energies"]
            )
            np.testing.assert_array_equal(
                h5[f"{geo}/training/step"][:],
                np.arange(150000 if geo == "disk" else 200000),
            )
        h5.flush()
    if sha256(old_path) != old_sha:
        raise ValueError("Original merge changed during import")
    temporary.replace(destination / "unified.h5")
    manifest.update(
        updated_at=datetime.now(UTC).isoformat(),
        evaluation_extension=workflow["run_id"],
    )
    save_json(destination / "manifest.json", manifest)
    save_json(destination / "schema.json", schema)
    report = dict(
        passed=True,
        workflow_sha256=identity,
        unified_sha256=sha256(destination / "unified.h5"),
        torus_unchanged=True,
        references_unchanged=True,
        include_fubini=include_fubini,
        pending_geometries=[] if include_fubini else ["disk/fubini", "sphere/fubini"],
        baseline_sha256=old_sha,
        geometries={
            "disk": dict(training_rows=150000, checkpoints=75),
            "sphere": dict(training_rows=200000, checkpoints=67),
        },
        known_issues=["球面密度及条件波函数扫描未计算", "Fubini 距离标准误差未计算"],
    )
    save_json(report_path, report)
    (destination / "README.md").write_text(
        "# 三种几何统一数据集\n\n"
        "圆盘和球面来自四层网络完整训练及固定参数评估，环面完整沿用原合并数据集。\n\n"
        "训练历史保存在各几何的 `training` 分组，固定网络评估保存在对应物理量分组。"
        "参考能量与原数据集一致，来源及校验值见 `manifest.json`。\n\n"
        "常规评估使用4096个行走样本、预热1000步、正式采样10000步。"
        "检查点态距离使用4096个行走样本、预热100步、采样2000步。"
        "标准误差采用连续时间分块统计，缺少的误差保持为空。\n\n"
        "圆盘关联函数为固定参考位置时关于目标电子位置的函数，其径向结果按距圆心的半径作角平均。"
        "球面关联函数使用无权重电子对计数和精确球面分箱面积归一化。"
        "条件波函数显示幅值不代表单电子密度。\n\n"
        "球面密度、球面条件波函数扫描、自然轨道占据和态距离标准误差未计算。"
        "HDF5 空数据集的 `shape` 为 `None`。\n"
    )
    with (destination / "README.md").open("a", encoding="utf-8") as stream:
        stream.write(
            ""
            if include_fubini
            else "\n圆盘和球面的态距离仍在计算，对应分组保留为空并标记未计算。"
            "其他训练数据和物理量已完成导入与校验。环面分组保持原数据不变。\n"
        )
    return report


def import_sphere_pair(
    h5: h5py.File,
    put: Any,
    arrays: dict[str, Any],
    summary: dict[str, Any],
    plan: dict[str, Any],
    folder: Path,
) -> None:
    radius = plan["config"]["system"].get("radius") or np.sqrt(
        plan["config"]["system"]["flux"] / 2
    )
    values = dict(
        radial__edges=arrays["theta_edges"],
        radial__coordinate=arrays["theta"],
        radial__coordinate_name="两电子夹角，弧度",
        radial__bin_probability=arrays["bin_probability"],
        radial__bin_measure=4 * np.pi * radius**2 * arrays["bin_probability"],
        radial__raw_counts=arrays["raw_counts"],
        radial__value=arrays["value"],
        radial__standard_error=arrays["standard_error"],
        radial__g_finite_n=arrays["g_finite_n"],
        radial__g_finite_n_standard_error=arrays["g_finite_n_standard_error"],
        radial__valid_mask=np.ones(len(arrays["value"]), bool),
        radial__normalization="g=2*C/(M*N^2*p); p=(cos(theta_left)-cos(theta_right))/2",
    )
    values.update(
        {
            k: arrays[k]
            for k in [
                "counts_by_step",
                "block_means",
                "tested_block_sizes",
                "tested_standard_errors",
                "error_stable",
            ]
        }
    )
    values.update(
        sample_count=summary["sample_count"],
        electron_number=summary["electron_number"],
        uncorrelated_level=(summary["electron_number"] - 1)
        / summary["electron_number"],
        n_steps=summary["steps"],
        actual_batch_size=4096,
        configured_batch_size=4096,
        burn_in=1000,
        sampler_steps=10,
        seed=plan["config"]["workflow"]["seed"],
        source_checkpoint_step=199999,
        source_checkpoint_sha256=plan["input"]["source_sha256"],
        evaluation_id=folder.name,
        config_yaml=(folder / "evaluation/evaluation_config.yaml").read_text(),
    )
    for name, value in values.items():
        put("sphere", "pair_correlation/" + name.replace("__", "/"), value)
    group = h5["sphere/pair_correlation"]
    if "legacy_weighted" in group:
        del group["legacy_weighted"]
    for node in [group, group["radial"]]:
        node.attrs.update(
            status="available", note="无权重电子对计数，按精确球面面积归一化"
        )


def plot(root: Path, baseline: Path, include_fubini: bool = True) -> None:
    """Recreate the 28 selected figure pairs using the existing plot programs.

    Raises:
        ValueError: If the requested input or result fails validation.
    """
    original = baseline.parent / "figures"
    destination = root / "figures"
    destination.mkdir(exist_ok=True)
    code_checksums = {}
    for path in original.glob("*.py"):
        target = destination / path.name
        shutil.copy2(path, target)
        code_checksums[path.name] = sha256(path)
    # The current source caption describes historical weighted sphere counts.
    # Update only that caption for the corrected dataset, retaining the plot style.
    pair = destination / "plot_pair_correlation.py"
    text = pair.read_text().replace(
        'title="Sphere: original weighted statistics"', 'title="Sphere"'
    )
    start = text.index(
        '        explanation = (\n            "Original sphere weighted counts'
    )
    stop = text.index("\n    else:", start)
    text = (
        text[:start]
        + (
            "        explanation = (\n"
            '            "Sphere pair correlation uses unweighted pair counts, "\n'
            '            "normalized by exact spherical bin areas. "\n'
            "        )"
        )
        + text[stop:]
    )
    pair.write_text(text)
    if not include_fubini:
        batch = destination / "plot_hall_observables.py"
        batch.write_text(
            batch.read_text().replace("    draw_fubini(groups, writer)\n", "")
        )
    python = original / ".venv/bin/python"
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    image = destination / "img"
    image.mkdir(exist_ok=True)
    for name in [
        "plot_hall_observables.py",
        "plot_multi_geo.py",
        "plot_kinetic_potential.py",
    ]:
        subprocess.run(
            [str(python), str(destination / name)], check=True, cwd=destination, env=env
        )
    # Use the original comparison renderer with only its HDF5 reader adapted.
    comparison = destination / "plot_training_energy.py"
    text = comparison.read_text()
    start = text.index("    path = Path(path).expanduser()")
    stop = text.index("\n\ndef plot_training_energy", start)
    text = (
        text[:start]
        + """    import h5py
    with h5py.File(Path(path).expanduser()) as data:
        return pd.DataFrame({name: data['disk/training/' + name][:] for name in FIELDS})
"""
        + text[stop:]
    )
    comparison.write_text(text)
    subprocess.run(
        [
            str(python),
            str(comparison),
            "--paths",
            str(baseline / "unified.h5"),
            str(root / "merge/unified.h5"),
            "--n",
            "2",
            "--labels",
            "2 layers",
            "4 layers",
            "--window",
            "500",
            "--output",
            str(image / "disk-training-energy-comparison"),
        ],
        check=True,
        cwd=destination,
        env=env,
    )
    expected = {
        p.relative_to(original / "img")
        for p in (original / "img").rglob("*.pdf")
        if p.name != "train-and-extrapolation-sphere-8det-test.pdf"
        and (include_fubini or "fubini" not in p.parts)
    }
    actual = {p.relative_to(image) for p in image.rglob("*.pdf")}
    if actual != expected:
        raise ValueError(
            f"Figure list differs: missing={expected - actual}, "
            f"extra={actual - expected}"
        )
    for path in actual:
        for suffix in [".pdf", ".png"]:
            output = image / path.with_suffix(suffix)
            if not output.is_file() or output.stat().st_size == 0:
                raise ValueError("Missing figure: " + str(output))
    save_json(
        destination / "reproduction.json",
        dict(
            passed=True,
            figure_pairs=len(expected),
            include_fubini=include_fubini,
            merge_sha256=sha256(root / "merge/unified.h5"),
            original_scripts=code_checksums,
            outputs={
                str(p.relative_to(image)): sha256(p)
                for p in image.rglob("*")
                if p.is_file()
            },
        ),
    )
