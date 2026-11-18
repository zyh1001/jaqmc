# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Check final-data source identity, schema clearing, and preservation behavior."""

import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "final_dataset", SCRIPTS / "final_dataset.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_parameter_identity_rejects_another_network(tmp_path):
    source, result = tmp_path / "train.npz", tmp_path / "evaluation.npz"
    arrays = {
        "params/weight": np.array([1.0, 2.0]),
        "batched_data/data/electrons": np.zeros((4, 2, 2)),
    }
    np.savez(source, **arrays)
    np.savez(result, **arrays)
    module.verify_parameters(source, result, 4)
    arrays["params/weight"][0] = 3.0
    np.savez(result, **arrays)
    with pytest.raises(AssertionError):
        module.verify_parameters(source, result, 4)
    np.savez(
        result,
        **{
            "params/other": np.array([1.0]),
            "batched_data/data/electrons": np.zeros((4, 2, 2)),
        },
    )
    with pytest.raises(ValueError, match="names differ"):
        module.verify_parameters(source, result, 4)


@pytest.mark.parametrize("changed", [None, "current", "reference", "walkers"])
def test_overlap_parameters_check_both_networks(tmp_path, changed):
    train = tmp_path / "train"
    train.mkdir()
    np.savez(train / "train_ckpt_000099.npz", **{"params/weight": [1.0, 2.0]})
    np.savez(train / "train_ckpt_000199.npz", **{"params/weight": [3.0, 4.0]})
    folder = tmp_path / "fubini"
    step = folder / "step_0000099"
    step.mkdir(parents=True)
    (folder / "fubini_plan.json").write_text(
        json.dumps(
            {
                "reference_checkpoint": "/remote/train_ckpt_000199.npz",
                "iterations": 2,
                "batch_size": 4,
                "checkpoints": [
                    {
                        "step": 99,
                        "input": {"source_checkpoint": "/remote/train_ckpt_000099.npz"},
                    }
                ],
            }
        )
    )
    arrays = {
        "params/current/weight": np.array([1.0, 2.0]),
        "params/reference/weight": np.array([3.0, 4.0]),
        "batched_data/data/electrons": np.zeros((4, 2, 2)),
    }
    if changed in ("current", "reference"):
        arrays[f"params/{changed}/weight"][0] += 1
    if changed == "walkers":
        arrays["batched_data/data/electrons"] = np.zeros((3, 2, 2))
    np.savez(step / "overlap_ckpt_000001.npz", **arrays)
    if changed is None:
        module.validate_overlap_parameters(folder, train)
    else:
        with pytest.raises((AssertionError, ValueError)):
            module.validate_overlap_parameters(folder, train)


def test_clean_measurements_and_exact_geometry_copy(tmp_path):
    with (
        h5py.File(tmp_path / "old.h5", "w") as old,
        h5py.File(tmp_path / "new.h5", "w") as new,
    ):
        geo = old.create_group("torus")
        geo.attrs["status"] = "available"
        geo["value"] = np.array([1.0, np.nan])
        geo["value"].attrs.update(status="available", source_files='["F0001"]')
        geo.create_dataset("missing", data=h5py.Empty("float64"))
        old.copy("torus", new)
        module.compare_groups(old["torus"], new["torus"])
        clean = new.create_group("sphere")
        module.empty_copy(geo, clean)
        assert clean["value"].shape is None
        assert clean["value"].attrs["source_files"] == "[]"
        assert clean["value"].attrs["status"] == "not_computed"
        new["torus/value"][0] = 7.0
        with pytest.raises(AssertionError):
            module.compare_groups(old["torus"], new["torus"])


def test_phase_winding_detects_a_known_single_vortex(tmp_path):
    axis = np.linspace(-1.0, 1.0, 12)
    x, y = np.meshgrid(axis, axis)
    z = x + 1j * y
    digest = {
        "wavefunction_scan:x": axis,
        "wavefunction_scan:y": axis,
        "wavefunction_scan:log_abs": np.log(abs(z)),
        "wavefunction_scan:phase": np.angle(z),
        "wavefunction_scan:selection_samples": 8192,
        "wavefunction_scan:selected_log_abs": 0.0,
    }
    summary, table = module.scan_postprocess(digest, tmp_path)
    assert summary["vortex_count"] == 1
    np.testing.assert_allclose(table[0], [0.0, 0.0, 1.0], atol=1e-14)


def test_source_registration_is_idempotent(tmp_path):
    p = tmp_path / "value.json"
    p.write_text(json.dumps({"result": 1}))
    manifest = {"source_root": str(tmp_path), "files": []}
    first = module.register(manifest, p, "disk", "evaluation")
    second = module.register(manifest, p, "disk", "evaluation")
    assert first == second
    assert len(manifest["files"]) == 1
    assert manifest["files"][0]["sha256"] == module.sha256(p)


def test_single_geometry_import_uses_explicit_directory_and_preserves_references(
    tmp_path, monkeypatch
):
    import merge_hall_evaluation as merger

    folder = tmp_path / "new_disk_evaluation"
    folder.mkdir()
    (folder / "evaluation_config.yaml").write_text("wf: {num_layers: 4}\n")
    with h5py.File(folder / "evaluation_stats.h5", "w") as stats:
        stats["one_rdm"] = np.ones((3, 1, 1), dtype=complex)
    config = dict(
        workflow=dict(batch_size=4096, seed=1),
        run=dict(iterations=10000, burn_in=1000),
        sampler=dict(steps=10),
    )
    statistics = {
        "one_rdm:" + key: np.array([1.0])
        for key in [
            "matrix:mean",
            "diagonal:mean",
            "landau_level_occupations:mean",
            "landau_level_occupations:standard_error",
            "trace:mean",
            "matrix:standard_error",
            "matrix:imaginary_standard_error",
        ]
    }
    stages = {
        "trained": (
            {"config": config},
            {"one_rdm": {"hermiticity_error": 0.0}},
            statistics,
            {},
        )
    }
    monkeypatch.setattr(
        merger,
        "write_energy",
        lambda put, geo, *_: put(geo, "energy/summary/total_energy", 3.0),
    )
    with h5py.File(tmp_path / "dataset.h5", "w") as data:
        for observable in ["energy", "structure_factor", "one_rdm"]:
            data.require_group("disk/" + observable)
        data.create_dataset(
            "disk/one_rdm/natural_occupations", data=h5py.Empty("float64")
        )
        data["disk/reference_energies/Laughlin"] = -4.25
        data["torus/value"] = np.array([7.0, 8.0])
        put = merger.make_writer(data, {"fields": {}}, {"disk": ["F0001"]})
        merger.write_geometry(
            data,
            put,
            "disk",
            {
                "source_checkpoint": "/new/train_ckpt_000123.npz",
                "source_sha256": "new-network",
            },
            stages,
            tmp_path,
            evaluation_directory=folder,
        )
        assert data["disk/energy/source_checkpoint_step"][()] == 123
        np.testing.assert_equal(data["disk/reference_energies/Laughlin"][()], -4.25)
        np.testing.assert_array_equal(data["torus/value"][:], [7.0, 8.0])
        assert "sphere" not in data


def test_pending_overlap_does_not_block_other_data_or_full_refresh(tmp_path):
    for geo, kind in [
        ("disk", "regular"),
        ("disk", "scan"),
        ("sphere", "regular"),
        ("sphere", "pair"),
    ]:
        directory = tmp_path / "raw" / geo / "evaluation" / kind
        directory.mkdir(parents=True)
        (directory / "download_validation.json").write_text('{"passed":true}')
    module.require_downloaded(tmp_path, include_fubini=False)
    with pytest.raises(FileNotFoundError):
        module.require_downloaded(tmp_path, include_fubini=True)
    merge = tmp_path / "merge"
    merge.mkdir()
    (merge / "unified.h5").write_bytes(b"example dataset identity")
    report = dict(
        workflow_sha256="run-identity",
        include_fubini=False,
        passed=True,
        unified_sha256=module.sha256(merge / "unified.h5"),
    )
    (merge / "validation.json").write_text(json.dumps(report))
    assert module.existing_report(merge, "run-identity", include_fubini=False) == report
    assert module.existing_report(merge, "run-identity", include_fubini=True) is None
