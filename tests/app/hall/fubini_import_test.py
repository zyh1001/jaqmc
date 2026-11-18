# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Check overlap imports against synthetic completed and corrupted runs."""

import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


@pytest.fixture
def completed_run(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[3] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "fubini_import_under_test", scripts / "merge_hall_fubini.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source, run, merge = (tmp_path / name for name in ("source", "run", "merge"))
    merge.mkdir()
    module.save_json(merge / "manifest.json", {"source_root": str(source), "files": []})
    module.save_json(merge / "schema.json", {"fields": {}})
    module.save_json(merge / "validation.json", {"passed": True, "checks": []})
    (merge / "README.md").write_text("Synthetic dataset\n")
    with h5py.File(merge / "unified.h5", "w") as handle:
        for geo, relative in module.SOURCES.items():
            final = source / relative
            final.parent.mkdir(parents=True)
            final_step = int(final.stem.rsplit("_", 1)[1])
            checkpoints = [final.with_name("train_ckpt_000099.npz"), final]
            for checkpoint in checkpoints:
                np.savez(checkpoint, params=np.array([1.0, 2.0]))
            directory = run / geo
            directory.mkdir(parents=True)
            items, records = [], []
            arrays = {
                key: []
                for key in (
                    "step",
                    "distance",
                    "cosine",
                    "modulus_cosine",
                    "phase_factor",
                )
            }
            for index, checkpoint in enumerate(checkpoints):
                step = int(checkpoint.stem.rsplit("_", 1)[1])
                step_dir = directory / f"step_{step:07d}"
                step_dir.mkdir()
                log_mean, phase = (0.2, 0.8) if index == 0 else (0.0, 1.0)
                modulus = np.exp(-log_mean)
                cosine = modulus * phase
                distance = np.arccos(cosine)
                with h5py.File(step_dir / "overlap_stats.h5", "w") as stats:
                    for key, value in zip(
                        module.STAT_FIELDS,
                        (log_mean, log_mean, complex(phase)),
                        strict=True,
                    ):
                        stats["fubini:" + key] = np.full(4, value)
                np.savez(
                    step_dir / "overlap_digest.npz",
                    fubini_distance=distance,
                    cosine=cosine,
                    cos_gamma_modulus=modulus,
                    f_phase=phase,
                )
                (step_dir / "requested_config.yaml").write_text("synthetic: true\n")
                items.append(
                    {
                        "step": step,
                        "input": {
                            "source_checkpoint": str(checkpoint),
                            "source_sha256": module.sha256(checkpoint),
                        },
                    }
                )
                records.append(
                    {
                        "step": step,
                        "distance": distance,
                        "digest_sha256": module.sha256(step_dir / "overlap_digest.npz"),
                    }
                )
                for key, value in zip(
                    arrays, (step, distance, cosine, modulus, phase), strict=True
                ):
                    arrays[key].append(value)
            identity = {
                "geometry": geo,
                "reference_step": final_step,
                "reference_sha256": module.sha256(final),
            }
            module.save_json(
                directory / "fubini_plan.json",
                dict(
                    identity,
                    checkpoints=items,
                    sampling_density="abs(psi_checkpoint * psi_final)",
                    iterations=4,
                    batch_size=2,
                    burn_in=1,
                    sampler_updates_per_step=10,
                ),
            )
            module.save_json(
                directory / "fubini_summary.json",
                dict(identity, records=records, total_configurations_per_checkpoint=8),
            )
            np.savez(
                directory / "fubini_series.npz", **arrays, reference_step=final_step
            )
            handle[f"{geo}/metadata/final_checkpoint_step"] = final_step
            handle[f"{geo}/training/total_energy_real"] = [1.0, 2.0, 3.0]
            handle[f"{geo}/reference_energies/3LL_ED"] = -4.283048198240584
    yield module, source, run, merge
    sys.modules.pop("hall_evaluation", None)


def test_complete_import_preserves_data_and_is_idempotent(completed_run):
    module, _, run, merge = completed_run
    report = module.import_results(run, merge)
    assert report["passed"] and report["checkpoints"] == dict.fromkeys(
        module.SOURCES, 2
    )
    with h5py.File(merge / "unified.h5", "r") as handle:
        for geo in module.SOURCES:
            np.testing.assert_array_equal(
                handle[f"{geo}/training/total_energy_real"], [1, 2, 3]
            )
            np.testing.assert_array_equal(
                handle[f"{geo}/reference_energies/3LL_ED"][()], -4.283048198240584
            )
            assert handle[f"{geo}/fubini/series/phase_mean"].shape == (2, 4)
            assert handle[f"{geo}/fubini/standard_error"].shape is None
            assert handle[f"{geo}/fubini/distance"][-1] == 0
    checksum = module.sha256(merge / "unified.h5")
    assert module.import_results(run, merge) == report
    assert module.sha256(merge / "unified.h5") == checksum
    assert len(list(merge.parent.glob("merge_fubini_backup_*"))) == 1


def test_mixed_sampling_lengths_preserve_completed_statistics(completed_run):
    module, _, run, merge = completed_run
    geometry = run / "disk"
    plan_path = geometry / "fubini_plan.json"
    summary_path = geometry / "fubini_summary.json"
    plan = json.loads(plan_path.read_text())
    plan["checkpoints"][1].update(iterations=2, burn_in=0)
    plan["reference_checkpoint"] = plan["checkpoints"][1]["input"]["source_checkpoint"]
    for item in plan["checkpoints"]:
        item["directory"] = str(geometry / f"step_{item['step']:07d}")
    final_dir = geometry / f"step_{plan['reference_step']:07d}"
    with h5py.File(final_dir / "overlap_stats.h5", "r+") as handle:
        for key in list(handle):
            values = handle[key][:2]
            del handle[key]
            handle[key] = values
    module.save_json(plan_path, plan)
    spec = importlib.util.spec_from_file_location(
        "fubini_driver_under_test", Path(module.__file__).with_name("hall_fubini.py")
    )
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    driver.collect(geometry)
    summary = json.loads(summary_path.read_text())
    assert summary["total_configurations_per_checkpoint"] is None
    assert summary["records"][1]["total_configurations"] == 4
    report = module.import_results(run, merge)
    assert report["passed"]
    with h5py.File(merge / "unified.h5", "r") as handle:
        group = handle["disk/fubini"]
        np.testing.assert_array_equal(group["iterations_per_checkpoint"], [4, 2])
        np.testing.assert_array_equal(group["burn_in_per_checkpoint"], [1, 0])
        np.testing.assert_array_equal(
            group["total_configurations_per_checkpoint"], [8, 4]
        )
        values = group["series/phase_mean"][()]
        np.testing.assert_array_equal(values[0], np.full(4, 0.8))
        np.testing.assert_array_equal(values[1, :2], [1, 1])
        assert np.isnan(values[1, 2:]).all()


@pytest.mark.parametrize("corruption", ["source", "sampling", "digest"])
def test_incomplete_or_changed_run_never_modifies_merge(completed_run, corruption):
    module, source, run, merge = completed_run
    before = module.sha256(merge / "unified.h5")
    if corruption == "source":
        (source / module.SOURCES["sphere"]).write_bytes(b"changed checkpoint")
    elif corruption == "sampling":
        path = run / "sphere/step_0000099/overlap_stats.h5"
        with h5py.File(path, "r+") as handle:
            del handle["fubini:phase_mean"]
            handle["fubini:phase_mean"] = np.ones(3, dtype=complex)
    else:
        (run / "sphere/step_0000099/overlap_digest.npz").write_bytes(b"changed digest")
    with pytest.raises(ValueError):
        module.import_results(run, merge)
    assert module.sha256(merge / "unified.h5") == before
    assert not list(merge.parent.glob("merge_fubini_backup_*"))
