# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Statistical and import checks using explicitly synthetic evaluation files."""

import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml


@pytest.fixture
def evaluation_modules():
    scripts = Path(__file__).resolve().parents[3] / "scripts"
    modules = []
    for name in ("hall_evaluation", "merge_hall_evaluation"):
        spec = importlib.util.spec_from_file_location(name, scripts / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    yield modules
    for module in modules:
        sys.modules.pop(module.__name__, None)


def test_connected_jackknife_includes_mean_density_bias(evaluation_modules):
    runner, _ = evaluation_modules
    real = np.repeat(np.array([-1.0, 1.0, -1.0, 1.0]), 100)[:, None]
    result = runner.structure_statistics(
        real,
        np.zeros_like(real),
        real**2,
        "torus",
        {
            "structure_factor:electron_number": np.array(1),
            "structure_factor:q": np.array([1.0]),
        },
    )
    np.testing.assert_allclose(result["connected_plugin"], [1.0])
    np.testing.assert_allclose(result["connected"], [4 / 3])
    np.testing.assert_allclose(result["shell_connected"], result["connected"])


def test_ed_import_rejects_incompatible_energy_conventions(
    evaluation_modules, tmp_path
):
    runner, _ = evaluation_modules
    entry = {
        "energy": 6.7,
        "n_electrons": 6,
        "flux": 15,
        "interaction_strength": 1,
        "energy_unit": "hbar_omega_c_total",
        "background_convention": "electron_electron_only",
        "basis": {"num_landau_levels": 1},
        "sector": {"Lz": 0},
        "source": "synthetic test",
    }
    path = tmp_path / "reference.yaml"
    path.write_text(yaml.safe_dump({"sphere": {"LLL_ED": entry}}), encoding="utf-8")
    output = tmp_path / "reference.json"
    runner.import_references(path, output)
    np.testing.assert_allclose(
        json.loads(output.read_text(encoding="utf-8"))["reference_energies"]["sphere"][
            "LLL_ED"
        ]["energy"],
        6.7,
    )
    entry["background_convention"] = "background_corrected"
    path.write_text(yaml.safe_dump({"sphere": {"LLL_ED": entry}}), encoding="utf-8")
    with pytest.raises(ValueError, match="background"):
        runner.import_references(path, output)


def test_complete_evaluations_import_with_backup_and_exact_sources(
    evaluation_modules, tmp_path
):
    runner, merger = evaluation_modules
    sources, runs, merged = (tmp_path / name for name in ("sources", "runs", "merge"))
    merged.mkdir()
    runner.save_json(
        merged / "manifest.json", {"source_root": str(sources), "files": []}
    )
    runner.save_json(merged / "schema.json", {"fields": {}})
    runner.save_json(merged / "validation.json", {"passed": True})
    with h5py.File(merged / "unified.h5", "w") as f:
        for geometry in runner.SOURCES:
            for name in ("energy", "structure_factor", "reference_energies"):
                f.create_group(geometry + "/" + name)
            f.create_dataset(geometry + "/one_rdm/matrix", data=np.eye(54))
            f.create_dataset(
                geometry + "/one_rdm/natural_occupations", data=h5py.Empty("f8")
            )
            f.create_dataset(geometry + "/metadata/retained", data=123)
            for reference in ("LLL_ED", "2LL_ED", "3LL_ED"):
                f.create_dataset(
                    geometry + "/reference_energies/" + reference, data=np.nan
                )
    before = runner.sha256(merged / "unified.h5")
    for geometry, relative in runner.SOURCES.items():
        source = sources / relative
        source.parent.mkdir(parents=True)
        params = {
            "params/test": np.array([0.4, 0.7]),
            "batched_data/data/electrons": np.zeros((4, 6, 2)),
        }
        np.savez(source, **params)
        root = runs / geometry
        root.mkdir(parents=True)
        runner.save_json(root / "completion.json", {"status": "complete"})
        runner.save_json(
            root / "provenance.json", {"source_sha256": runner.sha256(source)}
        )
        for state in ("trained", "laughlin"):
            stage = root / state
            stage.mkdir()
            config = {
                "run": {"iterations": 1000, "burn_in": 1000},
                "workflow": {"batch_size": 4, "seed": 42},
                "sampler": {"steps": 10},
                "system": {"flux": 15 if geometry == "sphere" else 18},
            }
            (stage / "evaluation_config.yaml").write_text(
                yaml.safe_dump(config), encoding="utf-8"
            )
            runner.save_json(
                stage / "run_manifest.json", {"status": "complete", "config": config}
            )
            np.savez(stage / "evaluation_ckpt_000999.npz", step=999, **params)
            digest = {}
            with h5py.File(stage / "evaluation_stats.h5", "w") as f:
                f["total_energy"] = np.full(1000, 5.0 + 0j)
                f["total_energy_real"] = np.full(1000, 5.0)
                f["energy:kinetic"] = np.full(1000, 3.0 + 0j)
                f["energy:potential"] = np.full(1000, 2.0)
                if state == "trained":
                    f["structure_factor:rho_real"] = np.full(
                        (1000, 1, 1) if geometry == "disk" else (1000, 1), 6.0
                    )
                    f["structure_factor:rho_imag"] = np.zeros(
                        f["structure_factor:rho_real"].shape
                    )
                    f["structure_factor:power"] = np.full(
                        f["structure_factor:rho_real"].shape, 36.0
                    )
                    digest = {
                        "structure_factor:electron_number": np.array(6),
                        "structure_factor:q": np.array([0.0]),
                    }
                    if geometry == "sphere":
                        digest.update(
                            {
                                "structure_factor:ell": np.array([0]),
                                "structure_factor:mode_ell": np.array([0]),
                                "structure_factor:mode_weight": np.array([1 / 6]),
                            }
                        )
                    if geometry != "torus":
                        size = 60 if geometry == "disk" else 54
                        counts = [20, 20, 20] if geometry == "disk" else [16, 18, 20]
                        matrix = np.diag(np.r_[np.ones(6), np.zeros(size - 6)]).astype(
                            np.complex64
                        )
                        f.create_dataset(
                            "one_rdm",
                            data=np.broadcast_to(matrix, (1000, size, size)),
                            compression="gzip",
                        )
                        digest["one_rdm:basis_landau_level"] = np.repeat(
                            np.arange(3), counts
                        )
            np.savez(stage / "evaluation_digest.npz", **digest)
            runner.summarize(stage, geometry)
    report = merger.import_results(runs, merged)
    assert report["passed"]
    assert runner.sha256(Path(report["backup"]) / "unified.h5") == before
    with h5py.File(merged / "unified.h5") as f:
        for geometry in runner.SOURCES:
            np.testing.assert_allclose(
                f[geometry + "/reference_energies/Laughlin"][()], 5.0
            )
            assert f[geometry + "/metadata/retained"][()] == 123
            assert np.isnan(f[geometry + "/reference_energies/3LL_ED"][()])
        np.testing.assert_allclose(
            f["disk/one_rdm/landau_level_occupations"][:], [6, 0, 0]
        )
        assert f["sphere/one_rdm/matrix"].shape == (54, 54)
