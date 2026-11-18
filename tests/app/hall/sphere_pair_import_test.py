# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Exact counts, additive dataset import, and original-data preservation."""

import importlib
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml


@pytest.fixture
def recorded_pair_run(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts"))
    module = importlib.import_module("sphere_pair_correlation")
    runner = importlib.import_module("hall_evaluation")
    sources, root, merge = (tmp_path / name for name in ("sources", "run", "merge"))
    evaluation = root / "evaluation"
    evaluation.mkdir(parents=True)
    source = sources / runner.SOURCES["sphere"]
    source.parent.mkdir(parents=True)
    params = {"params/example": np.array([0.7, 0.2], dtype=np.float32)}
    np.savez(
        source, **params, **{"batched_data/data/electrons": np.zeros((8192, 6, 2))}
    )
    steps, batch = 1000, 4096
    config = dict(
        run=dict(iterations=steps, burn_in=1000),
        workflow=dict(batch_size=batch, seed=17),
        system=dict(nspins=[6, 0], flux=15, radius=None),
        estimators=dict(pair_correlation=dict(bins=200)),
        sampler=dict(steps=10),
    )
    _, probability = module.bin_geometry(200)
    counts = np.random.default_rng(19).multinomial(batch * 15, probability, size=steps)
    with h5py.File(evaluation / "evaluation_stats.h5", "w") as f:
        f["pair_correlation:counts_per_configuration"] = (counts / batch).astype(
            np.float32
        )
        f["pair_correlation:pair_count"] = np.full(steps, 15, dtype=np.float32)
    np.savez(
        evaluation / "evaluation_digest.npz",
        pair_correlation=2 * counts.mean(axis=0) / (batch * 36 * probability),
    )
    np.savez(
        evaluation / "evaluation_ckpt_000999.npz",
        step=999,
        **params,
        **{"batched_data/data/electrons": np.zeros((batch, 6, 2))},
    )
    arrays, summary = module.inspect_stage(evaluation, config, source)
    np.savez_compressed(root / "pair_statistics.npz", **arrays)
    runner.save_json(root / "pair_summary.json", summary)
    runner.save_json(root / "completion.json", summary)
    runner.save_json(
        root / "pair_plan.json",
        dict(config=config, input=dict(source_sha256=runner.sha256(source))),
    )
    runner.save_json(root / "code_checksums.json", {})
    runner.save_json(evaluation / "validation.json", summary)
    (evaluation / "evaluation_config.yaml").write_text(yaml.safe_dump(config))
    paths = sorted(p for p in root.rglob("*") if p.is_file())
    (root / "result_checksums.sha256").write_text(
        "".join(
            runner.sha256(p) + "  " + str(p.relative_to(root)) + "\n" for p in paths
        )
    )
    merge.mkdir()
    runner.save_json(merge / "manifest.json", dict(source_root=str(sources), files=[]))
    runner.save_json(merge / "schema.json", dict(fields={}))
    runner.save_json(
        merge / "validation.json",
        dict(known_issues=["球面对关联等待修复", "another issue"]),
    )
    (merge / "README.md").write_text("数据集说明\n")
    (merge / "待补充清单.md").write_text("用户决定是否修复对关联，检查分块误差是否稳定")
    with h5py.File(merge / "unified.h5", "w") as f:
        f["sphere/pair_correlation/radial/raw_counts"] = np.full(200, -123.0)
        f["sphere/metadata/retained"] = 23
        f["disk/reference_energies/LLL_ED"] = -4.2627686054372385
    yield module, runner, root, merge, source, config, counts
    sys.modules.pop("sphere_pair_correlation", None)


def test_import_preserves_ed_and_legacy_counts_and_is_idempotent(recorded_pair_run):
    module, runner, root, merge, _, _, counts = recorded_pair_run
    before = runner.sha256(merge / "unified.h5")
    report = module.import_result(root, merge)
    assert report["passed"] and report["total_pair_count"] == 1000 * 4096 * 15
    assert runner.sha256(Path(report["backup"]) / "unified.h5") == before
    with h5py.File(merge / "unified.h5") as f:
        pair = f["sphere/pair_correlation"]
        assert pair.attrs["status"] == "available"
        np.testing.assert_array_equal(pair["radial/raw_counts"][:], counts.sum(axis=0))
        np.testing.assert_array_equal(pair["counts_by_step"][:], counts)
        np.testing.assert_array_equal(pair["legacy_weighted/raw_counts"][:], -123.0)
        assert f["sphere/metadata/retained"][()] == 23
        np.testing.assert_array_equal(
            f["disk/reference_energies/LLL_ED"][()], -4.2627686054372385
        )
        np.testing.assert_allclose(
            np.dot(pair["radial/value"][:], pair["radial/bin_probability"][:]), 5 / 6
        )
    digest = runner.sha256(merge / "unified.h5")
    assert module.import_result(root, merge) == report
    assert runner.sha256(merge / "unified.h5") == digest


def test_incomplete_pair_counts_are_rejected(recorded_pair_run):
    module, _, root, _, source, config, _ = recorded_pair_run
    with h5py.File(root / "evaluation/evaluation_stats.h5", "r+") as f:
        f["pair_correlation:counts_per_configuration"][10, -1] += 1 / 4096
    with pytest.raises(ValueError, match="Pair counts"):
        module.inspect_stage(root / "evaluation", config, source)
