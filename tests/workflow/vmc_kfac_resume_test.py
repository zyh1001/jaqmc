# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Check restoring training data with new KFAC curvature state."""

import logging
from dataclasses import replace

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.app.hall.data import HallData
from jaqmc.data import BatchedData
from jaqmc.optimizer.kfac import KFACOptimizer
from jaqmc.optimizer.kfac.kfac import KFACState
from jaqmc.utils.checkpoint import NumPyCheckpointManager, tree_to_npz
from jaqmc.workflow.stage.vmc import VMCState, VMCWorkStage, VMCWorkStageConfig


def make_state(value=1.0):
    return VMCState(
        params={"params": {"weight": jnp.array([value], dtype=jnp.float32)}},
        batched_data=BatchedData(
            HallData(electrons=jnp.full((2, 1, 2), value)), ("electrons",)
        ),
        sampler_state={"width": jnp.array([value])},
        estimator_state={},
        opt_state=KFACState(
            optax=({"count": jnp.array(150000, dtype=jnp.int32)},),
            kfac={"old_factor": jnp.array([value])},
        ),
    )


def make_stage(monkeypatch):
    stage = object.__new__(VMCWorkStage)
    stage.config = VMCWorkStageConfig(reset_kfac_state_on_restore=True)
    stage.optimizer = KFACOptimizer()
    stage.logger = logging.getLogger(__name__)
    observed = {}

    def initialize(_self, params, *, batched_data, rngs):
        del rngs
        observed["params"] = params
        observed["electrons"] = batched_data.data.electrons
        return KFACState(optax=(), kfac={"new_factor": jnp.array([9.0])})

    monkeypatch.setattr(KFACOptimizer, "init", initialize)
    return stage, observed


def save(path, state):
    np.savez(path, step=149999, **tree_to_npz(state))
    return NumPyCheckpointManager(path.parent, path, prefix="train")


def test_resume_restores_saved_arrays_and_reinitializes_kfac(tmp_path, monkeypatch):
    original = make_state(3.0)
    ckpt = save(tmp_path / "train_ckpt_149999.npz", original)
    stage, observed = make_stage(monkeypatch)
    step, result = stage.restore_for_run(ckpt, make_state(), jax.random.key(0))
    assert step == 150000
    np.testing.assert_array_equal(result.params["params"]["weight"], [3.0])
    np.testing.assert_array_equal(observed["params"]["params"]["weight"], [3.0])
    np.testing.assert_array_equal(
        observed["electrons"], original.batched_data.data.electrons
    )
    np.testing.assert_array_equal(result.sampler_state["width"], [3.0])
    assert int(result.opt_state.optax[0]["count"]) == 150000
    np.testing.assert_array_equal(result.opt_state.kfac["new_factor"], [9.0])


@pytest.mark.parametrize("change", ["name", "shape", "dtype", "optax"])
def test_resume_rejects_incompatible_checkpoint(tmp_path, monkeypatch, change):
    stage, _ = make_stage(monkeypatch)
    arrays = tree_to_npz(make_state(3.0))
    key = "params/params/weight"
    if change == "name":
        arrays["params/params/other"] = arrays.pop(key)
    elif change == "shape":
        arrays[key] = np.ones((2,), dtype=np.float32)
    elif change == "dtype":
        arrays[key] = np.ones((1,), dtype=np.float64)
    else:
        arrays.pop("opt_state/optax/0/count")
    path = tmp_path / "train_ckpt_149999.npz"
    np.savez(path, step=149999, **arrays)
    with pytest.raises(ValueError):
        stage.restore_for_run(
            NumPyCheckpointManager(tmp_path, path, prefix="train"),
            make_state(),
            jax.random.key(0),
        )


def test_default_resume_keeps_saved_kfac_state(tmp_path, monkeypatch):
    original = make_state(3.0)
    ckpt = save(tmp_path / "train_ckpt_149999.npz", original)
    stage, observed = make_stage(monkeypatch)
    stage.config = replace(stage.config, reset_kfac_state_on_restore=False)
    step, result = stage.restore_for_run(ckpt, make_state(), jax.random.key(0))
    assert step == 150000
    np.testing.assert_array_equal(result.opt_state.kfac["old_factor"], [3.0])
    assert not observed
