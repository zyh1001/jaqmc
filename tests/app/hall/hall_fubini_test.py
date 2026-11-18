# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Known overlaps and geometry-specific overlap workflow invariants."""

from operator import itemgetter

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.app.hall.data import HallData
from jaqmc.app.hall.estimator.disk.fubini import DiskFubini, FubiniParams
from jaqmc.app.hall.estimator.sphere.fubini import SphereFubini
from jaqmc.app.hall.estimator.torus.fubini import TorusFubini
from jaqmc.app.hall.overlap import HallOverlapWorkflow
from jaqmc.data import BatchedData
from jaqmc.estimator.base import EstimatorPipeline
from jaqmc.utils import parallel_jax
from jaqmc.utils.config import ConfigManager


@pytest.mark.parametrize("estimator_class", [DiskFubini, SphereFubini, TorusFubini])
@pytest.mark.parametrize(
    "current,reference,shift,expected",
    [
        ([1, 1], [1j, 1j], 0.0, 1.0),
        ([2, 1], [1, 2], 0.0, 0.8),
        ([2, 1], [1, 2], 100.0, 0.8),
        ([1, 1], [1, -1], 0.0, 0.0),
    ],
)
def test_exact_two_configuration_overlaps(
    estimator_class, current, reference, shift, expected
):
    # All cases have constant |psi * psi_reference| on the two configurations.
    def logpsi(params, data):
        return params["log_values"][data.electrons[0, 0].astype(int)]

    params = FubiniParams(
        {"log_values": jnp.log(jnp.asarray(current, dtype=jnp.complex64)) + shift},
        {"log_values": jnp.log(jnp.asarray(reference, dtype=jnp.complex64))},
    )
    electrons = jnp.zeros((8, 1, 2)).at[:, 0, 0].set(jnp.arange(8) % 2)
    data = BatchedData(
        data=HallData(electrons=electrons), fields_with_batch=["electrons"]
    )
    estimator = estimator_class(f_log_psi=logpsi)
    pipeline = EstimatorPipeline({"fubini": estimator})
    key = jax.random.key(0)
    state = pipeline.init(data, key)

    def evaluate(electrons):
        local = BatchedData(
            data=HallData(electrons=electrons), fields_with_batch=["electrons"]
        )
        return pipeline.evaluate(params, local, state, key)[0]

    sharded = parallel_jax.jit_sharded(
        evaluate,
        in_specs=(parallel_jax.DATA_PARTITION,),
        out_specs=parallel_jax.SHARE_PARTITION,
    )
    electrons = jax.device_put(
        electrons, parallel_jax.make_sharding(parallel_jax.DATA_PARTITION)
    )
    statistics = sharded(electrons)
    digest = pipeline.finalize_stats(jax.tree.map(itemgetter(None), statistics), state)
    np.testing.assert_allclose(digest["cosine"], expected, atol=3e-6)
    np.testing.assert_allclose(
        digest["fubini_distance"], np.arccos(expected), atol=3e-6
    )
    # Sampling amplitude must square to the requested symmetric density.
    point = HallData(electrons=jnp.zeros((1, 2)))
    amplitude = estimator.log_overlap_amplitude(params, point)
    np.testing.assert_allclose(
        2 * amplitude,
        (logpsi(params.current, point) + logpsi(params.reference, point)).real,
    )


@pytest.mark.parametrize(
    "geometry,flux,estimator_class",
    [("disk", 6, DiskFubini), ("sphere", 3, SphereFubini), ("torus", 6, TorusFubini)],
)
def test_analytic_self_overlap_workflow(tmp_path, geometry, flux, estimator_class):
    config = {
        "system": {
            "module": f"jaqmc.app.hall.config:Hall{geometry.capitalize()}Config",
            "nspins": [2, 0],
            "flux": flux,
        },
        "wf": {"module": f"jaqmc.app.hall.wavefunction.{geometry}.laughlin"},
        "workflow": {"save_path": str(tmp_path), "batch_size": 8, "seed": 9},
        "run": {"iterations": 2, "burn_in": 2},
        "sampler": {"steps": 1},
    }
    workflow = HallOverlapWorkflow(ConfigManager(config))
    assert isinstance(
        workflow.evaluation_stage.estimators.estimators["fubini"], estimator_class
    )
    workflow()
    with np.load(tmp_path / "overlap_digest.npz") as digest:
        np.testing.assert_allclose(digest["cosine"], 1.0, atol=2e-6)
        np.testing.assert_allclose(digest["fubini_distance"], 0.0, atol=2e-6)
