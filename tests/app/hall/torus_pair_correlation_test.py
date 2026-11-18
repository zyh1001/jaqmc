# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Torus pair correlations match the full sample across devices and steps."""

import jax
import numpy as np
from jax import numpy as jnp

from jaqmc.app.hall.data import HallData
from jaqmc.app.hall.estimator.torus import TorusPairCorrelation
from jaqmc.data import BatchedData
from jaqmc.estimator.base import EstimatorPipeline
from jaqmc.utils import parallel_jax


def test_sharded_pair_correlation_matches_full_batch() -> None:
    electrons = np.zeros((2 * jax.device_count(), 2, 2), dtype=np.float32)
    electrons[:, 1, 0] = np.linspace(0.1, 0.4, len(electrons))
    electrons[:, 1, 1] = np.linspace(0.4, 0.1, len(electrons))
    batch = BatchedData(
        data=HallData(electrons=jnp.asarray(electrons)), fields_with_batch=["electrons"]
    )
    estimator = TorusPairCorrelation(bins_u=4, bins_v=4)
    pipeline = EstimatorPipeline({"pair": estimator})
    shared = parallel_jax.make_sharding(parallel_jax.SHARE_PARTITION)
    state = jax.device_put(pipeline.init(batch, jax.random.key(0)), shared)
    assert state["pair"]["histogram"].shape == (4, 4)
    assert state["pair"]["compensation"].shape == (4, 4)
    batch = jax.device_put(batch, parallel_jax.make_sharding(batch.partition_spec))
    evaluate = parallel_jax.jit_sharded(
        pipeline.evaluate,
        in_specs=(
            parallel_jax.SHARE_PARTITION,
            batch.partition_spec,
            parallel_jax.SHARE_PARTITION,
            parallel_jax.DATA_PARTITION,
        ),
        out_specs=(parallel_jax.SHARE_PARTITION, parallel_jax.SHARE_PARTITION),
        check_vma=True,
        donate_argnums=2,
    )
    for step in range(2):
        _, state = evaluate(
            {}, batch, state, parallel_jax.shard_rngs(jax.random.key(step + 1))
        )
    actual = pipeline.digest({}, state, n_steps=2)["pair_correlation"]
    delta = electrons[:, 0] - electrons[:, 1]
    directed = (np.concatenate([delta, -delta]) + 0.5) % 1 - 0.5
    expected = (
        np.histogramdd(directed, bins=(4, 4), range=((-0.5, 0.5), (-0.5, 0.5)))[0]
        * 16
        / (len(electrons) * 4)
    )
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    np.testing.assert_allclose(np.mean(actual), 0.5, atol=1e-6)
