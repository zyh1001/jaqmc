# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Physical and sharding checks for unweighted sphere pair correlations."""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.app.hall.data import HallData
from jaqmc.app.hall.estimator.sphere.pair_correlation import (
    PairCorrelation,
    angular_bin_geometry,
)
from jaqmc.data import BatchedData
from jaqmc.estimator.base import EstimatorPipeline
from jaqmc.utils import parallel_jax


@pytest.mark.parametrize("separation", [0.0, 1e-4, np.pi - 1e-4, np.pi])
def test_endpoint_pairs_remain_finite_and_count_once(separation):
    data = HallData(
        electrons=jnp.asarray([[np.pi / 2, 0], [np.pi / 2, separation]], jnp.float32)
    )
    estimator = PairCorrelation()
    estimator.init(data, jax.random.key(0))
    stats, _ = estimator.evaluate_single_walker({}, data, {}, None, jax.random.key(1))
    counts = np.asarray(stats["pair_correlation:counts_per_configuration"])
    assert counts.sum() == 1
    assert counts[0 if separation < np.pi / 2 else -1] == 1
    digest = estimator.finalize_stats({k: v[None] for k, v in stats.items()}, None)
    assert np.isfinite(digest["pair_correlation"]).all()
    assert (np.asarray(digest["pair_correlation"]) >= 0).all()
    np.testing.assert_allclose(
        np.dot(digest["pair_correlation"], digest["pair_correlation:bin_probability"]),
        0.5,
        atol=1e-7,
    )


def test_uniform_sphere_has_finite_particle_baseline():
    size = 32768
    electrons = np.zeros((size, 2, 2), dtype=np.float32)
    electrons[:, 1, 0] = np.arccos(-1 + (np.arange(size) + 0.5) * 2 / size)
    batch = BatchedData(
        data=HallData(electrons=jnp.asarray(electrons)), fields_with_batch=["electrons"]
    )
    estimator = PairCorrelation(bins=20)
    pipeline = EstimatorPipeline({"pair": estimator})
    state = pipeline.init(batch, jax.random.key(0))
    stats, state = jax.jit(pipeline.evaluate)({}, batch, state, jax.random.key(1))
    digest = pipeline.digest({k: v[None] for k, v in stats.items()}, state, n_steps=1)
    np.testing.assert_allclose(digest["pair_correlation"], 0.5, rtol=0.006)
    np.testing.assert_allclose(digest["pair_correlation:g_finite_n"], 1, rtol=0.006)
    _, probability = angular_bin_geometry(20)
    np.testing.assert_allclose(probability.sum(), 1, atol=1e-15)


def test_sharded_counts_match_full_batch_and_preserve_all_pairs():
    size = 8 * jax.device_count()
    rng = np.random.default_rng(381)
    electrons = np.stack(
        [
            np.arccos(rng.uniform(-1, 1, (size, 6))),
            rng.uniform(-np.pi, np.pi, (size, 6)),
        ],
        axis=-1,
    ).astype(np.float32)
    batch = BatchedData(
        data=HallData(electrons=jnp.asarray(electrons)), fields_with_batch=["electrons"]
    )
    pipeline = EstimatorPipeline({"pair": PairCorrelation(bins=20)})
    state = pipeline.init(batch, jax.random.key(0))
    expected, _ = pipeline.evaluate({}, batch, state, jax.random.key(1))
    batch = jax.device_put(batch, parallel_jax.make_sharding(batch.partition_spec))
    state = jax.device_put(
        state, parallel_jax.make_sharding(parallel_jax.SHARE_PARTITION)
    )
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
    )
    actual, state = evaluate(
        {}, batch, state, parallel_jax.shard_rngs(jax.random.key(1))
    )
    np.testing.assert_array_equal(
        actual["pair_correlation:counts_per_configuration"],
        expected["pair_correlation:counts_per_configuration"],
    )
    assert actual["pair_correlation:pair_count"] == 15
    digest = pipeline.digest({k: v[None] for k, v in actual.items()}, state, n_steps=1)
    np.testing.assert_allclose(
        np.dot(digest["pair_correlation"], digest["pair_correlation:bin_probability"]),
        5 / 6,
        atol=1e-7,
    )
