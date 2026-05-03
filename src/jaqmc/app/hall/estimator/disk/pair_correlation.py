# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

r"""Pair correlation function estimator for the disk geometry.

Accumulates a histogram of euclidean distances :math:`r_{ij}`
to obtain the pair correlation function :math:`g(r)`.
"""

from collections.abc import Mapping
from typing import Any

from jax import numpy as jnp

from jaqmc.array_types import PRNGKey
from jaqmc.data import BatchedData, Data
from jaqmc.estimator.base import Estimator
from jaqmc.utils.config import configurable_dataclass


@configurable_dataclass
class DiskPairCorrelation(Estimator):
    r"""Pair correlation function :math:`g(r)` on the disk.

    For each pair of electrons :math:`(i < j)`, computes the euclidean
    distance :math:`r_{ij}` and accumulates a histogram.

    Args:
        bins: Number of histogram bins.
        max_r: Maximum distance for the histogram.
    """

    bins: int = 200
    max_r: float = 20.0

    def init(self, data: Data, rngs: PRNGKey) -> jnp.ndarray:
        return jnp.zeros(self.bins)

    def evaluate_batch(
        self,
        params: Any,
        batched_data: BatchedData,
        prev_local_stats: Mapping[str, Any],
        state: jnp.ndarray,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], jnp.ndarray]:
        del params, prev_local_stats, rngs
        electrons = batched_data.data.electrons
        batch_size, nelec, _ = electrons.shape
        x, y = electrons[..., 0], electrons[..., 1]

        r_sq = (x[:, :, None] - x[:, None, :])**2 + (y[:, :, None] - y[:, None, :])**2
        r_ij = jnp.sqrt(jnp.clip(r_sq, a_min=1e-8))

        pairs = r_ij[:, *jnp.triu_indices(nelec, 1)].reshape(-1)

        to_add, _ = jnp.histogram(pairs, self.bins, (0.0, self.max_r))
        
        norm_factor = 2 * self.bins / (batch_size * nelec**2 * self.max_r)
        
        return {}, state + to_add * norm_factor

    def reduce(self, local_stats: Mapping[str, Any]) -> dict[str, Any]:
        return {}

    def finalize_stats(
        self, mean_stats: Mapping[str, Any], state: jnp.ndarray
    ) -> dict[str, Any]:
        return {"pair_correlation": state}
