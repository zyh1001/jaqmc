# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Spherical pair correlations from unweighted angular-bin counts."""

from collections.abc import Mapping
from typing import Any

import numpy as np
from jax import numpy as jnp

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import Data
from jaqmc.estimator.base import PerWalkerEstimator, mean_reduce
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.wiring import runtime_dep


def angular_bin_geometry(bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Construct angular edges and exact fractions of the sphere area.

    Returns:
        Float64 edges and positive bin probabilities summing to one.

    Raises:
        ValueError: If the number of bins is not positive.
    """
    if bins < 1:
        raise ValueError("bins must be positive")
    edges = np.linspace(0.0, np.pi, bins + 1)
    probability = np.sin((edges[:-1] + edges[1:]) / 2) * np.sin(np.diff(edges) / 2)
    return edges, probability


@configurable_dataclass
class SpherePairCorrelation(PerWalkerEstimator):
    r"""Count unordered pairs and normalize by exact spherical bin areas.

    For a bin with area fraction :math:`p_b`, the returned correlation is
    :math:`g_b=2\langle C_b\rangle/(N^2p_b)`. Its independent uniform-point
    baseline is :math:`(N-1)/N`. The finite-N version has baseline one.
    Step records contain mean counts per configuration, allowing integer
    reconstruction and block-error analysis on the host.

    Args:
        bins: Number of equal angular bins between zero and pi.
        data_field: Coordinate field supplied at runtime.
        name: Prefix of saved statistics and normalized output.
    """

    bins: int = 200
    data_field: str = runtime_dep(default="electrons")
    name: str = "pair_correlation"

    def init(self, data: Data, rngs: PRNGKey) -> None:
        self._nelec = data[self.data_field].shape[-2]
        if self._nelec < 2:
            raise ValueError("Pair correlations require at least two electrons")
        edges, probability = angular_bin_geometry(self.bins)
        self._edges = jnp.asarray(edges)
        self._probability = jnp.asarray(probability)
        return None

    def extract(self, data: Data) -> jnp.ndarray:
        """Return geodesic pair angles with shape (..., n_pairs, 1)."""
        electrons = data[self.data_field]
        nelec = electrons.shape[-2]
        theta, phi = electrons[..., 0], electrons[..., 1]
        sin_t, cos_t = jnp.sin(theta), jnp.cos(theta)
        xyz = jnp.stack([sin_t * jnp.cos(phi), sin_t * jnp.sin(phi), cos_t], axis=-1)
        cos12 = jnp.sum(xyz[..., :, None, :] * xyz[..., None, :, :], axis=-1)
        pairs = cos12[..., *jnp.triu_indices(nelec, 1)]
        return jnp.arccos(jnp.clip(pairs, -1, 1))[..., None]

    def evaluate_single_walker(
        self,
        params: Params,
        data: Data,
        prev_walker_stats: Mapping[str, Any],
        state: None,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], None]:
        angles = self.extract(data)[..., 0]
        counts = jnp.histogram(angles, bins=self._edges)[0].astype(jnp.float32)
        return {
            self.name + ":counts_per_configuration": counts,
            self.name + ":pair_count": jnp.sum(counts),
        }, state

    def reduce(self, walker_stats: Mapping[str, Any]) -> dict[str, Any]:
        return mean_reduce(walker_stats, include_variance=False)

    def finalize_stats(
        self, batched_stats: Mapping[str, Any], state: None
    ) -> dict[str, Any]:
        counts = jnp.mean(
            batched_stats[self.name + ":counts_per_configuration"], axis=0
        )
        value = 2 * counts / (self._nelec**2 * self._probability)
        return {
            self.name: value,
            self.name + ":g_finite_n": value * self._nelec / (self._nelec - 1),
            self.name + ":counts_per_configuration": counts,
            self.name + ":theta_edges": self._edges,
            self.name + ":theta": (self._edges[:-1] + self._edges[1:]) / 2,
            self.name + ":bin_probability": self._probability,
            self.name + ":electron_number": self._nelec,
        }


PairCorrelation = SpherePairCorrelation

__all__ = ["PairCorrelation", "SpherePairCorrelation", "angular_bin_geometry"]
