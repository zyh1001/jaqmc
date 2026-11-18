# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Particle-density structure factor on the reciprocal torus lattice."""

from collections.abc import Mapping
from typing import Any

import numpy as np
from jax import numpy as jnp

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import Data
from jaqmc.estimator.base import PerWalkerEstimator, mean_reduce
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.wiring import runtime_dep


@configurable_dataclass
class TorusStructureFactor(PerWalkerEstimator):
    """Evaluate every reciprocal vector within a physical cutoff.

    Args:
        flux: Number of magnetic flux quanta.
        tau: Complex torus shape.
        q_max: Maximum wave number in inverse magnetic-length units.
    """

    flux: int = 18
    tau: complex = runtime_dep(default=1j)
    q_max: float = 5.0

    def init(self, data: Data, rngs: PRNGKey) -> None:
        if self.flux <= 0 or self.tau.imag <= 0 or self.q_max < 0:
            raise ValueError("Invalid flux, torus shape, or wave-number cutoff")
        l1 = np.sqrt(2 * np.pi * self.flux / self.tau.imag)
        scale = 2 * np.pi / l1
        hmax = int(np.ceil(self.q_max / scale))
        kmax = int(np.ceil(self.q_max / scale * abs(self.tau)))
        modes = []
        for h in range(-hmax, hmax + 1):
            for k in range(-kmax, kmax + 1):
                vector = scale * np.array([h, (k - h * self.tau.real) / self.tau.imag])
                if np.linalg.norm(vector) <= self.q_max + 1e-12:
                    modes.append((h, k, *vector))
        modes.sort(key=lambda row: (row[2] ** 2 + row[3] ** 2, row[0], row[1]))
        self._indices = jnp.asarray([row[:2] for row in modes], dtype=jnp.int32)
        self._vectors = jnp.asarray([row[2:] for row in modes])
        self._nelec = data.electrons.shape[-2]
        return None

    def evaluate_single_walker(
        self,
        params: Params,
        data: Data,
        prev_walker_stats: Mapping[str, Any],
        state: None,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], None]:
        phase = 2 * jnp.pi * (data.electrons @ self._indices.T)
        rho = jnp.sum(jnp.exp(-1j * phase), axis=0)
        return {
            "structure_factor:rho_real": rho.real,
            "structure_factor:rho_imag": rho.imag,
            "structure_factor:power": jnp.abs(rho) ** 2,
        }, state

    def reduce(self, walker_stats: Mapping[str, Any]) -> dict[str, Any]:
        return mean_reduce(walker_stats, include_variance=False)

    def finalize_stats(
        self,
        batched_stats: Mapping[str, Any],
        state: None,
    ) -> dict[str, Any]:
        rho = jnp.mean(
            batched_stats["structure_factor:rho_real"], axis=0
        ) + 1j * jnp.mean(batched_stats["structure_factor:rho_imag"], axis=0)
        raw = jnp.mean(batched_stats["structure_factor:power"], axis=0) / self._nelec
        disconnected = jnp.abs(rho) ** 2 / self._nelec
        return {
            "structure_factor": raw,
            "structure_factor:connected": raw - disconnected,
            "structure_factor:disconnected": disconnected,
            "structure_factor:rho_mean_real": rho.real,
            "structure_factor:rho_mean_imag": rho.imag,
            "structure_factor:q": jnp.linalg.norm(self._vectors, axis=-1),
            "structure_factor:q_vectors": self._vectors,
            "structure_factor:reciprocal_indices": self._indices,
            "structure_factor:electron_number": self._nelec,
        }


__all__ = ["TorusStructureFactor"]
