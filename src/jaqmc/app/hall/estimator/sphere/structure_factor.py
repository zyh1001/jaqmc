# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Particle-density structure factor in spherical harmonic modes."""

from collections.abc import Mapping
from typing import Any

import numpy as np
from jax import numpy as jnp

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import Data
from jaqmc.estimator.base import PerWalkerEstimator, mean_reduce
from jaqmc.utils.config import configurable_dataclass


def spherical_density_modes(electrons: jnp.ndarray, ell_max: int) -> jnp.ndarray:
    """Return summed normalized spherical harmonics with nonnegative m.

    Ordering is m-major, then ell=m,...,ell_max. Negative m modes are the
    conjugates up to a sign and are accounted for with multiplicity two.
    """
    theta, phi = electrons[..., 0], electrons[..., 1]
    x = jnp.cos(theta)
    phase = jnp.exp(1j * phi)
    diagonal = jnp.ones_like(phase) / jnp.sqrt(4 * jnp.pi)
    modes = []
    for m in range(ell_max + 1):
        if m:
            diagonal = (
                -np.sqrt((2 * m + 1) / (2 * m)) * jnp.sin(theta) * phase * diagonal
            )
        previous = diagonal
        modes.append(jnp.sum(previous, axis=-1))
        if m == ell_max:
            continue
        current = np.sqrt(2 * m + 3) * x * previous
        modes.append(jnp.sum(current, axis=-1))
        for ell in range(m + 2, ell_max + 1):
            a = np.sqrt((4 * ell**2 - 1) / (ell**2 - m**2))
            b = np.sqrt(
                (2 * ell + 1)
                * ((ell - 1) ** 2 - m**2)
                / ((2 * ell - 3) * (ell**2 - m**2))
            )
            following = a * x * current - b * previous
            modes.append(jnp.sum(following, axis=-1))
            previous, current = current, following
    return jnp.stack(modes, axis=-1)


@configurable_dataclass
class SphereStructureFactor(PerWalkerEstimator):
    """Angularly averaged spherical structure factor.

    Args:
        ell_max: Maximum density-mode angular momentum.
        radius: Sphere radius in magnetic-length units.
    """

    ell_max: int = 20
    radius: float = 1.0

    def init(self, data: Data, rngs: PRNGKey) -> None:
        if self.ell_max < 0 or self.radius <= 0:
            raise ValueError("ell_max must be nonnegative and radius positive")
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
        rho = spherical_density_modes(data.electrons, self.ell_max)
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
        power = jnp.mean(batched_stats["structure_factor:power"], axis=0)
        labels = [
            (ell, m)
            for m in range(self.ell_max + 1)
            for ell in range(m, self.ell_max + 1)
        ]
        ell = jnp.asarray([level for level, _ in labels], dtype=jnp.int32)
        m = jnp.asarray([m for _, m in labels], dtype=jnp.int32)
        weights = 4 * jnp.pi * jnp.where(m == 0, 1, 2) / (self._nelec * (2 * ell + 1))
        raw = jnp.zeros(self.ell_max + 1).at[ell].add(weights * power)
        disconnected = (
            jnp.zeros(self.ell_max + 1).at[ell].add(weights * jnp.abs(rho) ** 2)
        )
        axis = jnp.arange(self.ell_max + 1)
        return {
            "structure_factor": raw,
            "structure_factor:connected": raw - disconnected,
            "structure_factor:disconnected": disconnected,
            "structure_factor:rho_mean_real": rho.real,
            "structure_factor:rho_mean_imag": rho.imag,
            "structure_factor:ell": axis,
            "structure_factor:q": jnp.sqrt(axis * (axis + 1)) / self.radius,
            "structure_factor:mode_ell": ell,
            "structure_factor:mode_m": m,
            "structure_factor:mode_weight": weights,
            "structure_factor:electron_number": self._nelec,
        }


__all__ = ["SphereStructureFactor", "spherical_density_modes"]
