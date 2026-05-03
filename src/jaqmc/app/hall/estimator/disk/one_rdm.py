# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

r"""One-body reduced density matrix (1-RDM) estimator for the disk.

Computes the 1-RDM in the disk basis via Monte Carlo sampling.
"""

from collections.abc import Mapping
from typing import Any

import jax
import numpy as np
from jax import numpy as jnp
from scipy import special as ss

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import Data
from jaqmc.estimator.base import Estimator, mean_reduce
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.base import NumericWavefunctionEvaluate


def make_disk_basis(m: float, sigma: float = 0.25):
    r"""Build a disk basis function.

    Args:
        m: Angular momentum quantum number.
        sigma: Decay parameter for the gaussian envelope.

    Returns:
        A callable ``phi_m(electrons)``.
    """
    norm_factor = 1.0 / np.sqrt(ss.factorial(m))

    def phi_m(electrons: jnp.ndarray) -> jnp.ndarray:
        x, y = electrons[..., 0], electrons[..., 1]
        z = x - 1j * y
        r_sq = x**2 + y**2
        return norm_factor * z ** m * jnp.exp(-r_sq * sigma)

    return phi_m


def _uniform_sample_disk(key: PRNGKey, radius_max: float = 10.0) -> jnp.ndarray:
    """Sample a single point uniformly on a disk up to radius_max.

    Args:
        key: JAX random key.
        radius_max: Maximum radius to sample.

    Returns:
        Array of shape ``(2,)`` containing ``(x, y)``.
    """
    key1, key2 = jax.random.split(key)
    r = radius_max * jnp.sqrt(jax.random.uniform(key1, ()))
    theta = jax.random.uniform(key2, (), minval=0.0, maxval=2*jnp.pi)
    return jnp.stack([r * jnp.cos(theta), r * jnp.sin(theta)])


@configurable_dataclass
class DiskOneRDM(Estimator):
    r"""One-body reduced density matrix on the disk.

    Args:
        flux: Magnetic flux.
        f_log_psi: Complex log-psi function (runtime dep).
        data_field: Name of the data field (runtime dep).
        radius_max: Maximum radius for uniform sampling.
    """

    flux: int = 2
    f_log_psi: NumericWavefunctionEvaluate = runtime_dep()
    data_field: str = runtime_dep(default="electrons")
    radius_max: float = 15.0

    def init(self, data: Data, rngs: PRNGKey) -> None:
        """Build the list of disk basis closures."""
        norbs = self.flux + 1
        self._orbitals = [make_disk_basis(float(m)) for m in range(norbs)]
        return None

    def evaluate_local(
        self,
        params: Params,
        data: Data,
        prev_local_stats: Mapping[str, Any],
        state: Any,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], Any]:
        del prev_local_stats
        electrons = data[self.data_field]
        nelec = electrons.shape[0]

        # Sample r' uniformly on the disk
        r_prime = _uniform_sample_disk(rngs, self.radius_max)
        area = jnp.pi * self.radius_max**2

        # Build displaced configs
        data_prime_electrons = jnp.repeat(electrons[None], nelec, axis=0)
        idx = jnp.arange(nelec)
        data_prime_electrons = data_prime_electrons.at[idx, idx].set(r_prime)

        logpsi = self.f_log_psi(params, data)

        def eval_displaced(displaced_electrons: jnp.ndarray) -> jnp.ndarray:
            return self.f_log_psi(
                params, data.merge({self.data_field: displaced_electrons})
            )

        logpsi_prime = jax.vmap(eval_displaced)(data_prime_electrons)
        wf_ratio = jnp.exp(logpsi_prime - logpsi)

        # Orbitals at original positions: shape (nelec, norbs)
        varphi = jnp.stack([orb(electrons) for orb in self._orbitals], axis=-1)
        # Orbitals at r': shape (norbs,)
        varphi_prime = jnp.stack([orb(r_prime[None]) for orb in self._orbitals])[..., 0]

        one_rdm = area * jnp.sum(
            wf_ratio[:, None, None]
            * varphi[:, :, None]
            * jnp.conj(varphi_prime)[None, None, :],
            axis=0,
        )

        return {"one_rdm": one_rdm}, state

    def reduce(self, local_stats: Mapping[str, Any]) -> dict[str, Any]:
        return mean_reduce(local_stats, include_variance=False)

    def finalize_stats(
        self, batched_stats: Mapping[str, Any], state: Any
    ) -> dict[str, Any]:
        one_rdm = jnp.nanmean(batched_stats["one_rdm"], axis=0)
        return {
            "one_rdm": one_rdm,
            "one_rdm:diagonal": jnp.diagonal(one_rdm),
            "one_rdm:trace": jnp.trace(one_rdm),
        }
