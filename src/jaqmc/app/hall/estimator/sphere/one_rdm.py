# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

r"""One-body reduced density matrix (1-RDM) estimator for the Haldane sphere.

Computes the 1-RDM in the monopole harmonic basis via Monte Carlo sampling
of a single auxiliary point :math:`r'` per walker per step:

.. math::
    \rho_{ij} = 4\pi \left\langle
        \sum_a \frac{\Psi(R_a')}{\Psi(R)}
        \varphi_i(r_a)\,\varphi_j^*(r')
    \right\rangle

where :math:`R_a'` is the configuration with electron :math:`a` replaced
by :math:`r'`, and :math:`\varphi_i` are monopole harmonics
:math:`Y_{Q,Q+n,m}` for the retained Landau levels.

The diagonal gives orbital occupation numbers. The trace approaches the
electron number as the retained basis becomes complete.
"""

from collections.abc import Mapping
from typing import Any

import jax
import numpy as np
from jax import numpy as jnp
from scipy import special as ss

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import Data
from jaqmc.estimator.base import PerWalkerEstimator, mean_reduce
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.base import NumericWavefunctionEvaluate


def make_monopole_harm(q: float, ell: float, m: float):
    r"""Build a monopole harmonic closure :math:`Y_{qlm}`.

    The monopole harmonic is parameterized by half-integer quantum numbers
    :math:`(q, l, m)`.  The returned closure evaluates :math:`Y_{qlm}` on
    arrays of spherical coordinates.

    Args:
        q: Monopole charge (half-integer).
        ell: Angular momentum quantum number.
        m: Magnetic quantum number.

    Returns:
        A callable ``Y_qlm(electrons)`` where ``electrons[..., 0]`` is
        :math:`\theta` and ``electrons[..., 1]`` is :math:`\phi`.
    """
    norm_factor = np.sqrt(
        ((2 * ell + 1) / (4 * np.pi))
        * (ss.factorial(ell - m) * ss.factorial(ell + m))
        / (ss.factorial(ell - q) * ss.factorial(ell + q))
    )
    # Keep only nonzero terms. Their powers are nonnegative integers, so
    # the expression remains finite at both poles without clipping angles.
    s = np.arange(max(0, round(-m - q)), min(round(ell - q), round(ell - m)) + 1)
    sum_factors = jnp.array(
        (-1) ** (ell - m - s) * ss.comb(ell - q, s) * ss.comb(ell + q, ell - m - s)
    )

    def Y_qlm(electrons: jnp.ndarray) -> jnp.ndarray:
        theta, phi = electrons[..., 0], electrons[..., 1]
        sin_half = jnp.sin(theta / 2)
        cos_half = jnp.cos(theta / 2)
        theta_part = jnp.sum(
            sum_factors
            * sin_half[..., None]
            ** jnp.asarray(np.rint(2 * ell - 2 * s - m - q), dtype=jnp.int32)
            * cos_half[..., None]
            ** jnp.asarray(np.rint(2 * s + m + q), dtype=jnp.int32),
            axis=-1,
        )
        return norm_factor * theta_part * jnp.exp(1j * m * phi)

    return Y_qlm


def _uniform_sample_sphere(key: PRNGKey) -> jnp.ndarray:
    """Sample a single point uniformly on the unit sphere.

    Args:
        key: JAX random key.

    Returns:
        Array of shape ``(2,)`` containing ``(theta, phi)``.
    """
    key1, key2 = jax.random.split(key)
    theta = jnp.arccos(jax.random.uniform(key1, (), minval=-1, maxval=1))
    phi = jax.random.uniform(key2, (), minval=-jnp.pi, maxval=jnp.pi)
    return jnp.stack([theta, phi])


@configurable_dataclass
class SphereOneRDM(PerWalkerEstimator):
    r"""One-body reduced density matrix on the Haldane sphere.

    Computes the 1-RDM in the monopole harmonic basis
    :math:`\{Y_{Q,Q+n,m}\}` using stochastic sampling of an
    auxiliary point :math:`r'`.

    Each retained level ``n`` has ``flux + 2*n + 1`` orbitals.
    The diagonal gives orbital occupation numbers. The retained trace
    measures occupation within this finite basis.

    Args:
        flux: Magnetic flux :math:`2Q` (positive integer).
        num_landau_levels: Number of complete Landau levels to retain.
        f_log_psi: Complex log-psi function (runtime dep).
        data_field: Name of the data field (runtime dep, default ``"electrons"``).
    """

    flux: int = 2
    num_landau_levels: int = 1
    f_log_psi: NumericWavefunctionEvaluate = runtime_dep()
    data_field: str = runtime_dep(default="electrons")

    def init(self, data: Data, rngs: PRNGKey) -> None:
        """Build the list of monopole harmonic closures.

        Raises:
            ValueError: If the number of Landau levels is not positive.
        """
        Q = self.flux / 2
        if self.num_landau_levels < 1:
            raise ValueError("num_landau_levels must be positive")
        labels = [
            (n, float(m))
            for n in range(self.num_landau_levels)
            for m in np.arange(-Q - n, Q + n + 1)
        ]
        self._orbitals = [make_monopole_harm(Q, Q + n, m) for n, m in labels]
        self._orbital_landau_levels = jnp.asarray(
            [n for n, _ in labels], dtype=jnp.int32
        )
        self._orbital_angular_momenta = jnp.asarray([m for _, m in labels])
        return None

    def evaluate_single_walker(
        self,
        params: Params,
        data: Data,
        prev_walker_stats: Mapping[str, Any],
        state: Any,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], Any]:
        del prev_walker_stats
        electrons = data[self.data_field]
        nelec = electrons.shape[0]

        # Sample r' uniformly on the sphere
        r_prime = _uniform_sample_sphere(rngs)

        # Build displaced configs: for each electron a, replace r_a with r_prime
        data_prime_electrons = jnp.repeat(electrons[None], nelec, axis=0)
        idx = jnp.arange(nelec)
        data_prime_electrons = data_prime_electrons.at[idx, idx].set(r_prime)

        # Evaluate log-psi at original and displaced configs
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

        # rho_ij = 4pi * sum_a ratio_a * phi_i(r_a) * phi_j*(r')
        one_rdm = (4 * jnp.pi) * jnp.sum(
            wf_ratio[:, None, None]
            * varphi[:, :, None]
            * jnp.conj(varphi_prime)[None, None, :],
            axis=0,
        )

        return {"one_rdm": one_rdm}, state

    def reduce(self, walker_stats: Mapping[str, Any]) -> dict[str, Any]:
        """Mean over walkers without variance (complex matrix).

        Uses the default ``mean_reduce`` but skips variance since the
        1-RDM is a complex matrix where element-wise variance is not
        meaningful.

        Returns:
            Step-level mean of the 1-RDM across walkers.
        """
        return mean_reduce(walker_stats, include_variance=False)

    def finalize_stats(
        self, batched_stats: Mapping[str, Any], state: Any
    ) -> dict[str, Any]:
        """Average over steps and extract diagonal and trace.

        Returns:
            Dict with ``one_rdm``, ``one_rdm:diagonal``, and
            ``one_rdm:trace``.
        """
        one_rdm = jnp.nanmean(batched_stats["one_rdm"], axis=0)
        occupations = jnp.real(jnp.diagonal(one_rdm))
        level_occupations = (
            jnp.zeros(self.num_landau_levels)
            .at[self._orbital_landau_levels]
            .add(occupations)
        )
        return {
            "one_rdm": one_rdm,
            "one_rdm:diagonal": jnp.diagonal(one_rdm),
            "one_rdm:trace": jnp.trace(one_rdm),
            "one_rdm:landau_level_occupations": level_occupations,
            "one_rdm:basis_landau_level": self._orbital_landau_levels,
            "one_rdm:basis_angular_momentum": self._orbital_angular_momenta,
        }


OneRDM = SphereOneRDM

__all__ = ["SphereOneRDM"]
__all__ += ["OneRDM", "make_monopole_harm"]
