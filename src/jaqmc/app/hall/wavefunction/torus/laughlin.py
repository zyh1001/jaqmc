# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Analytic Laughlin states in the torus symmetric-gauge convention."""

import jax
from jax import numpy as jnp

from jaqmc.app.hall.data import HallData
from jaqmc.array_types import Params
from jaqmc.utils.torus import get_torus_lattice
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.base import ComplexWFOutput, Wavefunction


def log_theta(
    w: jnp.ndarray, tau: complex, a: float, b: float, terms: int
) -> jnp.ndarray:
    """Evaluate a theta series after centering and scaling its terms.

    Returns:
        The complex logarithm of the theta function.
    """
    centre = jax.lax.stop_gradient(jnp.rint(-jnp.imag(w) / tau.imag - a))
    k = centre[..., None] + jnp.arange(-terms, terms + 1) + a
    exponent = 1j * jnp.pi * tau * k**2 + 2j * jnp.pi * k * (w[..., None] + b)
    scale = jax.lax.stop_gradient(jnp.max(exponent.real, axis=-1))
    return scale + jnp.log(jnp.sum(jnp.exp(exponent - scale[..., None]), axis=-1))


class TorusLaughlin(Wavefunction[HallData, ComplexWFOutput]):
    """Fully polarized Laughlin state at flux m*N for odd positive m.

    Args:
        nspins: Electron counts, supplied by the Hall workflow.
        flux: Number of magnetic flux quanta.
        tau: Torus modular parameter.
        sector: Center-of-mass sector, from zero to m-1.
        theta_terms: Number of series terms on either side of the maximum.
    """

    nspins: tuple[int, int] = runtime_dep()
    flux: int = runtime_dep()
    tau: complex = runtime_dep()
    sector: int = 0
    theta_terms: int = 12

    def setup(self) -> None:
        nelec = sum(self.nspins)
        if nelec < 1 or min(self.nspins) != 0 or self.flux % nelec:
            raise ValueError("Laughlin requires full polarization and integer flux/N")
        m = self.flux // nelec
        if m < 1 or m % 2 != 1 or not 0 <= self.sector < m or self.theta_terms < 1:
            raise ValueError("Invalid Laughlin exponent, sector, or theta truncation")

    def __call__(self, data: HallData) -> ComplexWFOutput:
        nelec = sum(self.nspins)
        m = self.flux // nelec
        l1, l2, _ = get_torus_lattice(self.flux, self.tau)
        z = l1 * data.electrons[..., 0] + l2 * data.electrons[..., 1]
        i, j = jnp.triu_indices(nelec, 1)
        relative = (z[i] - z[j]) / l1
        pair = m * jnp.sum(log_theta(relative, self.tau, 0.5, 0.5, self.theta_terms))
        # These characteristics reproduce the existing orbitals' (-1)^flux
        # boundary twists for both lattice periods.
        centre = log_theta(
            m * jnp.sum(z) / l1,
            m * self.tau,
            0.5 + self.sector / m,
            0.5,
            self.theta_terms,
        )
        gaussian = jnp.sum(-(z.imag**2) / 2 + 0.5j * z.real * z.imag)
        return ComplexWFOutput(logpsi=gaussian + pair + centre)

    def logpsi(self, params: Params, data: HallData) -> jnp.ndarray:
        return self.evaluate(params, data)["logpsi"]


__all__ = ["TorusLaughlin"]
