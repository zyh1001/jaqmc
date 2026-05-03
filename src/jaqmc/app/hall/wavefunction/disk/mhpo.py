# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Product orbital wavefunction on the disk.

The input features are Cartesian coordinates on the disk:
``[x, y]``.
These are fed into a PsiformerBackbone (which appends spin encoding),
followed by disk orbital projection and a disk Jastrow factor.
"""

import enum
import numpy as np
from flax import linen as nn
from jax import numpy as jnp
from scipy import special as ss

from jaqmc.app.hall.data import HallData
from jaqmc.array_types import Params
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.backbone.psiformer import PsiformerBackbone
from jaqmc.wavefunction.base import ComplexWFOutput, Wavefunction
from jaqmc.wavefunction.output.orbital import SplitChannelDense

from .jastrow import DiskJastrow

__all__ = ["DiskMHPO", "EnvelopeType"]


class EnvelopeType(str, enum.Enum):
    """Types of envelope functions for disk orbitals."""
    no_m_learnable = "no_m_learnable"
    no_m_fixed = "no_m_fixed"
    m_learnable = "m_learnable"
    m_fixed = "m_fixed"


class DiskOrbitals(nn.Module):
    r"""Disk orbital construction.

    Combines learned features with disk basis functions
    :math:`z^m \exp(-r^2 \sigma)` where :math:`z = x - iy`.

    Args:
        Q: Maximum angular momentum (number of flux quanta).
        nspins: ``(n_up, n_down)`` electron counts.
        ndets: Number of determinants.
        envelope_type: The type of envelope to apply to the orbitals.
    """

    Q: float
    nspins: tuple[int, int]
    ndets: int
    envelope_type: EnvelopeType = EnvelopeType.m_learnable

    def setup(self) -> None:
        self.num_orbitals = int(self.Q + 1)
        m = np.arange(0, self.num_orbitals)
        self.norm_factor = jnp.array(1.0 / np.sqrt(ss.factorial(m)))
        features = [self.num_orbitals, sum(self.nspins), self.ndets]
        self.orbitals_real = SplitChannelDense(channels=self.nspins, features=features)
        self.orbitals_imag = SplitChannelDense(channels=self.nspins, features=features)

    def __call__(
        self, h_one: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray
    ) -> jnp.ndarray:
        orbitals = self.orbitals_real(h_one) + 1j * self.orbitals_imag(h_one)
        
        r_square = (x**2 + y**2)[..., None]
        
        sigma = self.param("sigma", nn.initializers.constant(0.25), (self.num_orbitals,))
        
        if self.envelope_type == EnvelopeType.no_m_learnable:
            envelope = jnp.exp(- r_square * jnp.abs(sigma))
            
        elif self.envelope_type == EnvelopeType.no_m_fixed:
            envelope = jnp.exp(- r_square * 0.25)
        
        elif self.envelope_type == EnvelopeType.m_learnable:
            m = jnp.arange(0, self.num_orbitals)
            z = (x - 1j * y)[..., None]
            envelope = self.norm_factor * z ** m * jnp.exp(- r_square * sigma)
            
        elif self.envelope_type == EnvelopeType.m_fixed:
            m = jnp.arange(0, self.num_orbitals)
            z = (x - 1j * y)[..., None]
            envelope = self.norm_factor * z ** m * jnp.exp(- r_square * 0.25)
        else:
            raise ValueError(f"Unknown envelope type: {self.envelope_type}")

        orbitals = jnp.sum(orbitals * envelope[..., None, None], axis=1)

        return jnp.moveaxis(orbitals, -1, 0)  # (ndets, nelec, nelec)


class DiskMHPO(Wavefunction[HallData, ComplexWFOutput]):
    r"""Product orbital ansatz on the disk.

    Architecture:
        1. Input: ``(x, y)`` Cartesian features
        2. Backbone: :class:`~jaqmc.wavefunction.backbone.psiformer.PsiformerBackbone`
        3. Orbitals: Disk orbital projection
        4. Jastrow: Disk Jastrow factor
        5. Output: ``slogdet`` of orbital matrix to complex ``log psi``

    Args:
        nspins: ``(n_up, n_down)`` electron counts.
        monopole_strength: :math:`Q`.
        flux: Magnetic flux :math:`Q`.
        ndets: Number of determinants.
        num_heads: Number of attention heads.
        heads_dim: Dimension of each attention head.
        num_layers: Number of Psiformer layers.
        flux_per_elec: Flux quanta attached per electron for composite fermions.
        envelope_type: The type of envelope to apply to the orbitals.
    """

    nspins: tuple[int, int] = runtime_dep()
    monopole_strength: float = runtime_dep()
    flux: int = runtime_dep()
    ndets: int = 1
    num_heads: int = 4
    heads_dim: int = 64
    num_layers: int = 2
    flux_per_elec: int = 0
    envelope_type: EnvelopeType = EnvelopeType.m_learnable

    def setup(self) -> None:
        reduced_flux = self.flux - self.flux_per_elec * (sum(self.nspins) - 1)
        self.backbone_layer = PsiformerBackbone(
            nspins=self.nspins,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            heads_dim=self.heads_dim,
        )
        self.orbital_layer = DiskOrbitals(
            Q=reduced_flux,
            nspins=self.nspins,
            ndets=self.ndets,
            envelope_type=self.envelope_type,
        )
        self.jastrow_layer = DiskJastrow(nspins=self.nspins)

    def __call__(self, data: HallData) -> ComplexWFOutput:
        electrons = data.electrons
        x, y = electrons[..., 0], electrons[..., 1]

        # Cartesian features
        h_one = jnp.stack([x, y], axis=-1)

        # Backbone (appends spin encoding internally)
        h_one = self.backbone_layer(h_one)

        # Disk orbitals: (ndets, nelec, nelec)
        orbitals = self.orbital_layer(h_one, x, y)

        # Jastrow factor
        jastrow = self.jastrow_layer(electrons)

        orbitals = jnp.exp(jastrow / sum(self.nspins)) * orbitals

        # Complex slogdet
        signs, logdets = jnp.linalg.slogdet(orbitals)
        logmax = jnp.max(logdets)
        logpsi = jnp.log(jnp.sum(signs * jnp.exp(logdets - logmax))) + logmax

        return ComplexWFOutput(logpsi=logpsi)

    def logpsi(self, params: Params, data: HallData) -> jnp.ndarray:
        return self.evaluate(params, data)["logpsi"]
