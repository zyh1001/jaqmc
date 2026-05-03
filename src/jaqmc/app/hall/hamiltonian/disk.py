# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Potential energy estimator for the Haldane sphere."""

from collections.abc import Mapping
from typing import Any, Literal

from jax import numpy as jnp
from jax.numpy import cos, sin

from DeepHall.conf_potential import single_quad
from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import Data
from jaqmc.estimator.base import Estimator
from jaqmc.utils.config import configurable_dataclass

from scipy.special import ellipk
from scipy.integrate import quad

import os
import numpy as np

@configurable_dataclass
class DiskPotential(Estimator):
    r"""Potential energy on the Haldane sphere.

    Converts spherical coordinates to Cartesian, computes pairwise
    cosines, then evaluates the chosen interaction.

    Args:
        interaction_type: Interaction potential form.
        monopole_strength: :math:`Q = \text{flux}`.
        radius: Sphere radius.
        interaction_strength: Overall scaling factor.
    """

    interaction_type: Literal["coulomb"] = "coulomb"
    monopole_strength: float = 1.0
    radius: float = 1.0
    N: int = 6
    flux: int = 18
    d: int = 0.1
    interaction_strength: float = 1.0

    def single_quad(self, r: float, 
                    limit: int=1000, 
                    epsrel: float=1e-4):
        a = self.radius
        d = self.d
        N = self.N
        def integrand(rp, r, d):
            """
            被积函数 f(r,r')
            """

            A = (r + rp)**2 + d**2
            B = (r - rp)**2 + d**2
            m1 = 4 * r * rp / A
            m2 = 4 * r * rp / B
            k1 = ellipk(m1)
            k2 = ellipk(-m2)
            # m1 = np.clip(m1, None, 1.0)   # 数值稳定
            # m2 = np.clip(m2, None, 1.0)
            f = 2*rp * (k1 / jnp.sqrt(A) 
                    + k2 / jnp.sqrt(B))

            return f
    
        f = lambda rp: integrand(rp, r, d)
        I, err = quad(f, 0.0, a, epsrel=epsrel, limit=limit)
        pref = - N / jnp.pi / a**2

        return pref * I, err

    def make_conf_potential(self, r_vals, 
            limit: int=1000, 
            epsrel: float=1e-4):
        Vc = jnp.zeros_like(r_vals)
        err = jnp.zeros_like(r_vals)
        i=0
        for r in r_vals:

            val, e = self.single_quad(r, limit, epsrel)

            Vc[i] = val
            err[i] = e

            if i % 10 == 0:

                print(f"{i}/{len(r_vals)}  err={e:.2e}")
            i+=1
        return Vc, err    

    def evaluate_local(
        self,
        params: Params,
        data: Data,
        prev_local_stats: Mapping[str, Any],
        state: None,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], None]:
        del params, prev_local_stats, rngs
        electrons = data["electrons"]
        x, y = electrons[..., 0], electrons[..., 1]

        # Convert to Cartesian on the unit sphere
        xy = jnp.stack(
            [x, y], axis=-1
        )
        
        r = jnp.sqrt(x**2 + y**2)
        
        folder_name = f"N{self.N}_a{self.radius:.2f}_d{self.d:.2f}"
        vc_path = os.path.join(folder_name, "Vc.npy")
        r_path = os.path.join(folder_name, "r.npy")
        
        if os.path.exists(folder_name) and os.path.exists(vc_path) and os.path.exists(r_path):
            Vc_table = jnp.load(vc_path)
            r_grid = jnp.load(r_path)
        else:
            os.makedirs(folder_name, exist_ok=True)
            r_grid_np = np.linspace(0.0, 16 * self.radius, 10000)
            Vc_table_np, _ = self.make_conf_potential(r_grid_np)
            np.save(vc_path, np.array(Vc_table_np))
            np.save(r_path, np.array(r_grid_np))
            Vc_table = jnp.array(Vc_table_np)
            r_grid = jnp.array(r_grid_np)

        # 插值计算 r<a*15 区域的势能
        V_small = jnp.interp(r, r_grid, Vc_table)  
        # r>=15a 时近似为点电荷势
        V_large = -self.N / jnp.sqrt(self.d**2 + r**2)
        # 根据条件选择势能
        V_vals = jnp.where(r < 15*self.radius, V_small, V_large)
        # 对所有电子求和（axis=-1对最后一个维度求和）
        
        potential_conf = jnp.sum(V_vals, axis=-1) 

        if self.interaction_type == "coulomb":
            r_ee = xy[None] - xy[:, None]
            eye = jnp.eye(r_ee.shape[0])
            r_ee = jnp.linalg.norm(r_ee + eye[..., None], axis=-1) * (1.0 - eye)
            potential_ee = jnp.sum(jnp.triu(1 / r_ee, k=1)) / self.radius
            potential = potential_ee + potential_conf
        
        else:
            raise ValueError(f"Unknown interaction type: {self.interaction_type}")


        return {"energy:potential": potential * self.interaction_strength}, state
