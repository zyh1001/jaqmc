# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Verify first-order tagging of both active disk Jastrow parameters."""

import jax
import kfac_jax
import numpy as np
from jax import numpy as jnp

from jaqmc.app.hall.wavefunction.disk.jastrow import DiskJastrow
from jaqmc.optimizer.kfac.tag_registration import make_graph_patterns


def test_disk_jastrow_first_order_tags_and_coordinate_derivatives():
    layer = DiskJastrow(nspins=(2, 0))
    positions = jnp.array([[0.2, 0.3], [1.1, -0.4]], dtype=jnp.float32)
    params = layer.init(jax.random.key(0), positions)
    batch = jnp.stack([positions, positions + jnp.array([[0.01, 0.0], [0.0, 0.02]])])

    def tagged_value(p, coords):
        result = jax.vmap(layer.apply, in_axes=(None, 0))(p, coords)
        kfac_jax.register_normal_predictive_distribution(result[:, None])
        return result

    graph = kfac_jax.tag_graph_matcher.auto_register_tags(
        tagged_value, (params, batch), graph_patterns=make_graph_patterns()
    )
    names = [
        "|".join(graph._param_labels.get(p, ["Orphan"]))
        for p in graph._func_graph.params_vars
    ]
    labels = jax.tree_util.tree_unflatten(graph._func_graph.params_tree, names)
    for key in ("ee_par", "beta_par"):
        assert "first_order" in labels["params"][key]

    alpha = params["params"]["ee_par"][0]
    beta = params["params"]["beta_par"][0]

    def reference(coords):
        distance = jnp.linalg.norm(coords[0] - coords[1])
        return -(beta * alpha**2) / (alpha + distance)

    actual = lambda coords: layer.apply(params, coords)
    np.testing.assert_allclose(actual(positions), reference(positions), rtol=1e-6)
    np.testing.assert_allclose(
        jax.grad(actual)(positions), jax.grad(reference)(positions), rtol=1e-6
    )
    np.testing.assert_allclose(
        jax.hessian(actual)(positions),
        jax.hessian(reference)(positions),
        rtol=1e-6,
        atol=1e-7,
    )
