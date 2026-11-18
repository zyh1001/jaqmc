# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Verify torus Jastrow registration and coordinate derivatives."""

import logging

import jax
import kfac_jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.app.hall.wavefunction.torus.jastrow import TorusJastrow
from jaqmc.optimizer.kfac.tag_registration import make_graph_patterns


@pytest.mark.parametrize("nspins", [(2, 0), (1, 1), (2, 1)])
def test_torus_jastrow_first_order_tags_and_coordinate_derivatives(
    nspins: tuple[int, int], caplog: pytest.LogCaptureFixture
) -> None:
    layer = TorusJastrow(nspins=nspins, l1=2.0, l2=3.0j)
    positions = jnp.array([[0.1, 0.2], [0.4, 0.7], [0.8, 0.3]], dtype=jnp.float32)[
        : sum(nspins)
    ]
    params = layer.init(jax.random.key(0), positions)
    batch = jnp.stack([positions, positions + 0.01])

    def tagged_value(p, coords):
        result = jax.vmap(layer.apply, in_axes=(None, 0))(p, coords)
        kfac_jax.register_normal_predictive_distribution(result[:, None])
        return result

    graph = kfac_jax.tag_graph_matcher.auto_register_tags(
        tagged_value, (params, batch), graph_patterns=make_graph_patterns()
    )
    with caplog.at_level(logging.INFO):
        graph.print_parameter_tags()
    assert caplog.text.count("first_order") == len(params["params"])
    assert "Orphan" not in caplog.text

    def reference(coords):
        result = jnp.asarray(0.0)
        for i in range(sum(nspins)):
            for j in range(i + 1, sum(nspins)):
                delta = coords[i] - coords[j]
                distance = (
                    jnp.linalg.norm(jnp.array([2.0, 3.0]) * jnp.sin(jnp.pi * delta))
                    / jnp.pi
                )
                parallel = (i < nspins[0]) == (j < nspins[0])
                alpha = params["params"]["ee_par" if parallel else "ee_anti"][0]
                result -= alpha**2 / (alpha + distance) / (3.0 if parallel else 1.0)
        return result

    def actual(coords):
        return layer.apply(params, coords)

    np.testing.assert_allclose(actual(positions), reference(positions), rtol=1e-6)
    np.testing.assert_allclose(
        jax.grad(actual)(positions),
        jax.grad(reference)(positions),
        rtol=1e-5,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        jax.hessian(actual)(positions),
        jax.hessian(reference)(positions),
        rtol=1e-5,
        atol=1e-6,
    )
