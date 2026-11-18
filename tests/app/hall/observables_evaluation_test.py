# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Physical invariants for multigeometry evaluation additions."""

import importlib.util
from operator import itemgetter
from pathlib import Path

import jax
import numpy as np
import pytest
from jax import numpy as jnp
from scipy.special import eval_legendre, sph_harm_y

from jaqmc.app.hall.data import HallData
from jaqmc.app.hall.estimator.disk.one_rdm import DiskOneRDM
from jaqmc.app.hall.estimator.disk.structure_factor import DiskStructureFactor
from jaqmc.app.hall.estimator.sphere.one_rdm import SphereOneRDM, make_monopole_harm
from jaqmc.app.hall.estimator.sphere.structure_factor import spherical_density_modes
from jaqmc.app.hall.estimator.torus.structure_factor import TorusStructureFactor
from jaqmc.app.hall.wavefunction.torus.laughlin import TorusLaughlin
from jaqmc.data import BatchedData
from jaqmc.estimator.base import EstimatorPipeline
from jaqmc.estimator.kinetic.torus import TorusKinetic
from jaqmc.utils import parallel_jax


def test_spherical_modes_and_legendre_addition():
    coordinates = jnp.asarray([[0.13, -1.7], [1.2, 0.8], [2.8, 2.0]])
    modes = spherical_density_modes(coordinates, 20)
    reference = np.array(
        [
            sph_harm_y(
                ell, m, np.asarray(coordinates[:, 0]), np.asarray(coordinates[:, 1])
            ).sum()
            for m in range(21)
            for ell in range(m, 21)
        ]
    )
    np.testing.assert_allclose(modes, reference, atol=3e-6)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    xyz = np.stack(
        [
            np.sin(coordinates[:, 0]) * np.cos(coordinates[:, 1]),
            np.sin(coordinates[:, 0]) * np.sin(coordinates[:, 1]),
            np.cos(coordinates[:, 0]),
        ],
        axis=-1,
    )
    for ell in (0, 1, 7, 20):
        labels = [(level, m) for m in range(21) for level in range(m, 21)]
        value = sum(
            (1 if m == 0 else 2) * abs(modes[k]) ** 2
            for k, (level, m) in enumerate(labels)
            if level == ell
        )
        direct = eval_legendre(ell, np.clip(xyz @ xyz.T, -1, 1)).sum()
        np.testing.assert_allclose(4 * np.pi * value / (2 * ell + 1), direct, atol=2e-5)


def test_three_sphere_levels_orthonormal_and_finite_poles():
    x, weights = np.polynomial.legendre.leggauss(64)
    phi = np.arange(64) * 2 * np.pi / 64
    theta, phi_grid = np.meshgrid(np.arccos(x), phi, indexing="ij")
    coords = jnp.asarray(np.stack([theta, phi_grid], axis=-1))
    orbitals = [
        make_monopole_harm(7.5, 7.5 + n, m)
        for n in range(3)
        for m in np.arange(-7.5 - n, 8.5 + n)
    ]
    values = np.stack([orb(coords) for orb in orbitals], axis=-1)
    overlap = (
        np.einsum("xpi,xpj,x->ij", values.conj(), values, weights) * 2 * np.pi / 64
    )
    np.testing.assert_allclose(overlap, np.eye(54), atol=3e-5)
    poles = jnp.asarray([[0.0, 0.3], [jnp.pi, 1.7]])
    assert all(np.isfinite(orb(poles)).all() for orb in orbitals)
    estimator = SphereOneRDM(flux=15, num_landau_levels=3)
    estimator.init(HallData(electrons=poles), jax.random.key(0))
    digest = estimator.finalize_stats({"one_rdm": jnp.eye(54)[None]}, None)
    np.testing.assert_allclose(digest["one_rdm:landau_level_occupations"], [16, 18, 20])


@pytest.mark.parametrize("sector", [0, 1, 2])
@pytest.mark.requires_x64
def test_torus_laughlin_boundaries_exchange_and_kinetic(sector):
    wf = TorusLaughlin(nspins=(3, 0), flux=9, tau=0.2 + 1.1j, sector=sector)
    electrons = jnp.asarray(
        [[0.13, 0.27], [0.48, 0.18], [0.79, 0.64]], dtype=jnp.float64
    )
    data = HallData(electrons=electrons)
    params = wf.init(jax.random.key(0), data)
    logpsi = wf.logpsi(params, data)
    l1 = np.sqrt(2 * np.pi * 9 / 1.1)
    z = l1 * (electrons[0, 0] + wf.tau * electrons[0, 1])
    for shift, period in (([1.0, 0.0], l1), ([0.0, 1.0], l1 * wf.tau)):
        displaced = data.merge({"electrons": electrons.at[0].add(jnp.asarray(shift))})
        ratio = jnp.exp(wf.logpsi(params, displaced) - logpsi)
        phase = (-1) ** 9 * jnp.exp((np.conj(period) * z - period * np.conj(z)) / 4)
        np.testing.assert_allclose(ratio, phase, atol=1e-9)
    swapped = data.merge({"electrons": electrons[jnp.array([1, 0, 2])]})
    np.testing.assert_allclose(
        jnp.exp(wf.logpsi(params, swapped) - logpsi), -1, atol=1e-10
    )
    kinetic = TorusKinetic(flux=9, tau=wf.tau, f_log_psi=wf.logpsi)
    result, _ = jax.jit(kinetic.evaluate_single_walker)(
        params, data, {}, None, jax.random.key(2)
    )
    np.testing.assert_allclose(result["energy:kinetic"], 1.5, atol=1e-8)
    refined = TorusLaughlin(
        nspins=(3, 0), flux=9, tau=wf.tau, sector=sector, theta_terms=18
    )
    np.testing.assert_allclose(
        jnp.exp(refined.logpsi(params, data) - logpsi), 1, atol=1e-10
    )


def test_structure_zero_mode_and_torus_translation():
    data = HallData(electrons=jnp.asarray([[0.2, 0.3], [0.4, 0.7], [0.8, 0.9]]))
    key = jax.random.key(1)
    torus = TorusStructureFactor(flux=9)
    torus.init(data, key)
    stats, _ = torus.evaluate_single_walker({}, data, {}, None, key)
    moved, _ = torus.evaluate_single_walker(
        {},
        data.merge({"electrons": data.electrons + jnp.array([1.0, -2.0])}),
        {},
        None,
        key,
    )
    np.testing.assert_allclose(
        stats["structure_factor:power"], moved["structure_factor:power"], atol=6e-5
    )
    final = torus.finalize_stats(jax.tree.map(itemgetter(None), stats), None)
    np.testing.assert_allclose(final["structure_factor"][0], 3)
    disk = DiskStructureFactor(num_q=3)
    state = disk.init(data, key)
    batch = BatchedData(
        data=HallData(electrons=jnp.stack([data.electrons] * 4)),
        fields_with_batch=["electrons"],
    )
    stats, state = disk.evaluate_batch_walkers({}, batch, {}, state, key)
    assert stats["structure_factor:rho_real"].shape[0] == batch.batch_size
    assert np.isclose(disk.reduce(stats)["structure_factor:power"][0, 0], 9)
    assert np.isclose(disk.finalize_state(state, n_steps=1)["structure_factor"][0], 3)


def test_occupations_without_natural_eigenvalues():
    estimator = DiskOneRDM(num_orbitals=20, num_landau_levels=3)
    estimator.init(HallData(electrons=jnp.zeros((6, 2))), jax.random.key(0))
    diagonal = jnp.zeros(60).at[jnp.array([0, 1, 2, 20, 21, 40])].set(1.0)
    result = estimator.finalize_stats({"one_rdm": jnp.diag(diagonal)[None]}, None)
    np.testing.assert_allclose(result["one_rdm:landau_level_occupations"], [3, 2, 1])
    assert "one_rdm:natural_occupations" not in result


def test_disk_structure_and_density_matrix_share_pipeline():
    """A later per-walker estimator must receive the original walker count."""

    def logpsi(params, data):
        return -jnp.sum(data.electrons**2) / 4 + 0j

    pipeline = EstimatorPipeline(
        {
            "structure_factor": DiskStructureFactor(num_q=3, num_angles=2),
            "one_rdm": DiskOneRDM(
                num_orbitals=1, num_landau_levels=1, f_log_psi=logpsi, vmap_chunk_size=2
            ),
        }
    )
    data = BatchedData(
        data=HallData(electrons=jnp.ones((4, 1, 2))), fields_with_batch=["electrons"]
    )
    key = jax.random.key(3)
    state = pipeline.init(data, key)
    stats, _ = pipeline.evaluate({}, data, state, key)
    assert stats["one_rdm"].shape == (1, 1)
    np.testing.assert_allclose(stats["structure_factor:power"][0], 1.0)


def test_legacy_sphere_spin_parameter_does_not_change_wavefunction():
    from jaqmc.app.hall.wavefunction.sphere._jastrow import SphericalCoulombJastrow

    coordinates = jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    wf = SphericalCoulombJastrow(
        nspins=(2, 0),
        interaction_strength=1.0,
        radius=2.0,
        retain_unused_spin_parameters=True,
    )
    params = wf.init(jax.random.key(0), coordinates)
    before = wf.apply(params, coordinates)
    params["params"]["ee_anti"] = jnp.array([123.0])
    np.testing.assert_allclose(wf.apply(params, coordinates), before)


def test_torus_legacy_features_retain_five_input_channels():
    from jaqmc.app.hall.wavefunction.torus.features import torus_features

    data = jnp.asarray([[0.2, 0.3], [0.7, 0.8]])
    features = torus_features(data, include_center_of_mass=False)
    assert features.shape == (2, 4)
    np.testing.assert_allclose(features, torus_features(data)[:, :4])


def test_block_errors_and_checkpoint_subset(tmp_path):
    path = Path(__file__).resolve().parents[3] / "scripts/hall_evaluation.py"
    spec = importlib.util.spec_from_file_location("hall_evaluation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    samples = np.repeat(np.array([-1.0, 1.0, -1.0, 1.0]), 100)
    np.testing.assert_allclose(
        module.blocking(samples)["standard_error"], 1 / np.sqrt(3)
    )
    source = tmp_path / "source.npz"
    np.savez(
        source,
        **{
            "params/a": np.arange(3),
            "batched_data/data/electrons": np.arange(48).reshape(8, 3, 2),
        },
    )
    output = tmp_path / "copy.npz"
    module.prepare_checkpoint(source, output, 4, 27)
    with np.load(output) as f, np.load(source) as original:
        np.testing.assert_array_equal(f["params/a"], original["params/a"])
        assert f["batched_data/data/electrons"].shape == (4, 3, 2)
    assert module.disk_tail_error() < 1e-15


@pytest.mark.parametrize("geometry", ["disk", "sphere", "torus"])
def test_single_and_two_device_structure_statistics_match(geometry):
    """Device partitioning must not duplicate normalization or counts."""
    from jaqmc.app.hall.estimator.sphere.structure_factor import SphereStructureFactor

    coords = jnp.asarray(np.random.default_rng(9).uniform(0.1, 0.9, (8, 3, 2)))
    batch = BatchedData(
        data=HallData(electrons=coords), fields_with_batch=["electrons"]
    )
    estimator = {
        "disk": lambda: DiskStructureFactor(num_q=5, num_angles=2),
        "sphere": lambda: SphereStructureFactor(ell_max=3),
        "torus": lambda: TorusStructureFactor(flux=9, q_max=2.0),
    }[geometry]()
    key = jax.random.key(0)
    initial = estimator.init(HallData(electrons=coords[0]), key)
    raw, expected_state = estimator.evaluate_batch_walkers({}, batch, {}, initial, key)
    expected = estimator.reduce(raw)

    def evaluate(electrons, state):
        local = BatchedData(
            data=HallData(electrons=electrons), fields_with_batch=["electrons"]
        )
        stats, following = estimator.evaluate_batch_walkers({}, local, {}, state, key)
        return estimator.reduce(stats), following

    sharded = parallel_jax.jit_sharded(
        evaluate,
        in_specs=(parallel_jax.DATA_PARTITION, parallel_jax.SHARE_PARTITION),
        out_specs=(parallel_jax.SHARE_PARTITION, parallel_jax.SHARE_PARTITION),
    )
    coords = jax.device_put(
        coords, parallel_jax.make_sharding(parallel_jax.DATA_PARTITION)
    )
    initial = jax.device_put(
        initial, parallel_jax.make_sharding(parallel_jax.SHARE_PARTITION)
    )
    actual, actual_state = sharded(coords, initial)
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], rtol=2e-6, atol=2e-6)
    if geometry == "disk":
        np.testing.assert_allclose(
            estimator.finalize_state(actual_state, n_steps=1)["structure_factor"],
            estimator.finalize_state(expected_state, n_steps=1)["structure_factor"],
            rtol=2e-6,
        )
