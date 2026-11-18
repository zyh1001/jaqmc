# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Prepare, run, validate, and summarize fixed-parameter Hall evaluations."""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml
from scipy.integrate import quad
from scipy.special import eval_genlaguerre, factorial

SOURCES = {
    "disk": "disk/n6_Q18_kapa1_8192_final/train_ckpt_149999.npz",
    "sphere": "spherical/mhpo-1det/train/train_ckpt_199999.npz",
    "torus": (
        "torus/interview/torus-n6-nu1_3-square-bugfix/"
        "raw/training/train_ckpt_1999999.npz"
    ),
}
PROJECT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    """Return a file checksum without loading the entire file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path: Path, value: Any) -> None:
    """Write readable JSON with explicit null for unavailable values."""
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def prepare_checkpoint(source: Path, output: Path, batch: int, seed: int) -> dict:
    """Copy exact parameters and a reproducible subset of saved walkers.

    Returns:
        Source checksums and walker-selection metadata.

    Raises:
        ValueError: If the batch size or copied parameters are invalid.
    """
    with np.load(source, allow_pickle=False) as data:
        electrons = data["batched_data/data/electrons"]
        if not 0 < batch <= len(electrons):
            raise ValueError("Requested walker count exceeds the source checkpoint")
        indices = np.random.default_rng(seed).choice(
            len(electrons), batch, replace=False
        )
        arrays = {
            k: data[k]
            for k in data.files
            if k.startswith(("params/", "batched_data/", "sampler_state/"))
        }
        for key, value in arrays.items():
            if (
                key.startswith("batched_data/")
                and value.ndim
                and len(value) == len(electrons)
            ):
                arrays[key] = value[indices]
        np.savez_compressed(output, step=np.array(0), **arrays)
        with np.load(output, allow_pickle=False) as copied:
            parameter_keys = [k for k in data.files if k.startswith("params/")]
            if not parameter_keys or not all(
                np.array_equal(data[k], copied[k]) for k in parameter_keys
            ):
                raise ValueError(
                    "Prepared parameters differ from the trained parameters"
                )
    np.save(output.parent / "walker_indices.npy", indices)
    return {
        "source_checkpoint": str(source),
        "source_sha256": sha256(source),
        "prepared_checkpoint": str(output),
        "prepared_sha256": sha256(output),
        "source_batch_size": len(electrons),
        "batch_size": batch,
        "selection_seed": seed,
        "parameter_arrays": len(parameter_keys),
    }


def wavefunction_config(train: dict, geometry: str, laughlin: bool) -> dict:
    """Resolve a checkpoint-compatible wavefunction configuration.

    Returns:
        Wavefunction fields with explicit legacy input conventions.

    Raises:
        ValueError: If nonzero legacy sphere flux attachment is requested.
    """
    wf = dict(train["wf"])
    if geometry == "sphere" and not laughlin and "flux_per_elec" in wf:
        # This dataset used zero flux attachment before dressing was renamed.
        attachment = wf.pop("flux_per_elec")
        if attachment != 0:
            raise ValueError(
                "Only the zero-attachment legacy sphere checkpoint is supported"
            )
        wf["dressing"] = {"degree": 0}
        wf["retain_unused_spin_parameters"] = True
    if geometry == "torus" and not laughlin:
        # The requested checkpoint predates the extra four shared features.
        wf.setdefault("include_center_of_mass", False)
    wf["module"] = f"jaqmc.app.hall.wavefunction.{geometry}." + (
        "laughlin" if laughlin else "mhpo"
    )
    if laughlin:
        wf = {"module": wf["module"]}
        if geometry == "sphere":
            wf["flux_per_elec"] = 1
        if geometry == "torus":
            wf.update(sector=0, theta_terms=12)
    return wf


def build_config(
    train: dict,
    geometry: str,
    output: Path,
    source: Path | None,
    batch: int,
    steps: int,
    burn_in: int,
    seed: int,
    laughlin: bool = False,
) -> dict:
    """Use the original physical configuration with evaluation-only settings.

    Returns:
        A complete evaluation configuration.

    """
    system = dict(train["system"])
    system["module"] = (
        "jaqmc.app.hall.config:Hall"
        + {"disk": "Disk", "sphere": "Sphere", "torus": "Torus"}[geometry]
        + "Config"
    )
    wf = wavefunction_config(train, geometry, laughlin)
    estimators = {
        "enabled": {
            "energy": True,
            "one_rdm": not laughlin and geometry != "torus",
            "structure_factor": not laughlin,
            "density": False,
            "pair_correlation": False,
        },
        "energy": {
            "kinetic": {"vmap_chunk_size": 64},
            "potential": {"vmap_chunk_size": 128},
        },
    }
    if geometry == "sphere":
        estimators["angular_momentum"] = {"vmap_chunk_size": 64}
    if not laughlin:
        if geometry == "disk":
            estimators["one_rdm"] = {
                "num_orbitals": 20,
                "num_landau_levels": 3,
                "radius_max": 15.0,
                "vmap_chunk_size": 32,
            }
        elif geometry == "sphere":
            estimators["one_rdm"] = {"num_landau_levels": 3, "vmap_chunk_size": 32}
            estimators["structure_factor"] = {"ell_max": 20, "vmap_chunk_size": 128}
        else:
            estimators["structure_factor"] = {"q_max": 5.0, "vmap_chunk_size": 128}
    sampler = dict(train.get("sampler", {}))
    sampler["steps"] = 10
    fields = "energy=total_energy_real:.7f"
    return {
        "system": system,
        "wf": wf,
        "workflow": {
            "batch_size": batch,
            "source_path": str(source) if source else None,
            "save_path": str(output),
            "seed": seed,
        },
        "sampler": sampler,
        "estimators": estimators,
        "run": {
            "iterations": steps,
            "burn_in": burn_in,
            "save_step_interval": 1000,
            "save_time_interval": 0,
            "digest_step_interval": 0,
        },
        "writers": {
            "console": {
                "module": "jaqmc.writer.console:ConsoleWriter",
                "fields": fields,
                "interval": 100,
            }
        },
    }


def blocking(values: np.ndarray, block_length: int = 100) -> dict[str, np.ndarray]:
    """Estimate mean errors from consecutive equal-length blocks.

    Returns:
        Mean, componentwise standard errors, and block means.

    Raises:
        ValueError: If samples are nonfinite or fewer than two blocks exist.
    """
    values = np.asarray(values)
    values = values.astype(np.complex128 if np.iscomplexobj(values) else np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite samples cannot be silently discarded")
    count = len(values) // block_length
    if count < 2:
        raise ValueError("At least two blocks are required")
    blocks = (
        values[: count * block_length]
        .reshape(count, block_length, *values.shape[1:])
        .mean(axis=1)
    )
    return {
        "mean": values.mean(axis=0),
        "standard_error": blocks.real.std(axis=0, ddof=1) / np.sqrt(count),
        "imaginary_standard_error": blocks.imag.std(axis=0, ddof=1) / np.sqrt(count),
        "block_means": blocks,
    }


def integrated_autocorrelation(values: np.ndarray) -> float:
    """Estimate integrated autocorrelation using positive adjacent pairs.

    Returns:
        Integrated correlation time in recorded-step units.
    """
    x = np.asarray(values, dtype=np.float64)
    x = x - x.mean()
    if np.dot(x, x) == 0:
        return 0.5
    fft = np.fft.rfft(x, n=2 * len(x))
    corr = np.fft.irfft(fft * fft.conj())[: len(x)] / np.arange(len(x), 0, -1)
    corr /= corr[0]
    pairs = corr[1 : len(corr) // 2 * 2 - 1].reshape(-1, 2).sum(axis=1)
    nonpositive = np.flatnonzero(pairs <= 0)
    stop = int(nonpositive[0]) if len(nonpositive) else len(pairs)
    return float(max(0.5, 0.5 + pairs[:stop].sum()))


def structure_statistics(
    real: np.ndarray,
    imag: np.ndarray,
    power: np.ndarray,
    geometry: str,
    digest: dict,
    block_length: int = 100,
) -> dict:
    """Compute density-mode averages and block jackknife connected errors.

    Returns:
        Mode averages, bias-corrected connected estimates, and errors.

    Raises:
        ValueError: If fewer than two complete time blocks exist.
    """
    count = len(power) // block_length
    if count < 2:
        raise ValueError("At least two blocks are required for structure-factor errors")
    n = float(digest["structure_factor:electron_number"])

    def group(values):
        if geometry == "disk":
            return values.mean(axis=-1) / n
        if geometry == "torus":
            return values / n
        ell = digest["structure_factor:mode_ell"].astype(int)
        weights = digest["structure_factor:mode_weight"]
        result = np.zeros((*values.shape[:-1], len(digest["structure_factor:ell"])))
        for mode, label in enumerate(ell):
            result[..., label] += weights[mode] * values[..., mode]
        return result

    rho = np.asarray(real, np.float64) + 1j * np.asarray(imag, np.float64)
    power = np.asarray(power, np.float64)
    # All formal runs use a step count divisible by the block length.
    rho = rho[: count * block_length]
    power = power[: count * block_length]
    rb = rho.reshape(count, block_length, *rho.shape[1:]).mean(axis=1)
    pb = power.reshape(count, block_length, *power.shape[1:]).mean(axis=1)
    rmean, pmean = rb.mean(axis=0), pb.mean(axis=0)
    raw = group(pmean)
    disconnected = group(np.abs(rmean) ** 2)
    rleave = (count * rmean - rb) / (count - 1)
    pleave = (count * pmean - pb) / (count - 1)
    leave = group(pleave - np.abs(rleave) ** 2)
    plugin = raw - disconnected
    connected = count * plugin - (count - 1) * leave.mean(axis=0)
    connected_error = np.sqrt(
        (count - 1) / count * np.sum((leave - leave.mean(axis=0)) ** 2, axis=0)
    )
    out = {
        "value": raw,
        "standard_error": group(pb).std(axis=0, ddof=1) / np.sqrt(count),
        "connected": connected,
        "connected_plugin": plugin,
        "connected_standard_error": connected_error,
        "disconnected": disconnected,
        "disconnected_bias_corrected": raw - connected,
        "rho_mean": rmean,
        "block_count": np.array(count),
    }
    if geometry == "torus":
        q = np.asarray(digest["structure_factor:q"])
        shell, inverse = np.unique(np.round(q, 6), return_inverse=True)
        out["shell_q"] = shell
        out["shell_index"] = inverse
        for key in ("value", "connected", "disconnected"):
            out["shell_" + key] = np.array(
                [out[key][inverse == s].mean() for s in range(len(shell))]
            )
        raw_blocks = group(pb)
        shell_blocks = np.stack(
            [raw_blocks[:, inverse == s].mean(axis=1) for s in range(len(shell))],
            axis=-1,
        )
        out["shell_standard_error"] = shell_blocks.std(axis=0, ddof=1) / np.sqrt(count)
        shell_leave = np.stack(
            [leave[:, inverse == s].mean(axis=1) for s in range(len(shell))], axis=-1
        )
        out["shell_connected_standard_error"] = np.sqrt(
            (count - 1)
            / count
            * ((shell_leave - shell_leave.mean(axis=0)) ** 2).sum(axis=0)
        )
    return out


def validate_output(directory: Path, config: dict, source: Path | None) -> dict:
    """Validate sample count, parameter identity, and per-step energy sums.

    Returns:
        The completed validation record.

    Raises:
        ValueError: If samples, parameters, or physical consistency checks fail.
    """
    steps = config["run"]["iterations"]
    checkpoint = directory / f"evaluation_ckpt_{steps - 1:06d}.npz"
    with np.load(checkpoint, allow_pickle=False) as data:
        if int(data["step"]) != steps - 1:
            raise ValueError("Evaluation checkpoint has an unexpected step")
        if (
            data["batched_data/data/electrons"].shape[0]
            != config["workflow"]["batch_size"]
        ):
            raise ValueError("Actual walker count differs from the requested count")
        if source:
            with np.load(source, allow_pickle=False) as original:
                keys = {k for k in original.files if k.startswith("params/")}
                if keys != {
                    k for k in data.files if k.startswith("params/")
                } or not all(np.array_equal(original[k], data[k]) for k in keys):
                    raise ValueError(
                        "Wavefunction parameters changed during evaluation"
                    )
    with h5py.File(directory / "evaluation_stats.h5") as f:
        if len(f["total_energy"]) != steps:
            raise ValueError("Incomplete evaluation records")
        total = f["total_energy"][:]
        summed = f["energy:kinetic"][:] + f["energy:potential"][:]
        if not np.isfinite(total).all() or not np.allclose(
            total, summed, rtol=1e-6, atol=2e-6
        ):
            raise ValueError("Energy records are nonfinite or E differs from T+V")
        for key in (
            "one_rdm",
            "structure_factor:rho_real",
            "structure_factor:rho_imag",
            "structure_factor:power",
        ):
            if key in f and not np.isfinite(f[key][:]).all():
                raise ValueError(f"Nonfinite samples in {key}")
    validate_zero_mode(directory, config)
    result = {
        "passed": True,
        "steps": steps,
        "actual_batch_size": config["workflow"]["batch_size"],
        "sample_count": steps * config["workflow"]["batch_size"],
        "parameters_unchanged": bool(source),
        "energy_sum_verified": True,
    }
    save_json(directory / "validation.json", result)
    return result


def validate_zero_mode(directory: Path, config: dict) -> None:
    """Check the exact zero-wave-number normalization when enabled.

    Raises:
        ValueError: If the full structure factor at zero is not N.
    """
    if config.get("estimators", {}).get("enabled", {}).get("structure_factor", False):
        with np.load(directory / "evaluation_digest.npz", allow_pickle=False) as digest:
            if not np.isclose(
                digest["structure_factor"][0],
                sum(config["system"]["nspins"]),
                atol=2e-4,
            ):
                raise ValueError(
                    "The zero density mode does not equal the electron number"
                )


def summarize(directory: Path, geometry: str) -> dict:
    """Write block errors, occupations, structure factors, and metadata.

    Returns:
        The scalar summary written beside the complete numerical arrays.
    """
    with np.load(directory / "evaluation_digest.npz", allow_pickle=False) as d:
        digest = dict(d)
    arrays = {}
    summary = {
        "geometry": geometry,
        "block_length": 100,
        "observables": {},
        "error_method": (
            "consecutive time blocks; connected structure factor uses block jackknife"
        ),
    }
    with h5py.File(directory / "evaluation_stats.h5") as f:
        for key in (
            "total_energy_real",
            "energy:kinetic",
            "energy:potential",
            "observable:angular_momentum_z",
            "observable:angular_momentum_z_square",
            "angular_momentum_z",
            "angular_momentum_z_square",
            "angular_momentum_square",
        ):
            if key not in f:
                continue
            series = f[key][:]
            stat = blocking(series)
            trials = {
                str(length): float(blocking(series, length)["standard_error"])
                for length in (50, 100, 200, 500)
            }
            errors = np.array(list(trials.values()))
            positive = errors[errors > 1e-12]
            stable = not len(positive) or positive.max() / positive.min() <= 1.5
            summary["observables"][key] = {
                "mean": float(np.real(stat["mean"])),
                "standard_error": float(stat["standard_error"]),
                "imaginary_mean": float(np.imag(stat["mean"])),
                "imaginary_standard_error": float(stat["imaginary_standard_error"]),
                "tested_standard_errors": trials,
                "error_stable": bool(stable),
                "status": "complete" if stable else "provisional_error",
                "integrated_autocorrelation_steps": integrated_autocorrelation(
                    series.real
                ),
            }
            arrays[key + ":block_means"] = stat["block_means"]
        if "one_rdm" in f:
            matrix = f["one_rdm"][:].astype(np.complex128)
            diagonal = np.diagonal(matrix, axis1=-2, axis2=-1).real
            labels = digest["one_rdm:basis_landau_level"].astype(int)
            levels = np.stack(
                [diagonal[:, labels == n].sum(axis=-1) for n in range(3)], axis=-1
            )
            for name, series in (
                ("matrix", matrix),
                ("diagonal", diagonal),
                ("landau_level_occupations", levels),
                ("trace", np.trace(matrix, axis1=-2, axis2=-1)),
            ):
                stat = blocking(series)
                for suffix in (
                    "mean",
                    "standard_error",
                    "imaginary_standard_error",
                    "block_means",
                ):
                    arrays["one_rdm:" + name + ":" + suffix] = stat[suffix]
            arrays["one_rdm:landau_level_series"] = levels
            mean_matrix = matrix.mean(axis=0)
            summary["one_rdm"] = {
                "landau_level_occupations": levels.mean(axis=0).tolist(),
                "standard_error": blocking(levels)["standard_error"].tolist(),
                "trace_real": float(np.trace(mean_matrix).real),
                "trace_imaginary": float(np.trace(mean_matrix).imag),
                "hermiticity_error": float(
                    np.linalg.norm(mean_matrix - mean_matrix.conj().T)
                    / np.linalg.norm(mean_matrix)
                ),
                "natural_occupations_computed": False,
                "basis_truncation": (
                    "three Landau levels; disk has 20 orbitals per level"
                ),
            }
            arrays["one_rdm:tested_block_sizes"] = np.array([50, 100, 200, 500])
            arrays["one_rdm:tested_occupation_standard_errors"] = np.stack(
                [
                    blocking(levels, length)["standard_error"]
                    for length in (50, 100, 200, 500)
                ]
            )
        if "structure_factor:power" in f:
            sf = structure_statistics(
                f["structure_factor:rho_real"][:],
                f["structure_factor:rho_imag"][:],
                f["structure_factor:power"][:],
                geometry,
                digest,
            )
            arrays.update({"structure_factor:" + k: v for k, v in sf.items()})
            for key, value in digest.items():
                if key.startswith("structure_factor:") and key not in arrays:
                    arrays[key] = value
            tested = [
                structure_statistics(
                    f["structure_factor:rho_real"][:],
                    f["structure_factor:rho_imag"][:],
                    f["structure_factor:power"][:],
                    geometry,
                    digest,
                    length,
                )
                for length in (50, 200, 500)
            ]
            arrays["structure_factor:tested_block_sizes"] = np.array(
                [50, 100, 200, 500]
            )
            arrays["structure_factor:tested_standard_errors"] = np.stack(
                [
                    tested[0]["standard_error"],
                    sf["standard_error"],
                    tested[1]["standard_error"],
                    tested[2]["standard_error"],
                ]
            )
            arrays["structure_factor:tested_connected_standard_errors"] = np.stack(
                [
                    tested[0]["connected_standard_error"],
                    sf["connected_standard_error"],
                    tested[1]["connected_standard_error"],
                    tested[2]["connected_standard_error"],
                ]
            )
    np.savez_compressed(directory / "statistics.npz", **arrays)
    save_json(directory / "summary.json", summary)
    return summary


def disk_tail_error(radius: float = 15.0) -> float:
    """Largest missing orbital norm beyond the auxiliary integration disk.

    Returns:
        Maximum omitted norm over the sixty retained disk orbitals.
    """
    errors = []
    for n in range(3):
        for q in range(20):
            m = q - n
            if m < 0:
                radial_n, alpha = q, -m
            else:
                radial_n, alpha = n, m
            norm = factorial(radial_n) / factorial(radial_n + alpha)
            value, _ = quad(
                lambda t: (
                    norm
                    * t**alpha
                    * eval_genlaguerre(radial_n, alpha, t) ** 2
                    * np.exp(-t)
                ),
                radius**2 / 2,
                np.inf,
            )
            errors.append(value)
    return float(max(errors))


def run_evaluation(config: dict, directory: Path, dry_run: bool = False) -> None:
    """Invoke the normal JaQMC command and retain its complete log.

    Raises:
        RuntimeError: If the evaluation command fails.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "requested_config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    if list(directory.glob("evaluation_ckpt_*.npz")) and not dry_run:
        config["workflow"]["restore_path"] = str(directory)
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    command = [
        sys.executable,
        "-c",
        "from jaqmc.app.cli import cli; cli()",
        "hall",
        "evaluate",
        "--yml",
        str(path),
    ]
    if dry_run:
        command.append("--dry-run")
    with (directory / ("dry_run.log" if dry_run else "evaluation.log")).open(
        "w"
    ) as stream:
        result = subprocess.run(
            command, cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT, check=False
        )
    if result.returncode:
        raise RuntimeError(f"JaQMC evaluation failed; see {directory}/evaluation.log")


def run(args: argparse.Namespace) -> None:
    """Run one geometry's trained state and analytic reference sequentially.

    Raises:
        ValueError: If the requested input or result fails validation.
    """
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = (
        args.train_dir.resolve() / Path(SOURCES[args.geometry]).name
        if getattr(args, "train_dir", None)
        else args.data_root.resolve() / SOURCES[args.geometry]
    )
    train = yaml.safe_load((source.parent / "train_config.yaml").read_text())
    input_dir = output / "inputs"
    input_dir.mkdir(exist_ok=True)
    prepared = input_dir / "train_ckpt_000000.npz"
    provenance = prepare_checkpoint(source, prepared, args.batch_size, args.seed)
    provenance.update(
        geometry=args.geometry,
        source_config_sha256=sha256(source.parent / "train_config.yaml"),
        job_id=os.environ.get("SLURM_JOB_ID"),
        seed=args.seed,
    )
    code = {
        str(p.relative_to(PROJECT)): sha256(p)
        for p in sorted((PROJECT / "src/jaqmc").rglob("*.py"))
    }
    code.update({"scripts/hall_evaluation.py": sha256(Path(__file__))})
    save_json(output / "code_checksums.json", code)
    provenance["code_checksums_sha256"] = sha256(output / "code_checksums.json")
    if args.geometry == "disk":
        provenance["maximum_orbital_norm_outside_auxiliary_disk"] = disk_tail_error()
    save_json(output / "provenance.json", provenance)
    states = (
        ((False, "trained"),)
        if getattr(args, "trained_only", False)
        else ((False, "trained"), (True, "laughlin"))
    )
    for analytic, name in states:
        directory = output / name
        config = build_config(
            train,
            args.geometry,
            directory,
            None if analytic else prepared,
            args.batch_size,
            args.steps,
            args.burn_in,
            args.seed + int(analytic),
            analytic,
        )
        if getattr(args, "disk_spatial", False) and not analytic:
            if args.geometry != "disk":
                raise ValueError("Disk spatial estimators require disk geometry")
            config["estimators"]["enabled"].update(density=True, pair_correlation=True)
            config["estimators"]["density"] = {
                "bins": 400,
                "range_max": 10.0,
                "radial_bin_width_pixels": 2.0,
            }
            config["estimators"]["pair_correlation"] = {
                "bins": 800,
                "max_r": 15.0,
                "r": 3.0,
                "reference_bin_width": None,
            }
        if args.dry_run:
            run_evaluation(config, directory, True)
            continue
        if args.smoke and not (output / (name + "_smoke") / "validation.json").exists():
            smoke_dir = output / (name + "_smoke")
            smoke = copy.deepcopy(config)
            smoke["workflow"]["save_path"] = str(smoke_dir)
            smoke["run"].update(iterations=3, burn_in=2)
            run_evaluation(smoke, smoke_dir)
            validate_output(smoke_dir, smoke, None if analytic else prepared)
        # A fully validated run is reusable after reconnecting to the cluster.
        if not (directory / "validation.json").exists():
            run_evaluation(config, directory)
            validate_output(directory, config, None if analytic else prepared)
        summary = summarize(directory, args.geometry)
        save_json(
            directory / "run_manifest.json",
            {
                **provenance,
                "status": "complete",
                "state": name,
                "steps": args.steps,
                "config": config,
                "summary": summary["observables"]["total_energy_real"],
            },
        )
    if not args.dry_run:
        save_json(
            output / "completion.json",
            {
                "geometry": args.geometry,
                "status": "complete",
                "steps": args.steps,
                "batch_size": args.batch_size,
            },
        )
        checksums = "".join(
            sha256(p) + "  " + str(p.relative_to(output)) + "\n"
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != "result_checksums.sha256"
        )
        (output / "result_checksums.sha256").write_text(checksums, encoding="utf-8")


def import_references(path: Path, output: Path) -> None:
    """Validate explicitly supplied ED conventions and retain their provenance.

    Raises:
        ValueError: If required conventions are missing or incompatible.
    """
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "n_electrons",
        "flux",
        "interaction_strength",
        "energy_unit",
        "background_convention",
        "basis",
        "sector",
        "source",
    }
    expected_flux = {"disk": 18, "sphere": 15, "torus": 18}
    for geometry, refs in payload.items():
        if geometry not in SOURCES:
            raise ValueError(f"Unknown geometry: {geometry}")
        for label, entry in refs.items():
            if label not in ("LLL_ED", "2LL_ED", "3LL_ED") or required - entry.keys():
                raise ValueError(
                    "Reference value lacks its complete physical conventions"
                )
            if (
                entry["n_electrons"] != 6
                or entry["flux"] != expected_flux[geometry]
                or entry["interaction_strength"] != 1
            ):
                raise ValueError(
                    "Reference system differs from the target training system"
                )
            if entry["energy_unit"] != "hbar_omega_c_total" or not np.isfinite(
                entry["energy"]
            ):
                raise ValueError("Supply a finite total energy in hbar*omega_c units")
            expected_bg = {
                "disk": "electron_background_no_background_self_energy",
                "sphere": "electron_electron_only",
                "torus": "ewald_neutral_background_and_self_images",
            }[geometry]
            if entry["background_convention"] != expected_bg:
                raise ValueError(
                    "Reference background convention requires explicit conversion"
                )
    save_json(output, {"input_sha256": sha256(path), "reference_energies": payload})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("run")
    evaluate.add_argument("geometry", choices=SOURCES)
    evaluate.add_argument("--data-root", type=Path, required=True)
    evaluate.add_argument("--train-dir", type=Path)
    evaluate.add_argument("--trained-only", action="store_true")
    evaluate.add_argument("--disk-spatial", action="store_true")
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--batch-size", type=int, default=4096)
    evaluate.add_argument("--steps", type=int, default=10000)
    evaluate.add_argument("--burn-in", type=int, default=1000)
    evaluate.add_argument("--seed", type=int, default=271828)
    evaluate.add_argument("--smoke", action="store_true")
    evaluate.add_argument("--dry-run", action="store_true")
    stats = commands.add_parser("summarize")
    stats.add_argument("geometry", choices=SOURCES)
    stats.add_argument("directory", type=Path)
    refs = commands.add_parser("import-references")
    refs.add_argument("input", type=Path)
    refs.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    elif args.command == "summarize":
        summarize(args.directory, args.geometry)
    else:
        import_references(args.input, args.output)


if __name__ == "__main__":
    main()
