# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Prepare and evaluate checkpoint distances to the final Hall training state."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml
from hall_evaluation import (
    PROJECT,
    SOURCES,
    build_config,
    prepare_checkpoint,
    save_json,
    sha256,
)


def prepare(args: argparse.Namespace) -> None:
    """Prepare one symmetric-density evaluation for each selected checkpoint.

    Raises:
        FileExistsError: If an existing plan could be overwritten.
        ValueError: If selected checkpoints are absent or incompatible.
    """
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "fubini_plan.json"
    if plan_path.exists():
        raise FileExistsError(f"An evaluation plan already exists: {plan_path}")
    final = (
        args.train_dir.resolve() / Path(SOURCES[args.geometry]).name
        if getattr(args, "train_dir", None)
        else args.data_root.resolve() / SOURCES[args.geometry]
    )
    checkpoints = {
        int(path.stem.rsplit("_", 1)[1]): path
        for path in final.parent.glob("train_ckpt_*.npz")
    }
    selected = sorted(
        checkpoints if args.checkpoint_steps is None else args.checkpoint_steps
    )
    if not selected or not set(selected).issubset(checkpoints):
        raise ValueError("Every requested training step must have a saved checkpoint")
    final_step = int(final.stem.rsplit("_", 1)[1])
    if max(checkpoints) != final_step:
        raise ValueError(
            "The specified final checkpoint is not the last saved training state"
        )
    with np.load(final, allow_pickle=False) as data:
        reference_shapes = {
            k: data[k].shape for k in data.files if k.startswith("params/")
        }
    train = yaml.safe_load((final.parent / "train_config.yaml").read_text())
    items = []
    for step in selected:
        source = checkpoints[step]
        with np.load(source, allow_pickle=False) as data:
            shapes = {k: data[k].shape for k in data.files if k.startswith("params/")}
        if not shapes or shapes != reference_shapes:
            raise ValueError(
                f"Checkpoint parameters do not match the reference: {source}"
            )
        directory = output / f"step_{step:07d}"
        directory.mkdir(exist_ok=True)
        prepared = directory / "input.npz"
        provenance = prepare_checkpoint(
            source, prepared, args.batch_size, args.seed + step
        )
        config = build_config(
            train,
            args.geometry,
            directory,
            prepared,
            args.batch_size,
            args.steps,
            args.burn_in,
            args.seed + step,
        )
        config["workflow"]["reference_path"] = str(final)
        config["estimators"] = {"fubini": {"vmap_chunk_size": 128}}
        config["writers"]["console"]["fields"] = (
            "phase=fubini:phase_mean,log_plus=fubini:log_mean_exp_delta"
        )
        config_path = directory / "requested_config.yaml"
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        items.append(
            {
                "step": step,
                "directory": str(directory),
                "config": str(config_path),
                "input": provenance,
            }
        )
    save_json(
        plan_path,
        {
            "geometry": args.geometry,
            "reference_step": final_step,
            "reference_checkpoint": str(final),
            "reference_sha256": sha256(final),
            "sampling_density": "abs(psi_checkpoint * psi_final)",
            "batch_size": args.batch_size,
            "iterations": args.steps,
            "burn_in": args.burn_in,
            "sampler_updates_per_step": 10,
            "checkpoints": items,
        },
    )
    print(f"Prepared {len(items)} checkpoint comparisons in {output}", flush=True)


def collect(output: Path) -> None:
    """Validate completed sampling and write the checkpoint-distance series.

    Raises:
        ValueError: If sampling is incomplete or the metrics are inconsistent.
    """
    plan = json.loads((output / "fubini_plan.json").read_text())
    if sha256(Path(plan["reference_checkpoint"])) != plan["reference_sha256"]:
        raise ValueError("The reference checkpoint changed after preparation")
    fields = {
        key: []
        for key in ("step", "distance", "cosine", "modulus_cosine", "phase_factor")
    }
    records = []
    for item in plan["checkpoints"]:
        directory = Path(item["directory"])
        iterations = item.get("iterations", plan["iterations"])
        burn_in = item.get("burn_in", plan["burn_in"])
        with h5py.File(directory / "overlap_stats.h5", "r") as statistics:
            keys = (
                "fubini:log_mean_exp_delta",
                "fubini:log_mean_exp_minus_delta",
                "fubini:phase_mean",
            )
            if any(statistics[key].shape != (iterations,) for key in keys):
                raise ValueError(f"Incomplete sampling: {directory}")
            if not all(np.isfinite(statistics[key][()]).all() for key in keys):
                raise ValueError(f"Nonfinite overlap statistics: {directory}")
        with np.load(directory / "overlap_digest.npz", allow_pickle=False) as digest:
            cosine = float(digest["cosine"])
            distance = float(digest["fubini_distance"])
            modulus = float(digest["cos_gamma_modulus"])
            phase = float(digest["f_phase"])
        np.testing.assert_allclose(cosine, modulus * phase, atol=2e-6)
        np.testing.assert_allclose(distance, np.arccos(cosine), atol=2e-6)
        if not 0 <= distance <= np.pi / 2 + 1e-6:
            raise ValueError(f"Invalid Fubini distance: {directory}")
        if item["step"] == plan["reference_step"]:
            np.testing.assert_allclose(distance, 0.0, atol=5e-4)
        for key, value in zip(
            fields, (item["step"], distance, cosine, modulus, phase), strict=True
        ):
            fields[key].append(value)
        records.append(
            {
                "step": item["step"],
                "distance": distance,
                "digest_sha256": sha256(directory / "overlap_digest.npz"),
                "iterations": iterations,
                "burn_in": burn_in,
                "total_configurations": plan["batch_size"] * iterations,
            }
        )
    iterations = np.array([record["iterations"] for record in records])
    burn_in = np.array([record["burn_in"] for record in records])
    np.savez(
        output / "fubini_series.npz",
        **fields,
        reference_step=plan["reference_step"],
        iterations_per_checkpoint=iterations,
        burn_in_per_checkpoint=burn_in,
        total_configurations_per_checkpoint=plan["batch_size"] * iterations,
    )
    save_json(
        output / "fubini_summary.json",
        {
            "geometry": plan["geometry"],
            "reference_step": plan["reference_step"],
            "reference_sha256": plan["reference_sha256"],
            "records": records,
            "total_configurations_per_checkpoint": (
                int(plan["batch_size"] * iterations[0])
                if np.all(iterations == iterations[0])
                else None
            ),
            "distance_unit": "rad",
            "standard_error": None,
        },
    )
    print(f"Validated {len(records)} checkpoint distances in {output}", flush=True)


def run(output: Path, dry_run: bool) -> None:
    """Run the native overlap command sequentially for a prepared geometry.

    Raises:
        RuntimeError: If a native overlap evaluation fails.
        ValueError: If an evaluation input changed after preparation.
    """
    plan = json.loads((output / "fubini_plan.json").read_text())
    if sha256(Path(plan["reference_checkpoint"])) != plan["reference_sha256"]:
        raise ValueError("The reference checkpoint changed after preparation")
    for item in plan["checkpoints"]:
        directory = Path(item["directory"])
        if (
            sha256(Path(item["input"]["prepared_checkpoint"]))
            != item["input"]["prepared_sha256"]
        ):
            raise ValueError(f"Prepared input changed: {directory}")
        if (directory / "overlap_digest.npz").exists() and not dry_run:
            continue
        config_path = Path(item["config"])
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if config["workflow"]["batch_size"] != plan["batch_size"] or any(
            config["run"][key] != item.get(key, plan[key])
            for key in ("iterations", "burn_in")
        ):
            raise ValueError(f"Sampling configuration differs from plan: {directory}")
        if list(directory.glob("overlap_ckpt_*.npz")) and not dry_run:
            config["workflow"]["restore_path"] = str(directory)
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
        command = [
            sys.executable,
            "-c",
            "from jaqmc.app.cli import cli; cli()",
            "hall",
            "overlap",
            "--yml",
            str(config_path),
        ]
        if dry_run:
            command.append("--dry-run")
        log = directory / ("dry_run.log" if dry_run else "overlap.log")
        print(f"Checkpoint {item['step']}: {log}", flush=True)
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                command,
                cwd=PROJECT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            raise RuntimeError(f"Overlap evaluation failed; see {log}")
    if not dry_run:
        collect(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("geometry", choices=SOURCES)
    prepare_parser.add_argument("--data-root", type=Path, required=True)
    prepare_parser.add_argument("--train-dir", type=Path)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--batch-size", type=int, default=4096)
    prepare_parser.add_argument("--steps", type=int, default=10000)
    prepare_parser.add_argument("--burn-in", type=int, default=1000)
    prepare_parser.add_argument("--seed", type=int, default=62119)
    prepare_parser.add_argument("--checkpoint-steps", type=int, nargs="+")
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--dry-run", action="store_true")
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "run":
        run(args.output.resolve(), args.dry_run)
    else:
        collect(args.output.resolve())


if __name__ == "__main__":
    main()
