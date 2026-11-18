# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Run one of two Hall training processes, each using four local GPUs."""

import os
import sys
from pathlib import Path

import jax
import yaml

from jaqmc.utils.config import ConfigManager
from jaqmc.utils.runtime import configure_runtime


def main() -> None:
    config, coordinator = sys.argv[1:]
    rank = int(os.environ["SLURM_PROCID"])
    cfg = ConfigManager(
        [yaml.safe_load(Path(config).read_text(encoding="utf-8"))],
        [
            f"distributed.coordinator_address={coordinator}",
            "distributed.num_processes=2",
            f"distributed.process_id={rank}",
        ],
    )
    configure_runtime(cfg)
    devices = jax.devices()
    local = jax.local_devices()
    print(
        f"Process {rank}: global devices={devices}; local devices={local}", flush=True
    )
    if (
        len(devices) != 8
        or len(local) != 4
        or any(d.platform != "gpu" for d in devices)
    ):
        raise RuntimeError("Two processes with four GPUs each are required")

    from jaqmc.app.hall.workflow import HallTrainWorkflow

    HallTrainWorkflow(cfg)()
    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
