# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Hall overlap evaluation with a dedicated sampling distribution."""

from dataclasses import fields
from functools import partial
from typing import ClassVar

from jaqmc.utils.config import ConfigManager, configurable_dataclass
from jaqmc.workflow.evaluation import EvaluationWorkflow, EvaluationWorkflowConfig
from jaqmc.workflow.stage.evaluation import EvaluationWorkStage

from .config import HallDiskConfig, HallSphereConfig, HallTorusConfig
from .data import data_init
from .estimator.disk.fubini import DiskFubini
from .estimator.fubini import HallFubini
from .estimator.sphere.fubini import SphereFubini
from .estimator.torus.fubini import TorusFubini
from .evaluation import HallOverlapWorkStage
from .workflow import configure_system, make_sampler


@configurable_dataclass
class HallOverlapConfig(EvaluationWorkflowConfig):
    """Checkpoint locations for two Hall wavefunctions of the same architecture.

    Args:
        reference_path: Reference training directory or checkpoint. If omitted,
            compare the sampling wavefunction with itself. The current parameters
            are restored from the inherited ``source_path`` setting.
    """

    reference_path: str | None = None


class HallOverlapWorkflow(EvaluationWorkflow):
    """Measure Hall overlap under the density ``abs(psi * psi_reference)``."""

    config_class: ClassVar[type[HallOverlapConfig]] = HallOverlapConfig
    config: HallOverlapConfig

    def __init__(self, cfg: ConfigManager) -> None:
        cfg.use_preset({"system": {"module": "jaqmc.app.hall.config:HallDiskConfig"}})
        super().__init__(cfg)
        system, wf = configure_system(cfg)
        estimator_class: type[HallFubini]
        if isinstance(system, HallDiskConfig):
            estimator_class = DiskFubini
        elif isinstance(system, HallSphereConfig):
            estimator_class = SphereFubini
        elif isinstance(system, HallTorusConfig):
            estimator_class = TorusFubini
        else:
            raise ValueError(f"Unsupported overlap geometry: {type(system)}")
        self.data_init = partial(data_init, system)
        estimator = cfg.get("estimators.fubini", estimator_class(f_log_psi=wf.logpsi))
        sampler = make_sampler(cfg, system)
        builder = EvaluationWorkStage.builder(cfg, wf, name="overlap")
        builder.configure_sample_plan(
            estimator.log_overlap_amplitude,
            {"electrons": sampler},
        )
        builder.configure_estimators(fubini=estimator)
        stage = builder.build()
        self.evaluation_stage = HallOverlapWorkStage(
            **{field.name: getattr(stage, field.name) for field in fields(stage)},
            reference_path=self.config.reference_path,
        )


# Keep the public disk configuration name for existing imports.
DiskOverlapConfig = HallOverlapConfig
