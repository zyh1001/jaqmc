# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Symmetric-density Fubini--Study estimator for disk wavefunctions."""

from jaqmc.utils.config import configurable_dataclass

from ..fubini import FubiniParams, HallFubini

__all__ = ["DiskFubini", "FubiniParams"]


@configurable_dataclass
class DiskFubini(HallFubini):
    """Evaluate disk overlaps under ``abs(psi * psi_reference)``.

    Args:
        f_log_psi: Complex log-wavefunction evaluator in the disk coordinates.
    """
