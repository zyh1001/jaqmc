# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Symmetric-density Fubini--Study estimator for torus wavefunctions."""

from jaqmc.utils.config import configurable_dataclass

from ..fubini import FubiniParams, HallFubini

__all__ = ["FubiniParams", "TorusFubini"]


@configurable_dataclass
class TorusFubini(HallFubini):
    """Evaluate torus overlaps under ``abs(psi * psi_reference)``.

    Args:
        f_log_psi: Complex log-wavefunction evaluator in the torus coordinates.
    """
