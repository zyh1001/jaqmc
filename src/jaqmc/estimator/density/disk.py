# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Disk density estimator for the disk geometry."""

import math

from jax import numpy as jnp

from jaqmc.data import Data
from jaqmc.estimator.histogram import HistogramEstimator
from jaqmc.utils.config import configurable_dataclass


@configurable_dataclass
class DiskDensity(HistogramEstimator):
    r"""Electron density on the disk.

    Accumulates a 2-D histogram of electron positions in Cartesian
    coordinates (x, y) on the disk.

    Args:
        bins: Number of bins for both x and y dimensions.
        range_max: The range of the histogram will be
            [-range_max, range_max] for both dimensions.
    """

    bins: int = 100
    range_max: float = 10.0

    def _histogram_spec(
        self,
    ) -> tuple[int | tuple[int, ...], list[tuple[float, float]]]:
        return (self.bins, self.bins), [
            (-self.range_max, self.range_max),
            (-self.range_max, self.range_max),
        ]

    def extract(self, data: Data) -> jnp.ndarray:
        return data["electrons"][..., :2]
