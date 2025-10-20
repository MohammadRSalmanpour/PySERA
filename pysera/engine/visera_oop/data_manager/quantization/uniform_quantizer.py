# -*- coding: utf-8 -*-
# data_manager/quantization/uniform_quantizer.py

from __future__ import annotations

import numpy as np
from typing import Tuple

from .base_quantizer import BaseQuantizer


class UniformQuantizer(BaseQuantizer):
    """
    Fixed-Bin-Number (FBN) quantization according to IBSI.

    Formula:
        bin = floor(N * (x - xmin) / (xmax - xmin)) + 1
        with levels 1 ... N

    Special cases:
        - If xmax == xmin, all valid voxels are assigned level 1.
        - NaNs (outside the intensity mask) are preserved.
    """

    def __init__(self, roi: np.ndarray, n_levels: int) -> None:
        super().__init__(roi)
        if int(n_levels) < 1:
            raise ValueError("n_levels must be >= 1.")
        self.n_levels: int = int(n_levels)

    def quantize(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply fixed-bin-number quantization to the ROI.

        Returns
        -------
        quantized_roi : np.ndarray
            Quantized ROI with the same shape as input.
        levels : np.ndarray
            Array of possible quantization levels [1 ... n_levels].
        """
        roi_data = self.roi
        quantized_roi = np.full(roi_data.shape, np.nan, dtype=np.float32)

        valid_mask = np.isfinite(roi_data)
        if not np.any(valid_mask):
            raise ValueError("ROI has no valid voxels for FBN quantization.")

        intensity_min = float(np.nanmin(roi_data))
        intensity_max = float(np.nanmax(roi_data))

        if np.isclose(intensity_max, intensity_min):
            quantized_roi[valid_mask] = 1.0
            return quantized_roi, np.arange(1, 2, dtype=np.int32)

        intensity_range = intensity_max - intensity_min
        bin_levels = (
            np.floor(
                self.n_levels * (roi_data[valid_mask] - intensity_min) / intensity_range
            )
            + 1.0
        )
        bin_levels = np.clip(bin_levels, 1.0, float(self.n_levels))

        quantized_roi[valid_mask] = bin_levels.astype(np.float32)
        levels = np.arange(1, self.n_levels + 1, dtype=np.int32)

        return quantized_roi, levels
