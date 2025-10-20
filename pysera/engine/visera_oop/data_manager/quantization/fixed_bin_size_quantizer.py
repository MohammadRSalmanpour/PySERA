# -*- coding: utf-8 -*-
# data_manager/quantization/fixed_bin_size_quantizer.py

from __future__ import annotations

import numpy as np
from typing import Tuple

from .base_quantizer import BaseQuantizer


class FixedBinSizeQuantizer(BaseQuantizer):
    """
    Fixed-Bin-Size (FBS) quantization according to IBSI.

    Formula:
        bin = floor((x - min_gl) / Δ) + 1
        where Δ > 0 and levels are contiguous (1 ... K)

    Notes
    -----
    - min_gl is the anchor (prefer lower bound of re-segmentation range
      for reproducibility).
    - NaNs (outside the intensity mask) are preserved.
    """

    def __init__(self, roi: np.ndarray, bin_size: float, min_gl: float) -> None:
        super().__init__(roi)
        if float(bin_size) <= 0.0:
            raise ValueError("bin_size must be > 0.")
        self.bin_size: float = float(bin_size)
        self.min_gl: float = float(min_gl)

    def quantize(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply fixed-bin-size quantization to the ROI.

        Returns
        -------
        quantized_roi : np.ndarray
            Quantized ROI with the same shape as input.
        levels : np.ndarray
            Array of possible quantization levels [1 ... K].
        """
        roi_data = self.roi
        quantized_roi = np.full(roi_data.shape, np.nan, dtype=np.float32)

        valid_mask = np.isfinite(roi_data)
        if not np.any(valid_mask):
            raise ValueError("ROI has no valid voxels for FBS quantization.")

        bin_levels = np.floor((roi_data[valid_mask] - self.min_gl) / self.bin_size) + 1.0
        bin_levels = np.maximum(bin_levels, 1.0)

        quantized_roi[valid_mask] = bin_levels.astype(np.float32)
        max_level = int(np.nanmax(quantized_roi))
        levels = np.arange(1, max_level + 1, dtype=np.int32)

        return quantized_roi, levels
