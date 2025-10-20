# -*- coding: utf-8 -*-
# core/sparsity/sparse_policy.py

from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class SparsePolicy:
    zero_frac_threshold: float = 0.85
    min_voxels_for_sparse: int = 1_000_000
    allow_dense_fallback: bool = True
    enabled: bool = True  # Optional, default True

    def should_use_sparse(self, cropped_block: np.ndarray, roi_mask_bin: np.ndarray) -> bool:
        """
        Decide whether to use a sparse representation for the given ROI.

        Args:
            cropped_block (np.ndarray): 3D float32 array with NaNs outside ROI.
            roi_mask_bin (np.ndarray): 3D uint8/bool mask (1 inside ROI, 0 outside).

        Returns:
            bool: True if sparse representation should be used.
        """
        total_voxels: int = cropped_block.size
        if total_voxels < self.min_voxels_for_sparse:
            return False

        # Compute fraction of empty voxels: mask==0 or non-finite intensity
        empty_voxels: int = np.count_nonzero(
            np.logical_or(roi_mask_bin == 0, ~np.isfinite(cropped_block))
        )

        empty_fraction: float = float(empty_voxels) / float(total_voxels)
        return empty_fraction >= self.zero_frac_threshold
