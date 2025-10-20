# -*- coding: utf-8 -*-
# data_manager/quantization/lloyd_quantizer.py

from __future__ import annotations

import numpy as np
from typing import Tuple
from sklearn.cluster import MiniBatchKMeans

from .base_quantizer import BaseQuantizer


class LloydQuantizer(BaseQuantizer):
    """
    Lloyd-Max (k-means) quantization on finite ROI voxels.

    - Clusters voxel intensities to `n_levels`.
    - Labels are mapped from 1 to `n_levels`.
    - NaNs are preserved outside the ROI mask.
    """

    def __init__(self, roi: np.ndarray, n_levels: int, random_state: int = 42) -> None:
        """
        Initialize the LloydQuantizer.

        Parameters
        ----------
        roi : np.ndarray
            The region-of-interest intensity array.
        n_levels : int
            Number of quantization levels (clusters).
        random_state : int, optional
            Random seed for k-means initialization (default: 42).
        """
        super().__init__(roi)
        if n_levels < 1:
            raise ValueError("n_levels must be >= 1.")
        self.n_levels = n_levels
        self.random_state = random_state

    def quantize(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform Lloyd-Max (k-means) quantization.

        Returns
        -------
        quantized_roi : np.ndarray
            ROI array with quantized labels (1 to n_levels), NaNs preserved.
        levels : np.ndarray
            Array of quantization levels (1 to n_levels).
        """
        roi_flat = self.roi.ravel()
        valid_mask = np.isfinite(roi_flat)
        valid_values = roi_flat[valid_mask]

        if valid_values.size == 0:
            raise ValueError(
                "ROI has no valid voxels for Lloyd quantization."
            )

        if np.allclose(valid_values, valid_values[0]):
            # All values identical, assign single cluster
            labels = np.zeros_like(valid_values, dtype=np.int32)
        else:
            kmeans = MiniBatchKMeans(
                n_clusters=self.n_levels,
                batch_size=8192,
                n_init=3,
                max_iter=50,
                random_state=self.random_state,
            )
            labels = kmeans.fit_predict(valid_values.reshape(-1, 1)).astype(np.int32)

        quantized_flat = np.full(roi_flat.shape, np.nan, np.float32)
        quantized_flat[valid_mask] = labels.astype(np.float32) + 1.0
        quantized_roi = quantized_flat.reshape(self.roi.shape)

        levels = np.arange(1, self.n_levels + 1, dtype=np.int32)
        return quantized_roi, levels
