# -*- coding: utf-8 -*-
# core/extractors/intensity_volume_histogram_features_extractor.py

from __future__ import annotations

import logging
import numpy as np
from typing import Any, Dict, Optional, Tuple

from pysera.engine.visera_oop.core.sparsity.view_planner import DataView
from pysera.engine.visera_oop.core.base_feature_extractor import BaseFeatureExtractor

logger = logging.getLogger("Dev_logger")


def asarray_compat(
    array_like: Any,
    dtype: Optional[np.dtype] = None,
    copy: bool = False
) -> np.ndarray:
    """
    Safely convert input to a NumPy array, with optional dtype and copy.
    Falls back gracefully if 'copy' is not supported.

    Args:
        array_like (Any): Input data to convert.
        dtype (Optional[np.dtype]): Desired NumPy data type.
        copy (bool): Whether to copy data.

    Returns:
        np.ndarray: Converted NumPy array.
    """
    try:
        return np.asarray(array_like, dtype=dtype, copy=copy)
    except TypeError:
        return np.asarray(array_like, dtype=dtype)


class IntensityVolumeHistogramExtractor(BaseFeatureExtractor):
    """
    Intensity Volume Histogram (IVH) features on the ROI 1D vector.
    Calculates fractional volumes and intensities.
    """

    ACCEPTS = DataView.ROI_VECTOR
    PREFERS = DataView.ROI_VECTOR
    NAME = "IntensityVolumeHistogramExtractor"

    def _init_subclass(
            self,
            bin_width: Optional[float] = None,
            apply_resegmentation: bool = False,
            resegmentation_interval: Optional[Tuple[float, float]] = None,
            **_,
    ) -> None:
        # Bin width (default 1.0 if not provided)
        self._bin_width: float = float(bin_width) if bin_width is not None else 1.0

        # Resegmentation flags
        self._enable_resegmentation: bool = bool(apply_resegmentation)
        self._resegmentation_interval: Optional[Tuple[float, float]] = (
            tuple(map(float, resegmentation_interval))
            if resegmentation_interval is not None and len(resegmentation_interval) == 2
            else None
        )

        # IVH (Intensity-Volume Histogram) cache per bin
        self._ivh_cache: Dict[int, Dict[str, np.ndarray]] = {}

        # Performance tracking
        self.last_perf: Dict[str, Any] = {}

    # -------- Input vector preparation --------
    def _get_roi_vector(self, roi_index: int) -> np.ndarray:
        """
        Returns the 1D vector of voxel intensities within the ROI for a given index.

        Args:
            roi_index (int): Index of the ROI.

        Returns:
            np.ndarray: 1D array of finite voxel values inside the ROI.
        """
        roi_vector: Optional[np.ndarray] = self.get_views(roi_index).get("roi_vector")

        if roi_vector is not None:
            return asarray_compat(roi_vector, dtype=np.float32, copy=False)

        image_block: Optional[np.ndarray] = self.get_image(roi_index)

        if image_block is None:
            return np.empty((0,), dtype=np.float32)

        flattened: np.ndarray = asarray_compat(image_block, dtype=np.float32, copy=False)
        finite_voxels: np.ndarray = flattened[np.isfinite(flattened)]

        return finite_voxels

    # -------- IVH computation --------
    def _prepare_ivh(self, roi_index: int) -> None:
        """
        Prepare intensity-volume histogram (IVH) for a given ROI index.

        Args:
            roi_index (int): ROI index to process.
        """
        if roi_index in self._ivh_cache:
            return

        roi_values: np.ndarray = self._get_roi_vector(roi_index)
        if roi_values.size == 0 or not np.isfinite(roi_values).any():
            self._ivh_cache[roi_index] = dict(
                intensity=np.array([0.0], dtype=np.float64),
                frac_volume=np.array([1.0], dtype=np.float64),
                bins=np.array([0.0], dtype=np.float64)
            )
            return

        if self._enable_resegmentation and self._resegmentation_interval:
            lower, upper = self._resegmentation_interval
            mask_in_range: np.ndarray = np.logical_and(roi_values >= lower, roi_values <= upper)
            roi_values = roi_values[mask_in_range]
            self.last_perf["enable_resegmentation"] = True
            self.last_perf["resegmentation_interval"] = np.array([float(lower), float(upper)], dtype=np.float64)
        else:
            self.last_perf["_enable_resegmentation"] = False

        roi_values = roi_values[np.isfinite(roi_values)]

        min_intensity: float = np.nanmin(roi_values)
        max_intensity: float = np.nanmax(roi_values)
        bin_width: float = self._bin_width

        intensity_bins: np.ndarray = np.arange(
            start=np.floor(min_intensity),
            stop=np.add(np.ceil(max_intensity), bin_width),
            step=bin_width,
            dtype=np.float64
        )

        # Small constant depending on feature_value_mode
        small_constant: float = np.finfo(np.float64).eps if self.feature_value_mode == "APPROXIMATE_VALUE" else 0.0

        # Fractional intensity [0,1]
        bins_max: float = np.nanmax(intensity_bins).item()  # convert to Python float
        bins_min: float = np.nanmin(intensity_bins).item()
        intensity_range: float = max(np.subtract(bins_max, bins_min), small_constant)

        frac_intensity: np.ndarray = np.divide(np.subtract(intensity_bins, np.nanmin(intensity_bins)), intensity_range)

        # Fractional volume (1 - fraction of voxels below intensity)
        num_voxels: int = roi_values.size
        frac_volume: np.ndarray = np.array([
            np.subtract(1.0, np.divide(np.count_nonzero(roi_values < intensity_val), num_voxels))
            for intensity_val in intensity_bins
        ], dtype=np.float64)

        self._ivh_cache[roi_index] = dict(
            intensity=frac_intensity,
            frac_volume=frac_volume,
            bins=intensity_bins
        )

    # -------- Queries --------
    def _volume_fraction_at_normalized_intensity(self, roi_index: int, norm_intensity: float) -> float:
        """
        Returns the fractional volume at a given normalized intensity for a specific ROI.

        Args:
            roi_index (int): Index of the ROI.
            norm_intensity (float): Normalized intensity in [0, 1].

        Returns:
            float: Fractional volume corresponding to the intensity.
        """
        self._prepare_ivh(roi_index)
        fractional_volume: np.ndarray = self._ivh_cache[roi_index]["frac_volume"]
        normalized_bins: np.ndarray = self._ivh_cache[roi_index]["intensity"]

        # Find insertion index in the normalized intensity array
        insert_index: int = int(np.searchsorted(normalized_bins, norm_intensity, side="left"))
        safe_index: int = min(insert_index, int(np.subtract(len(fractional_volume), 1)))

        return float(fractional_volume[safe_index])

    def _intensity_at_volume_fraction(self, roi_index: int, target_vf: float) -> float:
        """
        Returns the intensity corresponding to a given fractional volume (VF) for a specific ROI.

        Args:
            roi_index (int): Index of the ROI.
            target_vf (float): Desired fractional volume.

        Returns:
            float: Intensity value at the specified fractional volume.
        """
        self._prepare_ivh(roi_index)
        fractional_volume: np.ndarray = self._ivh_cache[roi_index]["frac_volume"]
        intensity_bins: np.ndarray = self._ivh_cache[roi_index]["bins"]

        matching_indices: np.ndarray = np.squeeze(np.where(np.less_equal(fractional_volume, target_vf)))

        if matching_indices.size == 0:
            if self.feature_value_mode == "APPROXIMATE_VALUE":
                fractional_volume_synth = np.append(fractional_volume, np.finfo(np.float64).eps)
                matching_indices: np.ndarray = np.where(np.less_equal(fractional_volume_synth, target_vf))[0]
                intensity_bins: np.ndarray = np.append(intensity_bins, int(intensity_bins[-1]+1))
            else:       # REAL_VALUE
                logger.warning(f"Small ROI causing empty fractional volume.")
                return np.nan

        return float(intensity_bins[int(np.atleast_1d(matching_indices)[0])])

    # -------- Feature getters --------
    def get_ivh_v10(self, roi_index: int) -> float:
        return self._volume_fraction_at_normalized_intensity(roi_index, 0.10)

    def get_ivh_v90(self, roi_index: int) -> float:
        return self._volume_fraction_at_normalized_intensity(roi_index, 0.90)

    def get_ivh_i10(self, roi_index: int) -> float:
        return self._intensity_at_volume_fraction(roi_index, 0.10)

    def get_ivh_i90(self, roi_index: int) -> float:
        return self._intensity_at_volume_fraction(roi_index, 0.90)

    def get_ivh_diff_v10_v90(self, roi_index: int) -> float:

        ivh_v10_value: float = self._get_or_compute_feature("ivh_v10", roi_index)
        ivh_v90_value: float = self._get_or_compute_feature("ivh_v90", roi_index)

        difference: float = np.subtract(ivh_v10_value, ivh_v90_value)

        return difference

    def get_ivh_diff_i10_i90(self, roi_index: int) -> float:
        ivh_i10_value: float = self._get_or_compute_feature("ivh_i10", roi_index)
        ivh_i90_value: float = self._get_or_compute_feature("ivh_i90", roi_index)

        difference: float = np.subtract(ivh_i10_value, ivh_i90_value)

        return difference

    def get_ivh_auc(self, roi_index: int) -> float:
        self._prepare_ivh(roi_index)
        frac_volume: np.ndarray = self._ivh_cache[roi_index]["frac_volume"]
        intensity_values: np.ndarray = self._ivh_cache[roi_index]["intensity"]

        if intensity_values.size <= 1:
            return 0.0

        try:
            ivh_auc_value: float = float(np.trapz(y=frac_volume,
                                                  x=intensity_values))
            return ivh_auc_value

        except Exception as exc:
            logger.warning("IVH AUC computation failed: %s", exc)
            return 0.0
