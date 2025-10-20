# -*- coding: utf-8 -*-
# core/extractors/statistics_features_extractor.py

from __future__ import annotations

import logging
import numpy as np
from typing import Any, Dict, Optional

from pysera.engine.visera_oop.core.sparsity.view_planner import DataView
from pysera.engine.visera_oop.core.base_feature_extractor import BaseFeatureExtractor

logger = logging.getLogger("Dev_logger")


class StatisticsFeaturesExtractor(BaseFeatureExtractor):
    """
    first-order statistics on ROI intensities.
    Implements the same set of 18 features.
    """

    ACCEPTS: DataView = DataView.ROI_VECTOR
    PREFERS: DataView = DataView.ROI_VECTOR
    NAME: str = "StatisticsExtractor"

    # ---- Vector accessor ----
    def _get_roi_vector(self, roi_index: int) -> np.ndarray:
        """
        Return ROI vector containing finite voxel intensities only.
        """
        self.last_perf: Dict[str, Dict[str, Any]] = {"used_view": {}}

        roi_vector: Optional[np.ndarray] = self.get_views(roi_index).get("dense_block")

        if roi_vector is not None:
            self.last_perf["used_view"] = {"value": "ROI_VECTOR"}
            return roi_vector.astype(np.float32, copy=False)

        image_data: Optional[np.ndarray] = self.get_image(roi_index)
        self.last_perf["used_view"] = {"value": "DENSE_BLOCK"}

        if image_data is None:
            return np.empty((0,), dtype=np.float32)

        finite_voxels: np.ndarray = image_data[np.isfinite(image_data)]
        return finite_voxels.astype(np.float32, copy=False)

    # ---- Legacy-compatible features ----
    def get_stat_mean(self, roi_index: int) -> float:
        roi_vector: np.ndarray = self._get_roi_vector(roi_index)
        return float(np.mean(roi_vector)) if roi_vector.size else np.nan

    def get_stat_var(self, roi_index: int) -> float:
        roi_vector: np.ndarray = self._get_roi_vector(roi_index)
        return float(np.var(roi_vector)) if roi_vector.size else np.nan

    def get_stat_skew(self, roi_index: int) -> float:
        roi_vector: np.ndarray = self._get_roi_vector(roi_index)

        if roi_vector.size == 0:
            return np.nan

        mean_value: float = self._get_or_compute_feature("stat_mean", roi_index)
        standard_deviation: float = float(np.std(roi_vector, ddof=0))

        if standard_deviation == 0:
            return 0.0

        return float(np.mean(np.power((roi_vector - mean_value) / standard_deviation, 3)))

    def get_stat_kurt(self, roi_index: int) -> float:
        roi_vector: np.ndarray = self._get_roi_vector(roi_index)

        if roi_vector.size == 0:
            return np.nan

        mean_value: float = self._get_or_compute_feature("stat_mean", roi_index)
        standard_deviation: float = float(np.std(roi_vector, ddof=0))

        if standard_deviation == 0:
            return 0.0

        return float(np.subtract(np.mean(np.power((roi_vector - mean_value) / standard_deviation, 4)), 3))

    def get_stat_median(self, roi_index: int) -> float:
        roi_vector: np.ndarray = self._get_roi_vector(roi_index)
        return float(np.median(roi_vector)) if roi_vector.size else np.nan

    def get_stat_min(self, roi_index: int) -> float:
        roi_vector: np.ndarray = self._get_roi_vector(roi_index)
        return float(np.min(roi_vector)) if roi_vector.size else np.nan

    def get_stat_p10(self, roi_index: int) -> float:
        roi_vector: np.ndarray = self._get_roi_vector(roi_index)
        return float(np.percentile(roi_vector, 10)) if roi_vector.size else np.nan

    def get_stat_p90(self, roi_index: int) -> float:
        roi_vector: np.ndarray = self._get_roi_vector(roi_index)
        return float(np.percentile(roi_vector, 90)) if roi_vector.size else np.nan

    def get_stat_max(self, roi_index: int) -> float:
        roi_vector: np.ndarray = self._get_roi_vector(roi_index)
        return float(np.max(roi_vector)) if roi_vector.size else np.nan

    def get_stat_iqr(self, roi_index: int) -> float:
        roi_vector: np.ndarray = self._get_roi_vector(roi_index)

        if roi_vector.size == 0:
            return np.nan

        percentile_75: float = float(np.percentile(roi_vector, 75))
        percentile_25: float = float(np.percentile(roi_vector, 25))

        return float(np.subtract(percentile_75, percentile_25))

    def get_stat_range(self, roi_index: int) -> float:
        maximum_value: float = self._get_or_compute_feature("stat_max", roi_index)
        minimum_value: float = self._get_or_compute_feature("stat_min", roi_index)

        if np.isnan(maximum_value) or np.isnan(minimum_value):
            return np.nan

        return float(np.subtract(maximum_value, minimum_value))

    def get_stat_mad(self, roi_index: int) -> float:
        roi_vector: np.ndarray = self._get_roi_vector(roi_index)

        if roi_vector.size == 0:
            return np.nan

        mean_value: float = self._get_or_compute_feature("stat_mean", roi_index)

        return float(np.mean(np.absolute(np.subtract(roi_vector, mean_value))))

    def get_stat_rmad(self, roi_index: int) -> float:
        roi_vector: np.ndarray = self._get_roi_vector(roi_index)

        if roi_vector.size == 0:
            return np.nan

        percentile_10: float = self._get_or_compute_feature("stat_p10", roi_index)
        percentile_90: float = self._get_or_compute_feature("stat_p90", roi_index)

        filtered_roi: np.ndarray = roi_vector[(roi_vector >= percentile_10) & (roi_vector <= percentile_90)]

        if filtered_roi.size == 0:
            return np.nan

        mean_filtered: float = float(np.mean(filtered_roi))

        return float(np.mean(np.absolute(np.subtract(filtered_roi, mean_filtered))))

    def get_stat_medad(self, roi_index: int) -> float:
        roi_vector: np.ndarray = self._get_roi_vector(roi_index)

        if roi_vector.size == 0:
            return np.nan

        median_value: float = self._get_or_compute_feature("stat_median", roi_index)

        return float(np.mean(np.absolute(np.subtract(roi_vector, median_value))))

    def get_stat_cov(self, roi_index: int) -> float:
        roi_vector: np.ndarray = self._get_roi_vector(roi_index)

        if roi_vector.size == 0:
            return np.nan

        variance_value: float = self._get_or_compute_feature("stat_var", roi_index)
        mean_value: float = self._get_or_compute_feature("stat_mean", roi_index)

        if variance_value == 0.0 or mean_value == 0.0:
            return 0.0

        return float(np.divide(np.sqrt(variance_value), mean_value))

    def get_stat_qcod(self, roi_index: int) -> float:
        roi_vector: np.ndarray = self._get_roi_vector(roi_index)

        if roi_vector.size == 0:
            return np.nan

        percentile_75: int = int(np.percentile(roi_vector, 75))
        percentile_25: int = int(np.percentile(roi_vector, 25))
        sum_percentiles: float = float(np.add(percentile_75, percentile_25))

        if sum_percentiles == 0.0:
            return float(np.finfo(np.float64).eps)

        return float(np.divide(np.subtract(percentile_75, percentile_25), sum_percentiles))

    def get_stat_energy(self, roi_index: int) -> float:
        roi_vector: np.ndarray = self._get_roi_vector(roi_index)
        return float(np.sum(np.power(roi_vector, 2))) if roi_vector.size else np.nan

    def get_stat_rms(self, roi_index: int) -> float:
        roi_vector: np.ndarray = self._get_roi_vector(roi_index)
        return float(np.sqrt(np.mean(np.power(roi_vector, 2)))) if roi_vector.size else np.nan
