# -*- coding: utf-8 -*-
# core/extractors/intensity_histogram_features_extractor.py

from __future__ import annotations

import logging
import numpy as np
from typing import Any, Dict

from pysera.engine.visera_oop.core.sparsity.view_planner import DataView
from pysera.engine.visera_oop.core.base_feature_extractor import BaseFeatureExtractor

logger = logging.getLogger("Dev_logger")


class IntensityHistogramFeatureExtractor(BaseFeatureExtractor):
    """
    Intensity Histogram features computed *on quantized labels* (1 ... K).
    This mirrors the legacy `getHist` behavior: all stats are based on the
    discrete label histogram produced by the pipeline (FBN/FBS/Lloyd).
    """
    ACCEPTS: DataView = DataView.ROI_VECTOR | DataView.DENSE_BLOCK
    PREFERS: DataView = DataView.ROI_VECTOR
    NAME: str = "IntensityHistogramExtractor"

    # ------------------ inputs ------------------

    def _labels_1_based(self, roi_index: int) -> np.ndarray:
        """
        Return 1-based integer labels for voxels inside ROI.
        If the extractor received a quantized block in `roi`, use it.
        Otherwise, fall back to a provided roi_vector (already masked).
        """
        self.last_perf: Dict[str, Dict[str, Any]] = {"used_view": {}}

        quantised_roi = self.get_roi(roi_index)  # quantized 3D block, NaNs outside ROI

        if quantised_roi is not None:
            finite_voxels = quantised_roi[np.isfinite(quantised_roi)].ravel()
            self.last_perf["used_view"] = {"value": "roi_quant"}

        else:
            roi_vector = self.get_views(roi_index).get("roi_vector")

            if roi_vector is None:
                image_data = self.get_image(roi_index)

                if image_data is None:
                    return np.empty(0, dtype=np.int64)
                finite_voxels = image_data[np.isfinite(image_data)].ravel()
                self.last_perf["used_view"] = {"value": "dense_block"}

            else:
                finite_voxels = roi_vector
                self.last_perf["used_view"] = {"value": "roi_vector"}

        if finite_voxels.size == 0:
            return np.empty(0, dtype=np.int64)

        # Ensure integer 1 ... K labels (round if needed)
        if not np.allclose(finite_voxels, np.round(finite_voxels)):
            finite_voxels = np.round(finite_voxels)

        finite_voxels = finite_voxels.astype(np.int64, copy=False)
        # safety: enforce minimum label 1
        finite_voxels[finite_voxels < 1] = 1

        return finite_voxels

    def _hist_from_labels(self, roi_index: int):
        """
        Build counts, probabilities, and bin labels 1..K from integer labels.
        """
        labels = self._labels_1_based(roi_index)

        if labels.size == 0:
            return (np.zeros(0, dtype=np.int64),
                    np.zeros(0, dtype=np.float64),
                    np.zeros(0, dtype=np.int64))

        max_labels = int(labels.max())
        counts = np.bincount(labels, minlength=max_labels + 1)[1:]  # drop index 0
        total = int(counts.sum())

        if total == 0:

            return (np.zeros(0, dtype=np.int64),
                    np.zeros(0, dtype=np.float64),
                    np.zeros(0, dtype=np.int64))

        probs = counts.astype(np.float64) / float(total)
        bins = np.arange(1, max_labels + 1, dtype=np.int64)

        self.last_perf["num_bins"] = {"value": max_labels}
        self.last_perf["total_voxels"] = total

        return counts, probs, bins

    # ------------------ helpers ------------------

    @staticmethod
    def _cumprob(probabilities: np.ndarray) -> np.ndarray:
        """
        Compute the cumulative probability distribution.
        """
        return np.cumsum(probabilities)

    def _percentile_bin(self, probabilities: np.ndarray, quantile: float) -> int | float:
        """
        Find the first (1-based) bin index where the cumulative probability
        is greater than or equal to the given quantile.
        """
        if probabilities.size == 0:
            return np.nan

        cumulative = self._cumprob(probabilities)
        index = int(np.searchsorted(cumulative, quantile, side="left"))

        return index + 1

    def _entropy(self, probabilities: np.ndarray) -> float:
        """
        Compute Shannon entropy (base-2) of a probability distribution.
        """
        if probabilities.size == 0:
            return np.nan

        small_constant = (
            np.finfo(np.float64).eps
            if self.feature_value_mode == "APPROXIMATE_VALUE"
            else 0.0
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            entropy_terms = probabilities * np.log2(probabilities + small_constant)
            entropy_terms[~np.isfinite(entropy_terms)] = 0.0

        return -float(np.sum(entropy_terms))

    @staticmethod
    def _gradients(counts: np.ndarray) -> np.ndarray:
        """
        Compute the legacy gradient vector of a 1D array.
        """
        num = counts.size

        if num == 0:
            return np.zeros(0, dtype=np.float64)

        if num == 1:
            return np.array([0.0], dtype=np.float64)

        gradients = np.empty(num, dtype=np.float64)
        gradients[0] = counts[1] - counts[0]
        gradients[-1] = counts[-1] - counts[-2]

        if num > 2:
            gradients[1:-1] = (counts[2:] - counts[:-2]) / 2.0

        return gradients

    # ------------------ features ------------------

    def get_ih_mean(self, roi_index: int) -> float:
        """
        Compute the intensity histogram mean for a given ROI.
        """
        _, probabilities, bin_centers = self._hist_from_labels(roi_index)

        if probabilities.size == 0:
            return np.nan

        return float(np.dot(probabilities, bin_centers))

    def get_ih_var(self, roi_index: int) -> float:
        """
        Compute the intensity histogram variance for a given ROI.
        """
        _, probabilities, bin_centers = self._hist_from_labels(roi_index)

        if probabilities.size == 0:
            return np.nan

        mean_value = self._get_or_compute_feature("ih_mean", roi_index)
        variance = np.dot(probabilities, (bin_centers - mean_value) ** 2)

        return float(variance)

    def get_ih_skew(self, roi_index: int) -> float:
        """
        Compute the intensity histogram skewness for a given ROI.
        """
        _, probabilities, bin_centers = self._hist_from_labels(roi_index)

        if probabilities.size == 0:
            return np.nan

        mean_value = self._get_or_compute_feature("ih_mean", roi_index)
        variance = self._get_or_compute_feature("ih_var", roi_index)

        if variance == 0.0:
            return 0.0

        third_moment = float(np.dot(probabilities, (bin_centers - mean_value) ** 3))
        skewness = third_moment / (variance ** 1.5)

        return float(skewness)

    def get_ih_kurt(self, roi_index: int) -> float:
        """
        Compute the intensity histogram kurtosis for a given ROI.
        """
        _, probabilities, bin_centers = self._hist_from_labels(roi_index)

        if probabilities.size == 0:
            return np.nan

        mean_value = self._get_or_compute_feature("ih_mean", roi_index)
        variance = self._get_or_compute_feature("ih_var", roi_index)

        if variance == 0.0:
            return 0.0

        fourth_moment = float(np.dot(probabilities, (bin_centers - mean_value) ** 4))
        kurtosis = fourth_moment / (variance ** 2) - 3.0  # excess kurtosis

        return float(kurtosis)

    def get_ih_median(self, roi_index: int) -> float:
        """
        Compute the intensity histogram median for a given ROI.
        """
        _, probabilities, _ = self._hist_from_labels(roi_index)

        if probabilities.size == 0:
            return np.nan

        median_bin = self._percentile_bin(probabilities, 0.5)

        return float(median_bin)

    def get_ih_min(self, roi_index: int) -> float:
        """
        Compute the minimum intensity bin for a given ROI based on the histogram.
        """
        counts, _, bin_centers = self._hist_from_labels(roi_index)

        if counts.size == 0:
            return np.nan

        non_zero_indices = np.flatnonzero(counts)

        if non_zero_indices.size == 0:
            return np.nan

        min_value = bin_centers[non_zero_indices[0]]

        return float(min_value)

    def get_ih_p10(self, roi_index: int) -> float:
        """
        Compute the 10th percentile intensity bin for a given ROI.
        """
        _, probabilities, _ = self._hist_from_labels(roi_index)

        if probabilities.size == 0:
            return np.nan

        percentile_10_bin = self._percentile_bin(probabilities, 0.1)

        return float(percentile_10_bin)

    def get_ih_p90(self, roi_index: int) -> float:
        """
        Compute the 90th percentile intensity bin for a given ROI.
        """
        _, probabilities, _ = self._hist_from_labels(roi_index)

        if probabilities.size == 0:
            return np.nan

        percentile_90_bin = self._percentile_bin(probabilities, 0.9)

        return float(percentile_90_bin)

    def get_ih_max(self, roi_index: int) -> float:
        """
        Compute the maximum intensity bin for a given ROI based on the histogram.
        """
        counts, _, bin_centers = self._hist_from_labels(roi_index)

        if bin_centers.size == 0:
            return np.nan

        max_value = bin_centers[-1]

        return float(max_value)

    def get_ih_mode(self, roi_index: int) -> float:
        """
        Compute the mode (most frequent intensity) of the histogram for a given ROI.
        """
        counts, probabilities, bin_centers = self._hist_from_labels(roi_index)

        if counts.size == 0:
            return np.nan

        mean_value = self._get_or_compute_feature("ih_mean", roi_index)
        max_indices = np.flatnonzero(counts == counts.max())

        if max_indices.size == 1:
            return float(bin_centers[max_indices[0]])

        # Multiple bins tied for max; choose the one closest to the mean
        tied_bins = bin_centers[max_indices]
        closest_to_mean = tied_bins[np.argmin(np.abs(tied_bins - mean_value))]

        return float(closest_to_mean)

    def get_ih_iqr(self, roi_index: int) -> float:
        """
        Compute the interquartile range (IQR) of the intensity histogram for a given ROI.
        """
        _, probabilities, _ = self._hist_from_labels(roi_index)

        if probabilities.size == 0:
            return np.nan

        percentile_25_bin = int(self._percentile_bin(probabilities, 0.25))
        percentile_75_bin = int(self._percentile_bin(probabilities, 0.75))
        interquartile = percentile_75_bin - percentile_25_bin

        return float(interquartile)

    def get_ih_range(self, roi_index: int) -> float:
        """
        Compute the range of the intensity histogram for a given ROI.
        """
        max_value = self._get_or_compute_feature("ih_max", roi_index)
        min_value = self._get_or_compute_feature("ih_min", roi_index)

        if np.isnan(max_value) or np.isnan(min_value):
            return np.nan

        return float(max_value - min_value)

    def get_ih_mad(self, roi_index: int) -> float:
        """
        Compute the mean absolute deviation (MAD) of the intensity histogram for a given ROI.
        """
        _, probabilities, bin_centers = self._hist_from_labels(roi_index)

        if probabilities.size == 0:
            return np.nan

        mean_value = self._get_or_compute_feature("ih_mean", roi_index)
        mad = np.dot(probabilities, np.abs(bin_centers - mean_value))

        return float(mad)

    def get_ih_rmad(self, roi_index: int) -> float:
        """
        Compute the robust mean absolute deviation (RMAD) of the intensity histogram for a given ROI.
        """
        _, probabilities, bin_centers = self._hist_from_labels(roi_index)

        if probabilities.size == 0:
            return np.nan

        percentile_10 = int(self._get_or_compute_feature("ih_p10", roi_index))
        percentile_90 = int(self._get_or_compute_feature("ih_p90", roi_index))

        # Validate percentile indices
        if percentile_10 < 1 or percentile_90 <= percentile_10 or percentile_90 > bin_centers.size:
            if self.feature_value_mode == 'REAL_VALUE':
                return np.nan
            else:
                return float(np.finfo(np.float64).eps)

        # Select the range between percentile 10 and percentile 90 (1-based indexing)
        sel_probs = probabilities[percentile_10 - 1:percentile_90]
        sel_bins = bin_centers[percentile_10 - 1:percentile_90]

        weight_sum = float(sel_probs.sum())

        if weight_sum == 0.0:
            if self.feature_value_mode == 'APPROXIMATE_VALUE':
                weight_sum += np.finfo(np.float64).eps
            else:
                return np.nan

        mean_selected = float(np.dot(sel_probs, sel_bins) / weight_sum)
        rmad = float(np.dot(sel_probs, np.abs(sel_bins - mean_selected)) / weight_sum)

        return rmad

    def get_ih_medad(self, roi_index: int) -> float:
        """
        Compute the median absolute deviation (MEDAD) of the intensity histogram for a given ROI.
        """
        _, probabilities, bin_centers = self._hist_from_labels(roi_index)

        if probabilities.size == 0:
            return np.nan

        median_value = float(self._get_or_compute_feature("ih_median", roi_index))
        medad = np.dot(probabilities, np.abs(bin_centers - median_value))

        return float(medad)

    def get_ih_cov(self, roi_index: int) -> float:
        """
        Compute the coefficient of variation (CoV) of the intensity histogram for a given ROI.
        """
        mean_value = self._get_or_compute_feature("ih_mean", roi_index)
        variance = self._get_or_compute_feature("ih_var", roi_index)

        if mean_value == 0.0 or variance == 0.0:
            return 0.0

        cov = np.sqrt(variance) / mean_value
        return float(cov)

    def get_ih_qcod(self, roi_index: int) -> float:
        """
        Compute the quartile coefficient of dispersion (QCoD) for the intensity histogram of a given ROI.
        """
        _, probabilities, _ = self._hist_from_labels(roi_index)

        if probabilities.size == 0:
            return np.nan

        percentile_25_bin = int(self._percentile_bin(probabilities, 0.25))
        percentile_75_bin = int(self._percentile_bin(probabilities, 0.75))

        sum_q = percentile_25_bin + percentile_75_bin

        if sum_q == 0:
            return 1e6  # legacy safeguard for division by zero

        qcod = (percentile_75_bin - percentile_25_bin) / sum_q

        return float(qcod)

    def get_ih_entropy(self, roi_index: int) -> float:
        """
        Compute the entropy of the intensity histogram for a given ROI.
        """
        _, probabilities, _ = self._hist_from_labels(roi_index)

        if probabilities.size == 0:
            return np.nan

        return float(self._entropy(probabilities))

    def get_ih_uniformity(self, roi_index: int) -> float:
        """
        Compute the uniformity (energy) of the intensity histogram for a given ROI.
        """
        _, probabilities, _ = self._hist_from_labels(roi_index)

        if probabilities.size == 0:
            return np.nan

        uniformity = np.sum(probabilities ** 2)

        return float(uniformity)

    def get_ih_max_grad(self, roi_index: int) -> float:
        """
        Compute the maximum gradient of the intensity histogram for a given ROI.
        """
        counts, _, _ = self._hist_from_labels(roi_index)

        if counts.size == 0:
            return np.nan

        gradients = self._gradients(counts.astype(np.float64))
        max_grad = np.max(gradients)

        return float(max_grad)

    def get_ih_max_grad_g(self, roi_index: int) -> float:
        """
        Compute the bin index of the maximum gradient in the intensity histogram for a given ROI.
        """
        counts, _, _ = self._hist_from_labels(roi_index)

        if counts.size == 0:
            return np.nan

        gradients = self._gradients(counts.astype(np.float64))
        max_grad_index = np.argmax(gradients) + 1  # 1-based index

        return float(max_grad_index)

    def get_ih_min_grad(self, roi_index: int) -> float:
        """
        Compute the minimum gradient of the intensity histogram for a given ROI.
        """
        counts, _, _ = self._hist_from_labels(roi_index)

        if counts.size == 0:
            return np.nan

        gradients = self._gradients(counts.astype(np.float64))
        min_grad = np.min(gradients)

        return float(min_grad)

    def get_ih_min_grad_g(self, roi_index: int) -> float:
        """
        Compute the bin index of the minimum gradient in the intensity histogram for a given ROI.
        """
        counts, _, _ = self._hist_from_labels(roi_index)

        if counts.size == 0:
            return np.nan

        gradients = self._gradients(counts.astype(np.float64))
        min_grad_index = np.argmin(gradients) + 1  # 1-based index

        return float(min_grad_index)
