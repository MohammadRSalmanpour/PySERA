# -*- coding: utf-8 -*-
# core/extractors/moment_invariants_features_extractor.py

from __future__ import annotations

import logging
import numpy as np
import concurrent.futures
from typing import Any, Dict, Optional, Tuple

from pysera.engine.visera_oop.core.sparsity.view_planner import DataView
from pysera.engine.visera_oop.core.base_feature_extractor import BaseFeatureExtractor

logger = logging.getLogger("Dev_logger")


def handle_math_operations(feature_vector, feature_value_mode, mode='divide', epsilon=1e-30):  # local replacement
    """Safeguard basic math operations for REAL/APPROXIMATE modes.

    If `feature_value_mode` is 'APPROXIMATE_VALUE', replaces illegal values with a small epsilon
    to avoid division-by-zero or sqrt-of-negative issues. Works with NumPy arrays or scalar-like
    inputs (they are treated as 0-D arrays).
    """
    if feature_value_mode == 'REAL_VALUE':
        return feature_vector

    feat_vect = np.asarray(feature_vector)

    if mode == 'divide':
        mask = (feat_vect == 0.0)
    elif mode == 'sqrt':
        mask = (feat_vect < 0.0)
    elif mode == 'both':
        mask = (feat_vect <= 0.0)
    else:
        mask = np.zeros_like(feat_vect, dtype=bool)

    if np.any(mask):
        feat_vect = feat_vect.copy()
        feat_vect[mask] = epsilon
        logger.warning(f"Using epsilon = {epsilon} to prevent mathematical errors.")
    return feat_vect if feat_vect.shape != () else feat_vect.item()


class MomentInvariantsFeaturesExtractor(BaseFeatureExtractor):
    """
    3D moment invariants (up to 3rd order) computed on the ROI intensity field and ROI shape.

    This extractor provides two invariant sets per ROI:
      - *Intensity*-based: computed from the normalized intensity field on the ROI support
      - *Shape*-based: computed from the binary ROI mask (within the same bounding box)

    Each set returns five invariants (IBSI-inspired tensor invariants of 2nd and 3rd order
    central moments):
        [i1, f2, i2, i3, f3]

    Where
      - invariant_i1 : trace of the 2nd-order central-moment tensor
      - invariant_f2 : Frobenius norm squared of the 2nd-order central-moment tensor
      - invariant_i2 : second principal invariant of the 2nd-order tensor
      - invariant_i3 : third principal invariant (determinant) of the 2nd-order tensor
      - invariant_f3 : Frobenius norm squared of the 3rd-order central-moment tensor (properly weighted)

    The full 10-vector is exposed via :meth:`get_moment_invariants_vector` as
        [i1,f2,i2,i3,f3,  si1,sf2,si2,si3,sf3]
    for *(intensity, shape)* respectively.
    """

    ACCEPTS: DataView = DataView.DENSE_BLOCK
    PREFERS: DataView = DataView.DENSE_BLOCK
    NAME: str = "MomentInvariantExtractor"

    # ----------------------------
    # Lifecycle / construction
    # ----------------------------
    def __init__(self, *, cache: Optional[Any] = None, feature_value_mode: str = "REAL_VALUE") -> None:
        super().__init__()
        self.cache = cache
        self.feature_value_mode = feature_value_mode

        # Internal per-ROI cache
        # structure: { roi_index: { 'intensity': np.ndarray(5,), 'shape': np.ndarray(5,), 'vector': np.ndarray(10,) } }
        self._mom_cache: Dict[int, Dict[str, Any]] = {}

        # Perf bookkeeping
        self.last_perf: Dict[str, Dict[str, Any]] = {"used_view": {}}
        self.last_feature_perf: Dict[str, Dict[str, Any]] = {}

    # ----------------------------
    # Utilities
    # ----------------------------
    def _fallback_value(self) -> float:
        """Return NaN or 0.0 depending on feature_value_mode for degenerate cases."""
        return float("nan") if self.feature_value_mode == "REAL_VALUE" else 0.0

    # ----------------------------
    # Core accessors (mask/image)
    # ----------------------------
    def _get_bbox_and_mask(self, roi_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        bbox : float64 ndarray (X,Y,Z)
            Intensity block in the ROI bounding box, background must be NaN.
        mask : bool ndarray (X,Y,Z)
            Binary ROI mask for the same bounding box (True inside ROI).
        """
        views = self.get_views(roi_index)
        bbox = views.get("dense_block")
        mask = views.get("binary_mask")

        if bbox is None:
            # Degenerate – return minimal arrays to keep downstream robust
            return np.full((0,), np.nan, dtype=np.float64), np.zeros((0,), dtype=bool)

        bbox = np.asarray(bbox, dtype=np.float64, order="C")

        if mask is None:
            # Derive mask from finite intensity
            mask_bool = np.isfinite(bbox)
        else:
            m = np.asarray(mask)
            if m.dtype != bool:
                # treat partial-volume as ROI >= 0.5 by default (IBSI recommendation)
                mask_bool = m >= 0.5
            else:
                mask_bool = m

        # Ensure background is NaN in bbox
        if np.any(~mask_bool):
            bbox = bbox.copy()
            bbox[~mask_bool] = np.nan

        return bbox, mask_bool

    # ----------------------------
    # Public ensure
    # ----------------------------
    def _ensure_moments(self, roi_index: int) -> None:
        """Compute and cache 3D intensity and shape moments for a given ROI."""

        if roi_index in self._mom_cache:
            return

        bbox, mask = self._get_bbox_and_mask(roi_index)
        if bbox.size == 0 or not np.any(mask):
            fallback = np.array([self._fallback_value()] * 5, dtype=np.float64)
            self._mom_cache[roi_index] = {
                "intensity": fallback,
                "shape": fallback,
                "vector": np.concatenate((fallback, fallback)),
            }
            return

        # Dimensions of the ROI
        dim_x, dim_y, dim_z = bbox.shape
        x_coords = np.arange(1, dim_x + 1, dtype=np.float64)
        y_coords = np.arange(1, dim_y + 1, dtype=np.float64)
        z_coords = np.arange(1, dim_z + 1, dtype=np.float64)

        # Intensity: normalize values within ROI, ignoring outside mask
        norm_bbox = self._normalize_bbox_inplace(bbox.copy(), ~mask)

        # Shape: binary mask array (1 inside ROI, 0 outside)
        shape_array = np.zeros_like(norm_bbox, dtype=np.float64)
        shape_array[mask] = 1.0

        # Compute intensity and shape invariants in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            fut_intensity = executor.submit(
                self._compute_invariants, norm_bbox, mask, x_coords, y_coords, z_coords
            )
            fut_shape = executor.submit(
                self._compute_invariants, shape_array, None, x_coords, y_coords, z_coords
            )
            intensity_invariants = fut_intensity.result()
            shape_invariants = fut_shape.result()

        combined_vector = np.concatenate((intensity_invariants, shape_invariants))
        self._mom_cache[roi_index] = {
            "intensity": intensity_invariants,
            "shape": shape_invariants,
            "vector": combined_vector,
        }

    # ----------------------------
    # Low-level helpers (adapted from the functional prototype)
    # ----------------------------
    def _normalize_bbox_inplace(self, arr: np.ndarray, background_mask: np.ndarray) -> np.ndarray:
        """
        Normalize ROI values to [0,1] on the ROI support by (x - min) / (mean - min).
        Background is set to 0. Uses `handle_math_operations` for safe division.
        """
        roi_mask = ~background_mask
        if not np.any(roi_mask):
            return arr

        roi_min = np.nanmin(arr)
        roi_mean = np.nanmean(arr, where=roi_mask)

        denom = roi_mean - roi_min
        denom = handle_math_operations(np.array(denom), self.feature_value_mode, mode='divide', epsilon=1e-30)

        # shift/scale only on ROI
        np.subtract(arr, roi_min, out=arr, where=roi_mask)
        np.divide(arr, denom, out=arr, where=roi_mask)

        # zero-out background and sanitize
        np.copyto(arr, 0.0, where=~roi_mask)
        np.nan_to_num(arr, copy=False, nan=0.0)
        return arr

    def _calculate_raw_mass(self, field: np.ndarray, support_mask: Optional[np.ndarray]) -> float:
        # Zeroth raw moment (mass). If support_mask is provided, it will be used to index field.
        if support_mask is not None:
            return float(np.nansum(field[support_mask]))
        return float(np.nansum(field))

    def _coordinate_first_raw(
            self,
            field: np.ndarray,
            x_coords: np.ndarray,
            y_coords: np.ndarray,
            z_coords: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Compute first-order raw moments along each axis.

        Parameters
        ----------
        field : np.ndarray
            3D array of intensities or binary values.
        x_coords, y_coords, z_coords : np.ndarray
            1D coordinate arrays along X, Y, Z axes.

        Returns
        -------
        tuple of float
            Raw moments (m100, m010, m001) along X, Y, Z.
        """
        # Sum along remaining axes to get marginal sums
        sum_x = np.nansum(field, axis=(1, 2))  # sum over Y and Z
        sum_y = np.nansum(field, axis=(0, 2))  # sum over X and Z
        sum_z = np.nansum(field, axis=(0, 1))  # sum over X and Y

        # Compute first-order raw moments
        m100 = float(np.dot(x_coords, sum_x))
        m010 = float(np.dot(y_coords, sum_y))
        m001 = float(np.dot(z_coords, sum_z))

        return m100, m010, m001

    def _higher_centered_moments(
            self,
            field: np.ndarray,
            x_coords: np.ndarray,
            y_coords: np.ndarray,
            z_coords: np.ndarray,
            mean_x: float,
            mean_y: float,
            mean_z: float
    ) -> Dict[str, float]:
        """
        Compute centered 3D moments up to third order for a given scalar field.

        Parameters
        ----------
        field : np.ndarray
            3D scalar field (X x Y x Z) values (e.g., ROI intensity or mask)
        x_coords, y_coords, z_coords : np.ndarray
            1D coordinate arrays for each axis
        mean_x, mean_y, mean_z : float
            Mean coordinates along each axis (centroid)

        Returns
        -------
        dict
            Dictionary of selected higher-order centered moments
        """

        # Centered coordinates
        x_dev = (x_coords - mean_x).astype(np.float64, copy=False)
        y_dev = (y_coords - mean_y).astype(np.float64, copy=False)
        z_dev = (z_coords - mean_z).astype(np.float64, copy=False)

        # Powers of centered coordinates up to 3rd order
        x_powers = np.column_stack([np.ones_like(x_dev), x_dev, x_dev ** 2, x_dev ** 3])  # (X,4)
        y_powers = np.column_stack([np.ones_like(y_dev), y_dev, y_dev ** 2, y_dev ** 3])  # (Y,4)
        z_powers = np.column_stack([np.ones_like(z_dev), z_dev, z_dev ** 2, z_dev ** 3])  # (Z,4)

        # Tensor contraction to compute moments
        moments_z = np.tensordot(field, z_powers, axes=([2], [0]))  # (X,Y,4)
        moments_yz = np.tensordot(moments_z, y_powers, axes=([1], [0]))  # (X,4,4)
        moments_tensor = np.tensordot(x_powers.T, moments_yz, axes=([1], [0]))  # (4,4,4)

        return {
            'm200': float(moments_tensor[2, 0, 0]),
            'm020': float(moments_tensor[0, 2, 0]),
            'm002': float(moments_tensor[0, 0, 2]),
            'm110': float(moments_tensor[1, 1, 0]),
            'm101': float(moments_tensor[1, 0, 1]),
            'm011': float(moments_tensor[0, 1, 1]),
            'm300': float(moments_tensor[3, 0, 0]),
            'm030': float(moments_tensor[0, 3, 0]),
            'm003': float(moments_tensor[0, 0, 3]),
            'm210': float(moments_tensor[2, 1, 0]),
            'm201': float(moments_tensor[2, 0, 1]),
            'm120': float(moments_tensor[1, 2, 0]),
            'm102': float(moments_tensor[1, 0, 2]),
            'm021': float(moments_tensor[0, 2, 1]),
            'm012': float(moments_tensor[0, 1, 2]),
            'm111': float(moments_tensor[1, 1, 1]),
        }

    def _normalize_center_moments(
            self, moments: Dict[str, float], m000: float
    ) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        """
        Normalize centered moments for 2nd and 3rd order.

        Parameters
        ----------
        moments : dict
            Dictionary of raw centered moments (m200, m020, ...)
        m000 : float
            Zeroth-order moment (volume/intensity sum)

        Returns
        -------
        tuple of 2nd-order normalized moments, tuple of 3rd-order normalized moments
        """

        # Compute normalization factors with safeguard
        factor_5_3 = handle_math_operations(np.array(m000 ** (5.0 / 3.0)),
                                            self.feature_value_mode, mode='divide')
        factor_2 = handle_math_operations(np.array(m000 ** 2),
                                          self.feature_value_mode, mode='divide')

        # 2nd-order normalized moments
        norm2_m200 = moments['m200'] / factor_5_3
        norm2_m020 = moments['m020'] / factor_5_3
        norm2_m002 = moments['m002'] / factor_5_3
        norm2_m110 = moments['m110'] / factor_5_3
        norm2_m101 = moments['m101'] / factor_5_3
        norm2_m011 = moments['m011'] / factor_5_3

        # 3rd-order normalized moments
        norm3_m300 = moments['m300'] / factor_2
        norm3_m030 = moments['m030'] / factor_2
        norm3_m003 = moments['m003'] / factor_2
        norm3_m210 = moments['m210'] / factor_2
        norm3_m201 = moments['m201'] / factor_2
        norm3_m120 = moments['m120'] / factor_2
        norm3_m102 = moments['m102'] / factor_2
        norm3_m021 = moments['m021'] / factor_2
        norm3_m012 = moments['m012'] / factor_2
        norm3_m111 = moments['m111'] / factor_2

        second_order = (norm2_m200, norm2_m020, norm2_m002,
                        norm2_m110, norm2_m101, norm2_m011)

        third_order = (norm3_m300, norm3_m030, norm3_m003,
                       norm3_m210, norm3_m201, norm3_m120,
                       norm3_m102, norm3_m021, norm3_m012, norm3_m111)

        return second_order, third_order

    def _tensor_invariants(
            self,
            second_order: Tuple[float, ...],
            third_order: Tuple[float, ...]
    ) -> Tuple[float, float, float, float, float]:
        """
        Compute 3D tensor invariants from normalized centered moments.

        Parameters
        ----------
        second_order : tuple of float
            Normalized second-order moments: (n200, n020, n002, n110, n101, n011)
        third_order : tuple of float
            Normalized third-order moments:
            (n300, n030, n003, n210, n201, n120, n102, n021, n012, n111)

        Returns
        -------
        tuple of float
            Five invariants: (i1, f2, i2, i3, f3)
        """

        # Unpack second-order normalized moments
        n200, n020, n002, n110, n101, n011 = second_order

        # Unpack third-order normalized moments
        n300, n030, n003, n210, n201, n120, n102, n021, n012, n111 = third_order

        # 2nd-order invariants
        invariant_i1 = float(n200 + n020 + n002)
        invariant_f2 = float(
            n200 ** 2 + n020 ** 2 + n002 ** 2
            + 2.0 * (n101 ** 2 + n110 ** 2 + n011 ** 2)
        )
        invariant_i2 = float(
            n200 * n020 + n200 * n002 + n020 * n002
            - n101 ** 2 - n110 ** 2 - n011 ** 2
        )
        invariant_i3 = float(
            n200 * n020 * n002
            - n002 * n110 ** 2
            + 2.0 * n110 * n101 * n011
            - n020 * n101 ** 2
            - n200 * n011 ** 2
        )

        # 3rd-order Frobenius norm squared
        invariant_f3 = float(
            n300 ** 2 + n030 ** 2 + n003 ** 2
            + 3.0 * (n210 ** 2 + n201 ** 2 + n120 ** 2 + n102 ** 2 + n021 ** 2 + n012 ** 2)
            + 6.0 * n111 ** 2
        )

        return invariant_i1, invariant_f2, invariant_i2, invariant_i3, invariant_f3

    def _compute_invariants(
            self,
            field: np.ndarray,
            support_mask: Optional[np.ndarray],
            x: np.ndarray,
            y: np.ndarray,
            z: np.ndarray
    ) -> np.ndarray:
        """
        Compute 3D geometric/intensity invariants for a given ROI field.

        Parameters
        ----------
        field : np.ndarray
            ROI intensity or binary mask array.
        support_mask : Optional[np.ndarray]
            Mask array to restrict computation (1 inside ROI, 0 outside).
        x, y, z : np.ndarray
            Coordinate grids corresponding to the ROI.

        Returns
        -------
        np.ndarray
            Array of five invariants: [i1, f2, i2, i3, f3]
        """

        # Zeroth-order raw moment (mass / total intensity)
        m000 = self._calculate_raw_mass(field, support_mask)
        if not np.isfinite(m000) or m000 == 0.0:
            val = self._fallback_value()
            return np.full(5, val, dtype=np.float64)

        # First-order raw moments and centroid coordinates
        m100, m010, m001 = self._coordinate_first_raw(field, x, y, z)
        m000_safe = handle_math_operations(np.array(m000),
                                           self.feature_value_mode, mode='divide')
        x_centroid = m100 / m000_safe
        y_centroid = m010 / m000_safe
        z_centroid = m001 / m000_safe

        # Higher-order centered moments (up to 3rd order)
        centered_moments = self._higher_centered_moments(field, x, y, z,
                                                         x_centroid, y_centroid, z_centroid)

        # Normalize centered moments and form invariants
        second_order, third_order = self._normalize_center_moments(centered_moments, m000)
        invariant_i1, invariant_f2, invariant_i2, invariant_i3, invariant_f3 = \
            self._tensor_invariants(second_order, third_order)

        return np.array([invariant_i1, invariant_f2, invariant_i2, invariant_i3, invariant_f3],
                        dtype=np.float64)

    # =============================
    #      Feature API methods
    # =============================
    # --- Intensity-only ---
    def get_moment_i1_intensity(self, roi_index: int) -> float:
        self._ensure_moments(roi_index)
        return float(self._mom_cache[roi_index]["intensity"][0])

    def get_moment_f2_intensity(self, roi_index: int) -> float:
        self._ensure_moments(roi_index)
        return float(self._mom_cache[roi_index]["intensity"][1])

    def get_moment_i2_intensity(self, roi_index: int) -> float:
        self._ensure_moments(roi_index)
        return float(self._mom_cache[roi_index]["intensity"][2])

    def get_moment_i3_intensity(self, roi_index: int) -> float:
        self._ensure_moments(roi_index)
        return float(self._mom_cache[roi_index]["intensity"][3])

    def get_moment_f3_intensity(self, roi_index: int) -> float:
        self._ensure_moments(roi_index)
        return float(self._mom_cache[roi_index]["intensity"][4])

    # --- Shape-only ---
    def get_moment_i1_shape(self, roi_index: int) -> float:
        self._ensure_moments(roi_index)
        return float(self._mom_cache[roi_index]["shape"][0])

    def get_moment_f2_shape(self, roi_index: int) -> float:
        self._ensure_moments(roi_index)
        return float(self._mom_cache[roi_index]["shape"][1])

    def get_moment_i2_shape(self, roi_index: int) -> float:
        self._ensure_moments(roi_index)
        return float(self._mom_cache[roi_index]["shape"][2])

    def get_moment_i3_shape(self, roi_index: int) -> float:
        self._ensure_moments(roi_index)
        return float(self._mom_cache[roi_index]["shape"][3])

    def get_moment_f3_shape(self, roi_index: int) -> float:
        self._ensure_moments(roi_index)
        return float(self._mom_cache[roi_index]["shape"][4])