# -*- coding: utf-8 -*-
# core/extractors/intensity_peak_features_extractor.py

from __future__ import annotations

import math
import logging
import numpy as np
from typing import Tuple, Optional

try:
    from scipy.signal import fftconvolve
    _HAS_SCIPY: bool = True
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    _HAS_SCIPY: bool = False

from pysera.engine.visera_oop.core.sparsity.view_planner import DataView
from pysera.engine.visera_oop.core.base_feature_extractor import BaseFeatureExtractor

logger = logging.getLogger("Dev_logger")


class IntensityPeakFeatureExtractor(BaseFeatureExtractor):
    """
    IBSI-compliant local/global intensity peak.

    - Neighbourhood: 1 cm^3 sphere in WORLD coordinates (uses (dz,dy,dx) in cm).
    - Centres: only voxels inside the ROI intensity mask.
    - Neighbourhood voxels: use *all* image voxels (not limited to ROI).
    """
    ACCEPTS = DataView.DENSE_BLOCK
    PREFERS  = DataView.DENSE_BLOCK | getattr(DataView, "DENSE_BLOCK_UNMASKED", 0)
    NAME = "IntensityPeakExtractor"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._cache: dict[tuple, np.ndarray] = {}

    # ---------- kernel utilities ----------

    @staticmethod
    def _compute_unit_sphere_radius_cm() -> float:
        """
        Computes the radius of a sphere whose volume is 1 cm³.

        Formula: (4/3) * π * r³ = 1  =>  r = (3 / (4π))^(1/3)

        Returns:
            float: Radius in centimeters.
        """
        pi_value: float = math.pi
        four_times_pi: float = math.prod([4.0, pi_value])  # 4 * π
        reciprocal_four_pi: float = math.pow(four_times_pi, -1.0)  # 1 / (4 * π)
        ratio_three_over_four_pi: float = math.prod([3.0, reciprocal_four_pi])  # 3 / (4 * π)
        radius_cm: float = math.pow(ratio_three_over_four_pi, math.pow(3.0, -1.0))  # (3 / (4π))^(1/3)

        return float(radius_cm)

    def _build_spherical_kernel(self,
                               voxel_spacing_z_cm: float,
                               voxel_spacing_y_cm: float,
                               voxel_spacing_x_cm: float
    ) -> np.ndarray:
        """
        Builds a 3D spherical kernel with radius corresponding to 1 cm³ volume.

        Args:
            voxel_spacing_z_cm (float): Voxel spacing in Z-direction (cm).
            voxel_spacing_y_cm (float): Voxel spacing in Y-direction (cm).
            voxel_spacing_x_cm (float): Voxel spacing in X-direction (cm).

        Returns:
            np.ndarray: 3D float32 spherical kernel with center voxel included.
        """
        radius_cm: float = self._compute_unit_sphere_radius_cm()

        small_constant: float = np.finfo(np.float64).eps if self.feature_value_mode == "APPROXIMATE_VALUE" else 0.0

        # Compute voxel radius in each dimension
        radius_voxels_z: int = max(1, int(math.ceil(radius_cm / max(voxel_spacing_z_cm, small_constant))))
        radius_voxels_y: int = max(1, int(math.ceil(radius_cm / max(voxel_spacing_y_cm, small_constant))))
        radius_voxels_x: int = max(1, int(math.ceil(radius_cm / max(voxel_spacing_x_cm, small_constant))))

        # Generate 3D coordinate grids using method-based operations
        z_coords: np.ndarray = np.arange(np.negative(radius_voxels_z), np.add(radius_voxels_z, 1), dtype=np.int32)
        y_coords: np.ndarray = np.arange(np.negative(radius_voxels_y), np.add(radius_voxels_y, 1), dtype=np.int32)
        x_coords: np.ndarray = np.arange(np.negative(radius_voxels_x), np.add(radius_voxels_x, 1), dtype=np.int32)
        grid_z, grid_y, grid_x = np.meshgrid(z_coords, y_coords, x_coords, indexing="ij")

        # Compute squared distances from center (cm²) using method-based operations
        dist_squared_z: np.ndarray = np.power(np.multiply(grid_z, voxel_spacing_z_cm), 2.0)
        dist_squared_y: np.ndarray = np.power(np.multiply(grid_y, voxel_spacing_y_cm), 2.0)
        dist_squared_x: np.ndarray = np.power(np.multiply(grid_x, voxel_spacing_x_cm), 2.0)

        distance_squared_cm: np.ndarray = np.add(dist_squared_z, np.add(dist_squared_y, dist_squared_x))

        # Build spherical kernel
        kernel: np.ndarray = (distance_squared_cm <= math.pow(radius_cm, 2.0)).astype(np.float32, copy=False)

        # Ensure center voxel is included
        kernel[radius_voxels_z, radius_voxels_y, radius_voxels_x] = 1.0

        return kernel

    def _kernel(self, voxel_spacing_cm: tuple[float, float, float]) -> np.ndarray:
        """
        Retrieve or build a spherical kernel for given voxel spacing.

        Args:
            voxel_spacing_cm (tuple[float, float, float]): Spacing along (Z, Y, X) in centimeters.

        Returns:
            np.ndarray: 3D float32 spherical kernel.
        """
        cache_key = ("intensity_peak_kernel",) + tuple(float(v) for v in voxel_spacing_cm)
        kernel: Optional[np.ndarray] = self._cache.get(cache_key)

        if kernel is None:
            kernel = self._build_spherical_kernel(*voxel_spacing_cm)
            self._cache[cache_key] = kernel

        return kernel

    # ---------- data assembly (shape-safe) ----------

    @staticmethod
    def _roi_mask_from_dense_block(dense_block: np.ndarray) -> np.ndarray:
        # ROI intensity mask has finite values inside ROI; NaN outside
        return np.isfinite(dense_block)

    def _get_grid_payload(
        self, index: int
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
        """
        Returns:
            - roi_masked: full-frame image with NaN outside ROI
            - image_unmasked: full-frame image (fallback: cropped)
            - voxel_spacing: (spacing_z_cm, spacing_y_cm, spacing_x_cm)
        """
        views = self.get_views(index)

        # --- spacing (z,y,x in cm) ---
        voxel_spacing = views.get("voxel_spacing_cm") or getattr(self, "voxel_spacing_cm", None)
        if voxel_spacing is None:
            raise RuntimeError("Voxel spacing (cm) is required for intensity peak calculations.")
        spacing_z, spacing_y, spacing_x = map(float, voxel_spacing)

        # --- prefer FULL resampled image & FULL mask ---
        img_full = views.get("image_resampled")
        mask_full = views.get("binary_mask_full")

        if (
            isinstance(img_full, np.ndarray) and img_full.ndim == 3 and img_full.size > 0 and
            isinstance(mask_full, np.ndarray) and mask_full.ndim == 3 and
            img_full.shape == mask_full.shape
        ):
            image_unmasked = np.asarray(img_full, dtype=np.float32, order="C")
            roi_masked = np.where(mask_full.astype(bool), image_unmasked, np.nan).astype(np.float32, copy=False)
            return roi_masked, image_unmasked, (spacing_z, spacing_y, spacing_x)

        # --- robust fallback: use cropped views (previous behavior) ---
        roi_masked = views.get("dense_block")
        if not (isinstance(roi_masked, np.ndarray) and roi_masked.ndim == 3):
            roi_masked = self.get_image(index)
        if not (isinstance(roi_masked, np.ndarray) and roi_masked.ndim == 3):
            raise RuntimeError("IntensityPeakFeatureExtractor requires 'dense_block' (ROI intensity mask).")
        roi_masked = np.asarray(roi_masked, dtype=np.float32, order="C")

        image_unmasked = views.get("dense_block_unmasked")
        if not (isinstance(image_unmasked, np.ndarray) and image_unmasked.ndim == 3):
            image_unmasked = np.nan_to_num(roi_masked, nan=0.0).astype(np.float32, copy=False)

        return roi_masked, np.asarray(image_unmasked, dtype=np.float32, order="C"), (spacing_z, spacing_y, spacing_x)

    # ---------- convolution-based spherical mean ----------

    @staticmethod
    def _spherical_mean_map(image_unmasked: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Compute per-voxel spherical mean using a convolution numerator and a robust,
        spatially-varying denominator (convolution of ones with the same kernel).

        Args:
            image_unmasked (np.ndarray): Input image (NaNs outside ROI allowed).
            kernel (np.ndarray): 3D spherical kernel (float32).

        Returns:
            np.ndarray: Per-voxel spherical mean map.
        """
        if _HAS_SCIPY:
            img_clean = np.nan_to_num(image_unmasked, nan=0.0, copy=False).astype(np.float32, copy=False)
            numerator = fftconvolve(img_clean, kernel, mode="same")
            ones_block = np.ones_like(img_clean, dtype=np.float32)
            denominator = fftconvolve(ones_block, kernel, mode="same")
            with np.errstate(divide="ignore", invalid="ignore"):
                mean_map = np.divide(numerator, denominator)
            return mean_map.astype(np.float32, copy=False)

        # Fallback: sliding window (slow)
        radius_voxels_z = np.floor_divide(kernel.shape[0], 2)
        radius_voxels_y = np.floor_divide(kernel.shape[1], 2)
        radius_voxels_x = np.floor_divide(kernel.shape[2], 2)

        size_z, size_y, size_x = image_unmasked.shape
        output_map = np.full_like(image_unmasked, np.nan, dtype=np.float32)
        kernel_mask = np.greater(kernel, 0.0)

        for center_z in range(size_z):
            z_start = np.maximum(np.subtract(center_z, radius_voxels_z), 0)
            z_end = np.minimum(np.add(center_z, np.add(radius_voxels_z, 1)), size_z)
            kernel_start_z = np.add(radius_voxels_z, np.subtract(0, np.subtract(z_start, center_z)))
            kernel_end_z = np.add(radius_voxels_z, np.subtract(z_end, center_z))

            for center_y in range(size_y):
                y_start = np.maximum(np.subtract(center_y, radius_voxels_y), 0)
                y_end = np.minimum(np.add(center_y, np.add(radius_voxels_y, 1)), size_y)
                kernel_start_y = np.add(radius_voxels_y, np.subtract(0, np.subtract(y_start, center_y)))
                kernel_end_y = np.add(radius_voxels_y, np.subtract(y_end, center_y))

                for center_x in range(size_x):
                    x_start = np.maximum(np.subtract(center_x, radius_voxels_x), 0)
                    x_end = np.minimum(np.add(center_x, np.add(radius_voxels_x, 1)), size_x)
                    kernel_start_x = np.add(radius_voxels_x, np.subtract(0, np.subtract(x_start, center_x)))
                    kernel_end_x = np.add(radius_voxels_x, np.subtract(x_end, center_x))

                    sub_block = image_unmasked[z_start:z_end, y_start:y_end, x_start:x_end]
                    kernel_sub = kernel_mask[
                        kernel_start_z:kernel_end_z, kernel_start_y:kernel_end_y, kernel_start_x:kernel_end_x]

                    if not np.any(kernel_sub):
                        continue

                    values = sub_block[kernel_sub]
                    if values.size == 0:
                        continue

                    output_map[center_z, center_y, center_x] = float(np.mean(values))

        return output_map

    # ---------- helpers ----------

    @staticmethod
    def _roi_max_coords(roi_masked: np.ndarray) -> tuple[float, np.ndarray]:
        max_value: float = float(np.nanmax(roi_masked))
        max_coords: np.ndarray = np.argwhere(np.equal(roi_masked, max_value))

        return max_value, max_coords

    # ---------- features ----------
    def get_loc_peak_loc(self, roi_index: int) -> float:
        roi_masked, image_unmasked, spacing_cm = self._get_grid_payload(roi_index)
        roi_binary = np.isfinite(roi_masked)
        if not np.any(roi_binary):
            return float("nan")

        kernel = self._kernel(spacing_cm)
        mean_map = self._spherical_mean_map(image_unmasked, kernel)

        # maxima of *ROI-masked* intensities (full frame)
        max_val = float(np.nanmax(roi_masked))
        max_centers = np.argwhere(np.equal(roi_masked, max_val))
        if max_centers.size == 0:
            return float("nan")

        vals = [
            float(mean_map[int(z), int(y), int(x)])
            for z, y, x in max_centers
            if np.isfinite(mean_map[int(z), int(y), int(x)])
        ]
        return float(np.max(vals)) if vals else float("nan")

    def get_loc_peak_glob(self, roi_index: int) -> float:
        roi_masked, image_unmasked, spacing_cm = self._get_grid_payload(roi_index)
        roi_binary = np.isfinite(roi_masked)
        if not np.any(roi_binary):
            return float("nan")

        kernel = self._kernel(spacing_cm)
        mean_map = self._spherical_mean_map(image_unmasked, kernel)

        vals = mean_map[roi_binary]
        vals = vals[np.isfinite(vals)]
        return float(np.nanmax(vals)) if vals.size else float("nan")