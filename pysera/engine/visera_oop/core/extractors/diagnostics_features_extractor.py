# -*- coding: utf-8 -*-
# core/extractors/diagnostics_features_extractor.py

from __future__ import annotations

import logging
import numpy as np
from typing import Tuple, Any, Dict

from pysera.engine.visera_oop.core.sparsity.view_planner import DataView
from pysera.engine.visera_oop.core.base_feature_extractor import BaseFeatureExtractor

logger = logging.getLogger("Dev_logger")


def _finite(array: np.ndarray) -> np.ndarray:
    """
    Check element-wise finiteness of a NumPy array.

    Args:
        array (np.ndarray): Input array to check.

    Returns:
        np.ndarray: Boolean mask where True indicates finite (not NaN or ±Inf) values.
    """
    return np.isfinite(array)


def _stats_whole(array: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute basic statistics (mean, min, max) for all finite elements in an array.

    Args:
        array (np.ndarray): Input numeric array.

    Returns:
        Tuple[float, float, float]: A tuple containing:
            - mean (float): Mean of finite values.
            - minimum (float): Minimum of finite values.
            - maximum (float): Maximum of finite values.
            If the input is empty or contains no finite values, all outputs are NaN.
    """
    if not isinstance(array, np.ndarray) or array.size == 0:
        return float("nan"), float("nan"), float("nan")

    finite_values = array[np.isfinite(array)]

    if finite_values.size == 0:
        return float("nan"), float("nan"), float("nan")

    mean_value = float(np.mean(finite_values))
    min_value = float(np.min(finite_values))
    max_value = float(np.max(finite_values))

    return mean_value, min_value, max_value


def _stats_masked(array: np.ndarray, mask: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute basic statistics (mean, min, max) of elements in `array`
    where `mask` is True, considering only finite (non-NaN, non-inf) values.

    Args:
        array (np.ndarray): Input numeric array.
        mask (np.ndarray): Boolean mask array (same shape as `array`).

    Returns:
        Tuple[float, float, float]: A tuple containing:
            - mean (float): Mean of masked finite values.
            - minimum (float): Minimum of masked finite values.
            - maximum (float): Maximum of masked finite values.
            If input arrays are invalid or no valid values exist, all outputs are NaN.
    """
    # Validate inputs
    if (
        not isinstance(array, np.ndarray)
        or not isinstance(mask, np.ndarray)
        or array.size == 0
        or mask.size == 0
    ):
        return float("nan"), float("nan"), float("nan")

    # Ensure mask is boolean
    if mask.dtype is not bool:
        mask = mask.astype(bool, copy=False)

    # Ensure matching shapes
    if mask.shape != array.shape:
        return float("nan"), float("nan"), float("nan")

    # Select finite, masked values
    finite_values = array[mask & np.isfinite(array)]

    if finite_values.size == 0:
        return float("nan"), float("nan"), float("nan")

    # Compute statistics
    mean_value = float(np.mean(finite_values))
    min_value = float(np.min(finite_values))
    max_value = float(np.max(finite_values))

    return mean_value, min_value, max_value

class DiagnosticsFeaturesExtractor(BaseFeatureExtractor):
    """
    Reports diagnostics for:
      • Initial image/ROI  -> STRICTLY raw/full inputs (pre-ROI ops, pre-resample)
      • Interpolated image/ROI -> resampled grid (unmasked/morph mask)
      • Resegmented ROI   -> intensity block after reseg + morph mask

    Arrays come as (z, y, x); we report (X, Y, Z) = (axis2, axis1, axis0).
    Spacing keys:
      - views["voxel_spacing_cm_initial"] = original header spacing (dz, dy, dx)
      - views["voxel_spacing_cm"]         = effective spacing after interpolation (dz, dy, dx)
    """
    ACCEPTS = DataView.DENSE_BLOCK
    PREFERS = DataView.DENSE_BLOCK
    NAME = "DiagnosticsExtractor"

    # ---------- spacing helpers ----------

    @staticmethod
    def _centimeters_to_millimeters(length_cm: float) -> float:
        """
        Convert a length from centimeters to millimeters.

        Args:
            length_cm (float): Length in centimeters.

        Returns:
            float: Length converted to millimeters.
        """
        return length_cm * 10.0

    def _spacing_mm_xyz_from_views(self, views: Dict[str, Any], spacing_key: str) -> Tuple[float, float, float]:
        """
        Extract voxel spacing from a dictionary of views and convert from centimeters to millimeters.

        Args:
            views (Dict[str, Any]): Dictionary containing spacing information.
            spacing_key (str): Key to access the spacing tuple/list in the form (dz, dy, dx) in cm.

        Returns:
            Tuple[float, float, float]: Spacing in millimeters in the order (dx, dy, dz).
            Returns NaNs if the key is missing or the spacing is invalid.
        """
        spacing_cm = views.get(spacing_key)
        if spacing_cm is None or len(spacing_cm) != 3:
            return float("nan"), float("nan"), float("nan")

        dz_cm, dy_cm, dx_cm = map(float, spacing_cm)
        return (
            self._centimeters_to_millimeters(dx_cm),
            self._centimeters_to_millimeters(dy_cm),
            self._centimeters_to_millimeters(dz_cm),
        )

    # ---------- generic shape/bbox helpers ----------

    @staticmethod
    def _dims_xyz(array: np.ndarray) -> Tuple[int, int, int]:
        """
        Return the dimensions of a 3D array in (X, Y, Z) order.

        Args:
            array (np.ndarray): Input 3D array.

        Returns:
            Tuple[int, int, int]: Dimensions in order (X, Y, Z).
            Returns (0, 0, 0) if input is not a 3D array.
        """
        if not isinstance(array, np.ndarray) or array.ndim != 3:
            return 0, 0, 0
        # Convert (z, y, x) -> (X, Y, Z)
        return int(array.shape[2]), int(array.shape[1]), int(array.shape[0])

    @staticmethod
    def _bbox_dims_xyz(mask: np.ndarray) -> Tuple[int, int, int]:
        """
        Compute the bounding box dimensions of a 3D mask in (X, Y, Z) order.

        Args:
            mask (np.ndarray): 3D binary mask array.

        Returns:
            Tuple[int, int, int]: Bounding box dimensions (width_x, height_y, depth_z).
            Returns (0, 0, 0) if mask is invalid or empty.
        """
        if not isinstance(mask, np.ndarray) or mask.ndim != 3 or mask.size == 0:
            return 0, 0, 0

        binary_mask = mask > 0 if mask.dtype is not bool else mask

        if not binary_mask.any():
            return 0, 0, 0

        z_indices, y_indices, x_indices = np.where(binary_mask)
        width_x = int(x_indices.max() - x_indices.min() + 1)
        height_y = int(y_indices.max() - y_indices.min() + 1)
        depth_z = int(z_indices.max() - z_indices.min() + 1)

        return width_x, height_y, depth_z

    # ---------- view accessors ----------

    def _views(self, roi_index: int) -> Dict[str, Any]:
        """
        Retrieve all views for a given ROI index.

        Args:
            roi_index (int): Index of the region of interest (ROI).

        Returns:
            Dict[str, Any]: Dictionary of views. Returns an empty dictionary if no views are found.
        """
        return self.get_views(roi_index) or {}

    def _initial_image_full(self, roi_index: int) -> np.ndarray:
        """
        Retrieve the raw, full 3D image for a given ROI.

        Tries the keys 'raw_image_full' and '_raw_image_full'.
        If neither is available nor valid, logs an error and returns an empty array.

        Args:
            roi_index (int): Index of the region of interest (ROI).

        Returns:
            np.ndarray: 3D array of the raw image. Returns empty array if not available.
        """
        views = self._views(roi_index)

        image = views.get("raw_image_full")
        if isinstance(image, np.ndarray) and image.ndim == 3 and image.size > 0:
            return image

        backup_image = views.get("_raw_image_full")
        if isinstance(backup_image, np.ndarray) and backup_image.ndim == 3 and backup_image.size > 0:
            return backup_image

        logger.error(
            "Diagnostics: 'raw_image_full' missing; initial-image metrics will be invalid."
        )
        return np.array([], dtype=np.float32)

    def _initial_roi_mask_full(self, roi_index: int) -> np.ndarray:
        """
        Retrieve the full 3D ROI mask for a given ROI index.

        Tries the keys 'raw_roi_mask_full' and '_raw_roi_mask_full'.
        Returns a boolean mask where values > 0 are True.
        If no valid mask is found, logs an error and returns a zeros mask
        with the same shape as the initial image.

        Args:
            roi_index (int): Index of the region of interest (ROI).

        Returns:
            np.ndarray: 3D boolean array representing the ROI mask.
        """
        views = self._views(roi_index)

        mask = views.get("raw_roi_mask_full")
        if isinstance(mask, np.ndarray) and mask.ndim == 3 and mask.size > 0:
            return mask > 0

        backup_mask = views.get("_raw_roi_mask_full")
        if isinstance(backup_mask, np.ndarray) and backup_mask.ndim == 3 and backup_mask.size > 0:
            return backup_mask > 0

        logger.error(
            "Diagnostics: 'raw_roi_mask_full' missing; initial-ROI metrics will be invalid."
        )
        # Return a boolean mask of zeros with the same shape as the initial image
        return np.zeros_like(self._initial_image_full(roi_index), dtype=bool)

    def _interp_image_full(self, roi_index: int) -> np.ndarray:
        """
        Retrieve the interpolated (resampled) full 3D image for a given ROI.

        Prefers the following keys in order:
        1. 'image_resampled_full'
        2. 'image_resampled'
        3. 'dense_block_unmasked'

        If none of these are available, falls back to the initial full image.

        Args:
            roi_index (int): Index of the region of interest (ROI).

        Returns:
            np.ndarray: 3D array of the interpolated image.
        """
        views = self._views(roi_index)
        preferred_keys = ("image_resampled_full", "image_resampled", "dense_block_unmasked")

        for key in preferred_keys:
            image = views.get(key)
            if isinstance(image, np.ndarray) and image.ndim == 3 and image.size > 0:
                return image

        # Fallback to initial full image
        return self._initial_image_full(roi_index)

    def _interp_image_and_mask(self, roi_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return a full resampled 3D image and a boolean mask of the same shape for a given ROI.
        """
        views = self._views(roi_index)

        # ---- Image selection (full preferred) ----
        image_full = views.get("image_resampled")
        if not (isinstance(image_full, np.ndarray) and image_full.ndim == 3 and image_full.size > 0):
            image_full = views.get("dense_block_unmasked")
        if not (isinstance(image_full, np.ndarray) and image_full.ndim == 3 and image_full.size > 0):
            return np.array([], dtype=np.float32), np.zeros((0, 0, 0), dtype=bool)
        image_full = np.asarray(image_full, dtype=np.float32, order="C")

        # ---- Mask selection (full preferred) ----
        mask_full_resampled = views.get("binary_mask_full")
        if isinstance(mask_full_resampled, np.ndarray) and mask_full_resampled.ndim == 3 and mask_full_resampled.shape == image_full.shape:
            return image_full, mask_full_resampled > 0

        # ---- Rebuild mask from cropped mask + bbox ----
        mask_cropped = views.get("binary_mask")
        crop_bbox = views.get("crop_bbox")
        if (
            isinstance(mask_cropped, np.ndarray) and mask_cropped.ndim == 3 and
            isinstance(crop_bbox, (tuple, list)) and len(crop_bbox) == 6
        ):
            x_start, y_start, z_start, x_end, y_end, z_end = map(int, crop_bbox)
            depth_image, height_image, width_image = image_full.shape

            # Clamp bbox to image dimensions
            z_start = max(0, min(z_start, depth_image))
            z_end = max(0, min(z_end, depth_image))
            y_start = max(0, min(y_start, height_image))
            y_end = max(0, min(y_end, height_image))
            x_start = max(0, min(x_start, width_image))
            x_end = max(0, min(x_end, width_image))

            depth_mask = max(0, min(z_end - z_start, mask_cropped.shape[0], depth_image - z_start))
            height_mask = max(0, min(y_end - y_start, mask_cropped.shape[1], height_image - y_start))
            width_mask = max(0, min(x_end - x_start, mask_cropped.shape[2], width_image - x_start))

            mask_rebuilt = np.zeros_like(image_full, dtype=bool)
            if depth_mask > 0 and height_mask > 0 and width_mask > 0:
                mask_rebuilt[z_start:z_start + depth_mask,
                             y_start:y_start + height_mask,
                             x_start:x_start + width_mask] = mask_cropped[:depth_mask, :height_mask, :width_mask] > 0
            return image_full, mask_rebuilt

        # ---- Use cropped mask if shape already matches ----
        if isinstance(mask_cropped, np.ndarray) and mask_cropped.ndim == 3 and mask_cropped.shape == image_full.shape:
            return image_full, mask_cropped > 0

        # ---- Last resort: empty mask ----
        return image_full, np.zeros_like(image_full, dtype=bool)

    # Reseg: intensity block passed to extract (post-reseg)
    def _reseg_image(self, roi_index: int) -> np.ndarray:
        """
        Retrieve the resegmented 3D image for a given ROI.

        Tries the following sources in order:
        1) self.get_image(roi_index)
        2) views[DataView.DENSE_BLOCK]

        Returns an empty array if no valid 3D image is found.

        Args:
            roi_index (int): Index of the region of interest (ROI).

        Returns:
            np.ndarray: 3D array of the resegmented image (float32).
                        Returns empty array if no valid image exists.
        """
        image = self.get_image(roi_index)
        if isinstance(image, np.ndarray) and image.ndim == 3 and image.size > 0:
            return image

        views = self._views(roi_index)
        dense_block_image = views.get(DataView.DENSE_BLOCK)
        if isinstance(dense_block_image, np.ndarray) and dense_block_image.ndim == 3:
            return dense_block_image

        return np.array([], dtype=np.float32)

    def _morph_mask_interp_full(self, roi_index: int) -> np.ndarray:
        """
        Full-size resampled morphological mask aligned to the full resampled image.

        Prefers 'binary_mask_full'; otherwise reconstructs it from the cropped mask + crop_bbox.
        Falls back to cropped mask if needed.

        Args:
            roi_index (int): Index of the region of interest (ROI).

        Returns:
            np.ndarray: Boolean 3D array representing the full-size morphological mask.
        """
        views = self._views(roi_index)

        # ---- Full mask preferred ----
        mask_full_resampled = views.get("binary_mask_full")
        if isinstance(mask_full_resampled, np.ndarray) and mask_full_resampled.ndim == 3:
            return mask_full_resampled > 0

        # ---- Rebuild from cropped mask + bbox ----
        mask_cropped = views.get("binary_mask")
        crop_bbox = views.get("crop_bbox")
        full_image = views.get("image_resampled")

        if (
            isinstance(mask_cropped, np.ndarray) and mask_cropped.ndim == 3 and
            isinstance(crop_bbox, (tuple, list)) and len(crop_bbox) == 6 and
            isinstance(full_image, np.ndarray) and full_image.ndim == 3
        ):
            x_start, y_start, z_start, x_end, y_end, z_end = map(int, crop_bbox)
            depth_img, height_img, width_img = full_image.shape

            # Clamp bbox
            z_start = max(0, min(z_start, depth_img))
            z_end = max(0, min(z_end, depth_img))
            y_start = max(0, min(y_start, height_img))
            y_end = max(0, min(y_end, height_img))
            x_start = max(0, min(x_start, width_img))
            x_end = max(0, min(x_end, width_img))

            depth_mask = max(0, min(z_end - z_start, mask_cropped.shape[0], depth_img - z_start))
            height_mask = max(0, min(y_end - y_start, mask_cropped.shape[1], height_img - y_start))
            width_mask = max(0, min(x_end - x_start, mask_cropped.shape[2], width_img - x_start))

            mask_full = np.zeros_like(full_image, dtype=bool)
            if depth_mask > 0 and height_mask > 0 and width_mask > 0:
                mask_full[z_start:z_start + depth_mask,
                          y_start:y_start + height_mask,
                          x_start:x_start + width_mask] = mask_cropped[:depth_mask, :height_mask, :width_mask] > 0
            return mask_full

        # ---- Fallback: use cropped mask ----
        return self._morph_mask_interp(roi_index)

    def _morph_mask_interp(self, roi_index: int) -> np.ndarray:
        """
        Retrieve the cropped or interpolated morphological mask for a given ROI.

        Prefers the following sources in order:
        1) views["binary_mask"]
        2) self.get_roi(roi_index)
        3) Fallback: zeros mask with the same shape as the interpolated full image

        Args:
            roi_index (int): Index of the region of interest (ROI).

        Returns:
            np.ndarray: 3D boolean mask array.
        """
        views = self._views(roi_index)

        # ---- Cropped mask from views ----
        mask_cropped = views.get("binary_mask")
        if isinstance(mask_cropped, np.ndarray) and mask_cropped.ndim == 3:
            return mask_cropped > 0

        # ---- ROI from manager ----
        roi_mask = self.get_roi(roi_index)
        if isinstance(roi_mask, np.ndarray) and roi_mask.ndim == 3:
            return roi_mask > 0

        # ---- Fallback: zeros mask matching interpolated full image ----
        return np.zeros_like(self._interp_image_full(roi_index), dtype=bool)

    # ---------- INITIAL IMAGE (raw/full) ----------

    def get_img_dim_x_init_img(self, roi_index: int) -> int:
        return self._dims_xyz(self._initial_image_full(roi_index))[2]

    def get_img_dim_y_init_img(self, roi_index: int) -> int:
        return self._dims_xyz(self._initial_image_full(roi_index))[1]

    def get_img_dim_z_init_img(self, roi_index: int) -> int:
        return self._dims_xyz(self._initial_image_full(roi_index))[0]

    def get_vox_dim_x_init_img(self, roi_index: int) -> float:
        return float(np.round(self._spacing_mm_xyz_from_views(self._views(roi_index), "voxel_spacing_cm_initial")[0], 3))

    def get_vox_dim_y_init_img(self, roi_index: int) -> float:
        return float(np.round(self._spacing_mm_xyz_from_views(self._views(roi_index), "voxel_spacing_cm_initial")[1], 3))

    def get_vox_dim_z_init_img(self, roi_index: int) -> float:
        return float(np.round(self._spacing_mm_xyz_from_views(self._views(roi_index), "voxel_spacing_cm_initial")[2], 3))

    def get_mean_int_init_img(self, roi_index: int) -> float:
        mean_value, _, _ = _stats_whole(self._initial_image_full(roi_index))
        return float(np.round(mean_value, 3)) if np.isfinite(mean_value) else float("nan")

    def get_min_int_init_img(self, roi_index: int) -> float:
        _, min_value, _ = _stats_whole(self._initial_image_full(roi_index))
        return float(np.round(min_value, 3)) if np.isfinite(min_value) else float("nan")

    def get_max_int_init_img(self, roi_index: int) -> float:
        _, _, max_value = _stats_whole(self._initial_image_full(roi_index))
        return float(np.round(max_value, 3)) if np.isfinite(max_value) else float("nan")

    # ---------- INTERPOLATED IMAGE (resampled) ----------

    def get_img_dim_x_interp_img(self, roi_index: int) -> int:
        return self._dims_xyz(self._interp_image_full(roi_index))[2]

    def get_img_dim_y_interp_img(self, roi_index: int) -> int:
        return self._dims_xyz(self._interp_image_full(roi_index))[1]

    def get_img_dim_z_interp_img(self, roi_index: int) -> int:
        return self._dims_xyz(self._interp_image_full(roi_index))[0]

    def get_vox_dim_x_interp_img(self, roi_index: int) -> float:
        return float(np.round(self._spacing_mm_xyz_from_views(self._views(roi_index), "voxel_spacing_cm")[0], 3))

    def get_vox_dim_y_interp_img(self, roi_index: int) -> float:
        return float(np.round(self._spacing_mm_xyz_from_views(self._views(roi_index), "voxel_spacing_cm")[1], 3))

    def get_vox_dim_z_interp_img(self, roi_index: int) -> float:
        return float(np.round(self._spacing_mm_xyz_from_views(self._views(roi_index), "voxel_spacing_cm")[2], 3))

    def get_mean_int_interp_img(self, roi_index: int) -> float:
        mean_value, _, _ = _stats_whole(self._interp_image_full(roi_index))
        return float(np.round(mean_value, 3)) if np.isfinite(mean_value) else float("nan")

    def get_min_int_interp_img(self, roi_index: int) -> float:
        _, min_value, _ = _stats_whole(self._interp_image_full(roi_index))
        return float(np.round(min_value, 3)) if np.isfinite(min_value) else float("nan")

    def get_max_int_interp_img(self, roi_index: int) -> float:
        _, _, max_value = _stats_whole(self._interp_image_full(roi_index))
        return float(np.round(max_value, 3)) if np.isfinite(max_value) else float("nan")

    # ---------- INITIAL ROI (raw/full grid + raw mask) ----------

    def get_int_mask_dim_x_init_roi(self, roi_index: int) -> int:
        return self.get_img_dim_x_init_img(roi_index)

    def get_int_mask_dim_y_init_roi(self, roi_index: int) -> int:
        return self.get_img_dim_y_init_img(roi_index)

    def get_int_mask_dim_z_init_roi(self, roi_index: int) -> int:
        return self.get_img_dim_z_init_img(roi_index)

    def get_int_mask_bb_dim_x_init_roi(self, roi_index: int) -> int:
        return self._bbox_dims_xyz(self._initial_roi_mask_full(roi_index))[2]

    def get_int_mask_bb_dim_y_init_roi(self, roi_index: int) -> int:
        return self._bbox_dims_xyz(self._initial_roi_mask_full(roi_index))[1]

    def get_int_mask_bb_dim_z_init_roi(self, roi_index: int) -> int:
        return self._bbox_dims_xyz(self._initial_roi_mask_full(roi_index))[0]

    def get_morph_mask_bb_dim_x_init_roi(self, roi_index: int) -> int:
        return self.get_int_mask_bb_dim_x_init_roi(roi_index)

    def get_morph_mask_bb_dim_y_init_roi(self, roi_index: int) -> int:
        return self.get_int_mask_bb_dim_y_init_roi(roi_index)

    def get_morph_mask_bb_dim_z_init_roi(self, roi_index: int) -> int:
        return self.get_int_mask_bb_dim_z_init_roi(roi_index)

    def get_int_mask_vox_count_init_roi(self, roi_index: int) -> int:
        return int(np.count_nonzero(self._initial_roi_mask_full(roi_index)))

    def get_morph_mask_vox_count_init_roi(self, roi_index: int) -> int:
        return self.get_int_mask_vox_count_init_roi(roi_index)

    def get_int_mask_mean_int_init_roi(self, roi_index: int) -> float:
        mean_value, _, _ = _stats_masked(self._initial_image_full(roi_index), self._initial_roi_mask_full(roi_index))
        return float(np.round(mean_value, 3)) if np.isfinite(mean_value) else float("nan")

    def get_int_mask_min_int_init_roi(self, roi_index: int) -> float:
        _, min_value, _ = _stats_masked(self._initial_image_full(roi_index), self._initial_roi_mask_full(roi_index))
        return float(np.round(min_value, 3)) if np.isfinite(min_value) else float("nan")

    def get_int_mask_max_int_init_roi(self, roi_index: int) -> float:
        _, _, max_value = _stats_masked(self._initial_image_full(roi_index), self._initial_roi_mask_full(roi_index))
        return float(np.round(max_value, 3)) if np.isfinite(max_value) else float("nan")

    # ---------- INTERPOLATED ROI (resampled grid + morph mask) ----------

    def get_int_mask_dim_x_interp_roi(self, roi_index: int) -> int:
        return self.get_img_dim_x_interp_img(roi_index)

    def get_int_mask_dim_y_interp_roi(self, roi_index: int) -> int:
        return self.get_img_dim_y_interp_img(roi_index)

    def get_int_mask_dim_z_interp_roi(self, roi_index: int) -> int:
        return self.get_img_dim_z_interp_img(roi_index)

    def get_int_mask_bb_dim_x_interp_roi(self, roi_index: int) -> int:
        return self._bbox_dims_xyz(self._morph_mask_interp(roi_index))[2]

    def get_int_mask_bb_dim_y_interp_roi(self, roi_index: int) -> int:
        return self._bbox_dims_xyz(self._morph_mask_interp(roi_index))[1]

    def get_int_mask_bb_dim_z_interp_roi(self, roi_index: int) -> int:
        return self._bbox_dims_xyz(self._morph_mask_interp(roi_index))[0]

    def get_morph_mask_bb_dim_x_interp_roi(self, roi_index: int) -> int:
        return self.get_int_mask_bb_dim_x_interp_roi(roi_index)

    def get_morph_mask_bb_dim_y_interp_roi(self, roi_index: int) -> int:
        return self.get_int_mask_bb_dim_y_interp_roi(roi_index)

    def get_morph_mask_bb_dim_z_interp_roi(self, roi_index: int) -> int:
        return self.get_int_mask_bb_dim_z_interp_roi(roi_index)

    def get_int_mask_vox_count_interp_roi(self, roi_index: int) -> int:
        return int(np.count_nonzero(self._morph_mask_interp(roi_index)))

    def get_morph_mask_vox_count_interp_roi(self, roi_index: int) -> int:
        return self.get_int_mask_vox_count_interp_roi(roi_index)

    def get_int_mask_mean_int_interp_roi(self, roi_index: int) -> float:
        img, mask = self._interp_image_and_mask(roi_index)
        mean_value, _, _ = _stats_masked(img, mask)
        return float(np.round(mean_value, 3)) if np.isfinite(mean_value) else float("nan")

    def get_int_mask_min_int_interp_roi(self, roi_index: int) -> float:
        img, mask = self._interp_image_and_mask(roi_index)
        _, min_value, _ = _stats_masked(img, mask)
        return float(np.round(min_value, 3)) if np.isfinite(min_value) else float("nan")

    def get_int_mask_max_int_interp_roi(self, roi_index: int) -> float:
        img, mask = self._interp_image_and_mask(roi_index)
        _, _, max_value = _stats_masked(img, mask)
        return float(np.round(max_value, 3)) if np.isfinite(max_value) else float("nan")

    # ---------- RESEGMENTED ROI (post-reseg intensity block + morph mask) ----------

    # CHANGED — report full resampled grid dims for the "reseg" dims (matches your expected values)
    def get_int_mask_dim_x_reseg_roi(self, roi_index: int) -> int:
        return self._dims_xyz(self._interp_image_full(roi_index))[2]

    def get_int_mask_dim_y_reseg_roi(self, roi_index: int) -> int:
        return self._dims_xyz(self._interp_image_full(roi_index))[1]

    def get_int_mask_dim_z_reseg_roi(self, roi_index: int) -> int:
        return self._dims_xyz(self._interp_image_full(roi_index))[0]

    def get_int_mask_bb_dim_x_reseg_roi(self, roi_index: int) -> int:
        return self._bbox_dims_xyz(self._morph_mask_interp(roi_index))[2]

    def get_int_mask_bb_dim_y_reseg_roi(self, roi_index: int) -> int:
        return self._bbox_dims_xyz(self._morph_mask_interp(roi_index))[1]

    def get_int_mask_bb_dim_z_reseg_roi(self, roi_index: int) -> int:
        return self._bbox_dims_xyz(self._morph_mask_interp(roi_index))[0]

    def get_morph_mask_bb_dim_x_reseg_roi(self, roi_index: int) -> int:
        return self.get_int_mask_bb_dim_x_reseg_roi(roi_index)

    def get_morph_mask_bb_dim_y_reseg_roi(self, roi_index: int) -> int:
        return self.get_int_mask_bb_dim_y_reseg_roi(roi_index)

    def get_morph_mask_bb_dim_z_reseg_roi(self, roi_index: int) -> int:
        return self.get_int_mask_bb_dim_z_reseg_roi(roi_index)

    def get_int_mask_vox_count_reseg_roi(self, roi_index: int) -> int:
        reseg_image = self._reseg_image(roi_index)
        morph_mask = self._morph_mask_interp(roi_index)

        if reseg_image.size and morph_mask.size and reseg_image.shape == morph_mask.shape:
            return int(np.count_nonzero(morph_mask & _finite(reseg_image)))

        return 0

    def get_morph_mask_vox_count_reseg_roi(self, roi_index: int) -> int:
        return int(np.count_nonzero(self._morph_mask_interp(roi_index)))

    def get_int_mask_mean_int_reseg_roi(self, roi_index: int) -> float:
        mean_value, _, _ = _stats_masked(self._reseg_image(roi_index), self._morph_mask_interp(roi_index))
        return float(np.round(mean_value, 3)) if np.isfinite(mean_value) else float("nan")

    def get_int_mask_min_int_reseg_roi(self, roi_index: int) -> float:
        _, min_value, _ = _stats_masked(self._reseg_image(roi_index), self._morph_mask_interp(roi_index))
        return float(np.round(min_value, 3)) if np.isfinite(min_value) else float("nan")

    def get_int_mask_max_int_reseg_roi(self, roi_index: int) -> float:
        _, _, max_value = _stats_masked(self._reseg_image(roi_index), self._morph_mask_interp(roi_index))
        return float(np.round(max_value, 3)) if np.isfinite(max_value) else float("nan")
