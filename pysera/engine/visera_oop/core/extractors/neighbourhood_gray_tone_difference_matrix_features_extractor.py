# # -*- coding: utf-8 -*-
# # core/extractors/neighbourhood_gray_tone_difference_matrix_features_extractor.py

from __future__ import annotations

import logging
import numpy as np
from scipy.ndimage import convolve
from typing import Any, Dict, List, Optional, Tuple, Callable

from pysera.engine.visera_oop.core.sparsity.view_planner import DataView
from pysera.engine.visera_oop.core.base_feature_extractor import BaseFeatureExtractor

logger = logging.getLogger("Dev_logger")


class NeighbourhoodGrayToneDifferenceMatrixFeatureExtractor(BaseFeatureExtractor):
    """NGTDM feature extractor with dynamic getters."""

    ACCEPTS: DataView = DataView.DENSE_BLOCK
    PREFERS: DataView = DataView.DENSE_BLOCK

    # ----------------------------------------------------------------------
    # Dynamic getter
    # ----------------------------------------------------------------------
    def __getattr__(self, name: str) -> Callable[[int], float]:
        """Return a callable for a dynamically resolved feature."""

        if not name.startswith("get_ngtdm_"):
            raise AttributeError(f"{type(self).__name__} has no attribute {name}")

        core = name[len("get_ngtdm_"):]

        mode: Optional[str] = None
        feature: Optional[str] = None
        for m in ("2_5D", "3D", "2D"):
            if core.endswith(f"_{m}"):
                mode = m
                feature = core[: -(len(m) + 1)]
                break

        if mode is None or feature is None:
            logger.error(f"Invalid mode or feature: {name}")
            return lambda *_args, **_kwargs: float(np.nan)

        calc_func: Optional[Callable[[Tuple[np.ndarray, np.ndarray]], float]] = getattr(
            self, f"_calc_{feature}", None
        )
        if not callable(calc_func):
            logger.error(f"Feature '{feature}' calculation function is missing.")
            return lambda *_args, **_kwargs: float(np.nan)

        def getter(roi_index: int) -> float:
            """Resolve the requested NGTDM feature for the given ROI."""

            if mode == "2D":
                result = self._mean_over_slices(roi_index, calc_func)
            elif mode == "2_5D":
                matrix = self._get_m25_feature(roi_index)
                result = calc_func(matrix) if matrix is not None else float(np.nan)
            else:  # "3D"
                matrix = self._get_m3_feature(roi_index)
                result = calc_func(matrix) if matrix is not None else float(np.nan)

            return result

        return getter

    # ----------------------------------------------------------------------
    # Feature discovery
    # ----------------------------------------------------------------------
    def _discover_feature_names(self) -> List[str]:
        """Discover implemented NGTDM feature names in a canonical order."""

        # List all methods in the class, including inherited ones
        methods = []
        for cls in self.__class__.mro():  # Traverse MRO to include base classes
            methods.extend(name for name, obj in cls.__dict__.items() if callable(obj))

        logger.debug(f"Methods found in the class (including parent classes): {methods}")

        # Look for the '_calc_' methods
        metric_names: List[str] = [
            name[len("_calc_"):]
            for name in methods
            if name.startswith("_calc_")
        ]
        logger.debug(f"Found _calc_ methods: {metric_names}")

        # Only include features that match the allowed modes
        dimensions = self._allowed_modes
        feature_names: List[str] = [
            f"ngtdm_{metric}_{dim}" for dim in dimensions for metric in metric_names
        ]

        return feature_names

    def _get_empty_value(self) -> float:
        """Return default empty feature value based on feature value mode."""
        return 0.0 if str(self.feature_value_mode).upper() == "APPROXIMATE_VALUE" else float(np.nan)

    @classmethod
    def _is_mode_allowed(cls, mode: str) -> bool:
        """Check if the subclass supports the given mode."""
        result = getattr(cls, "_allowed_modes", []) and mode in cls._allowed_modes

        return result

    # ----------------------------------------------------------------------
    # Cache management
    # ----------------------------------------------------------------------
    def _ensure_cache(self, roi_id: int) -> None:
        """Ensure that the NGTDM cache for a given ROI is populated."""

        if not hasattr(self, "_ngtdm_cache"):
            self._ngtdm_cache: Dict[int, Dict[str, Any]] = {}

        roi_quant: Optional[np.ndarray] = self.get_roi(roi_id)
        roi_views: Dict[str, Any] = self.get_views(roi_id)

        mask, levels = self._prepare_mask_and_levels(roi_quant, roi_views)

        if roi_quant is None or roi_quant.ndim != 3 or mask is None or levels is None or mask.shape != roi_quant.shape:
            self._ngtdm_cache[roi_id] = {}
            logger.warning(f"Invalid data for ROI {roi_id}, skipping cache population.")
            return

        mats_2d, mat_25d = self._build_2d_and_25d_matrices(roi_quant, mask, levels)
        mat_3d = self._build_3d_matrix(roi_quant, mask, levels)

        self._ngtdm_cache[roi_id] = {
            "ngtdm_matrices_2d": mats_2d,
            "ngtdm_matrices_25d": mat_25d,
            "ngtdm_matrices_3d": mat_3d,
            "LEVEL_ROWS": len(levels),
        }

    # ----------------------------------------------------------------------
    # Cached matrix retrieval
    # ----------------------------------------------------------------------
    def _get_m25_feature(self, roi_id: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Retrieve the 2.5D aggregated NGTDM for a given ROI."""
        self._ensure_cache(roi_id)
        return self._ngtdm_cache.get(roi_id, {}).get("ngtdm_matrices_25d", None)

    def _get_m3_feature(self, roi_id: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Retrieve the 3D NGTDM for a given ROI."""
        self._ensure_cache(roi_id)
        return self._ngtdm_cache.get(roi_id, {}).get("ngtdm_matrices_3d", None)

    # ----------------------------------------------------------------------
    # 2D aggregator
    # ----------------------------------------------------------------------
    def _mean_over_slices(
        self, roi_id: int, metric_func: Callable[[Tuple[np.ndarray, np.ndarray]], float]
    ) -> float:
        """Compute the mean of a given metric across all 2D slices of an ROI."""
        self._ensure_cache(roi_id)
        mats: List[Tuple[np.ndarray, np.ndarray]] = self._ngtdm_cache.get(roi_id, {}).get(
            "ngtdm_matrices_2d", []
        )
        if not mats:
            return float(np.nan)
        vals: List[float] = []
        for s_i, n_i in mats:
            v = metric_func((s_i, n_i))
            vals.append(v if np.isfinite(v) else np.nan)
        return float(np.nanmean(vals)) if vals else float(np.nan)

    def _build_global_level_indexer(
        self, levels: np.ndarray
    ) -> Tuple[np.ndarray, Dict[float, int], int]:
        """Build a mapping from gray levels to integer row indices.

        The returned dictionary maps each level value to a 0‑based index.  A
        sentinel value is appended to the level list to represent NaNs and
        out‑of‑mask voxels.  Rounding is applied to stabilise floating point
        equality comparisons.
        """
        num_levels: int = int(levels.size)
        scale_factor: int = 10000 if num_levels > 100 else 1000
        sentinel_value: float = float(np.max(levels) + 1.0)
        levels_with_sentinel = np.append(levels.astype(np.float32, copy=False), sentinel_value)
        rounded_levels = self._round_levels(levels_with_sentinel, scale_factor)
        level_to_index: Dict[float, int] = {
            float(val): idx for idx, val in enumerate(rounded_levels)
        }
        return rounded_levels, level_to_index, scale_factor

    def _prepare_mask_and_levels(
            self, quant_array: Optional[np.ndarray], views: Dict[str, Any]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare binary mask and level list for the NGTDM computation."""

        if quant_array is None or quant_array.ndim != 3:
            return None, None

        mask_array: Optional[np.ndarray] = views.get("binary_mask")
        levels_array: Optional[np.ndarray] = views.get("levels")

        if mask_array is None:
            mask_array = np.isfinite(quant_array).astype(np.uint8, copy=False)

        if levels_array is None or isinstance(levels_array, np.ndarray) and levels_array.size == 0:
            unique_levels = np.unique(quant_array[np.isfinite(quant_array)])
            levels_array = unique_levels.astype(np.float32, copy=False) if unique_levels.size else None

        return mask_array, levels_array

    @staticmethod
    def _round_levels(values: np.ndarray, scale_factor: int) -> np.ndarray:
        """Round intensity levels using a scaling factor.

        Parameters
        ----------
        values : np.ndarray
            Input array of intensity values.
        scale_factor : int
            Scaling factor used for rounding (e.g., 1000 or 10000).

        Returns
        -------
        np.ndarray
            Array of rounded intensity values.
        """
        rounded_values = np.asarray(values, dtype=np.float32).copy()
        rounded_values *= scale_factor
        np.rint(rounded_values, out=rounded_values)
        rounded_values /= scale_factor
        return rounded_values

    def _compute_ngtdm(self, roi_data: np.ndarray, roi_mask: np.ndarray, gray_levels: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray]:
        # --- Fast value→index mapping (vectorized; 0 outside ROI) ---
        # Ensure levels are sorted unique
        levels = np.asarray(np.unique(gray_levels), dtype=np.float32)
        # Replace NaN outside ROI to -inf, then map to indices [1..N] or 0 if unmatched
        arr = roi_data.astype(np.float32, copy=False)
        arr = np.where(roi_mask > 0, arr, np.nan)

        # searchsorted on rounded grid (same idea as your GLCM mapping)
        scale = 10000 if levels.size > 100 else 1000
        lvl = np.rint(levels * scale) / scale
        val = np.rint(np.nan_to_num(arr, nan=np.inf) * scale) / scale  # inf ensures "unmatched"
        pos = np.searchsorted(lvl, val)
        inside = (pos < lvl.size) & np.isfinite(val) & (val == lvl[np.clip(pos, 0, lvl.size - 1)])
        idx = np.where(inside & (roi_mask > 0), pos + 1, 0).astype(np.int32)

        # --- Neighborhood mean via one convolution (8-neigh 2D / 26-neigh 3D) ---
        if roi_data.ndim == 2:
            kernel = np.ones((3, 3), np.float32)
            kernel[1, 1] = 0.0
        else:
            kernel = np.ones((3, 3, 3), np.float32)
            kernel[1, 1, 1] = 0.0

        valid = (idx > 0).astype(np.float32)
        idx_f = idx.astype(np.float32)

        # Sum of neighbor levels & neighbor counts (masked)
        neigh_sum = convolve(idx_f, kernel, mode="constant", cval=0.0)  # Σ neighbor level-numbers
        neigh_cnt = convolve(valid, kernel, mode="constant", cval=0.0)  # number of valid neighbors

        has_nb = neigh_cnt > 0
        mean_nb = np.zeros_like(neigh_sum, dtype=np.float32)
        mean_nb[has_nb] = neigh_sum[has_nb] / neigh_cnt[has_nb]

        # Absolute difference between center "level number" and neighbor mean
        diff = np.abs(idx_f - mean_nb)
        used = (idx > 0) & has_nb

        # Bin by center level (1..N -> 0..N-1)
        n_levela = int(levels.size)
        centers = (idx[used] - 1).astype(np.int64)
        s = np.bincount(centers, weights=diff[used].astype(np.float64), minlength=n_levela)
        n_levela = np.bincount(centers, minlength=n_levela).astype(np.int64)
        return s, n_levela

    # ----------------------------------------------------------------------
    # Matrix builders
    # ----------------------------------------------------------------------
    def _build_2d_and_25d_matrices(
            self, quant_array: np.ndarray, mask_array: np.ndarray, levels_array: np.ndarray
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray]]:
        """Compute per‑slice and aggregated 2.5D NGTDM matrices."""

        num_slices: int = quant_array.shape[2]
        mats_2d: List[Tuple[np.ndarray, np.ndarray]] = []
        agg_s = np.zeros(len(levels_array), dtype=np.float64)
        agg_n = np.zeros(len(levels_array), dtype=np.int64)

        for s in range(num_slices):
            slice_quant = quant_array[:, :, s]
            slice_mask = mask_array[:, :, s]
            if np.count_nonzero(slice_mask) == 0:
                mats_2d.append((np.zeros_like(agg_s), np.zeros_like(agg_n)))
                continue
            s_i, n_i = self._compute_ngtdm(slice_quant, slice_mask, levels_array)
            mats_2d.append((s_i, n_i))
            agg_s += s_i
            agg_n += n_i

        return mats_2d, (agg_s, agg_n)

    def _build_3d_matrix(
            self, quant_array: np.ndarray, mask_array: np.ndarray, levels_array: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the NGTDM matrix for the entire 3D volume."""
        result = self._compute_ngtdm(quant_array, mask_array, levels_array)

        return result

    # Feature calculations (coarseness, contrast, etc.)
    def _calc_coarseness(self, matrix: Tuple[np.ndarray, np.ndarray]) -> float:
        """Calculate the coarseness feature."""
        s_arr, n_arr = matrix
        n_voxels = int(np.sum(n_arr))
        if n_voxels == 0:
            return self._get_empty_value()
        p_i = n_arr.astype(np.float64) / n_voxels
        denom = float(np.dot(p_i, s_arr))
        if denom <= 0:
            return 1e6 if str(self.feature_value_mode).upper() == "APPROXIMATE_VALUE" else float(np.nan)
        coarseness = 1.0 / denom

        return float(min(coarseness, 1e6))

    def _calc_contrast(self, matrix: Tuple[np.ndarray, np.ndarray]) -> float:
        """Calculate the contrast feature."""
        s_arr, n_arr = matrix
        n_voxels = int(np.sum(n_arr))
        if n_voxels == 0:
            return self._get_empty_value()
        p_i = n_arr.astype(np.float64) / n_voxels
        valid_indices = np.where(p_i > 0)[0]
        n_gp = len(valid_indices)
        if n_gp < 2:
            return 0.0 if str(self.feature_value_mode).upper() == "APPROXIMATE_VALUE" else 0.0
        indices = np.arange(1, len(p_i) + 1, dtype=np.float64)
        diff_sq = (indices[:, None] - indices[None, :]) ** 2
        term1 = float(np.sum((p_i[:, None] * p_i[None, :]) * diff_sq))
        term1 /= (n_gp * (n_gp - 1))
        term2 = float(np.sum(s_arr)) / n_voxels
        contrast = term1 * term2

        return float(contrast)

    def _calc_busyness(self, matrix: Tuple[np.ndarray, np.ndarray]) -> float:
        """Calculate the busyness feature."""
        s_arr, n_arr = matrix
        n_voxels = int(np.sum(n_arr))
        if n_voxels == 0:
            return self._get_empty_value()
        p_i = n_arr.astype(np.float64) / n_voxels
        numerator = float(np.dot(p_i, s_arr))
        indices = np.arange(1, len(p_i) + 1, dtype=np.float64)
        valid = p_i > 0
        if valid.sum() < 2:
            return 0.0 if str(self.feature_value_mode).upper() == "APPROXIMATE_VALUE" else 0.0
        i_mesh, j_mesh = np.meshgrid(indices, indices, indexing="ij")
        pi_mesh, pj_mesh = np.meshgrid(p_i, p_i, indexing="ij")
        denom_matrix = np.abs(i_mesh * pi_mesh - j_mesh * pj_mesh)
        denom = float(np.sum(denom_matrix[valid[:, None] & valid[None, :]]))
        if denom == 0.0:
            if str(self.feature_value_mode).upper() == "APPROXIMATE_VALUE":
                return numerator / np.finfo(float).eps
            else:
                return 0.0
        return float(numerator / denom)

    def _calc_complexity(self, matrix: Tuple[np.ndarray, np.ndarray]) -> float:
        """Calculate the complexity feature (NaN-safe)."""
        s_arr, n_arr = matrix
        n_voxels = int(np.sum(n_arr))
        if n_voxels == 0:
            return self._get_empty_value()

        p_i = n_arr.astype(np.float64) / n_voxels
        indices = np.arange(1, len(p_i) + 1, dtype=np.float64)

        i1_grid, i2_grid = np.meshgrid(indices, indices, indexing="ij")
        p1_grid, p2_grid = np.meshgrid(p_i, p_i, indexing="ij")
        s1_grid, s2_grid = np.meshgrid(s_arr, s_arr, indexing="ij")

        # Valid entries: both probabilities > 0
        valid_mask = (p1_grid > 0) & (p2_grid > 0)
        if not valid_mask.any():
            return 0.0 if str(self.feature_value_mode).upper() == "APPROXIMATE_VALUE" else 0.0

        diff_abs = np.abs(i1_grid - i2_grid)
        numer = (p1_grid * s1_grid + p2_grid * s2_grid)
        denom = (p1_grid + p2_grid)

        # Safe division to avoid invalid value warnings
        with np.errstate(divide='ignore', invalid='ignore'):
            summand = np.where(denom != 0, diff_abs * (numer / denom), 0.0)

        complexity = float(np.sum(summand[valid_mask]) / n_voxels)
        return complexity

    def _calc_strength(self, matrix: Tuple[np.ndarray, np.ndarray]) -> float:
        """Calculate the strength feature."""
        s_arr, n_arr = matrix
        n_voxels = int(np.sum(n_arr))
        if n_voxels == 0:
            return self._get_empty_value()
        p_i = n_arr.astype(np.float64) / n_voxels
        indices = np.arange(1, len(p_i) + 1, dtype=np.float64)
        i1_grid, i2_grid = np.meshgrid(indices, indices, indexing="ij")
        p1_grid, p2_grid = np.meshgrid(p_i, p_i, indexing="ij")
        valid_mask = (p1_grid > 0) & (p2_grid > 0)
        if not valid_mask.any():
            return 0.0 if str(self.feature_value_mode).upper() == "APPROXIMATE_VALUE" else 0.0
        numerator_matrix = (p1_grid + p2_grid) * ((i1_grid - i2_grid) ** 2)
        numerator = float(np.sum(numerator_matrix[valid_mask]))
        denom = float(np.sum(s_arr))
        if denom == 0.0:
            return 0.0 if str(self.feature_value_mode).upper() == "APPROXIMATE_VALUE" else float(np.nan)
        return float(numerator / denom)


class NeighbourhoodGrayToneDifferenceMatrixFeature2DExtractor(NeighbourhoodGrayToneDifferenceMatrixFeatureExtractor):
    """NGTDM 2D Extractor."""
    NAME: str = "NGTDM2DExtractor"
    _allowed_modes = ["2D"]

    def _mean_over_slices(
            self, roi_id: int, metric_func: Callable[[Tuple[np.ndarray, np.ndarray]], float]
    ) -> float:
        """Safely compute the mean of a given NGTDM metric across all 2D slices of an ROI."""

        # Ensure the cache is populated
        self._ensure_cache(roi_id)

        # Retrieve 2D matrices from cache
        mats: List[Tuple[np.ndarray, np.ndarray]] = self._ngtdm_cache.get(roi_id, {}).get(
            "ngtdm_matrices_2d", []
        )

        if not mats:
            return float(np.nan)

        # Compute metric for each slice
        values: List[float] = []
        for s_i, n_i in mats:
            try:
                v = metric_func((s_i, n_i))
                if np.isfinite(v):
                    values.append(v)
            except Exception as e:
                logger.warning(f"[NGTDM] Metric computation failed for ROI {roi_id}: {e}")
                continue

        # Return NaN if no valid values exist
        if len(values) == 0:
            return float(np.nan)

        # Safe mean without warnings
        return float(np.mean(values))


class NeighbourhoodGrayToneDifferenceMatrixFeature25DExtractor(NeighbourhoodGrayToneDifferenceMatrixFeatureExtractor):
    """NGTDM 2.5D Extractor."""
    NAME: str = "NGTDM25DExtractor"
    _allowed_modes = ["2_5D"]

    def _get_m25_feature(self, roi_id: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Retrieve the 2.5D aggregated NGTDM for a given ROI."""
        self._ensure_cache(roi_id)
        return self._ngtdm_cache.get(roi_id, {}).get("ngtdm_matrices_25d", None)


class NeighbourhoodGrayToneDifferenceMatrixFeature3DExtractor(NeighbourhoodGrayToneDifferenceMatrixFeatureExtractor):
    """NGTDM 3D Extractor."""
    NAME: str = "NGTDM3DExtractor"
    _allowed_modes = ["3D"]

    def _get_m3_feature(self, roi_id: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Retrieve the 3D NGTDM for a given ROI."""
        self._ensure_cache(roi_id)
        return self._ngtdm_cache.get(roi_id, {}).get("ngtdm_matrices_3d", None)

    def _mean_over_slices(self, roi_id: int, metric_func: Callable[[Tuple[np.ndarray, np.ndarray]], float]) -> float:
        """Override mean for 3D: 3D mode doesn't need to average across slices."""
        return float(np.nan)  # No slice-wise averaging in 3D
