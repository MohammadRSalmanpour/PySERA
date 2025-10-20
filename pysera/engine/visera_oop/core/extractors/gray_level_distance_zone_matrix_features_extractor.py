# # -*- coding: utf-8 -*-
# # core/extractors/gray_level_distance_zone_matrix_features_extractor.py

from __future__ import annotations

import logging
import numpy as np
import scipy.ndimage as ndi
from typing import Any, Dict, Optional, List, Tuple, Callable

from pysera.engine.visera_oop.core.base_feature_extractor import BaseFeatureExtractor
from pysera.engine.visera_oop.core.sparsity.view_planner import DataView

logger = logging.getLogger("Dev_logger")


class GrayLevelDistanceZoneMatrixFeatureExtractor(BaseFeatureExtractor):
    """GLDZM feature extractor with dynamic getters for multiple dimensional modes."""

    ACCEPTS: DataView = DataView.DENSE_BLOCK
    PREFERS: DataView = DataView.DENSE_BLOCK

    # IBSI behaviour knob preserved (default False)
    use_compact_levels_3d: bool = False

    # ------------------------------------------------------------------
    # Dynamic getter: get_dzm_<metric>_<mode>
    # ------------------------------------------------------------------
    def __getattr__(self, name: str) -> Callable[[int], float]:
        """Dynamic getter for GLDZM-based features (e.g. get_dzm_szhge_3D, get_dzm_z_perc_2D, ...)."""

        # Validate prefix
        if not name.startswith("get_dzm_"):
            raise AttributeError(f"{type(self).__name__} has no attribute '{name}'")

        # Extract feature and mode
        core_name = name[len("get_dzm_"):]
        mode: Optional[str] = None
        feature_name: Optional[str] = None

        for mode_candidate in ("2_5D", "3D", "2D"):
            if core_name.endswith(f"_{mode_candidate}"):
                mode = mode_candidate
                feature_name = core_name[: -(len(mode_candidate) + 1)]
                break

        # Invalid attribute name
        if mode is None or feature_name is None:
            logger.error(f"[GLDZM] Invalid getter name: '{name}'")
            return lambda *_args, **_kwargs: float(np.nan)

        # Unsupported computation mode
        if not self._is_mode_allowed(mode):
            logger.error(f"[GLDZM] Mode '{mode}' not allowed for {self.__class__.__name__}")
            return lambda *_args, **_kwargs: float(np.nan)

        # Feature calculation function lookup
        calc_func = getattr(self, f"_calc_{feature_name}", None)
        if not callable(calc_func):
            logger.error(f"[GLDZM] Missing calculation method for feature '{feature_name}'")
            return lambda *_args, **_kwargs: float(np.nan)

        # --- Feature Getters --------------------------------------------------------

        # z_perc: requires ROI/slice context
        if feature_name == "z_perc":
            def getter(roi_index: int) -> float:
                if mode == "2D":
                    return self._mean_over_slices(
                        roi_index,
                        lambda mat, s: calc_func(mat, roi_index, s),
                    )
                if mode == "2_5D":
                    matrix_25d = self._get_m25_feature(roi_index)
                    return calc_func(matrix_25d, roi_index, None) if matrix_25d is not None else float(np.nan)
                # 3D
                matrix_3d = self._get_m3_feature(roi_index)
                return calc_func(matrix_3d, roi_index, None) if matrix_3d is not None else float(np.nan)

        # Other DZM features (no slice context needed)
        else:
            def getter(roi_index: int) -> float:
                if mode == "2D":
                    return self._mean_over_slices(roi_index, lambda mat, _: calc_func(mat))
                if mode == "2_5D":
                    matrix_25d = self._get_m25_feature(roi_index)
                    return calc_func(matrix_25d) if matrix_25d is not None else float(np.nan)
                # 3D
                matrix_3d = self._get_m3_feature(roi_index)
                return calc_func(matrix_3d) if matrix_3d is not None else float(np.nan)

        return getter

    # ------------------------------------------------------------------
    # Feature discovery
    # ------------------------------------------------------------------
    def _discover_feature_names(self) -> List[str]:
        """Discover implemented GLDZM feature names for this subclass' allowed modes."""

        methods: List[str] = []
        for cls in self.__class__.mro():
            methods.extend(name for name, obj in cls.__dict__.items() if callable(obj))
        metric_names = [name[len("_calc_"):] for name in methods if name.startswith("_calc_")]

        dims = getattr(self, "_allowed_modes", []) or ["2D", "2_5D", "3D"]
        feature_names = [f"dzm_{metric}_{dim}" for dim in dims for metric in metric_names]

        return feature_names

    @classmethod
    def _is_mode_allowed(cls, mode: str) -> bool:
        """Check if the subclass supports the given mode."""
        return hasattr(cls, "_allowed_modes") and mode in cls._allowed_modes

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------
    def _ensure_cache(self, roi_id: int) -> None:
        """Ensure GLDZM matrices are computed and cached for a given ROI."""

        if not hasattr(self, "_gldzm_cache"):
            self._gldzm_cache: Dict[int, Dict[str, Any]] = {}

        roi_quant: Optional[np.ndarray] = self.get_roi(roi_id)
        roi_views: Dict[str, Any] = self.get_views(roi_id)

        mask, levels = self._prepare_mask_and_levels(roi_quant, roi_views)
        if roi_quant is None or roi_quant.ndim != 3 or mask is None or levels is None or mask.shape != roi_quant.shape:
            self._gldzm_cache[roi_id] = {}
            logger.warning(f"Invalid data for ROI {roi_id}, skipping cache population.")
            return

        mats_2d, mat_25d = self._build_2d_and_25d_matrices(roi_quant, mask, levels)
        mat_3d = self._build_3d_matrix(roi_quant, mask, levels)

        self._gldzm_cache[roi_id] = {
            "gldzm_matrices_2d": mats_2d,
            "gldzm_matrices_25d": mat_25d,
            "gldzm_matrices_3d": mat_3d,
            "LEVEL_ROWS": len(self._build_global_level_indexer(levels)[0]) - 1,
        }

    # ------------------------------------------------------------------
    # Cached matrix retrieval
    # ------------------------------------------------------------------
    def _get_m25_feature(self, roi_id: int) -> Optional[np.ndarray]:
        self._ensure_cache(roi_id)
        return self._gldzm_cache.get(roi_id, {}).get("gldzm_matrices_25d", None)

    def _get_m3_feature(self, roi_id: int) -> Optional[np.ndarray]:
        self._ensure_cache(roi_id)
        return self._gldzm_cache.get(roi_id, {}).get("gldzm_matrices_3d", None)

    # ------------------------------------------------------------------
    # 2D aggregator
    # ------------------------------------------------------------------

    def _mean_over_slices(self, roi_index: int, metric_function: Callable[[np.ndarray, int], float]) -> float:
        """Compute the mean of a metric across all 2D slices of a given ROI."""
        self._ensure_cache(roi_index)

        matrices_2d = self._gldzm_cache.get(roi_index, {}).get("gldzm_matrices_2d")
        if not matrices_2d:
            return np.nan

        # Preallocate a NumPy array for speed instead of appending to a list
        n = len(matrices_2d)
        metric_values = np.empty(n, dtype=float)

        for i, mat in enumerate(matrices_2d):
            val = metric_function(mat, i)
            metric_values[i] = val if np.isfinite(val) else np.nan

        # Using np.nanmean directly is optimal for numeric arrays
        return float(np.nanmean(metric_values)) if n > 0 else np.nan

    # ------------------------------------------------------------------
    # Mask & level helpers
    # ------------------------------------------------------------------

    def _prepare_mask_and_levels(
            self, quant_array: Optional[np.ndarray], views: Dict[str, Any]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare binary mask and intensity levels (optimized for speed)."""
        if quant_array is None or quant_array.ndim != 3:
            return None, None

        mask_array = views.get("binary_mask")
        levels_array = views.get("levels")

        # Use quant_array directly to avoid redundant memory reads
        if mask_array is None:
            # np.isfinite is fast and avoids copying if we use where+astype in one step
            mask_array = np.isfinite(quant_array)
            mask_array = mask_array.astype(np.uint8, copy=False)

        if levels_array is None or (
                isinstance(levels_array, np.ndarray) and levels_array.size == 0
        ):
            # Use ravel to avoid creating a full new array view and process in one pass
            finite_vals = quant_array[np.isfinite(quant_array)]
            if finite_vals.size:
                # np.unique is already highly optimized in C
                levels_array = np.unique(finite_vals).astype(np.float32, copy=False)
            else:
                levels_array = None

        return mask_array, levels_array

    # ------------------------------------------------------------------
    # Level rounding / indexing
    # ------------------------------------------------------------------
    @staticmethod
    def _round_levels(values: np.ndarray, scale_factor: int) -> np.ndarray:
        """Optimized rounding with minimal memory overhead."""
        # Convert to float32 only if necessary, and round in-place
        if values.dtype != np.float32:
            values = values.astype(np.float32, copy=False)
        values = np.multiply(values, scale_factor, out=values, casting="unsafe")
        np.rint(values, out=values)
        np.divide(values, scale_factor, out=values)
        return values

    def _build_global_level_indexer(
            self, levels: np.ndarray
    ) -> Tuple[np.ndarray, Dict[float, int], int]:
        """Build a global indexer for quantized levels (optimized)."""
        num_levels = levels.size
        scale_factor = 10000 if num_levels > 100 else 1000

        # Use float32 once, avoid multiple conversions
        levels = np.asarray(levels, dtype=np.float32)

        # Append sentinel efficiently
        sentinel_value = np.float32(np.max(levels) + 1.0)
        levels_with_sentinel = np.empty(num_levels + 1, dtype=np.float32)
        levels_with_sentinel[:-1] = levels
        levels_with_sentinel[-1] = sentinel_value

        # Round in place (no new allocations)
        rounded_levels = self._round_levels(levels_with_sentinel, scale_factor)

        # Build level-to-index mapping using vectorized indexing for speed
        # np.arange + tolist avoids Python loop overhead
        level_to_index = dict(zip(map(float, rounded_levels.tolist()), range(rounded_levels.size)))

        return rounded_levels, level_to_index, scale_factor

    # ------------------------------------------------------------------
    # Distance helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _distance_map(padded_mask: np.ndarray) -> np.ndarray:
        """Compute distance map from ROI edges (4-conn in 2D, 6-conn in 3D), optimized version."""
        if padded_mask.ndim not in (2, 3):
            raise ValueError("Only 2D or 3D masks are supported.")

        # Ensure binary mask
        mask = (padded_mask > 0).astype(np.uint8, copy=False)

        # Use cityblock distance (4-conn in 2D, 6-conn in 3D)
        dist_inside = ndi.distance_transform_cdt(mask, metric='taxicab')

        # Replace 0 (outside or edge) with NaN
        dist_map = dist_inside.astype(np.float32)
        dist_map[mask == 0] = np.nan

        return dist_map

    def _compute_distance_map(self, roi_mask: np.ndarray) -> np.ndarray:
        """Optimized computation of the distance map for a given ROI mask with 1-voxel padding."""
        # Pad mask with one voxel border
        padding = ((1, 1),) * roi_mask.ndim
        padded_mask = np.pad((roi_mask > 0.5).astype(np.uint8, copy=False), padding)

        # Compute distances once (highly optimized)
        distance_map = self._distance_map(padded_mask)

        # Remove padding to restore original ROI size
        slices = tuple(slice(1, -1) for _ in range(roi_mask.ndim))
        return distance_map[slices]

    @staticmethod
    def _zone_structure(num_dims: int) -> np.ndarray:
        """Return the binary connectivity structure for zone labeling.

        Uses 8-connectivity for 2D and 26-connectivity for 3D.
        """
        connectivity: int = 3 if num_dims == 3 else 2
        return ndi.generate_binary_structure(num_dims, connectivity)

    # ------------------------------------------------------------------
    # Matrix builders
    # ------------------------------------------------------------------
    def _prepare_rounded_box(self, roi_box: np.ndarray, uniq_levels: np.ndarray, adjust: int) -> np.ndarray:
        sentinel = float(uniq_levels[-1])
        filled_box = np.nan_to_num(roi_box, nan=sentinel).astype(np.float32, copy=False)
        return self._round_levels(filled_box, adjust)

    @staticmethod
    def _get_max_distance(distance_map: np.ndarray) -> int:
        return int(np.nanmax(distance_map)) if np.isfinite(distance_map).any() else 0

    @staticmethod
    def _get_present_levels(rounded_box: np.ndarray, roi_box: np.ndarray, uniq_levels: np.ndarray) -> np.ndarray:
        sentinel = float(uniq_levels[-1])
        present = np.unique(rounded_box[np.isfinite(roi_box)])
        return present[present != sentinel]

    @staticmethod
    def _add_gray_level_zones(
            gldzm_matrix: np.ndarray,
            rounded_box: np.ndarray,
            distance_map: np.ndarray,
            gray_val: float,
            value_to_row: dict,
            struct_zone: np.ndarray,
    ) -> None:
        """Optimized version — same result, minimal runtime."""

        # Early exit: skip all unnecessary operations if gray_val not present
        if gray_val not in rounded_box:
            return

        # Binary mask for current gray value
        level_mask = (rounded_box == gray_val)
        if not level_mask.any():
            return

        # Label connected components (zones)
        labeled, n_zones = ndi.label(level_mask, structure=struct_zone)
        if n_zones == 0:
            return

        row_idx = value_to_row.get(float(gray_val))
        if row_idx is None or row_idx >= gldzm_matrix.shape[0]:
            return

        # Use vectorized zone min-distance computation (avoid Python loop)
        # Compute min distance per label efficiently
        zone_min_distances = ndi.minimum(distance_map, labels=labeled, index=np.arange(1, n_zones + 1))
        zone_min_distances = np.floor(zone_min_distances).astype(int) - 1  # distances start at 1
        valid_mask = (zone_min_distances >= 0) & (zone_min_distances < gldzm_matrix.shape[1])

        # Efficient bincount update
        if np.any(valid_mask):
            counts = np.bincount(zone_min_distances[valid_mask], minlength=gldzm_matrix.shape[1])
            gldzm_matrix[row_idx, :len(counts)] += counts

    def _build_gldzm_global_rows(self, roi_box: np.ndarray, roi_mask: np.ndarray, levels: np.ndarray) -> np.ndarray:
        """Optimized GLDZM builder with minimal runtime (same output)."""

        # Step 1. Precompute unique levels and adjustment mapping
        uniq_levels, value_to_row, adjust = self._build_global_level_indexer(levels)

        # Early exit if no valid levels
        n_levels = len(uniq_levels) - 1
        if n_levels <= 0:
            return np.zeros((0, 1), dtype=np.int64)

        # Step 2. Prepare rounded intensity box (already quantized)
        rounded_box = self._prepare_rounded_box(roi_box, uniq_levels, adjust)

        # Step 3. Compute distance map and its max distance
        dmap = self._compute_distance_map(roi_mask)
        max_distance = self._get_max_distance(dmap)
        if max_distance <= 0:
            return np.zeros((n_levels, 1), dtype=np.int64)

        # Step 4. Preallocate GLDZM matrix efficiently
        gldzm = np.zeros((n_levels, max_distance), dtype=np.int64)

        # Step 5. Determine present gray levels (skip ones not in mask)
        present_levels = self._get_present_levels(rounded_box, roi_box, uniq_levels)
        if len(present_levels) == 0:
            return gldzm

        # Step 6. Precompute zone structure (connectivity)
        struct_zone = self._zone_structure(roi_box.ndim)

        # Step 7. Vectorized loop over present gray levels
        add_zones = self._add_gray_level_zones
        for gray_val in np.asarray(present_levels, dtype=rounded_box.dtype):
            add_zones(gldzm, rounded_box, dmap, float(gray_val), value_to_row, struct_zone)

        return gldzm

    def _build_2d_and_25d_matrices(
            self, quant_img: np.ndarray, mask_img: np.ndarray, gray_levels: np.ndarray
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Optimized: Build per-slice 2D and aggregated 2.5D GLDZM matrices (identical output, minimal runtime)."""

        num_slices = quant_img.shape[2]

        # --- Step 1: Precompute global gray level indexer once ---
        unique_levels, _, _ = self._build_global_level_indexer(gray_levels)
        num_gray_levels = len(unique_levels) - 1
        if num_gray_levels <= 0 or num_slices == 0:
            return [], np.zeros((0, 0), dtype=np.int64)

        # --- Step 2: Prepare result containers efficiently ---
        gldzm_2d_list: List[np.ndarray] = [None] * num_slices  # preallocate list slots
        max_zone_length = 0

        # Local aliasing to minimize attribute lookup overhead
        build_rows = self._build_gldzm_global_rows
        count_nonzero = np.count_nonzero

        # --- Step 3: Vectorized per-slice loop ---
        for i in range(num_slices):
            mask_slice = mask_img[:, :, i]
            if not count_nonzero(mask_slice):
                # If mask empty, skip compute
                gldzm_matrix = np.zeros((num_gray_levels, 1), dtype=np.int64)
            else:
                gldzm_matrix = build_rows(quant_img[:, :, i], mask_slice, gray_levels)

            gldzm_2d_list[i] = gldzm_matrix
            if gldzm_matrix.shape[1] > max_zone_length:
                max_zone_length = gldzm_matrix.shape[1]

        # --- Step 4: Aggregate all slices efficiently ---
        if max_zone_length > 0:
            gldzm_25d = np.zeros((num_gray_levels, max_zone_length), dtype=np.int64)
            # Use in-place accumulation to avoid temporary arrays
            for mat in gldzm_2d_list:
                if mat.size:
                    gldzm_25d[:, :mat.shape[1]] += mat
        else:
            gldzm_25d = np.zeros((num_gray_levels, 0), dtype=np.int64)

        return gldzm_2d_list, gldzm_25d

    def _build_3d_matrix(
            self, quant_img: np.ndarray, mask_img: np.ndarray, gray_levels: np.ndarray
    ) -> np.ndarray:
        """Build the 3D GLDZM matrix for the entire volume.

        Parameters
        ----------
        quant_img : np.ndarray
            Quantized 3D image volume (H, W, D).
        mask_img : np.ndarray
            Binary mask of the same shape as `quant_img`.
        gray_levels : np.ndarray
            Discretized gray levels used for GLDZM computation.

        Returns
        -------
        np.ndarray
            The 3D GLDZM matrix representing zone sizes and gray levels for the entire volume.
        """
        gldzm_3d: np.ndarray = self._build_gldzm_global_rows(
            quant_img, mask_img, gray_levels
        )
        return gldzm_3d

    # ------------------------------------------------------------------
    # Matrix utilities for metric calculations
    # ------------------------------------------------------------------
    @staticmethod
    def _trim_trailing_zero_cols(matrix: np.ndarray) -> np.ndarray:
        if matrix.size == 0:
            return matrix
        col_sums = matrix.sum(axis=0)
        nonzero_cols = np.where(col_sums > 0)[0]
        return matrix[:, :1] if nonzero_cols.size == 0 else matrix[:, : nonzero_cols[-1] + 1]

    @staticmethod
    def _get_total_count(matrix: np.ndarray) -> int:
        return int(matrix.sum())

    @staticmethod
    def _get_row_and_column_sums(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        row_sums = matrix.sum(axis=1).astype(np.float64)
        col_sums = matrix.sum(axis=0).astype(np.float64)
        return row_sums, col_sums

    @staticmethod
    def _get_indices_and_grids(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        num_rows, num_cols = matrix.shape
        row_indices = np.arange(1, num_rows + 1, dtype=np.float64)
        col_indices = np.arange(1, num_cols + 1, dtype=np.float64)
        row_grid, col_grid = np.meshgrid(row_indices, col_indices, indexing="ij")
        return row_indices, col_indices, row_grid, col_grid

    def _get_empty_value(self) -> float:
        return 0.0 if str(self.feature_value_mode).upper() == "APPROXIMATE_VALUE" else float(np.nan)

    # ------------------------------------------------------------------
    # Metric preparation
    # ------------------------------------------------------------------
    def _prepare_matrix(
            self, matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        matrix = self._trim_trailing_zero_cols(matrix)
        total = self._get_total_count(matrix)
        row_idx = col_idx = row_grid = col_grid = np.array([], dtype=np.float64)
        if total > 0 and matrix.size:
            row_idx, col_idx, row_grid, col_grid = self._get_indices_and_grids(matrix)
        return matrix, row_idx, col_idx, row_grid, col_grid, total

    # ------------------------------------------------------------------
    # _calc_* metrics (GLDZM)
    # ------------------------------------------------------------------
    # 1 Small Distance Emphasis
    def _calc_sde(self, matrix: np.ndarray) -> float:
        """
        Calculate the Short Distance Emphasis (SDE) feature from a GLDZM matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Input 2D GLDZM matrix.

        Returns
        -------
        float
            SDE feature value. Returns empty value if matrix is empty.
        """
        matrix, row_indices, col_indices, row_grid, col_grid, total_count = self._prepare_matrix(matrix)

        if total_count == 0:
            return self._get_empty_value()

        col_sums: np.ndarray = matrix.sum(axis=0).astype(np.float64)
        sde_value: float = float((col_sums / (col_indices ** 2)).sum() / total_count)
        return sde_value

    # 2 Large Distance Emphasis
    def _calc_lde(self, matrix: np.ndarray) -> float:
        """
        Calculate the Long Distance Emphasis (LDE) feature from a GLDZM matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Input 2D GLDZM matrix.

        Returns
        -------
        float
            LDE feature value. Returns empty value if the matrix is empty.
        """
        matrix, row_indices, col_indices, row_grid, col_grid, total_count = self._prepare_matrix(matrix)

        if total_count == 0:
            return self._get_empty_value()

        col_sums: np.ndarray = matrix.sum(axis=0).astype(np.float64)
        lde_value: float = float((col_sums * (col_indices ** 2)).sum() / total_count)
        return lde_value

    # 3 Low Grey Level Zone Emphasis
    def _calc_lgze(self, matrix: np.ndarray) -> float:
        """
        Calculate the Low Gray-Level Zone Emphasis (LGZE) feature from a GLDZM matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Input 2D GLDZM matrix.

        Returns
        -------
        float
            LGZE feature value. Returns empty value if the matrix is empty.
        """
        matrix, row_indices, col_indices, row_grid, col_grid, total_count = self._prepare_matrix(matrix)

        if total_count == 0:
            return self._get_empty_value()

        row_sums: np.ndarray = matrix.sum(axis=1).astype(np.float64)
        lgze_value: float = float((row_sums / (row_indices ** 2)).sum() / total_count)

        return lgze_value

    # 4 High Grey Level Zone Emphasis
    def _calc_hgze(self, matrix: np.ndarray) -> float:
        """
        Calculate the High Gray-Level Zone Emphasis (HGZE) feature from a GLDZM matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Input 2D GLDZM matrix.

        Returns
        -------
        float
            HGZE feature value. Returns empty value if the matrix is empty.
        """
        matrix, row_indices, col_indices, row_grid, col_grid, total_count = self._prepare_matrix(matrix)

        if total_count == 0:
            return self._get_empty_value()

        row_sums: np.ndarray = matrix.sum(axis=1).astype(np.float64)
        hgze_value: float = float((row_sums * (row_indices ** 2)).sum() / total_count)
        return hgze_value

    # 5 Small Distance Low Grey Level Emphasis
    def _calc_sdlge(self, matrix: np.ndarray) -> float:
        """
        Calculate the Short Distance Low Gray-Level Emphasis (SDLGE) feature
        from a GLDZM matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Input 2D GLDZM matrix.

        Returns
        -------
        float
            SDLGE feature value. Returns empty value if the matrix is empty.
        """
        matrix, row_indices, col_indices, row_grid, col_grid, total_count = self._prepare_matrix(matrix)

        if total_count == 0:
            return self._get_empty_value()

        sdlge_value: float = float((matrix / ((row_grid ** 2) * (col_grid ** 2))).sum() / total_count)
        return sdlge_value

    # 6 Small Distance High Grey Level Emphasis
    def _calc_sdhge(self, matrix: np.ndarray) -> float:
        """
        Calculate the Short Distance High Gray-Level Emphasis (SDHGE) feature
        from a GLDZM matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Input 2D GLDZM matrix.

        Returns
        -------
        float
            SDHGE feature value. Returns empty value if the matrix is empty.
        """
        matrix, row_indices, col_indices, row_grid, col_grid, total_count = self._prepare_matrix(matrix)

        if total_count == 0:
            return self._get_empty_value()

        sdhge_value: float = float(((matrix * (row_grid ** 2)) / (col_grid ** 2)).sum() / total_count)
        return sdhge_value

    # 7 Large Distance Low Grey Level Emphasis
    def _calc_ldlge(self, matrix: np.ndarray) -> float:
        """
        Calculate the Long Distance Low Gray-Level Emphasis (LDLGE) feature
        from a GLDZM matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Input 2D GLDZM matrix.

        Returns
        -------
        float
            LDLGE feature value. Returns empty value if the matrix is empty.
        """
        matrix, row_indices, col_indices, row_grid, col_grid, total_count = self._prepare_matrix(matrix)

        if total_count == 0:
            return self._get_empty_value()

        ldlge_value: float = float(((matrix * (col_grid ** 2)) / (row_grid ** 2)).sum() / total_count)
        return ldlge_value

    # 8 Large Distance High Grey Level Emphasis
    def _calc_ldhge(self, matrix: np.ndarray) -> float:
        """
        Calculate the Long Distance High Gray-Level Emphasis (LDHGE) feature
        from a GLDZM matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Input 2D GLDZM matrix.

        Returns
        -------
        float
            LDHGE feature value. Returns empty value if the matrix is empty.
        """
        matrix, row_indices, col_indices, row_grid, col_grid, total_count = self._prepare_matrix(matrix)

        if total_count == 0:
            return self._get_empty_value()

        ldhge_value: float = float((matrix * (row_grid ** 2) * (col_grid ** 2)).sum() / total_count)
        return ldhge_value

    # 9 Grey Level Non-uniformity
    def _calc_glnu(self, matrix: np.ndarray) -> float:
        """
        Calculate the Gray-Level Non-Uniformity (GLNU) feature from a GLDZM matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Input 2D GLDZM matrix.

        Returns
        -------
        float
            GLNU feature value. Returns empty value if the matrix is empty.
        """
        matrix, row_indices, col_indices, row_grid, col_grid, total_count = self._prepare_matrix(matrix)

        if total_count == 0:
            return self._get_empty_value()

        row_sums: np.ndarray = matrix.sum(axis=1).astype(np.float64)
        glnu_value: float = float((row_sums ** 2).sum() / total_count)
        return glnu_value

    # 10 Normalised Grey Level Non-uniformity
    def _calc_glnu_norm(self, matrix: np.ndarray) -> float:
        """
        Calculate the Normalized Gray-Level Non-Uniformity (GLNU_norm) feature
        from a GLDZM matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Input 2D GLDZM matrix.

        Returns
        -------
        float
            Normalized GLNU feature value. Returns empty value if the matrix is empty.
        """
        matrix, row_indices, col_indices, row_grid, col_grid, total_count = self._prepare_matrix(matrix)

        if total_count == 0:
            return self._get_empty_value()

        row_sums: np.ndarray = matrix.sum(axis=1).astype(np.float64)
        glnu_norm_value: float = float((row_sums ** 2).sum() / (total_count ** 2))
        return glnu_norm_value

    # 11 Zone Distance Non-uniformity
    def _calc_zdnu(self, matrix: np.ndarray) -> float:
        """
        Calculate the Zone Distance Non-Uniformity (ZDNU) feature from a GLDZM matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Input 2D GLDZM matrix.

        Returns
        -------
        float
            ZDNU feature value. Returns empty value if the matrix is empty.
        """
        matrix, row_indices, col_indices, row_grid, col_grid, total_count = self._prepare_matrix(matrix)

        if total_count == 0:
            return self._get_empty_value()

        col_sums: np.ndarray = matrix.sum(axis=0).astype(np.float64)
        zdnu_value: float = float((col_sums ** 2).sum() / total_count)
        return zdnu_value

    # 12 Normalised Zone Distance Non-uniformity
    def _calc_zdnu_norm(self, matrix: np.ndarray) -> float:
        """
        Calculate the Normalized Zone Distance Non-Uniformity (ZDNU_norm) feature
        from a GLDZM matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Input 2D GLDZM matrix.

        Returns
        -------
        float
            Normalized ZDNU feature value. Returns empty value if the matrix is empty.
        """
        matrix, row_indices, col_indices, row_grid, col_grid, total_count = self._prepare_matrix(matrix)

        if total_count == 0:
            return self._get_empty_value()

        col_sums: np.ndarray = matrix.sum(axis=0).astype(np.float64)
        zdnu_norm_value: float = float((col_sums ** 2).sum() / (total_count ** 2))
        return zdnu_norm_value

    # 13 Zone Percentage (needs voxel count)
    def _calc_z_perc(self, matrix: np.ndarray, roi_index: int, slice_index: int | None) -> float:
        """
        Calculate the zone percentage feature (ratio of zone voxels to total ROI voxels).

        Parameters
        ----------
        matrix : np.ndarray
            Input 2D GLDZM matrix.
        roi_index : int
            Index of the ROI to compute total voxels.

        Returns
        -------
        float
            Zone percentage value. Returns empty value according to feature_value_mode.
        """
        matrix, _, _, _, _, total_z = self._prepare_matrix(matrix)
        if total_z == 0:
            return self._get_empty_value()

        roi = self.get_roi(roi_index)  # quantized 3D array (NaN outside)
        if roi is None or roi.size == 0:
            return float(np.nan)

        if slice_index is not None:
            # 2D mode: only voxels on this slice
            roi_slice = roi[:, :, int(slice_index)]
            n_voxels = int(np.count_nonzero(np.isfinite(roi_slice)))
        else:
            # 2.5D / 3D: all voxels in the whole ROI volume
            n_voxels = int(np.count_nonzero(np.isfinite(roi)))

        if n_voxels > 0:
            return float(total_z / n_voxels)
        return self._get_empty_value()

    # 14 Grey Level Variance
    def _calc_gl_var(self, matrix: np.ndarray) -> float:
        """
        Calculate the Gray-Level Variance (GLVar) feature from a GLDZM matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Input 2D GLDZM matrix.

        Returns
        -------
        float
            Gray-Level Variance feature value. Returns empty value if the matrix is empty.
        """
        matrix, row_indices, col_indices, row_grid, col_grid, total_count = self._prepare_matrix(matrix)

        if total_count == 0:
            return self._get_empty_value()

        prob_matrix: np.ndarray = matrix.astype(np.float64) / total_count
        mean_level: float = (row_grid * prob_matrix).sum()
        gl_var_value: float = float(((row_grid - mean_level) ** 2 * prob_matrix).sum())

        return gl_var_value

    # 15 Zone Distance Variance
    def _calc_zd_var(self, matrix: np.ndarray) -> float:
        """
        Calculate the Zone Distance Variance (ZDVar) feature from a GLDZM matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Input 2D GLDZM matrix.

        Returns
        -------
        float
            Zone Distance Variance feature value. Returns empty value if the matrix is empty.
        """
        matrix, row_indices, col_indices, row_grid, col_grid, total_count = self._prepare_matrix(matrix)

        if total_count == 0:
            return self._get_empty_value()

        prob_matrix: np.ndarray = matrix.astype(np.float64) / total_count
        mean_distance: float = (col_grid * prob_matrix).sum()
        zd_var_value: float = float(((col_grid - mean_distance) ** 2 * prob_matrix).sum())

        return zd_var_value

    # 16 Zone Distance Entropy
    def _calc_zd_entr(self, matrix: np.ndarray) -> float:
        """
        Calculate the Zone Distance Entropy (ZD_Entr) feature from a GLDZM matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Input 2D GLDZM matrix.

        Returns
        -------
        float
            Zone Distance Entropy feature value. Returns empty value if the matrix is empty.
        """
        matrix, row_indices, col_indices, row_grid, col_grid, total_count = self._prepare_matrix(matrix)

        if total_count == 0:
            return self._get_empty_value()

        prob_matrix: np.ndarray = matrix.astype(np.float64) / total_count
        small_constant: float = np.finfo(float).eps  # to avoid log2(0)
        zd_entr_value: float = float(-(prob_matrix * np.log2(prob_matrix + small_constant)).sum())

        return zd_entr_value


# ======================================================================
# Dimension-specific extractors
# ======================================================================
class GrayLevelDistanceZoneMatrixFeature2DExtractor(GrayLevelDistanceZoneMatrixFeatureExtractor):
    """GLDZM 2D Extractor (slice-wise average)."""
    NAME: str = "GLDZM2DExtractor"
    _allowed_modes = ["2D"]

    # Inherit everything; 2D uses _mean_over_slices in the base.


class GrayLevelDistanceZoneMatrixFeature25DExtractor(GrayLevelDistanceZoneMatrixFeatureExtractor):
    """GLDZM 2.5D Extractor (sum of 2D matrices then compute)."""
    NAME: str = "GLDZM25DExtractor"
    _allowed_modes = ["2_5D"]

    def _get_m25_feature(self, roi_id: int) -> Optional[np.ndarray]:
        return super()._get_m25_feature(roi_id)


class GrayLevelDistanceZoneMatrixFeature3DExtractor(GrayLevelDistanceZoneMatrixFeatureExtractor):
    """GLDZM 3D Extractor (single 3D matrix)."""
    NAME: str = "GLDZM3DExtractor"
    _allowed_modes = ["3D"]

    def _get_m3_feature(self, roi_id: int) -> Optional[np.ndarray]:
        return super()._get_m3_feature(roi_id)

    def _mean_over_slices(self, roi_id: int, metric_func: Callable[[np.ndarray, int], float]) -> float:
        # 3D mode does not use slice-wise averaging
        return float(np.nan)
