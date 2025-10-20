# -*- coding: utf-8 -*-
# core/extractors/neighbouring_gray_level_dependence_matrix_features_extractor_refactored.py

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, List, Tuple, Callable

import numpy as np

from pysera.engine.visera_oop.core.base_feature_extractor import BaseFeatureExtractor
from pysera.engine.visera_oop.core.sparsity.view_planner import DataView

logger = logging.getLogger("Dev_logger")


class NeighbouringGrayLevelDependenceMatrixFeatureExtractor(BaseFeatureExtractor):
    """
    NGLDM (Neighbouring Gray Level Dependence Matrix) feature extractor
    with dynamic getters and per-dimension cache plumbing.

    Getters follow: get_ngl_<metric>_<mode>(roi_index)
    where <mode> ∈ {2D, 2_5D, 3D}.
    """

    ACCEPTS: DataView = DataView.DENSE_BLOCK
    PREFERS: DataView = DataView.DENSE_BLOCK

    # Neighbourhood radius (Chebyshev) and coarseness parameter (α)
    neighbourhood_radius: int = 1
    coarseness_param: int = 0

    # ------------------------------------------------------------------
    # Dynamic getter: get_ngl_<feature>_<mode>
    # ------------------------------------------------------------------
    def __getattr__(self, name: str) -> Callable[[int], float]:
        """Dynamic getter for NGLDM features (2D, 2.5D, 3D)."""

        prefix = "get_ngl_"
        if not name.startswith(prefix):
            raise AttributeError(f"{type(self).__name__} has no attribute {name}")

        core_name = name[len(prefix):]
        mode: Optional[str] = None
        feature_name: Optional[str] = None

        for m in ("2_5D", "3D", "2D"):
            if core_name.endswith(f"_{m}"):
                mode = m
                feature_name = core_name[: -(len(m) + 1)]
                break

        # Invalid pattern
        if mode is None or feature_name is None:
            logger.error(f"Invalid getter name: {name}")
            return lambda *_: float(np.nan)

        # Check mode allowed by subclass
        if not self._is_mode_allowed(mode):
            logger.error(f"Requested mode '{mode}' not allowed for {self.__class__.__name__}")
            return lambda *_: float(np.nan)

        calc_func = getattr(self, f"_calc_{feature_name}", None)
        if not callable(calc_func):
            logger.error(f"Feature calculator '_calc_{feature_name}' not found.")
            return lambda *_: float(np.nan)

        # Features requiring ROI/slice context
        roi_slice_features = {"dcperc"}

        if feature_name in roi_slice_features:
            def getter(roi_index: int) -> float:
                if mode == "2D":
                    return self._mean_over_slices(roi_index, lambda mat, s: calc_func(mat, roi_index, s))
                elif mode == "2_5D":
                    mat_25d = self._get_m25_feature(roi_index)
                    return calc_func(mat_25d, roi_index, None) if mat_25d is not None else float(np.nan)
                else:  # 3D
                    mat_3d = self._get_m3_feature(roi_index)
                    return calc_func(mat_3d, roi_index, None) if mat_3d is not None else float(np.nan)

            return getter

        # Standard metrics (do not need ROI/slice context)
        def getter(roi_index: int) -> float:
            if mode == "2D":
                return self._mean_over_slices(roi_index, lambda mat, s: calc_func(mat))
            elif mode == "2_5D":
                mat_25d = self._get_m25_feature(roi_index)
                return calc_func(mat_25d) if mat_25d is not None else float(np.nan)
            else:  # 3D
                mat_3d = self._get_m3_feature(roi_index)
                return calc_func(mat_3d) if mat_3d is not None else float(np.nan)

        return getter

    # ------------------------------------------------------------------
    # Feature discovery (without 'get_')
    # ------------------------------------------------------------------
    def _discover_feature_names(self) -> List[str]:

        # collect _calc_* across MRO so subclass overrides are seen
        methods: List[str] = []
        for cls in self.__class__.mro():
            methods.extend(name for name, obj in cls.__dict__.items() if callable(obj))
        metrics = [name[len("_calc_"):] for name in methods if name.startswith("_calc_")]

        dims = getattr(self, "_allowed_modes", []) or ["2D", "2_5D", "3D"]
        names = [f"ngl_{metric}_{dim}" for dim in dims for metric in metrics]

        return names

    @classmethod
    def _is_mode_allowed(cls, mode: str) -> bool:
        return hasattr(cls, "_allowed_modes") and mode in cls._allowed_modes

    # ------------------------------------------------------------------
    # Cache plumbing
    # ------------------------------------------------------------------
    def _ensure_cache(self, roi_id: int) -> None:
        if not hasattr(self, "_ngldm_cache"):
            self._ngldm_cache: Dict[int, Dict[str, Any]] = {}

        roi_q: Optional[np.ndarray] = self.get_roi(roi_id)
        views: Dict[str, Any] = self.get_views(roi_id)

        mask, levels = self._prepare_mask_and_levels(roi_q, views)

        if roi_q is None or roi_q.ndim != 3 or mask is None or levels is None or mask.shape != roi_q.shape:
            self._ngldm_cache[roi_id] = {}
            logger.warning("Invalid inputs for ROI %s; cache left empty.", roi_id)
            return

        mats_2d, mat_25d = self._build_2d_and_25d_matrices(roi_q, mask, levels)
        mat_3d = self._build_3d_matrix(roi_q, mask, levels)

        self._ngldm_cache[roi_id] = {
            "MATRICES_2D": mats_2d,
            "MATRIX_25D": mat_25d,
            "MATRIX_3D": mat_3d,
            "NUM_GRAY_LEVEL_ROWS": len(self._build_global_level_indexer(levels)[0]) - 1,
        }

    def _get_m25_feature(self, roi_id: int) -> Optional[np.ndarray]:
        self._ensure_cache(roi_id)
        return self._ngldm_cache.get(roi_id, {}).get("MATRIX_25D", None)

    def _get_m3_feature(self, roi_id: int) -> Optional[np.ndarray]:
        self._ensure_cache(roi_id)
        return self._ngldm_cache.get(roi_id, {}).get("MATRIX_3D", None)

    # ------------------------------------------------------------------
    # Views → mask/levels
    # ------------------------------------------------------------------
    def _prepare_mask_and_levels(
            self, quant_array: Optional[np.ndarray], views: Dict[str, Any]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if quant_array is None or quant_array.ndim != 3:
            return None, None

        mask_array: Optional[np.ndarray] = views.get("binary_mask")
        levels_array: Optional[np.ndarray] = views.get("levels")

        if mask_array is None:
            mask_array = np.isfinite(quant_array).astype(np.uint8, copy=False)
        if levels_array is None or (isinstance(levels_array, np.ndarray) and levels_array.size == 0):
            levels_array = self._levels_from_quant(quant_array)

        return mask_array, levels_array

    @staticmethod
    def _levels_from_quant(quantized_values: np.ndarray) -> Optional[np.ndarray]:
        u = np.unique(quantized_values[np.isfinite(quantized_values)])
        return u.astype(np.float32, copy=False) if u.size else None

    # ------------------------------------------------------------------
    # Level rounding / indexing
    # ------------------------------------------------------------------
    @staticmethod
    def _round_levels(values_array: np.ndarray, scale_factor: int) -> np.ndarray:
        """Round values to discrete levels using a given scale factor.

        Parameters
        ----------
        values_array : np.ndarray
            Array of float values to be rounded.
        scale_factor : int
            Factor used to scale before rounding and then divide back.

        Returns
        -------
        np.ndarray
            Rounded array of the same shape as input.
        """
        rounded_array = np.asarray(values_array, dtype=np.float32).copy()
        rounded_array *= scale_factor
        np.rint(rounded_array, out=rounded_array)
        rounded_array /= scale_factor
        return rounded_array

    def _build_global_level_indexer(
            self, levels: np.ndarray
    ) -> Tuple[np.ndarray, Dict[float, int], int]:
        n = int(levels.size)
        scale = 10000 if n > 100 else 1000
        sentinel = float(np.max(levels) + 1.0)
        with_sentinel = np.append(levels.astype(np.float32, copy=False), sentinel)
        rounded = self._round_levels(with_sentinel, scale)
        mapping = {float(v): i for i, v in enumerate(rounded)}
        return rounded, mapping, scale

    # ------------------------------------------------------------------
    # NGLDM builders
    # ------------------------------------------------------------------
    def _prepare_rounded_box(self, roi_box: np.ndarray, uniq_levels: np.ndarray, adjust: int) -> np.ndarray:
        sentinel = float(uniq_levels[-1])
        filled = np.nan_to_num(roi_box, nan=sentinel).astype(np.float32, copy=False)
        return self._round_levels(filled, adjust)

    def _compute_ngldm_global_rows(
            self, roi_array: np.ndarray, roi_mask: np.ndarray, levels_array: np.ndarray
    ) -> np.ndarray:

        """Compute NG-LDM (neighbourhood gray-level dependence matrix) for the ROI."""

        # Build global indexer + round box exactly as before
        unique_levels, level_to_row, adjust = self._build_global_level_indexer(levels_array)

        rounded_array = self._prepare_rounded_box(roi_array, unique_levels, adjust)
        sentinel_value = float(unique_levels[-1])

        num_rows = len(unique_levels) - 1
        r = int(self.neighbourhood_radius)
        nd = roi_array.ndim

        # Valid centres: inside mask, finite, not sentinel
        roi_mask_binary = (roi_mask > 0.5)
        centre_valid = roi_mask_binary & np.isfinite(rounded_array) & (rounded_array != sentinel_value)

        # Count same-level neighbours (excluding centre) via shifted equality
        dep_same = np.zeros_like(rounded_array, dtype=np.uint16)

        # runner = RuntimeRAMLogger()
        if nd == 2:
            H, W = rounded_array.shape
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    if di == 0 and dj == 0:
                        continue
                    si0, di0 = (max(0, -di), max(0, di))
                    sj0, dj0 = (max(0, -dj), max(0, dj))
                    si1, di1 = (H - max(0, di), H - max(0, -di))
                    sj1, dj1 = (W - max(0, dj), W - max(0, -dj))
                    src = (slice(si0, si1), slice(sj0, sj1))
                    dst = (slice(di0, di1), slice(dj0, dj1))

                    # neighbour must be valid inside ROI and not sentinel; centre must be valid
                    vsrc = roi_mask_binary[src] & np.isfinite(rounded_array[src]) & (
                            rounded_array[src] != sentinel_value)
                    vdst = centre_valid[dst]

                    eq = (rounded_array[src] == rounded_array[dst]) & vsrc & vdst
                    dep_same[dst] += eq

        else:  # 3D
            H, W, D = rounded_array.shape
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    for dk in range(-r, r + 1):
                        if di == 0 and dj == 0 and dk == 0:
                            continue
                        si0, di0 = (max(0, -di), max(0, di))
                        sj0, dj0 = (max(0, -dj), max(0, dj))
                        sk0, dk0 = (max(0, -dk), max(0, dk))
                        si1, di1 = (H - max(0, di), H - max(0, -di))
                        sj1, dj1 = (W - max(0, dj), W - max(0, -dj))
                        sk1, dk1 = (D - max(0, dk), D - max(0, -dk))
                        src = (slice(si0, si1), slice(sj0, sj1), slice(sk0, sk1))
                        dst = (slice(di0, di1), slice(dj0, dj1), slice(dk0, dk1))

                        vsrc = roi_mask_binary[src] & np.isfinite(rounded_array[src]) & (
                                rounded_array[src] != sentinel_value)
                        vdst = centre_valid[dst]

                        eq = (rounded_array[src] == rounded_array[dst]) & vsrc & vdst
                        dep_same[dst] += eq

        # Dependence includes the centre --> +1 where centre is valid
        dep_total = np.zeros_like(dep_same, dtype=np.uint16)
        dep_total[centre_valid] = dep_same[centre_valid] + 1

        # Map centre gray levels to row indices (0..num_rows-1); ignore sentinel
        centres = rounded_array[centre_valid]
        pos = np.searchsorted(unique_levels, centres, side="left")
        # exclude sentinel row
        row_idx = pos[pos < num_rows]
        dep_vals = dep_total[centre_valid][pos < num_rows]

        # Columns are j-1 (j >= 1). Guarantee at least 1 column as in the original.
        max_dep = int(dep_vals.max(initial=0))
        if max_dep < 1:
            max_dep = 1

        # Prepare integer indices
        row_idx_i64 = row_idx.astype(np.int64)
        col_idx_i64 = np.clip(dep_vals.astype(np.int64) - 1, 0, max_dep - 1)

        # Flatten 2D indices to 1D raveled index
        raveled_idx = row_idx_i64 * np.int64(max_dep) + col_idx_i64

        # Use bincount to accumulate counts
        if raveled_idx.size == 0:
            matrix = np.zeros((num_rows, max_dep), dtype=np.int64)
        else:
            counts = np.bincount(raveled_idx, minlength=int(num_rows) * int(max_dep))
            matrix = counts.reshape((num_rows, max_dep)).astype(np.int64, copy=False)

        return matrix

    def _build_2d_and_25d_matrices(
            self, roi_array: np.ndarray, roi_mask: np.ndarray, levels_array: np.ndarray
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Compute per-slice 2D NG-LDM matrices and aggregated 2.5D NG-LDM matrix."""

        num_slices = roi_array.shape[2]
        unique_levels, _, _ = self._build_global_level_indexer(levels_array)
        num_rows = len(unique_levels) - 1

        matrices_2d: List[np.ndarray] = []
        max_cols = 0

        for slice_idx in range(num_slices):
            slice_quant = roi_array[:, :, slice_idx]
            slice_mask = roi_mask[:, :, slice_idx]

            if np.count_nonzero(slice_mask) == 0:
                matrix_2d = np.zeros((num_rows, 1), dtype=np.int64)
            else:
                matrix_2d = self._compute_ngldm_global_rows(slice_quant, slice_mask, levels_array)

            matrices_2d.append(matrix_2d)
            max_cols = max(max_cols, matrix_2d.shape[1])

        if max_cols > 0:
            matrix_25d = np.zeros((num_rows, max_cols), dtype=np.int64)
            for matrix_2d in matrices_2d:
                matrix_25d[:, : matrix_2d.shape[1]] += matrix_2d
        else:
            matrix_25d = np.zeros((num_rows, 0), dtype=np.int64)

        return matrices_2d, matrix_25d

    def _build_3d_matrix(
            self, roi_array: np.ndarray, roi_mask: np.ndarray, levels_array: np.ndarray
    ) -> np.ndarray:
        """Compute NG-LDM for the entire 3D ROI volume."""

        matrix_3d = self._compute_ngldm_global_rows(roi_array, roi_mask, levels_array)
        return matrix_3d

    # ------------------------------------------------------------------
    # 2D aggregation helper
    # ------------------------------------------------------------------
    def _mean_over_slices(
            self, roi_index: int, metric_function: Callable[[np.ndarray, int], float]
    ) -> float:
        """Compute the mean of a metric across all 2D slices of an ROI."""

        self._ensure_cache(roi_index)
        matrices_2d: List[np.ndarray] = self._ngldm_cache.get(roi_index, {}).get("MATRICES_2D", [])

        if not matrices_2d:
            return float(np.nan)

        values: List[float] = []
        for slice_index, matrix in enumerate(matrices_2d):
            val = metric_function(matrix, slice_index)
            values.append(val if np.isfinite(val) else float(np.nan))

        return float(np.nanmean(values)) if values else float(np.nan)

    # ------------------------------------------------------------------
    # Matrix utilities for metrics
    # ------------------------------------------------------------------
    @staticmethod
    def _trim_trailing_zero_cols(matrix: np.ndarray) -> np.ndarray:
        """Remove trailing columns that are all zeros."""

        if matrix.size == 0:
            return matrix

        column_sums = matrix.sum(axis=0)
        nonzero_indices = np.where(column_sums > 0)[0]

        if nonzero_indices.size == 0:
            return matrix[:, :1]

        return matrix[:, : nonzero_indices[-1] + 1]

    @staticmethod
    def _get_total_count(matrix: np.ndarray) -> int:
        return int(matrix.sum())

    @staticmethod
    def _get_row_and_column_sums(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return matrix.sum(axis=1).astype(np.float64), matrix.sum(axis=0).astype(np.float64)

    @staticmethod
    def _get_indices_and_grids(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return row indices, column indices, and corresponding 2D grids for a matrix."""

        num_rows, num_cols = matrix.shape
        row_indices = np.arange(1, num_rows + 1, dtype=np.float64)
        col_indices = np.arange(1, num_cols + 1, dtype=np.float64)

        row_grid, col_grid = np.meshgrid(row_indices, col_indices, indexing="ij")

        return row_indices, col_indices, row_grid, col_grid

    def _get_empty_value(self) -> float:
        return 0.0 if str(getattr(self, "feature_value_mode", "")).upper() == "APPROXIMATE_VALUE" else float(np.nan)

    def _prepare_matrix(
            self, matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """Trim matrix, compute total, and prepare row/column indices and grids."""

        # Remove trailing zero columns
        matrix_trimmed = self._trim_trailing_zero_cols(matrix)

        # Total count (sum of all elements)
        total_count = self._get_total_count(matrix_trimmed)

        # Initialize empty arrays
        row_indices = col_indices = row_grid = col_grid = np.array([], dtype=np.float64)

        # Only compute indices/grids if matrix is non-empty and has positive total
        if total_count > 0 and matrix_trimmed.size > 0:
            row_indices, col_indices, row_grid, col_grid = self._get_indices_and_grids(matrix_trimmed)

        return matrix_trimmed, row_indices, col_indices, row_grid, col_grid, total_count

    # ------------------------------------------------------------------
    # Feature computation methods
    # ------------------------------------------------------------------
    def _calc_lde(self, matrix: np.ndarray) -> float:
        """Low dependence emphasis (NGLDM small number emphasis)【701412440027074†L5305-L5326】."""
        matrix, row_indices, col_indices, row_grid, col_grid, total_count = self._prepare_matrix(matrix)
        if total_count == 0:
            return self._get_empty_value()
        _, col_sums = self._get_row_and_column_sums(matrix)
        return float((col_sums / (col_indices ** 2)).sum() / total_count)

    def _calc_hde(self, matrix: np.ndarray) -> float:
        """High dependence emphasis (large number emphasis)【701412440027074†L5327-L5348】."""
        matrix, row_indices, col_indices, row_grid, col_grid, total_count = self._prepare_matrix(matrix)
        if total_count == 0:
            return self._get_empty_value()
        _, col_sums = self._get_row_and_column_sums(matrix)
        return float((col_sums * (col_indices ** 2)).sum() / total_count)

    def _calc_lgce(self, matrix: np.ndarray) -> float:
        """Low gray level count emphasis【701412440027074†L5350-L5369】."""
        matrix, row_indices, col_indices, row_grid, col_grid, total_count = self._prepare_matrix(matrix)
        if total_count == 0:
            return self._get_empty_value()
        row_sums, _ = self._get_row_and_column_sums(matrix)
        return float((row_sums / (row_indices ** 2)).sum() / total_count)

    def _calc_hgce(self, matrix: np.ndarray) -> float:
        """High gray level count emphasis【701412440027074†L5371-L5391】."""
        matrix, row_indices, col_indices, row_grid, col_grid, total_count = self._prepare_matrix(matrix)
        if total_count == 0:
            return self._get_empty_value()
        row_sums, _ = self._get_row_and_column_sums(matrix)
        return float((row_sums * (row_indices ** 2)).sum() / total_count)

    def _calc_ldlge(self, matrix: np.ndarray) -> float:
        """Low dependence low gray level emphasis【701412440027074†L5392-L5413】."""
        matrix, row_indices, col_indices, row_grid, col_grid, total_count = self._prepare_matrix(matrix)
        if total_count == 0:
            return self._get_empty_value()
        return float((matrix / ((row_grid ** 2) * (col_grid ** 2))).sum() / total_count)

    def _calc_ldhge(self, matrix: np.ndarray) -> float:
        """Low dependence high gray level emphasis【701412440027074†L5415-L5436】."""
        matrix, row_indices, col_indices, row_grid, col_grid, total_count = self._prepare_matrix(matrix)
        if total_count == 0:
            return self._get_empty_value()
        return float(((matrix * (row_grid ** 2)) / (col_grid ** 2)).sum() / total_count)

    def _calc_hdlge(self, matrix: np.ndarray) -> float:
        """High dependence low gray level emphasis【701412440027074†L5438-L5459】."""
        matrix, row_indices, col_indices, row_grid, col_grid, total_count = self._prepare_matrix(matrix)
        if total_count == 0:
            return self._get_empty_value()
        return float(((matrix * (col_grid ** 2)) / (row_grid ** 2)).sum() / total_count)

    def _calc_hdhge(self, matrix: np.ndarray) -> float:
        """High dependence high gray level emphasis【701412440027074†L5461-L5483】."""
        matrix, row_indices, col_indices, row_grid, col_grid, total_count = self._prepare_matrix(matrix)
        if total_count == 0:
            return self._get_empty_value()
        return float((matrix * (row_grid ** 2) * (col_grid ** 2)).sum() / total_count)

    def _calc_glnu(self, matrix: np.ndarray) -> float:
        """Gray level non‑uniformity【701412440027074†L5485-L5504】."""
        matrix, row_indices, col_indices, row_grid, col_grid, total_count = self._prepare_matrix(matrix)
        if total_count == 0:
            return self._get_empty_value()
        row_sums, _ = self._get_row_and_column_sums(matrix)
        return float((row_sums ** 2).sum() / total_count)

    def _calc_glnu_norm(self, matrix: np.ndarray) -> float:
        """Normalised gray level non‑uniformity【701412440027074†L5507-L5529】."""
        matrix, row_indices, col_indices, row_grid, col_grid, total_count = self._prepare_matrix(matrix)
        if total_count == 0:
            return self._get_empty_value()
        row_sums, _ = self._get_row_and_column_sums(matrix)
        return float((row_sums ** 2).sum() / (total_count ** 2))

    def _calc_dcnu(self, matrix: np.ndarray) -> float:
        """Dependence count non‑uniformity【701412440027074†L5532-L5553】."""
        matrix, row_indices, col_indices, row_grid, col_grid, total_count = self._prepare_matrix(matrix)
        if total_count == 0:
            return self._get_empty_value()
        _, col_sums = self._get_row_and_column_sums(matrix)
        return float((col_sums ** 2).sum() / total_count)

    def _calc_dcnu_norm(self, matrix: np.ndarray) -> float:
        """Normalised dependence count non‑uniformity【701412440027074†L5555-L5574】."""
        matrix, row_indices, col_indices, row_grid, col_grid, total_count = self._prepare_matrix(matrix)
        if total_count == 0:
            return self._get_empty_value()
        _, col_sums = self._get_row_and_column_sums(matrix)
        return float((col_sums ** 2).sum() / (total_count ** 2))

    def _calc_dcperc(self, matrix: np.ndarray, roi_index: int, slice_index: Optional[int]) -> float:
        """Dependence count percentage【701412440027074†L5576-L5597】.

        This feature measures the fraction of realised neighbourhoods with
        respect to the maximum number of possible neighbourhoods.  Under
        IBSI’s definition every voxel inside the mask constitutes a
        neighbourhood【701412440027074†L5230-L5233】, so this feature always equals one, but
        the calculation is performed explicitly for completeness.
        """
        matrix, row_indices, col_indices, row_grid, col_grid, total_count = self._prepare_matrix(matrix)
        if total_count == 0:
            return self._get_empty_value()
        # total_count corresponds to N_s, the number of neighbourhoods
        roi = self.get_roi(roi_index)
        if roi is None or roi.size == 0:
            return float(np.nan)
        if slice_index is not None:
            roi_slice = roi[:, :, int(slice_index)]
            n_voxels = int(np.count_nonzero(np.isfinite(roi_slice)))
        else:
            n_voxels = int(np.count_nonzero(np.isfinite(roi)))
        return float(total_count / n_voxels) if n_voxels > 0 else self._get_empty_value()

    def _calc_gl_var(self, matrix: np.ndarray) -> float:
        """Gray level variance【701412440027074†L5599-L5623】."""
        matrix, row_indices, col_indices, row_grid, col_grid, total_count = self._prepare_matrix(matrix)
        if total_count == 0:
            return self._get_empty_value()
        prob_matrix = matrix.astype(np.float64) / total_count
        mean_level = (row_grid * prob_matrix).sum()
        return float(((row_grid - mean_level) ** 2 * prob_matrix).sum())

    def _calc_dc_var(self, matrix: np.ndarray) -> float:
        """Dependence count variance【701412440027074†L5626-L5649】."""
        matrix, row_indices, col_indices, row_grid, col_grid, total_count = self._prepare_matrix(matrix)
        if total_count == 0:
            return self._get_empty_value()
        prob_matrix = matrix.astype(np.float64) / total_count
        mean_dep = (col_grid * prob_matrix).sum()
        return float(((col_grid - mean_dep) ** 2 * prob_matrix).sum())

    def _calc_dc_entr(self, matrix: np.ndarray) -> float:
        """Dependence count entropy【701412440027074†L5651-L5675】."""
        matrix, row_indices, col_indices, row_grid, col_grid, total_count = self._prepare_matrix(matrix)
        if total_count == 0:
            return self._get_empty_value()
        prob_matrix = matrix.astype(np.float64) / total_count
        eps = np.finfo(float).eps
        return float(-(prob_matrix * np.log2(prob_matrix + eps)).sum())

    def _calc_dc_energy(self, matrix: np.ndarray) -> float:
        """Dependence count energy (second moment)【701412440027074†L5677-L5701】."""
        matrix, row_indices, col_indices, row_grid, col_grid, total_count = self._prepare_matrix(matrix)
        if total_count == 0:
            return self._get_empty_value()
        prob_matrix = matrix.astype(np.float64) / total_count
        return float((prob_matrix ** 2).sum())


# ======================================================================
# Dimension-specific extractors
# ======================================================================
class NeighbouringGrayLevelDependenceMatrixFeature2DExtractor(NeighbouringGrayLevelDependenceMatrixFeatureExtractor):
    """NGLDM 2D Extractor (slice-wise average)."""
    NAME: str = "NGLDM2DExtractor"
    _allowed_modes = ["2D"]


class NeighbouringGrayLevelDependenceMatrixFeature25DExtractor(NeighbouringGrayLevelDependenceMatrixFeatureExtractor):
    """NGLDM 2.5D Extractor (sum 2D matrices then compute)."""
    NAME: str = "NGLDM25DExtractor"
    _allowed_modes = ["2_5D"]

    def _get_m25_feature(self, roi_id: int) -> Optional[np.ndarray]:
        return super()._get_m25_feature(roi_id)


class NeighbouringGrayLevelDependenceMatrixFeature3DExtractor(NeighbouringGrayLevelDependenceMatrixFeatureExtractor):
    """NGLDM 3D Extractor (single 3D matrix)."""
    NAME: str = "NGLDM3DExtractor"
    _allowed_modes = ["3D"]

    def _get_m3_feature(self, roi_id: int) -> Optional[np.ndarray]:
        return super()._get_m3_feature(roi_id)

    # 3D does not use slice-wise averaging, but keep signature consistent
    def _mean_over_slices(
            self, roi_index: int, metric_function: Callable[[np.ndarray, int], float]
    ) -> float:
        return float(np.nan)
