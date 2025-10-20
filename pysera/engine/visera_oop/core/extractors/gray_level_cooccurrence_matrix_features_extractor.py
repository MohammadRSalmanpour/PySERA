# -*- coding: utf-8 -*-
# core/extractors/gray_level_cooccurrence_matrix_features_extractor.py

from __future__ import annotations

import logging
import numpy as np
from functools import lru_cache
from typing import Any, Dict, Optional, List, Tuple, Callable

from pysera.engine.visera_oop.core.sparsity.view_planner import DataView
from pysera.engine.visera_oop.core.base_feature_extractor import BaseFeatureExtractor

logger = logging.getLogger("Dev_logger")


@lru_cache(maxsize=1)
def _cached_offsets_2d() -> np.ndarray:
    # (dy, dx) for 0°, 45°, 90°, 135°, distance=1
    return np.asarray(
        [
            [0, 1],     # 0°
            [-1, 1],    # 45°
            [-1, 0],    # 90°
            [-1, -1],   # 135°
        ],
        dtype=np.int32,
    )


@lru_cache(maxsize=1)
def _cached_offsets_3d() -> np.ndarray:
    # 13 unique 3D offsets (IBSI-consistent), distance=1
    return np.asarray(
        [
            [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 1, 0], [1, -1, 0],
            [1, 0, 1], [1, 0, -1],
            [0, 1, 1], [0, 1, -1],
            [1, 1, 1], [1, 1, -1],
            [1, -1, 1], [1, -1, -1],
        ],
        dtype=np.int32,
    )


@lru_cache(maxsize=64)
def _cached_ij_grids(grid_size: int) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(1, grid_size + 1, dtype=np.float64)
    return np.meshgrid(idx, idx, indexing="ij")


@lru_cache(maxsize=64)
def _cached_sum_k_indices(grid_size: int) -> np.ndarray:
    # 2, 3, ..., (2 x grid_size)
    return np.arange(2, 2 * grid_size + 1, dtype=np.float64)


class GrayLevelCooccurrenceMatrixFeaturesExtractor(BaseFeatureExtractor):
    """GLCM feature extractor with dynamic getters for multiple dimensional modes (2D, 2.5D, 3D)."""

    ACCEPTS: DataView = DataView.DENSE_BLOCK
    PREFERS: DataView = DataView.DENSE_BLOCK

    def __getattr__(self, name: str) -> Callable[[int], float]:
        """Return a callable for a dynamically resolved GLCM feature (mode-specific)."""
        if not name.startswith("get_glcm_"):
            raise AttributeError(f"{type(self).__name__} has no attribute {name}")
        core = name[len("get_glcm_"):]
        mode: Optional[str] = None
        feature: Optional[str] = None
        # Check for mode suffixes including aggregation
        for m in ("2D_avg", "2D_comb", "2_5D_avg", "2_5D_comb", "3D_avg", "3D_comb"):
            if core.endswith(f"_{m}"):
                mode = m
                feature = core[: -(len(m) + 1)]
                break
        if mode is None or feature is None:
            logger.error("Invalid GLCM feature name: %s", name)
            return lambda *_args, **_kwargs: float(np.nan)

        calc_func = getattr(self, f"_calc_{feature}", None)
        if not callable(calc_func):
            logger.error("GLCM feature '%s' is not implemented.", feature)
            return lambda *_args, **_kwargs: float(np.nan)

        def getter(roi_index: int) -> float:
            # Ensure GLCM matrices are computed for this ROI
            self._ensure_cache(roi_index)
            try:
                cache = self._glcm_cache.get(roi_index, {})
                if mode == "2D_avg":
                    # Average over direction-specific per-slice 2D GLCMs (strict IBSI 2D-avg)
                    glcm_matrices = self._glcm_cache.get(roi_index, {}).get("glcm_matrices_2d_dirs", [])
                    if not glcm_matrices:
                        return float(np.nan)

                    values = []
                    for matrix in glcm_matrices:
                        try:
                            values.append(calc_func(matrix))
                        except Exception as error:
                            logger.error(
                                "2D_avg feature computation failed for ROI %d: %s", roi_index, error
                            )
                            values.append(np.nan)

                    return float(np.nanmean(values)) if values else float(np.nan)

                if mode == "2D_comb":
                    # Combine and average over all 2D GLCMs computed per slice (IBSI 2D-combined)
                    glcm_matrices = cache.get("glcm_matrices_2d", [])
                    if not glcm_matrices:
                        return float(np.nan)

                    values = []
                    for matrix in glcm_matrices:
                        try:
                            values.append(calc_func(matrix))
                        except Exception as error:
                            logger.error(
                                "2D_comb feature computation failed for ROI %d: %s", roi_index, error
                            )
                            values.append(np.nan)

                    return float(np.nanmean(values)) if values else float(np.nan)

                if mode == "2_5D_avg":
                    # Average over direction-specific 2.5D GLCMs (IBSI 2.5D-avg)
                    glcm_matrices = cache.get("glcm_matrices_25d_dirs", [])
                    if not glcm_matrices:
                        return float(np.nan)

                    values = []
                    for matrix in glcm_matrices:
                        try:
                            values.append(calc_func(matrix))
                        except Exception as error:
                            logger.error(
                                "2_5D_avg feature computation failed for ROI %d: %s",
                                roi_index,
                                error,
                            )
                            values.append(np.nan)

                    return float(np.nanmean(values)) if values else float(np.nan)

                if mode == "2_5D_comb":
                    # Compute feature from the combined 2.5D GLCM (IBSI 2.5D-combined)
                    glcm_matrix = cache.get("glcm_matrices_25d")
                    if isinstance(glcm_matrix, np.ndarray) and glcm_matrix.size:
                        return float(calc_func(glcm_matrix))
                    return float(np.nan)

                if mode == "3D_avg":
                    # Average over direction-specific 3D GLCMs (IBSI 3D-avg)
                    glcm_matrices = cache.get("glcm_matrices_3d_dirs", [])
                    if not glcm_matrices:
                        return float(np.nan)

                    values = []
                    for matrix in glcm_matrices:
                        try:
                            values.append(calc_func(matrix))
                        except Exception as error:
                            logger.error(
                                "3D_avg feature computation failed for ROI %d: %s",
                                roi_index,
                                error,
                            )
                            values.append(np.nan)

                    return float(np.nanmean(values)) if values else float(np.nan)

                # 3D_comb
                glcm_matrix = cache.get("glcm_matrices_3d")
                if isinstance(glcm_matrix, np.ndarray) and glcm_matrix.size:
                    return float(calc_func(glcm_matrix))
                return float(np.nan)

            except Exception as e:
                logger.error("Error computing '%s' for ROI %d: %s", feature, roi_index, e)
                return float(np.nan)

        return getter

    def _discover_feature_names(self) -> List[str]:
        methods: List[str] = []
        for cls in self.__class__.mro():
            methods.extend(name for name, obj in cls.__dict__.items() if callable(obj))
        metric_names = [name[len("_calc_"):] for name in methods if name.startswith("_calc_")]
        dimensions = getattr(self, "_allowed_modes", [])
        return [f"glcm_{metric}_{dim}" for dim in dimensions for metric in metric_names]

    def _ensure_cache(self, roi_id: int) -> None:
        """
        Build & cache (once):
          - 2D: per-slice merged GLCMs
          - 2D_avg: per-slice, per-direction GLCMs
          - 2.5D: merged + 4 per-direction (across slices)
          - 3D: merged + 13 per-direction
        """
        if not hasattr(self, "_glcm_cache"):
            self._glcm_cache: Dict[int, Dict[str, Any]] = {}
        if roi_id in self._glcm_cache:
            return

        quant: Optional[np.ndarray] = self.get_roi(roi_id)
        views: Dict[str, Any] = self.get_views(roi_id)
        mask, levels = self._prepare_mask_and_levels(quant, views)

        if quant is None or quant.ndim != 3 or mask is None or levels is None or mask.shape != quant.shape:
            self._glcm_cache[roi_id] = {}
            logger.warning("[GLCM] Invalid or missing data for ROI %d.", roi_id)
            return

        # ---- 2D & 2.5D in one pass (plus per-direction per-slice for 2D_avg) ----
        mats_2d, m25_comb, m25_dirs, mats_2d_dirs = self._build_2d_and_25d_glcms(quant, mask, levels)

        # ---- 3D merged + 13 dirs in one pass ----
        m3_comb, m3_dirs = self._build_3d_glcm(quant, mask, levels)

        self._glcm_cache[roi_id] = {
            "glcm_matrices_2d": mats_2d,                 # list per slice (merged dirs), normalized
            "glcm_matrices_2d_dirs": mats_2d_dirs,       # list per-slice, per-dir normalized
            "glcm_matrices_25d": m25_comb,               # merged across slices & dirs, normalized
            "glcm_matrices_25d_dirs": m25_dirs,          # list[4] per in-plane dir (across slices), normalized
            "glcm_matrices_3d": m3_comb,                 # merged 13 dirs, normalized
            "glcm_matrices_3d_dirs": m3_dirs,            # list[13] per dir, normalized
        }

    @staticmethod
    def _prepare_mask_and_levels(
            quant_array: Optional[np.ndarray], views: Dict[str, Any]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if quant_array is None or quant_array.ndim != 3:
            return None, None

        mask: Optional[np.ndarray] = views.get("binary_mask")
        levels: Optional[np.ndarray] = views.get("levels")

        if mask is None:
            mask = np.isfinite(quant_array)

        if levels is None or (isinstance(levels, np.ndarray) and levels.size == 0):
            finite_vals = quant_array[np.isfinite(quant_array)]
            unique_levels = np.unique(finite_vals) if finite_vals.size else np.array([])
            levels = unique_levels.astype(np.float32) if unique_levels.size else None

        return mask, levels

    @staticmethod
    def _offsets_2d() -> np.ndarray:
        return _cached_offsets_2d()

    @staticmethod
    def _offsets_3d() -> np.ndarray:
        return _cached_offsets_3d()

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        total_count = matrix.sum(dtype=np.float64)
        if total_count <= 0.0:
            return np.zeros_like(matrix, dtype=np.float64)
        return matrix.astype(np.float64, copy=False) / total_count

    @staticmethod
    def _quant_to_index(arr: np.ndarray, levels: np.ndarray) -> np.ndarray:
        """
        Map array values to 1..N indices based on `levels`, vectorized and float-safe.
        Values outside the ROI are mapped to 0.

        Parameters
        ----------
        arr : np.ndarray
            Input array containing values to be mapped.
        levels : np.ndarray
            Array of reference levels. Must be 1D.

        Returns
        -------
        np.ndarray
            Array of same shape as `arr`, containing integer indices 1..N,
            or 0 for values outside the levels.
        """
        # Choose scaling factor to avoid floating-point issues
        scale = 10000 if levels.size > 100 else 1000

        # Round levels and input array to the scaled float precision
        scaled_levels = np.rint(levels.astype(np.float32) * scale) / scale
        scaled_arr = np.rint(arr.astype(np.float32) * scale) / scale

        # Sort levels and keep the sort order
        sort_order = np.argsort(scaled_levels)
        sorted_levels = scaled_levels[sort_order]

        # Find positions where each array value would be inserted
        positions = np.searchsorted(sorted_levels, scaled_arr)
        inside = positions < sorted_levels.size

        matched = np.zeros(arr.shape, dtype=np.int32)

        if inside.any():
            # Identify exact matches
            hits = inside & (
                    scaled_arr == sorted_levels[np.clip(positions, 0, sorted_levels.size - 1)]
            )

            # Compute inverse permutation to map back to original levels
            inv_order = np.empty_like(sort_order)
            inv_order[sort_order] = np.arange(sort_order.size, dtype=sort_order.dtype)

            # Assign indices (1-based), 0 for unmatched values
            matched = np.where(
                hits,
                inv_order[np.clip(positions, 0, inv_order.size - 1)] + 1,
                0,
            ).astype(np.int32)

        return matched

    @staticmethod
    def _crop_valid_pairs(indexed_image: np.ndarray, offset: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract aligned pairs of central and neighbor indices given an offset (2D or 3D).
        """
        offset_list = offset.tolist()
        src_slices, dst_slices = [], []
        for axis, delta in enumerate(offset_list):
            if delta >= 0:
                src_slices.append(slice(0, indexed_image.shape[axis] - delta))
                dst_slices.append(slice(delta, indexed_image.shape[axis]))
            else:
                src_slices.append(slice(-delta, indexed_image.shape[axis]))
                dst_slices.append(slice(0, indexed_image.shape[axis] + delta))

        central_indices = indexed_image[tuple(src_slices)]
        neighbor_indices = indexed_image[tuple(dst_slices)]

        return central_indices.ravel(order="F"), neighbor_indices.ravel(order="F")

    @staticmethod
    def _accumulate_pairs(i_values: np.ndarray, j_values: np.ndarray, n_levels: int, symmetric: bool) -> np.ndarray:
        """
        Fast accumulation of co-occurrence counts with np.bincount.
        """
        if i_values.size == 0:
            return np.zeros((n_levels, n_levels), dtype=np.int64)

        valid = (i_values > 0) & (j_values > 0)
        if not np.any(valid):
            return np.zeros((n_levels, n_levels), dtype=np.int64)

        row_idx = (i_values[valid] - 1).astype(np.int64)
        col_idx = (j_values[valid] - 1).astype(np.int64)

        flat_indices = row_idx * n_levels + col_idx
        counts = np.bincount(flat_indices, minlength=n_levels * n_levels)
        # Reshape counts into a square matrix
        matrix = counts.reshape(n_levels, n_levels)

        # Make the matrix symmetric if requested
        if symmetric:
            matrix = matrix + matrix.T

        return matrix

    def _build_2d_and_25d_glcms(
        self, quant: np.ndarray, mask: np.ndarray, levels: np.ndarray
    ) -> Tuple[List[np.ndarray], np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        Compute in one pass:
          - per-slice merged 2D GLCMs (normalized)
          - merged 2.5D GLCM (normalized)
          - 4 per-direction 2.5D GLCMs across slices (normalized)
          - per-slice, per-direction 2D GLCMs (normalized) -> for 2D_avg
        """
        n_levels  = int(levels.size)
        mats_2d: List[np.ndarray] = []
        mats_2d_dirs: List[np.ndarray] = []
        counts_25d_comb = np.zeros((n_levels , n_levels ), dtype=np.int64)
        counts_25d_dirs = [np.zeros((n_levels , n_levels ), dtype=np.int64) for _ in range(4)]

        # Get image dimensions: Height, Width, Slices
        height, width, num_slices = quant.shape

        # Retrieve 2D offsets for GLCM directions
        offsets_2d = self._offsets_2d()

        for slice_idx  in range(num_slices):
            slice_mask = mask[:, :, slice_idx ]

            if np.count_nonzero(slice_mask) == 0:
                # Keep placeholder for empty slice to maintain consistent slice indexing
                mats_2d.append(np.zeros((n_levels, n_levels), dtype=np.float64))
                continue

            idx = self._quant_to_index(quant[:, :, slice_idx ], levels)
            np.multiply(idx, (slice_mask > 0).astype(np.int32), out=idx)

            # Per-direction counts (slice), also accumulate into 2.5D dir counts
            counts_slice_dirs = []
            # Compute per-direction counts for the slice and accumulate for 2.5D
            for direction_idx, offset in enumerate(offsets_2d):
                row_indices, col_indices = self._crop_valid_pairs(idx, offset)

                counts_dir = self._accumulate_pairs(row_indices, col_indices, n_levels, symmetric=True)

                counts_slice_dirs.append(counts_dir)
                counts_25d_dirs[direction_idx] += counts_dir

            # Sum slice directions -> merged 2D slice, normalize & store
            c_slice_merged = sum(counts_slice_dirs)
            mats_2d.append(self._normalize(c_slice_merged))

            # Per-slice, per-direction normalized GLCMs for 2D_avg
            mats_2d_dirs.extend(self._normalize(counts_dir) for counts_dir in counts_slice_dirs)

            # Aggregate for 2.5D merged
            counts_25d_comb += c_slice_merged

        # Normalize 2.5D products
        m25_comb = self._normalize(counts_25d_comb)
        m25_dirs = [self._normalize(counts_dir) for counts_dir in counts_25d_dirs]

        return mats_2d, m25_comb, m25_dirs, mats_2d_dirs

    def _build_3d_glcm(
        self, quant: np.ndarray, mask: np.ndarray, levels: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Compute merged 3D GLCM and 13 direction-specific GLCMs in a single pass.
        """
        n_levels = int(levels.size)

        # Return empty matrix if the mask has no nonzero elements
        if np.count_nonzero(mask) == 0:
            matrix_size = max(1, n_levels)
            return np.zeros((matrix_size, matrix_size), dtype=np.float64), []

        idx = self._quant_to_index(quant, levels)
        np.multiply(idx, (mask > 0).astype(np.int32), out=idx)

        counts_dirs: List[np.ndarray] = []
        counts_merged = np.zeros((n_levels, n_levels), dtype=np.int64)
        # Compute per-direction 3D GLCMs and aggregate for merged 3D
        for offset in self._offsets_3d():
            row_indices, col_indices = self._crop_valid_pairs(idx, offset)

            counts_dir = self._accumulate_pairs(row_indices, col_indices, n_levels, symmetric=True)

            counts_dirs.append(counts_dir)
            counts_merged += counts_dir

        m3_dirs = [self._normalize(counts_dir) for counts_dir in counts_dirs]
        m3_comb = self._normalize(counts_merged)
        return m3_comb, m3_dirs

    # ---------------- Marginals / helper distributions ----------------

    @staticmethod
    def _p_x(glcm_prob: np.ndarray) -> np.ndarray:
        return glcm_prob.sum(axis=1)

    @staticmethod
    def _p_y(glcm_prob: np.ndarray) -> np.ndarray:
        return glcm_prob.sum(axis=0)

    @staticmethod
    def _p_xplusy(glcm_prob: np.ndarray) -> np.ndarray:
        n_levels = glcm_prob.shape[0]

        # Return empty array if GLCM has no levels
        if n_levels == 0:
            return np.zeros(0, dtype=np.float64)

        i_index, j_index = np.meshgrid(np.arange(1, n_levels + 1), np.arange(1, n_levels + 1), indexing="ij")
        k_indices = (i_index + j_index - 2).astype(np.int64)
        p_xplusy = np.zeros(2 * n_levels - 1, dtype=np.float64)
        np.add.at(p_xplusy, k_indices.ravel(), glcm_prob.ravel())
        return p_xplusy

    @staticmethod
    def _p_xminusy(glcm_prob: np.ndarray) -> np.ndarray:
        """
        Vectorized p_{|x-y|} via bincount over |i-j|.
        """
        n_levels = glcm_prob.shape[0]
        if n_levels == 0:
            return np.zeros(0, dtype=np.float64)
        i_grid, j_grid = _cached_ij_grids(n_levels)
        diff = np.abs(i_grid - j_grid).astype(np.int64)
        return np.bincount(diff.ravel(), weights=glcm_prob.ravel(), minlength=n_levels).astype(np.float64)

    @staticmethod
    def _safe_entropy(prob: np.ndarray, small_constant: float = np.finfo(np.float64).eps) -> float:
        stabilized = prob + small_constant
        return float(-(stabilized * np.log2(stabilized)).sum())

    # -------------------------------------------------------------------------
    # Feature calculation methods (_calc_* for each GLCM metric)
    # Each method expects a normalized GLCM probability matrix P as input.
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Feature calculation methods (_calc_* for each GLCM metric)
    # Each method expects a normalized GLCM probability matrix P as input.
    # -------------------------------------------------------------------------

    # 1. Joint Maximum (Maximum Probability)
    @staticmethod
    def _calc_joint_max(glcm_prob: np.ndarray) -> float:
        """
        Compute the Joint Maximum (Maximum Probability) from a GLCM.

        The Joint Maximum is the largest probability in the GLCM, indicating the
        most frequently co-occurring gray-level pair.

        Args:
            glcm_prob (np.ndarray): 2D normalized GLCM (n_levels x n_levels).

        Returns:
            float: Maximum probability value in the GLCM. Returns np.nan if the GLCM is empty.
        """
        if glcm_prob.size == 0:
            return np.nan
        return float(np.max(glcm_prob))

    # 2. Joint Average
    def _calc_joint_avg(self, glcm_prob: np.ndarray) -> float:
        """
        Compute the Joint Average (grey-level weighted mean) from a GLCM.

        The Joint Average is the expected value of grey levels weighted by their
        marginal probabilities. By symmetry, it can be computed along rows or columns:
            Joint Average = sum_i i * p_x(i) = sum_j j * p_y(j)

        Args:
            glcm_prob (np.ndarray): 2D normalized GLCM (n_levels x n_levels).

        Returns:
            float: Joint Average value. Returns 0 if the GLCM is empty.
        """
        n_levels = glcm_prob.shape[0]
        if n_levels == 0:
            return 0.0

        px = self._p_x(glcm_prob)  # Row marginal probabilities
        gray_levels = np.arange(1, n_levels + 1, dtype=np.float64)  # 1-based indices
        joint_avg = np.sum(gray_levels * px)

        return float(joint_avg)

    # 3. Joint Variance
    def _calc_joint_var(self, glcm_prob: np.ndarray) -> float:
        """
        Compute the Joint Variance of a GLCM.

        Joint Variance measures the variance of grey levels weighted by their
        marginal probabilities:
            Joint Variance = sum_i (i - μ)^2 * p_x(i)
        where μ is the Joint Average (mean grey level).

        Args:
            glcm_prob (np.ndarray): 2D normalized GLCM (n_levels x n_levels).

        Returns:
            float: Joint Variance value. Returns 0 if the GLCM is empty.
        """
        n_levels = glcm_prob.shape[0]
        if n_levels == 0:
            return 0.0

        x_prob = self._p_x(glcm_prob)  # Row marginal probabilities
        gray_levels = np.arange(1, n_levels + 1, dtype=np.float64)  # 1-based indices
        joint_avg = float(np.sum(gray_levels * x_prob))
        joint_var = float(np.sum(((gray_levels - joint_avg) ** 2) * x_prob))

        return joint_var

    # 4. Joint Entropy
    def _calc_joint_entr(self, glcm_prob: np.ndarray) -> float:
        """
        Compute the Joint Entropy of a GLCM.

        Joint Entropy quantifies the uncertainty in the joint grey-level
        distribution of the GLCM:
            Joint Entropy = -sum_{i,j} P(i,j) * log2(P(i,j) + eps)
        A small constant is added to avoid log2(0).

        Args:
            glcm_prob (np.ndarray): 2D normalized GLCM (n_levels x n_levels).

        Returns:
            float: Joint Entropy in bits. Returns 0 if the GLCM is empty.
        """
        if glcm_prob.size == 0:
            return 0.0
        return self._safe_entropy(glcm_prob)

    # 5. Difference Average (Dissimilarity)
    def _calc_diff_avg(self, glcm_prob: np.ndarray) -> float:
        """
        Compute the Difference Average (mean of absolute intensity differences) from a GLCM.

        The Difference Average is the expected value of |i - j| weighted by its probability:
            Diff Average = sum_{k=0}^{n-1} k * p_{|x-y|}(k)
        This is mathematically equivalent to the GLCM dissimilarity measure.

        Args:
            glcm_prob (np.ndarray): 2D normalized GLCM (n_levels x n_levels).

        Returns:
            float: Difference Average value. Returns 0 if the GLCM is empty.
        """
        n_levels = glcm_prob.shape[0]
        if n_levels == 0:
            return 0.0

        # Compute |i-j| distribution
        p_diff = self._p_xminusy(glcm_prob)
        diff_indices = np.arange(0, n_levels, dtype=np.float64)

        diff_avg = float(np.sum(diff_indices * p_diff))
        return diff_avg

    # 6. Difference Variance
    def _calc_diff_var(self, glcm_prob: np.ndarray) -> float:
        """
        Compute the Difference Variance of a GLCM.

        Difference Variance measures the variance of |i - j| distribution:
            Diff Variance = sum_k (k - μ_d)^2 * p_{|x-y|}(k)
        where μ_d is the Difference Average (mean of |i-j|).

        Args:
            glcm_prob (np.ndarray): 2D normalized GLCM (n_levels x n_levels).

        Returns:
            float: Difference Variance value. Returns 0 if the GLCM is empty.
        """
        n_levels = glcm_prob.shape[0]
        if n_levels == 0:
            return 0.0

        # Compute |i-j| distribution
        p_diff = self._p_xminusy(glcm_prob)
        diff_indices = np.arange(0, n_levels, dtype=np.float64)
        diff_avg = float(np.sum(diff_indices * p_diff))

        diff_var = float(np.sum(((diff_indices - diff_avg) ** 2) * p_diff))
        return diff_var

    # 7. Difference Entropy
    def _calc_diff_entr(self, glcm_prob: np.ndarray) -> float:
        """
        Compute the Difference Entropy of a GLCM.

        Difference Entropy quantifies the uncertainty in the |i-j| (absolute intensity difference)
        distribution:
            Diff Entropy = -sum_k p_{|x-y|}(k) * log2(p_{|x-y|}(k) + eps)
        A small constant is added to avoid log2(0).

        Args:
            glcm_prob (np.ndarray): 2D normalized GLCM (n_levels x n_levels).

        Returns:
            float: Difference Entropy in bits. Returns 0 if the GLCM is empty.
        """
        n_levels = glcm_prob.shape[0]
        if n_levels == 0:
            return 0.0

        # Compute |i-j| distribution
        p_diff = self._p_xminusy(glcm_prob)

        # Compute entropy with numerical stability
        return self._safe_entropy(p_diff)

    # 8. Sum Average
    def _calc_sum_avg(self, glcm_prob: np.ndarray) -> float:
        """
        Compute the Sum Average of a GLCM.

        Sum Average is the expected value of (i + j) weighted by the probability distribution:
            Sum Average = sum_{k=2}^{2N} k * p_{x+y}(k)
        where p_{x+y} is the distribution of sums of row and column indices.

        Args:
            glcm_prob (np.ndarray): 2D normalized GLCM (n_levels x n_levels).

        Returns:
            float: Sum Average value. Returns 0 if the GLCM is empty or all probabilities are zero.
        """
        if glcm_prob.size == 0:
            return 0.0

        p_sum = self._p_xplusy(glcm_prob)
        if np.sum(p_sum) == 0.0:
            return 0.0

        k_indices = _cached_sum_k_indices(glcm_prob.shape[0])
        sum_avg = float(np.sum(k_indices * p_sum))
        return sum_avg

    # 9. Sum Variance
    def _calc_sum_var(self, glcm_prob: np.ndarray) -> float:
        """
        Compute the Sum Variance of a GLCM.

        Sum Variance measures the variance of the sum distribution of indices (i + j):
            Sum Variance = sum_k (k - μ_s)^2 * p_{x+y}(k)
        where μ_s is the Sum Average.

        Args:
            glcm_prob (np.ndarray): 2D normalized GLCM (n_levels x n_levels).

        Returns:
            float: Sum Variance value. Returns 0 if the GLCM is empty or all probabilities are zero.
        """
        if glcm_prob.size == 0:
            return 0.0

        p_sum = self._p_xplusy(glcm_prob)
        if np.sum(p_sum) == 0.0:
            return 0.0

        k_indices = _cached_sum_k_indices(glcm_prob.shape[0])
        sum_avg = float(np.sum(k_indices * p_sum))
        sum_var = float(np.sum(((k_indices - sum_avg) ** 2) * p_sum))
        return sum_var

    # 10. Sum Entropy
    def _calc_sum_entr(self, glcm_prob: np.ndarray) -> float:
        """
        Compute the Sum Entropy of a GLCM.

        Sum Entropy quantifies the uncertainty in the sum distribution of indices (i + j):
            Sum Entropy = -sum_k p_{x+y}(k) * log2(p_{x+y}(k) + eps)
        A small constant is added to avoid log2(0).

        Args:
            glcm_prob (np.ndarray): 2D normalized GLCM (n_levels x n_levels).

        Returns:
            float: Sum Entropy in bits. Returns 0 if the GLCM is empty.
        """
        if glcm_prob.size == 0:
            return 0.0

        p_sum = self._p_xplusy(glcm_prob)
        return self._safe_entropy(p_sum)

    # 11. Angular Second Moment (Energy)
    @staticmethod
    def _calc_energy(glcm_prob: np.ndarray) -> float:
        """
        Compute the Angular Second Moment (Energy) of a GLCM.

        Energy measures the uniformity of the GLCM:
            Energy = sum_{i,j} P(i,j)^2
        Ranges from 0 to 1, reaching 1 for a perfectly uniform matrix (only one non-zero element).

        Args:
            glcm_prob (np.ndarray): 2D normalized GLCM (n_levels x n_levels).

        Returns:
            float: Energy value. Returns 0 if the GLCM is empty.
        """
        if glcm_prob.size == 0:
            return 0.0

        energy = float(np.sum(glcm_prob ** 2))
        return energy

    # 12. Contrast
    def _calc_contrast(self, glcm_prob: np.ndarray) -> float:
        """
        Compute the Contrast of a GLCM.

        Contrast measures the intensity contrast between voxel pairs:
            Contrast = sum_{i,j} (i - j)^2 * P(i,j)
        Higher values indicate greater intensity differences.

        Args:
            glcm_prob (np.ndarray): 2D normalized GLCM (n_levels x n_levels).

        Returns:
            float: Contrast value. Returns 0 if the GLCM is empty.
        """
        n_levels = glcm_prob.shape[0]
        if n_levels == 0:
            return 0.0

        i_index, j_index = _cached_ij_grids(n_levels)
        contrast = float(np.sum(((i_index - j_index) ** 2) * glcm_prob))
        return contrast

    # 13. Dissimilarity
    def _calc_dissimilarity(self, glcm_prob: np.ndarray) -> float:
        """
        Compute the Dissimilarity of a GLCM.

        Dissimilarity measures the mean absolute difference between intensity pairs:
            Dissimilarity = sum_{i,j} |i - j| * P(i,j)
        Equivalent to the Difference Average.

        Args:
            glcm_prob (np.ndarray): 2D normalized GLCM (n_levels x n_levels).

        Returns:
            float: Dissimilarity value. Returns 0 if the GLCM is empty.
        """
        n_levels = glcm_prob.shape[0]
        if n_levels == 0:
            return 0.0

        i_index, j_index = _cached_ij_grids(n_levels)
        dissimilarity = float(np.sum(np.abs(i_index - j_index) * glcm_prob))
        return dissimilarity

    # 14. Inverse Difference (Homogeneity)
    def _calc_inv_diff(self, glcm_prob: np.ndarray) -> float:
        """
        Compute the Inverse Difference (Homogeneity) of a GLCM.

        Inverse Difference emphasizes contributions from neighboring intensity pairs:
            Homogeneity = sum_{i,j} P(i,j) / (1 + |i - j|)

        Args:
            glcm_prob (np.ndarray): 2D normalized GLCM (n_levels x n_levels).

        Returns:
            float: Inverse Difference (Homogeneity) value. Returns 0 if the GLCM is empty.
        """
        n_levels = glcm_prob.shape[0]
        if n_levels == 0:
            return 0.0

        i_index, j_index = _cached_ij_grids(n_levels)
        inv_diff = float(np.sum(glcm_prob / (1.0 + np.abs(i_index - j_index))))
        return inv_diff

    # 15. Inverse Difference Normalized
    def _calc_inv_diff_norm(self, glcm_prob: np.ndarray) -> float:
        """
        Compute the Inverse Difference Normalized (IDN) of a GLCM.

        IDN (Normalized Homogeneity) emphasizes neighboring intensity pairs,
        normalized by the number of gray levels:
            IDN = sum_{i,j} P(i,j) / (1 + |i - j| / (Ng - 1))
        where Ng is the number of gray levels (matrix size).

        Args:
            glcm_prob (np.ndarray): 2D normalized GLCM (Ng x Ng).

        Returns:
            float: IDN value. For Ng <= 1, falls back to unnormalized Inverse Difference.
        """
        n_levels = glcm_prob.shape[0]
        if n_levels == 0:
            return 0.0

        i_index, j_index = _cached_ij_grids(n_levels)
        if n_levels <= 1:
            # fallback to unnormalized inverse difference
            return float(np.sum(glcm_prob / (1.0 + np.abs(i_index - j_index))))

        denom = float(n_levels - 1)
        idn_value = float(np.sum(glcm_prob / (1.0 + np.abs(i_index - j_index) / denom)))
        return idn_value

    # 16. Inverse Difference Moment
    def _calc_inv_diff_mom(self, glcm_prob: np.ndarray) -> float:
        """
        Compute the Inverse Difference Moment (IDM) of a GLCM.

        Inverse Difference Moment is a homogeneity measure that emphasizes
        contributions from neighboring intensity pairs more strongly:
            IDM = sum_{i,j} P(i,j) / (1 + (i - j)^2)

        Args:
            glcm_prob (np.ndarray): 2D normalized GLCM (n_levels x n_levels).

        Returns:
            float: IDM value. Returns 0 if the GLCM is empty.
        """
        n_levels = glcm_prob.shape[0]
        if n_levels == 0:
            return 0.0

        i_index, j_index = _cached_ij_grids(n_levels)
        idm_value = float(np.sum(glcm_prob / (1.0 + (i_index - j_index) ** 2)))
        return idm_value

    # 17. Inverse Difference Moment Normalized
    def _calc_inv_diff_mom_norm(self, glcm_prob: np.ndarray) -> float:
        """
        Compute the Inverse Difference Moment Normalized (IDMN) of a GLCM.

        IDMN is the normalized variant of the Inverse Difference Moment (IDM),
        conforming to IBSI standards. It emphasizes neighboring intensity pairs
        with normalization by the number of gray levels:
            IDMN = sum_{i,j} P(i,j) / [1 + ((i - j) / (Ng - 1))^2]
        where Ng is the number of gray levels. For Ng <= 1, falls back to
        the unnormalized Inverse Difference Moment.

        Args:
            glcm_prob (np.ndarray): 2D normalized GLCM (Ng x Ng).

        Returns:
            float: IDMN value. Returns 0 if the GLCM is empty.
        """
        n_levels = glcm_prob.shape[0]
        if n_levels == 0:
            return 0.0

        i_index, j_index = _cached_ij_grids(n_levels)

        if n_levels <= 1:
            # fallback to unnormalized inverse difference moment
            return float(np.sum(glcm_prob / (1.0 + (i_index - j_index) ** 2)))

        denom = float(n_levels - 1)
        idmn_value = float(np.sum(glcm_prob / (1.0 + ((i_index - j_index) / denom) ** 2)))
        return idmn_value

    # 18. Inverse Variance
    def _calc_inv_var(self, glcm_prob: np.ndarray) -> float:
        """
        Compute the Inverse Variance (IV) of a GLCM.

        Inverse Variance emphasizes near-diagonal elements (i ≈ j),
        excluding the diagonal itself:
            IV = 2 * sum_{i<j} P(i,j) / (i - j)^2
        Returns 0 if all energy is on the diagonal.

        Args:
            glcm_prob (np.ndarray): 2D normalized GLCM (n_levels x n_levels).

        Returns:
            float: Inverse Variance value. Returns 0 if the GLCM is empty.
        """
        n_levels = glcm_prob.shape[0]
        if n_levels == 0:
            return 0.0

        i_index, j_index = _cached_ij_grids(n_levels)
        denom = (i_index - j_index) ** 2

        # Compute element-wise P / (i-j)^2, set diagonal to 0
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_var_matrix = np.divide(glcm_prob, denom, out=np.zeros_like(glcm_prob, dtype=np.float64),
                                       where=denom != 0)

        # Sum only the upper triangle (i<j) and double for symmetry
        upper_triangle = np.triu(inv_var_matrix, k=1)
        inv_var_value = float(2.0 * upper_triangle.sum())
        return inv_var_value

    # 19. Correlation (Haralick's)
    def _calc_corr(self, glcm_prob: np.ndarray) -> float:
        """
        Compute the Correlation of a GLCM.

        Correlation measures the linear dependency between row and column intensities:
            Corr = sum_{i,j} [(i - μ_x) * (j - μ_y) * P(i,j)] / (σ_x * σ_y)
        where μ_x, μ_y are the marginal means and σ_x, σ_y are the standard deviations
        of row and column marginals. Returns 0 if variance is zero.

        Args:
            glcm_prob (np.ndarray): 2D normalized GLCM (n_levels x n_levels).

        Returns:
            float: Correlation value, ranging from -1 to 1 (0 if variance is zero).
        """
        n_levels = glcm_prob.shape[0]
        if n_levels == 0:
            return 0.0

        x_prob = self._p_x(glcm_prob)
        y_prob = self._p_y(glcm_prob)
        i_index = np.arange(1, n_levels + 1, dtype=np.float64)
        j_index = i_index.copy()

        # Marginal means
        mean_x = float((i_index * x_prob).sum())
        mean_y = float((j_index * y_prob).sum())

        # Marginal standard deviations
        std_x = float(np.sqrt(((i_index - mean_x) ** 2 * x_prob).sum()))
        std_y = float(np.sqrt(((j_index - mean_y) ** 2 * y_prob).sum()))

        if std_x == 0.0 or std_y == 0.0:
            return 0.0

        # Compute correlation
        i_grid, j_grid = _cached_ij_grids(n_levels)
        numerator = float(((i_grid - mean_x) * (j_grid - mean_y) * glcm_prob).sum())
        correlation = numerator / (std_x * std_y)
        return float(correlation)

    # 20. Autocorrelation
    def _calc_auto_corr(self, glcm_prob: np.ndarray) -> float:
        """
        Compute the Autocorrelation of a GLCM.

        Autocorrelation measures the correlation of voxel intensities within the ROI:
            Autocorr = sum_{i,j} i * j * P(i,j)

        Args:
            glcm_prob (np.ndarray): 2D normalized GLCM (n_levels x n_levels).

        Returns:
            float: Autocorrelation value. Returns 0 if the GLCM is empty.
        """
        n_levels = glcm_prob.shape[0]
        if n_levels == 0:
            return 0.0

        i_index, j_index = _cached_ij_grids(n_levels)
        return float((i_index * j_index * glcm_prob).sum())

    # 21. Cluster Tendency (Sum of Squares)
    def _calc_clust_tend(self, glcm_prob: np.ndarray) -> float:
        """
        Compute the Cluster Tendency (Cluster Shade / Sum Variance) of a GLCM.

        Measures the grouping of similar intensity pairs:
            ClusterTend = sum_{i,j} (i + j - 2*μ)^2 * P(i,j)
        where μ is the mean gray level (row marginal mean).

        Args:
            glcm_prob (np.ndarray): 2D normalized GLCM (n_levels x n_levels).

        Returns:
            float: Cluster Tendency value.
        """
        n_levels = glcm_prob.shape[0]
        if n_levels == 0:
            return 0.0

        px = self._p_x(glcm_prob)
        i_index = np.arange(1, n_levels + 1, dtype=np.float64)
        mean_gray = float((i_index * px).sum())

        i_grid, j_grid = _cached_ij_grids(n_levels)
        cluster_tend = float((((i_grid + j_grid - 2.0 * mean_gray) ** 2) * glcm_prob).sum())
        return cluster_tend

    # 22. Cluster Shade
    def _calc_clust_shade(self, glcm_prob: np.ndarray) -> float:
        """
        Compute the Cluster Shade of a GLCM.

        Cluster Shade measures the skewness of clusters in the GLCM:
            ClusterShade = sum_{i,j} (i + j - 2*μ)^3 * P(i,j)
        where μ is the mean gray level (row marginal mean). Sign indicates skew direction.

        Args:
            glcm_prob (np.ndarray): 2D normalized GLCM (n_levels x n_levels).

        Returns:
            float: Cluster Shade value.
        """
        n_levels = glcm_prob.shape[0]
        if n_levels == 0:
            return 0.0

        px = self._p_x(glcm_prob)
        i_index = np.arange(1, n_levels + 1, dtype=np.float64)
        mean_gray = float((i_index * px).sum())

        i_grid, j_grid = _cached_ij_grids(n_levels)
        cluster_shade = float((((i_grid + j_grid - 2.0 * mean_gray) ** 3) * glcm_prob).sum())
        return cluster_shade

    # 23. Cluster Prominence
    def _calc_clust_prom(self, glcm_prob: np.ndarray) -> float:
        """
        Compute the Cluster Prominence of a GLCM.

        Cluster Prominence measures the peakedness (fourth moment) of clusters:
            ClusterProm = sum_{i,j} (i + j - 2*μ)^4 * P(i,j)
        where μ is the mean gray level (row marginal mean). Higher values indicate more outliers.

        Args:
            glcm_prob (np.ndarray): 2D normalized GLCM (n_levels x n_levels).

        Returns:
            float: Cluster Prominence value.
        """
        n_levels = glcm_prob.shape[0]
        if n_levels == 0:
            return 0.0

        px = self._p_x(glcm_prob)
        i_index = np.arange(1, n_levels + 1, dtype=np.float64)
        mean_gray = float((i_index * px).sum())

        i_grid, j_grid = _cached_ij_grids(n_levels)
        cluster_prom = float((((i_grid + j_grid - 2.0 * mean_gray) ** 4) * glcm_prob).sum())
        return cluster_prom

    # 24. Information Measure of Correlation 1 (IMC1)
    def _calc_info_corr1(self, glcm_prob: np.ndarray) -> float:
        """
        Compute Information Measure of Correlation 1 (IMC1) from a GLCM.

        IMC1 quantifies correlation between row and column probabilities:
            IMC1 = (HXY - HXY1) / max(HX, HY)
        where:
            HX, HY = entropies of row and column marginals,
            HXY = joint entropy,
            HXY1 = -sum_{i,j} P(i,j) * log2(P_i * P_j) (cross-entropy).

        Args:
            glcm_prob (np.ndarray): 2D normalized GLCM (n_levels x n_levels).

        Returns:
            float: IMC1 value.
        """
        small_constant: float = np.finfo(np.float64).eps

        # Marginal probabilities
        prob_x = self._p_x(glcm_prob)
        prob_y = self._p_y(glcm_prob)

        # Marginal entropies
        entropy_x = float(-(prob_x * np.log2(prob_x + small_constant)).sum())
        entropy_y = float(-(prob_y * np.log2(prob_y + small_constant)).sum())

        # Joint entropy
        joint_entropy = self._safe_entropy(glcm_prob)

        # Cross-entropy of joint vs independent marginals
        prob_indep = np.outer(prob_x, prob_y)
        joint_entropy_indep = float(-(glcm_prob * np.log2(prob_indep + small_constant)).sum())

        # Compute IMC1 with safe denominator
        denom = max(entropy_x, entropy_y) if max(entropy_x, entropy_y) > 0 else 1.0
        imc1_value = (joint_entropy - joint_entropy_indep) / denom

        return float(0.0 if not np.isfinite(imc1_value) else imc1_value)

    # 25. Information Measure of Correlation 2 (IMC2)
    def _calc_info_corr2(self, glcm_prob: np.ndarray) -> float:
        """
        Compute Information Measure of Correlation 2 (IMC2) from a GLCM.

        IMC2 quantifies correlation between row and column probabilities:
            IMC2 = sqrt(1 - exp(-2 * (HXY2 - HXY)))
        where:
            HXY  = joint entropy,
            HXY2 = entropy of the product of marginal probabilities (independent distribution).

        Args:
            glcm_prob (np.ndarray): 2D normalized GLCM (n_levels x n_levels).

        Returns:
            float: IMC2 value.
        """
        small_constant: float = np.finfo(np.float64).eps

        # Marginal probabilities
        prob_x = self._p_x(glcm_prob)
        prob_y = self._p_y(glcm_prob)

        # Joint entropy
        joint_entropy = self._safe_entropy(glcm_prob)

        # Entropy of independent distribution
        prob_indep = np.outer(prob_x, prob_y)
        joint_entropy_indep = float(-(prob_indep * np.log2(prob_indep + small_constant)).sum())

        # Compute IMC2
        val = 1.0 - np.exp(-2.0 * (joint_entropy_indep - joint_entropy))
        imc2_value = np.sqrt(val) if val >= 0.0 else 0.0

        return float(imc2_value)

# ------------ Thin wrappers selecting allowed modes ------------
class GrayLevelCooccurrenceMatrixFeatures2DExtractor(GrayLevelCooccurrenceMatrixFeaturesExtractor):
    NAME: str = "GLCM2DExtractor"
    _allowed_modes = ["2D_avg", "2D_comb"]


class GrayLevelCooccurrenceMatrixFeatures25DExtractor(GrayLevelCooccurrenceMatrixFeaturesExtractor):
    NAME: str = "GLCM25DExtractor"
    _allowed_modes = ["2_5D_avg", "2_5D_comb"]


class GrayLevelCooccurrenceMatrixFeatures3DExtractor(GrayLevelCooccurrenceMatrixFeaturesExtractor):
    NAME: str = "GLCM3DExtractor"
    _allowed_modes = ["3D_avg", "3D_comb"]
