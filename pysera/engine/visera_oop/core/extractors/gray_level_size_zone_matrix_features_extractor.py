# # -*- coding: utf-8 -*-
# # core/extractors/gray_level_size_zone_matrix_features_extractor.py

from __future__ import annotations

import logging
import numpy as np
import scipy.ndimage as ndi
from typing import Any, Dict, List, Optional, Tuple, Callable

from pysera.engine.visera_oop.core.base_feature_extractor import BaseFeatureExtractor
from pysera.engine.visera_oop.core.sparsity.view_planner import DataView

logger = logging.getLogger("Dev_logger")


class GrayLevelSizeZoneMatrixFeaturesExtractor(BaseFeatureExtractor):
    """GLSZM feature extractor with dynamic getters for multiple dimensional modes."""
    ACCEPTS: DataView = DataView.DENSE_BLOCK
    PREFERS: DataView = DataView.DENSE_BLOCK

    def __getattr__(self, name: str) -> Callable[[int], float]:
        """Return a callable for a dynamically resolved GLSZM feature."""

        if not name.startswith("get_szm_"):
            raise AttributeError(f"{type(self).__name__} has no attribute {name}")

        core = name[len("get_szm_"):]
        mode: Optional[str] = None
        feature: Optional[str] = None

        for m in ("2_5D", "3D", "2D"):
            if core.endswith(f"_{m}"):
                mode = m
                feature = core[: -(len(m) + 1)]
                break

        if mode is None or feature is None:
            logger.error(f"Invalid feature name format: {name}")
            return lambda *_args, **_kwargs: float(np.nan)

        calc_func = getattr(self, f"_calc_{feature}", None)

        if not callable(calc_func):
            logger.error(f"Feature '{feature}' calculation function is missing.")
            return lambda *_args, **_kwargs: float(np.nan)

        # Special handling for zone percentage feature (requires ROI context)
        if feature == "z_perc":
            def getter(roi_index: int) -> float:

                if mode == "2D":
                    result = self._mean_over_slices(roi_index, lambda mat, s: calc_func(mat, roi_index, s))

                elif mode == "2_5D":
                    matrix = self._get_m25_feature(roi_index)
                    result = calc_func(matrix, roi_index, None) if matrix is not None else float(np.nan)

                else:  # 3D
                    matrix = self._get_m3_feature(roi_index)
                    result = calc_func(matrix, roi_index, None) if matrix is not None else float(np.nan)

                return result

        else:
            def getter(roi_index: int) -> float:
                if mode == "2D":
                    result = self._mean_over_slices(roi_index, lambda mat, s: calc_func(mat))

                elif mode == "2_5D":
                    matrix = self._get_m25_feature(roi_index)
                    result = calc_func(matrix) if matrix is not None else float(np.nan)

                else:  # 3D
                    matrix = self._get_m3_feature(roi_index)
                    result = calc_func(matrix) if matrix is not None else float(np.nan)

                return result

        return getter

    def _discover_feature_names(self) -> List[str]:
        """Discover implemented GLSZM feature names for allowed modes."""

        methods = []
        for cls in self.__class__.mro():
            methods.extend(name for name, obj in cls.__dict__.items() if callable(obj))

        metric_names = [name[len("_calc_"):] for name in methods if name.startswith("_calc_")]
        dimensions = getattr(self, "_allowed_modes", [])
        feature_names = [f"szm_{metric}_{dim}" for dim in dimensions for metric in metric_names]

        return feature_names

    @classmethod
    def _is_mode_allowed(cls, mode: str) -> bool:
        """Check if the subclass supports the given mode."""
        return getattr(cls, "_allowed_modes", []) and mode in cls._allowed_modes

    def _ensure_cache(self, roi_id: int) -> None:
        """Ensure that GLSZM matrices are computed and cached for a given ROI."""
        if not hasattr(self, "_glszm_cache"):
            self._glszm_cache: Dict[int, Dict[str, Any]] = {}

        roi_quant: Optional[np.ndarray] = self.get_roi(roi_id)
        roi_views: Dict[str, Any] = self.get_views(roi_id)
        mask_array, levels_array = self._prepare_mask_and_levels(roi_quant, roi_views)

        if roi_quant is None or roi_quant.ndim != 3 or mask_array is None or levels_array is None:
            self._glszm_cache[roi_id] = {}
            logger.warning(f"Invalid or missing data for ROI {roi_id}. Cache entry set to empty.")
            return

        mats_2d, matrix_25d = self._build_2d_and_25d_matrices(roi_quant, mask_array, levels_array)
        matrix_3d = self._build_3d_matrix(roi_quant, mask_array, levels_array)
        self._glszm_cache[roi_id] = {"MATS_2D": mats_2d, "M25": matrix_25d, "M3": matrix_3d}

    def _get_m25_feature(self, roi_id: int) -> Optional[np.ndarray]:
        """Retrieve the 2.5D GLSZM matrix for a given ROI."""
        self._ensure_cache(roi_id)
        return self._glszm_cache.get(roi_id, {}).get("M25", None)

    def _get_m3_feature(self, roi_id: int) -> Optional[np.ndarray]:
        """Retrieve the 3D GLSZM matrix for a given ROI."""
        self._ensure_cache(roi_id)
        return self._glszm_cache.get(roi_id, {}).get("M3", None)

    def _mean_over_slices(self, roi_id: int, func: Callable[[np.ndarray, int], float]) -> float:
        """Compute the mean of a metric across all 2D slices of an ROI."""
        self._ensure_cache(roi_id)
        matrices: List[np.ndarray] = self._glszm_cache.get(roi_id, {}).get("MATS_2D", [])
        if not matrices:
            return float(np.nan)
        values: List[float] = []
        for slice_index, mat in enumerate(matrices):
            value = func(mat, slice_index)
            values.append(value if np.isfinite(value) else np.nan)
        return float(np.nanmean(values)) if values else float(np.nan)

    def _prepare_mask_and_levels(
            self, quant_array: Optional[np.ndarray], views: Dict[str, Any]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare binary mask and intensity levels for GLSZM computation (optimized)."""

        # Fast checks
        if quant_array is None or quant_array.ndim != 3:
            return None, None

        mask_array = views.get("binary_mask")
        levels_array = views.get("levels")

        # Avoid repeated np.isfinite calls
        if mask_array is None:
            mask_array = np.isfinite(quant_array)

        # Fast path if levels_array already valid
        if (
                levels_array is None
                or not isinstance(levels_array, np.ndarray)
                or levels_array.size == 0
        ):
            # Use np.unique directly on masked finite elements (saves memory & time)
            finite_vals = quant_array[mask_array]
            if finite_vals.size:
                levels_array = np.unique(finite_vals).astype(np.float32, copy=False)
            else:
                levels_array = None

        return mask_array, levels_array

    @staticmethod
    def _build_glszm_matrix(quant: np.ndarray, mask: np.ndarray, levels: np.ndarray) -> np.ndarray:
        """Optimized GLSZM matrix builder with identical results but much faster runtime."""
        import numpy as np
        from typing import Tuple

        uniq_levels = np.unique(levels).astype(np.float32, copy=False)
        n_rows = uniq_levels.size

        # Use fast binary connectivity structure
        struct = ndi.generate_binary_structure(quant.ndim, 2 if quant.ndim == 2 else 3)

        zone_size_dict: Dict[Tuple[int, int], int] = {}

        # Preallocate reused arrays to avoid reallocations
        for row_idx, gray_level in enumerate(uniq_levels):
            mask_gray_level = (quant == gray_level) & mask
            if not np.any(mask_gray_level):
                continue

            # Label connected regions once per gray level
            labeled, n_zones = ndi.label(mask_gray_level, structure=struct)
            if n_zones == 0:
                continue

            # Compute region sizes efficiently using bincount
            zone_sizes = np.bincount(labeled.ravel())[1:]  # Skip background (0)

            # Count occurrences of each zone size
            size_counts = np.bincount(zone_sizes)
            for zone_size, count in enumerate(size_counts):
                if count == 0:
                    continue
                zone_size_dict[(row_idx, zone_size)] = zone_size_dict.get((row_idx, zone_size), 0) + count

        # Allocate final GLSZM matrix
        max_size = max((size for (_, size) in zone_size_dict.keys()), default=1)
        glszm_matrix = np.zeros((n_rows, max_size), dtype=np.int64)

        # Fill GLSZM efficiently
        for (row_idx, zone_size), count in zone_size_dict.items():
            glszm_matrix[row_idx, zone_size - 1] = count

        return glszm_matrix

    def _build_2d_and_25d_matrices(
            self, quant_array: np.ndarray, mask_array: np.ndarray, levels_array: np.ndarray
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Compute per-slice 2D GLSZMs and aggregated 2.5D GLSZM (optimized for speed)."""

        num_slices = quant_array.shape[2]
        n_levels = len(levels_array)
        matrices_2d = [None] * num_slices  # Preallocate list for speed
        max_cols = 0

        # Precompute function reference for faster local lookup
        _build_glszm = self._build_glszm_matrix

        # Iterate slices efficiently
        for s in range(num_slices):
            slice_mask = mask_array[:, :, s]
            if not np.any(slice_mask):  # faster than count_nonzero == 0
                slice_matrix = np.zeros((n_levels, 1), dtype=np.int64)
            else:
                slice_matrix = _build_glszm(quant_array[:, :, s], slice_mask, levels_array)

            matrices_2d[s] = slice_matrix
            cols = slice_matrix.shape[1]
            if cols > max_cols:
                max_cols = cols

        # Efficiently aggregate into 2.5D matrix
        if max_cols > 0:
            matrix_25d = np.zeros((n_levels, max_cols), dtype=np.int64)
            for mat in matrices_2d:
                c = mat.shape[1]
                if c:
                    matrix_25d[:, :c] += mat
        else:
            matrix_25d = np.zeros((n_levels, 0), dtype=np.int64)

        return matrices_2d, matrix_25d

    def _build_3d_matrix(
            self, quant_array: np.ndarray, mask_array: np.ndarray, levels_array: np.ndarray
    ) -> np.ndarray:
        """Compute the GLSZM matrix for the entire 3D volume."""

        matrix_3d = self._build_glszm_matrix(quant_array, mask_array, levels_array)

        return matrix_3d

    @staticmethod
    def _trim_trailing_zero_columns(matrix: np.ndarray) -> np.ndarray:
        """Trim trailing columns consisting of all zeros in a matrix."""
        if matrix.size == 0:
            return matrix

        col_sums = matrix.sum(axis=0)
        nonzero_cols = np.where(col_sums > 0)[0]

        return matrix[:, :1] if nonzero_cols.size == 0 else matrix[:, : nonzero_cols[-1] + 1]

    @staticmethod
    def _get_total_count(matrix: np.ndarray) -> int:
        """Get the total sum of all elements in the matrix."""
        return int(matrix.sum())

    @staticmethod
    def _get_row_and_column_sums(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute sum of matrix elements across rows and columns."""
        row_sums = matrix.sum(axis=1).astype(np.float64)
        col_sums = matrix.sum(axis=0).astype(np.float64)

        return row_sums, col_sums

    @staticmethod
    def _get_indices_and_grids(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate 1-based indices and meshgrids for rows and columns of a matrix."""
        num_rows, num_cols = matrix.shape
        row_indices = np.arange(1, num_rows + 1, dtype=np.float64)
        col_indices = np.arange(1, num_cols + 1, dtype=np.float64)
        row_grid, col_grid = np.meshgrid(row_indices, col_indices, indexing="ij")

        return row_indices, col_indices, row_grid, col_grid

    def _get_empty_value(self) -> float:
        """Return default empty feature value based on feature value mode."""
        return 0.0 if str(self.feature_value_mode).upper() == "APPROXIMATE_VALUE" else float(np.nan)

    def _prepare_matrix(
            self, matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """Prepare matrix by trimming zeros and computing indices/grids/total count."""
        matrix = self._trim_trailing_zero_columns(matrix)
        total_count = self._get_total_count(matrix)

        if total_count > 0 and matrix.size:
            row_idx, col_idx, row_grid, col_grid = self._get_indices_and_grids(matrix)

        else:
            row_idx = col_idx = row_grid = col_grid = np.array([], dtype=np.float64)

        return matrix, row_idx, col_idx, row_grid, col_grid, total_count

    # -------------------------------------------------------------------------
    # Feature implementations (_calc_*)
    # -------------------------------------------------------------------------
    # 1 Small Size Emphasis
    def _calc_sze(self, matrix: np.ndarray) -> float:
        """
        Calculate the Short Size Emphasis (SDE) feature from a GLSM matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Input 2D GLSM matrix.

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

    # 2 Large Size Emphasis
    def _calc_lze(self, matrix: np.ndarray) -> float:
        """
        Calculate the Long Size Emphasis (LDE) feature from a GLSM matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Input 2D GLSM matrix.

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
        Calculate the Low Gray-Level Zone Emphasis (LGZE) feature from a GLSM matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Input 2D GLSM matrix.

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
        Calculate the High Gray-Level Zone Emphasis (HGZE) feature from a GLSM matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Input 2D GLSM matrix.

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

    # 5 Small Size Low Grey Level Emphasis
    def _calc_sdlge(self, matrix: np.ndarray) -> float:
        """
        Calculate the Short Size Low Gray-Level Emphasis (SDLGE) feature
        from a GLSM matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Input 2D GLSM matrix.

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

    # 6 Small Size High Grey Level Emphasis
    def _calc_sdhge(self, matrix: np.ndarray) -> float:
        """
        Calculate the Short Size High Gray-Level Emphasis (SDHGE) feature
        from a GLSM matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Input 2D GLSM matrix.

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

    # 7 Large Size Low Grey Level Emphasis
    def _calc_ldlge(self, matrix: np.ndarray) -> float:
        """
        Calculate the Long Size Low Gray-Level Emphasis (LDLGE) feature
        from a GLSM matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Input 2D GLSM matrix.

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

    # 8 Large Size High Grey Level Emphasis
    def _calc_ldhge(self, matrix: np.ndarray) -> float:
        """
        Calculate the Long Size High Gray-Level Emphasis (LDHGE) feature
        from a GLSM matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Input 2D GLSM matrix.

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
        Calculate the Gray-Level Non-Uniformity (GLNU) feature from a GLSM matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Input 2D GLSM matrix.

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
        from a GLSM matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Input 2D GLSM matrix.

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

    # 11 Zone Size Non-uniformity
    def _calc_zsnu(self, matrix: np.ndarray) -> float:
        """
        Calculate the Zone Size Non-Uniformity (zsNU) feature from a GLSM matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Input 2D GLSM matrix.

        Returns
        -------
        float
            zsNU feature value. Returns empty value if the matrix is empty.
        """
        matrix, row_indices, col_indices, row_grid, col_grid, total_count = self._prepare_matrix(matrix)

        if total_count == 0:
            return self._get_empty_value()

        col_sums: np.ndarray = matrix.sum(axis=0).astype(np.float64)
        zsnu_value: float = float((col_sums ** 2).sum() / total_count)
        return zsnu_value

    # 12 Normalised Zone Size Non-uniformity
    def _calc_zsnu_norm(self, matrix: np.ndarray) -> float:
        """
        Calculate the Normalized Zone Size Non-Uniformity (zsNU_norm) feature
        from a GLSM matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Input 2D GLSM matrix.

        Returns
        -------
        float
            Normalized zsNU feature value. Returns empty value if the matrix is empty.
        """
        matrix, row_indices, col_indices, row_grid, col_grid, total_count = self._prepare_matrix(matrix)

        if total_count == 0:
            return self._get_empty_value()

        col_sums: np.ndarray = matrix.sum(axis=0).astype(np.float64)
        zsnu_norm_value: float = float((col_sums ** 2).sum() / (total_count ** 2))
        return zsnu_norm_value

    # 13 Zone Percentage (needs voxel count)
    def _calc_z_perc(self, matrix: np.ndarray, roi_index: int, slice_index: int | None) -> float:
        """
        Calculate the zone percentage feature (ratio of zone voxels to total ROI voxels).

        Parameters
        ----------
        matrix : np.ndarray
            Input 2D GLSM matrix.
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
        Calculate the Gray-Level Variance (GLVar) feature from a GLSM matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Input 2D GLSM matrix.

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

    # 15 Zone Size Variance
    def _calc_zs_var(self, matrix: np.ndarray) -> float:
        """
        Calculate the Zone Size Variance (zs_var) feature from a GLSM matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Input 2D GLSM matrix.

        Returns
        -------
        float
            Zone Size Variance feature value. Returns empty value if the matrix is empty.
        """
        matrix, row_indices, col_indices, row_grid, col_grid, total_count = self._prepare_matrix(matrix)

        if total_count == 0:
            return self._get_empty_value()

        prob_matrix: np.ndarray = matrix.astype(np.float64) / total_count
        mean_size: float = (col_grid * prob_matrix).sum()
        zs_var_value: float = float(((col_grid - mean_size) ** 2 * prob_matrix).sum())

        return zs_var_value

    # 16 Zone Size Entropy
    def _calc_zs_entr(self, matrix: np.ndarray) -> float:
        """
        Calculate the Zone Size Entropy (zs_entr) feature from a GLSM matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Input 2D GLSM matrix.

        Returns
        -------
        float
            Zone Size Entropy feature value. Returns empty value if the matrix is empty.
        """
        matrix, row_indices, col_indices, row_grid, col_grid, total_count = self._prepare_matrix(matrix)

        if total_count == 0:
            return self._get_empty_value()

        prob_matrix: np.ndarray = matrix.astype(np.float64) / total_count
        small_constant: float = np.finfo(float).eps  # to avoid log2(0)
        zs_entr_value: float = float(-(prob_matrix * np.log2(prob_matrix + small_constant)).sum())

        return zs_entr_value


class GrayLevelSizeZoneMatrixFeatures2DExtractor(GrayLevelSizeZoneMatrixFeaturesExtractor):
    """GLSZM 2D feature extractor."""
    NAME: str = "GLSZM2DExtractor"
    _allowed_modes = ["2D"]

    def _mean_over_slices(self, roi_id: int, func: Callable[[np.ndarray, int], float]) -> float:
        """Compute mean of metric across 2D slices (overridden for clarity in 2D mode)."""
        return super()._mean_over_slices(roi_id, func)


class GrayLevelSizeZoneMatrixFeatures25DExtractor(GrayLevelSizeZoneMatrixFeaturesExtractor):
    """GLSZM 2.5D feature extractor."""
    NAME: str = "GLSZM25DExtractor"
    _allowed_modes = ["2_5D"]

    def _get_m25_feature(self, roi_id: int) -> Optional[np.ndarray]:
        """Retrieve the 2.5D GLSZM matrix for a given ROI (overridden for 2.5D mode)."""
        return super()._get_m25_feature(roi_id)


class GrayLevelSizeZoneMatrixFeatures3DExtractor(GrayLevelSizeZoneMatrixFeaturesExtractor):
    """GLSZM 3D feature extractor."""
    NAME: str = "GLSZM3DExtractor"
    _allowed_modes = ["3D"]

    def _get_m3_feature(self, roi_id: int) -> Optional[np.ndarray]:
        """Retrieve the 3D GLSZM matrix for a given ROI (overridden for 3D mode)."""
        return super()._get_m3_feature(roi_id)

    def _mean_over_slices(self, roi_id: int, func: Callable[[np.ndarray, int], float]) -> float:
        """No slice-wise averaging in 3D mode (override to disable)."""
        return float(np.nan)
