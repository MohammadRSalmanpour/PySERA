# -*- coding: utf-8 -*-
# core/sparsity/view_factory.py

from __future__ import annotations

import logging
import numpy as np
from typing import Any, Dict, Iterable, List, Tuple

# Optional: Numba JIT
try:
    from numba import njit  # noqa: F401
except ImportError:  # pragma: no cover
    njit = None

logger = logging.getLogger(__name__)


class ViewFactory:
    """
    Build on-demand representations of a cropped ROI block.

    Inputs must be shape-aligned 2D or 3D arrays:
      - intensity_block : float32 array, NaNs outside ROI
      - roi_mask        : uint8/bool mask (1 inside ROI, 0 outside)
      - quantized_block : integer/float array of discretized levels
    """

    def __init__(
        self,
        intensity_block: np.ndarray,
        roi_mask: np.ndarray,
        quantized_block: np.ndarray,
        **extra_config: Any,
    ):
        self.intensity_block = np.asarray(intensity_block, dtype=np.float32)
        self.roi_mask = np.asarray(roi_mask)
        self.quantized_block = np.asarray(quantized_block)
        self.extra_config = extra_config

        if (
            self.intensity_block.shape != self.roi_mask.shape
            or self.intensity_block.shape != self.quantized_block.shape
        ):
            raise ValueError(
                f"Input shapes must match: intensity_block={self.intensity_block.shape}, "
                f"roi_mask={self.roi_mask.shape}, quantized_block={self.quantized_block.shape}"
            )

        if self.roi_mask.dtype not in (np.uint8, np.bool_):
            self.roi_mask = self.roi_mask.astype(np.uint8, copy=False)

        if not (
            np.issubdtype(self.quantized_block.dtype, np.integer)
            or np.issubdtype(self.quantized_block.dtype, np.floating)
        ):
            self.quantized_block = self.quantized_block.astype(np.float32, copy=False)

    # --------------------------------------------------------------------- #
    # Simple views
    # --------------------------------------------------------------------- #
    def roi_vector(self) -> np.ndarray:
        """
        Return 1D array of finite intensity values inside the ROI.
        """
        roi_values = self.intensity_block[self.roi_mask.astype(bool, copy=False)]

        if np.isnan(roi_values).any():
            roi_values = roi_values[np.isfinite(roi_values)]

        return roi_values.astype(np.float32, copy=False)

    def dense_block(self) -> np.ndarray:
        """
        Return the cropped dense intensity block (view).
        """
        return self.intensity_block

    # --------------------------------------------------------------------- #
    # Sparse encodings
    # --------------------------------------------------------------------- #
    def sparse_coords(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        COO-like encoding of finite voxels within ROI.

        Returns
        -------
        coords : (N, D) int32
            Coordinates of valid voxels (D=2 for 2D, D=3 for 3D)
        values : (N, ) float32
            Corresponding intensity values
        """
        valid_mask = (self.roi_mask > 0) & np.isfinite(self.intensity_block)
        coords = np.argwhere(valid_mask).astype(np.int32, copy=False)
        values = self.intensity_block[valid_mask].astype(np.float32, copy=False)

        return coords, values

    # --------------------------------------------------------------------- #
    # Run-length encoding
    # --------------------------------------------------------------------- #
    def run_lengths(self, directions: Iterable[str] = ("0", "45", "90", "135")) -> List[Dict[str, Any]]:
        """
        Compute run-length encoding per 2D slice and direction.
        """
        quantized_data = np.asarray(self.quantized_block)
        mask_data = self.roi_mask
        run_length_results: List[Dict[str, Any]] = []

        # Prepare slices
        if quantized_data.ndim == 2:
            slices = [quantized_data]
            masks = [mask_data]
            slice_indices = [0]

        elif quantized_data.ndim == 3:
            slices = [quantized_data[z] for z in range(quantized_data.shape[0])]
            masks = [mask_data[z] for z in range(mask_data.shape[0])]
            slice_indices = list(range(quantized_data.shape[0]))

        else:
            raise ValueError("run_lengths supports only 2D or 3D arrays.")

        # ---------- line scanner ----------
        def _scan_line(values_line: np.ndarray, mask_line: np.ndarray) -> Dict[Tuple[int, int], int]:
            valid_voxels = (mask_line != 0) & np.isfinite(values_line)

            if not valid_voxels.any():
                return {}

            finite_voxels = values_line[valid_voxels].astype(np.int32, copy=False)

            if finite_voxels.size == 0:
                return {}

            is_run_start = np.empty(finite_voxels.size, dtype=bool)
            is_run_start[0] = True
            is_run_start[1:] = finite_voxels[1:] != finite_voxels[:-1]

            start_indices = np.flatnonzero(is_run_start)
            end_indices = np.append(start_indices[1:], finite_voxels.size)
            run_lengths_array = (end_indices - start_indices).astype(np.int32, copy=False)
            run_levels_array = finite_voxels[start_indices]

            level_length_pairs = np.stack((run_levels_array, run_lengths_array), axis=1)
            unique_pairs, counts_array = np.unique(level_length_pairs, axis=0, return_counts=True)

            return {
                (int(level_val), int(length_val)): int(count_val)
                for (level_val, length_val), count_val in zip(unique_pairs, counts_array)
            }

        scan_line_func = _scan_line

        if njit is not None:

            # Optional Numba JIT version
            @njit(cache=True)  # type: ignore[misc]
            def _compute_run_boundaries(finite_voxels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
                """
                Compute the start indices and lengths of consecutive runs in a 1D array.

                Parameters
                ----------
                finite_voxels : np.ndarray
                    1D array of finite integer values.

                Returns
                -------
                start_indices : np.ndarray
                    Start indices of each run.
                run_lengths : np.ndarray
                    Lengths of each run.
                """
                n_voxels = finite_voxels.size
                is_run_start = np.empty(n_voxels, dtype=np.uint8)
                is_run_start[0] = 1

                for idx in range(1, n_voxels):
                    is_run_start[idx] = 1 if finite_voxels[idx] != finite_voxels[idx - 1] else 0

                num_runs = np.sum(is_run_start)
                start_indices = np.empty(num_runs, dtype=np.int64)
                run_index = 0

                for idx in range(n_voxels):
                    if is_run_start[idx]:
                        start_indices[run_index] = idx
                        run_index += 1

                end_indices = np.empty(num_runs, dtype=np.int64)

                for idx in range(num_runs - 1):
                    end_indices[idx] = start_indices[idx + 1]
                end_indices[num_runs - 1] = n_voxels

                run_lengths = end_indices - start_indices

                return start_indices, run_lengths

            def _scan_line_numba(values_line_numba: np.ndarray, mask_line_numba: np.ndarray) -> Dict[
                Tuple[int, int], int]:
                valid_voxels = (mask_line_numba != 0) & np.isfinite(values_line_numba)

                if not valid_voxels.any():
                    return {}

                finite_voxels = values_line_numba[valid_voxels].astype(np.int32, copy=False)

                if finite_voxels.size == 0:
                    return {}

                run_starts, run_lengths_array = _compute_run_boundaries(finite_voxels)
                run_levels_array = finite_voxels[run_starts]

                level_length_pairs = np.stack((run_levels_array, run_lengths_array.astype(np.int32)), axis=1)
                unique_pairs, counts_array = np.unique(level_length_pairs, axis=0, return_counts=True)

                return {
                    (int(level_val), int(length_val)): int(count_val)
                    for (level_val, length_val), count_val in zip(unique_pairs, counts_array)
                }

            scan_line_func = _scan_line_numba

        # ---------- directional generators ----------
        def _horizontal_lines(slice_data_input: np.ndarray, slice_mask_input: np.ndarray):
            """
            Generator that yields rows of a 2D slice as 1D arrays for horizontal run-length scanning.

            Parameters
            ----------
            slice_data_input : np.ndarray
                2D array representing one slice of the quantized block.
            slice_mask_input : np.ndarray
                2D array mask of the same shape as slice_data.

            Yields
            ------
            row_values_flat : np.ndarray
                1D array of values for the current row.
            row_mask_flat : np.ndarray
                1D array of mask values for the current row.
            """
            for row_values_2d, row_mask_2d in zip(slice_data_input, slice_mask_input):
                yield row_values_2d.ravel(), row_mask_2d.ravel()

        def _vertical_lines(slice_data_input: np.ndarray, slice_mask_input: np.ndarray):
            """
            Generator that yields columns of a 2D slice as 1D arrays for vertical run-length scanning.

            Parameters
            ----------
            slice_data_input : np.ndarray
                2D array representing one slice of the quantized block.
            slice_mask_input : np.ndarray
                2D array mask of the same shape as slice_data.

            Yields
            ------
            col_values_flat : np.ndarray
                1D array of values for the current column.
            col_mask_flat : np.ndarray
                1D array of mask values for the current column.
            """
            for col_values_2d, col_mask_2d in zip(slice_data_input.T, slice_mask_input.T):
                yield col_values_2d.ravel(), col_mask_2d.ravel()

        def _main_diagonals(slice_data_input: np.ndarray, slice_mask_input: np.ndarray):
            """
            Generator that yields main diagonals (↘) of a 2D slice as 1D arrays
            for run-length scanning.

            Parameters
            ----------
            slice_data_input : np.ndarray
                2D array representing one slice of the quantized block.
            slice_mask_input : np.ndarray
                2D array mask of the same shape as slice_data.

            Yields
            ------
            diagonal_values : np.ndarray
                1D array of values along the main diagonal.
            diagonal_mask : np.ndarray
                1D array of mask values along the main diagonal.
            """
            height, width = slice_data_input.shape
            for diag_offset in range(-(height - 1), width):
                yield np.diagonal(slice_data_input, offset=diag_offset), np.diagonal(slice_mask_input, offset=diag_offset)

        def _anti_diagonals(slice_data_input: np.ndarray, slice_mask_input: np.ndarray):
            """
            Generator that yields anti-diagonals (↙) of a 2D slice as 1D arrays
            for run-length scanning.

            Parameters
            ----------
            slice_data_input : np.ndarray
                2D array representing one slice of the quantized block.
            slice_mask_input : np.ndarray
                2D array mask of the same shape as slice_data.

            Yields
            ------
            diagonal_values : np.ndarray
                1D array of values along the anti-diagonal.
            diagonal_mask : np.ndarray
                1D array of mask values along the anti-diagonal.
            """
            flipped_data = np.fliplr(slice_data_input)
            flipped_mask = np.fliplr(slice_mask_input)
            height, width = flipped_data.shape

            for diag_offset in range(-(height - 1), width):
                yield np.diagonal(flipped_data, offset=diag_offset), np.diagonal(flipped_mask, offset=diag_offset)

        line_generators = {
            "0": _horizontal_lines,
            "90": _vertical_lines,
            "135": _main_diagonals,
            "45": _anti_diagonals,
        }

        # ---------- process slices ----------
        for slice_data, slice_mask, slice_idx in zip(slices, masks, slice_indices):
            for direction_key in directions:
                generator_func = line_generators.get(direction_key)
                if generator_func is None:
                    continue

                for line_values_1d, line_mask_1d in generator_func(slice_data, slice_mask):
                    run_dict = scan_line_func(line_values_1d, line_mask_1d)
                    for (run_level, run_length), run_count in run_dict.items():
                        run_length_results.append(
                            {
                                "slice": int(slice_idx),
                                "direction": direction_key,
                                "level": int(run_level),
                                "length": int(run_length),
                                "count": int(run_count),
                            }
                        )

        return run_length_results
