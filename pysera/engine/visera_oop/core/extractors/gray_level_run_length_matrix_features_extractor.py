# -*- coding: utf-8 -*-
# core/extractors/gray_level_run_length_matrix_features_extractor.py

from __future__ import annotations

import logging
import numpy as np
from functools import lru_cache
from typing import Any, Dict, Optional, List, Tuple, Callable, Iterable

from pysera.engine.visera_oop.core.sparsity.view_planner import DataView
from pysera.engine.visera_oop.core.base_feature_extractor import BaseFeatureExtractor

logger = logging.getLogger("Dev_logger")


# -----------------------------------------------------------------------------
# Cached helpers (directions, index vectors)
# -----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _dirs_2d() -> Tuple[Tuple[int, int], ...]:
    return ((1, 0), (0, 1), (1, 1), (1, -1))


@lru_cache(maxsize=1)
def _dirs_3d() -> Tuple[Tuple[int, int, int], ...]:
    # 13 unique 3D directions (IBSI-consistent, positive-first convention)
    return (
        (1, 0, 0), (0, 1, 0), (0, 0, 1),
        (1, 1, 0), (1, -1, 0),
        (1, 0, 1), (1, 0, -1),
        (0, 1, 1), (0, 1, -1),
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
    )


@lru_cache(maxsize=256)
def _idx_vec(n: int) -> np.ndarray:
    return np.arange(1, n + 1, dtype=np.float64)


# -----------------------------------------------------------------------------
# Fast 1D RLE -> (labels, lengths) for a single line
# -----------------------------------------------------------------------------
def _rle_1d(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if arr.size == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    c = np.concatenate(([True], arr[1:] != arr[:-1]))
    starts = np.flatnonzero(c)
    lengths = np.diff(np.append(starts, arr.size))
    labels = arr[starts]
    m = labels > 0
    if not m.any():
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    return labels[m].astype(np.int32, copy=False), lengths[m].astype(np.int32, copy=False)


# -----------------------------------------------------------------------------
# Lines iterators
# -----------------------------------------------------------------------------
def _lines_2d(img: np.ndarray, dx: int, dy: int) -> Iterable[np.ndarray]:
    # 2D already matched your references — keep the fast path
    h, w = img.shape
    if (dx, dy) == (1, 0):      # rows
        for r in range(h):
            yield img[r, :]
    elif (dx, dy) == (0, 1):    # cols
        for c in range(w):
            yield img[:, c]
    elif (dx, dy) == (1, 1):    # main diagonals
        for k in range(-h + 1, w):
            d = np.diagonal(img, offset=k)
            if d.size:
                yield d
    elif (dx, dy) == (1, -1):   # anti-diagonals
        fl = np.fliplr(img)
        for k in range(-h + 1, w):
            d = np.diagonal(fl, offset=k)
            if d.size:
                yield d
    else:
        raise ValueError(f"Unsupported 2D direction {(dx, dy)}")


def _lines_3d(vol: np.ndarray, dx: int, dy: int, dz: int) -> Iterable[np.ndarray]:
    """
    3D iterator FIXED to avoid duplicates/misses:
    A voxel (x,y,z) is a start iff (x-dx, y-dy, z-dz) is out of bounds.
    We then walk forward in (dx,dy,dz) until out of bounds, yielding that line.
    """
    X, Y, Z = vol.shape

    def in_bounds(x: int, y: int, z: int) -> bool:
        return 0 <= x < X and 0 <= y < Y and 0 <= z < Z

    # Enumerate all voxels once and start only at valid "predecessor-out" positions
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                px, py, pz = x - dx, y - dy, z - dz
                if in_bounds(px, py, pz):
                    continue  # not a start, predecessor exists inside
                # walk this line
                a = []
                cx, cy, cz = x, y, z
                while in_bounds(cx, cy, cz):
                    a.append(vol[cx, cy, cz])
                    cx += dx; cy += dy; cz += dz
                if a:
                    yield np.asarray(a, dtype=vol.dtype)


# -----------------------------------------------------------------------------
# Counts accumulation from many lines
# -----------------------------------------------------------------------------
def _accumulate_counts_from_lines(lines: Iterable[np.ndarray], n_labels: int) -> np.ndarray:
    labels_all: List[np.ndarray] = []
    lens_all: List[np.ndarray] = []
    for ln in lines:
        lab, lnth = _rle_1d(ln)
        if lab.size:
            labels_all.append(lab)
            lens_all.append(lnth)

    if not labels_all:
        return np.zeros((n_labels, 1), dtype=np.int64)

    labels_cat = np.concatenate(labels_all)
    lens_cat = np.concatenate(lens_all)

    max_len = int(lens_cat.max())
    flat = (labels_cat - 1) * max_len + (lens_cat - 1)
    counts = np.bincount(flat, minlength=n_labels * max_len).reshape(n_labels, max_len)
    return counts.astype(np.int64, copy=False)


# -----------------------------------------------------------------------------
# Main extractor
# -----------------------------------------------------------------------------
class GrayLevelRunLengthMatrixFeaturesExtractor(BaseFeatureExtractor):
    ACCEPTS: DataView = DataView.DENSE_BLOCK
    PREFERS: DataView = DataView.DENSE_BLOCK

    @staticmethod
    def _nanmean_silent(values: List[float]) -> float:
        if not values:
            return float(np.nan)
        arr = np.asarray(values, dtype=np.float64)
        m = np.nanmean(arr)
        return float(m) if np.isfinite(m) else float(np.nan)

    @staticmethod
    def _pad_to_cols(matrix: np.ndarray, target_width: int) -> np.ndarray:
        n_rows, n_cols = matrix.shape
        if n_cols >= target_width:
            return matrix
        out = np.zeros((n_rows, target_width), dtype=matrix.dtype)
        out[:, :n_cols] = matrix
        return out

    @staticmethod
    def _merge_counts(count_matrices: List[np.ndarray]) -> np.ndarray:
        if not count_matrices:
            return np.zeros((1, 1), dtype=np.int64)
        n = count_matrices[0].shape[0]
        L = max(m.shape[1] for m in count_matrices)
        out = np.zeros((n, L), dtype=np.int64)
        for m in count_matrices:
            if m.shape[1] < L:
                tmp = np.zeros((m.shape[0], L), dtype=m.dtype)
                tmp[:, :m.shape[1]] = m
                m = tmp
            out += m
        return out

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        s = matrix.sum(dtype=np.float64)
        if s <= 0.0:
            return np.zeros_like(matrix, dtype=np.float64)
        return matrix.astype(np.float64, copy=False) / s

    @staticmethod
    def _quant_to_index(arr: np.ndarray, levels: np.ndarray) -> np.ndarray:
        scale = 10000 if levels.size > 100 else 1000
        lv = np.rint(levels.astype(np.float32) * scale) / scale
        x = np.rint(arr.astype(np.float32) * scale) / scale
        order = np.argsort(lv)
        lv_sorted = lv[order]
        pos = np.searchsorted(lv_sorted, x)
        inside = pos < lv_sorted.size
        idx = np.zeros(x.shape, dtype=np.int32)
        if inside.any():
            hits = inside & (x == lv_sorted[np.clip(pos, 0, lv_sorted.size - 1)])
            inv = np.empty_like(order)
            inv[order] = np.arange(order.size, dtype=order.dtype)
            idx[hits] = (inv[np.clip(pos[hits], 0, inv.size - 1)] + 1).astype(np.int32)
        return idx

    # ---------------------- dynamic getters ----------------------
    def __getattr__(self, name: str) -> Callable[[int], float]:
        """
        Modes:
          2D_avg:     per-slice & per-direction P → feature; average over (slices × 4 dirs)
          2D_comb:    per-slice merged (4 dirs) counts → P → feature; average over slices
          2_5D_avg:   per-direction merged-across-slices counts → P → feature; avg over 4 dirs
          2_5D_comb:  merged (4 dirs, all slices) counts → P → feature
          3D_avg:     per 3D-direction P → feature; avg over 13 dirs
          3D_comb:    merged (13 dirs) counts → P → feature

          r_perc EXCEPTIONS (to match your tables):
            - 2D:  mean_{slice,dir}( Ns(slice,dir) / Nv(slice) )
            - 2.5D: dir sums across slices with Nv_total: avg over dirs for *_avg; combined for *_comb
            - 3D:  mean_dir( Ns(dir) / Nv_total )  (also for *_comb)
        """
        if not name.startswith("get_glrlm_"):
            raise AttributeError(f"{type(self).__name__} has no attribute {name}")

        core = name[len("get_glrlm_"):]
        mode, feature = None, None
        for suf in ("2_5D_comb", "2_5D_avg", "3D_comb", "3D_avg", "2D_comb", "2D_avg"):
            if core.endswith(f"_{suf}"):
                mode, feature = suf, core[:-(len(suf) + 1)]
                break
        if mode is None or feature is None:
            logger.error("Invalid GLRLM feature name: %s", name)
            return lambda *_a, **_k: float(np.nan)

        calc_func = getattr(self, f"_calc_{feature}", None)

        def _glnu_from_counts(C: np.ndarray) -> float:
            S = float(C.sum())
            if S <= 0:
                return float(np.nan)
            gs = np.sum(C, axis=1, dtype=np.float64)
            return float(np.sum(gs ** 2) / S)

        def _rlnu_from_counts(C: np.ndarray) -> float:
            S = float(C.sum())
            if S <= 0:
                return float(np.nan)
            ls = np.sum(C, axis=0, dtype=np.float64)
            return float(np.sum(ls ** 2) / S)

        def getter(roi_index: int) -> float:
            self._ensure_cache(roi_index)
            try:
                cache = self._glrlm_cache.get(roi_index) or {}
                quant = self.get_roi(roi_index)
                views = self.get_views(roi_index)

                # P-based features
                if feature not in ("glnu", "rlnu", "r_perc"):
                    if mode == "2D_avg":
                        mats = cache.get("glcm_matrices_2d_dirs") or cache.get("glcm_matrices_2d") or []
                        if not mats:
                            return float(np.nan)
                        return self._nanmean_silent([float(calc_func(P)) for P in mats])
                    if mode == "2D_comb":
                        mats = cache.get("glcm_matrices_2d", [])
                        if not mats:
                            return float(np.nan)
                        return self._nanmean_silent([float(calc_func(P)) for P in mats])
                    if mode == "2_5D_avg":
                        mats = cache.get("glcm_matrices_25d_dirs", [])
                        if not mats:
                            return float(np.nan)
                        return self._nanmean_silent([float(calc_func(P)) for P in mats])
                    if mode == "2_5D_comb":
                        P = cache.get("glcm_matrices_25d")
                        return float(calc_func(P)) if isinstance(P, np.ndarray) and P.size else float(np.nan)
                    if mode == "3D_avg":
                        mats = cache.get("glcm_matrices_3d_dirs", [])
                        if not mats:
                            return float(np.nan)
                        return self._nanmean_silent([float(calc_func(P)) for P in mats])
                    P3 = cache.get("glcm_matrices_3d")
                    return float(calc_func(P3)) if isinstance(P3, np.ndarray) and P3.size else float(np.nan)

                # Counts-based features (GLNU / RLNU / R_PERC)
                mask, levels = self._prepare_mask_and_levels(quant, views)
                if quant is None or mask is None or levels is None or quant.ndim != 3:
                    return float(np.nan)
                n_levels = int(levels.size)
                if n_levels == 0:
                    return float(np.nan)

                # ---------- GLNU / RLNU ----------
                if feature in ("glnu", "rlnu"):
                    reducer = _glnu_from_counts if feature == "glnu" else _rlnu_from_counts

                    if mode == "2D_avg":
                        vals: List[float] = []
                        for z in range(quant.shape[2]):
                            if not np.count_nonzero(mask[:, :, z]):
                                continue
                            idx = self._quant_to_index(quant[:, :, z], levels)
                            idx *= (mask[:, :, z] > 0).astype(np.int32)
                            for dx, dy in _dirs_2d():
                                C = _accumulate_counts_from_lines(_lines_2d(idx, dx, dy), n_levels)
                                vals.append(reducer(C))
                        return self._nanmean_silent(vals)

                    if mode == "2D_comb":
                        vals: List[float] = []
                        for z in range(quant.shape[2]):
                            if not np.count_nonzero(mask[:, :, z]):
                                continue
                            idx = self._quant_to_index(quant[:, :, z], levels)
                            idx *= (mask[:, :, z] > 0).astype(np.int32)
                            Cs = [_accumulate_counts_from_lines(_lines_2d(idx, dx, dy), n_levels) for dx, dy in _dirs_2d()]
                            C = self._merge_counts(Cs)
                            vals.append(reducer(C))
                        return self._nanmean_silent(vals)

                    if mode == "2_5D_avg":
                        acc = [None, None, None, None]
                        for z in range(quant.shape[2]):
                            if not np.count_nonzero(mask[:, :, z]):
                                continue
                            idx = self._quant_to_index(quant[:, :, z], levels)
                            idx *= (mask[:, :, z] > 0).astype(np.int32)
                            for k, (dx, dy) in enumerate(_dirs_2d()):
                                C = _accumulate_counts_from_lines(_lines_2d(idx, dx, dy), n_levels)
                                if acc[k] is None:
                                    acc[k] = C
                                else:
                                    L = max(acc[k].shape[1], C.shape[1])
                                    acc[k] = self._pad_to_cols(acc[k], L)
                                    acc[k][:, :C.shape[1]] += C
                        vals = [reducer(C) for C in acc if C is not None]
                        return self._nanmean_silent(vals)

                    if mode == "2_5D_comb":
                        allC: List[np.ndarray] = []
                        for z in range(quant.shape[2]):
                            if not np.count_nonzero(mask[:, :, z]):
                                continue
                            idx = self._quant_to_index(quant[:, :, z], levels)
                            idx *= (mask[:, :, z] > 0).astype(np.int32)
                            allC.extend(_accumulate_counts_from_lines(_lines_2d(idx, dx, dy), n_levels)
                                        for dx, dy in _dirs_2d())
                        C = self._merge_counts(allC)
                        return reducer(C)

                    if mode == "3D_avg":
                        idx3 = self._quant_to_index(quant, levels)
                        idx3 *= (mask > 0).astype(np.int32)
                        Cs = [_accumulate_counts_from_lines(_lines_3d(idx3, dx, dy, dz), n_levels)
                              for dx, dy, dz in _dirs_3d()]
                        return self._nanmean_silent([reducer(C) for C in Cs])

                    # 3D_comb
                    idx3 = self._quant_to_index(quant, levels)
                    idx3 *= (mask > 0).astype(np.int32)
                    Cs = [_accumulate_counts_from_lines(_lines_3d(idx3, dx, dy, dz), n_levels)
                          for dx, dy, dz in _dirs_3d()]
                    C = self._merge_counts(Cs)
                    return reducer(C)

                # ---------- R_PERC ----------
                if feature == "r_perc":
                    # 2D: average per-slice, per-direction ratios
                    if mode in ("2D_avg", "2D_comb"):
                        vals: List[float] = []
                        for z in range(quant.shape[2]):
                            Nv_slice = int(np.count_nonzero(mask[:, :, z]))
                            if Nv_slice == 0:
                                continue
                            idx = self._quant_to_index(quant[:, :, z], levels)
                            idx *= (mask[:, :, z] > 0).astype(np.int32)
                            for dx, dy in _dirs_2d():
                                C = _accumulate_counts_from_lines(_lines_2d(idx, dx, dy), n_levels)
                                vals.append(float(C.sum()) / Nv_slice)
                        return self._nanmean_silent(vals)

                    # 2.5D: aggregate runs across slices; divide by Nv_total
                    if mode in ("2_5D_avg", "2_5D_comb"):
                        Nv_total = int(np.count_nonzero(mask))
                        if Nv_total == 0:
                            return float(np.nan)
                        # accumulate per-direction across slices
                        acc_dirs: List[Optional[np.ndarray]] = [None, None, None, None]
                        for z in range(quant.shape[2]):
                            if not np.count_nonzero(mask[:, :, z]):
                                continue
                            idx = self._quant_to_index(quant[:, :, z], levels)
                            idx *= (mask[:, :, z] > 0).astype(np.int32)
                            for k, (dx, dy) in enumerate(_dirs_2d()):
                                C = _accumulate_counts_from_lines(_lines_2d(idx, dx, dy), n_levels)
                                if acc_dirs[k] is None:
                                    acc_dirs[k] = C
                                else:
                                    L = max(acc_dirs[k].shape[1], C.shape[1])
                                    acc_dirs[k] = self._pad_to_cols(acc_dirs[k], L)
                                    acc_dirs[k][:, :C.shape[1]] += C
                        dir_vals = [float(C.sum()) / Nv_total for C in acc_dirs if C is not None]
                        if mode == "2_5D_avg":
                            return self._nanmean_silent(dir_vals)
                        # 2_5D_comb: use combined runs across directions
                        total_runs = sum(float(C.sum()) for C in acc_dirs if C is not None)
                        return float(total_runs / Nv_total)

                    # 3D: average over directions of Ns(dir)/Nv_total (also for *_comb)
                    if mode in ("3D_avg", "3D_comb"):
                        Nv_total = int(np.count_nonzero(mask))
                        if Nv_total == 0:
                            return float(np.nan)
                        idx3 = self._quant_to_index(quant, levels)
                        idx3 *= (mask > 0).astype(np.int32)
                        Cs = [_accumulate_counts_from_lines(_lines_3d(idx3, dx, dy, dz), n_levels)
                              for dx, dy, dz in _dirs_3d()]
                        vals = [float(C.sum()) / Nv_total for C in Cs]
                        return self._nanmean_silent(vals)

                return float(np.nan)

            except Exception as e:
                logger.error("Error computing GLRLM feature '%s' for ROI %d: %s", feature, roi_index, e)
                return float(np.nan)

        return getter

    # ---------------------- discovery ----------------------
    def _discover_feature_names(self) -> List[str]:
        methods: List[str] = []
        for cls in self.__class__.mro():
            methods.extend(name for name, obj in cls.__dict__.items() if callable(obj))
        metric_names = [name[len("_calc_"):] for name in methods if name.startswith("_calc_")]
        dims = getattr(self, "_allowed_modes", [])
        return [f"glrlm_{m}_{d}" for d in dims for m in metric_names]

    # ---------------------- cache ----------------------
    def _ensure_cache(self, roi_id: int) -> None:
        if not hasattr(self, "_glrlm_cache"):
            self._glrlm_cache: Dict[int, Dict[str, Any]] = {}
        if roi_id in self._glrlm_cache:
            return

        quant: Optional[np.ndarray] = self.get_roi(roi_id)
        views: Dict[str, Any] = self.get_views(roi_id)
        mask, levels = self._prepare_mask_and_levels(quant, views)

        if quant is None or quant.ndim != 3 or mask is None or levels is None or mask.shape != quant.shape:
            self._glrlm_cache[roi_id] = {}
            logger.warning("[GLRLM] Invalid data for ROI %d", roi_id)
            return

        mats_2d, mats_2d_dirs, prob_25d, prob_25d_dirs = self._build_2d_and_25d_glrlms(quant, mask, levels)
        prob_3d, mats_3d_dirs = self._build_3d_glrlm(quant, mask, levels)

        self._glrlm_cache[roi_id] = {
            "glcm_matrices_2d": mats_2d,                 # per-slice merged (2D_comb) P
            "glcm_matrices_2d_dirs": mats_2d_dirs,       # per-slice per-direction (2D_avg) P
            "glcm_matrices_25d": prob_25d,               # merged 2.5D P
            "glcm_matrices_25d_dirs": prob_25d_dirs,     # per-direction 2.5D P
            "glcm_matrices_3d": prob_3d,                 # merged 3D P
            "glcm_matrices_3d_dirs": mats_3d_dirs,       # per-direction 3D P
        }

    # ---------------------- data prep ----------------------
    @staticmethod
    def _prepare_mask_and_levels(quant_array: Optional[np.ndarray], views: Dict[str, Any]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if quant_array is None or quant_array.ndim != 3:
            return None, None
        mask = views.get("binary_mask")
        levels = views.get("levels")
        if mask is None:
            mask = np.isfinite(quant_array)
        if levels is None or (isinstance(levels, np.ndarray) and levels.size == 0):
            vals = quant_array[np.isfinite(quant_array)]
            levels = np.unique(vals).astype(np.float32) if vals.size else None
        return mask, levels

    # ---------------------- 2D/2.5D builders ----------------------
    def _build_2d_and_25d_glrlms(
        self, quant: np.ndarray, mask: np.ndarray, levels: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray], Optional[np.ndarray], List[np.ndarray]]:
        n = int(levels.size)

        slice_ps_merged: List[np.ndarray] = []
        slice_ps_dirs: List[np.ndarray] = []

        acc_25d_combined: Optional[np.ndarray] = None
        acc_25d_dirs: List[Optional[np.ndarray]] = [None, None, None, None]

        for z in range(quant.shape[2]):
            m = mask[:, :, z]
            if not np.count_nonzero(m):
                continue

            idx = self._quant_to_index(quant[:, :, z], levels)
            idx *= (m > 0).astype(np.int32)

            counts_dirs = [_accumulate_counts_from_lines(_lines_2d(idx, dx, dy), n) for dx, dy in _dirs_2d()]

            merged = self._merge_counts(counts_dirs)
            slice_ps_merged.append(self._normalize(merged))
            slice_ps_dirs.extend([self._normalize(C) for C in counts_dirs])

            # accumulate for 2.5D (raw counts, not normalized)
            if acc_25d_combined is None:
                acc_25d_combined = merged.copy()
            else:
                L = max(acc_25d_combined.shape[1], merged.shape[1])
                acc_25d_combined = self._pad_to_cols(acc_25d_combined, L)
                acc_25d_combined[:, :merged.shape[1]] += merged

            for k, C in enumerate(counts_dirs):
                if acc_25d_dirs[k] is None:
                    acc_25d_dirs[k] = C.copy()
                else:
                    L = max(acc_25d_dirs[k].shape[1], C.shape[1])
                    acc_25d_dirs[k] = self._pad_to_cols(acc_25d_dirs[k], L)
                    acc_25d_dirs[k][:, :C.shape[1]] += C

        prob_25d = self._normalize(acc_25d_combined) if acc_25d_combined is not None else None
        prob_25d_dirs = [self._normalize(C) for C in acc_25d_dirs if C is not None]

        return slice_ps_merged, slice_ps_dirs, prob_25d, prob_25d_dirs

    # ---------------------- 3D builder ----------------------
    def _build_3d_glrlm(
        self, quant: np.ndarray, mask: np.ndarray, levels: np.ndarray
    ) -> Tuple[Optional[np.ndarray], List[np.ndarray]]:
        n = int(levels.size)
        if not np.count_nonzero(mask):
            return None, []

        idx = self._quant_to_index(quant, levels)
        idx *= (mask > 0).astype(np.int32)

        # NEW: unique-lines enumeration via predecessor-out rule
        counts_dirs = [_accumulate_counts_from_lines(_lines_3d(idx, dx, dy, dz), n) for dx, dy, dz in _dirs_3d()]
        p_dirs = [self._normalize(C) for C in counts_dirs]

        merged = self._merge_counts(counts_dirs)
        p_3d = self._normalize(merged)
        return p_3d, p_dirs

    # -------------------------------------------------------------------------
    # GLRLM feature formulas (expect normalized P)
    # -------------------------------------------------------------------------
    def _calc_sre(self, P: np.ndarray) -> float:
        L = _idx_vec(P.shape[1])
        return float(np.sum(P.sum(axis=0) / (L ** 2)))

    def _calc_lre(self, P: np.ndarray) -> float:
        L = _idx_vec(P.shape[1])
        return float(np.sum(P.sum(axis=0) * (L ** 2)))

    def _calc_lgre(self, P: np.ndarray) -> float:
        G = _idx_vec(P.shape[0])
        return float(np.sum(P.sum(axis=1) / (G ** 2)))

    def _calc_hgre(self, P: np.ndarray) -> float:
        G = _idx_vec(P.shape[0])
        return float(np.sum(P.sum(axis=1) * (G ** 2)))

    def _calc_srlge(self, P: np.ndarray) -> float:
        G = _idx_vec(P.shape[0])[:, None]
        L = _idx_vec(P.shape[1])[None, :]
        return float(np.sum(P / (G ** 2 * L ** 2)))

    def _calc_srhge(self, P: np.ndarray) -> float:
        G = _idx_vec(P.shape[0])[:, None]
        L = _idx_vec(P.shape[1])[None, :]
        return float(np.sum((G ** 2 * P) / (L ** 2)))

    def _calc_lrlge(self, P: np.ndarray) -> float:
        G = _idx_vec(P.shape[0])[:, None]
        L = _idx_vec(P.shape[1])[None, :]
        return float(np.sum((L ** 2 * P) / (G ** 2)))

    def _calc_lrhge(self, P: np.ndarray) -> float:
        G = _idx_vec(P.shape[0])[:, None]
        L = _idx_vec(P.shape[1])[None, :]
        return float(np.sum((G ** 2) * (L ** 2) * P))

    def _calc_glnu(self, P: np.ndarray) -> float:
        return float(np.sum(np.sum(P, axis=1, dtype=np.float64) ** 2))

    def _calc_glnu_norm(self, P: np.ndarray) -> float:
        return float(np.sum(np.sum(P, axis=1, dtype=np.float64) ** 2))

    def _calc_rlnu(self, P: np.ndarray) -> float:
        return float(np.sum(np.sum(P, axis=0, dtype=np.float64) ** 2))

    def _calc_rlnu_norm(self, P: np.ndarray) -> float:
        return float(np.sum(np.sum(P, axis=0, dtype=np.float64) ** 2))

    def _calc_r_perc(self, P: np.ndarray, n_voxels: int) -> float:
        total_runs = float(P.sum())
        return float(total_runs / n_voxels) if n_voxels > 0 else float(np.nan)

    def _calc_gl_var(self, P: np.ndarray) -> float:
        gl_prob = np.sum(P, axis=1, dtype=np.float64)
        G = _idx_vec(gl_prob.size)
        mu = float(np.sum(G * gl_prob))
        return float(np.sum(((G - mu) ** 2) * gl_prob))

    def _calc_rl_var(self, P: np.ndarray) -> float:
        rl_prob = np.sum(P, axis=0, dtype=np.float64)
        L = _idx_vec(rl_prob.size)
        mu = float(np.sum(L * rl_prob))
        return float(np.sum(((L - mu) ** 2) * rl_prob))

    def _calc_rl_entropy(self, P: np.ndarray) -> float:
        eps = np.finfo(np.float64).eps
        return float(-(P * np.log2(P + eps)).sum())


# ------------ Thin wrappers selecting allowed modes ------------
class GrayLevelRunLengthMatrixFeatures2DExtractor(GrayLevelRunLengthMatrixFeaturesExtractor):
    NAME: str = "GLRLM2DExtractor"
    _allowed_modes = ["2D_avg", "2D_comb"]


class GrayLevelRunLengthMatrixFeatures25DExtractor(GrayLevelRunLengthMatrixFeaturesExtractor):
    NAME: str = "GLRLM25DExtractor"
    _allowed_modes = ["2_5D_avg", "2_5D_comb"]


class GrayLevelRunLengthMatrixFeatures3DExtractor(GrayLevelRunLengthMatrixFeaturesExtractor):
    NAME: str = "GLRLM3DExtractor"
    _allowed_modes = ["3D_avg", "3D_comb"]
