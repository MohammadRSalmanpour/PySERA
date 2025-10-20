# -*- coding: utf-8 -*-
# core/feature_extraction_manager.py

from __future__ import annotations

import gc
import importlib
import inspect
import json
import logging
import os
import pkgutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import psutil

from .base_feature_extractor import BaseFeatureExtractor
from .sparsity.sparse_policy import SparsePolicy
from .sparsity.view_factory import ViewFactory
from .sparsity.view_planner import DataView, ViewPlanner
from ..data_manager.feature_data_manager import FeatureDataManager

logger = logging.getLogger("Dev_logger")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def log_memory(label: str, cache: dict):
    """Log current memory usage and cache size."""
    proc = psutil.Process()
    mem_kb = proc.memory_info().rss / 1024.0
    cache_keys = len(cache)
    logger.info("[MEMORY] %s | RSS KB: %.1f | cache items: %d | Time: %s",
                label, mem_kb, cache_keys, datetime.now().strftime("%H:%M:%S.%f"))


@dataclass
class ManagerConfig:
    raw_image: np.ndarray
    roi_masks: List[np.ndarray]
    voxel_size_info: Tuple[float, float, float]

    bin_sizes: List[int | float]
    data_type: str
    isotropic_voxel_size_3d: float
    isotropic_voxel_size_2d: float
    discretization_type: str
    quantization_method: str
    voxel_interp: Optional[str]
    roi_interp: Optional[str]
    perform_rescale: bool
    perform_gl_rounding: bool
    quantize_statistics: bool
    perform_resegmentation: bool
    remove_outliers: bool
    use_isotropic_2d: bool
    resegmentation_interval: Tuple[float, float]
    roi_partial_volume: float

    profile_name: str
    feature_dimensions_mask: str = "None"
    extractor_mask: str = "None"

    config_path: Optional[Path] = None
    config_bit_path: Optional[Path] = None

    ivh_configuration: Optional[List[int | float]] = None

    feature_value_mode: str = "REAL_VALUE"
    scale_type: str = "XYZscale"
    backend: str = "scipy"
    align_to_center: bool = True
    zero_frac_threshold: float = 0.85
    min_voxels_for_sparse: int = 1_000_000
    sparse_policy_config: Optional[Dict[str, Any]] = None
    view_planner_config: Optional[Dict[str, Any]] = None
    view_factory_config: Optional[Dict[str, Any]] = None
    max_workers: Optional[int] = None
    log_level: int = logging.INFO


class FeatureExtractionManager:
    """Coordinates preprocessing and executes enabled feature extractors (RAM-only)."""

    @staticmethod
    def _filter_kwargs(callable_object, config: Dict[str, Any]) -> Dict[str, Any]:
        if not config:
            return {}
        try:
            params = inspect.signature(callable_object).parameters
            return {k: v for k, v in config.items() if k in params}
        except (ValueError, TypeError):
            return {}

    def __init__(self, config: ManagerConfig):
        self.config = config
        logger.setLevel(config.log_level)
        self.cache: Dict[str, Any] = {}

        self.raw_image = np.ascontiguousarray(config.raw_image, dtype=np.float32)

        self.roi_masks = [np.asarray(m, dtype=bool) for m in config.roi_masks]
        self.pixel_width_mm, self.pixel_height_mm, self.slice_thickness_mm = config.voxel_size_info

        self.feature_dimensions_mask = config.feature_dimensions_mask
        self.extractor_mask = config.extractor_mask

        default_cfg_dir = Path(__file__).parent.parent / "config" / "materials"
        self.config_path = config.config_path or (default_cfg_dir / "feature_modes_mapping.json")
        self.config_bit_path = config.config_bit_path or (default_cfg_dir / "bit_mappings.json")

        (self.feature_type_bit_mapping,
         self.extractor_bit_mapping,
         self.extractor_group_names,
         self.extractor_group_expansion) = self._load_bit_mappings()

        self.profile = self._load_profile()

        # Profile defaults (names, not class names yet)
        self.order: List[str] = list(self.profile.get("order", []))
        self.feature_type_mapping: Dict[str, List[str]] = dict(self.profile.get("feature_type_mapping", {}))
        self.profile_feature_types: List[str] = list(self.profile.get("active_feature_types", []))
        self.profile_extractors: List[str] = list(self.profile.get("active_extractors", []))

        self._import_all_extractors()

        # 1) Apply masks → compute active types + final extractor name list (dimension-gated ∩ mask)
        self.active_feature_types, self.active_extractors = self._apply_bit_selections()

        # 2) Resolve to classnames (drop invalids)
        self.order = self._resolve_names_to_classnames(self.order)
        self.active_extractors = self._resolve_names_to_classnames(self.active_extractors)

        # 3) Enable only those names (empty list disables all; None would mean "everything")
        BaseFeatureExtractor.enable_extractors_from_list(self.active_extractors)
        self.registry = BaseFeatureExtractor.get_enabled_extractors()
        logger.debug("Enabled extractors: %s", self.active_extractors)

        sp_kwargs = dict(
            zero_frac_threshold=self.config.zero_frac_threshold,
            min_voxels_for_sparse=self.config.min_voxels_for_sparse,
        )
        if self.config.sparse_policy_config:
            sp_kwargs.update(self.config.sparse_policy_config)
        self._sparse_policy = SparsePolicy(**self._filter_kwargs(SparsePolicy.__init__, sp_kwargs))

        self._view_planner = ViewPlanner(
            **self._filter_kwargs(ViewPlanner.__init__, self.config.view_planner_config or {}))
        self._view_factory_cfg = self._filter_kwargs(ViewFactory.__init__, self.config.view_factory_config or {})

    # ---------- config helpers ----------
    def _filter_extractors_by_feature_types(
            self,
            *,
            extractor_names: List[str],
            selected_feature_types: List[str],
            feature_type_mapping: Dict[str, List[str]],
    ) -> List[str]:
        if not feature_type_mapping or not selected_feature_types:
            return extractor_names

        ext2ft: Dict[str, str] = {}
        for ft, ext_list in feature_type_mapping.items():
            for en in ext_list or []:
                ext2ft[en] = ft

        out: List[str] = []
        for en in extractor_names:
            ft = ext2ft.get(en)
            if ft in selected_feature_types:
                out.append(en)
        return out

    def _allowed_extractors_from_dimensions(
            self,
            active_types: List[str],
            ft_map: Dict[str, List[str]],
    ) -> List[str]:
        """
        Union of extractors across all active feature dimensions, preserving profile order.
        """
        if not active_types:
            return []

        # Union (preserve order seen in profile.active_extractors if present)
        union_set = []
        seen = set()
        for ft in active_types:
            for en in ft_map.get(ft, []) or []:
                if en not in seen:
                    union_set.append(en)
                    seen.add(en)
        return union_set

    def _mask_select(self, mask: Optional[str], mapping: List[str]) -> List[str]:
        """
        Apply a positional '01' mask to mapping names; returns selected names.
        If mask is 'None' or None → return [] to signal 'no override'.
        """
        if not mask or mask == "None":
            return []
        sel = [name for i, name in enumerate(mapping) if i < len(mask) and mask[i] == "1"]
        return sel

    def _load_bit_mappings(self) -> Tuple[List[str], List[str], List[str], Dict[str, List[str]]]:
        """
        Load bit mappings.

        Returns:
            feature_dimension_names (list[str]),
            extractor_names (list[str]),
            extractor_group_names (list[str]),
            extractor_group_expansion (dict[str, list[str]])
        """
        feat_list: List[str] = []
        ext_list: List[str] = []
        group_names: List[str] = []
        group_expansion: Dict[str, List[str]] = {}

        try:
            with open(self.config_bit_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
        except Exception as e:
            logger.warning("Bit mapping load issue from %s: %s", self.config_bit_path, e)
            return feat_list, ext_list, group_names, group_expansion

        # Accept BOTH legacy & current keys
        feat_keys = ["feature_dimensions_mask_mapping", "feature_dimensions_mapping"]
        ext_keys = ["extractor_mask_mapping", "extractor_mapping"]

        for k in feat_keys:
            if isinstance(mapping.get(k), list):
                feat_list = mapping[k]
                break
        for k in ext_keys:
            if isinstance(mapping.get(k), list):
                ext_list = mapping[k]
                break

        # NEW: optional group layer
        if isinstance(mapping.get("extractor_group_mapping"), list):
            group_names = mapping["extractor_group_mapping"]

        if isinstance(mapping.get("extractor_group_expansion"), dict):
            # normalize values to list[str]
            for g, v in mapping["extractor_group_expansion"].items():
                if isinstance(v, list) and all(isinstance(x, str) for x in v):
                    group_expansion[g] = v

        # Validate types
        if not all(isinstance(x, str) for x in feat_list):
            logger.warning("feature-dimension mapping malformed; falling back to empty.")
            feat_list = []
        if not all(isinstance(x, str) for x in ext_list):
            logger.warning("extractor mapping malformed; falling back to empty.")
            ext_list = []
        if not all(isinstance(x, str) for x in group_names):
            logger.warning("extractor_group_mapping malformed; ignoring.")
            group_names = []
            group_expansion = {}

        if not feat_list:
            logger.warning("Feature-dimension mapping is empty. Positional masks cannot be applied.")
        if not ext_list:
            logger.warning("Extractor mapping is empty. Positional masks cannot be applied.")

        # Groups are optional; ok if absent.
        return feat_list, ext_list, group_names, group_expansion

    def _apply_bit_selections(self) -> Tuple[List[str], List[str]]:
        """
        Returns (active_feature_types, active_extractors) honoring both masks
        and now also supporting group masks (e.g., 13-bit) that expand into
        concrete extractors.
        """
        # -------- detect whether masks were explicitly provided ----------
        type_mask_provided = bool(self.feature_dimensions_mask) and self.feature_dimensions_mask != "None"
        ext_mask_provided = bool(self.extractor_mask) and self.extractor_mask != "None"

        # -------- Step A: resolve active feature types ----------
        types_from_mask = self._mask_select(self.feature_dimensions_mask, self.feature_type_bit_mapping)
        if type_mask_provided:
            active_feature_types = types_from_mask
        else:
            active_feature_types = list(self.profile_feature_types)

        # -------- Step B: dimension gating (union of extractors in active types) ----------
        allowed_by_dims = self._allowed_extractors_from_dimensions(active_feature_types, self.feature_type_mapping)

        # Keep intersection with profile's allowed extractors if profile lists them explicitly
        if self.profile_extractors:
            allowed_by_dims = [e for e in allowed_by_dims if e in self.profile_extractors]

        # -------- Step C: apply extractor mask (per-extractor OR group) ----------
        active_extractors: List[str]

        if ext_mask_provided:
            mask_str = self.extractor_mask
            # Try to interpret as a GROUP mask first if lengths match
            used_group_mask = False
            if getattr(self, "extractor_group_names", None):
                if len(mask_str) == len(self.extractor_group_names):
                    selected_groups = self._mask_select(mask_str, self.extractor_group_names)
                    expanded: List[str] = []
                    seen = set()
                    for g in selected_groups:
                        for name in self.extractor_group_expansion.get(g, []):
                            if name not in seen:
                                expanded.append(name)
                                seen.add(name)
                    # Intersect with dimension gating
                    mask_set = set(expanded)
                    active_extractors = [e for e in allowed_by_dims if e in mask_set]
                    used_group_mask = True
                # else: fall through to per-extractor logic

            if not used_group_mask:
                # classic: mask points at the concrete extractor list
                exts_from_mask = self._mask_select(mask_str, self.extractor_bit_mapping)
                mask_set = set(exts_from_mask)
                active_extractors = [e for e in allowed_by_dims if e in mask_set]

        else:
            # no extractor mask: honor only dimension gating
            active_extractors = allowed_by_dims

        # -------- Step D: de-duplicate & keep profile order if available ----------
        if self.profile.get("order"):
            order_ref = list(self.profile["order"])
            active_extractors = [e for e in order_ref if e in active_extractors]
        else:
            seen = set()
            active_extractors = [e for e in allowed_by_dims if
                                 e in active_extractors and not (e in seen or seen.add(e))]

        logger.debug("Active feature types: %s", active_feature_types)
        logger.debug("Allowed (by dims): %s", allowed_by_dims)
        logger.debug("Final active extractors: %s", active_extractors)

        return active_feature_types, active_extractors

    def _load_profile(self) -> Dict[str, Any]:
        with open(self.config_path, "r", encoding="utf-8") as f:
            profiles = json.load(f).get("profiles", {})
        profile = profiles.get(self.config.profile_name)
        if profile is None:
            raise ValueError(f"Profile '{self.config.profile_name}' not found in {self.config_path}")
        return profile

    @staticmethod
    def _package_root() -> str:
        return __package__ or "pysera.engine.visera_oop.core"

    def _import_all_extractors(self, package_name: Optional[str] = None) -> None:
        base_package = self._package_root()
        pkg_name = package_name or f"{base_package}.extractors"
        try:
            pkg = importlib.import_module(pkg_name)
            if hasattr(pkg, "__path__"):
                for _, module_name, is_pkg in pkgutil.iter_modules(pkg.__path__, pkg_name + "."):
                    if not is_pkg:
                        importlib.import_module(module_name)
        except Exception as e:
            logger.warning("Extractor package import issue '%s': %s", pkg_name, e)
            for m in ("statistics_features_extractor", "intensity_peak_features_extractor"):
                try:
                    importlib.import_module(f"{base_package}.extractors.{m}")
                except Exception:
                    pass

    @staticmethod
    def _resolve_to_classname(extractor_name: str) -> Optional[str]:
        reg = BaseFeatureExtractor.EXTRACTOR_REGISTRY
        if extractor_name in reg:
            return extractor_name
        cand = f"{extractor_name}FeatureExtractor"
        if cand in reg:
            return cand
        for clsname, cls in reg.items():
            if getattr(cls, "NAME", None) == extractor_name:
                return clsname
        return None

    def _resolve_names_to_classnames(self, names: List[str]) -> List[str]:
        out: List[str] = []
        for n in names:
            c = self._resolve_to_classname(n)
            if c:
                out.append(c)
        return out

    # ---------- preprocessing ----------

    def _preprocess_roi(
            self,
            roi_mask: np.ndarray,
            bin_size: int | float,
            bin_index: int = 0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:

        fdm = FeatureDataManager(
            image=self.raw_image,
            roi_mask_array=np.asarray(roi_mask, dtype=np.float32),
            pixel_width=self.pixel_width_mm,
            slice_thickness=self.slice_thickness_mm,
            data_type=self.config.data_type,
            isotropic_voxel_size=self.config.isotropic_voxel_size_3d,
            roi_partial_volume=self.config.roi_partial_volume,
            scale_type=self.config.scale_type,
            is_isotropic_2d=self.config.use_isotropic_2d,
            apply_scaling=self.config.perform_rescale,
            apply_rounding=self.config.perform_gl_rounding,
            discretization_type=self.config.discretization_type,
            quantizer_type=self.config.quantization_method,
            apply_resegmentation=self.config.perform_resegmentation,
            resegmentation_interval=self.config.resegmentation_interval,
            remove_outliers=self.config.remove_outliers,
        )

        state = fdm.prepare_volume_for_bin(
            bin_size=bin_size,
            bin_index=bin_index,
            isotropic_voxel_size_2d=self.config.isotropic_voxel_size_2d,
            voxel_interp=(self.config.voxel_interp or "linear"),
            roi_interp=(self.config.roi_interp or "nearest"),
        )

        dense = np.asarray(state["intensity_mask"], dtype=np.float32)
        quant = np.asarray(state["quantized"], dtype=np.float32)

        mask = state.get("morph_mask_resampled")
        if mask is not None:
            m = np.asarray(mask)
            if m.dtype == bool:
                binmask = m.astype(np.uint8, copy=False)
            else:
                m = m.astype(np.float32, copy=False)
                pv = float(self.config.roi_partial_volume or 0.0)

                # If mask is effectively binary already, keep it binary.
                if np.all((m >= -1e-6) & (m <= 1 + 1e-6)) and \
                        np.all(np.isclose(m, 0.0, atol=1e-6) | np.isclose(m, 1.0, atol=1e-6)):
                    binmask = (m >= 0.5).astype(np.uint8, copy=False)
                else:
                    # Avoid linear-fringe flood for PV==0.0 by using 0.5 when ROIInterp is not nearest
                    if pv <= 0.0 and (self.config.roi_interp and self.config.roi_interp.lower() != "nearest"):
                        thr = 0.5
                    else:
                        thr = pv
                    binmask = (m >= thr).astype(np.uint8, copy=False)
        else:
            binmask = np.isfinite(dense).astype(np.uint8, copy=False)

        # augment raw
        state["_raw_image_full"] = self.raw_image
        state["_raw_roi_mask_full"] = roi_mask.astype(bool)
        return dense, quant, binmask, state

    # ---------- views ----------

    def _compute_voxel_spacing_cm(self) -> tuple[float, float, float]:
        return self.slice_thickness_mm / 10.0, self.pixel_height_mm / 10.0, self.pixel_width_mm / 10.0

    def _build_views_dict(
            self,
            *,
            dense_block: np.ndarray,
            quantized_block: np.ndarray,
            binary_mask: np.ndarray,
            fdm_state: Optional[Dict[str, Any]],
            required_views: int,
    ) -> Dict[DataView | str, Any]:
        """
        Build the view's dictionary. Ensures both CROPPED (for most extractors) and FULL
        (for diagnostics) representations are available and shape-consistent.

        Conventions:
          - Arrays are (z, y, x)
          - crop_bbox is stored as (x_min, y_min, z_min, x_max, y_max, z_max)
        """
        fdm_state = fdm_state or {}

        # If the intensity block inside ROI is all-NaN, but we have an unmasked image,
        # fill ROI with unmasked intensities as a safe fallback.
        # Pick the first valid 3D array, don't use Python `or` on numpy arrays
        unmasked_pref = None
        for k in ("image_resampled_unmasked_crop", "image_resampled"):
            arr = fdm_state.get(k)
            if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.size > 0:
                unmasked_pref = arr
                break

        if (
                (not np.isfinite(dense_block).any())
                and (np.count_nonzero(binary_mask) > 0)
                and isinstance(unmasked_pref, np.ndarray)
        ):
            logger.info(
                "Intensity block is all-NaN inside ROI; using unmasked resampled image within ROI as fallback."
            )
            dense_block = np.where(binary_mask.astype(bool), unmasked_pref, np.nan).astype(np.float32, copy=False)

        view_factory = ViewFactory(
            intensity_block=dense_block,
            roi_mask=binary_mask,
            quantized_block=quantized_block,
            **self._view_factory_cfg
        )

        views: Dict[DataView | str, Any] = {DataView.DENSE_BLOCK: view_factory.dense_block(),
                                            DataView.ROI_VECTOR: view_factory.roi_vector(),
                                            "binary_mask": np.asarray(binary_mask, dtype=np.uint8, order="C"),
                                            "levels": (fdm_state.get("levels")),
                                            "quantized_block": np.asarray(quantized_block, dtype=np.float32, order="C"),
                                            "use_quantized_for_statistics": bool(
                                                getattr(self.config, "quantize_statistics", False))}

        # NEW: expose the quantized block & quantization meta

        # If FDM provides quantization info, forward it (optional)
        for k in ("discretization_type", "quantization_method"):
            if k in fdm_state:
                views[k] = fdm_state[k]


        # ---- Resolve FULL resampled image robustly (unmasked), expose under BOTH keys ----
        full_candidates = [
            fdm_state.get("image_resampled_full"),
            fdm_state.get("image_resampled_unmasked_full"),
            fdm_state.get("image_resampled_unmasked"),
            fdm_state.get("image_resampled"),
        ]
        img_full = next(
            (a for a in full_candidates if isinstance(a, np.ndarray) and a.ndim == 3 and a.size > 0),
            None,
        )
        if isinstance(img_full, np.ndarray):
            img_full = img_full.astype(np.float32, copy=False)

        # Both names point to the same full grid (if available)
        views["image_resampled"] = img_full
        views["image_resampled_full"] = img_full

        # ---- Crop unmasked image to align with DENSE_BLOCK (if explicit crop not provided) ----
        cropped_unmasked = fdm_state.get("image_resampled_unmasked_crop")
        if not (isinstance(cropped_unmasked, np.ndarray) and cropped_unmasked.ndim == 3):
            bbox = None
            for key in ("crop_bbox", "resampled_crop_bbox", "roi_crop_bbox"):
                cb = fdm_state.get(key)
                if isinstance(cb, (tuple, list)) and len(cb) == 6:
                    bbox = tuple(map(int, cb))
                    break

            if isinstance(img_full, np.ndarray) and bbox is not None:
                # bbox: (x_min, y_min, z_min, x_max, y_max, z_max)
                x0, y0, z0, x1, y1, z1 = bbox
                zf, yf, xf = img_full.shape

                # Clamp to bounds
                z0 = max(0, min(z0, zf))
                z1 = max(0, min(z1, zf))
                y0 = max(0, min(y0, yf))
                y1 = max(0, min(y1, yf))
                x0 = max(0, min(x0, xf))
                x1 = max(0, min(x1, xf))

                if (z1 > z0) and (y1 > y0) and (x1 > x0):
                    cropped_unmasked = img_full[z0:z1, y0:y1, x0:x1]

        if not (isinstance(cropped_unmasked, np.ndarray) and cropped_unmasked.ndim == 3):
            # Last resort: use dense_block with NaNs replaced
            cropped_unmasked = np.nan_to_num(dense_block, nan=0.0)

        views["dense_block_unmasked"] = np.asarray(cropped_unmasked, dtype=np.float32, order="C")

        # ---- crop bbox (x_min, y_min, z_min, x_max, y_max, z_max) ----
        crop_bbox = None
        for key in ("crop_bbox", "resampled_crop_bbox", "roi_crop_bbox"):
            cb = fdm_state.get(key)
            if isinstance(cb, (tuple, list)) and len(cb) == 6:
                crop_bbox = tuple(map(int, cb))
                break
        views["crop_bbox"] = crop_bbox  # may be None

        # ---- FULL binary mask (z,y,x), rebuilt from cropped mask + bbox; else pass-through if same shape ----
        binary_mask_full = None
        if isinstance(img_full, np.ndarray):
            if crop_bbox is not None and isinstance(binary_mask, np.ndarray) and binary_mask.ndim == 3:
                x0, y0, z0, x1, y1, z1 = crop_bbox
                zf, yf, xf = img_full.shape

                # Clamp bbox to image bounds
                z0 = max(0, min(z0, zf))
                z1 = max(0, min(z1, zf))
                y0 = max(0, min(y0, yf))
                y1 = max(0, min(y1, yf))
                x0 = max(0, min(x0, xf))
                x1 = max(0, min(x1, xf))

                # Extents that fit both bbox and cropped mask
                dz = max(0, min(z1 - z0, binary_mask.shape[0], zf - z0))
                dy = max(0, min(y1 - y0, binary_mask.shape[1], yf - y0))
                dx = max(0, min(x1 - x0, binary_mask.shape[2], xf - x0))

                full_m = np.zeros_like(img_full, dtype=np.uint8)
                if dz > 0 and dy > 0 and dx > 0:
                    full_m[z0:z0 + dz, y0:y0 + dy, x0:x0 + dx] = (binary_mask[:dz, :dy, :dx] > 0).astype(np.uint8,
                                                                                                         copy=False)
                binary_mask_full = full_m

            elif isinstance(binary_mask, np.ndarray) and binary_mask.shape == img_full.shape:
                # Already full-sized
                binary_mask_full = (binary_mask > 0).astype(np.uint8, copy=False)

        views["binary_mask_full"] = binary_mask_full  # may be None

        # ---- Spacing (effective and initial): derive from manager config (not from FDM) ----
        dx_mm, dy_mm, dz_mm = self._effective_voxel_spacing_mm()

        # Optionally log if FDM provided different spacing (no override; for visibility only)
        fdm_vox = fdm_state.get("voxel_spacing_mm")
        if fdm_vox is not None:
            try:
                fdx, fdy, fdz = float(fdm_vox[0]), float(fdm_vox[1]), float(fdm_vox[2])
                tol = 1e-6
                if max(abs(fdx - dx_mm), abs(fdy - dy_mm), abs(fdz - dz_mm)) > tol:
                    lvl = logging.DEBUG if not self.config.perform_rescale else logging.INFO
                    logger.log(
                        lvl,
                        "Effective spacing (mm) from config: dx=%.3f, dy=%.3f, dz=%.3f; FDM reported: (%s)",
                        dx_mm, dy_mm, dz_mm, fdm_vox
                    )
            except Exception:
                logger.debug("Non-numeric fdm_vox: %r", fdm_vox)

        # Algorithms expect (dz,dy,dx) in CM for array (z,y,x)
        views["voxel_spacing_cm"] = (dz_mm / 10.0, dy_mm / 10.0, dx_mm / 10.0)
        logger.debug("Effective voxel spacing (mm): dx=%.3f, dy=%.3f, dz=%.3f", dx_mm, dy_mm, dz_mm)

        # Raw inputs & initial spacing (pre-resample)
        views["raw_image_full"] = fdm_state.get("_raw_image_full")
        views["raw_roi_mask_full"] = fdm_state.get("_raw_roi_mask_full")
        dz0 = self.slice_thickness_mm / 10.0
        dy0 = self.pixel_height_mm / 10.0
        dx0 = self.pixel_width_mm / 10.0
        views["voxel_spacing_cm_initial"] = (dz0, dy0, dx0)

        # ---- Optional views for sparse workflows ----
        if required_views & DataView.SPARSE_COORDS:
            views[DataView.SPARSE_COORDS] = view_factory.sparse_coords()
        if required_views & DataView.RUN_LENGTHS:
            views[DataView.RUN_LENGTHS] = view_factory.run_lengths()

        return views

    def _effective_voxel_spacing_mm(self) -> tuple[float, float, float]:
        """
        Returns (dx_mm, dy_mm, dz_mm) derived from manager config and
        original header spacing, independent of FDM's state.
        """
        dx0, dy0, dz0 = float(self.pixel_width_mm), float(self.pixel_height_mm), float(self.slice_thickness_mm)

        # No rescaling → keep original header spacing
        if not self.config.perform_rescale:
            return dx0, dy0, dz0

        iso3d = float(self.config.isotropic_voxel_size_3d)
        iso2d = float(self.config.isotropic_voxel_size_2d) if self.config.isotropic_voxel_size_2d else iso3d

        # 2D isotropic (in-plane only)
        if self.config.use_isotropic_2d:
            return iso2d, iso2d, dz0

        # 3D isotropic
        return iso3d, iso3d, iso3d

    @staticmethod
    def _extractor_view_payload(
            extractor_class: Type[BaseFeatureExtractor],
            available_views: Dict[DataView | str, Any],
    ) -> Dict[str, Any]:

        accepts = getattr(extractor_class, "ACCEPTS", DataView.DENSE_BLOCK)
        prefers = getattr(extractor_class, "PREFERS", accepts)
        mask = int(prefers or accepts)

        payload: Dict[str, Any] = {}
        if mask & DataView.ROI_VECTOR:
            payload["roi_vector"] = available_views.get(DataView.ROI_VECTOR)
        if mask & DataView.DENSE_BLOCK:
            payload["dense_block"] = available_views.get(DataView.DENSE_BLOCK)
        if mask & getattr(DataView, "DENSE_BLOCK_UNMASKED", 0):
            payload["dense_block_unmasked"] = available_views.get("dense_block_unmasked")
        if mask & DataView.SPARSE_COORDS:
            payload["sparse_coords"] = available_views.get(DataView.SPARSE_COORDS)
        if mask & DataView.RUN_LENGTHS:
            payload["run_lengths"] = available_views.get(DataView.RUN_LENGTHS)

        # Spacing
        payload["voxel_spacing_cm"] = available_views.get("voxel_spacing_cm")
        payload["voxel_spacing_cm_initial"] = available_views.get("voxel_spacing_cm_initial")

        # Raw/full variants
        payload["raw_image_full"] = available_views.get("raw_image_full")
        payload["raw_roi_mask_full"] = available_views.get("raw_roi_mask_full")
        payload["raw_bbox_slices"] = available_views.get("raw_bbox_slices")
        payload["raw_block"] = available_views.get("raw_block")
        payload["raw_mask_block"] = available_views.get("raw_mask_block")

        # Resampled grid (full + bbox) and masks
        payload["image_resampled"] = available_views.get("image_resampled")
        payload["crop_bbox"] = available_views.get("crop_bbox")
        payload["binary_mask"] = available_views.get("binary_mask")
        payload["binary_mask_full"] = available_views.get("binary_mask_full")

        payload["levels"] = available_views.get("levels")

        # Quantization: always pass; extractor may ignore
        payload["quantized_block"] = available_views.get("quantized_block")
        payload["use_quantized_for_statistics"] = available_views.get("use_quantized_for_statistics")
        payload["discretization_type"] = available_views.get("discretization_type")
        payload["quantization_method"] = available_views.get("quantization_method")
        return payload

    # ---------- orchestrate extractors ----------
    def _resolve_extractor_classes(self) -> List[Tuple[str, Type[BaseFeatureExtractor]]]:
        """Return (name, class) in the exact order we decided in _apply_bit_selections()."""
        reg = BaseFeatureExtractor.get_enabled_extractors()
        return [(n, reg[n]) for n in self.active_extractors if n in reg]

    @staticmethod
    def _to_primitive(v: Any) -> Any:
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.bool_):
            return bool(v)
        if isinstance(v, (list, tuple)):
            return [FeatureExtractionManager._to_primitive(x) for x in v]
        return v

    @staticmethod
    def _evaluate_feature_map(ext_instance: Any, fmap: Dict[str, Any]) -> Dict[str, Any]:
        def eval_item(iv: Any) -> Any:
            if isinstance(iv, str):
                attr = getattr(ext_instance, iv, None)
                return FeatureExtractionManager._to_primitive(attr()) if callable(attr) else iv
            if callable(iv):
                return FeatureExtractionManager._to_primitive(iv())
            return FeatureExtractionManager._to_primitive(iv)

        return {k: eval_item(v) for k, v in fmap.items()}

    # ---------- public ----------
    # def run(self, image_name: str = "image_0") -> Dict[str, Any]:
    #     """Run all enabled feature extractors on all ROIs and bin sizes, returning results
    #     and performance statistics, while aggressively freeing memory to minimize usage.
    #     """
    #     results: Dict[str, Any] = {image_name: []}
    #     proc = psutil.Process()
    #     wall_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    #     cpu_start = psutil.cpu_times().user
    #     mem_start_kb = proc.memory_info().rss / 1024.0
    #
    #     exts = self._resolve_extractor_classes()
    #     if not exts:
    #         logger.warning("No extractors enabled; returning empty result.")
    #         results[image_name + "_overall_perf"] = {
    #             "start_time": wall_start,
    #             "end_time": wall_start,
    #             "total_time_sec": 0.0,
    #             "start_memory_kb": mem_start_kb,
    #             "end_memory_kb": mem_start_kb,
    #             "total_memory_kb": 0.0,
    #         }
    #         return results
    #
    #     max_workers = self.config.max_workers or min(8, os.cpu_count() or 1)
    #
    #     for r_idx, roi_mask in enumerate(self.roi_masks):
    #         for _, bin_size in enumerate(self.config.bin_sizes):
    #             # ---- Preprocess ROI ----
    #             try:
    #                 dense, quant, binmask, fdm_state = self._preprocess_roi(roi_mask, bin_size, r_idx)
    #             except Exception as exc:
    #                 logger.error("Preprocessing failed (ROI %d, bin %s): %s", r_idx, bin_size, exc)
    #                 continue
    #
    #             if binmask is None or np.count_nonzero(binmask) == 0:
    #                 logger.warning("ROI %d bin %s empty; skipping.", r_idx, bin_size)
    #                 del dense, quant, binmask, fdm_state
    #                 gc.collect()
    #                 continue
    #
    #             # ---- Determine required views ----
    #             use_sparse = self._sparse_policy.should_use_sparse(dense, binmask)
    #             class_list = [cls for _, cls in exts]
    #             plan = self._view_planner.build_plan(class_list, use_sparse)
    #             views = self._build_views_dict(
    #                 dense_block=dense,
    #                 quantized_block=quant,
    #                 binary_mask=binmask,
    #                 fdm_state=fdm_state,
    #                 required_views=plan.required,
    #             )
    #
    #             # ---- Run extractors in parallel ----
    #             roi_vals: Dict[str, Any] = {}
    #             roi_feat_perf: Dict[str, Dict[str, Any]] = {}
    #             roi_perf_log: Dict[str, Any] = {}
    #
    #             def run_one(idx: int, name: str, cls: Type[BaseFeatureExtractor]):
    #                 inst = cls(cache=self.cache, feature_value_mode=self.config.feature_value_mode)
    #                 payload = self._extractor_view_payload(cls, views)
    #                 out = inst.extract(image=dense, roi=quant, views=payload, roi_index=r_idx, selected_features=None)
    #                 return idx, name, inst, out
    #
    #             buf: list[tuple[str, Any, Dict[int, Dict[str, Any]]]] = [None] * len(exts)  # temporary storage
    #             with ThreadPoolExecutor(max_workers=max_workers) as pool:
    #                 futures = [pool.submit(run_one, idx, name, cls) for idx, (name, cls) in enumerate(exts)]
    #                 for fut in futures:
    #                     idx, name, inst, out = fut.result()
    #                     buf[idx] = (name, inst, out)
    #
    #             # ---- Collect results in original order ----
    #             for idx in range(len(exts)):
    #                 name, inst, out = buf[idx]
    #                 vals_for_roi = out.get(r_idx) or out.get(0) or {}
    #                 if vals_for_roi:
    #                     roi_vals.update(self._evaluate_feature_map(inst, vals_for_roi))
    #
    #                 if hasattr(inst, "last_perf"):
    #                     roi_perf_log[name] = dict(getattr(inst, "last_perf"))
    #                 if hasattr(inst, "last_feature_perf"):
    #                     roi_feat_perf[name] = dict(getattr(inst, "last_feature_perf"))
    #
    #                 # ---- Delete extractor instance immediately ----
    #                 del inst
    #                 gc.collect()
    #
    #             results[image_name].append({
    #                 "roi_index": r_idx,
    #                 "bin_size": bin_size,
    #                 "value": roi_vals,
    #                 "feature_perf": roi_feat_perf,
    #                 "roi_perf": roi_perf_log,
    #             })
    #
    #             # ---- Delete large temporary arrays ----
    #             del dense, quant, binmask, fdm_state, views, buf, roi_vals, roi_feat_perf, roi_perf_log
    #             # ---- Clear cache for next ROI ----
    #             self.cache.clear()
    #             gc.collect()
    #
    #     # ---- Final memory/cpu stats ----
    #     wall_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    #     cpu_end = psutil.cpu_times().user
    #     mem_end_kb = proc.memory_info().rss / 1024.0
    #
    #     results[image_name + "_overall_perf"] = {
    #         "start_time": wall_start,
    #         "end_time": wall_end,
    #         "total_time_sec": round(cpu_end - cpu_start, 6),
    #         "start_memory_kb": round(mem_start_kb, 6),
    #         "end_memory_kb": round(mem_end_kb, 6),
    #         "total_memory_kb": round(mem_end_kb - mem_start_kb, 6),
    #     }
    #
    #     return results

    def _selected_features_for(self, cls: Type[BaseFeatureExtractor]) -> Optional[list[str]]:
        sel = getattr(self.config, "features_to_extract", None)
        if sel is None:
            return None
        # list -> same subset for all extractors
        if isinstance(sel, (list, tuple)):
            return list(sel)
        # dict -> per-extractor subset
        if isinstance(sel, dict):
            # match by class name or by NAME alias if present
            clsname = cls.__name__
            alias = getattr(cls, "NAME", None)
            for key, feats in sel.items():
                if key == clsname or (alias and key == alias):
                    return list(feats)
        return None

    def run(self, image_name: str = "image_0", roi_name: str = "label_0_lesion_0") -> Dict[str, Any]:
        """
        Run all enabled feature extractors on all ROIs and bin sizes,
        while logging memory usage and cache size continuously.
        """
        results: Dict[str, Any] = {image_name: []}
        proc = psutil.Process()

        wall_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        cpu_start = psutil.cpu_times().user
        mem_start_kb = proc.memory_info().rss / 1024.0

        exts = self._resolve_extractor_classes()
        if not exts:
            results[image_name + "_overall_perf"] = {
                "start_time": wall_start,
                "end_time": wall_start,
                "total_time_sec": 0.0,
                "total_memory_kb": 0.0,
                "max_memory_kb": 0.0,
                "memory_log": [],
            }
            return results

        max_workers = self.config.max_workers or min(8, os.cpu_count() or 1)
        memory_log: List[Dict[str, Any]] = []
        max_mem_kb = mem_start_kb

        # --- single executor for all ROIs ---
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for r_idx, roi_mask in enumerate(self.roi_masks):
                for bin_size in self.config.bin_sizes:

                    mem_before_kb = proc.memory_info().rss / 1024.0

                    try:
                        dense, quant, binmask, fdm_state = self._preprocess_roi(roi_mask, bin_size, r_idx)
                    except Exception as exc:
                        logger.warning("Preprocessing failed ROI %d, bin %s: %s", r_idx, bin_size, exc)
                        continue

                    if binmask is None or np.count_nonzero(binmask) == 0:
                        del dense, quant, binmask, fdm_state
                        gc.collect()
                        continue

                    use_sparse = self._sparse_policy.should_use_sparse(dense, binmask)
                    class_list = [cls for _, cls in exts]
                    plan = self._view_planner.build_plan(class_list, use_sparse)
                    views = self._build_views_dict(
                        dense_block=dense,
                        quantized_block=quant,
                        binary_mask=binmask,
                        fdm_state=fdm_state,
                        required_views=plan.required
                    )

                    def run_one(cls: Type[BaseFeatureExtractor]):
                        extractor_name = cls.__name__
                        inst = cls(cache=self.cache, feature_value_mode=self.config.feature_value_mode)
                        try:
                            payload = self._extractor_view_payload(cls, views)
                            selected = self._selected_features_for(cls)
                            out = inst.extract(image=dense, roi=quant, views=payload, roi_index=r_idx,
                                               selected_features=selected)
                            vals: Dict[str, Any] = {}
                            for vmap in out.values():
                                for k, v in vmap.items():
                                    vals[k] = self._to_primitive(v)
                            logger.info(
                                f"[%s] ROI '%s' Extractor {extractor_name} completed successfully", image_name,
                                roi_name
                            )
                            return vals

                        except Exception as e:
                            logger.error(
                                f"[%s] ROI '%s' Extractor {extractor_name} failed : {e}", image_name, roi_name
                            )
                            return {}
                        finally:
                            del inst

                    futures = [pool.submit(run_one, cls) for _, cls in exts]
                    roi_vals: Dict[str, Any] = {}
                    for fut in futures:
                        roi_vals.update(fut.result())

                    mem_after_kb = proc.memory_info().rss / 1024.0
                    max_mem_kb = max(max_mem_kb, mem_after_kb)
                    memory_log.append({
                        "roi_index": r_idx,
                        "bin_size": bin_size,
                        "mem_before_kb": mem_before_kb,
                        "mem_after_kb": mem_after_kb,
                        "cache_size": len(self.cache)
                    })

                    results[image_name].append({
                        "roi_index": r_idx,
                        "bin_size": bin_size,
                        "value": roi_vals
                    })

                    # --- cleanup ---
                    del dense, quant, binmask, fdm_state, views, roi_vals
                    self.cache.clear()
                    gc.collect()

        wall_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        cpu_end = psutil.cpu_times().user
        mem_end_kb = proc.memory_info().rss / 1024.0

        results[image_name + "_overall_perf"] = {
            "start_time": wall_start,
            "end_time": wall_end,
            "total_time_sec": round(cpu_end - cpu_start, 6),
            "total_memory_kb": round(mem_end_kb - mem_start_kb, 6),
            "max_memory_kb": round(max_mem_kb, 6),
            "memory_log": memory_log
        }

        return results
