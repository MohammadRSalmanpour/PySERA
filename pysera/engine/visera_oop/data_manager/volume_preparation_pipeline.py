# -*- coding: utf-8 -*-
# data_manager/volume_preparation_pipeline.py

from __future__ import annotations

import logging
import numpy as np
from typing import Any, Dict, Optional, Tuple, Type

from .resampling.volume_resizer import VolumeResizer
from .quantization.lloyd_quantizer import LloydQuantizer
from .quantization.uniform_quantizer import UniformQuantizer
from .quantization.fixed_bin_size_quantizer import FixedBinSizeQuantizer


class VolumePreparationPipeline:
    """
    IBSI-aligned 3D preparation pipeline (RAM-only).

    Stages:
      1) Select quantizer (FBN/FBS/Lloyd)
      2) Load ROI (provided as array)
      3) Compute scaling factors
      4) Resample (image & ROI) — align to grid center, accept interpolation aliases
      5) Intensity preprocessing (resegmentation + outliers + rounding) → intensity_mask (NaN outside ROI)
      6) Quantize (FBN/FBS/Lloyd). FBS anchor resolution:
           reseg lower bound → fbs_default_anchor → PET=0.0 → data-driven min
      7) Crop to ROI bbox → return crops + bbox slices (and unmasked resampled crop)
    """

    # ---------------- lifecycle ----------------

    def __init__(self, config: Dict[str, Any], volume: np.ndarray, roi_mask: Optional[np.ndarray] = None) -> None:
        self.config: Dict[str, Any] = dict(config or {})
        self.volume: np.ndarray = np.asarray(volume, dtype=np.float32)
        self._roi_mask: Optional[np.ndarray] = (
            np.asarray(roi_mask, dtype=np.float32) if roi_mask is not None else None
        )
        self.state: Dict[str, Any] = {}
        self.log = logging.getLogger(self.__class__.__name__)

    # ---------------- Utilities ----------------

    def _get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    @staticmethod
    def _to_bool(arr: np.ndarray) -> np.ndarray:
        return arr if arr.dtype == bool else arr.astype(bool, copy=False)

    @staticmethod
    def _normalize_interp(name: Optional[str]) -> str:
        """
        Map common aliases to the resizer's supported set: {'nearest','linear','cubic'}.
        """
        key = (name or "linear").strip().lower()
        alias_map = {
            "nearest": "nearest",
            "nn": "nearest",

            "linear": "linear",
            "bilinear": "linear",
            "trilinear": "linear",

            "cubic": "cubic",
            "bicubic": "cubic",
            "tricubic": "cubic",

            # approximations when true spline/lanczos aren't available
            "spline": "cubic",
            "bspline": "cubic",
            "tricubic-spline": "cubic",
            "lanczos": "cubic",
            "lanczos4": "cubic",
        }
        return alias_map.get(key, "linear")

    # --------------- Stage 1 -------------------

    def stage_select_quantizer(self) -> Type:
        """
        Select the quantizer class based on discretization and quantizer type.
        """
        discretization_type = str(self._get("discretization_type") or "").strip().upper()
        quantizer_type = str(self._get("quantizer_type") or "").strip()

        if quantizer_type == "Lloyd":
            quantizer_cls: Type = LloydQuantizer
        elif discretization_type == "FBS":
            quantizer_cls = FixedBinSizeQuantizer
        elif discretization_type == "FBN":
            quantizer_cls = UniformQuantizer
        else:
            raise ValueError('discretization_type must be one of {"FBS","FBN"} or quantizer_type == "Lloyd".')

        self.state["quantizer_cls"] = quantizer_cls
        return quantizer_cls

    # --------------- Stage 2 -------------------

    def stage_load_roi(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the ROI mask and original volume.
        Returns:
            morph_mask (float32 0/1), image (float32)
        """
        if self._roi_mask is None:
            raise ValueError("ROI mask array must be provided in-memory.")
        if self._roi_mask.ndim != 3:
            raise ValueError("ROI mask must be a 3D array.")
        if self.volume.ndim != 3:
            raise ValueError("Input volume must be a 3D array.")
        if self.volume.shape != self._roi_mask.shape:
            raise ValueError("Volume and ROI mask must have identical shapes.")

        morph_mask = (self._roi_mask > 0.5).astype(np.float32, copy=False)

        self.state["morph_mask_orig"] = morph_mask
        self.state["image_orig"] = self.volume
        return morph_mask, self.volume

    # --------------- Stage 3 -------------------

    def stage_compute_scaling(self) -> Tuple[float, float, float]:
        """
        Compute scale factors depending on requested resampling mode.
        Returns (sx, sy, sz).
        """
        apply_scaling = bool(self._get("apply_scaling"))
        scale_mode = str(self._get("scale_type") or "NoRescale")

        if (not apply_scaling) or scale_mode == "NoRescale":
            scales = (1.0, 1.0, 1.0)
            self.state["scales"] = scales
            return scales

        voxel_w = float(self._get("pixel_width"))
        voxel_z = float(self._get("slice_thickness"))
        target = float(self._get("new_voxel_size"))

        sx = sy = sz = 1.0
        if scale_mode == "XYZscale":
            sx = sy = (voxel_w / target)
            sz = (voxel_z / target)
        elif scale_mode == "XYscale":
            sx = sy = (voxel_w / target)
        elif scale_mode == "Zscale":
            sz = (voxel_z / voxel_w)
        else:
            raise ValueError('scale_type must be in {"XYZscale","XYscale","Zscale","NoRescale"}')

        if bool(self._get("is_isotropic_2d")):
            sz = 1.0  # force in-plane only

        scales = (float(sx), float(sy), float(sz))
        self.state["scales"] = scales
        return scales

    # --------------- Stage 4 -------------------

    def stage_resample(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample image & ROI to target spacing (center aligned).
        Returns (image_resampled, mask_resampled).
        """
        scales = self.state.get("scales")
        mask0 = self.state.get("morph_mask_orig")
        img0 = self.state.get("image_orig")
        if any(x is None for x in (scales, mask0, img0)):
            raise RuntimeError("Run stages 2 and 3 before resampling.")

        voxel_w = float(self._get("pixel_width"))
        voxel_z = float(self._get("slice_thickness"))
        target_vs = float(self._get("new_voxel_size"))
        is_2d = bool(self._get("is_isotropic_2d"))

        original_spacing = (voxel_w, voxel_w, voxel_z)
        if is_2d:
            target_spacing = (target_vs, target_vs, voxel_z)  # in-plane only
        else:
            target_spacing = (target_vs, target_vs, target_vs)

        self.state["voxel_spacing_mm"] = tuple(float(s) for s in target_spacing)

        mask_resizer = VolumeResizer()
        image_resizer = VolumeResizer()

        sx, sy, sz = scales
        needs = not np.isclose(sx + sy + sz, 3.0)

        if not needs:
            mask_r = mask0
            img_r = img0
        else:
            # ROI mask
            roi_interp = self._normalize_interp(self._get("roi_interp"))
            mask_tmp = mask_resizer.resize(
                volume=mask0,
                original_spacing=original_spacing,
                target_spacing=target_spacing,
                interpolation=roi_interp,
                align_to_center=True,
            )
            pv = self._get("roi_partial_volume")
            pv_thresh = 0.5 if pv is None else float(pv)
            if roi_interp == "linear":
                mask_r = (mask_tmp >= pv_thresh).astype(np.float32, copy=False)
            else:  # nearest/cubic → hard binary
                mask_r = (mask_tmp > 0.5).astype(np.float32, copy=False)

            # Image
            vox_interp = self._normalize_interp(self._get("voxel_interp"))
            img_clean = np.nan_to_num(img0, nan=0.0, copy=True)
            img_r = image_resizer.resize(
                volume=img_clean,
                original_spacing=original_spacing,
                target_spacing=target_spacing,
                interpolation=vox_interp,
                align_to_center=True,
            ).astype(np.float32, copy=False)

        if bool(self._get("apply_rounding")):
            # e.g., CT HU should be integers after interpolation
            np.rint(img_r, out=img_r)

        self.state["morph_mask_resampled"] = mask_r
        self.state["image_resampled"] = img_r
        return img_r, mask_r

    # --------------- Stage 5 -------------------

    def stage_preprocess_intensity(self) -> np.ndarray:
        """
        Apply ROI masking → NaN outside, then re-segmentation and outlier filtering (3σ).
        """
        mask_r = self.state.get("morph_mask_resampled")
        img_r = self.state.get("image_resampled")
        if any(x is None for x in (mask_r, img_r)):
            raise RuntimeError("Resampling must precede intensity preprocessing.")

        mask_bool = self._to_bool(mask_r)
        inten = np.where(mask_bool, img_r, np.nan).astype(np.float32, copy=False)

        # Range re-segmentation
        if bool(self._get("apply_resegmentation")):
            lo, hi = self._get("resegmentation_interval") or (None, None)
            if lo is None and hi is None:
                raise ValueError("apply_resegmentation=True but 'resegmentation_interval' is not set.")
            if lo is not None:
                inten[inten < float(lo)] = np.nan
            if hi is not None:
                inten[inten > float(hi)] = np.nan

        # Outlier removal (3σ over valid voxels only)
        if bool(self._get("remove_outliers")):
            mu = np.nanmean(inten)
            sd = np.nanstd(inten)
            if np.isfinite(sd) and sd > 0.0:
                low = mu - 3.0 * sd
                high = mu + 3.0 * sd
                out = (inten < low) | (inten > high)
                inten[out] = np.nan

        # Optional rounding (e.g., CT integer intensities)
        if bool(self._get("apply_rounding")):
            np.rint(inten, out=inten)

        self.state["intensity_mask"] = inten
        return inten

    # --------------- Stage 6 -------------------

    def _resolve_fbs_anchor(self, roi_vals: np.ndarray) -> float:
        """
        IBSI-consistent FBS anchor resolution:
        1) re-segmentation lower bound if present
        2) modality default from config 'fbs_default_anchor' (e.g., CT:-1000, PET:0)
        3) PET fallback 0.0 if modality looks like PET
        4) data-driven min of ROI values
        """
        if bool(self._get("apply_resegmentation")):
            lo, _ = self._get("resegmentation_interval") or (None, None)
            if lo is not None:
                return float(lo)

        default_anchor = self._get("fbs_default_anchor")

        if default_anchor is not None:
            return float(default_anchor)

        if str(self._get("data_type") or "").strip().upper().startswith("PET"):
            return 0.0

        return float(np.nanmin(roi_vals))

    def stage_quantize(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Discretise intensity mask using selected quantiser.
        """
        roi_values = self.state.get("intensity_mask")
        quantizer_cls: Optional[Type] = self.state.get("quantizer_cls")
        if roi_values is None or quantizer_cls is None:
            raise RuntimeError("Run quantizer selection and intensity preprocessing first.")

        bin_size = self._get("bin_size")
        if bin_size is None:
            raise ValueError("Missing 'bin_size' for quantization.")

        anchor = None
        if quantizer_cls is FixedBinSizeQuantizer:
            anchor = self._resolve_fbs_anchor(roi_values)
            quantizer = FixedBinSizeQuantizer(roi_values, float(bin_size), anchor)
        elif quantizer_cls is UniformQuantizer:
            quantizer = UniformQuantizer(roi_values, int(bin_size))
        else:
            quantizer = LloydQuantizer(roi_values, int(bin_size))

        quantized, levels = quantizer.quantize()

        # ----------------- شرط offset -----------------
        reseg_lo, reseg_hi = self._get("resegmentation_interval")

        if quantizer_cls is FixedBinSizeQuantizer and reseg_lo is None and reseg_hi is None and anchor is not None:
            quantized += (anchor - 1)
        # ----------------------------------------------

        self.state["quantized"] = quantized
        self.state["levels"] = levels
        return quantized, levels

    # --------------- Stage 7 -------------------

    def stage_crop(self) -> None:
        """
        Crop all outputs to the tight bbox of the (resampled) morphological mask.
        Adds:
          - crop_bbox (xmin, ymin, zmin, xmax, ymax, zmax)
          - morph_mask_resampled        (cropped)
          - intensity_mask              (cropped)
          - quantized                   (cropped)
          - image_resampled_unmasked_crop (cropped, unmasked)
        """
        mask_r = self.state.get("morph_mask_resampled")
        inten = self.state.get("intensity_mask")
        quant = self.state.get("quantized")
        img_r = self.state.get("image_resampled")

        if any(x is None for x in (mask_r, inten, quant, img_r)):
            raise RuntimeError("Quantization and resampling must precede cropping.")

        coords = np.argwhere(mask_r > 0.5)
        if coords.size == 0:
            raise ValueError("ROI empty after preprocessing; cannot crop.")

        x_min, y_min, z_min = coords.min(axis=0)
        x_max, y_max, z_max = coords.max(axis=0) + 1
        bbox = (slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max))

        mask_c = mask_r[bbox].astype(np.float32, copy=False)
        inten_c = inten[bbox].astype(np.float32, copy=False)
        quant_c = quant[bbox].astype(np.float32, copy=False)
        img_c = img_r[bbox].astype(np.float32, copy=False)

        self.state.update({
            "crop_bbox": (int(x_min), int(y_min), int(z_min), int(x_max), int(y_max), int(z_max)),
            "morph_mask_resampled": mask_c,
            "intensity_mask": inten_c,
            "quantized": quant_c,
            # downstream expects this key; avoid ambiguous `a or b` in callers
            "image_resampled_unmasked_crop": img_c,
            # keep full resampled image too
            "image_resampled": img_r,
        })

    # --------------- Orchestrator ----------------

    def run_pipeline(self) -> Dict[str, Any]:
        self.stage_select_quantizer()
        self.stage_load_roi()
        self.stage_compute_scaling()
        self.stage_resample()
        self.stage_preprocess_intensity()
        self.stage_quantize()
        self.stage_crop()
        return self.state

    def run(self) -> Dict[str, Any]:
        return self.run_pipeline()
