# -*- coding: utf-8 -*-
# data_manager/feature_data_manager.py

from __future__ import annotations

import gc
import logging
import numpy as np
from typing import Any, Dict, Optional, Tuple

from .volume_preparation_pipeline import VolumePreparationPipeline

log = logging.getLogger(__name__)


def _is_calibrated_units(modality: str) -> bool:
    """
    Return True if modality represents calibrated intensity units (e.g., CT (HU), PET (SUV)).
    All other/unknown modalities are treated as 'arbitrary' units per IBSI Table 3.1.
    """
    m = str(modality or "").strip().upper()
    return m in {"CT", "PET", "SPECT", "PT", "SPECT-CT", "PET-CT"}


def _default_fbs_anchor_for_modality(modality: str) -> Optional[float]:
    """
    Provide a stable, modality-specific FBS anchor when re-seg lower bound is not set.
    - CT  : -1000 HU (air)
    - PET : 0.0 SUV
    """
    m = str(modality or "").strip().upper()
    if m == "CT":
        return -1000.0
    if m == "PET" or m == "PET-CT" or m == "SPECT" or m == "SPECT-CT":
        return 0.0
    return None


class FeatureDataManager:
    """
    RAM-only orchestrator to prepare volumes for feature extraction (IBSI-aligned).

    - Interpolation (2D in-plane or 3D) aligned to grid centers (implementation-independent).
    - ROI mask handled with nearest or trilinear; partial-volume threshold applied for trilinear.
    - Intensity rounding (e.g., CT HU to nearest integer) after interpolation.
    - Re-segmentation (range) and outlier filtering (3σ) applied to INTENSITY mask only.
      Morphological mask preserves original segmentation shape (post-resample).
    - Discrimination: FBN, FBS, or Lloyd (with IBSI rules and recommendations).
    - Crops to ROI bbox and returns aligned, cropped blocks and metadata.
    """

    def __init__(
        self,
        *,
        image: np.ndarray,
        roi_mask_array: np.ndarray,
        pixel_width: float,
        slice_thickness: float,
        data_type: str,
        isotropic_voxel_size: float,
        roi_partial_volume: float,
        scale_type: str,
        is_isotropic_2d: bool,
        apply_scaling: bool,
        apply_rounding: bool,
        discretization_type: str,
        quantizer_type: str,
        apply_resegmentation: bool,
        resegmentation_interval: Tuple[Optional[float], Optional[float]],
        remove_outliers: bool,
    ) -> None:
        self.image = np.asarray(image, np.float32)
        self.roi_mask = np.asarray(roi_mask_array, np.float32)

        if self.image.ndim != 3 or self.roi_mask.ndim != 3 or self.image.shape != self.roi_mask.shape:
            raise ValueError("image and roi_mask_array must be non-empty 3D arrays with identical shapes.")

        self.pixel_width = float(pixel_width)
        self.slice_thickness = float(slice_thickness)
        self.data_type = str(data_type)
        self._units_calibrated = _is_calibrated_units(self.data_type)

        self.isotropic_voxel_size = float(isotropic_voxel_size)
        self.roi_partial_volume = float(roi_partial_volume)
        self.scale_type = str(scale_type)
        self.is_isotropic_2d = bool(is_isotropic_2d)
        self.apply_scaling = bool(apply_scaling)
        self.apply_rounding = bool(apply_rounding)
        self.discretization_type = str(discretization_type).upper()
        self.quantizer_type = str(quantizer_type)

        self.apply_resegmentation = bool(apply_resegmentation)
        if resegmentation_interval is None or len(resegmentation_interval) != 2:
            self.resegmentation_interval = (None, None)
        else:
            lo, hi = resegmentation_interval
            self.resegmentation_interval = (
                None if lo is None else float(lo),
                None if hi is None else float(hi),
            )

        self.remove_outliers = bool(remove_outliers)
        self._cache: Dict[int, Dict[str, Any]] = {}

        # IBSI Table 3.1 guardrails for FBS
        self._fbs_default_anchor: Optional[float] = None
        if self.discretization_type == "FBS":
            lower_bound = self.resegmentation_interval[0]
            if lower_bound is None:
                if not self._units_calibrated:
                    # Arbitrary units + FBS without range: not recommended → block
                    raise ValueError(
                        "Fixed Bin Size (FBS) with arbitrary intensity units and no defined re-segmentation "
                        "range is not recommended (IBSI Table 3.1). Either switch to FBN or set a re-seg range."
                    )
                # Calibrated but no range → use stable modality default (CT:-1000HU, PET:0SUV)
                self._fbs_default_anchor = _default_fbs_anchor_for_modality(self.data_type)
                if self._fbs_default_anchor is not None:
                    log.info(
                        "FBS selected without a re-seg lower bound; using modality default anchor %.3f for %s.",
                        self._fbs_default_anchor, self.data_type
                    )
                else:
                    # Last resort (still calibrated): fall back to data-driven min; warn once
                    log.warning(
                        "FBS selected without a defined re-seg lower bound and no known modality default. "
                        "Anchor will fall back to data-driven min; define a re-seg range for reproducibility."
                    )

    # -------------------------- main entry --------------------------

    def prepare_volume_for_bin(
        self,
        *,
        bin_size: float,
        bin_index: int,
        isotropic_voxel_size_2d: Optional[float] = None,
        voxel_interp: str = "linear",
        roi_interp: str = "nearest",
    ) -> Dict[str, Any]:
        """
        Build the full IBSI-aligned preparation pipeline for a single (ROI, bin) setup.
        Returns a state dict with cropped volumes and spacing metadata.
        """
        # Resolve target spacing (2D vs 3D)
        if self.is_isotropic_2d and isotropic_voxel_size_2d is not None:
            target_voxel = float(isotropic_voxel_size_2d)
        else:
            target_voxel = self.isotropic_voxel_size

        config: Dict[str, Any] = {
            # acquisition / modality
            "data_type": self.data_type,
            "_units_calibrated": self._units_calibrated,

            # original spacing (mm)
            "pixel_width": self.pixel_width,
            "slice_thickness": self.slice_thickness,

            # interpolation & alignment
            "new_voxel_size": target_voxel,
            "voxel_interp": str(voxel_interp or "linear"),
            "roi_interp": str(roi_interp or "nearest"),
            "roi_partial_volume": self.roi_partial_volume,
            "scale_type": self.scale_type,
            "is_isotropic_2d": self.is_isotropic_2d,
            "apply_scaling": self.apply_scaling,
            "align_to_center": True,  # IBSI recommendation

            # intensity processing
            "apply_rounding": self.apply_rounding,
            "apply_resegmentation": self.apply_resegmentation,
            "resegmentation_interval": self.resegmentation_interval,
            "remove_outliers": self.remove_outliers,

            # discrimination
            "discretization_type": self.discretization_type,
            "quantizer_type": self.quantizer_type,
            "bin_size": float(bin_size),

            # pass default FBS anchor (if any) so pipeline can use it
            "fbs_default_anchor": self._fbs_default_anchor,
        }

        pipeline = VolumePreparationPipeline(config=config, volume=self.image, roi_mask=self.roi_mask)
        state = pipeline.run_pipeline()

        self._cache[bin_index] = state
        gc.collect()
        return state

    # -------------------------- accessors --------------------------

    def _get_cached(self, bin_index: int) -> Dict[str, Any]:
        if bin_index not in self._cache:
            raise ValueError(f"No cached results for bin {bin_index}. Call prepare_volume_for_bin() first.")
        return self._cache[bin_index]

    def get_roi_mask(self, bin_index: int) -> np.ndarray:
        return self._get_cached(bin_index)["morph_mask_resampled"]

    def get_roi_values(self, bin_index: int) -> np.ndarray:
        return self._get_cached(bin_index)["intensity_mask"]

    def get_quantized_volume(self, bin_index: int) -> np.ndarray:
        return self._get_cached(bin_index)["quantized"]

    def get_levels(self, bin_index: int) -> Optional[np.ndarray]:
        return self._get_cached(bin_index).get("levels")
