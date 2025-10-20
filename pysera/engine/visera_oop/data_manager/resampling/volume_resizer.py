# -*- coding: utf-8 -*-
# data_manager/resampling/volume_resizer.py

from __future__ import annotations

import warnings
import numpy as np
import scipy.ndimage as ndi
from typing import Sequence, Union

from .base_resizer import BaseResizer


class VolumeResizer(BaseResizer):
    """
    SciPy-based 3D resampler with 'align grid centers' mapping and rich interpolation alias support.

    - Computes target grid by ceil( orig_shape * (orig_spacing / target_spacing) )
    - Builds destination coordinates with center alignment
    - Uses map_coordinates(mode="nearest") to pad outside field with edge values
    - Interpolation aliases supported:
        nearest:  "nearest", "nn"
        linear:   "linear", "bilinear", "trilinear"
        cubic:    "cubic", "bicubic", "tricubic"
        spline:   "spline", "bspline", "tricubic-spline"   -> mapped to cubic (order=3)
        lanczos:  "lanczos", "lanczos4"                    -> downgraded to cubic (order=3) with warning
    """

    _ALIASES_LINEAR = {"linear", "bilinear", "trilinear"}
    _ALIASES_NEAREST = {"nearest", "nn"}
    _ALIASES_CUBIC = {"cubic", "bicubic", "tricubic"}
    _ALIASES_SPLINE = {"spline", "bspline", "tricubic-spline"}
    _ALIASES_LANCZOS = {"lanczos", "lanczos4"}

    @classmethod
    def normalize_interpolation(cls, interp: str) -> str:
        """Normalize a user interpolation string to one of: 'nearest' | 'linear' | 'cubic'."""
        s = str(interp or "").strip().lower()
        if s in cls._ALIASES_NEAREST:
            return "nearest"
        if s in cls._ALIASES_LINEAR:
            return "linear"
        if s in cls._ALIASES_CUBIC:
            return "cubic"
        if s in cls._ALIASES_SPLINE:
            # map to cubic for map_coordinates
            return "cubic"
        if s in cls._ALIASES_LANCZOS:
            warnings.warn("Lanczos interpolation not supported by scipy.ndimage.map_coordinates; using cubic instead.",
                          RuntimeWarning, stacklevel=2)
            return "cubic"
        # default conservative
        warnings.warn(f"Unknown interpolation '{interp}', defaulting to linear.", RuntimeWarning, stacklevel=2)
        return "linear"

    @staticmethod
    def _order_for(interp_norm: str) -> int:
        # SciPy spline order: 0..5. We use: nearest=0, linear=1, cubic=3
        return {"nearest": 0, "linear": 1, "cubic": 3}.get(interp_norm, 1)

    def resize(
        self,
        volume: Union[str, np.ndarray],
        original_spacing: Sequence[float],
        target_spacing: Sequence[float],
        interpolation: str = "linear",
        align_to_center: bool = True,
    ) -> np.ndarray:
        vol = self._validate_input(volume).astype(np.float32, copy=False)

        os_ = np.asarray(original_spacing, np.float32)
        ts_ = np.asarray(target_spacing, np.float32)
        if os_.size != 3 or ts_.size != 3:
            raise ValueError("Spacings must be 3 elements (sx, sy, sz).")

        orig_shape = np.asarray(vol.shape, np.float32)
        target_shape = np.ceil(orig_shape * (os_ / ts_)).astype(int)

        interp_norm = self.normalize_interpolation(interpolation)
        order = self._order_for(interp_norm)
        return self._interpolate(vol, orig_shape, os_, target_shape, ts_, order, align_to_center)

    @staticmethod
    def _interpolate(
        volume: np.ndarray,
        orig_shape: np.ndarray,
        orig_spacing: np.ndarray,
        target_shape: np.ndarray,
        target_spacing: np.ndarray,
        order: int,
        align_to_center: bool,
    ) -> np.ndarray:
        # destination → source coordinate mapping (grid centers)
        scale_factors  = target_spacing / orig_spacing
        if align_to_center:
            origin_offset  = (
                0.5 * (orig_shape - 1.0)
                - 0.5 * (target_shape.astype(np.float32) - 1.0) * scale_factors
            )
        else:
            origin_offset  = np.zeros(3, np.float32)

        z_coords = np.arange(target_shape[0], dtype=np.float32) * scale_factors [0] + origin_offset [0]
        y_coords = np.arange(target_shape[1], dtype=np.float32) * scale_factors [1] + origin_offset [1]
        x_coords = np.arange(target_shape[2], dtype=np.float32) * scale_factors [2] + origin_offset [2]
        coords = np.vstack(np.meshgrid(z_coords, y_coords, x_coords, indexing="ij")).reshape(3, -1)

        resampled_volume  = ndi.map_coordinates(volume, coords, order=order, mode="nearest")
        return resampled_volume .reshape(target_shape).astype(np.float32, copy=False)
