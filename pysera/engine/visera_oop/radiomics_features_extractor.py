# -*- coding: utf-8 -*-
# radiomics_features_extractor.py

from __future__ import annotations

import collections.abc
import gc
import logging
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .core.feature_extraction_manager import FeatureExtractionManager, ManagerConfig

logger = logging.getLogger("Dev_logger")


def _as_list(value: Iterable[float] | float) -> List[float]:
    if isinstance(value, collections.abc.Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [float(v) for v in value]
    return [float(value)]


def save_value_dict_to_excel(value_dict: dict, output_path: str = "features_only.xlsx"):
    df = pd.DataFrame([value_dict])
    df.to_excel(output_path, index=False)
    print(f"Only 'value' data saved to: {output_path}")


def SERA_FE_main(  # noqa: N802
        image_array: np.ndarray,
        roi_mask_array: np.ndarray,
        feature_dimensions_mask: str,
        extractor_mask: str,
        voxel_dimensions: Tuple[float, float, float],
        bin_sizes_input,
        data_type_str: str,
        isotropic_voxel_size_3d: float,
        isotropic_voxel_size_2d: float,
        discretization_method: str,
        quantization_method: str,
        voxel_interpolation_method: str,
        roi_interpolation_method: str,
        enable_scaling: bool,
        enable_gl_rounding: bool,
        enable_resegmentation: bool,
        remove_outliers: bool,
        quantize_statistics: bool,
        use_isotropic_2d: bool,
        resegmentation_interval: Tuple[float, float],
        roi_partial_volume_fraction: float,
        feature_value_mode_str,
        image_name: str,
        roi_name=None,
        ivh_type=None,
        ivh_disc_cont=None,
        ivh_bin_size=None,
) -> Dict[str, Any]:
    if image_array is None or image_array.size == 0:
        raise ValueError("image_array is empty.")

    if roi_mask_array is None or roi_mask_array.size == 0:
        raise ValueError("roi_mask_array is empty.")

    bin_sizes_list = _as_list(bin_sizes_input)
    pixel_width, pixel_height, slice_thickness = voxel_dimensions

    # Convert IVH parameters to float
    ivh_type = float(ivh_type) if ivh_type is not None else None
    ivh_disc_cont = float(ivh_disc_cont) if ivh_disc_cont is not None else None
    ivh_bin_size = float(ivh_bin_size) if ivh_bin_size is not None else None

    # Build IVH configuration from numeric values
    ivh_configuration = [ivh_type, ivh_disc_cont, ivh_bin_size]

    manager_cfg = ManagerConfig(
        raw_image=np.ascontiguousarray(image_array, dtype=np.float32),
        roi_masks=[np.asarray(roi_mask_array, dtype=bool)],
        voxel_size_info=(float(pixel_width), float(pixel_height), float(slice_thickness)),

        bin_sizes=bin_sizes_list,
        data_type=str(data_type_str),
        isotropic_voxel_size_3d=float(isotropic_voxel_size_3d),
        isotropic_voxel_size_2d=float(isotropic_voxel_size_2d),
        discretization_type=str(discretization_method),
        quantization_method=str(quantization_method),
        voxel_interp=str(voxel_interpolation_method),
        roi_interp=str(roi_interpolation_method),
        perform_rescale=bool(enable_scaling),
        perform_gl_rounding=bool(enable_gl_rounding),
        quantize_statistics=bool(quantize_statistics),
        perform_resegmentation=bool(enable_resegmentation),
        remove_outliers=bool(remove_outliers),
        use_isotropic_2d=bool(use_isotropic_2d),
        resegmentation_interval=tuple(resegmentation_interval),
        roi_partial_volume=float(roi_partial_volume_fraction),

        profile_name="Full_IBSI",
        feature_value_mode=str(feature_value_mode_str),

        feature_dimensions_mask=feature_dimensions_mask,
        extractor_mask=extractor_mask,

        ivh_configuration=ivh_configuration,

        scale_type="XYZscale",
        backend="scipy",
        align_to_center=True,
        zero_frac_threshold=0.85,
        min_voxels_for_sparse=1_000_000,
        sparse_policy_config=None,
        view_planner_config=None,
        view_factory_config=None,
        max_workers=None,
        log_level=logging.INFO,
    )

    manager = FeatureExtractionManager(manager_cfg)
    result = manager.run(image_name=image_name, roi_name=roi_name)
    gc.collect()
    return result
