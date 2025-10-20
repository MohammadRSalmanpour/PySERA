"""
Intensity preprocessing functionality for radiomics analysis.
"""

import logging
import numpy as np
from ..config.settings import INTENSITY_PERCENTILE_LOW, INTENSITY_PERCENTILE_HIGH

logger = logging.getLogger("Dev_logger")


def apply_image_intensity_preprocessing(
        image_array: np.ndarray,
        roi_array: np.ndarray,
        normalize_intensities: bool = True,
        clip_outliers: bool = True,
) -> np.ndarray:
    logger.info("INTENSITY_PREPROCESS - Starting intensity preprocessing")
    processed_image = image_array.copy()

    # Build combined ROI mask
    roi_voxel_count = np.count_nonzero(roi_array)

    if roi_voxel_count == 0:
        logger.error("INTENSITY_PREPROCESS - CRITICAL: No ROI voxels found!")
        return processed_image

    logger.info(f"INTENSITY_PREPROCESS - ROI contains {roi_voxel_count} voxels")
    roi_intensities = processed_image[roi_array]
    _log_intensity_stats("Original", roi_intensities)

    # Outlier clipping
    if clip_outliers:
        processed_image = _clip_intensity_outliers(processed_image, roi_array, roi_intensities)

    # Normalization
    if normalize_intensities:
        processed_image = _normalize_intensities(processed_image, roi_array)

    logger.info("INTENSITY_PREPROCESS - Completed intensity preprocessing")
    return processed_image


# ---------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------

def _log_intensity_stats(stage: str, intensities: np.ndarray) -> None:
    """Log basic intensity statistics for ROI voxels."""
    logger.info(
        f"INTENSITY_PREPROCESS - {stage} intensities | "
        f"Min: {np.min(intensities):.2f}, Max: {np.max(intensities):.2f}, "
        f"Mean: {np.mean(intensities):.2f}, Std: {np.std(intensities):.2f}"
    )


# ---------------------------------------------------------------------
# Outlier clipping
# ---------------------------------------------------------------------

def _clip_intensity_outliers(
        image_array: np.ndarray, roi_mask: np.ndarray, roi_intensities: np.ndarray
) -> np.ndarray:
    """Clip intensity outliers in ROI region."""
    p_low, p_high = np.percentile(
        roi_intensities, [INTENSITY_PERCENTILE_LOW, INTENSITY_PERCENTILE_HIGH]
    )
    clipped_image = np.clip(image_array, p_low, p_high)

    _log_intensity_stats("After clipping", clipped_image[roi_mask])
    logger.info(f"INTENSITY_PREPROCESS - Clipped to range [{p_low:.2f}, {p_high:.2f}]")
    return clipped_image


# ---------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------

def _normalize_intensities(image_array: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    """Normalize intensities within ROI region."""
    roi_vals = image_array[roi_mask]
    mean, std = float(np.mean(roi_vals)), float(np.std(roi_vals))

    if std <= 0:
        logger.warning("INTENSITY_PREPROCESS - Cannot normalize (zero variance in ROI)")
        return image_array

    normalized = image_array.copy()
    normalized[roi_mask] = (roi_vals - mean) / std
    normalized[roi_mask] += abs(np.min(normalized[roi_mask])) + 1

    _log_intensity_stats("After normalization", normalized[roi_mask])
    return normalized


def apply_mask_roundup(mask_array: np.ndarray) -> np.ndarray:

    logger.info("Apply preprocessing: Round Mask array to the nearest integer number of Mask")
    return np.rint(mask_array).astype(mask_array.dtype)
