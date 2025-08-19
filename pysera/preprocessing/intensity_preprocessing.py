"""
Intensity preprocessing functionality for radiomics analysis.
"""

import logging
import numpy as np
import tempfile
from ..utils.utils import create_process_safe_tempfile
try:
    from ..config.settings import INTENSITY_PERCENTILE_LOW, INTENSITY_PERCENTILE_HIGH
except ImportError:
    # Fallback for when running as standalone
    INTENSITY_PERCENTILE_LOW = 1
    INTENSITY_PERCENTILE_HIGH = 99

logger = logging.getLogger("Dev_logger")

def apply_intensity_preprocessing(image_array: np.ndarray, roi_array: np.ndarray, 
                                normalize_intensities: bool = True, 
                                clip_outliers: bool = True) -> np.ndarray:
    """
    Apply intensity preprocessing to image data within ROI regions.
    
    Args:
        image_array: Input image array
        roi_array: ROI mask array or RTSTRUCT dictionary {roi_name: mask_array}
        normalize_intensities: Whether to normalize intensities
        clip_outliers: Whether to clip outliers
        
    Returns:
        Processed image array
    """
    logger.info(f"INTENSITY_PREPROCESS- Starting intensity preprocessing")
    
    processed_image = image_array.copy()
    
    # Handle RTSTRUCT case (dictionary of ROIs)
    if isinstance(roi_array, dict):
        logging.info(f"INTENSITY_PREPROCESS- Processing RTSTRUCT with {len(roi_array)} ROIs")
        return _process_rtstruct_intensities(processed_image, roi_array, normalize_intensities, clip_outliers)
    
    # Handle single ROI case
    roi_mask = roi_array > 0
    roi_voxel_count = np.sum(roi_mask)
    
    logger.info(f"INTENSITY_PREPROCESS- ROI contains {roi_voxel_count} voxels")
    
    if roi_voxel_count == 0:
        logger.error(f"INTENSITY_PREPROCESS- CRITICAL: No ROI voxels found!")
        return processed_image
    
    roi_intensities = processed_image[roi_mask]
    _log_initial_intensity_statistics(roi_intensities)
    
    if clip_outliers:
        processed_image = _clip_intensity_outliers(processed_image, roi_mask, roi_intensities)
    
    if normalize_intensities:
        processed_image = _normalize_intensities(processed_image, roi_mask)
    
    logger.info(f"INTENSITY_PREPROCESS- Intensity preprocessing completed")
    return processed_image


def _process_rtstruct_intensities(image_array: np.ndarray, rtstruct_dict: dict, 
                                normalize_intensities: bool, clip_outliers: bool) -> np.ndarray:
    """
    Process intensities for RTSTRUCT data with multiple ROIs.
    
    Args:
        image_array: Input image array
        rtstruct_dict: Dictionary of {roi_name: mask_array}
        normalize_intensities: Whether to normalize intensities
        clip_outliers: Whether to clip outliers
        
    Returns:
        Processed image array
    """
    logging.info(f"INTENSITY_PREPROCESS- Processing RTSTRUCT intensities for {len(rtstruct_dict)} ROIs")

    # Combine all ROIs for intensity preprocessing
    combined_roi_mask = np.zeros_like(image_array, dtype=bool)
    total_voxels = 0
    
    for roi_name, roi_mask in rtstruct_dict.items():
        if roi_mask is not None and roi_mask.size > 0:
            roi_bool_mask = roi_mask > 0
            combined_roi_mask |= roi_bool_mask
            roi_voxels = np.sum(roi_bool_mask)
            total_voxels += roi_voxels
            logging.info(f"INTENSITY_PREPROCESS- ROI '{roi_name}': {roi_voxels} voxels")
    
    if total_voxels == 0:
        logging.error(f"INTENSITY_PREPROCESS- CRITICAL: No ROI voxels found in RTSTRUCT!")
        return image_array
    
    logging.info(f"INTENSITY_PREPROCESS- Combined RTSTRUCT ROIs contain {total_voxels} voxels")
    
    # Get intensities from all ROIs
    roi_intensities = image_array[combined_roi_mask]
    _log_initial_intensity_statistics(roi_intensities)
    
    if clip_outliers:
        image_array = _clip_intensity_outliers(image_array, combined_roi_mask, roi_intensities)
    
    if normalize_intensities:
        image_array = _normalize_intensities(image_array, combined_roi_mask)
    
    logging.info(f"INTENSITY_PREPROCESS- RTSTRUCT intensity preprocessing completed")
    return image_array


def _chunked_boolean_operation(roi_mask: np.ndarray) -> np.ndarray:
    """
    Chunked boolean operation for very large arrays.
    
    Args:
        roi_mask: ROI mask array
        
    Returns:
        Boolean mask
    """
    chunk_size = 1000000  # 1M elements per chunk
    result = np.zeros_like(roi_mask, dtype=bool)
    
    for i in range(0, roi_mask.size, chunk_size):
        end_idx = min(i + chunk_size, roi_mask.size)
        chunk = roi_mask.flat[i:end_idx]
        result.flat[i:end_idx] = chunk > 0
    
    return result


def _sampled_boolean_operation(roi_mask: np.ndarray) -> np.ndarray:
    """
    Sampled boolean operation for large arrays.
    
    Args:
        roi_mask: ROI mask array
        
    Returns:
        Boolean mask
    """
    # Use sparse approach
    result = np.zeros_like(roi_mask, dtype=bool)
    non_zero_indices = np.where(roi_mask > 0)
    result[non_zero_indices] = True
    return result


def _log_initial_intensity_statistics(roi_intensities: np.ndarray) -> None:
    """Log initial intensity statistics."""
    logger.info(f"INTENSITY_PREPROCESS- Original intensity stats - Min: {np.min(roi_intensities):.2f}, Max: {np.max(roi_intensities):.2f}, Mean: {np.mean(roi_intensities):.2f}, Std: {np.std(roi_intensities):.2f}")


def _clip_intensity_outliers(image_array: np.ndarray, roi_mask: np.ndarray, 
                           roi_intensities: np.ndarray) -> np.ndarray:
    """
    Clip intensity outliers in the ROI region.
    
    Args:
        image_array: Input image array
        roi_mask: ROI mask
        roi_intensities: Intensities within ROI
        
    Returns:
        Image array with clipped outliers
    """
    logger.info(f"INTENSITY_PREPROCESS- Clipping outliers...")
    
    p1, p99 = np.percentile(roi_intensities, [INTENSITY_PERCENTILE_LOW, INTENSITY_PERCENTILE_HIGH])
    clipped_image = np.clip(image_array, p1, p99)
    
    roi_intensities_clipped = clipped_image[roi_mask]
    _log_clipping_results(p1, p99, roi_intensities_clipped)
    
    return clipped_image

def _log_clipping_results(p1: float, p99: float, roi_intensities_clipped: np.ndarray) -> None:
    """Log clipping results."""
    logger.info(f"INTENSITY_PREPROCESS- Clipped intensities to range [{p1:.2f}, {p99:.2f}]")
    logger.info(f"INTENSITY_PREPROCESS- After clipping - Min: {np.min(roi_intensities_clipped):.2f}, Max: {np.max(roi_intensities_clipped):.2f}, Mean: {np.mean(roi_intensities_clipped):.2f}, Std: {np.std(roi_intensities_clipped):.2f}")


def _normalize_intensities(image_array: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    """
    Normalize intensities within the ROI region.
    
    Args:
        image_array: Input image array
        roi_mask: ROI mask
        
    Returns:
        Image array with normalized intensities
    """
    logger.info(f"INTENSITY_PREPROCESS- Applying intensity normalization...")
    
    roi_intensities = image_array[roi_mask]
    mean_intensity = float(np.mean(roi_intensities))
    std_intensity = float(np.std(roi_intensities))
    
    logger.info(f"INTENSITY_PREPROCESS- ROI mean: {mean_intensity:.2f}, std: {std_intensity:.2f}")
    
    if std_intensity <= 0:
        logger.warning(f"INTENSITY_PREPROCESS- Cannot normalize - zero variance in ROI intensities!")
        return image_array
    
    normalized_image = _perform_normalization(image_array, roi_mask, mean_intensity, std_intensity)
    _log_normalization_results(normalized_image, roi_mask)

    return normalized_image


def _perform_normalization(image_array: np.ndarray, roi_mask: np.ndarray, 
                         mean_intensity: float, std_intensity: float) -> np.ndarray:
    """
    Perform the actual normalization operation.
    
    Args:
        image_array: Input image array
        roi_mask: ROI mask
        mean_intensity: Mean intensity value
        std_intensity: Standard deviation of intensity
        
    Returns:
        Normalized image array
    """
    normalized_image = image_array.copy()
    normalized_image[roi_mask] = (image_array[roi_mask] - mean_intensity) / std_intensity
    normalized_image[roi_mask] = normalized_image[roi_mask] + abs(np.min(normalized_image[roi_mask])) + 1
    
    return normalized_image


def _log_normalization_results(normalized_image: np.ndarray, roi_mask: np.ndarray) -> None:
    """Log normalization results."""
    roi_intensities_norm = normalized_image[roi_mask]
    logger.info(f"INTENSITY_PREPROCESS- After normalization - Min: {np.min(roi_intensities_norm):.2f}, Max: {np.max(roi_intensities_norm):.2f}, Mean: {np.mean(roi_intensities_norm):.2f}, Std: {np.std(roi_intensities_norm):.2f}")
    logger.info(f"INTENSITY_PREPROCESS- Applied intensity normalization within ROI")


def get_intensity_statistics(image_array: np.ndarray, roi_array: np.ndarray) -> dict:
    """
    Get intensity statistics within ROI regions.
    
    Args:
        image_array: Input image array
        roi_array: ROI mask array or RTSTRUCT dictionary {roi_name: mask_array}
        
    Returns:
        Dictionary containing intensity statistics
    """
    # Handle RTSTRUCT case (dictionary of ROIs)
    if isinstance(roi_array, dict):
        return _get_rtstruct_intensity_statistics(image_array, roi_array)
    
    # Handle single ROI case
    roi_mask = roi_array > 0
    roi_intensities = image_array[roi_mask]
    
    if len(roi_intensities) == 0:
        return _get_empty_intensity_stats()
    
    return _calculate_intensity_stats(roi_intensities)


def _get_rtstruct_intensity_statistics(image_array: np.ndarray, rtstruct_dict: dict) -> dict:
    """
    Get intensity statistics for RTSTRUCT data with multiple ROIs.
    
    Args:
        image_array: Input image array
        rtstruct_dict: Dictionary of {roi_name: mask_array}
        
    Returns:
        Dictionary containing combined intensity statistics
    """
    # Combine all ROIs for statistics
    combined_roi_mask = np.zeros_like(image_array, dtype=bool)
    
    for roi_name, roi_mask in rtstruct_dict.items():
        if roi_mask is not None and roi_mask.size > 0:
            roi_bool_mask = roi_mask > 0
            combined_roi_mask |= roi_bool_mask
    
    roi_intensities = image_array[combined_roi_mask]
    
    if len(roi_intensities) == 0:
        return _get_empty_intensity_stats()
    
    return _calculate_intensity_stats(roi_intensities)


def _get_empty_intensity_stats() -> dict:
    """Get empty intensity statistics."""
    return {
        'mean': 0.0,
        'std': 0.0,
        'min': 0.0,
        'max': 0.0,
        'median': 0.0,
        'voxel_count': 0
    }


def _calculate_intensity_stats(roi_intensities: np.ndarray) -> dict:
    """
    Calculate intensity statistics from ROI intensities.
    
    Args:
        roi_intensities: Intensity values within ROI
        
    Returns:
        Dictionary containing calculated statistics
    """
    return {
        'mean': float(np.mean(roi_intensities)),
        'std': float(np.std(roi_intensities)),
        'min': float(np.min(roi_intensities)),
        'max': float(np.max(roi_intensities)),
        'median': float(np.median(roi_intensities)),
        'voxel_count': int(len(roi_intensities))
    } 