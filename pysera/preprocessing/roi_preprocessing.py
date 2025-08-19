"""
ROI preprocessing functionality for radiomics analysis.
"""

import logging
import numpy as np
from scipy import ndimage
import gc
from typing import Optional, Dict, Any, List, Tuple, Union
from ..utils.utils import save_numpy_on_disk
try:
    from ..config.settings import (
        MIN_COMPONENT_SIZE, MIN_COMPONENT_SIZE_PERCENTAGE, DEFAULT_MIN_ROI_VOLUME
)
except ImportError:
    # Fallback for when running as standalone
    MIN_COMPONENT_SIZE = 10
    MIN_COMPONENT_SIZE_PERCENTAGE = 0.05

try:
    from ..utils.utils import (
        get_memory_usage, check_memory_available, estimate_array_memory,
        safe_array_operation, optimize_array_dtype, memory_efficient_unique,
        log_memory_usage, remove_temp_file
    )
except ImportError:
    # Fallback if utils not available
    def get_memory_usage(): return 0.0
    def check_memory_available(required_mb): return True
    def estimate_array_memory(shape, dtype): return 0.0
    def safe_array_operation(func, *args, **kwargs): return func(*args, **kwargs)
    def optimize_array_dtype(array, target_dtype=None): return array
    def memory_efficient_unique(array_path):
        array = np.load(array_path, mmap_mode='r')
        uniques = np.unique(array)
        # Clean RAM
        del array
        gc.collect()
        return uniques
    def log_memory_usage(operation_name): pass


logger = logging.getLogger("Dev_logger")

def optimize_roi_preprocessing(roi_array_path: str, min_roi_volume: int = DEFAULT_MIN_ROI_VOLUME,
                             # apply_morphological_operations: bool = True
                               ) -> np.ndarray:
    """
    Optimize ROI preprocessing with memory-efficient operations.
    
    Args:
        roi_array: Input ROI array
        min_roi_volume: Minimum ROI volume threshold
        apply_morphological_operations: Whether to apply morphological operations
        
    Returns:
        Processed ROI array
    """
    roi_array = np.load(roi_array_path, mmap_mode='r')

    logger.info(f"ROI_PREPROCESS- Starting ROI preprocessing")
    logger.info(f"ROI_PREPROCESS- Debug - min_roi_volume type: {type(min_roi_volume)}, value: {min_roi_volume}")
    logger.info(f"ROI_PREPROCESS- Debug - roi_array type: {type(roi_array)}, shape: {roi_array.shape}, dtype: {roi_array.dtype}")

    # Clean RAM
    del roi_array
    gc.collect()

    _log_initial_roi_statistics(roi_array_path)

    # Optimize data type for memory efficiency          toto returns path
    processed_roi_path = optimize_array_dtype(roi_array_path)

    # if apply_morphological_operations:
    #     processed_roi_path = _apply_morphological_operations(processed_roi_path)        # returns path


    processed_roi = np.load(processed_roi_path, mmap_mode='r')
    final_volume = np.sum(processed_roi > 0)

    _log_final_roi_statistics(processed_roi_path, final_volume)

    # Clean RAM
    del processed_roi
    gc.collect()

    logger.info(f"ROI_PREPROCESS- Debug - final_volume type: {type(final_volume)}, value: {final_volume}")
    logger.info(f"ROI_PREPROCESS- Debug - min_roi_volume type: {type(min_roi_volume)}, value: {min_roi_volume}")

    if final_volume < min_roi_volume:
        logger.warning(f"ROI_PREPROCESS- CRITICAL: Final ROI volume ({final_volume}) is below minimum threshold ({min_roi_volume}) for reliable calculation")
        logger.warning(f"ROI_PREPROCESS- Returning original ROI without morphological operations")
        return roi_array_path

    logger.info(f"ROI_PREPROCESS- ROI preprocessing completed successfully")


    return processed_roi_path


def _log_initial_roi_statistics(roi_array_path: str) -> None:
    """Log initial ROI statistics with memory monitoring."""
    try:
        roi_array = np.load(roi_array_path, mmap_mode='r')
        total_voxels = roi_array.size
        non_zero_voxels = np.sum(roi_array > 0)
        memory_mb = roi_array.nbytes / (1024 * 1024)

        # Clean RAM
        del roi_array
        gc.collect()

        logger.info(f"ROI_PREPROCESS - Initial ROI statistics:")
        logger.info(f"ROI_PREPROCESS - Total voxels: {total_voxels}")
        logger.info(f"ROI_PREPROCESS - Non-zero voxels: {non_zero_voxels}")
        logger.info(f"ROI_PREPROCESS - Memory usage: {memory_mb:.1f} MB")

    except Exception as e:
        logger.warning(f"ROI_PREPROCESS - Could not log initial statistics: {e}")


# def _apply_morphological_operations(roi_array_path: str) -> str:
#     """
#     Apply morphological operations with memory-efficient processing.
#     """
#     logger.info(f"ROI_PREPROCESS - Applying morphological operations...")
#     roi_array = np.load(roi_array_path, mmap_mode='r')
#     # processed_roi = np.load(roi_array_path, mmap_mode='r')
#     # processed_roi = roi_array.copy()          # toto CMed
#     struct_elem = _create_structural_element(roi_array.ndim)
#     # Clean RAM
#     del roi_array
#     gc.collect()
#     # Memory-efficient unique label detection
#     unique_labels = memory_efficient_unique(roi_array_path)
#     logger.info(f"ROI_PREPROCESS - Debug - unique_labels type: {type(unique_labels)}, dtype: {unique_labels.dtype}")
#     logger.info(f"ROI_PREPROCESS - Debug - unique_labels: {unique_labels}")
#
#     # Check types before comparison
#     logger.info(f"ROI_PREPROCESS - Debug - About to filter unique_labels > 0")
#     logger.info(f"ROI_PREPROCESS - Debug - unique_labels dtype: {unique_labels.dtype}")
#
#     unique_labels = unique_labels[unique_labels > 0]
#     logger.info(f"ROI_PREPROCESS - Debug - After filtering: {unique_labels}")
#
#     # Process each ROI label separately to preserve individual labels
#     for label_value in unique_labels:       # returns path on disk
#         processed_roi = _process_single_roi_label(
#             roi_array_path, label_value, struct_elem
#         )
#     # Clean disk
#     remove_temp_file(roi_array_path)
#     return processed_roi


def _create_structural_element(dimensions: int) -> np.ndarray:
    """
    Create structural element for morphological operations.
    
    Args:
        dimensions: Number of dimensions (2 or 3)
        
    Returns:
        Structural element array
    """
    if dimensions == 3:
        struct_elem = ndimage.generate_binary_structure(3, 1)
        struct_elem = ndimage.iterate_structure(struct_elem, 2)
    else:
        struct_elem = ndimage.generate_binary_structure(2, 1)
        struct_elem = ndimage.iterate_structure(struct_elem, 1)
    
    return struct_elem


# def _process_single_roi_label(roi_array_path: str, label_value: int,
#                             struct_elem: np.ndarray) -> np.ndarray:
#     """
#     Process a single ROI label with morphological operations.
#
#     Args:
#         roi_array: ROI array to process
#         label_value: Label value to process
#         struct_elem: Structural element for morphological operations
#
#     Returns:
#         Updated ROI array
#     """
#     roi_array = np.load(roi_array_path, mmap_mode='r+')
#     label_mask = (roi_array == label_value)
#     before_closing = np.sum(label_mask)
#
#     # Apply binary closing to this specific label
#     closed_mask = ndimage.binary_closing(label_mask, structure=struct_elem)
#     after_closing = np.sum(closed_mask)
#     logger.info(f"ROI_PREPROCESS - Label {label_value} - After binary closing: {before_closing} -> {after_closing} voxels")
#
#     # Remove small components for this label
#     closed_mask = _remove_small_components(closed_mask, label_value)
#
#     # Update the ROI array: first clear the original label, then set the processed regions
#     roi_array[roi_array == label_value] = 0
#     roi_array[closed_mask] = label_value
#
#     new_array_path = save_numpy_on_disk(roi_array, prefix='_process_single_roi_label', suffix='.npy')
#     # Clean RAM
#     del roi_array, label_mask
#     gc.collect()
#
#     return new_array_path


def _remove_small_components(mask: np.ndarray, label_value: int) -> np.ndarray:
    """
    Remove small components from a binary mask.
    
    Args:
        mask: Binary mask
        label_value: Label value for logging
        
    Returns:
        Mask with small components removed
    """
    labeled_array, num_features = ndimage.label(mask)
    logger.info(f"ROI_PREPROCESS - Label {label_value} - Found {num_features} connected components")
    
    if num_features <= 1:
        return mask
    
    component_sizes = _analyze_component_sizes(labeled_array, num_features, label_value)
    min_component_size = _calculate_min_component_size(mask)
    
    logger.info(f"ROI_PREPROCESS - Label {label_value} - Minimum component size threshold: {min_component_size}")
    
    return _filter_small_components(mask, labeled_array, component_sizes, min_component_size, label_value)


def _analyze_component_sizes(labeled_array: np.ndarray, num_features: int, 
                           label_value: int) -> list:
    """
    Analyze sizes of connected components.
    
    Args:
        labeled_array: Labeled array
        num_features: Number of features
        label_value: Label value for logging
        
    Returns:
        List of (component_id, size) tuples
    """
    component_sizes = []
    for i in range(1, num_features + 1):
        component_size = np.sum(labeled_array == i)
        component_sizes.append((i, component_size))
        logger.info(f"ROI_PREPROCESS - Label {label_value} - Component {i}: {component_size} voxels")
    
    return component_sizes


def _calculate_min_component_size(mask: np.ndarray) -> int:
    """
    Calculate minimum component size threshold.
    
    Args:
        mask: Binary mask
        
    Returns:
        Minimum component size threshold
    """
    total_voxels = np.sum(mask)
    percentage_threshold = max(MIN_COMPONENT_SIZE, total_voxels * MIN_COMPONENT_SIZE_PERCENTAGE)
    return int(percentage_threshold)


def _filter_small_components(mask: np.ndarray, labeled_array: np.ndarray, 
                           component_sizes: list, min_component_size: int, 
                           label_value: int) -> np.ndarray:
    """
    Filter out small components from mask.
    
    Args:
        mask: Binary mask
        labeled_array: Labeled array
        component_sizes: List of component sizes
        min_component_size: Minimum component size threshold
        label_value: Label value for logging
        
    Returns:
        Filtered mask
    """
    filtered_mask = mask.copy()
    
    for component_id, size in component_sizes:
        if size < min_component_size:
            filtered_mask[labeled_array == component_id] = False
            logger.info(f"ROI_PREPROCESS - Label {label_value} - Removed small component {component_id} of size {size} voxels")
    
    return filtered_mask


def _log_final_roi_statistics(roi_array_path: str, final_volume: int) -> None:
    """
    Log final ROI statistics with memory-efficient processing.
    """
    try:
        roi_array = np.load(roi_array_path, mmap_mode='c')
        # Use memory-efficient statistics calculation
        unique_labels_final = memory_efficient_unique(roi_array_path)
        num_labels = len(unique_labels_final[unique_labels_final > 0])

        logger.info(f"ROI_PREPROCESS - Final ROI statistics:")
        logger.info(f"ROI_PREPROCESS - Total voxels: {roi_array.size}")
        logger.info(f"ROI_PREPROCESS - Non-zero voxels: {final_volume}")
        logger.info(f"ROI_PREPROCESS - Unique labels: {num_labels}")
        logger.info(f"ROI_PREPROCESS - Memory usage: {roi_array.nbytes / (1024*1024):.1f} MB")

        # Clean RAM
        del roi_array
        gc.collect()

    except MemoryError:
        logger.warning(f"ROI_PREPROCESS - Memory constraints prevented detailed statistics")
        logger.info(f"ROI_PREPROCESS - Final ROI volume: {final_volume} voxels")


def get_roi_statistics(roi_array: Optional[Union[str, Dict]]) -> dict:      # gets array path   toto
    """
    Get statistics about ROIs in the array.
    
    Args:
        roi_array: Input ROI array or RTSTRUCT dictionary or path to mask on disk {roi_name: mask_array}

        path2roi: Path to RoI on disk
        
    Returns:
        Dictionary containing ROI statistics
    """
    # Load RoI from disk
    # Handle RTSTRUCT case (dictionary of ROIs)
    if isinstance(roi_array, dict):
        return _get_rtstruct_roi_statistics(roi_array)

    # Load roi array from disk
    roi_array_disk = np.load(roi_array, mmap_mode='r')

    # Handle single ROI case
    unique_labels = memory_efficient_unique(roi_array)
    unique_labels = unique_labels[unique_labels > 0]
    
    roi_stats = {
        'total_rois': len(unique_labels),
        'total_voxels': np.sum(roi_array_disk > 0),
        'roi_volumes': {},
        'roi_labels': list(unique_labels)
    }
    
    for label_value in unique_labels:
        volume = np.sum(roi_array_disk == label_value)
        roi_stats['roi_volumes'][int(label_value)] = volume

    # Clean RAM
    del roi_array_disk
    gc.collect
    return roi_stats


def _get_rtstruct_roi_statistics(rtstruct_dict: dict) -> dict:
    """
    Get statistics about ROIs in RTSTRUCT data.

    Args:
        rtstruct_dict: Dictionary of {roi_name: mask_array}

    Returns:
        Dictionary containing RTSTRUCT ROI statistics
    """
    roi_stats = {
        'total_rois': len(rtstruct_dict),
        'total_voxels': 0,
        'roi_volumes': {},
        'roi_labels': list(rtstruct_dict.keys())
    }

    for roi_name, roi_mask in rtstruct_dict.items():
        # Load RoI from disk        toto
        roi_mask_disk = np.load(roi_mask, mmap_mode='r').astype(np.float32)
        if roi_mask_disk is not None and roi_mask_disk.size > 0:
            volume = np.sum(roi_mask_disk > 0)
            roi_stats['roi_volumes'][roi_name] = volume
            roi_stats['total_voxels'] += volume

        # Clean RAM
        del roi_mask_disk
        gc.collect()

    return roi_stats