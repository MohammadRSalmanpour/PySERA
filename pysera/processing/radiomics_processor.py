"""
Main radiomics processing module that orchestrates the entire pipeline.
"""

import os
import sys
import logging
import time
import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Dict, Any, List, Tuple, Union
import pandas as pd
from datetime import datetime
import numpy as np
from pathlib import Path
from scipy.ndimage import label
from ..processing.synthesize_RoIs import synthesis_small_RoI
try:
    from ..config.settings import (
        DEFAULT_RADIOICS_PARAMS, DEFAULT_MIN_ROI_VOLUME, EXPECTED_FEATURE_COUNTS,
        OUTPUT_FILENAME_TEMPLATE, get_visera_pythoncode_path, get_default_output_path
    )
except ImportError:
    from pysera.config.settings import (
        DEFAULT_RADIOICS_PARAMS, DEFAULT_MIN_ROI_VOLUME, EXPECTED_FEATURE_COUNTS,
        OUTPUT_FILENAME_TEMPLATE, get_visera_pythoncode_path, get_default_output_path
    )

try:
    from ..utils.file_utils import (
        detect_file_format, find_files_by_format, match_image_mask_pairs, ensure_directory_exists
    )
    from ..utils.utils import save_numpy_on_disk, remove_temp_file
    # from ..utils.ram_runtime_logging import RuntimeRAMLogger
except ImportError:
    from pysera.utils.file_utils import (
        detect_file_format, find_files_by_format, match_image_mask_pairs, ensure_directory_exists
    )
    from pysera.utils.utils import save_numpy_on_disk

try:
    from ..data.dicom_loader import convert_dicom_to_arrays
except ImportError:
    from pysera.data.dicom_loader import convert_dicom_to_arrays

try:
    from ..preprocessing.roi_preprocessing import optimize_roi_preprocessing, get_roi_statistics
except ImportError:
    from pysera.preprocessing.roi_preprocessing import optimize_roi_preprocessing, get_roi_statistics

# Handle optional imports that might not exist
try:
    from ..preprocessing.intensity_preprocessing import apply_intensity_preprocessing
except ImportError:
    # Provide a simple fallback for intensity preprocessing
    def apply_intensity_preprocessing(image_array, image_id, params):
        logger.info(f"[{image_id}] Using simple intensity preprocessing (fallback)")
        logger.info(f"INTENSITY_PREPROCESS - Debug - image_array type: {type(image_array)}, shape: {image_array.shape}, dtype: {image_array.dtype}")
        logger.info(f"INTENSITY_PREPROCESS - Debug - image_id type: {type(image_id)}, value: {image_id}")
        logger.info(f"INTENSITY_PREPROCESS - Debug - params type: {type(params)}")
        if isinstance(params, dict):
            logger.info(f"INTENSITY_PREPROCESS - Debug - params keys: {list(params.keys())}")
        return image_array

try:
    from ..utils.mock_modules import setup_mock_modules
except ImportError:
    def setup_mock_modules(): pass

try:
    from ..utils.save_params import write_to_excel
except ImportError:
    def write_to_excel(*args, **kwargs): pass

try:
    from ..features.feature_names import get_feature_names
except ImportError:
    def get_feature_names(*args, **kwargs): return []

try:
    from ..utils.log_logging import (
        log_logger,
        MemoryLogHandler,
        setup_multiprocessing_logging,
        init_worker_logging,
    )
except ImportError:
    MemoryLogHandler = None
logger = logging.getLogger("Dev_logger")

class RadiomicsProcessor:
    """
    Main class for orchestrating radiomics feature extraction.
    """

    def __init__(self, output_path: Optional[str] = None, memory_handler: Optional[MemoryLogHandler] = None):
        """
        Initialize the radiomics processor.

        Args:
            output_path: Output directory path (optional)
        """
        self.output_path = output_path or get_default_output_path()
        ensure_directory_exists(self.output_path)

        # define memory handler to log on excel
        self.memory_handler = memory_handler

        # Set up mock modules for dependencies
        setup_mock_modules()

        # Import SERA module
        self._setup_sera_module()

        # Initialize parameters
        self.params = DEFAULT_RADIOICS_PARAMS.copy()
        self.params['radiomics_destfolder'] = self.output_path

    def _setup_sera_module(self) -> None:
        """Set up the SERA module for feature extraction."""
        sera_pythoncode_dir = get_visera_pythoncode_path()
        if sera_pythoncode_dir not in sys.path:
            sys.path.insert(0, sera_pythoncode_dir)

        try:
            import importlib
            rf_main_module = importlib.import_module('pysera.engine.visera.RF_main')
            self.SERA_FE_main = rf_main_module.SERA_FE_main
            logger.info("RF_main module imported successfully")
        except ImportError as e:
            logger.error(f"Failed to import RF_main module: {e}")
            raise

    def process_single_image_pair(self, args_tuple: Tuple) -> Optional[pd.DataFrame]:
        """
        Process a single image-mask pair.

        Args:
            args_tuple: Tuple containing (image_input, mask_input, params, output_folder,
                        apply_preprocessing, min_roi_volume, folder_name, feats2out, roi_num, roi_selection_mode)

        Returns:
            DataFrame with extracted features or None if failed
        """
        (image_input, mask_input, params, output_folder, apply_preprocessing,
         min_roi_volume, folder_name, feats2out, roi_num, roi_selection_mode, feature_value_mode) = args_tuple

        # Add unique identifier for this processing
        image_id = os.path.basename(image_input)
        try:
            from pysera.utils.log_logging import set_image_context
            set_image_context(image_id)
        except Exception:
            pass
        logger.info(f"=== PROCESSING IMAGE {image_id} ===")
        logger.info(f"Image path: {image_input}")
        logger.info(f"Mask path: {mask_input}")

        try:
            start_time = time.time()

            # Load and convert image and mask data
            image_data = self._load_image_data(image_id, image_input, mask_input)
            if image_data is None:
                return None
            image_array, image_metadata, mask_array, mask_metadata = image_data

            # Get ROI statistics
            if isinstance(mask_array, dict):
                total_rois = 0
                for roi_name, roi_mask in mask_array.items():
                    roi_stats = get_roi_statistics(roi_mask)
                    logger.info(f"[{image_id}] RTSTRUCT ROI '{roi_name}': {roi_stats['total_voxels']} voxels")
                    total_rois += 1 if roi_stats['total_voxels'] > 0 else 0
                logger.info(f"[{image_id}] Found {total_rois} Unique ROI(s) in RTSTRUCT mask")
                if total_rois == 0:
                    logger.warning(f"[{image_id}] No ROIs found in RTSTRUCT mask")
                    return None
            else:
                roi_stats = get_roi_statistics(mask_array)
                logger.info(f"[{image_id}] Found {roi_stats['total_rois']} Unique ROI(s) in mask")
                if roi_stats['total_rois'] == 0:
                    logger.warning(f"[{image_id}] No ROIs found in mask")
                    return None

            # Prepare parameters for SERA
            params_copy = self._prepare_sera_parameters(params, image_array, image_metadata,
                                                       mask_array, mask_metadata, feats2out, roi_num, roi_selection_mode)
            params_copy['feature_value_mode'] = feature_value_mode
            # Process with SERA
            result = self._run_sera_processing(image_id, params_copy, apply_preprocessing, min_roi_volume)

            end_time = time.time()
            processing_time = end_time - start_time
            logger.info(f"[{image_id}] Processing completed in {processing_time:.2f} seconds")

            if result is not None:
                df = result
                df.insert(0, 'PatientID', os.path.basename(image_input))

                # Final quality check
                self._perform_final_quality_check(image_id, df)

                return df
            else:
                logger.error(f"[{image_id}] FAILED: No result returned from SERA processing")
                return None

        except Exception as e:
            logger.error(f"[{image_id}] CRITICAL ERROR: {e}")
            import traceback
            logger.error(f"[{image_id}] Traceback: {traceback.format_exc()}")
            return None

    def _load_image_data(self, image_id: str, image_input: str, mask_input: str) -> Optional[Tuple]:
        """Load and convert image and mask data."""
        logger.info(f"[{image_id}] Loading and converting image and mask data...")
        image_array, image_metadata, mask_array_path, mask_metadata = convert_dicom_to_arrays(image_input, mask_input)         # array path

        if image_array is None or mask_array_path is None:
            logger.error(f"[{image_id}] CRITICAL: Failed to load image or mask data")
            return None

        if isinstance(mask_array_path, dict):
            logger.info(f"[{image_id}] Successfully loaded - Image shape: {image_array.shape}, RTSTRUCT mask with {len(mask_array_path)} ROIs: {list(mask_array_path.keys())}")
        else:
            mask_array = np.load(mask_array_path, mmap_mode='r')
            logger.info(f"[{image_id}] Successfully loaded - Image shape: {image_array.shape}, Mask shape: {mask_array.shape}")
            # Clean ram 
            del mask_array
            gc.collect()

        return image_array, image_metadata, mask_array_path, mask_metadata

    def _prepare_sera_parameters(self, params: Dict[str, Any], image_array: np.ndarray,
                               image_metadata: Dict, mask_array: np.ndarray,
                               mask_metadata: Dict, feats2out: int, roi_num: int,
                               roi_selection_mode: str) -> Dict[str, Any]:
        """Prepare parameters for SERA processing."""
        params_copy = params.copy()
        params_copy['da_original'] = [image_array, image_metadata, image_metadata['format'].title(), None]
        params_copy['da_label'] = [mask_array, mask_metadata, mask_metadata['format'].title(), None]
        params_copy['radiomics_Feats2out'] = feats2out
        params_copy['radiomics_ROI_num'] = roi_num
        params_copy['radiomics_ROI_selection_mode'] = roi_selection_mode
        params_copy['feature_value_mode'] = params.get('feature_value_mode', 'REAL_VALUE')
        return params_copy

    def _perform_final_quality_check(self, image_id: str, df: pd.DataFrame) -> None:
        """Perform final quality check on results."""
        nan_count = df.isnull().sum().sum()
        total_values = len(df) * (len(df.columns) - 3)  # Exclude metadata columns
        logger.info(f"[{image_id}] FINAL RESULT: {100*nan_count/total_values:.1f}% missing values")

    def _run_sera_processing(self, image_id: str, params: Dict[str, Any],
                           apply_preprocessing: bool, min_roi_volume: int) -> Optional[pd.DataFrame]:
        """
        Run SERA feature extraction with preprocessing.

        Args:
            image_id: Image identifier for logging
            params: Processing parameters
            apply_preprocessing: Whether to apply preprocessing
            min_roi_volume: Minimum ROI volume threshold

        Returns:
            DataFrame with extracted features or None if failed
        """
        def optimized_sera_with_preprocessing(da_original, da_label, *args, **kwargs):
            logger.info(f"[{image_id}] Running optimized SERA function with preprocessing")

            # Extract parameters from args
            sera_params = self._extract_sera_parameters(args)
            data_original = da_original[0].copy()
            data_label = da_label[0]
            VoxelSizeInfo = da_original[1]['spacing']
            # Cleam RAM
            del da_original
            gc.collect()

            # Add VoxelSizeInfo to sera_params so it can be accessed in feature extraction
            sera_params['VoxelSizeInfo'] = VoxelSizeInfo

            # Log input data shapes
            if isinstance(data_label, dict):
                roi_shapes = {}
                for roi, mask in data_label.items():
                    # Load mask from disk
                    mask_disk = np.load(mask, mmap_mode='r')
                    roi_shapes[roi] = mask_disk.shape
                logger.info(f"[{image_id}] Input data shapes - Image: {data_original.shape}, RTSTRUCT mask with {len(data_label)} ROIs: {roi_shapes}")
            else:
                # Load mask from disk
                mask_disk = np.load(data_label, mmap_mode='r')
                logger.info(f"[{image_id}] Input data shapes - Image: {data_original.shape}, Label: {mask_disk.shape}")
            # Clean RAM
            del mask_disk
            gc.collect()

            logger.info(f"[{image_id}] Voxel spacing: {VoxelSizeInfo}")

            # Apply preprocessing if requested          # toto returns path to numpy arrays
            if apply_preprocessing:
                data_original, data_label = self._apply_preprocessing_to_data(
                    image_id, data_original, data_label, min_roi_volume
                )

            return self._extract_features_for_rois(
                image_id, data_original, data_label, VoxelSizeInfo, min_roi_volume, sera_params
            )


        # Call SERA function
        result = optimized_sera_with_preprocessing(
            params['da_original'],
            params['da_label'],
            params['radiomics_BinSize'],
            params['radiomics_isotVoxSize'],
            params['radiomics_isotVoxSize2D'],
            params['radiomics_DataType'],
            params['radiomics_DiscType'],
            params['radiomics_qntz'],
            params['radiomics_VoxInterp'],
            params['radiomics_ROIInterp'],
            params['radiomics_isScale'],
            params['radiomics_isGLround'],
            params['radiomics_isReSegRng'],
            params['radiomics_isOutliers'],
            params['radiomics_isQuntzStat'],
            params['radiomics_isIsot2D'],
            params['radiomics_ReSegIntrvl01'],
            params['radiomics_ReSegIntrvl02'],
            params['radiomics_ROI_PV'],
            params['radiomics_IVH_Type'],
            params['radiomics_IVH_DiscCont'],
            params['radiomics_IVH_binSize'],
            params['radiomics_ROI_num'],
            params['radiomics_ROI_selection_mode'],
            params['radiomics_isROIsCombined'],
            params['radiomics_Feats2out'],
            params['radiomics_destfolder'],
            params.get('feature_value_mode', 'REAL_VALUE')
        )

        return result

    def _extract_sera_parameters(self, args: Tuple) -> Dict[str, Any]:
        """Extract SERA parameters from args tuple."""
        return {
            'BinSize': args[0],
            'isotVoxSize': args[1],
            'isotVoxSize2D': args[2],
            'DataType': args[3],
            'DiscType': args[4],
            'qntz': args[5],
            'VoxInterp': args[6],
            'ROIInterp': args[7],
            'isScale': args[8],
            'isGLround': args[9],
            'isReSegRng': args[10],
            'isOutliers': args[11],
            'isQuntzStat': args[12],
            'isIsot2D': args[13],
            'ReSegIntrvl01': args[14],
            'ReSegIntrvl02': args[15],
            'ROI_PV': args[16],
            'IVH_Type': None if args[17] is None else int(args[17]),
            'IVH_DiscCont': None if args[18] is None else int(args[18]),
            'IVH_binSize': None if args[19] is None else float(args[19]),
            'ROI_num': args[20],
            'ROI_selection_mode': args[21],
            'isROIsCombined': args[22],
            'Feats2out': args[23],
            'destfolder': args[24],
            'feature_value_mode': args[25] if len(args) > 25 else 'REAL_VALUE'
        }

    def _apply_preprocessing_to_data(self, image_id: str, data_original: np.ndarray, data_label: Any, min_roi_volume: int) -> Tuple[np.ndarray, Any]:
        """
        Apply preprocessing to image and mask data.
        For RTSTRUCT, data_label is a dict of {roi_name: mask_array}.
        For other masks, data_label is a single array.
        """
        processed_image = data_original.copy()
        if isinstance(data_label, dict):
            processed_label = {}        # toto path to mask on disk
            for roi_name, roi_mask in data_label.items():
                processed_label[roi_name] = optimize_roi_preprocessing(roi_mask, min_roi_volume)       # returns path
            return processed_image, processed_label
        else:
            processed_label = optimize_roi_preprocessing(data_label, min_roi_volume)
            return processed_image, processed_label

    def _extract_features_for_rois(self, image_id: str, data_original: np.ndarray,
                                 data_label: Any, VoxelSizeInfo: list,
                                 min_roi_volume: int, sera_params: dict) -> Optional[pd.DataFrame]:
        """
        Extract features for all ROIs in the image.
        For RTSTRUCT, data_label is a dict of {roi_name: mask_array}.
        For other masks, data_label is a single array.
        """
        if isinstance(data_label, dict):
            # RTSTRUCT: each ROI is a separate mask
            roi_num = sera_params.get('ROI_num', 10)
            roi_selection_mode = sera_params.get('ROI_selection_mode', 'per_Img')

            # Collect all ROIs with their volumes
            all_rois = []
            for roi_name, roi_mask in data_label.items():
                # Load roi from disk
                roi_mask_disk = np.load(roi_mask, mmap_mode='r')
                volume = np.sum(roi_mask_disk > 0)
                # clean RAM
                del roi_mask_disk
                gc.collect()

                if volume < min_roi_volume:
                    logger.info(f"[{image_id}] RTSTRUCT ROI '{roi_name}' skipped (volume {volume} < min_roi_volume {min_roi_volume})")
                    # toto: maybe add smth here e.g. logger
                    continue
                all_rois.append((roi_name, roi_mask, volume))
                

            if not all_rois:
                logger.error(f"[{image_id}] CRITICAL: No RTSTRUCT ROIs meet the volume threshold!")
                return None

            # Apply ROI selection policy for RTSTRUCT               toto returns path
            selected_rois = self._apply_rtstruct_roi_selection_policy(
                image_id, all_rois, roi_num, roi_selection_mode
            )

            # Clear RAM
            del all_rois
            gc.collect()

            # Process selected ROIs
            all_features = []
            roi_names = []
            processed_rois = []
            skipped_rois = []
            for roi_name, roi_mask, volume in selected_rois:        # toto gets path to np arrays
                
                # Set ROI context for logging
                try:
                    from pysera.utils.log_logging import set_roi_context
                    set_roi_context(roi_name)
                except Exception:
                    pass
                processed_rois.append((roi_name, volume))
                feature_result = self._extract_features_for_single_roi(
                    image_id, data_original, roi_mask, roi_name, sera_params
                )
                if feature_result is not None:
                    all_features.append(feature_result)
                    roi_names.append(roi_name)
                else:
                    skipped_rois.append((roi_name, volume))

                # Clean disk
                remove_temp_file(roi_mask)
                # Clear ROI context after processing this ROI
                try:
                    from pysera.utils.log_logging import clear_roi_context
                    clear_roi_context()
                except Exception:
                    pass

            if not all_features:
                logger.error(f"[{image_id}] CRITICAL: No features extracted from any RTSTRUCT ROI")
                return None

            return self._create_results_dataframe(
                image_id, all_features, roi_names, sera_params, processed_rois, skipped_rois, min_roi_volume
            )
        else:
            # Non-RTSTRUCT: old behavior
            all_rois = self._get_all_rois_from_mask(data_label, image_id, use_disk=True)
            # Clean disk
            remove_temp_file(data_label)
            if len(all_rois) == 0:
                logger.error(f"[{image_id}] CRITICAL: No ROIs found in mask!")
                return None
            self._log_roi_volume_summary(image_id, all_rois, min_roi_volume)
            selected_rois_disk, skipped_rois = self._select_rois(image_id, all_rois, min_roi_volume, sera_params)

            # Process single roi        toto
            processed_rois = []
            all_features = []
            roi_names = []
            for label_idx, (label_value, roi_id, volume, mask, roi_identifier) in enumerate(selected_rois_disk):
                processed_rois.append((roi_identifier, volume))
                roi_name = f"{roi_identifier}"

                # Set ROI context for logging
                try:
                    from pysera.utils.log_logging import set_roi_context
                    set_roi_context(roi_name)
                except Exception:
                    pass

                feature_result = self._extract_features_for_single_roi(
                    image_id, data_original, mask, roi_name, sera_params
                )

                if feature_result is not None:
                    all_features.append(feature_result)
                    roi_names.append(roi_name)

                # Clea disk
                remove_temp_file(mask)
                # Clear ROI context after processing this ROI
                try:
                    from pysera.utils.log_logging import clear_roi_context
                    clear_roi_context()
                except Exception:
                    pass

            if not all_features:
                logger.error(f"[{image_id}] CRITICAL: No features extracted from any ROI")
                return None
            return self._create_results_dataframe(
                image_id, all_features, roi_names, sera_params, processed_rois, skipped_rois, min_roi_volume
            )

    def _apply_rtstruct_roi_selection_policy(self, image_id: str, valid_rois: List[Tuple],
                                           roi_num: int, selection_mode: str) -> List[Tuple]:
        """
        Apply ROI selection policy for RTSTRUCT ROIs.

        Args:
            image_id: Image identifier for logging
            valid_rois: List of (roi_name, mask, volume) tuples
            roi_num: Number of ROIs to select
            selection_mode: 'per_Img' or 'per_region'

        Returns:
            List of selected (roi_name, mask, volume) tuples
        """
        if selection_mode == "per_Img":
            return self._select_rtstruct_rois_per_image(image_id, valid_rois, roi_num)
        elif selection_mode == "per_region":
            return self._select_rtstruct_rois_per_region(image_id, valid_rois, roi_num)
        else:
            logger.warning(f"[{image_id}] Unknown selection mode: {selection_mode}. Using per_Img.")
            return self._select_rtstruct_rois_per_image(image_id, valid_rois, roi_num)

    def _select_rtstruct_rois_per_image(self, image_id: str, valid_rois: List[Tuple], roi_num: int) -> List[Tuple]:
        """Select RTSTRUCT ROIs per image (ignore region grouping)."""
        # Sort by volume (largest first) and take top roi_num
        sorted_rois = sorted(valid_rois, key=lambda x: x[2], reverse=True)  # x[2] is volume
        selected = sorted_rois[:roi_num]

        logger.info(f"[{image_id}] RTSTRUCT per-image selection: {len(selected)}/{len(valid_rois)} ROIs selected")
        for roi_name, _, volume in selected:
            logger.info(f"[{image_id}]   Selected RTSTRUCT ROI '{roi_name}': {volume} voxels")

        return selected

    def _select_rtstruct_rois_per_region(self, image_id: str, valid_rois: List[Tuple], roi_num: int) -> List[Tuple]:
        """Group RTSTRUCT ROIs by name prefix and select from each group."""
        # Group ROIs by name prefix (e.g., "GTV", "CTV", "PTV")
        region_groups = self._group_rtstruct_rois_by_region(image_id, valid_rois)

        logger.info(f"[{image_id}] RTSTRUCT per-region selection: {len(region_groups)} region groups found")
        logger.info(f"[{image_id}] Will select up to {roi_num} ROIs from each region group")

        selected_rois = []

        for group_idx, group_rois in enumerate(region_groups):
            # Sort group ROIs by volume and select up to roi_num from this group
            sorted_group = sorted(group_rois, key=lambda x: x[2], reverse=True)  # x[2] is volume
            logger.info(f"[{image_id}] RTSTRUCT Group {group_idx} has {len(sorted_group)} ROIs")
            # Select up to roi_num ROIs from this group (or all if less than roi_num)
            group_selected = sorted_group[:roi_num]
            selected_rois.extend(group_selected)

            region_name = group_rois[0][0].split('_')[0] if '_' in group_rois[0][0] else group_rois[0][0]
            logger.info(f"[{image_id}]   RTSTRUCT Region Group {group_idx + 1} ({region_name}): {len(group_selected)}/{len(group_rois)} ROIs selected")
            for roi_name, _, volume in group_selected:
                logger.info(f"[{image_id}]     Selected RTSTRUCT ROI '{roi_name}': {volume} voxels")

        logger.info(f"[{image_id}] Total RTSTRUCT ROIs selected across all region groups: {len(selected_rois)}")

        return selected_rois

    def _group_rtstruct_rois_by_region(self, image_id: str, valid_rois: List[Tuple]) -> List[List[Tuple]]:
        """
        Group RTSTRUCT ROIs by name prefix (e.g., "GTV", "CTV", "PTV").

        Args:
            image_id: Image identifier for logging
            valid_rois: List of (roi_name, mask, volume) tuples

        Returns:
            List of ROI groups (each group is a list of tuples)
        """
        if len(valid_rois) <= 1:
            return [valid_rois]

        # Group ROIs by name prefix (before underscore or use full name)
        region_groups = {}
        for roi_name, mask, volume in valid_rois:
            # Extract region prefix (e.g., "GTV" from "GTV_Primary")
            region_prefix = roi_name.split('_')[0] if '_' in roi_name else roi_name
            if region_prefix not in region_groups:
                region_groups[region_prefix] = []
            region_groups[region_prefix].append((roi_name, mask, volume))

        # Convert to list of groups
        groups = list(region_groups.values())

        # Sort groups by the region prefix for consistent ordering
        groups.sort(key=lambda group: group[0][0])  # Sort by first ROI's name

        logger.info(f"[{image_id}] Grouped {len(valid_rois)} RTSTRUCT ROIs into {len(groups)} region groups")
        for i, group in enumerate(groups):
            region_prefix = group[0][0].split('_')[0] if '_' in group[0][0] else group[0][0]
            logger.info(f"[{image_id}]   RTSTRUCT Region Group {i + 1} ({region_prefix}): {len(group)} ROIs")
            for roi_name, _, volume in group:
                logger.info(f"[{image_id}]     ROI '{roi_name}': {volume} voxels")

        return groups

    # def _get_all_rois_from_mask(self, data_label: str, image_id: str) -> List[Tuple]:         # toto CMec     replaced with _save_all_RoIs_on_disk
    #     """
    #     Get all ROIs from mask using connected components analysis.

    #     Args:
    #         data_label: Label mask array
    #         image_id: Image identifier for logging

    #     Returns:
    #         List of (label_value, roi_id, volume, mask) tuples
    #     """
    #     from scipy.ndimage import label

    #     all_rois = []

    #     # Get unique labels (excluding 0)
    #     unique_labels = np.unique(data_label)
    #     unique_labels = unique_labels[unique_labels > 0]

    #     logger.info(f"[{image_id}] Found {len(unique_labels)} unique label values: {unique_labels}")

    #     # Process each label value
    #     for lbl in unique_labels:
    #         binary_mask = (data_label == lbl)  # binary mask for current label
    #         labeled_array, num_features = label(binary_mask)  # connected components

    #         logger.info(f"[{image_id}] Label {lbl}: {num_features} connected ROIs found")

    #         # Process each connected component for this label
    #         for roi_id in range(1, num_features + 1):
    #             roi_mask = (labeled_array == roi_id).astype(np.float32)
    #             volume = np.sum(roi_mask)

    #             # Create a unique identifier for this ROI
    #             roi_identifier = f"label_{lbl}_lesion_{roi_id}"

    #             all_rois.append((lbl, roi_id, volume, roi_mask, roi_identifier))

    #     logger.info(f"[{image_id}] Total ROIs found: {len(all_rois)}")
    #     return all_rois



    def _get_all_rois_from_mask(self, data_label: str, image_id: str, use_disk=True) -> List[Tuple]:           # toto
        """
        Get all ROIs from mask using connected components analysis.

        Args:
            data_label: Label mask array
            image_id: Image identifier for logging

        Returns:
            List of (label_value, roi_id, volume, mask) tuples
        """
        from scipy.ndimage import label

        # Get unique labels (excluding 0)
        unique_labels = self._get_roi_unique_labels(data_label)
        logger.info(f"[{image_id}] Found {len(unique_labels)} unique label values: {unique_labels}")
            
        all_rois = []
        for lbl in unique_labels:
            data_label_disk = np.load(data_label, mmap_mode='r')
            binary_mask = (data_label_disk == lbl)  # binary mask for current label
            del data_label_disk
            gc.collect()
            labeled_array, num_features = label(binary_mask)  # connected components
            logger.info(f"[{image_id}] Label {lbl}: {num_features} connected ROIs found")
            # Clean RAM
            del binary_mask
            gc.collect()

            # Process each connected component for this label
            for roi_id in range(1, num_features + 1):
                roi_mask = (labeled_array == roi_id).astype(np.float32)
                volume = np.sum(roi_mask)

                # Create a unique identifier for this ROI
                roi_identifier = f"label_{lbl}_lesion_{roi_id}"

                # save RoI on disk
                if use_disk:
                    roi_mask_input = save_numpy_on_disk(roi_mask, prefix=roi_identifier, suffix=".npy")
                    all_rois.append((lbl, roi_id, volume, roi_mask_input, roi_identifier))
                else:
                    all_rois.append((lbl, roi_id, volume, roi_mask, roi_identifier))

                # Clean RAM
                del roi_mask
                gc.collect()

        logger.info(f"[{image_id}] Total ROIs found: {len(all_rois)}")
        
        return all_rois



    def _get_roi_unique_labels(self, data_label):

        # load rois from disk
        data_label = np.load(data_label, mmap_mode='r')
        
        unique_labels = np.unique(data_label)
        unique_labels = unique_labels[unique_labels > 0]

        # delete rois from ram
        del data_label
        gc.collect()

        return unique_labels

    def _log_roi_volume_summary(self, image_id: str, all_rois: List[Tuple], min_roi_volume: int) -> None:
        """Log ROI volume summary."""
        logger.info(f"[{image_id}] ROI Volume Summary (min_roi_volume = {min_roi_volume}):")
        for label_value, roi_id, volume, _, roi_identifier in all_rois:
            status = "✓ PROCESS" if volume >= min_roi_volume else "✗ SKIP"
            logger.info(f"[{image_id}]   ROI '{roi_identifier}': {volume} voxels - {status}")


    def _filter_rois(self, image_id: str, all_rois: List[Tuple], min_roi_volume: int):      # toto      come from _process_all_rois

        # First, filter ROIs by volume threshold
        valid_rois = []
        skipped_rois = []           # toto should we keep it?

        for label_value, roi_id, volume, mask, roi_identifier in all_rois:
            if volume < min_roi_volume:
                skipped_rois.append((roi_identifier, volume))
                logger.info(f"[{image_id}] ROI '{roi_identifier}' skipped (volume {volume} < min_roi_volume {min_roi_volume})")
            else:
                valid_rois.append((label_value, roi_id, volume, mask, roi_identifier))

        if not valid_rois:
            logger.warning(f"[{image_id}] No ROIs meet the volume threshold ({min_roi_volume})")
            return [], [], [], skipped_rois

        return valid_rois, skipped_rois

    # def _process_single_rois(self, selected_roi: Tuple, image_id, data_original, sera_params):            # toto CMed
        # Process selected ROI

        # all_features = []
        # roi_names = []
        # processed_rois = []

        # for label_idx, (label_value, roi_id, volume, mask, roi_identifier) in enumerate(selected_rois):
            # processed_rois.append((roi_identifier, volume))
            # roi_name = f"{roi_identifier}"

            # feature_result = self._extract_features_for_single_roi(
            #     image_id, data_original, mask, roi_name, sera_params
            # )

            # if feature_result is not None:
            #     all_features.append(feature_result)
            #     roi_names.append(roi_name)

        # return all_features, roi_names, processed_rois, skipped_rois
    

    def _select_rois(self, image_id: str, all_rois: List[Tuple],
                          min_roi_volume: int, sera_params: Dict[str, Any]) -> Tuple:
        """Process all ROIs and return results with new selection policy."""
        # Filter RoIs
        valid_rois, skipped_rois = self._filter_rois(image_id, all_rois, min_roi_volume)

        # Get ROI selection parameters
        roi_num = sera_params.get('ROI_num', 10)
        roi_selection_mode = sera_params.get('ROI_selection_mode', 'per_Img')

        # Apply ROI selection policy
        selected_rois = self._apply_roi_selection_policy(
                        image_id, valid_rois, roi_num, roi_selection_mode
                        )

        del valid_rois

        # Save RoIs on disk to load one by one      toto
        selected_rois_disk = []
        for label, roi_id, volume, mask, roi_identifier in selected_rois:
            # mask_array_path = save_numpy_on_disk(mask, prefix=roi_identifier, suffix=".npy")
            selected_rois_disk.append((label, roi_id, volume, mask, roi_identifier))

        # Clear RAM
        del selected_rois

        return selected_rois_disk, skipped_rois


        
    
    # def _process_all_rois(self, image_id: str, data_original: np.ndarray, data_label: np.ndarray,
    #                      all_rois: List[Tuple], min_roi_volume: int, sera_params: Dict[str, Any]) -> Tuple:
    #     """Process all ROIs and return results with new selection policy."""
    #     # Get ROI selection parameters
    #     roi_num = sera_params.get('ROI_num', 10)
    #     roi_selection_mode = sera_params.get('ROI_selection_mode', 'per_Img')

    #     # First, filter ROIs by volume threshold
    #     valid_rois = []
    #     skipped_rois = []

    #     for label_value, roi_id, volume, mask, roi_identifier in all_rois:
    #         if volume < min_roi_volume:
    #             skipped_rois.append((roi_identifier, volume))
    #         elif volume < 10.:
    #             synth_volume, synth_mask = synthesis_small_RoI(int(volume), mask)        # Synthesize small RoIs (4<=, <=10)
    #             valid_rois.append((label_value, roi_id, synth_volume, synth_mask, roi_identifier))      # append synthesized RoI
    #             logger.warning(f"Insufficient volume to extract features (e.g. 'ivh_i10'). The ROI must contain at least 10 voxels. If the volume is 4 or more, the ROI will be synthesized up to 10 voxles.")
    #         else:
    #             valid_rois.append((label_value, roi_id, volume, mask, roi_identifier))

    #     if not valid_rois:
    #         logger.warning(f"[{image_id}] No ROIs meet the volume threshold ({min_roi_volume})")
    #         return [], [], [], skipped_rois

    #     # Apply ROI selection policy
    #     selected_rois = self._apply_roi_selection_policy(
    #         image_id, valid_rois, roi_num, roi_selection_mode
    #     )

    #     # Process selected ROIs
    #     all_features = []
    #     roi_names = []
    #     processed_rois = []

    #     for label_idx, (label_value, roi_id, volume, mask, roi_identifier) in enumerate(selected_rois):
    #         processed_rois.append((roi_identifier, volume))
    #         roi_name = f"{roi_identifier}"

    #         feature_result = self._extract_features_for_single_roi(
    #             image_id, data_original, mask, roi_name, sera_params
    #         )

    #         if feature_result is not None:
    #             all_features.append(feature_result)
    #             roi_names.append(roi_name)

    #     return all_features, roi_names, processed_rois, skipped_rois

    def _apply_roi_selection_policy(self, image_id: str, valid_rois: List[Tuple],
                                  roi_num: int, selection_mode: str) -> List[Tuple]:
        """
        Apply ROI selection policy based on mode.

        Args:
            image_id: Image identifier for logging
            valid_rois: List of (label_value, volume, mask) tuples
            roi_num: Number of ROIs to select
            selection_mode: 'per_Img' or 'per_region'

        Returns:
            List of selected (label_value, volume, mask) tuples
        """
        if selection_mode == "per_Img":
            return self._select_rois_per_image(image_id, valid_rois, roi_num)
        elif selection_mode == "per_region":
            return self._select_rois_per_region(image_id, valid_rois, roi_num)
        else:
            logger.warning(f"[{image_id}] Unknown selection mode: {selection_mode}. Using per_Img.")
            return self._select_rois_per_image(image_id, valid_rois, roi_num)

    def _select_rois_per_image(self, image_id: str, valid_rois: List[Tuple], roi_num: int) -> List[Tuple]:
        """Select ROIs per image (ignore region grouping)."""
        # Sort by volume (largest first) and take top roi_num
        sorted_rois = sorted(valid_rois, key=lambda x: x[2], reverse=True)  # x[2] is volume
        selected = sorted_rois[:roi_num]

        logger.info(f"[{image_id}] Per-image selection: {len(selected)}/{len(valid_rois)} ROIs selected")
        for label_value, roi_id, volume, _, roi_identifier in selected:
            logger.info(f"[{image_id}]   Selected ROI '{roi_identifier}': {volume} voxels")

        return selected

    def _select_rois_per_region(self, image_id: str, valid_rois: List[Tuple], roi_num: int) -> List[Tuple]:
        """Group ROIs by color and select from each group."""
        # Group ROIs by color (label value)
        color_groups = self._group_rois_by_region(image_id, valid_rois)

        logger.info(f"[{image_id}] Per-color selection: {len(color_groups)} color groups found")
        logger.info(f"[{image_id}] Will select up to {roi_num} ROIs from each color group")

        selected_rois = []

        for group_idx, group_rois in enumerate(color_groups):
            # Sort group ROIs by volume and select up to roi_num from this group
            sorted_group = sorted(group_rois, key=lambda x: x[2], reverse=True)  # x[2] is volume
            logger.info(f"[{image_id}] Group {group_idx} has {len(sorted_group)} ROIs")
            # Select up to roi_num ROIs from this group (or all if less than roi_num)
            group_selected = sorted_group[:roi_num]
            selected_rois.extend(group_selected)

            label_value = group_rois[0][0]  # Get the color/label value for this group
            logger.info(f"[{image_id}]   Color Group {group_idx + 1} (Label {label_value}): {len(group_selected)}/{len(group_rois)} ROIs selected")
            for label_value, roi_id, volume, _, roi_identifier in group_selected:
                logger.info(f"[{image_id}]     Selected ROI '{roi_identifier}': {volume} voxels")

        logger.info(f"[{image_id}] Total ROIs selected across all color groups: {len(selected_rois)}")

        return selected_rois

    def _group_rois_by_region(self, image_id: str, valid_rois: List[Tuple]) -> List[List[Tuple]]:
        """
        Group ROIs by color (label value).

        Args:
            image_id: Image identifier for logging
            valid_rois: List of (label_value, roi_id, volume, mask, roi_identifier) tuples

        Returns:
            List of ROI groups (each group is a list of tuples)
        """
        if len(valid_rois) <= 1:
            return [valid_rois]

        # Group ROIs by label value (color)
        color_groups = {}
        for label_value, roi_id, volume, mask, roi_identifier in valid_rois:
            if label_value not in color_groups:
                color_groups[label_value] = []
            color_groups[label_value].append((label_value, roi_id, volume, mask, roi_identifier))

        # Convert to list of groups
        groups = list(color_groups.values())

        # Sort groups by the label value for consistent ordering
        groups.sort(key=lambda group: group[0][0])  # Sort by first ROI's label value

        logger.info(f"[{image_id}] Grouped {len(valid_rois)} ROIs into {len(groups)} color groups")
        for i, group in enumerate(groups):
            label_value = group[0][0]  # Get label value for this group
            logger.info(f"[{image_id}]   Color Group {i + 1} (Label {label_value}): {len(group)} ROIs")
            for label_value, roi_id, volume, _, roi_identifier in group:
                logger.info(f"[{image_id}]     ROI '{roi_identifier}': {volume} voxels")

        return groups

    def _extract_features_for_single_roi(self, image_id: str, data_original: np.ndarray,
                                       mask_input: str, roi_name: str, sera_params: Dict[str, Any]) -> Optional[List]:
        """Extract features for a single ROI."""
        try:
            logger.info(f"[{image_id}] Extracting features for ROI '{roi_name}'")

            # Check mask and image properties before feature extraction
            self._validate_roi_data(image_id, data_original, mask_input, roi_name)

            # Try the feature extraction with error handling
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                logger.info(f"[{image_id}] Calling SERA_FE_main for ROI '{roi_name}'")
                result = self._call_sera_feature_extraction(data_original, mask_input, sera_params, roi_name)
                logger.info(f"[{image_id}] SERA function completed for ROI '{roi_name}'")

            if isinstance(result, list) and len(result) > 0:
                feature_vector = result[0]
                if isinstance(feature_vector, (list, np.ndarray)):
                    processed_vector = self._process_feature_vector(image_id, feature_vector, roi_name, sera_params['Feats2out'])
                    return processed_vector if processed_vector is not None else None
                else:
                    logger.warning(f"[{image_id}] Unexpected result format for ROI '{roi_name}': {type(feature_vector)}")
            else:
                logger.warning(f"[{image_id}] No features returned for ROI '{roi_name}'")

        except Exception as e:
            logger.error(f"[{image_id}] Error processing ROI '{roi_name}': {str(e)}")
            import traceback
            logger.error(f"[{image_id}] Traceback: {traceback.format_exc()}")

        return None

    def _validate_roi_data(self, image_id: str, data_original: np.ndarray,
                          mask_input: str, roi_name: str) -> None:
        """Validate ROI data before feature extraction."""
        # Load RoI
        mask = np.load(mask_input, mmap_mode='r')
        roi_intensities = data_original[mask > 0]
        logger.info(f"[{image_id}] ROI '{roi_name}' - Intensity stats: min={np.min(roi_intensities):.2f}, max={np.max(roi_intensities):.2f}, mean={np.mean(roi_intensities):.2f}, std={np.std(roi_intensities):.2f}")
        # Clean RAM
        del mask
        gc.collect()
        # Check for problematic intensity values
        if np.any(np.isnan(roi_intensities)) or np.any(np.isinf(roi_intensities)):
            logger.warning(f"[{image_id}] ROI '{roi_name}' contains NaN or Inf values!")

        if np.std(roi_intensities) == 0:
            logger.warning(f"[{image_id}] ROI '{roi_name}' has zero intensity variance!")

    def _call_sera_feature_extraction(self, data_original: np.ndarray, mask: str,
                                    sera_params: Dict[str, Any], roi_name: str) -> Any:
        """Call SERA feature extraction function."""
        return self.SERA_FE_main(
            data_original,
            mask,
            sera_params['VoxelSizeInfo'],
            sera_params['BinSize'],
            sera_params['DataType'],
            sera_params['isotVoxSize'],
            sera_params['isotVoxSize2D'],
            sera_params['DiscType'],
            sera_params['qntz'],
            sera_params['VoxInterp'],
            sera_params['ROIInterp'],
            sera_params['isScale'],
            sera_params['isGLround'],
            sera_params['isReSegRng'],
            sera_params['isOutliers'],
            sera_params['isQuntzStat'],
            sera_params['isIsot2D'],
            [sera_params['ReSegIntrvl01'], sera_params['ReSegIntrvl02']],
            sera_params['ROI_PV'],
            sera_params['Feats2out'],
            [sera_params['IVH_Type'], sera_params['IVH_DiscCont'], sera_params['IVH_binSize']],
            sera_params['feature_value_mode'],
            roi_name,
            sera_params['IVH_Type'],
            sera_params['IVH_DiscCont'],
            sera_params['IVH_binSize'],
            sera_params['isROIsCombined']
        )

    def _process_feature_vector(self, image_id: str, feature_vector: Any,
                              roi_name: str, feats2out: int) -> Optional[List]:
        """Process and validate feature vector."""
        # Count non-NaN features
        non_nan_count = np.sum(~np.isnan(feature_vector))
        total_count = len(feature_vector)
        logger.info(f"[{image_id}] ROI '{roi_name}': Extracted {non_nan_count}/{total_count} valid features")

        # Adjust feature vector size if needed
        expected_features = EXPECTED_FEATURE_COUNTS.get(feats2out, 215)
        if len(feature_vector) < expected_features:
            logger.warning(f"[{image_id}] ROI '{roi_name}': Got {len(feature_vector)} features, expected {expected_features}. Padding with NaN.")
            feature_vector = list(feature_vector) + [np.nan] * (expected_features - len(feature_vector))
        elif len(feature_vector) > expected_features:
            logger.warning(f"[{image_id}] ROI '{roi_name}': Got {len(feature_vector)} features, expected {expected_features}. Truncating.")
            feature_vector = feature_vector

        logger.info(f"[{image_id}] Successfully processed ROI '{roi_name}'")
        return feature_vector

    def _create_results_dataframe(self, image_id: str, all_features: List[List],
                                roi_names: List[str], sera_params: Dict[str, Any],
                                processed_rois: List[Tuple], skipped_rois: List[Tuple],
                                min_roi_volume: int) -> Optional[pd.DataFrame]:
        """Create results DataFrame."""
        try:
            feature_names = get_feature_names(sera_params['Feats2out'])
            actual_feature_count = len(all_features[0])
            expected_feature_count = len(feature_names)

            logger.info(f"[{image_id}] Feature extraction summary:")
            logger.info(f"[{image_id}]   - Expected features: {expected_feature_count}")
            logger.info(f"[{image_id}]   - Actual features: {actual_feature_count}")
            logger.info(f"[{image_id}]   - ROIs processed: {len(all_features)}")

            # Adjust feature names if needed
            feature_names = self._adjust_feature_names(feature_names, actual_feature_count, expected_feature_count, image_id)

            # Handle duplicate column names by making them unique
            unique_feature_names = self._make_column_names_unique(feature_names, image_id)

            df = pd.DataFrame(all_features, columns=unique_feature_names)
            df['ROI'] = roi_names
            df['File'] = os.path.basename(image_id)
            df['Bin_Size_Used'] = sera_params['BinSize']

            # Reorder columns to match expected format
            columns = ['File', 'ROI', 'Bin_Size_Used'] + unique_feature_names
            df = df.reindex(columns=columns)

            # Log processing summary
            self._log_processing_summary(image_id, processed_rois, skipped_rois, min_roi_volume)

            return df

        except Exception as e:
            logger.error(f"[{image_id}] Error creating DataFrame: {str(e)}")
            import traceback
            logger.error(f"[{image_id}] Traceback: {traceback.format_exc()}")
            return None

    def _adjust_feature_names(self, feature_names: List[str], actual_count: int,
                            expected_count: int, image_id: str) -> List[str]:
        """Adjust feature names based on actual feature count."""
        if actual_count != expected_count:
            logger.warning(f"[{image_id}] Feature count mismatch: expected {expected_count}, got {actual_count} features")

            if actual_count > expected_count:
                extended_names = feature_names.copy()
                for i in range(expected_count, actual_count):
                    extended_names.append(f"additional_feature_{i + 1 - expected_count}")
                feature_names = extended_names
            elif actual_count < expected_count:
                feature_names = feature_names[:actual_count]

        return feature_names

    def _make_column_names_unique(self, feature_names: List[str], image_id: str) -> List[str]:
        """Make column names unique by adding suffixes to duplicates."""
        from collections import Counter

        # Count occurrences of each name
        name_counts = Counter(feature_names)
        unique_names = []
        name_occurrences = {}

        for name in feature_names:
            if name_counts[name] > 1:
                # This is a duplicate, add suffix
                if name not in name_occurrences:
                    name_occurrences[name] = 1
                else:
                    name_occurrences[name] += 1
                unique_name = f"{name}_{name_occurrences[name]}"
                unique_names.append(unique_name)
            else:
                # This is unique, keep as is
                unique_names.append(name)

        # Log if duplicates were found
        duplicates = [name for name, count in name_counts.items() if count > 1]
        if duplicates:
            logger.warning(
                f"[{image_id}] Found duplicate feature names: {duplicates[:5]}{'...' if len(duplicates) > 5 else ''}")
            logger.warning(f"[{image_id}] Added suffixes to make column names unique")

        return unique_names

    def _log_processing_summary(self, image_id: str, processed_rois: List[Tuple],
                              skipped_rois: List[Tuple], min_roi_volume: int) -> None:
        """
        Log processing summary with helpful suggestions.

        Args:
            image_id: Image identifier
            processed_rois: List of processed ROIs
            skipped_rois: List of skipped ROIs
            min_roi_volume: Minimum ROI volume threshold
        """
        logger.info(f"[{image_id}] ═══ PROCESSING SUMMARY ═══")
        logger.info(f"[{image_id}] ✓ Processed ROIs: {len(processed_rois)}")
        if processed_rois:
            processed_volumes = [vol for _, vol in processed_rois]
            logger.info(f"[{image_id}]   └─ Volume range: {min(processed_volumes)} - {max(processed_volumes)} voxels")

        if skipped_rois:
            logger.warning(f"[{image_id}] ✗ Skipped ROIs: {len(skipped_rois)}")
            skipped_volumes = [vol for _, vol in skipped_rois]
            min_skipped = min(skipped_volumes)
            max_skipped = max(skipped_volumes)
            logger.warning(f"[{image_id}]   └─ Skipped volume range: {min_skipped} - {max_skipped} voxels")

            # Provide helpful suggestions
            suggested_threshold = max(min_skipped, DEFAULT_MIN_ROI_VOLUME)  # At least 10 voxels minimum
            if max_skipped < min_roi_volume:
                logger.warning(
                    f"[{image_id}] 💡 SUGGESTION: To include all ROIs, try --min-roi-volume {suggested_threshold}")
            logger.warning(
                f"[{image_id}] 💡 CURRENT: --min-roi-volume {min_roi_volume} (skipping {len(skipped_rois)} ROIs)")
            logger.warning(
                f"[{image_id}] 💡 TO INCLUDE ALL: --min-roi-volume {suggested_threshold} (would process {len(processed_rois) + len(skipped_rois)} ROIs)")
        else:
            logger.info(f"[{image_id}] ✓ All {len(processed_rois)} ROIs processed (none skipped)")

    def process_batch(self, image_input: Optional[Union[str, np.ndarray]] = None, mask_input: Optional[Union[str, np.ndarray]] = None,
                     apply_preprocessing: bool = True, min_roi_volume: int = DEFAULT_MIN_ROI_VOLUME,
                     num_workers: str = "auto", enable_parallelism: bool = True,
                     feats2out: int = DEFAULT_RADIOICS_PARAMS["radiomics_Feats2out"], bin_size: int = DEFAULT_RADIOICS_PARAMS["radiomics_BinSize"], roi_num: int = DEFAULT_RADIOICS_PARAMS["radiomics_ROI_num"],
                     roi_selection_mode: str = DEFAULT_RADIOICS_PARAMS["radiomics_ROI_selection_mode"], feature_value_mode: str = "REAL_VALUE",
                     optional_params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Process a batch of images and masks.

        Args:
            image_input: Path to image file/directory or numpy array
            mask_input: Path to mask file/directory or numpy array
            apply_preprocessing: Whether to apply preprocessing
            min_roi_volume: Minimum ROI volume threshold
            num_workers: Number of parallel workers
            enable_parallelism: Whether to disable parallel processing
            feats2out: Feature extraction mode
            bin_size: Bin size for discretization
            roi_num: Number of ROIs to select for feature extraction
            roi_selection_mode: ROI selection strategy
            feature_value_mode: Value type for feature extraction
            optional_params: Optional dictionary of additional radiomics parameter overrides

        Returns:
            Dictionary with processing results or None if failed
        """
        # start_total_time = time.time()        toto CMed

        # Set up logging        toto CMed
        # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        # log runtime and ram usage             toto 
        # RR_logger = RuntimeRAMLogger()

        # Merge optional parameter overrides first (if provided), then override with explicit function args
        if optional_params:
            for key, value in optional_params.items():
                # Only update known radiomics parameters to avoid accidental pollution
                if key in DEFAULT_RADIOICS_PARAMS or key in {
                    'feature_value_mode', 'radiomics_destfolder', 'radiomics_Feats2out', 'radiomics_ROI_num', 'radiomics_ROI_selection_mode'
                }:
                    # Skip overriding destination folder here; output path is controlled separately
                    if key == 'radiomics_destfolder':
                        continue
                    self.params[key] = value

        # Now set explicit function arguments to take precedence over any optional overrides
        self.params['radiomics_BinSize'] = bin_size
        self.params['radiomics_Feats2out'] = feats2out
        self.params['radiomics_ROI_num'] = roi_num
        self.params['radiomics_ROI_selection_mode'] = roi_selection_mode
        self.params['feature_value_mode'] = feature_value_mode

        # Check if we're working with numpy arrays directly
        if isinstance(image_input, np.ndarray) and isinstance(mask_input, np.ndarray):
            logger.info("Processing numpy arrays directly")

            # Validate array compatibility
            if image_input.shape != mask_input.shape:
                logger.error(f"Image and mask arrays must have the same shape. "
                           f"Image: {image_input.shape}, Mask: {mask_input.shape}")
                return None

            # Process the arrays directly
            return self._process_numpy_arrays(
                image_input, mask_input, apply_preprocessing, min_roi_volume,
                num_workers, enable_parallelism, apply_intensity_preprocessing=False
            )

        # Original file-based processing
        # Find image and mask files
        image_files, mask_files = self._find_input_files(image_input, mask_input)
        if not image_files or not mask_files:
            return None

        # Match image-mask pairs
        matched_pairs = match_image_mask_pairs(image_files, mask_files)
        if not matched_pairs:
            logger.error("No matching image-mask pairs found")
            return None

        # Sort pairs and extract file lists
        matched_pairs.sort(key=lambda x: os.path.basename(x[1]))
        image_files = [pair[0] for pair in matched_pairs]
        mask_files = [pair[1] for pair in matched_pairs]

        logger.info(f"Processing {len(matched_pairs)} matched image-mask pairs")

        # Prepare arguments for processing
        folder_name = os.path.basename(os.path.dirname(image_files[0])) if image_input else os.path.basename(os.getcwd())

        args_list = [(img_path, mask_input, self.params, self.output_path,
                     apply_preprocessing, min_roi_volume, folder_name, feats2out, roi_num, roi_selection_mode, feature_value_mode)
                    for img_path, mask_input in zip(image_files, mask_files)]

        # Process files
        use_parallel = (not enable_parallelism and len(image_files) > 1)
        if not use_parallel:
            logger.info("Using sequential processing")
            all_results = []
            for args in args_list:
                result = self.process_single_image_pair(args)
                if result is not None:
                    all_results.append(result)
        else:
            if num_workers == "auto":
                num_workers = min(mp.cpu_count(), len(image_files))

            logger.info(f"Using parallel processing with {num_workers} workers")

            # Set up multiprocessing-safe logging: queue + listener in main
            from pysera.config.settings import LOGGING_CONFIG
            original_memory_handler = self.memory_handler
            log_queue, listener = setup_multiprocessing_logging(original_memory_handler)
            listener.start()

            # IMPORTANT: remove unpicklable references from self before spawning workers
            self.memory_handler = None

            all_results = []
            try:
                # Initialize workers with queue-based logger
                with ProcessPoolExecutor(max_workers=int(num_workers), initializer=init_worker_logging,
                                         initargs=(log_queue, LOGGING_CONFIG['console_level'], LOGGING_CONFIG['console_format'])) as executor:
                    futures = [executor.submit(worker_process_single_image_pair, args) for args in args_list]

                    for future in futures:
                        try:
                            result = future.result()
                            if result is not None:
                                all_results.append(result)
                        except Exception as e:
                            logger.error(f"Error in parallel processing: {e}")
            finally:
                # Restore memory handler and stop listener
                self.memory_handler = original_memory_handler
                listener.stop()

        # Generate output file      toto
        output_path = self._generate_output_file(folder_name, apply_preprocessing, enable_parallelism, len(image_files))

        # log runtime and ram usage     toto
        # RR_logger.log('total_runtime')
        # RR_logger.save()


        # Save results      toto
        if all_results:
            saved_results = self._finalize_results(all_results, output_path)
        else:
            logger.error("No results returned from radiomics processing")
            saved_results = None

        # Save arguments and parameters     toto
        self._save_parameters(output_path, [args_list[0]])

        # Log errors (and warnings)
        logs = self.memory_handler.get_logs() if self.memory_handler else None
        log_logger(output_path, logs)
        return saved_results



    def _find_input_files(self, image_input: Optional[str], mask_input: Optional[str]) -> Tuple[List[str], List[str]]:
        """
        Find input image and mask files.

        Args:
            image_input: Path to image file or directory
            mask_input: Path to mask file or directory

        Returns:
            Tuple of (image_files, mask_files) lists
        """
        # Find image files
        image_files = self._find_image_files(image_input)
        if not image_files:
            return [], []

        # Find mask files
        mask_files = self._find_mask_files(mask_input)
        if not mask_files:
            return [], []

        return image_files, mask_files

    def _find_image_files(self, image_input: Optional[str]) -> List[str]:
        """Find image files from the given path or default directory, supporting multi-dcm and NRRD/NHDR."""
        if image_input:
            format_type = detect_file_format(image_input)
            if format_type == 'multi-dcm':
                # Return list of patient subfolders (each is a DICOM folder)
                from pysera.utils.file_utils import find_files_by_format
                return find_files_by_format(image_input, 'multi-dcm')
            else:
                return self._find_files_from_path(image_input, "image")
        else:
            return self._find_files_from_default_directory("image")

    def _find_mask_files(self, mask_input: Optional[str]) -> List[str]:
        """Find mask files from the given path or default directory, supporting NRRD/NHDR and multi-dcm."""
        if mask_input:
            format_type = detect_file_format(mask_input)
            if format_type == 'multi-dcm':
                from pysera.utils.file_utils import find_files_by_format
                return find_files_by_format(mask_input, 'multi-dcm')
            else:
                return self._find_files_from_path(mask_input, "mask")
        else:
            return self._find_files_from_default_directory("mask")

    def _find_files_from_path(self, path: str, file_type: str) -> List[str]:
        """Find files from a specific path, supporting NRRD/NHDR."""
        if os.path.isfile(path):
            files = [path]
            logger.info(f"Using single {file_type} file: {path}")
        elif os.path.isdir(path):
            file_format = detect_file_format(path)
            logger.info(f"{file_type.capitalize()} directory format detected: {file_format}")
            from pysera.utils.file_utils import find_files_by_format
            files = find_files_by_format(path, file_format)
            if not files:
                logger.error(f"No supported files found in {path}")
                return []
        else:
            logger.error(f"{file_type.capitalize()} input path does not exist: {path}")
            return []
        return files

    def _find_files_from_default_directory(self, file_type: str) -> List[str]:
        """Find files from default directory, supporting NRRD/NHDR and multi-dcm."""
        from ..config.settings import get_default_image_dir, get_default_mask_dir
        if file_type == "image":
            default_dir = get_default_image_dir()
        else:
            default_dir = get_default_mask_dir()
        if not os.path.isdir(default_dir):
            logger.error(f"Default {file_type} directory not found at {default_dir}")
            return []
        file_format = detect_file_format(default_dir)
        from pysera.utils.file_utils import find_files_by_format
        files = find_files_by_format(default_dir, file_format)
        if not files:
            logger.error(f"No supported files found in {default_dir}")
            return []
        return files

    def _finalize_results(self, all_results: List[pd.DataFrame], output_path: str) -> Dict[str, Any]:
        """
        Finalize and save processing results.

        Args:
            all_results: List of result DataFrames
            output_path: Path to save the final Excel file

        Returns:
            Dictionary with processing results
        """
        final_df = pd.concat(all_results, ignore_index=True)

        # Post-process the DataFrame
        final_df = self._post_process_dataframe(final_df)

        # Save results      toto
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            final_df.to_excel(writer, sheet_name="Sheet_1", index=False)

        # Log results summary
        self._log_results_summary(final_df, output_path)

        return {"out": ["Radiomics", final_df, output_path, output_path]}

    def _post_process_dataframe(self, final_df: pd.DataFrame) -> pd.DataFrame:
        """Post-process the final DataFrame."""
        # Renumber ROIs to match reference format

        # Fix column names to match reference format
        if 'Bin_Size_Used' in final_df.columns:
            final_df = final_df.rename(columns={'Bin_Size_Used': 'Bin Size'})
            logger.info("Renamed 'Bin_Size_Used' column to 'Bin Size' to match reference format")

        # Remove 'File' column if present (not in reference) - do this AFTER ROI renumbering
        if 'File' in final_df.columns:
            final_df = final_df.drop(columns=['File'])
            logger.info("Removed 'File' column to match reference format")

        return final_df

    def _renumber_rois(self, final_df: pd.DataFrame) -> pd.DataFrame:
        """Renumber ROIs to match reference format."""
        logger.info("Renumbering ROIs to match reference format (label_X_lesion_Y per image)")
        new_roi_names = []

        # Group by file and restart ROI numbering for each file
        for file_name in final_df['File'].unique():
            file_mask = final_df['File'] == file_name
            file_data = final_df[file_mask]

            roi_counter = 1
            for idx in file_data.index:
                new_roi_name = f"label {roi_counter} salam"
                new_roi_names.append((idx, new_roi_name))
                roi_counter += 1

            logger.info(f"File {file_name}: Renumbered {roi_counter - 1} ROIs")

        # Apply the new ROI names
        for idx, new_name in new_roi_names:
            final_df.loc[idx, 'ROI'] = new_name

        logger.info(f"Total ROIs renumbered: {len(new_roi_names)}")
        return final_df

    def _generate_output_file(self, folder_name: str,
                            apply_preprocessing: bool, enable_parallelism: bool,
                            num_images: int) -> str:
        """Generate output file path and save results."""
        # Generate output filename
        preprocessing_suffix = "_preprocessed" if apply_preprocessing else ""
        parallel_suffix = "_parallel" if not enable_parallelism and num_images > 1 else "_sequential"
        timestamp = datetime.now().strftime("%m-%d-%Y_%H%M%S")

        output_filename = OUTPUT_FILENAME_TEMPLATE.format(
            preprocessing_suffix=preprocessing_suffix,
            parallel_suffix=parallel_suffix,
            folder_name=folder_name,
            timestamp=timestamp
        )
        output_path = os.path.join(self.output_path, output_filename)

        return output_path

    def _save_parameters(self, excel_path, args_list) -> None:       # toto
        try:
            # Convert path to Path object for robust handling
            excel_path = Path(excel_path)
            # Create parent directories if they don't exist
            excel_path.parent.mkdir(parents=True, exist_ok=True)

            # Check if file exists
            if excel_path.is_file():
                write_to_excel(excel_path, args_list, "Sheet_2")

        except ValueError as e:
            logger.error(f"Value error: {e}")
            raise
        except PermissionError as e:
            logger.error(f"Permission error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise



    def _log_results_summary(self, final_df: pd.DataFrame, output_path: str) -> None:
        """Log results summary."""
        # Log feature information
        feature_columns = [col for col in final_df.columns if col not in ['ROI', 'Bin Size', 'PatientID']]
        logger.info(f"Total features extracted: {len(feature_columns)}")

        # Log feature quality metrics
        self._log_feature_quality_metrics(final_df)

        # Global processing summary
        self._log_global_summary(final_df, min_roi_volume=DEFAULT_MIN_ROI_VOLUME, total_processing_time=time.time() - time.time())

        logger.info(f"🎉 Optimized radiomics features extracted successfully!")
        logger.info(f"📁 Results saved to: {output_path}")

    def _log_feature_quality_metrics(self, final_df: pd.DataFrame) -> None:
        """Log feature quality metrics."""
        nan_count = final_df.isnull().sum()
        nan_features = nan_count[nan_count > 0]
        if len(nan_features) > 0:
            logger.warning(f"Features with missing values: {nan_features.to_dict()}")
        else:
            logger.info("No missing values found in extracted features!")

    def _log_global_summary(self, final_df: pd.DataFrame, min_roi_volume: int,
                            total_processing_time: float) -> None:
        """
        Log global processing summary.

        Args:
            final_df: Final results DataFrame
            min_roi_volume: Minimum ROI volume used
            total_processing_time: Total processing time
        """
        logger.info("╔══════════════════════════════════════════════════════════════════╗")
        logger.info("║                        GLOBAL SUMMARY                            ║")
        logger.info("╚══════════════════════════════════════════════════════════════════╝")

        total_rois_processed = len(final_df)
        roi_counts = final_df['ROI'].value_counts().sort_index()
        unique_roi_types = len(roi_counts)

        logger.info(f"📊 FINAL RESULTS:")
        logger.info(f"   ✓ Total ROIs processed: {total_rois_processed}")
        logger.info(
            f"   ✓ Unique ROI types: {unique_roi_types} (label_1_lesion_Y through label_{unique_roi_types}_lesion_Y)")

        # Estimate how many ROIs might have been skipped globally
        if min_roi_volume > 50:  # If using a high threshold
            logger.warning(f"⚠️  NOTICE: Using min_roi_volume = {min_roi_volume}")
            logger.warning(f"   💡 If you're missing expected ROIs, try a lower threshold like:")
            logger.warning(f"   💡   --min-roi-volume 50   (for small ROIs)")
            logger.warning(f"   💡   --min-roi-volume 10   (for very small ROIs)")
            logger.warning(f"   💡   --min-roi-volume 1    (to include all detected ROIs)")

        logger.info(f"⏱️  Total processing time: {total_processing_time:.2f} seconds")

    def _process_numpy_arrays(self, image_array: np.ndarray, mask_array: np.ndarray,
                            apply_preprocessing: bool, min_roi_volume: int,
                            num_workers: str, enable_parallelism: bool,
                            apply_intensity_preprocessing: bool = False,
                            use_disk: bool = False, memap_mode: str = 'r+') -> Optional[Dict[str, Any]]:
        """
        Process image and mask numpy arrays directly.

        Args:
            image_array: 3D numpy array containing image data
            mask_array: 3D numpy array containing mask data
            apply_preprocessing: Whether to apply preprocessing
            min_roi_volume: Minimum ROI volume threshold
            num_workers: Number of parallel workers
            enable_parallelism: Whether to enable parallel processing
            apply_intensity_preprocessing: Whether to apply intensity preprocessing (default: False for safety)
            use_disk: Whether to write arrays on disk (default: False)

        Returns:
            Dictionary with processing results or None if failed
        """
        # if use_disk:
        # Save mask array on disk
        mask_array_path = save_numpy_on_disk(mask_array)
        loaded_array = np.load(mask_array_path, mmap_mode=memap_mode)
        # else:
        #     loaded_array = mask_array
        # Clear RAM
        del mask_array

        logger.info(f"Processing numpy arrays - Image: {image_array.shape}, Mask: {loaded_array.shape}")

        # Create a unique identifier for this array pair
        image_id = "numpy_array_001"

        # Generate metadata for numpy arrays
        metadata = {
            'patient_id': 'numpy_patient',
            'image_id': image_id,
            'image_input': 'numpy_array_input',
            'mask_input': 'numpy_array_input',
            'image_shape': image_array.shape,
            'mask_shape': loaded_array.shape,
            'image_dtype': str(image_array.dtype),
            'mask_dtype': str(loaded_array.dtype),
            'processing_timestamp': datetime.now().isoformat()
        }

        # Process single image-mask pair
        try:
            # Apply preprocessing if requested
            if apply_preprocessing:
                logger.info(
                    f"[{image_id}] Starting preprocessing - image shape: {image_array.shape}, mask shape: {loaded_array.shape}")
                logger.info(f"[{image_id}] Min ROI volume type: {type(min_roi_volume)}, value: {min_roi_volume}")

                if apply_intensity_preprocessing:
                    processed_image = apply_intensity_preprocessing(
                        image_array,
                        image_id,
                        self.params
                    )
                    logger.info(f"[{image_id}] Intensity preprocessing completed")
                else:
                    logger.info(
                        f"[{image_id}] Skipping intensity preprocessing for numpy arrays (avoiding type conflicts)")
                    processed_image = image_array

                processed_mask_input = optimize_roi_preprocessing(
                    mask_array_path,
                    min_roi_volume
                )
                logger.info(f"[{image_id}] ROI preprocessing completed")
            else:
                processed_image = image_array
                processed_mask_input = mask_array_path

            # Process the single pair
            result_df = self._process_numpy_arrays_core(
                processed_image, processed_mask_input, image_id,
                min_roi_volume, apply_preprocessing
            )

            if result_df is not None and not result_df.empty:
                # Finalize results
                output_filename = f"radiomics_features_numpy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                output_path = os.path.join(self.output_path, output_filename)

                final_result = self._finalize_results([result_df], output_path)
                final_result['metadata'] = metadata

                return final_result
            else:
                logger.error("Failed to process numpy arrays")
                return None

        except Exception as e:
            logger.error(f"Error processing numpy arrays: {e}")
            return None

    def _process_numpy_arrays_core(self, image_array: np.ndarray, mask_array: str,
                                 image_id: str, min_roi_volume: int,
                                 apply_preprocessing: bool) -> Optional[pd.DataFrame]:
        """
        Core processing logic for numpy arrays.

        Args:
            image_array: 3D numpy array containing image data
            mask_array: 3D numpy array containing mask data
            image_id: Identifier for logging
            min_roi_volume: Minimum ROI volume threshold
            apply_preprocessing: Whether preprocessing was applied

        Returns:
            DataFrame with extracted features or None if failed
        """
        try:
            logger.info(f"[{image_id}] Core processing for numpy arrays")

            # Get ROI statistics
            roi_stats = get_roi_statistics(mask_array)
            logger.info(f"[{image_id}] Found {roi_stats['total_rois']} Unique ROI(s) in mask")

            if roi_stats['total_rois'] == 0:
                logger.warning(f"[{image_id}] No ROIs found in mask")
                return None

            # Create basic metadata for numpy arrays
            image_metadata = {
                'format': 'numpy',
                'spacing': [1.0, 1.0, 1.0],  # Default spacing
                'origin': [0.0, 0.0, 0.0],  # Default origin
                'direction': np.eye(3),  # Default direction matrix
                'shape': image_array.shape
            }

            mask_metadata = {
                'format': 'numpy',
                'spacing': [1.0, 1.0, 1.0],
                'origin': [0.0, 0.0, 0.0],
                'direction': np.eye(3),
                'shape': mask_array.shape
            }

            # Prepare parameters for SERA processing
            params_copy = self._prepare_sera_parameters(
                self.params, image_array, image_metadata,
                mask_array, mask_metadata,
                self.params['radiomics_Feats2out'],
                self.params['radiomics_ROI_num'],
                self.params['radiomics_ROI_selection_mode']
            )
            params_copy['feature_value_mode'] = self.params.get('feature_value_mode', 'REAL_VALUE')

            # Process with SERA
            result = self._run_sera_processing(image_id, params_copy, apply_preprocessing, min_roi_volume)

            if result is not None:
                df = result
                df.insert(0, 'PatientID', 'numpy_patient')

                # Final quality check
                self._perform_final_quality_check(image_id, df)

                return df
            else:
                logger.error(f"[{image_id}] FAILED: No result returned from SERA processing")
                return None

        except Exception as e:
            logger.error(f"[{image_id}] Error in core numpy processing: {e}")
            import traceback
            logger.error(f"[{image_id}] Traceback: {traceback.format_exc()}")
            return None

def worker_process_single_image_pair(args_tuple: Tuple) -> Optional[pd.DataFrame]:
    """Top-level worker to avoid pickling the main RadiomicsProcessor instance.
    It creates a fresh processor without a memory handler and processes one pair.
    """
    try:
        # Lazy import inside worker to avoid circular imports at module import time
        from pysera.processing.radiomics_processor import RadiomicsProcessor  # type: ignore
        # args_tuple layout matches producer's args_list
        output_folder = args_tuple[3]
        rp = RadiomicsProcessor(output_path=output_folder, memory_handler=None)
        # Use rp.process_single_image_pair directly with the provided args
        return rp.process_single_image_pair(args_tuple)
    except Exception as e:
        logger.error(f"Worker failed: {e}")
        return None