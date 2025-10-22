import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd

from ..config.settings import (
    DEFAULT_RADIOICS_PARAMS,
    OUTPUT_FILENAME_TEMPLATE, get_visera_pythoncode_path, get_default_output_path, DEFAULT_EXTRACTION_MODES,
    EXTRACTION_MODES, DEEP_LEARNING_MODELS, DEFAULT_DEEP_LEARNING_MODELS, DEFAULT_AGGREGATION_LESION,
    AGGREGATED_FEATURES_BLACK_LIST, AGGREGATED_FEATURES_WHITE_LIST
)
from ..data.data_loader import find_connected_rois, DataLoader
from ..preprocessing.data_preprocessing import apply_mask_roundup
from ..utils.helpers import (
    match_image_mask_pairs, ensure_directory_exists, find_input_files, remove_temp_file, convert_dimensions_bit_map,
    convert_categories_bit_map
)
from ..utils.log_record import log_to_excel, init_worker_logging, setup_multiprocessing_logging
from ..utils.mock_modules import setup_mock_modules
from ..utils.save_params import write_to_excel

logger = logging.getLogger("Dev_logger")


def sort_and_split_pairs(matched_pairs: List[Tuple[str, str]]) -> Tuple[List[str], List[str], str]:
    sorted_pairs = sorted(matched_pairs, key=lambda pair: os.path.basename(pair[1]))
    images, masks = zip(*sorted_pairs) if sorted_pairs else ([], [])
    return list(images), list(masks), os.path.basename(os.path.dirname(images[0])) if images else os.path.basename(
        os.getcwd())


def extract_voxel_size_info(data_image: list, sera_params: dict):
    """Extract voxel size information from image metadata."""
    metadata = data_image[1]

    # Case 1: Direct spacing
    if "spacing" in metadata:
        return metadata["spacing"]

    header = metadata.get("header", {})

    # Case 2: Space directions → compute norm of vectors
    if "space directions" in header:
        space_directions = header.get("space directions", [])
        spacing = [np.linalg.norm(vec) for vec in space_directions if vec is not None]
        if spacing:
            return tuple(spacing)

    # Case 3: Kinds
    if "kinds" in header:
        return header.get("kinds")

    # Case 4: Thicknesses
    if "thicknesses" in header:
        return header.get("thicknesses")

    # Case 5: Missing info → fall back
    if sera_params["radiomics_feature_value_mode"] == "APPROXIMATE_VALUE":
        voxel_size = (1, 1, 1)
        logger.warning(
            f"Couldn't find voxel size info. Using approximate value {voxel_size}."
        )
        return voxel_size

    logger.error("Couldn't find voxel size info in metadata.")
    return None


def import_sera_module() -> Any:
    # toggle to select OOP or procedural SERA implementation if needed
    sera_pythoncode_dir = get_visera_pythoncode_path()
    if sera_pythoncode_dir not in sys.path:
        sys.path.insert(0, sera_pythoncode_dir)

    module_path = 'pysera.engine.visera_oop.radiomics_features_extractor'
    try:
        import importlib
        mod = importlib.import_module(module_path)
        return mod.SERA_FE_main
    except ImportError as exc:
        logger.exception("Failed to import SERA module '%s': %s", module_path, exc)
        raise


def _create_results_dataframe(image_id: str, all_features: List[Dict]) -> Optional[pd.DataFrame]:
    """Create results DataFrame with 'PatientID' and 'ROI' as the first two columns."""
    try:
        if not all_features:
            return None

        # Convert list of dicts to DataFrame
        df = pd.DataFrame(all_features)

        # Ensure 'ROI' column exists in the data
        if 'ROI' not in df.columns:
            raise KeyError("Missing 'ROI' key in feature dictionaries.")

        # Insert PatientID column
        df.insert(0, 'PatientID', image_id)

        # Reorder columns: PatientID, ROI, then all other features
        columns = ['PatientID', 'ROI'] + [col for col in df.columns if col not in ['PatientID', 'ROI']]
        df = df.reindex(columns=columns)

        return df

    except Exception as e:
        logger.error(f"[{image_id}] Error creating DataFrame: {str(e)}")
        import traceback
        logger.error(f"[{image_id}] Traceback: {traceback.format_exc()}")
        return None


class RadiomicsProcessor:
    """A robust processor for batch radiomic feature extraction and parameter management."""

    def __init__(
            self,
            output_path: Optional[str] = None,
            memory_handler: Optional[Any] = None,
            temporary_files_path: Optional[str] = None,
            apply_preprocessing: Optional[bool] = None,
            min_roi_volume: Optional[int] = None,
            num_workers: Optional[Union[str, int]] = None,
            enable_parallelism: Optional[bool] = None,
            categories: Optional[str] = None,
            dimensions: Optional[str] = None,
            bin_size: Optional[int] = None,
            roi_num: Optional[int] = None,
            roi_selection_mode: Optional[str] = None,
            feature_value_mode: Optional[str] = None,
            report: Optional[str] = None,
            callback_fn: Optional[Callable[..., None]] = None,
            extraction_mode: str = DEFAULT_EXTRACTION_MODES,
            deep_learning_model: str = DEFAULT_DEEP_LEARNING_MODELS,
            aggregation_lesion: bool = DEFAULT_AGGREGATION_LESION,
            IBSI_based_parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the RadiomicsProcessor with configurable output path and optional memory logging.
        """
        # ------------------------------------------------------------------
        # Core setup
        # ------------------------------------------------------------------
        self.output_path = Path(output_path or get_default_output_path())
        ensure_directory_exists(self.output_path)
        self.memory_handler = memory_handler
        self.callback_fn = None
        self.aggregation_lesion = aggregation_lesion

        # initiate extraction mode
        if extraction_mode is not None and extraction_mode in EXTRACTION_MODES:
            self.extraction_mode = extraction_mode

        # deep model selection
        self.deep_learning_extractor = None
        if (deep_learning_model is not None and deep_learning_model in DEEP_LEARNING_MODELS and self.extraction_mode ==
                "deep_feature"):
            from ..engine.deep_features.radiomics_df_extractor import deep_learning_feature_extractor
            self.deep_learning_extractor = deep_learning_feature_extractor(deep_learning_model)

        # Load radiomics engine
        setup_mock_modules()
        self.SERA_FE_main = import_sera_module()

        # Initialize default parameters
        self.params: Dict[str, Any] = DEFAULT_RADIOICS_PARAMS.copy()
        self.params["radiomics_destination_folder"] = str(self.output_path)

        # set callback function
        if callback_fn is not None:
            self.callback_fn = callback_fn

        # Apply provided optional parameters first
        if IBSI_based_parameters:
            for key, value in IBSI_based_parameters.items():
                if key in DEFAULT_RADIOICS_PARAMS and key != "radiomics_destination_folder":
                    self.params[key] = value

        # Explicit parameter overrides
        param_updates = {
            "radiomics_BinSize": bin_size,
            "radiomics_categories": categories,
            "radiomics_dimensions": dimensions,
            "radiomics_roi_num": roi_num,
            "radiomics_roi_selection_mode": roi_selection_mode,
            "radiomics_feature_value_mode": feature_value_mode,
            "radiomics_report": report,
            "radiomics_temporary_files_path": temporary_files_path,
            "radiomics_min_roi_volume": min_roi_volume,
            "radiomics_num_workers": num_workers,
            "radiomics_enable_parallelism": enable_parallelism,
            "radiomics_apply_preprocessing": apply_preprocessing,
        }

        for key, value in param_updates.items():
            if value is not None:
                self.params[key] = value

    # -------------------------------------------------------------------------
    # Save Parameter Configuration
    # -------------------------------------------------------------------------
    def save_parameters(self, excel_path: Union[str, Path]) -> None:
        """
        Save current radiomics configuration to an Excel file.

        Parameters
        ----------
        excel_path : Union[str, Path]
            Destination Excel file path.
        """
        excel_path = Path(excel_path)
        excel_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            write_to_excel(excel_path, self.params)
        except (ValueError, PermissionError) as e:
            logger.error("Error saving parameters: %s", e)
            raise
        except Exception as e:
            logger.exception("Unexpected error while saving parameters.")
            raise

    def _generate_output_file(self, folder_name: str, num_images: int) -> str:
        preprocessing_suffix = "_preprocessed" if self.params["radiomics_apply_preprocessing"] else ""
        parallel_suffix = "_parallel" if self.params[
                                             "radiomics_enable_parallelism"] and num_images > 1 else "_sequential"
        timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

        output_filename = OUTPUT_FILENAME_TEMPLATE.format(
            preprocessing_suffix=preprocessing_suffix,
            parallel_suffix=parallel_suffix,
            folder_name=folder_name,
            timestamp=timestamp
        )
        output_file_path = os.path.join(self.output_path, output_filename)

        return output_file_path

    # -------------------------------------------------------------------------
    # Main Processing API
    # -------------------------------------------------------------------------
    def process_batch(
            self,
            image_input: Optional[Union[str, np.ndarray]] = None,
            mask_input: Optional[Union[str, np.ndarray]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Process a batch of radiomics extractions, either from NumPy arrays or file inputs.
        """
        # Direct array processing mode
        if isinstance(image_input, np.ndarray) and isinstance(mask_input, np.ndarray):
            return self._process_in_memory_pair(image_input, mask_input)

        # File-based processing mode
        image_files, mask_files = find_input_files(image_input, mask_input)
        if not image_files or not mask_files:
            logger.error("No image or mask files found for inputs: %s, %s", image_input, mask_input)
            return None

        matched_pairs = match_image_mask_pairs(image_files, mask_files)
        if not matched_pairs:
            logger.error("No matching image-mask pairs found.")
            return None

        image_files, mask_files, folder_name = sort_and_split_pairs(matched_pairs)
        logger.info("Processing %d image-mask pairs", len(image_files))

        # Process all pairs
        results = self._process_pairs(image_files, mask_files)

        if not results:
            logger.warning("No radiomics results generated.")
            return None

        # Generate output file and finalize results
        output_path = self._generate_output_file(folder_name, len(image_files))
        saved_results = self._finalize_results(results, output_path)

        # Save parameters and logs
        self.save_parameters(output_path)
        if self.memory_handler:
            logs = self.memory_handler.get_logs()
            log_to_excel(output_path, logs)

        return saved_results

    # ---------- Pair processing helpers ----------
    def _process_pairs(self, image_files: list[str], mask_files: list[str]) -> List[pd.DataFrame]:
        """Process all image-mask pairs, either sequentially or in parallel."""

        if not self.params["radiomics_enable_parallelism"] or len(image_files) == 1:
            logger.info("Processing sequentially")
            return self._process_sequential(image_files, mask_files)

        logger.info("Processing in parallel")
        return self._process_parallel(image_files, mask_files)

    def _process_sequential(self, image_files: list[str], mask_files: list[str]) -> List[pd.DataFrame]:
        """Process image-mask pairs sequentially."""
        return [
            df for image_path, mask_path in zip(image_files, mask_files)
            if (df := self.process_file_pair(image_path=image_path, mask_path=mask_path)) is not None
        ]

    def _process_parallel(self, image_files: list[str], mask_files: list[str]) -> List[pd.DataFrame]:
        """Process image-mask pairs in parallel using multiple workers."""

        workers = min(os.cpu_count() or 1, len(image_files)) if self.params["radiomics_num_workers"] == "auto" else int(
            self.params["radiomics_num_workers"])
        logger.info("Processing in parallel with %d workers", workers)

        original_handler = self.memory_handler
        log_queue, listener = setup_multiprocessing_logging(original_handler, self.params['radiomics_report'])

        if listener is not None:
            listener.start()
        self.memory_handler = None  # Prevent pickling issues

        try:
            results = self._run_parallel_workers(image_files, mask_files, workers, log_queue)
        finally:
            self._cleanup_parallel(original_handler, listener)

        return results

    def _run_parallel_workers(self, image_files: list[str], mask_files: list[str], workers: int, log_queue) -> List[
        pd.DataFrame]:
        """Streamlined parallel processing with minimal memory overhead."""
        results = []

        with ProcessPoolExecutor(
                max_workers=workers,
                initializer=init_worker_logging,
                initargs=(log_queue, self.params["radiomics_report"],),
        ) as executor:
            for result in executor.map(self.worker_process_single_image_pair, image_files, mask_files, chunksize=1):
                if result is not None:
                    results.append(result)

        return results

    def worker_process_single_image_pair(self, image_path: str, mask_path: str) -> \
            Optional[pd.DataFrame]:
        try:
            output_folder = self.params["radiomics_destination_folder"]

            rp = RadiomicsProcessor(
                output_path=output_folder,
                memory_handler=None,
                aggregation_lesion=self.aggregation_lesion,
                extraction_mode=self.extraction_mode
            )

            rp.deep_learning_extractor = self.deep_learning_extractor
            rp.params = self.params
            if self.callback_fn is not None:
                rp.callback_fn = self.callback_fn

            return rp.process_file_pair(image_path, mask_path)

        except Exception as e:
            logger.error(f"Worker failed: {e}")
            return None

    def _cleanup_parallel(self, original_handler, listener):
        """Restore processor state and stop logging listener."""
        self.memory_handler = original_handler
        if listener is not None:
            listener.stop()

    def prepare_sera_parameters(self, image_array: np.ndarray, image_metadata: Dict,
                                mask_array: np.ndarray | Dict, mask_metadata: Dict) -> Dict[str, Any]:
        """Prepare parameters for SERA processing."""
        params_copy = self.params.copy()
        params_copy['data_image'] = [image_array, image_metadata, image_metadata['format'].title(), None]
        params_copy['data_mask'] = [mask_array, mask_metadata, mask_metadata['format'].title(), None]
        return params_copy

    def process_file_pair(self, image_path: str, mask_path: str) -> Optional[pd.DataFrame]:
        """Load file-based image+mask, prepare SERA params and extract features."""

        image_id = os.path.basename(image_path)

        try:
            loader = DataLoader(
                roi_num=self.params["radiomics_roi_num"],
                roi_selection_mode=self.params["radiomics_roi_selection_mode"],
                min_roi_volume=self.params["radiomics_min_roi_volume"],
                temporary_files_path=self.params["radiomics_temporary_files_path"],
                apply_preprocessing=self.params["radiomics_apply_preprocessing"],
            )
            image_array, image_metadata, mask, mask_metadata = loader.convert(image_input=image_path,
                                                                              mask_input=mask_path)
            # clean loader from memory
            del loader

            if image_array is None or mask is None:
                logger.error("[%s] Failed to load image or mask", image_id)
                return None

            params_copy = self.prepare_sera_parameters(image_array, image_metadata, mask, mask_metadata)
            return self._prepare_and_extract(image_id, params_copy)

        except Exception as e:
            logger.exception(f"[%s] Error processing file pair: {e}", image_id)
            return None

    def _process_in_memory_pair(self, image_array: np.ndarray, mask_array: np.ndarray) -> Optional[Dict[str, Any]]:
        """Process a single image/mask provided as numpy arrays. Returns dict with results+metadata."""

        logger.info("Processing in-memory numpy arrays")
        if image_array.shape != mask_array.shape:
            logger.error("Image and mask shapes do not match: %s vs %s", image_array.shape, mask_array.shape)
            return None

        processed_image = image_array
        if self.params["radiomics_apply_preprocessing"]:
            # processed_image = apply_image_intensity_preprocessing(image_array, mask_array)
            mask_array = apply_mask_roundup(mask_array)

        loader = DataLoader(
            roi_num=self.params["radiomics_roi_num"],
            roi_selection_mode=self.params["radiomics_roi_selection_mode"],
            min_roi_volume=self.params["radiomics_min_roi_volume"],
            temporary_files_path=self.params["radiomics_temporary_files_path"],
            apply_preprocessing=self.params["radiomics_apply_preprocessing"],
        )
        image_array, _, mask_array, _ = loader.convert(image_input=image_array, mask_input=mask_array)
        image_id = f"numpy_array_{datetime.now().strftime('%Y_%m_%d_%H%M%S')}"

        image_meta = self._make_default_array_metadata(processed_image)
        mask_meta = self._make_default_array_metadata(mask_array)

        params_copy = self.prepare_sera_parameters(processed_image, image_meta, mask_array, mask_meta)

        df = self._prepare_and_extract(image_id, params_copy)
        if df is None:
            return None

        output_filename = f"radiomics_features_numpy_{datetime.now().strftime('%Y_%m_%d_%H%M%S')}.xlsx"
        output_path = os.path.join(self.output_path, output_filename)
        final = self._finalize_results([df], output_path)
        self.save_parameters(output_path)
        if self.memory_handler:
            logs = self.memory_handler.get_logs()
            log_to_excel(output_path, logs)

        final['metadata'] = {
            'image_shape': image_array.shape,
            'mask_shape': mask_array.shape,
            'processed_at': datetime.now().isoformat(),
        }

        return final

    # ---------- Core preparation & extraction ----------
    def _prepare_and_extract(self, image_id: str, params_copy: dict) -> Optional[pd.DataFrame]:
        """Shared logic: compute voxel info, call ROI extraction, return DataFrame or None."""
        try:
            params_copy['radiomics_VoxelSizeInfo'] = extract_voxel_size_info(params_copy['data_image'], params_copy)
            logger.info("[%s] Voxel spacing: %s", image_id, params_copy['radiomics_VoxelSizeInfo'])

            df = self._extract_features_for_rois(image_id, params_copy)
            if df is None:
                logger.error("[%s] No features were extracted", image_id)
                return None

            return df

        except Exception as e:
            logger.exception(f"[%s] Error preparing/extracting SERA features: {e}", image_id)
            return None

    # ---------- ROI extraction flows ----------
    def _extract_features_for_rois(self, image_id: str, sera_params: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Dispatch to the correct ROI iteration strategy (RTSTRUCT dict or labeled mask)."""
        mask = sera_params['data_mask'][0]

        if isinstance(mask, dict):
            features = self._extract_from_rtstruct_rois(image_id, mask, sera_params)
        else:
            features = self._extract_from_labeled_mask(image_id, mask, sera_params)

        if not features:
            logger.error("[%s] No features collected from any ROI", image_id)
            return None

        return _create_results_dataframe(image_id, features)

    def _extract_from_rtstruct_rois(self, image_id: str, rois: Dict[str, str],
                                    sera_params: Dict[str, Any]) -> List:
        """Load each ROI file (numpy .npy path), extract features, and remove temp file."""
        if not rois:
            logger.error("[%s] No ROIs found in RTSTRUCT", image_id)
            return []

        all_features: List = []

        for roi_name, roi_path in rois.items():
            mask_arr = None
            try:
                mask_arr = np.load(roi_path, mmap_mode='r')
                output_features = self._extract_features_single_roi(image_id, mask_arr, roi_name, sera_params)
                if output_features is not None:
                    output_features["ROI"] = roi_name
                    all_features.append(output_features)
            except Exception as e:
                logger.exception(f"[%s] Error loading/extracting ROI '%s' from %s: %s", image_id, roi_name, roi_path, e)

            finally:
                if mask_arr is not None:
                    try:
                        del mask_arr
                    except Exception:
                        pass

                import gc
                gc.collect()

                try:
                    remove_temp_file(roi_path)
                except Exception as e:
                    logger.warning(f"[%s] Could not remove temp file %s: %s", image_id, roi_path, e)

        return all_features

    def _extract_from_labeled_mask(self, image_id: str, mask_array: np.ndarray,
                                   sera_params: Dict[str, Any]) -> List:
        """Find connected components for each label and extract features per lesion."""
        labels = [lbl for lbl in np.unique(mask_array) if lbl != 0]
        if not labels:
            logger.error("[%s] No labeled ROIs found in mask", image_id)
            return []

        all_features: List = []

        for label_value in labels:
            labeled_mask, roi_ids = find_connected_rois(mask_array, label_value)
            for lesion_id in roi_ids:
                roi_arr = (labeled_mask == lesion_id).astype(np.float32)
                roi_name = f"label_{label_value}_lesion_{lesion_id}"
                output_features = self._extract_features_single_roi(image_id, roi_arr, roi_name, sera_params)
                if output_features is not None:
                    output_features["ROI"] = roi_name
                    all_features.append(output_features)

        return all_features

    def _extract_features_single_roi(self, image_id: str, mask_array: np.ndarray,
                                     roi_name: str, sera_params: Dict[str, Any]) -> Optional[dict]:
        """Wrap call to SERA and convert result to feature vector list or None."""
        try:
            self.invoke_callback_fn("START", image_id, roi_name)
            logger.info("[%s] Extracting ROI '%s'", image_id, roi_name)

            extraction_result = self._call_sera_feature_extraction(mask_array, sera_params, image_id, roi_name)
            feature_dictionary = None

            if self.extraction_mode != "deep_feature":
                for v in extraction_result.values():
                    if isinstance(v, list) and v and "value" in v[0]:
                        feature_dictionary = v[0]["value"]
                        break
            else:
                feature_dictionary = extraction_result

            del extraction_result

            if not (isinstance(feature_dictionary, dict) and feature_dictionary):
                logger.warning("[%s] SERA returned no data for ROI '%s'", image_id, roi_name)
                return None
            else:
                return feature_dictionary

        except Exception as e:
            logger.exception(f"[%s] Error extracting ROI '%s': {e}", image_id, roi_name)
            return None
        finally:
            self.invoke_callback_fn("END", image_id, roi_name)

    def _call_sera_feature_extraction(self, mask_array: np.ndarray, sera_params: Dict[str, Any], image_name: str,
                                      roi_name: str) -> Any:
        """Persist mask temporarily and call SERA_FE_main with flattened args."""
        try:
            if self.extraction_mode == "deep_feature":
                return self.deep_learning_extractor.process_single_image_pair(image_array=sera_params['data_image'][0],
                                                                              mask_array=mask_array, roi_name=roi_name,
                                                                              image_name=image_name)
            else:
                return self.SERA_FE_main(
                    image_array=sera_params['data_image'][0],
                    roi_mask_array=mask_array,
                    voxel_dimensions=sera_params['radiomics_VoxelSizeInfo'],
                    bin_sizes_input=sera_params['radiomics_BinSize'],
                    data_type_str=sera_params['radiomics_DataType'],
                    isotropic_voxel_size_3d=sera_params['radiomics_isotVoxSize'],
                    isotropic_voxel_size_2d=sera_params['radiomics_isotVoxSize2D'],
                    discretization_method=sera_params['radiomics_DiscType'],
                    quantization_method=sera_params['radiomics_qntz'],
                    voxel_interpolation_method=sera_params['radiomics_VoxInterp'],
                    roi_interpolation_method=sera_params['radiomics_ROIInterp'],
                    enable_scaling=sera_params['radiomics_isScale'],
                    enable_gl_rounding=sera_params['radiomics_isGLround'],
                    enable_resegmentation=sera_params['radiomics_isReSegRng'],
                    remove_outliers=sera_params['radiomics_isOutliers'],
                    quantize_statistics=sera_params['radiomics_isQuntzStat'],
                    use_isotropic_2d=sera_params['radiomics_isIsot2D'],
                    resegmentation_interval=[sera_params['radiomics_ReSegIntrvl01'],
                                             sera_params['radiomics_ReSegIntrvl02']],
                    roi_partial_volume_fraction=sera_params['radiomics_ROI_PV'],
                    extractor_mask=convert_categories_bit_map(sera_params['radiomics_categories']),
                    feature_dimensions_mask=convert_dimensions_bit_map(sera_params['radiomics_dimensions']),

                    feature_value_mode_str=sera_params['radiomics_feature_value_mode'],
                    image_name=image_name,
                    roi_name=roi_name,
                    ivh_type=sera_params['radiomics_IVH_Type'],
                    ivh_disc_cont=sera_params['radiomics_IVH_DiscCont'],
                    ivh_bin_size=sera_params['radiomics_IVH_binSize'],
                )
        except Exception as exc:
            logger.error(f"error in extracting ROI: {exc}")

    def _finalize_results(self, all_results: List[pd.DataFrame], output_path: str) -> Dict[str, Any]:

        # === Concatenate all results ===
        final_df = pd.concat(all_results, ignore_index=True)

        # === Early exit if no aggregation is required ===
        if not getattr(self, "aggregation_lesion", False):
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                final_df.to_excel(writer, sheet_name="Radiomics_Features", index=False)
            return {"out": ["Radiomics", final_df, output_path, output_path]}

        # === Grouping logic ===
        mode = self.params["radiomics_roi_selection_mode"]

        def extract_roi_group(roi_value: str) -> str:
            """Extracts the group identifier from ROI string."""
            if pd.isna(roi_value):
                return "unknown"
            return str(roi_value).split("_lesion_")[0]

        if mode == "per_Img":
            final_df["GroupKey"] = final_df["PatientID"]
        else:  # "per_region"
            final_df["GroupKey"] = (
                    final_df["PatientID"].astype(str) + "_" + final_df["ROI"].apply(extract_roi_group)
            )

        # === Aggregation logic ===
        def aggregate_group(df_group: pd.DataFrame) -> pd.Series:
            # Always keep PatientID (use the first one)
            agg_result = {"PatientID": df_group["PatientID"].iloc[0]}

            # If per_region mode: add the label name (ROI)
            if mode == "per_region":
                roi_name = extract_roi_group(df_group["ROI"].iloc[0])
                agg_result["ROI"] = roi_name

            if self.extraction_mode == "deep_feature":
                # Simple mean across all features
                for col in df_group.columns:
                    if col in ["PatientID", "ROI", "GroupKey"]:
                        continue
                    agg_result[col] = df_group[col].mean(skipna=True)
                return pd.Series(agg_result)

            # Radiomics feature aggregation
            volume_col = "morph_volume_mesh"
            volumes = (
                df_group[volume_col].fillna(0).to_numpy()
                if volume_col in df_group else np.ones(len(df_group))
            )

            for col in df_group.columns:
                if col in ["PatientID", "ROI", "GroupKey"]:
                    continue

                values = df_group[col].to_numpy(dtype=float)
                if np.all(np.isnan(values)):
                    agg_result[col] = np.nan
                    continue

                if col in AGGREGATED_FEATURES_BLACK_LIST:
                    first_valid_idx = np.where(~np.isnan(values))[0]
                    agg_result[col] = (
                        values[first_valid_idx[0]] if len(first_valid_idx) > 0 else np.nan
                    )
                elif col in AGGREGATED_FEATURES_WHITE_LIST:
                    valid_mask = ~np.isnan(values)
                    if valid_mask.any():
                        weights = volumes[valid_mask]
                        vals = values[valid_mask]
                        agg_result[col] = np.average(vals, weights=weights)
                    else:
                        agg_result[col] = np.nan
                else:
                    agg_result[col] = np.nansum(values)

            return pd.Series(agg_result)

        # === Apply aggregation ===
        aggregated_df = (
            final_df.groupby("GroupKey", group_keys=False)
            .apply(aggregate_group, include_groups=False)
            .reset_index(drop=True)
        )

        # === Ensure ROI column is right after PatientID ===
        if mode == "per_region" and "ROI" in aggregated_df.columns:
            cols = aggregated_df.columns.tolist()
            cols.insert(cols.index("PatientID") + 1, cols.pop(cols.index("ROI")))
            aggregated_df = aggregated_df[cols]

        # === Save results ===
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            aggregated_df.to_excel(writer, sheet_name="Radiomics_Features", index=False)

        return {"out": ["Radiomics", aggregated_df, output_path, output_path]}

    def invoke_callback_fn(
            self,
            flag: str = "START",
            image_id: Optional[str] = None,
            roi_name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """
        Parameters
        ----------
        flag : str, default="START"
            Indicates the current stage of processing. Common values include:
            - "START": Called at the beginning of a process.
            - "END": Called at the completion of a process.
            - "ERROR": Called when an exception occurs.
            - "UPDATE": Called for progress updates or intermediate results.

        image_id : str, optional
            Identifier of the image being processed.

        roi_name : str, optional
            Name of the current ROI being processed.

        **kwargs : Any
            Additional keyword arguments to pass to the callback function,
            such as progress, message, features, or patient metadata.
        """
        callback_fn = getattr(self, "callback_fn", None)
        if not callable(callback_fn):
            return

        try:
            callback_fn(
                flag=flag,
                image_id=image_id,
                roi_name=roi_name,
                **kwargs
            )
        except Exception as e:
            # Do not interrupt main execution if callback fails
            logger.warning(f"Callback function raised an exception: {e}")

    # ---------- Utility ----------
    @staticmethod
    def _make_default_array_metadata(arr: np.ndarray) -> Dict[str, Any]:
        """Construct default metadata for in-memory numpy arrays."""
        return {
            'type_file': arr.dtype,
            'shape_file': arr.shape,
            'format': 'npy',
            'origin': (0.0, 0.0, 0.0),
            'spacing': (1.0, 1.0, 1.0),
            'direction': (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        }
