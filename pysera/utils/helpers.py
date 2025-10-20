"""
File Handling and Utility Functions for Radiomics Processing
============================================================

This module provides comprehensive file management, format detection,
image–mask pairing, and logging utilities for the radiomics processing pipeline.
It serves as the core backbone for loading, matching, and organizing
medical imaging data (DICOM, NIfTI, NRRD, NumPy, or OpenCV-compatible formats)
in a robust and scalable way.

--------------------------------------------------------------------
📁 1. File Format Detection
--------------------------------------------------------------------
Functions for identifying the format of files or directories.
They detect supported medical image formats using extensions,
content-based heuristics, or library read tests.

Includes:
    - _normalize_extension
    - _detect_single_file_format
    - _detect_directory_format
    - _detect_consistent_format
    - _detect_file_format_by_content
    - detect_file_format

--------------------------------------------------------------------
🔍 2. File Search and Retrieval
--------------------------------------------------------------------
Utilities for locating files across directories and subdirectories.
They recursively find supported imaging and segmentation files
based on detected formats or default project configurations.

Includes:
    - find_files_by_format
    - _find_nifti_files
    - _find_dicom_files
    - _find_nrrd_files
    - _find_npy_files
    - _find_opencv_supported_images
    - files_catching
    - find_files_from_path
    - find_files_from_default_directory
    - find_input_files

--------------------------------------------------------------------
🧠 3. Image–Mask Matching
--------------------------------------------------------------------
Functions to match images with their corresponding segmentation masks
based on shared basenames, patient IDs, or directory structures.
Supports multi-format and mixed-format pairing strategies.

Includes:
    - _extract_patient_id
    - _extract_base_names
    - _create_matched_pairs
    - _match_nifti_pairs
    - _match_nrrd_pairs
    - _match_dicom_pairs
    - _match_opencv_supported_pairs
    - _match_mixed_format_pairs
    - match_image_mask_pairs

--------------------------------------------------------------------
🗂 4. Temporary File and Directory Management
--------------------------------------------------------------------
Handles creation, maintenance, and cleanup of process-specific
temporary directories and files. Ensures safe concurrent file writes.

Includes:
    - get_process_safe_temp_dir
    - create_tmp_dir
    - _cleanup_temp_dir
    - create_process_safe_temp_file
    - save_numpy_on_disk
    - remove_temp_file

--------------------------------------------------------------------
🧩 5. General Utilities
--------------------------------------------------------------------
Miscellaneous supporting functions used across the pipeline.

Includes:
    - ensure_directory_exists
"""

import glob
import logging
import os
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union

import cv2
import nrrd
import numpy as np
import pydicom
from nibabel.loadsave import load as nib_load

from pysera.config.settings import DEFAULT_RADIOICS_PARAMS

# --------------------------------------------------------------------
# 📁 0. Constants
# --------------------------------------------------------------------

logger = logging.getLogger("Dev_logger")

NIFTI_EXTENSIONS = (".nii", ".nii.gz")
DICOM_EXTENSIONS = (".dcm", ".dicom")
NRRD_EXTENSIONS = (".nrrd", ".nhdr")
NPY_EXTENSIONS = (".npy",)
OPENCV_EXTENSIONS = (".png", ".jpg", ".jpeg")

# Combined extension-to-format map
EXTENSION_MAP = {
    **{ext: "nifti" for ext in NIFTI_EXTENSIONS},
    **{ext: "dicom" for ext in DICOM_EXTENSIONS},
    **{ext: "nrrd" for ext in NRRD_EXTENSIONS},
    **{ext: "npy" for ext in NPY_EXTENSIONS},
    **{ext: "OpenCV_supported" for ext in OPENCV_EXTENSIONS},
}


# --------------------------------------------------------------------
# 📁 1. File Format Detection
# --------------------------------------------------------------------

def _normalize_extension(file_path: str) -> str:
    """Normalize file extension (e.g., handle .nii.gz correctly)."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".gz" and file_path.lower().endswith(".nii.gz"):
        return ".nii.gz"
    return ext


def _detect_single_file_format(file_path: str) -> str:
    """Detect format of a single file using extension and fallback content check."""
    logger.info(f"Checking file: {file_path}")

    ext = _normalize_extension(file_path)
    if ext in EXTENSION_MAP:
        fmt = EXTENSION_MAP[ext]
        logger.info(f"File {file_path} detected as {fmt.upper()} by extension.")
        return fmt

    if cv2.haveImageReader(file_path):
        logger.info(f"File {file_path} detected as OpenCV-supported image.")
        return "OpenCV_supported"

    return _detect_file_format_by_content(file_path)


def _detect_directory_format(directory: str) -> str:
    """Detect format of a directory containing files or subdirectories."""
    entries = [
        os.path.join(directory, e)
        for e in os.listdir(directory)
        if not e.startswith(".")
    ]
    files = [e for e in entries if os.path.isfile(e)]
    subdirs = [e for e in entries if os.path.isdir(e)]

    if files and not subdirs:
        return _detect_consistent_format(files)
    if subdirs and not files:
        return _detect_consistent_format(
            [detect_file_format(d) for d in subdirs]
        )
    if files and subdirs:
        return "mixed"
    return "unknown"


def _detect_consistent_format(items) -> str:
    """Check if a list of files/subformats is consistent or mixed."""
    detected = {_detect_single_file_format(f) if os.path.isfile(f) else f for f in items}
    if len(detected) == 1:
        return detected.pop()
    if len(detected) > 1:
        return "mixed"
    return "unknown"


def _detect_file_format_by_content(file_path: str) -> str:
    """Fallback detection by trying to read the file with different libraries."""
    try:
        pydicom.dcmread(file_path)
        logger.info(f"File {file_path} detected as DICOM by content.")
        return "dicom"
    except Exception:
        pass

    try:
        nib_load(file_path)
        logger.info(f"File {file_path} detected as NIfTI by content.")
        return "nifti"
    except Exception:
        pass

    try:
        nrrd.read(file_path)
        logger.info(f"File {file_path} detected as NRRD by content.")
        return "nrrd"
    except Exception:
        pass

    try:
        np.load(file_path)
        logger.info(f"File {file_path} detected as NumPy by content.")
        return "npy"
    except Exception:
        pass

    if cv2.haveImageReader(file_path):
        logger.info(f"File {file_path} is supported by OpenCV (content check).")
        return "OpenCV_supported"

    logger.warning(f"File {file_path} format could not be detected.")
    return "unknown"


def detect_file_format(input_path: str) -> str:
    input_path = os.path.normpath(input_path)

    if os.path.isfile(input_path):
        return _detect_single_file_format(input_path)

    if os.path.isdir(input_path):
        return _detect_directory_format(input_path)

    logger.warning(f"Input path {input_path} is neither file nor directory.")
    return "unknown"


# --------------------------------------------------------------------
# 🔍 2. File Search and Retrieval
# --------------------------------------------------------------------

def find_files_by_format(directory: str, format_type: str) -> List[str]:
    if format_type == 'nrrd':
        return _find_nrrd_files(directory)
    elif format_type == 'nifti':
        return _find_nifti_files(directory)
    elif format_type == 'dicom':
        return _find_dicom_files(directory)
    elif format_type == 'npy':
        return _find_npy_files(directory)
    elif format_type == 'OpenCV_supported':
        return _find_opencv_supported_images(directory)
    else:
        return []


def _find_nifti_files(directory: str) -> List[str]:
    """Find NIfTI files in directory and all subdirectories."""
    nii_files = glob.glob(os.path.join(directory, "**", "*.nii.gz"), recursive=True)
    nii_files += glob.glob(os.path.join(directory, "**", "*.nii"), recursive=True)
    return nii_files


def _find_dicom_files(directory: str) -> List[str]:
    if os.path.isdir(directory):
        entries = [os.path.join(directory, e) for e in os.listdir(directory) if not e.startswith('.')]
        subdirs = [e for e in entries if os.path.isdir(e)]
        files = [e for e in entries if os.path.isfile(e)]
        dcm_files = [f for f in files if f.lower().endswith(DICOM_EXTENSIONS)]
        # Case 1: sub_folders (multi-patient)
        if subdirs and not dcm_files:
            return subdirs
        # Case 2: only DICOM files (e.g., segmentation masks)
        elif dcm_files and not subdirs:
            return dcm_files
        # Case 3: DICOM series (image: a folder of DICOM slices)
        elif dcm_files and not subdirs:
            return [directory]
        # Mixed: both sub_dirs and files (rare, treat as sub_dirs)
        elif subdirs and dcm_files:
            return subdirs
        else:
            return []
    return []


def _find_nrrd_files(directory: str) -> List[str]:
    """Find NRRD files in directory and all subdirectories."""
    nrrd_files = glob.glob(os.path.join(directory, "**", "*.nrrd"), recursive=True)
    nrrd_files += glob.glob(os.path.join(directory, "**", "*.nhdr"), recursive=True)
    return nrrd_files


def _find_npy_files(directory: str) -> List[str]:
    """Find NumPy files in directory and all subdirectories."""
    npy_files = glob.glob(os.path.join(directory, "**", "*.npy"), recursive=True)
    return npy_files


def _find_opencv_supported_images(directory: str) -> List[str]:
    """Find all image files in directory (and subdirectories) that OpenCV can read."""
    supported_files = []

    # Walk through all subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            # Check if OpenCV can read this file
            if cv2.haveImageReader(filepath):
                supported_files.append(filepath)

    return supported_files


def files_catching(file_input: Optional[str], file_type: str) -> List[str]:
    """Find image/mask files from the given path or default directory, supporting multi-dcm and NRRD/NHDR."""
    if file_input:
        format_type = detect_file_format(file_input)
        if format_type == 'multi-dcm':
            return find_files_by_format(file_input, format_type)
        else:
            return find_files_from_path(file_input, file_type)
    else:
        return find_files_from_default_directory(file_type)


def find_files_from_path(path: str, file_type: str) -> List[str]:
    """Find files from a specific path, supporting NRRD/NHDR."""
    if os.path.isfile(path):
        files = [path]
        logger.info(f"Using single {file_type} file: {path}")
    elif os.path.isdir(path):
        file_format = detect_file_format(path)
        logger.info(f"{file_type.capitalize()} directory format detected: {file_format}")
        files = find_files_by_format(path, file_format)
        if not files:
            logger.error(f"No supported files found in {path}")
            return []
    else:
        logger.error(f"{file_type.capitalize()} input path does not exist: {path}")
        return []
    return files


def find_files_from_default_directory(file_type: str) -> List[str]:
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
    files = find_files_by_format(default_dir, file_format)
    if not files:
        logger.error(f"No supported files found in {default_dir}")
        return []
    return files


def find_input_files(image_input: Optional[str], mask_input: Optional[str]) -> Tuple[List[str], List[str]]:
    # Find image files
    image_files = files_catching(image_input, "image")
    if not image_files:
        return [], []

    # Find mask files
    mask_files = files_catching(mask_input, "mask")
    if not mask_files:
        return [], []

    return image_files, mask_files


# --------------------------------------------------------------------
# 🧠 3. Image–Mask Matching
# --------------------------------------------------------------------

def _extract_patient_id(path: str) -> str:
    base = os.path.basename(path)
    match = re.search(r"(\d+|[A-Za-z0-9_-]+)", base)
    return match.group(0) if match else base


def _extract_base_names(files: List[str], strip_extensions: Tuple[str, ...] = ()) -> Dict[str, str]:
    base_names = {}
    for file_path in files:
        base = os.path.splitext(os.path.basename(file_path))[0]
        # Handle cases like .nii.gz where extra stripping is needed
        for ext in strip_extensions:
            if base.endswith(ext):
                base = os.path.splitext(base)[0]
        base_names[base] = file_path
    return base_names


def _create_matched_pairs(
        image_base_names: Dict[str, str],
        mask_base_names: Dict[str, str],
        format_name: str
) -> List[Tuple[str, str]]:
    """Match images with corresponding masks by basename."""
    matched_pairs: List[Tuple[str, str]] = []

    for basename, img_path in image_base_names.items():
        mask_path = mask_base_names.get(basename)
        if mask_path:
            matched_pairs.append((img_path, mask_path))
            logger.info(f"Matched {format_name} pair found for {basename}")
        else:
            logger.warning(f"No matching mask found for image: {basename}")

    return matched_pairs


def _match_nifti_pairs(image_files: List[str], mask_files: List[str]) -> List[Tuple[str, str]]:
    return _create_matched_pairs(
        _extract_base_names(image_files, strip_extensions=(".nii",)),
        _extract_base_names(mask_files, strip_extensions=(".nii",)),
        "NIfTI"
    )


def _match_nrrd_pairs(image_files: List[str], mask_files: List[str]) -> List[Tuple[str, str]]:
    return _create_matched_pairs(
        _extract_base_names(image_files),
        _extract_base_names(mask_files),
        "NRRD"
    )


def _match_dicom_pairs(image_files: List[str], mask_files: List[str]) -> List[Tuple[str, str]]:
    return _create_matched_pairs(
        {os.path.basename(f): f for f in image_files},
        {os.path.basename(f): f for f in mask_files},
        "DICOM"
    )


def _match_opencv_supported_pairs(image_files: List[str], mask_files: List[str]) -> List[Tuple[str, str]]:
    return _create_matched_pairs(
        _extract_base_names(image_files),
        _extract_base_names(mask_files),
        "OpenCV"
    )


def _match_mixed_format_pairs(image_files: List[str], mask_files: List[str]) -> List[Tuple[str, str]]:
    """Handle mismatched formats gracefully."""
    image_format = detect_file_format(image_files[0])
    mask_format = detect_file_format(mask_files[0])

    logger.info("Different formats detected - checking pairing strategy")

    if len(image_files) == 1 and len(mask_files) > 1:
        logger.info(f"Single {image_format} image with {len(mask_files)} {mask_format} masks")
        return [(image_files[0], mask) for mask in mask_files]

    if len(mask_files) == 1 and len(image_files) > 1:
        logger.info(f"Single {mask_format} mask with {len(image_files)} {image_format} images")
        return [(img, mask_files[0]) for img in image_files]

    logger.info("Defaulting to sequential pairing for mixed formats")
    return list(zip(image_files, mask_files))


def match_image_mask_pairs(image_files: List[str], mask_files: List[str]) -> List[Tuple[str, str]]:
    """ Match image and mask files based on patient IDs or base_names. """
    if len(image_files) == 1 and len(mask_files) == 1:
        return [(image_files[0], mask_files[0])]

    image_format = detect_file_format(image_files[0])
    mask_format = detect_file_format(mask_files[0])

    # Special case: image dirs vs mask dicom files
    if image_format == mask_format == "dicom":
        if all(os.path.isdir(f) for f in image_files) and all(os.path.isfile(f) for f in mask_files):
            image_ids = {_extract_patient_id(f): f for f in image_files}
            mask_ids = {_extract_patient_id(f): f for f in mask_files}
            return _create_matched_pairs(image_ids, mask_ids, "DICOM (by patient ID)")
        return _match_dicom_pairs(image_files, mask_files)

    matchers = {
        "nifti": _match_nifti_pairs,
        "nrrd": _match_nrrd_pairs,
    }

    if image_format in matchers and mask_format == image_format:
        return matchers[image_format](image_files, mask_files)

    if cv2.haveImageReader(image_files[0]) and cv2.haveImageReader(mask_files[0]):
        return _match_opencv_supported_pairs(image_files, mask_files)

    return _match_mixed_format_pairs(image_files, mask_files)


# --------------------------------------------------------------------
# 🧾 4. Logging and Configuration Management
# --------------------------------------------------------------------

def load_ibsi_based_parameters(args):
    """Load optional parameters from file or JSON string."""
    if not getattr(args, 'IBSI_based_parameters', None):
        return None

    import json
    opt_arg = args.IBSI_based_parameters
    if os.path.isfile(opt_arg):
        with open(opt_arg, 'r') as f:
            return json.load(f)
    return json.loads(opt_arg)


def merge_cli_overrides(args, IBSI_based_parameters):
    """Merge CLI arguments into optional parameters, overriding JSON when necessary."""
    cli_overrides = {
        'radiomics_DataType': args.data_type,
        'radiomics_DiscType': args.disc_type,
        'radiomics_VoxInterp': args.vox_interp,
        'radiomics_ROIInterp': args.roi_interp,
        'radiomics_isScale': args.is_scale,
        'radiomics_isotVoxSize': args.isot_vox_size,
        'radiomics_isotVoxSize2D': args.isot_vox_size_2d,
        'radiomics_isIsot2D': args.is_isot_2d,
        'radiomics_isGLround': args.is_glround,
        'radiomics_isReSegRng': args.is_reseg_rng,
        'radiomics_isOutliers': args.is_outliers,
        'radiomics_isQuntzStat': args.is_quntz_stat,
        'radiomics_ReSegIntrvl01': args.reseg_intrvl01,
        'radiomics_ReSegIntrvl02': args.reseg_intrvl02,
        'radiomics_ROI_PV': args.roi_pv,
        'radiomics_qntz': args.qntz,
        'radiomics_IVH_Type': args.ivh_type,
        'radiomics_IVH_DiscCont': args.ivh_disc_cont,
        'radiomics_IVH_binSize': args.ivh_bin_size,
        'radiomics_isROIsCombined': args.is_rois_combined,
    }

    cli_overrides = {k: v for k, v in cli_overrides.items() if v is not None}

    if not IBSI_based_parameters:
        return cli_overrides or None

    if cli_overrides:
        IBSI_based_parameters.update(cli_overrides)
    return IBSI_based_parameters


# --------------------------------------------------------------------
# 🗂 4. Temporary File and Directory Management
# --------------------------------------------------------------------

def get_process_safe_temp_dir(prefix):
    base_temp_dir = create_tmp_dir()  # toto
    process_temp_dir = os.path.join(
        base_temp_dir, f"radiomics_proc_{os.getpid()}"
    )

    # Create directory if it doesn't exist
    os.makedirs(process_temp_dir, exist_ok=True)

    # Register for cleanup at exit
    import atexit
    atexit.register(lambda: _cleanup_temp_dir(process_temp_dir))

    return process_temp_dir


def create_tmp_dir() -> str:
    # Get the main directory
    base_temp_dir = os.path.join(os.getcwd(), 'temp')
    os.makedirs(base_temp_dir, exist_ok=True)
    return base_temp_dir


def _cleanup_temp_dir(temp_dir):
    """Clean up temporary directory at exit."""
    try:
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
    except:
        pass  # Silent cleanup


def create_process_safe_temp_file(prefix="temp", suffix=".tmp", custom_path=None):
    import uuid
    temp_dir = custom_path if custom_path else get_process_safe_temp_dir(prefix)
    os.makedirs(temp_dir, exist_ok=True)

    filename = os.path.join(
        temp_dir,
        f"{prefix}_{uuid.uuid4().hex[:8]}{suffix}"
    )

    # Create and immediately close the file
    with open(filename, 'wb'):
        pass

    return None, filename


def save_numpy_on_disk(array: np.ndarray, prefix="", suffix=".npy", custom_path=None):
    _, array_path = create_process_safe_temp_file(prefix, suffix, custom_path)
    np.save(array_path, array)
    return array_path


def remove_temp_file(file_path):
    try:
        os.remove(file_path)
    except Exception as e:
        logger.error(f"Error occurred while trying to remove the file: {e}")


# --------------------------------------------------------------------
# 🧩 5. General Utilities
# --------------------------------------------------------------------

def ensure_directory_exists(directory: Union[str, Path]) -> None:
    """Ensure a directory exists, creating it if necessary."""
    os.makedirs(directory, exist_ok=True)


def _load_bit_mappings(key: str):
    """Load and cache bit_mappings.json content once."""
    import importlib.resources as resources
    import json
    package = "pysera.engine.visera_oop.config.materials"
    with resources.open_text(package, "bit_mappings.json", encoding="utf-8") as f:
        data = json.load(f)
    return data[key]


def convert_dimensions_bit_map(dimensions: str) -> str:
    """Convert comma-separated dimension names to a binary bit map string with first and last bits always 1."""
    dim_set = {dim.strip().lower() for dim in dimensions.split(',')}
    all_dims = _load_bit_mappings("feature_dimensions_mapping")
    all_key = DEFAULT_RADIOICS_PARAMS["radiomics_dimensions"].lower()

    # If 'all' is present → all bits 1
    if all_key in dim_set:
        return "1" * len(all_dims)

    # Construct middle bits only
    middle_bits = ''.join(
        '1' if key.lower() in dim_set else '0'
        for key in all_dims[1:-1]
    )

    # Ensure first and last bit are always 1
    return f"1{middle_bits}1"


def convert_categories_bit_map(categories: str) -> str:
    """Convert comma-separated dimension names to a binary bit map string."""
    cat_set = set(cat.strip().lower() for cat in categories.split(','))

    all_cats = _load_bit_mappings("extractor_group_mapping")
    all_key = DEFAULT_RADIOICS_PARAMS["radiomics_categories"].lower()

    # If "all" dimensions selected → all bits = 1
    if all_key in cat_set:
        return "1" * len(all_cats)

    # Build binary map efficiently
    return ''.join('1' if key.lower() in cat_set else '0' for key in all_cats)
