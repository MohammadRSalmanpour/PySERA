"""
File handling utilities for the radiomics processing pipeline.
"""

import os
import logging
from typing import List, Tuple, Literal
import pydicom

# Always define these extensions at the top so they are available everywhere
NIFTI_EXTENSIONS = ('.nii', '.nii.gz')
DICOM_EXTENSIONS = ('.dcm', '.dicom')
NRRD_EXTENSIONS = ('.nrrd', '.nhdr')
NPY_EXTENSIONS = ('.npy',)
ALL_IMAGE_EXTENSIONS = NIFTI_EXTENSIONS + DICOM_EXTENSIONS + NRRD_EXTENSIONS + NPY_EXTENSIONS

try:
    from ..config.settings import MAX_FILES_TO_CHECK_FOR_FORMAT
    from nibabel.loadsave import load as nib_load
except ImportError:
    # Fallback for when running as standalone
    MAX_FILES_TO_CHECK_FOR_FORMAT = 10

logger = logging.getLogger("Dev_logger")

def detect_file_format(input_path: str) -> Literal['nifti', 'nrrd', 'npy', 'dicom', 'multi-dcm', 'multi-nifti', 'multi-nrrd', 'multi-npy', 'mixed', 'other', 'unknown']:
    """
    Robustly detect the format of a file or directory.
    Handles:
      - Single file (any supported format)
      - Folder of files (all nii, nrrd, npy, dcm, etc.)
      - Folder of subfolders (each subfolder = one patient/case)
    Returns one of: 'nifti', 'nrrd', 'npy', 'dicom', 'multi-dcm', 'multi-nifti', 'multi-nrrd', 'multi-npy', 'mixed', 'other', 'unknown'.
    """
    import os
    input_path = os.path.normpath(input_path)
    if os.path.isfile(input_path):
        return _detect_single_file_format(input_path)
    elif os.path.isdir(input_path):
        entries = [os.path.join(input_path, e) for e in os.listdir(input_path) if not e.startswith('.')]
        files = [e for e in entries if os.path.isfile(e)]
        dirs = [e for e in entries if os.path.isdir(e)]
        # If only files
        if files and not dirs:
            ext_map = {'.nii': 'nifti', '.nii.gz': 'nifti', '.nrrd': 'nrrd', '.nhdr': 'nrrd', '.npy': 'npy', '.dcm': 'dicom', '.dicom': 'dicom'}
            types = set()
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext == '.gz' and f.lower().endswith('.nii.gz'):
                    ext = '.nii.gz'
                types.add(ext_map.get(ext, 'other'))
            if len(types) == 1:
                t = types.pop()
                if t == 'dicom':
                    return 'dicom'  # treat as DICOM series (one patient)
                elif t == 'nifti':
                    return 'nifti'
                elif t == 'nrrd':
                    return 'nrrd'
                elif t == 'npy':
                    return 'npy'
                elif t == 'other':
                    return 'other'
            elif len(types) > 1:
                return 'mixed'
            else:
                return 'unknown'
        # If only dirs (multi-case folder)
        elif dirs and not files:
            subformats = set()
            for d in dirs:
                fmt = detect_file_format(d)
                subformats.add(fmt)
            if len(subformats) == 1:
                t = subformats.pop()
                if t == 'dicom':
                    return 'dicom'
                elif t == 'nifti':
                    return 'nifti'
                elif t == 'nrrd':
                    return 'nrrd'
                elif t == 'npy':
                    return 'npy'
                elif t == 'other':
                    return 'other'
            elif len(subformats) > 1:
                return 'mixed'
            else:
                return 'unknown'
        # If both files and dirs, treat as mixed
        elif files and dirs:
            return 'mixed'
        else:
            return 'unknown'
    else:
        logger.warning(f"Input path {input_path} is neither file nor directory.")
        return 'unknown'
    return 'unknown'


def _detect_single_file_format(file_path: str) -> Literal['nifti', 'dicom', 'nrrd', 'npy', 'other', 'unknown']:
    """
    Detect format of a single file.
    Returns one of: 'nifti', 'dicom', 'nrrd', 'npy', 'other', 'unknown'.
    """
    logger.info(f"Checking file: {file_path}")
    if file_path.lower().endswith(NIFTI_EXTENSIONS):
        logger.info(f"File {file_path} detected as NIfTI by extension.")
        return 'nifti'
    elif file_path.lower().endswith(DICOM_EXTENSIONS):
        logger.info(f"File {file_path} detected as DICOM by extension.")
        return 'dicom'
    elif file_path.lower().endswith(NRRD_EXTENSIONS):
        logger.info(f"File {file_path} detected as NRRD by extension.")
        return 'nrrd'
    elif file_path.lower().endswith(NPY_EXTENSIONS):
        logger.info(f"File {file_path} detected as NumPy by extension.")
        return 'npy'
    return _detect_file_format_by_content(file_path)


def _detect_file_format_by_content(file_path: str) -> Literal['nifti', 'dicom', 'nrrd', 'npy', 'other', 'unknown']:
    """
    Detect file format by reading file content.

    Args:
        file_path: Path to the file

    Returns:
        Detected format string
    """
    try:
        pydicom.dcmread(file_path)
        logger.info(f"File {file_path} detected as DICOM by content.")
        return 'dicom'
    except Exception:
        pass

    try:
        nib_load(file_path)
        logger.info(f"File {file_path} detected as NIfTI by content.")
        return 'nifti'
    except Exception:
        pass

    try:
        import nrrd
        nrrd.read(file_path)
        logger.info(f"File {file_path} detected as NRRD by content.")
        return 'nrrd'
    except Exception:
        pass

    try:
        import numpy as np
        np.load(file_path)
        logger.info(f"File {file_path} detected as NumPy by content.")
        return 'npy'
    except Exception:
        pass

    logger.warning(f"File {file_path} could not be detected as NIfTI, DICOM, or NRRD.")
    return 'unknown'


def _detect_directory_format(directory_path: str) -> Literal['nifti', 'dicom', 'nrrd', 'npy', 'multi-dcm', 'mixed', 'other', 'unknown']:
    """
    Detect format of files in a directory.
    Returns one of: 'nifti', 'dicom', 'nrrd', 'npy', 'multi-dcm', 'mixed', 'other', 'unknown'.
    """
    files = _get_files_in_directory(directory_path)
    logger.info(f"Files found in directory {directory_path}: {files}")

    nifti_files, dcm_files, nrrd_files, npy_files = _categorize_files_by_extension(files)

    # If no files found by extension, try content-based detection
    if not (nifti_files or dcm_files or nrrd_files or npy_files):
        nifti_files, dcm_files, nrrd_files, npy_files = _categorize_files_by_content(files)

    return _determine_format_from_categories(nifti_files, dcm_files, nrrd_files, npy_files, directory_path)


def _get_files_in_directory(directory_path: str) -> List[str]:
    """
    Get list of files in directory.

    Args:
        directory_path: Path to directory

    Returns:
        List of file paths
    """
    return [
        os.path.join(directory_path, f)
        for f in os.listdir(directory_path)
        if os.path.isfile(os.path.join(directory_path, f))
    ]


def _categorize_files_by_extension(files: List[str]):
    """
    Categorize files by their extensions.

    Args:
        files: List of file paths

    Returns:
        Tuple of (nifti_files, dcm_files, nrrd_files, npy_files)
    """
    nifti_files = [f for f in files if f.lower().endswith(NIFTI_EXTENSIONS)]
    dcm_files = [f for f in files if f.lower().endswith(DICOM_EXTENSIONS)]
    nrrd_files = [f for f in files if f.lower().endswith(NRRD_EXTENSIONS)]
    npy_files = [f for f in files if f.lower().endswith(NPY_EXTENSIONS)]
    return nifti_files, dcm_files, nrrd_files, npy_files


def _categorize_files_by_content(files: List[str]):
    """
    Categorize files by reading their content.

    Args:
        files: List of file paths

    Returns:
        Tuple of (nifti_files, dcm_files, nrrd_files, npy_files)
    """
    nifti_files = []
    dcm_files = []
    nrrd_files = []
    npy_files = []

    # Only check first few files for performance
    files_to_check = files[:MAX_FILES_TO_CHECK_FOR_FORMAT]

    for file_path in files_to_check:
        try:
            pydicom.dcmread(file_path)
            dcm_files.append(file_path)
            logger.info(f"File {file_path} detected as DICOM by content.")
        except Exception:
            pass
        try:
            nib_load(file_path)
            nifti_files.append(file_path)
            logger.info(f"File {file_path} detected as NIfTI by content.")
        except Exception:
            pass
        try:
            import nrrd
            nrrd.read(file_path)
            nrrd_files.append(file_path)
            logger.info(f"File {file_path} detected as NRRD by content.")
        except Exception:
            pass
        try:
            import numpy as np
            np.load(file_path)
            npy_files.append(file_path)
            logger.info(f"File {file_path} detected as NumPy by content.")
        except Exception:
            pass

    return nifti_files, dcm_files, nrrd_files, npy_files


def _determine_format_from_categories(nifti_files: List[str], dcm_files: List[str], nrrd_files: List[str], npy_files: List[str], directory_path: str) -> Literal['nifti', 'dicom', 'nrrd', 'npy', 'multi-dcm', 'mixed', 'other', 'unknown']:
    """
    Determine format based on categorized files.

    Args:
        nifti_files: List of NIfTI files
        dcm_files: List of DICOM files
        nrrd_files: List of NRRD files
        npy_files: List of NumPy files
        directory_path: Original directory path for logging

    Returns:
        Detected format string
    """
    total_files = len(nifti_files) + len(dcm_files) + len(nrrd_files) + len(npy_files)

    if total_files == 0:
        logger.warning(f"No supported files found in {directory_path}")
        return 'unknown'

    # Determine the dominant format
    if len(nifti_files) > 0 and len(dcm_files) == 0 and len(nrrd_files) == 0 and len(npy_files) == 0:
        logger.info(f"Directory {directory_path} detected as NIfTI format.")
        return 'nifti'
    elif len(dcm_files) > 0 and len(nifti_files) == 0 and len(nrrd_files) == 0 and len(npy_files) == 0:
        logger.info(f"Directory {directory_path} detected as DICOM format.")
        return 'dicom'
    elif len(nrrd_files) > 0 and len(nifti_files) == 0 and len(dcm_files) == 0 and len(npy_files) == 0:
        logger.info(f"Directory {directory_path} detected as NRRD format.")
        return 'nrrd'
    elif len(npy_files) > 0 and len(nifti_files) == 0 and len(dcm_files) == 0 and len(nrrd_files) == 0:
        logger.info(f"Directory {directory_path} detected as NumPy format.")
        return 'npy'
    else:
        logger.warning(f"Mixed or unsupported formats found in {directory_path}")
        return 'mixed'


def find_files_by_format(directory: str, format_type: str) -> List[str]:
    """
    Find files of a specific format in a directory.

    Args:
        directory: Directory to search
        format_type: Format to search for ('nifti', 'dicom', 'nrrd', or 'npy')

    Returns:
        List of file paths
    """
    if format_type == 'nrrd':
        return _find_nrrd_files(directory)
    elif format_type == 'nifti':
        return _find_nifti_files(directory)
    elif format_type == 'dicom':
        return _find_dicom_files(directory)
    elif format_type == 'npy':
        return _find_npy_files(directory)
    else:
        return []


def _find_nifti_files(directory: str) -> List[str]:
    """Find NIfTI files in directory and all subdirectories."""
    import glob
    import os
    nii_files = glob.glob(os.path.join(directory, "**", "*.nii.gz"), recursive=True)
    nii_files += glob.glob(os.path.join(directory, "**", "*.nii"), recursive=True)
    return nii_files


def _find_dicom_files(directory: str) -> List[str]:
    """Find DICOM files in directory.
    Handles three cases:
    1. Directory with subfolders (each subfolder is a patient/case) -> return subfolder paths
    2. Directory with only DICOM files (e.g., segmentation masks) -> return all DICOM files
    3. Directory with DICOM series (image: a folder of DICOM slices) -> return [directory]
    """
    if os.path.isdir(directory):
        entries = [os.path.join(directory, e) for e in os.listdir(directory) if not e.startswith('.')]
        subdirs = [e for e in entries if os.path.isdir(e)]
        files = [e for e in entries if os.path.isfile(e)]
        dcm_files = [f for f in files if f.lower().endswith(DICOM_EXTENSIONS)]
        # Case 1: subfolders (multi-patient)
        if subdirs and not dcm_files:
            return subdirs
        # Case 2: only DICOM files (e.g., segmentation masks)
        elif dcm_files and not subdirs:
            return dcm_files
        # Case 3: DICOM series (image: a folder of DICOM slices)
        elif dcm_files and not subdirs:
            return [directory]
        # Mixed: both subdirs and files (rare, treat as subdirs)
        elif subdirs and dcm_files:
            return subdirs
        else:
            return []
    return []


def _find_nrrd_files(directory: str) -> List[str]:
    """Find NRRD files in directory and all subdirectories."""
    import glob
    import os
    nrrd_files = glob.glob(os.path.join(directory, "**", "*.nrrd"), recursive=True)
    nrrd_files += glob.glob(os.path.join(directory, "**", "*.nhdr"), recursive=True)
    # print(nrrd_files)
    return nrrd_files


def _find_npy_files(directory: str) -> List[str]:
    """Find NumPy files in directory and all subdirectories."""
    import glob
    import os
    npy_files = glob.glob(os.path.join(directory, "**", "*.npy"), recursive=True)
    return npy_files


def _extract_patient_id(path: str) -> str:
    """Extract patient ID from a file or folder name (digits, letters, or custom logic)."""
    import re
    base = os.path.basename(path)
    # Try to extract a patient ID (customize this regex as needed)
    match = re.search(r'(\d+|[A-Za-z0-9_-]+)', base)
    return match.group(0) if match else base


def match_image_mask_pairs(image_files: List[str], mask_files: List[str]) -> List[Tuple[str, str]]:
    """
    Match image and mask files based on patient IDs extracted from folder/file names.
    Handles cases where image_files are directories (patient folders) and mask_files are DICOM files in a single folder.
    """
    if len(image_files) == 1 and len(mask_files) == 1:
        return [(image_files[0], mask_files[0])]

    image_format = detect_file_format(image_files[0])
    mask_format = detect_file_format(mask_files[0])

    # Special case: image_files are directories (patients), mask_files are DICOM files (single folder)
    if image_format == 'dicom' and mask_format == 'dicom':
        # If image_files are directories and mask_files are files
        if all(os.path.isdir(f) for f in image_files) and all(os.path.isfile(f) for f in mask_files):
            # Extract patient IDs
            image_ids = {_extract_patient_id(f): f for f in image_files}
            mask_ids = {_extract_patient_id(f): f for f in mask_files}
            # Match by patient ID
            matched = []
            for pid, img_path in image_ids.items():
                if pid in mask_ids:
                    matched.append((img_path, mask_ids[pid]))
                else:
                    logger.warning(f"No matching mask found for image patient ID: {pid}")
            for pid in mask_ids:
                if pid not in image_ids:
                    logger.warning(f"No matching image found for mask patient ID: {pid}")
            return matched
        # Fallback to original logic
        return _match_dicom_pairs(image_files, mask_files)
    elif image_format == 'nifti' and mask_format == 'nifti':
        return _match_nifti_pairs(image_files, mask_files)
    elif image_format == 'nrrd' and mask_format == 'nrrd':
        return _match_nrrd_pairs(image_files, mask_files)
    else:
        return _match_mixed_format_pairs(image_files, mask_files)


def _match_nifti_pairs(image_files: List[str], mask_files: List[str]) -> List[Tuple[str, str]]:
    """Match NIfTI image and mask files by basename."""
    image_basenames = _extract_nifti_basenames(image_files)
    mask_basenames = _extract_nifti_basenames(mask_files)
    
    return _create_matched_pairs(image_basenames, mask_basenames, "NIfTI")


def _extract_nifti_basenames(files: List[str]) -> dict:
    """Extract basenames from NIfTI files."""
    basenames = {}
    for file_path in files:
        basename = os.path.splitext(os.path.basename(file_path))[0]
        if basename.endswith('.nii'):
            basename = os.path.splitext(basename)[0]
        basenames[basename] = file_path
    return basenames


def _match_dicom_pairs(image_files: List[str], mask_files: List[str]) -> List[Tuple[str, str]]:
    """Match DICOM image and mask files by directory name."""
    image_basenames = {os.path.basename(f): f for f in image_files}
    mask_basenames = {os.path.basename(f): f for f in mask_files}
    
    return _create_matched_pairs(image_basenames, mask_basenames, "DICOM")


def _match_nrrd_pairs(image_files: List[str], mask_files: List[str]) -> List[Tuple[str, str]]:
    """Match NRRD image and mask files by basename."""
    image_basenames = _extract_nrrd_basenames(image_files)
    mask_basenames = _extract_nrrd_basenames(mask_files)
    
    return _create_matched_pairs(image_basenames, mask_basenames, "NRRD")


def _extract_nrrd_basenames(files: List[str]) -> dict:
    """Extract basenames from NRRD files."""
    basenames = {}
    for file_path in files:
        basename = os.path.splitext(os.path.basename(file_path))[0]
        basenames[basename] = file_path
    return basenames


def _create_matched_pairs(image_basenames: dict, mask_basenames: dict, format_name: str) -> List[Tuple[str, str]]:
    """Create matched pairs and log results."""
    matched_pairs = []
    
    for basename in image_basenames:
        if basename in mask_basenames:
            matched_pairs.append((image_basenames[basename], mask_basenames[basename]))
            logger.info(f"Matched {format_name} pair found for {basename}")
        else:
            logger.warning(f"No matching mask found for image: {basename}")
    
    for basename in mask_basenames:
        if basename not in image_basenames:
            logger.warning(f"No matching image found for mask: {basename}")
    
    return matched_pairs


def _match_mixed_format_pairs(image_files: List[str], mask_files: List[str]) -> List[Tuple[str, str]]:
    """Match files of different formats."""
    image_format = detect_file_format(image_files[0])
    mask_format = detect_file_format(mask_files[0])
    
    logger.info("Different formats detected - checking for single image with multiple masks")
    
    if len(image_files) == 1 and len(mask_files) > 1:
        logger.info(f"Single {image_format} image with {len(mask_files)} {mask_format} masks")
        return [(image_files[0], mask_file) for mask_file in mask_files]
    elif len(mask_files) == 1 and len(image_files) > 1:
        logger.info(f"Single {mask_format} mask with {len(image_files)} {image_format} images")
        return [(image_file, mask_files[0]) for image_file in image_files]
    else:
        logger.info("Using sequential pairing for mixed formats")
        return list(zip(image_files, mask_files))


def ensure_directory_exists(directory: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path to ensure exists
    """
    os.makedirs(directory, exist_ok=True)


def get_safe_filename(filename: str) -> str:
    """
    Convert a filename to a safe version by removing/replacing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename
    """
    # Replace invalid characters with underscores
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename 