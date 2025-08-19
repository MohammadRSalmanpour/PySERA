"""
Configuration settings for the radiomics processing pipeline.
"""

import os

# =============================================================================
# PROCESSING PARAMETERS
# =============================================================================

# Default radiomics processing parameters
DEFAULT_RADIOICS_PARAMS = {
    "TOOLTYPE": "Handcrafted radiomic",
    'radiomics_DataType': "OTHER",
    'radiomics_DiscType': "FBS",
    'radiomics_BinSize': 25,
    'radiomics_isScale': 0,
    'radiomics_VoxInterp': "Nearest",
    'radiomics_ROIInterp': "Nearest",
    'radiomics_isotVoxSize': 2,
    'radiomics_isotVoxSize2D': 2,
    'radiomics_isIsot2D': 0,
    'radiomics_isGLround': 0,
    'radiomics_isReSegRng': 0,
    'radiomics_isOutliers': 0,
    'radiomics_isQuntzStat': 1,
    'radiomics_ReSegIntrvl01': -1000,
    'radiomics_ReSegIntrvl02': 400,
    'radiomics_ROI_PV': 0.5,
    'radiomics_qntz': "Uniform",
    'radiomics_IVH_Type': 3,
    'radiomics_IVH_DiscCont': 1,
    'radiomics_IVH_binSize': 2,
    'radiomics_ROI_num': 10,
    'radiomics_ROI_selection_mode': "per_Img",
    'radiomics_isROIsCombined': 0,
    'radiomics_Feats2out': 2,
    'radiomics_destfolder': None
}

# =============================================================================
# THRESHOLDS AND LIMITS
# =============================================================================

# ROI volume thresholds
DEFAULT_MIN_ROI_VOLUME = 10
MIN_COMPONENT_SIZE = 10
MIN_ROI_VOLUME_FOR_PROCESSING = 50

# Intensity preprocessing thresholds
INTENSITY_PERCENTILE_LOW = 1
INTENSITY_PERCENTILE_HIGH = 99
MIN_COMPONENT_SIZE_PERCENTAGE = 0.05

# File processing limits
MAX_FILES_TO_CHECK_FOR_FORMAT = 10
MIN_VOXELS_FOR_COMPONENT = 10

# =============================================================================
# FILE FORMATS AND EXTENSIONS
# =============================================================================

NIFTI_EXTENSIONS = ('.nii', '.nii.gz')
DICOM_EXTENSIONS = ('.dcm',)

# =============================================================================
# FEATURE EXTRACTION CONFIGURATION
# =============================================================================

# Feature extraction modes with descriptions
FEATURE_EXTRACTION_MODES = {
    1: "all IBSI_Evaluation",
    2: "1st+3D+2.5D", 
    3: "1st+2D+2.5D",
    4: "1st+3D+selected2D+2.5D",
    5: "all+Moment",
    6: "1st+2D+2.5D",
    7: "1st+2.5D",
    8: "1st only",
    9: "2D only",
    10: "2.5D only",
    11: "3D only",
    12: "Moment only"
}

# Expected feature counts for different modes (for validation)
EXPECTED_FEATURE_COUNTS = {
    5: 497,  # all+Moment
    2: 215,  # 1st+3D+2.5D (default)
    8: 65    # 1st only
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOGGING_CONFIG = {
    'console_level': 'INFO',
    'memory_level': 'WARNING',
    'console_format': '%(asctime)s - %(levelname)s - %(message)s',
    'memoty_format': '%(asctime)s - %(levelname)s - %(message)s'
}

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================

# Output file naming template
OUTPUT_FILENAME_TEMPLATE = "All_extracted_features_OPTIMIZED{preprocessing_suffix}{parallel_suffix}_{folder_name}_{timestamp}.xlsx"

# Directory structure constants
DEFAULT_DIRECTORIES = {
    'CT': 'CT',
    'SEG': 'SEG', 
    'output': 'output_optimized',
    'visera_backend': 'pysera/engine/visera'
}

# =============================================================================
# PATH UTILITY FUNCTIONS
# =============================================================================

def get_project_root() -> str:
    """Get the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_visera_pythoncode_path() -> str:
    """Get the path to ViSERA PythonCode directory."""
    project_root = get_project_root()
    return os.path.join(project_root, DEFAULT_DIRECTORIES['visera_backend'])

def get_default_output_path() -> str:
    """Get the default output directory path."""
    project_root = get_project_root()
    return os.path.join(project_root, DEFAULT_DIRECTORIES['output'])

def get_default_image_dir() -> str:
    """Get the default image directory path."""
    project_root = get_project_root()
    return os.path.join(project_root, DEFAULT_DIRECTORIES['CT'])

def get_default_mask_dir() -> str:
    """Get the default mask directory path."""
    project_root = get_project_root()
    return os.path.join(project_root, DEFAULT_DIRECTORIES['SEG'])

# =============================================================================
# VALIDATION CONSTANTS
# =============================================================================

# Valid feature extraction mode range
MIN_FEATURE_MODE = 1
MAX_FEATURE_MODE = 12

# Valid bin size range
MIN_BIN_SIZE = 1
MAX_BIN_SIZE = 1000

# Valid ROI volume range
MIN_ROI_VOLUME = 10
MAX_ROI_VOLUME = 1000000

# Valid number of workers range
MIN_WORKERS = 1
MAX_WORKERS = 32

# =============================================================================
# ROI SELECTION CONFIGURATION
# =============================================================================

# ROI selection modes
ROI_SELECTION_MODES = {
    "per_Img": "Select ROIs per image (ignore color grouping)",
    "per_region": "Group ROIs by color first, then select from each group"
}

# Valid ROI selection modes
VALID_ROI_SELECTION_MODES = ["per_Img", "per_region"]

# Valid ROI number range
MIN_ROI_NUM = 1
MAX_ROI_NUM = 1000 