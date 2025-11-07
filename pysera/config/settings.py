"""
Configuration settings for the radiomics processing pipeline.
"""

import os

# =============================================================================
# PROCESSING PARAMETERS
# =============================================================================

# Default radiomics processing parameters
DEFAULT_RADIOICS_PARAMS = {
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
    'radiomics_roi_num': 10,
    'radiomics_roi_selection_mode': "per_img",
    'radiomics_feature_value_mode': "REAL_VALUE",
    'radiomics_categories': "diag,morph,glcm,glrlm,glszm,ngtdm,ngldm",
    'radiomics_dimensions': "1st,2d",
    'radiomics_destination_folder': "./output_result",
    'radiomics_temporary_files_path': "./temporary_files_path",
    'radiomics_report': "all",
    'radiomics_min_roi_volume': 10,
    'radiomics_apply_preprocessing': False,
    'radiomics_enable_parallelism': True,
    'radiomics_num_workers': "auto"
}

# =============================================================================
# THRESHOLDS AND LIMITS
# =============================================================================

# Feature value mode
FEATURE_VALUE_MODES = ['APPROXIMATE_VALUE', 'REAL_VALUE']

# extraction mode
EXTRACTION_MODES = ["handcrafted_feature", "deep_feature"]
DEFAULT_EXTRACTION_MODES = "handcrafted_feature"

# deep learning model
DEEP_LEARNING_MODELS = ["resnet50", "vgg16", "densenet121"]
DEFAULT_DEEP_LEARNING_MODELS = "resnet50"

# aggregation lesion default value
DEFAULT_AGGREGATION_LESION = False

# Intensity preprocessing thresholds
INTENSITY_PERCENTILE_LOW = 1
INTENSITY_PERCENTILE_HIGH = 99

# Valid feature extraction mode range
MIN_FEATURE_MODE = 1
MAX_FEATURE_MODE = 12

# Valid bin size range
MIN_BIN_SIZE = 1
MAX_BIN_SIZE = 1000

# Valid ROI volume range
MIN_ROI_VOLUME = 1
MAX_ROI_VOLUME = 1000000

# Valid number of workers range
MIN_WORKERS = 1
MAX_WORKERS = 32

# ROI selection modes
ROI_SELECTION_MODES = {
    "per_img": "Select ROIs per image (ignore color grouping)",
    "per_region": "Group ROIs by color first, then select from each group"
}

# Valid ROI number range
MIN_ROI_NUM = 1
MAX_ROI_NUM = 1000

# features should be or should not be aggregated
AGGREGATED_FEATURES_BLACK_LIST = [
        "img_dim_x_init_img", "img_dim_y_init_img", "img_dim_z_init_img",
        "vox_dim_x_init_img", "vox_dim_y_init_img", "vox_dim_z_init_img",
        "mean_int_init_img", "min_int_init_img", "max_int_init_img",
        "img_dim_x_interp_img", "img_dim_y_interp_img", "img_dim_z_interp_img",
        "vox_dim_x_interp_img", "vox_dim_y_interp_img", "vox_dim_z_interp_img",
        "mean_int_interp_img", "min_int_interp_img", "max_int_interp_img",
        "int_mask_dim_x_init_roi", "int_mask_dim_y_init_roi", "int_mask_dim_z_init_roi",
        "int_mask_bb_dim_x_init_roi", "int_mask_bb_dim_y_init_roi", "int_mask_bb_dim_z_init_roi",
        "morph_mask_bb_dim_x_init_roi", "morph_mask_bb_dim_y_init_roi", "morph_mask_bb_dim_z_init_roi",
        "int_mask_vox_count_init_roi", "morph_mask_vox_count_init_roi",
        "int_mask_mean_int_init_roi", "int_mask_min_int_init_roi", "int_mask_max_int_init_roi",
        "int_mask_dim_x_interp_roi", "int_mask_dim_y_interp_roi", "int_mask_dim_z_interp_roi",
        "int_mask_bb_dim_x_interp_roi", "int_mask_bb_dim_y_interp_roi", "int_mask_bb_dim_z_interp_roi",
        "morph_mask_bb_dim_x_interp_roi", "morph_mask_bb_dim_y_interp_roi", "morph_mask_bb_dim_z_interp_roi",
        "int_mask_vox_count_interp_roi", "morph_mask_vox_count_interp_roi",
        "int_mask_mean_int_interp_roi", "int_mask_min_int_interp_roi", "int_mask_max_int_interp_roi",
        "int_mask_dim_x_reseg_roi", "int_mask_dim_y_reseg_roi", "int_mask_dim_z_reseg_roi",
        "int_mask_bb_dim_x_reseg_roi", "int_mask_bb_dim_y_reseg_roi", "int_mask_bb_dim_z_reseg_roi",
        "morph_mask_bb_dim_x_reseg_roi", "morph_mask_bb_dim_y_reseg_roi", "morph_mask_bb_dim_z_reseg_roi",
        "int_mask_vox_count_reseg_roi", "morph_mask_vox_count_reseg_roi",
        "int_mask_mean_int_reseg_roi", "int_mask_min_int_reseg_roi", "int_mask_max_int_reseg_roi",
    ]

AGGREGATED_FEATURES_WHITE_LIST = [
            "morph_volume_mesh",
            "morph_volume_count",
            "morph_surface_area",
            "morph_max_3d_diameter",
            "morph_major_axis_length",
            "morph_minor_axis_length",
            "morph_least_axis_length",
        ]

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
    8: 65  # 1st only
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Map your mode to actual log levels
LOG_LEVEL_MAP = {
    "none": {'console_level': None, 'memory_level': None},  # No logs
    "error": {'console_level': 'ERROR', 'memory_level': 'ERROR'},  # Errors only
    "warning": {'console_level': 'WARNING', 'memory_level': 'WARNING'},  # Warnings only
    "info": {'console_level': 'INFO', 'memory_level': 'INFO'},  # Info only
    "all": {'console_level': 'INFO', 'memory_level': 'INFO'},  # All (INFO, WARNING, ERROR)
}

LOGGING_CONFIG = {
    'console_format': '%(asctime)s - %(levelname)s - %(message)s',
    'memory_format': '%(asctime)s - %(levelname)s - %(message)s'
}

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================

# Output file naming template
OUTPUT_FILENAME_TEMPLATE = ("extracted_radiomics_features{preprocessing_suffix}{parallel_suffix}_{"
                            "timestamp}.xlsx")

# Directory structure constants
DEFAULT_DIRECTORIES = {
    'CT': 'CT',
    'SEG': 'SEG',
    'output': 'output_result',
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
