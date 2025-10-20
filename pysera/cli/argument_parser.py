"""
Command line argument parsing for the radiomics processing pipeline.
"""

import argparse
import logging
import os

from .. import DEFAULT_EXTRACTION_MODES, DEFAULT_DEEP_LEARNING_MODELS
from ..config.settings import (
    FEATURE_EXTRACTION_MODES, MIN_FEATURE_MODE, MAX_FEATURE_MODE, MIN_BIN_SIZE, MAX_BIN_SIZE,
    MIN_ROI_VOLUME, MAX_ROI_VOLUME, MIN_WORKERS, MAX_WORKERS,
    ROI_SELECTION_MODES, MIN_ROI_NUM, MAX_ROI_NUM, DEFAULT_RADIOICS_PARAMS,
    FEATURE_VALUE_MODES, LOG_LEVEL_MAP, EXTRACTION_MODES, DEEP_LEARNING_MODELS)

logger = logging.getLogger("Dev_logger")


def parse_arguments() -> argparse.Namespace:
    parser = _create_argument_parser()
    return parser.parse_args()


def _create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Extract radiomics features from medical images (Optimized Version with DICOM support).'
    )

    _add_required_arguments(parser)
    _add_optional_arguments(parser)

    return parser


def _add_required_arguments(parser: argparse.ArgumentParser) -> None:
    """Add required command line arguments."""
    parser.add_argument(
        '--image-input',
        required=True,
        help='Path to the input image file or directory (NIfTI or DICOM format)'
    )
    parser.add_argument(
        '--mask-input',
        required=True,
        help='Path to the input mask file or directory (NIfTI or DICOM RT-STRUCT format)'
    )


def _add_optional_arguments(parser: argparse.ArgumentParser) -> None:
    """Add optional command line arguments."""
    parser.add_argument(
        '--output',
        default="./output_result",
        help='Path to the output directory. Default is "./output_result"'
    )
    parser.add_argument(
        '--temporary-files-path',
        default="./temporary_files_path",
        help='Path to the temporary files directory. Default is "./temporary_files_path"'
    )
    parser.add_argument(
        '--apply-preprocessing',
        action='store_true',
        help='Apply ROI and intensity preprocessing'
    )
    parser.add_argument(
        '--min-roi-volume',
        type=int,
        default=DEFAULT_RADIOICS_PARAMS['radiomics_min_roi_volume'],
        help=f'Minimum ROI volume for reliable feature extraction (default: {DEFAULT_RADIOICS_PARAMS["radiomics_min_roi_volume"]})'
    )
    parser.add_argument(
        '--num-workers',
        type=str,
        default=DEFAULT_RADIOICS_PARAMS['radiomics_num_workers'],
        help='Number of parallel workers (default: number of CPU cores)'
    )
    parser.add_argument(
        '--enable-parallelism',
        action="store_false",
        help='parallel processing'
    )
    parser.add_argument(
        '--categories',
        type=str,
        default=DEFAULT_RADIOICS_PARAMS["radiomics_categories"],
        help='Comma-separated list of feature categories to extract (e.g., GLCM,GLRLM).'
    )

    parser.add_argument(
        '--dimensions',
        type=str,
        default=DEFAULT_RADIOICS_PARAMS["radiomics_dimensions"],
        help='Comma-separated list of feature dimensions (e.g., 2D, 3D, 2.5D, slice, voxel).'
    )
    parser.add_argument(
        '--bin-size',
        type=int,
        default=DEFAULT_RADIOICS_PARAMS["radiomics_BinSize"],
        help=f'Intensity discretion bin size (default: {DEFAULT_RADIOICS_PARAMS["radiomics_BinSize"]})'
    )
    parser.add_argument(
        '--roi-num',
        type=int,
        default=DEFAULT_RADIOICS_PARAMS["radiomics_roi_num"],
        help=f'Number of ROIs to select for feature extraction (default: {DEFAULT_RADIOICS_PARAMS["radiomics_roi_num"]}, range: {MIN_ROI_NUM}-{MAX_ROI_NUM})'
    )
    parser.add_argument(
        '--roi-selection-mode',
        type=str,
        default=DEFAULT_RADIOICS_PARAMS["radiomics_roi_selection_mode"],
        choices=ROI_SELECTION_MODES.keys(),
        help=f'ROI selection mode: {_format_roi_selection_modes_help()} (default: {DEFAULT_RADIOICS_PARAMS["radiomics_roi_selection_mode"]})'
    )
    parser.add_argument(
        '--feature-value-mode',
        type=str,
        default=DEFAULT_RADIOICS_PARAMS["radiomics_feature_value_mode"],
        choices=FEATURE_VALUE_MODES,
        help=f'Type of value to use: {FEATURE_VALUE_MODES} (default: {DEFAULT_RADIOICS_PARAMS["radiomics_feature_value_mode"]})'
    )
    parser.add_argument(
        '--report',
        type=str,
        default=DEFAULT_RADIOICS_PARAMS["radiomics_report"],
        choices=list(LOG_LEVEL_MAP.keys()),
        help=f'Type of report logs to use (default: All (which represents INFO, WARNING, ERROR Logs))'
    )
    parser.add_argument(
        '--extraction-mode',
        type=str,
        default=DEFAULT_EXTRACTION_MODES,
        choices=EXTRACTION_MODES,
        help=f'selection of extraction tool to use (default: {DEFAULT_EXTRACTION_MODES})'
    )
    parser.add_argument(
        '--deep-learning-model',
        type=str,
        default=DEFAULT_DEEP_LEARNING_MODELS,
        choices=DEEP_LEARNING_MODELS,
        help=f'selection of deep learning model to use (default: {DEFAULT_DEEP_LEARNING_MODELS})'
    )
    parser.add_argument(
        '--optional-params',
        type=str,
        default=None,
        help='JSON string or path to a JSON file with additional radiomics parameter overrides'
    )

    # Additional radiomics parameters (set to None by default to defer to settings defaults)
    parser.add_argument('--data-type', dest='data_type', type=str,
                        default=DEFAULT_RADIOICS_PARAMS["radiomics_DataType"],
                        help='radiomics_DataType (e.g., OTHER)')

    parser.add_argument('--disc-type', dest='disc_type', type=str,
                        default=DEFAULT_RADIOICS_PARAMS["radiomics_DiscType"],
                        help='radiomics_DiscType (e.g., FBS)')

    parser.add_argument('--vox-interp', dest='vox_interp', type=str,
                        default=DEFAULT_RADIOICS_PARAMS["radiomics_VoxInterp"],
                        help='radiomics_VoxInterp (e.g., Nearest)')

    parser.add_argument('--roi-interp', dest='roi_interp', type=str,
                        default=DEFAULT_RADIOICS_PARAMS["radiomics_ROIInterp"],
                        help='radiomics_ROIInterp (e.g., Nearest)')

    parser.add_argument('--is-scale', dest='is_scale', type=int, choices=[0, 1],
                        default=DEFAULT_RADIOICS_PARAMS["radiomics_isScale"],
                        help='radiomics_isScale (0 or 1)')

    parser.add_argument('--isot-vox-size', dest='isot_vox_size', type=int,
                        default=DEFAULT_RADIOICS_PARAMS["radiomics_isotVoxSize"],
                        help='radiomics_isotVoxSize (int)')

    parser.add_argument('--isot-vox-size-2d', dest='isot_vox_size_2d', type=int,
                        default=DEFAULT_RADIOICS_PARAMS["radiomics_isotVoxSize2D"],
                        help='radiomics_isotVoxSize2D (int)')

    parser.add_argument('--is-isot-2d', dest='is_isot_2d', type=int, choices=[0, 1],
                        default=DEFAULT_RADIOICS_PARAMS["radiomics_isIsot2D"],
                        help='radiomics_isIsot2D (0 or 1)')

    parser.add_argument('--is-glround', dest='is_glround', type=int, choices=[0, 1],
                        default=DEFAULT_RADIOICS_PARAMS["radiomics_isGLround"],
                        help='radiomics_isGLround (0 or 1)')

    parser.add_argument('--is-reseg-rng', dest='is_reseg_rng', type=int, choices=[0, 1],
                        default=DEFAULT_RADIOICS_PARAMS["radiomics_isReSegRng"],
                        help='radiomics_isReSegRng (0 or 1)')

    parser.add_argument('--is-outliers', dest='is_outliers', type=int, choices=[0, 1],
                        default=DEFAULT_RADIOICS_PARAMS["radiomics_isOutliers"],
                        help='radiomics_isOutliers (0 or 1)')

    parser.add_argument('--is-quntz-stat', dest='is_quntz_stat', type=int, choices=[0, 1],
                        default=DEFAULT_RADIOICS_PARAMS["radiomics_isQuntzStat"],
                        help='radiomics_isQuntzStat (0 or 1)')

    parser.add_argument('--reseg-intrvl01', dest='reseg_intrvl01', type=float,
                        default=DEFAULT_RADIOICS_PARAMS["radiomics_ReSegIntrvl01"],
                        help='radiomics_ReSegIntrvl01 (float)')

    parser.add_argument('--reseg-intrvl02', dest='reseg_intrvl02', type=float,
                        default=DEFAULT_RADIOICS_PARAMS["radiomics_ReSegIntrvl02"],
                        help='radiomics_ReSegIntrvl02 (float)')

    parser.add_argument('--roi-pv', dest='roi_pv', type=float, default=DEFAULT_RADIOICS_PARAMS["radiomics_ROI_PV"],
                        help='radiomics_ROI_PV (float)')

    parser.add_argument('--qntz', dest='qntz', type=str, default=DEFAULT_RADIOICS_PARAMS["radiomics_qntz"],
                        help='radiomics_qntz (e.g., Uniform)')

    parser.add_argument('--ivh-type', dest='ivh_type', type=int, default=DEFAULT_RADIOICS_PARAMS["radiomics_IVH_Type"],
                        help='radiomics_IVH_Type (int)')

    parser.add_argument('--ivh-disc-cont', dest='ivh_disc_cont', type=int,
                        default=DEFAULT_RADIOICS_PARAMS["radiomics_IVH_DiscCont"],
                        help='radiomics_IVH_DiscCont (int)')

    parser.add_argument('--ivh-bin-size', dest='ivh_bin_size', type=float,
                        default=DEFAULT_RADIOICS_PARAMS["radiomics_IVH_binSize"],
                        help='radiomics_IVH_binSize (float)')

    parser.add_argument('--is-rois-combined', dest='is_rois_combined', type=int, choices=[0, 1],
                        default=DEFAULT_RADIOICS_PARAMS["radiomics_isROIsCombined"],
                        help='radiomics_isROIsCombined (0 or 1)')


def _format_feature_modes_help() -> str:
    help_parts = []
    for mode, description in FEATURE_EXTRACTION_MODES.items():
        help_parts.append(f"{mode}={description}")
    return ", ".join(help_parts)


def _format_roi_selection_modes_help() -> str:
    help_parts = []
    for mode, description in ROI_SELECTION_MODES.items():
        help_parts.append(f"{mode}={description}")
    return ", ".join(help_parts)


def validate_arguments(args: argparse.Namespace) -> bool:
    validation_checks = [
        _validate_input_paths,
        _validate_output_directory,
        _validate_numeric_arguments
    ]

    for check in validation_checks:
        if not check(args):
            return False

    return True


def _validate_input_paths(args: argparse.Namespace) -> bool:
    """Validate input file/directory paths."""
    if not os.path.exists(args.image_input):
        logger.error(f"Error: Image input path does not exist: {args.image_input}")
        return False

    if not os.path.exists(args.mask_input):
        logger.error(f"Error: Mask input path does not exist: {args.mask_input}")
        return False

    return True


def _validate_output_directory(args: argparse.Namespace) -> bool:
    """Validate output directory creation."""
    if args.output:
        try:
            os.makedirs(args.output, exist_ok=True)
        except Exception as e:
            logger.error(f"Error: Cannot create output directory {args.output}: {e}")
            return False

    return True


def _validate_numeric_arguments(args: argparse.Namespace) -> bool:
    """Validate numeric argument ranges."""
    if not MIN_ROI_VOLUME <= args.min_roi_volume <= MAX_ROI_VOLUME:
        logger.error(
            f"Error: min_roi_volume must be between {MIN_ROI_VOLUME} and {MAX_ROI_VOLUME}, got {args.min_roi_volume}")
        return False

    if not MIN_BIN_SIZE <= args.bin_size <= MAX_BIN_SIZE:
        logger.error(f"Error: bin_size must be between {MIN_BIN_SIZE} and {MAX_BIN_SIZE}, got {args.bin_size}")
        return False

    if args.num_workers != DEFAULT_RADIOICS_PARAMS["radiomics_num_workers"]:
        if not MIN_WORKERS <= int(args.num_workers) <= MAX_WORKERS:
            logger.error(f"Error: num_workers must be between {MIN_WORKERS} and {MAX_WORKERS}, got {args.num_workers}")
            return False

    if not MIN_ROI_NUM <= args.roi_num <= MAX_ROI_NUM:
        logger.error(f"Error: roi_num must be between {MIN_ROI_NUM} and {MAX_ROI_NUM}, got {args.roi_num}")
        return False

    if args.roi_selection_mode not in ROI_SELECTION_MODES.keys():
        logger.error(
            f"Error: roi_selection_mode must be one of {ROI_SELECTION_MODES.keys()}, got {args.roi_selection_mode}")
        return False

    return True


def print_usage_examples() -> None:
    """Print usage library_examples for the radiomics processing pipeline."""
    _print_header()
    _print_basic_examples()
    _print_advanced_examples()
    _print_feature_modes()
    _print_footer()


def _print_header() -> None:
    """Print the header section."""
    print("\n" + "=" * 80)
    print("RADIOMICS PROCESSING PIPELINE - USAGE EXAMPLES")
    print("=" * 80)


def _print_basic_examples() -> None:
    """Print basic usage library_examples."""
    print("\n1. Basic usage with NIfTI files:")
    print("   python radiomics_standalone.py --image-input /path/to/images --mask-input /path/to/masks")

    print("\n2. Process DICOM files:")
    print("   python radiomics_standalone.py --image-input /path/to/dicom/series --mask-input /path/to/rtstruct")

    print("\n3. Specify output directory:")
    print("   python radiomics_standalone.py --image-input images --mask-input masks --output /path/to/results")


def _print_advanced_examples() -> None:
    """Print advanced usage library_examples."""
    print("\n4. Adjust ROI volume threshold:")
    print("   python radiomics_standalone.py --image-input images --mask-input masks --min-roi-volume 50")

    print("\n5. Use specific feature extraction mode:")
    print("   python radiomics_standalone.py --image-input images --mask-input masks --feats2out 5")

    print("\n6. Disable preprocessing:")
    print("   python radiomics_standalone.py --image-input images --mask-input masks --no-apply-preprocessing")

    print("\n7. Control parallel processing:")
    print("   python radiomics_standalone.py --image-input images --mask-input masks --num-workers 4")
    print("   python radiomics_standalone.py --image-input images --mask-input masks --enable-parallelism")

    print("\n8. Adjust bin size for discretion:")
    print("   python radiomics_standalone.py --image-input images --mask-input masks --bin-size 50")


def _print_feature_modes() -> None:
    """Print feature extraction modes."""
    print("\n" + "=" * 80)
    print("FEATURE EXTRACTION MODES (--feats2out):")
    print("=" * 80)

    for mode, description in FEATURE_EXTRACTION_MODES.items():
        print(f"  {mode}: {description}")


def _print_footer() -> None:
    """Print the footer section."""
    print("\n" + "=" * 80)
    print("For more information, see the README.md file.")
    print("=" * 80 + "\n")
