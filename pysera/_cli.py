#!/usr/bin/env python3
"""
CLI interface for PySera library.

This module provides the command-line interface that wraps the original 
radiomics_standalone.py functionality while using the library's clean API.
"""

import sys
import logging
from typing import Optional

from .cli.argument_parser import parse_arguments, validate_arguments, print_usage_examples
from .processing.radiomics_processor import RadiomicsProcessor
from .config.settings import LOGGING_CONFIG
from .utils.log_logging import MemoryLogHandler


def setup_logging(memory_handler: Optional[MemoryLogHandler] = None) -> tuple:
    """Set up logging configuration for CLI usage."""
    logger = logging.getLogger('Dev_logger')
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, LOGGING_CONFIG['console_level']))
    console_format = logging.Formatter(LOGGING_CONFIG['console_format'])
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # Add context injection filter for consistent message tagging
    try:
        from .utils.log_logging import ContextInjectFilter
        logger.addFilter(ContextInjectFilter())
    except Exception:
        pass

    # Memory handler (write in excel)
    if memory_handler:
        memory_handler.setLevel(getattr(logging, LOGGING_CONFIG['memory_level']))
        memory_format = logging.Formatter(LOGGING_CONFIG['console_format'])  # Fixed typo from original
        memory_handler.setFormatter(memory_format)
        logger.addHandler(memory_handler)

    return logger, memory_handler


def main() -> int:
    """
    Main entry point for the PySera CLI.
    
    This function maintains compatibility with the original radiomics_standalone.py
    while using the new library architecture.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Set up logging with in-memory capture for bug sheet
        memory_handler = MemoryLogHandler()
        logger, memory_handler = setup_logging(memory_handler)
        
        # Parse command line arguments
        args = parse_arguments()

        # Parse optional parameters if provided
        optional_params = None
        if getattr(args, 'optional_params', None):
            import json, os
            opt_arg = args.optional_params
            if os.path.isfile(opt_arg):
                with open(opt_arg, 'r') as f:
                    optional_params = json.load(f)
            else:
                optional_params = json.loads(opt_arg)

        # Merge individual CLI flags into optional_params while preserving explicit JSON overrides
        cli_overrides = {
            'TOOLTYPE': args.tooltype,
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
        # Drop None values so defaults remain
        cli_overrides = {k: v for k, v in cli_overrides.items() if v is not None}
        if optional_params is None:
            optional_params = cli_overrides if cli_overrides else None
        elif cli_overrides:
            # CLI-specific flags should override JSON when both provided
            optional_params.update(cli_overrides)
        
        # Validate arguments
        if not validate_arguments(args):
            print_usage_examples()
            return 1
        
        # Create processor instance using the library API
        processor = RadiomicsProcessor(output_path=args.output, memory_handler=memory_handler)
        
        print(f"[DEBUG] About to detect file format for: {args.image_input}")
        
        # Process the batch using the library API
        result = processor.process_batch(
            image_input=args.image_input,
            mask_input=args.mask_input,
            apply_preprocessing=args.apply_preprocessing,
            min_roi_volume=args.min_roi_volume,
            num_workers=args.num_workers,
            enable_parallelism=args.enable_parallelism,
            feats2out=args.feats2out,
            bin_size=args.bin_size,
            roi_num=args.roi_num,
            roi_selection_mode=args.roi_selection_mode,
            feature_value_mode=args.feature_value_mode,
            optional_params=optional_params
        )
        
        # Persist bug sheet logs in the same Excel after processing (Sheet_3)
        try:
            from .utils.log_logging import log_logger
            # If processor returned an 'out' with paths, reuse its Excel path
            if result and 'out' in result and len(result['out']) >= 4:
                excel_path = result['out'][2]  # same path used in RadiomicsProcessor
            else:
                # Fallback to output directory with a default filename
                from .config.settings import OUTPUT_FILENAME_TEMPLATE
                from datetime import datetime
                folder_name = "results"
                output_filename = OUTPUT_FILENAME_TEMPLATE.format(
                    preprocessing_suffix="",
                    parallel_suffix="",
                    folder_name=folder_name,
                    timestamp=datetime.now().strftime("%m-%d-%Y_%H%M%S")
                )
                import os
                excel_path = os.path.join(args.output, output_filename)
            log_logger(excel_path, memory_handler.get_logs() if memory_handler else None)
        except Exception as e:
            logger.warning(f"Failed to write bug sheet logs from CLI: {e}")
        
        if result:
            logger.info(f"Processing complete. Results saved to {result['out'][3]}")
            return 0
        else:
            logger.info("Processing failed. Check logs for details.")
            return 1
            
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        print(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)