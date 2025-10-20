#!/usr/bin/env python3

import sys

from pysera.utils.helpers import load_ibsi_based_parameters, merge_cli_overrides
from pysera.utils.log_record import initialize_logging
from .cli.argument_parser import parse_arguments, validate_arguments, print_usage_examples
from .processing.radiomics_processor import RadiomicsProcessor


def main() -> int:
    """Main CLI entry point for the Radiomics processing pipeline."""
    try:
        args = parse_arguments()
        logger, memory_handler = initialize_logging(args.report)

        IBSI_based_parameters = load_ibsi_based_parameters(args)
        IBSI_based_parameters = merge_cli_overrides(args, IBSI_based_parameters)

        if not validate_arguments(args):
            print_usage_examples()
            return 1

        processor = RadiomicsProcessor(
            output_path=args.output,
            memory_handler=memory_handler,
            temporary_files_path=args.temporary_files_path,
            apply_preprocessing=args.apply_preprocessing,
            min_roi_volume=args.min_roi_volume,
            num_workers=args.num_workers,
            enable_parallelism=args.enable_parallelism,
            categories=args.categories,
            dimensions=args.dimensions,
            bin_size=args.bin_size,
            roi_num=args.roi_num,
            roi_selection_mode=args.roi_selection_mode,
            feature_value_mode=args.feature_value_mode,
            report=args.report,
            extraction_mode=args.extraction_mode,
            deep_learning_model=args.deep_learning_model,
            IBSI_based_parameters=IBSI_based_parameters
        )

        logger.debug(f"Detecting file format for: {args.image_input}")
        result = processor.process_batch(
            image_input=args.image_input,
            mask_input=args.mask_input
        )

        if result:
            logger.info(f"Processing complete. Results saved to {result['out'][3]}")
            return 0
        else:
            logger.warning("Processing failed. Check logs for details.")
            return 1

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
