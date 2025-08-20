"""
pysera - A Python library for radiomics feature extraction with multiprocessing support

This library provides a simple, focused interface for radiomics feature extraction
from medical imaging data with built-in multiprocessing and logging capabilities.

Main Features:
- Support for multiple medical image formats (NIfTI, DICOM, NRRD, NumPy arrays)
- Built-in multiprocessing for efficient batch processing
- Comprehensive logging with Excel export functionality
- Extensive radiomics feature extraction capabilities
- Simple, single-function API

Example Usage:
    >>> import pysera
    >>> 
    >>> # Process single image-mask pair
    >>> result = pysera.process_batch(
    ...     image_input="image.nii.gz",
    ...     mask_input="mask.nii.gz",
    ...     output_path="./results"
    ... )
    >>> 
    >>> # Process batch with multiprocessing
    >>> result = pysera.process_batch(
    ...     image_input="./images",
    ...     mask_input="./masks",
    ...     output_path="./results",
    ...     num_workers="4",
    ...     feats2out=2,
    ...     apply_preprocessing=True
    ... )
"""

__version__ = "1.0.2"
__author__ = "Mohammad R. Salmanpour, Amir Hossein Pouria"
__email__ = "m.salmanpoor66@gmail.com"

# Import internal components (not exposed to users)
from .processing.radiomics_processor import RadiomicsProcessor
from .utils.log_logging import MemoryLogHandler
from .config.settings import LOGGING_CONFIG

def process_batch(
    image_input,
    mask_input,
    output_path=None,
    apply_preprocessing=False,
    min_roi_volume=10,
    num_workers="auto",
    enable_parallelism=True,
    feats2out=2,
    bin_size=25,
    roi_num=10,
    roi_selection_mode="per_Img",
    feature_value_mode="REAL_VALUE",
    report=True,
    optional_params=None
):
    """
    Process radiomics feature extraction for single or batch of medical images.
    
    This is the main function for pysera library that handles all radiomics processing
    with multiprocessing and logging capabilities built-in.
    
    Args:
        image_input (str): Path to image file or directory containing images.
                         Supported formats: NIfTI (.nii, .nii.gz), DICOM (.dcm), 
                         NRRD (.nrrd), NumPy (.npy), and directories with multiple files.
        
        mask_input (str): Path to mask file or directory containing masks.
                        Must have same format and dimensions as corresponding images.
        
        output_path (str, optional): Output directory for results. 
                                   Defaults to "./output_optimized" if not specified.
        
        apply_preprocessing (bool, optional): Apply ROI and intensity preprocessing.
                                            Defaults to False.
        
        min_roi_volume (int, optional): Minimum ROI volume threshold for processing.
                                      ROIs smaller than this will be filtered out.
                                      Defaults to 10.
        
        num_workers (str, optional): Number of parallel workers for multiprocessing.
                                   If auto, uses all available CPU cores.
                                   Set to 1 to disable multiprocessing.
                                   Defaults to None (auto-detect).
        
        enable_parallelism (bool, optional): Explicitly enable parallel processing.
                                         Defaults to True.
        
        feats2out (int, optional): Feature extraction mode (1-12).
                                 1: all IBSI_Evaluation, 2: 1st+3D+2.5D (default), 3: 1st+2D+2.5D,
                                 4: 1st+3D+selected2D+2.5D, 5: all+Moment, 6: 1st+2D+2.5D,
                                 7: 1st+2.5D, 8: 1st only, 9: 2D only, 10: 2.5D only,
                                 11: 3D only, 12: Moment only.
                                 Defaults to 2.
        
        bin_size (int, optional): Intensity discretization bin size for texture analysis.
                                Defaults to 25.
        
        roi_num (int, optional): Number of ROIs to select for processing.
                               Defaults to 10.
        
        roi_selection_mode (str, optional): ROI selection strategy.
                                          "per_Img": Select largest ROIs across entire image.
                                          "per_region": Group ROIs by color/label first.
                                          Defaults to "per_Img".
        
        feature_value_mode (str, optional): Value type for feature extraction.
                                  "APPROXIMATE_VALUE": Fast computation (recommended).
                                  "REAL_VALUE": Precise computation (slower).
                                  Defaults to "REAL_VALUE".
        
        report (bool, optional): Enable comprehensive logging during processing.
                                       Defaults to True.

    
    Returns:
        dict: Processing results containing:
            - 'success': bool indicating if processing completed successfully
            - 'output_path': str path to the output directory
            - 'processed_files': int number of files processed
            - 'features_extracted': int number of features extracted
            - 'processing_time': float total processing time in seconds
            - 'logs': list of log messages if logging was enabled
    
    Example:
        >>> import pysera
        >>> 
        >>> # Basic usage
        >>> result = pysera.process_batch(
        ...     image_input="scan.nii.gz",
        ...     mask_input="mask.nii.gz",
        ...     output_path="./results"
        ... )
        >>> 
        >>> # Advanced usage with multiprocessing and custom parameters
        >>> result = pysera.process_batch(
        ...     image_input="./patient_scans",
        ...     mask_input="./patient_masks",
        ...     output_path="./radiomics_results",
        ...     num_workers="4",
        ...     feats2out=5,  # Extract all features + Moment
        ...     min_roi_volume=10,
        ...     apply_preprocessing=True,
        ... )
        >>> 
        >>> print(f"Success: {result['success']}")
        >>> print(f"Processed {result['processed_files']} files")
        >>> print(f"Results saved to: {result['output_path']}")
    
    Raises:
        FileNotFoundError: If image_input or mask_input do not exist.
        ValueError: If parameters are invalid (e.g., feats2out not in range 1-12).
        RuntimeError: If processing fails due to incompatible image/mask dimensions.
    
    Note:
        This function automatically handles:
        - File format detection and loading
        - Multiprocessing setup and management
        - Logging configuration and Excel export
        - ROI preprocessing and optimization
        - Feature extraction and result compilation
        - Error handling and cleanup
    """
    import time
    from pathlib import Path
    import logging
    logger = logging.getLogger("Dev_logger")

    start_time = time.time()
    
    # Set up logging if requested
    memory_handler = None
    if report:
        memory_handler = MemoryLogHandler()
        logger = _setup_logging(memory_handler=memory_handler)
        logger.info("Starting pysera radiomics processing")
    
    try:
        # Create processor with logging if enabled
        processor = RadiomicsProcessor(
            output_path=output_path,
            memory_handler=memory_handler if report else None
        )
        
        # Process the batch
        result = processor.process_batch(
            image_input=image_input,
            mask_input=mask_input,
            apply_preprocessing=apply_preprocessing,
            min_roi_volume=min_roi_volume,
            num_workers=num_workers,
            enable_parallelism=enable_parallelism,
            feats2out=feats2out,
            bin_size=bin_size,
            roi_num=roi_num,
            roi_selection_mode=roi_selection_mode,
            feature_value_mode=feature_value_mode,
            optional_params=optional_params
        )
        
        processing_time = time.time() - start_time
        
        # Prepare return results
        return_result = {
            'success': bool(result),
            'output_path': processor.output_path,
            'processed_files': result.get('processed_files', 0) if result else 0,
            'features_extracted': result["out"][1] if result else 0,
            'processing_time': processing_time,
            'logs': memory_handler.get_logs() if memory_handler else []
        }
        
        if report:
            if result:
                logger.info(f"Processing completed successfully in {processing_time:.2f} seconds")
            else:
                logger.error("Processing failed")
        
        return return_result
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        if report and 'logger' in locals():
            logger.error(f"Processing failed with error: {e}")
        
        return {
            'success': False,
            'output_path': output_path,
            'processed_files': 0,
            'features_extracted': 0,
            'processing_time': processing_time,
            'logs': memory_handler.get_logs() if memory_handler else [],
            'error': str(e)
        }


def _setup_logging(memory_handler=None):
    """Internal function to set up logging configuration."""
    import logging
    
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

    # Memory handler (write in excel)
    if memory_handler:
        memory_handler.setLevel(getattr(logging, LOGGING_CONFIG['memory_level']))
        memory_format = logging.Formatter(LOGGING_CONFIG['console_format'])
        memory_handler.setFormatter(memory_format)
        logger.addHandler(memory_handler)

    return logger


# Feature extraction modes information (for documentation/reference)
_FEATURE_EXTRACTION_MODES = {
    1: "all IBSI_Evaluation",
    2: "1st+3D+2.5D (default)", 
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

# Main exports - only process_batch is public
__all__ = [
    'process_batch',  # Main public function
    '__version__',
    '__author__',
    '__email__'
] 