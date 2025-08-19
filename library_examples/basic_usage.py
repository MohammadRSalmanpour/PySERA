#!/usr/bin/env python3
"""
Basic usage library_examples for PySera library.

This script demonstrates how to use PySera's simple, focused API with just
the process_batch function for all radiomics feature extraction needs.
"""

import pysera
import time


def example_1_basic_processing():
    """Example 1: Basic radiomics processing with default settings."""
    print("=== Example 1: Basic Processing ===")
    
    # Simple processing with just image and mask paths
    # Note: Replace these paths with your actual image and mask files
    image_input = "path/to/your/image.nii.gz"
    mask_input = "path/to/your/mask.nii.gz"
    
    try:
        result = pysera.process_batch(
            image_input=image_input,
            mask_input=mask_input,
            output_path="./results"
        )
        
        if result['success']:
            print(f"✅ Processing completed successfully!")
            print(f"   Results saved to: {result['output_path']}")
            print(f"   Processed files: {result['processed_files']}")
            print(f"   Features extracted: {result['features_extracted']}")
            print(f"   Processing time: {result['processing_time']:.2f} seconds")
        else:
            print("❌ Processing failed")
            if 'error' in result:
                print(f"   Error: {result['error']}")
            
    except FileNotFoundError:
        print("⚠️ Example files not found. Please update the paths to your actual data.")
    except Exception as e:
        print(f"❌ Error during processing: {e}")


def example_2_batch_processing_with_multiprocessing():
    """Example 2: Batch processing with multiprocessing and logging."""
    print("\n=== Example 2: Batch Processing with Multiprocessing ===")
    
    try:
        result = pysera.process_batch(
            image_input="path/to/image/directory",
            mask_input="path/to/mask/directory",
            output_path="./batch_results",
            num_workers="4",  # Use 4 CPU cores for parallel processing
            min_roi_volume=10,  # Minimum ROI volume threshold
            feats2out=2,  # Feature extraction mode: 1st+3D+2.5D
            apply_preprocessing=True,
            report=True
        )
        
        if result['success']:
            print(f"✅ Batch processing completed!")
            print(f"   Processed {result['processed_files']} files")
            print(f"   Total features extracted: {result['features_extracted']}")
            print(f"   Processing time: {result['processing_time']:.2f} seconds")
            print(f"   Log messages captured: {len(result['logs'])}")
            print("✅ Logs exported to Excel")
        else:
            print("❌ Batch processing failed")
            
    except FileNotFoundError:
        print("⚠️ Example directories not found. Please update the paths to your actual data.")
    except Exception as e:
        print(f"❌ Error during batch processing: {e}")


def example_3_high_performance_processing():
    """Example 3: High-performance processing with all CPU cores."""
    print("\n=== Example 3: High-Performance Processing ===")
    
    try:
        # Use all available CPU cores and comprehensive feature extraction
        result = pysera.process_batch(
            image_input="path/to/image.nii.gz",
            mask_input="path/to/mask.nii.gz",
            output_path="./high_performance_results",
            num_workers="auto",  # Use all available CPU cores
            feats2out=5,  # Extract all features + Moment (most comprehensive)
            apply_preprocessing=True,
            report=True
        )
        
        if result['success']:
            print("✅ High-performance processing completed!")
            print(f"   Processing time: {result['processing_time']:.2f} seconds")
            print(f"   Features extracted: {result['features_extracted']}")
        else:
            print("❌ High-performance processing failed")
            
    except FileNotFoundError:
        print("⚠️ Example files not found. Please update the paths to your actual data.")
    except Exception as e:
        print(f"❌ Error during processing: {e}")


def example_4_custom_configuration():
    """Example 4: Custom configuration with specific parameters."""
    print("\n=== Example 4: Custom Configuration ===")
    
    print("Available feature extraction modes:")
    modes = {
        1: "all IBSI_Evaluation",
        2: "1st+3D+2.5D (default)", 
        3: "1st+2D+2.5D",
        4: "1st+3D+selected2D+2.5D",
        5: "all+Moment",
        8: "1st order only (fast)",
        12: "Moment only"
    }
    for mode, description in modes.items():
        print(f"  {mode}: {description}")
    
    try:
        result = pysera.process_batch(
            image_input="path/to/image.nii.gz",
            mask_input="path/to/mask.nii.gz",
            output_path="./custom_results",
            bin_size=32,  # Custom bin size
            roi_num=5,   # Process top 5 ROIs
            feats2out=8,  # Extract only 1st order features (fast)
            min_roi_volume=100,  # Higher threshold
            roi_selection_mode="per_region",  # Group ROIs by color/label
            feature_value_mode="REAL_VALUE",  # More precise computation
            report=True
        )
        
        if result['success']:
            print("✅ Custom configuration processing completed!")
            print(f"   Custom parameters applied successfully")
        else:
            print("❌ Custom configuration processing failed")
            
    except FileNotFoundError:
        print("⚠️ Example files not found. Please update the paths to your actual data.")
    except Exception as e:
        print(f"❌ Error during custom processing: {e}")


def example_5_single_core_processing():
    """Example 5: Single-core processing (no multiprocessing)."""
    print("\n=== Example 5: Single-Core Processing ===")
    
    try:
        result = pysera.process_batch(
            image_input="path/to/image.nii.gz",
            mask_input="path/to/mask.nii.gz",
            output_path="./single_core_results",
            num_workers="1",  # Disable multiprocessing
            feats2out=8,  # Fast feature extraction
            report=False,  # Disable logging for speed
            apply_preprocessing=False  # Skip preprocessing for speed
        )
        
        if result['success']:
            print("✅ Single-core processing completed!")
            print(f"   Processing time: {result['processing_time']:.2f} seconds")
            print("   (Optimized for simplicity and speed)")
        else:
            print("❌ Single-core processing failed")
            
    except FileNotFoundError:
        print("⚠️ Example files not found. Please update the paths to your actual data.")
    except Exception as e:
        print(f"❌ Error during single-core processing: {e}")


def example_6_comprehensive_analysis():
    """Example 6: Comprehensive analysis with full logging and features."""
    print("\n=== Example 6: Comprehensive Analysis ===")
    
    try:
        start_time = time.time()
        
        result = pysera.process_batch(
            image_input="path/to/image.nii.gz",
            mask_input="path/to/mask.nii.gz",
            output_path="./comprehensive_results",
            num_workers="4",
            feats2out=5,  # All features + Moment
            bin_size=25,
            roi_num=10,
            min_roi_volume=4,  # Include small ROIs
            apply_preprocessing=True,
            report=True
        )
        
        total_time = time.time() - start_time
        
        if result['success']:
            print("✅ Comprehensive analysis completed!")
            print(f"   Total execution time: {total_time:.2f} seconds")
            print(f"   Processing time: {result['processing_time']:.2f} seconds")
            print(f"   Features extracted: {result['features_extracted']}")
            print(f"   Log entries: {len(result['logs'])}")
            print(f"   Detailed logs saved to: ./comprehensive_results/detailed_logs.xlsx")
        else:
            print("❌ Comprehensive analysis failed")
            
    except FileNotFoundError:
        print("⚠️ Example files not found. Please update the paths to your actual data.")
    except Exception as e:
        print(f"❌ Error during comprehensive analysis: {e}")


if __name__ == "__main__":
    print("PySera Library Usage Examples")
    print("=" * 50)
    print(f"PySera version: {pysera.__version__}")
    print("Simple, focused API - just one function: pysera.process_batch()")
    print("Note: Update file paths in the library_examples to point to your actual data")
    print()
    
    # Run all library_examples
    example_1_basic_processing()
    example_2_batch_processing_with_multiprocessing()
    example_3_high_performance_processing()
    example_4_custom_configuration()
    example_5_single_core_processing()
    example_6_comprehensive_analysis()
    
    print("\n" + "=" * 50)
    print("Examples completed! Check the generated results directories.")
    print()
    print("✨ PySera API Summary:")
    print("   import pysera")
    print("   result = pysera.process_batch(image_input, mask_input, **options)")
    print()
    print("🚀 Key features automatically handled:")
    print("   • Multiprocessing for performance")
    print("   • Comprehensive logging with Excel export")
    print("   • All file format support (NIfTI, DICOM, NRRD, etc.)")
    print("   • ROI preprocessing and optimization")
    print("   • 400+ radiomics features with 12 extraction modes")
    print()
    print("📚 For more information, see the PySera documentation.")