# PySERA: Python-Based Standardized Extraction for Radiomics Analysis – Python Radiomics Script and Library

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Development Status](https://img.shields.io/badge/status-stable-green.svg)](https://pypi.org/project/pysera/)

**PySERA** (Python-based Standardized Extraction for Radiomics Analysis) is a comprehensive Python library for radiomics feature extraction from medical imaging data. It provides a **simple, single-function API** with built-in multiprocessing support and comprehensive logging capabilities.

## 🔍 Table of Contents
- [🧩IBSI (Image Biomarker Standardisation Initiative) Standardization-1.0](#IBSI-Standardization)
- [🛠️Key Features](#key-features)
- [📥Installation](#installation)
  - [🌐GitHub Installation](#github-installation)
  - [💻Python Script - Command Line Interface (CLI)](#python-script---command-line-interface-cli)
  - [📦Library Installation via pip](#library-installation-via-pip)
- [📚Library Usage](#library-usage)
  - [📂Single File Processing](#single-file-processing)
  - [🧠In-Memory Array Processing](#in-memory-array-processing)
  - [⚡Parallel Batch Processing](#parallel-batch-processing)
  - [🔧Advanced Configuration](#advanced-configuration)
- [📂Data Structure Requirements](#data-structure-requirements)
- [📋PySERA Parameters Reference](#pysera-parameters-reference)
- [📚API Reference](#api-reference)
- [📊Output Structure](#output-structure)
- [🔢Feature Extraction Modes](#feature-extraction-modes)
- [📁Supported File Formats](#supported-file-formats)
- [🎯Library Examples](#library-examples)
- [⚡Performance Tips](#performance-tips)
- [❓Troubleshooting](#troubleshooting)
- [🕒Version History](#Version-History)
- [📬Contact](#contact)
- [👥Authors](#authors)
- [🙏Acknowledgment](#acknowledgment)
- [📜License](#license)

## ✨IBSI Standardization
Both the script and library have been rigorously standardized based on **IBSI** (Image Biomarker Standardisation Initiative) Standardization 1.0.
See the detailed evaluation and test cases here: [IBSI_Evaluation Folder](https://github.com/MohammadRSalmanpour/PySERATest/tree/main/IBSI_Evaluation)

## 🛠️Key Features

PySERA provides a **single-function API** that handles all radiomics processing:

```python
import pysera

result = pysera.process_batch(
    image_input="image.nii.gz",
    mask_input="mask.nii.gz",
    output_path="./results"
)
```

That's it! 🎉 All the complexity of multiprocessing, error & warning reports, file format handling, and feature extraction is handled automatically.

- **Single Function API**: One function does everything - `pysera.process_batch()`
- **Multi-format Support**: NIfTI, DICOM, NRRD, RTSTRUCT, NumPy arrays, and more
- **Automatic Multiprocessing**: Built-in parallel processing for maximum performance
- **Comprehensive Logging**: Excel export functionality for detailed analysis
- **Extensive Features**: 497 radiomics features with 12 extraction modes
- **Medical Image Optimized**: Designed for CT, MRI, PET, and other medical imaging

## 📥Installation

PySERA can be installed as a Python library for integration into your projects or as a standalone script for command-line usage. It supports Windows, macOS, and Linux. Below are the installation options.

### 🌐GitHub Installation 

For users who want to develop with the source code or run PySERA as a standalone command-line tool (CLI) without installing it as a Python package, you can clone the repository from GitHub.
This gives you access to the standalone script radiomics_standalone.py and all example files. After installing the dependencies, you can run the script directly (see the [💻Python Script - Command Line Interface (CLI)](#python-script---command-line-interface-cli) section).

```bash
# Clone the repository
git clone https://github.com/MohammadRSalmanpour/PySERA.git
cd pysera
```
### macOS/Linux Installation
#### Quick Setup (Recommended):

```bash
# Quick setup (creates a virtual environment and installs everything)
./dev_setup.sh
```
#### Manual Setup:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .

```
### Windows Setup
#### Quick Setup (Recommended):

```bash
# Quick setup
./dev_setup.sh
```

#### Manual Setup

```bash

python3 -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

```

### 💻Python Script - Command Line Interface (CLI)

If you just want to run the CLI without installing the library into Python,the standalone script 'radiomics_standalone.py' provides a command-line interface for radiomics processing :

```bash
# Process single files
python radiomics_standalone.py \
    --image_input image.nii.gz \
    --mask_input mask.nii.gz \
    --output ./results

# Batch processing (folders)
python radiomics_standalone.py \
    --image_input ./images \
    --mask_input ./masks \
    --output ./results \
    --num_workers 4
```

### 📦Library Installation via pip

Install the PySERA library directly from PyPI:

```bash
pip install pysera
```

## 📚Library Usage

Once installed, you can use PySERA directly in your Python code.

### 📂Single File Processing

```python
import pysera

# Process single image-mask pair
result = pysera.process_batch(
    image_input="scan.nii.gz",
    mask_input="mask.nii.gz",
    output_path="./results"
)

print(f"Success: {result['success']}")
print(f"Features extracted: {result['features_extracted']}")
print(f"Processing time: {result['processing_time']:.2f} seconds")
```

### 🧠In-Memory Array Processing

```python
import numpy as np
import nibabel as nib
import pysera

# Load image and mask as NumPy arrays (for example, using nibabel)
image_array = nib.load("patient002_image.nii.gz").get_fdata()
mask_array = nib.load("patient002_mask.nii.gz").get_fdata()

# Process the image and mask directly from memory
result = pysera.process_batch(
    image_input=image_array,
    mask_input=mask_array,
    output_path="./results"
)

# Display results
print(f"Success: {result['success']}")
print(f"Features extracted: {result['features_extracted']}")
print(f"Processing time: {result['processing_time']:.2f} seconds")
```

### ⚡Parallel Batch Processing

```python
import pysera

# Process multiple files with 4 CPU cores
result = pysera.process_batch(
    image_input="./patient_scans",
    mask_input="./patient_masks", 
    output_path="./results",
    num_workers="4",              # Use 4 CPU cores
    feats2out=5,               # Extract all features + Moment
    apply_preprocessing=True,   # Apply ROI preprocessing
)

print(f"Processed {result['processed_files']} files")
print(f"Total processing time: {result['processing_time']:.2f} seconds")
```

## 🔧Advanced Configuration

```python
import pysera

# Comprehensive processing with custom parameters
result = pysera.process_batch(
    image_input="image.nii.gz",
    mask_input="mask.nii.gz",
    output_path="./results",
    
    # Performance settings
    num_workers="auto",           # Use all available CPU cores
    enable_parallelism=True,     # Enable multiprocessing
    
    # Image feature extraction settings
    feats2out=2,               # 1st+3D+2.5D features (default)
    bin_size=25,               # Texture analysis bin size
    roi_num=10,                # Number of ROIs to process
    roi_selection_mode="per_Img",  # ROI selection strategy
    min_roi_volume=10,          # Minimum ROI volume threshold
    
    # Processing options
    apply_preprocessing=True,   # Apply ROI preprocessing
    feature_value_mode="REAL_VALUE",  # Fast computation

    # Optional parameters (advanced, overrides defaults)
    optional_params={
        "radiomics_DataType": "CT",
        "radiomics_DiscType": "FBN",
        "radiomics_isScale": 1
    },
    
    # Logging options
    report=True        # Enable detailed logging
)
```


## 📂Data Structure Requirements

For batch processing or multi-DICOM inputs, the folder structure for images and masks must follow these rules:
   - The **final folders** containing images and masks (e.g., ``images/`` and ``masks/``) must **not contain additional subfolders**. Only the image and mask files should be present in these folders.
   - There must be **only one folder level** between the parent folder and the image/mask files (e.g., ``parent/images/image001.nii.gz`` or ``parent/masks/mask001.nii.gz``).
   - **Warning**: Any additional internal subfolders within the final images or masks folders will cause PySERA to **malfunction** and fail to process the data correctly.

## Patient-Subfolder Organization (NIfTI/DICOM)

**Works with both:**

1. **DICOM Series** (multiple `.dcm` files per patient)  
2. **NIfTI Files** (single `.nii.gz` per patient)


### 🏷️Example Structures

**Note:**  PySERA supports all major formats, including DICOM, multi-slice DICOM, NIfTI, NRRD, RT Struct, and NumPy arrays.

#### 1️⃣**Flat NIfTI/NRRD Structure** 

**✅Correct:**
    
      parent/
      ├── images/ # All scan files directly here
      │   ├── patient001.nii.gz
      │   ├── patient002.nii.gz
      │   └── patient003.nii.gz
      └── masks/  # All mask files directly here
          ├── patient001.nii.gz
          ├── patient002.nii.gz
          └── patient003.nii.gz

#### 2️⃣**Patient-Subfolder NIfTI Structure**

**✅Correct:**

    parent/
    ├── CT_Images/ # Each patient has own folder
    │ ├── patient_01/
    │ │ └── scan.nii.gz # Single NIfTI file
    │ └── patient_02/
    │ └── scan.nii.gz
    └── CT_Masks/ # Mirrored structure
    ├── patient_01/
    │ └── segmentation.nii.gz
    └── patient_02/
    └── segmentation.nii.gz
    
**Notes:**  

- PySERA automatically processes DICOM series organized in patient subfolders.  
- **Patient subfolders are required** (one folder per patient).  
- **All DICOM slices for one series must be in the same patient folder.**  
- **Mask files must mirror the image folder structure.**  
  If there is a folder for `patient_01` under `CT_Images/`, there must be a corresponding `patient_01` folder under `CT_Masks/` containing the RTSTRUCT or mask.
    
    
### 3️⃣DICOM Series Structure

**✅Correct:**
    

    parent/
    ├── CT_Images/  # --image-input
    │ ├── patient_01/ # DICOM series folder
    │ │ ├── slice1.dcm  # Any number of slices
    │ │ ├── slice2.dcm
    │ │ └── slice3.dcm
    │ └── patient_02/
    │ ├── slice1.dcm
    │ └── slice2.dcm
    └── CT_Masks/   # --mask-input
    ├── patient_01/
    │ └── mask.dcm 
    └── patient_02/
    └── mask.dcm

   
**❌Incorrect Structure (Will Fail):**

      parent/
      ├── images/
      │   ├── subfolder1/
      │   │   ├── patient001.nii.gz
      │   └── subfolder2/
      │       ├── patient002.nii.gz
      └── masks/
          ├── subfolderA/
          │   ├── patient001.nii.gz
          └── patient002.nii.gz

### 📋PySERA Parameters Reference


| Parameter            | Type       | Default            | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
|----------------------|------------|--------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **image_input**       | str / .npy | Required           | Path to the image file, directory, or NumPy file containing the image data.                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| **mask_input**        | str / .npy | Required           | Path to the mask file, directory, or NumPy file defining the regions of interest.                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| **output_path**      | str        | ./output_optimized | Directory where the processing results will be saved.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **num_workers**      | str        | auto               | Number of CPU cores to use for processing. If auto, uses all available cores.                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **feats2out**           | int        | 2                  | Feature extraction mode (integer value from 1 to 12).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |                                                                                                                                                                                                                                                                                                                          
|  **apply_preprocessing** | bool       | False              | If True, applies preprocessing to the regions of interest (ROI).**Full preprocessing pipeline control:**<br>- When True:<br>  • Applies intensity normalization (CT: [-1000,400] rescaling)<br>  • Filters small ROIs (< min_roi_volume voxels)<br>  • Optimizes mask connectivity<br>  • Logs all preprocessing steps<br>- When False:<br>  • Uses raw image/mask without modifications<br>                                                                                                                                         |                                                                                                                                                                                                                                                                                                   
| **enable_parallelism**  | bool       | True               | If True, enables parallel processing for the analysis.                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| **min_roi_volume**      | int        | 10                 | Minimum volume threshold for regions of interest (ROI).                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| **bin_size**            | int        | 25                 | Bin size used for texture analysis.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| **roi_selection_mode**  | str        | per_Img            | **ROI selection strategy:**<br>- **"per_Img"** (default): Selects the top `roi_num` ROIs per image based on size, regardless of label category.<br>  • Suitable for single or dominant lesions per scan.<br>  • Preserves original spatial relationships.<br>- **"per_region"**: Selects up to `roi_num` ROIs separately for each label category, ensuring balanced representation across regions.<br>  • Useful in multi-lesion, multi-label, or longitudinal studies.<br>  • Requires consistent ROI labeling across datasets.<br> |
| **roi_num**             | int        | 10                 | Number of ROIs to process.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| **feature_value_mode**  | str        | REAL_VALUE         | Strategy for handling NaN values. Options: "APPROXIMATE_VALUE" or "REAL_VALUE". **"APPROXIMATE_VALUE"**: Replaces NaN features with substitutes (e.g., very small constants like `1e-30` or synthetic masks) to maintain pipeline continuity.<br>- **"REAL_VALUE"** (default): Keeps NaN values whenever feature extraction fails (e.g., small ROI, numerical instability), preserving the raw outcome without substitution.<br>                                                                                                     |
| **report**              | bool       | True               | If True, enables comprehensive report of the processing steps.                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| **optional_params**     | dict / JSON | See defaults       | Advanced configuration parameters. See the table below for detailed descriptions.                                                                                                                                                                                                                                                                                                                                                                                                                                                    |

#### 🔧Advanced configuration parameters (optional_params)


| Parameter                   | Type   | Default                | Description                                                                 |
|-----------------------------|--------|------------------------|-----------------------------------------------------------------------------|
| **TOOLTYPE**                | str    | "Handcrafted radiomic" | Tool classification.                                                        |
| **radiomics_DataType**      | str    | "OTHER"                | Image modality type (CT / PET / MRI / OTHER).                               |
| **radiomics_DiscType**      | str    | "FBS"                  |  Discretization type: either 'FBN' (fixed bin numbers) or 'FBS' (fixed bin size or fixed bin width).  |
| **radiomics_isScale**       | int    | 0                      | whether to do scaling. Has to be 1 to perform any resampling. If 0, always uses the original voxel dimension.                |
| **radiomics_VoxInterp**     | str    | "Nearest"              | Image resampling interpolation type  ('nearest', 'linear', or 'cubic'). Note: 'cubic' yeilds inconsistensies with IBSI results.                    |
| **radiomics_ROIInterp**     | str    | "Nearest"              | ROI resampling interpolation type  ('nearest', 'linear', or 'cubic'), default: 'linear'                                                 |
| **radiomics_isotVoxSize**   | int    | 2                      | New isotropic voxel size for resampling in 3D. This will be the new voxel size in X, Y and Z dimension.                                |
| **radiomics_isotVoxSize2D** | int    | 2                      | New voxel size for resampling slices in 2D. This maintains the Z dimension, and rescales both X and Y to this number.                                   |
| **radiomics_isIsot2D**      | int    | 0                      | (default 0) whether to resample image to isotropic 2D voxels (=1, i.e.keep the original slice thickness) or resample to isotropic 3D voxels (=0). (This is for 1st order features. Higher order 2D features are always calculated with original slice thickness).                            |
| **radiomics_isGLround**     | int    | 0                      | whether to round voxel intensities to the nearest integer (usually =1 for CT images, =0 for PET and SPECT)                          |
| **radiomics_isReSegRng**    | int    | 0                      | whether to perform range re-segmentation. The range is defined below in ReSegIntrvl. NOTE: Re-segmentation generally cannot be provided for arbitrary-unit modalities (MRI, SPECT)                               |
| **radiomics_isOutliers**    | int    | 0                      | Outlier removal flag (1 = remove ±3σ intensities).                          |
| **radiomics_isQuntzStat**   | int    | 1                      | (default 1) whether to use quantized image to calculate first order statistical features. If 0, no image resample/interp for calculating statistical features. (0 is preferrable for PET images)                   |
| **radiomics_ReSegIntrvl01** | int    | -1000                  | Range resegmentation interval. Intensity values outside this interval would be replaced by NaN.                             |
| **radiomics_ReSegIntrvl02** | int    | 400                    | Range resegmentation interval. Intensity values outside this interval would be replaced by NaN. Resegmentation upper bound (e.g., 400 for CT).                              |
| **radiomics_ROI_PV**        | float  | 0.5                    | (default 0.5) ROI partial volume threshold. Used to threshold ROI after resampling: i.e. ROI(ROI<ROI_PV) = 0, ROI(ROI>ROI_PV) = 1.                        |
| **radiomics_qntz**          | str    | "Uniform"              | An extra option for FBN Discretization Type: Either 'Uniform' quantization or 'Lloyd' for Max-Lloyd quantization. (defualt: Uniform)                                  |
| **radiomics_IVH_Type**      | int    | 3                      | Setting for Intensity Volume Histogram (IVH) Unit type={0: Definite(PET,CT), 1:Arbitrary(MRI,SPECT. This is FNB), 2: use 1000 bins, 3: use same discritization as histogram (for CT)}                             |
| **radiomics_IVH_DiscCont**  | int    | 1                      | Disc/Cont = {0:Discrete(for CT), 1:Continuous(for CT,PET. This is FBS)}                                    |
| **radiomics_IVH_binSize**   | float    | 2.0                    | Bin size for Intensity Volumen Histogram in case choosing setting 1 for FNB, or setting 0 and either IVH_DiscCont options.                                                       |
| **radiomics_isROIsCombined**| int    | 0                      | Whether to combine ROIs for multiple tumors to one.                           |



## 📚API Reference

### `pysera.process_batch()`

The main and only function you need for radiomics processing.


## 📊Output Structure

The ``pysera.process_batch()`` function produces two types of output: a **Python dictionary** with processing results and an **Excel file** containing detailed analysis data. Ensure your data follows `Data Structure Requirements` to avoid errors affecting output.

**Python Dictionary Output**

The function returns a dictionary with the following keys:

```python
{
    'success': bool,              # True if processing completed
    'output_path': str,           # Path to results directory
    'processed_files': int,       # Number of files processed
    'features_extracted': Dataframe,    # extracted features
    'processing_time': float,     # Processing time in seconds
    'logs': list,                # Log messages (if logging enabled)
    'error': str                 # Error message (if failed)
}
```
**Excel File Output**

**PySERA** generates an Excel file with three sheets:

📑1. **Radiomics Features**: Lists feature names (e.g., first-order, texture, shape) and their computed values for each processed image-mask pair.

⚙️2. **Selected Parameters**: Details the parameters used for the run (e.g., ``feats2out``, ``bin_size``, ``min_roi_volume``, ``roi_selection_mode``).

⚠️3. **Bugs, Warnings, and Errors**: Logs issues for each patient sample, including ROI labels, warnings (e.g., small ROI volume), and errors (e.g., “No matching mask found for patient001.nii.gz”).


## 🔢Feature Extraction Modes


| Mode | Feature Set | Approx. Features | Best Use Case | Performance Impact |
|------|-------------|------------------|---------------|--------------------|
| **1** | All IBSI-compliant features | 487              | Research studies requiring IBSI compliance | High computational load |
| **2** | 1st-order + 3D + 2.5D (default) | 215              | General purpose radiomics | Balanced performance |
| **3** | 1st-order + 2D + 2.5D | 351              | 2D slice analysis | Moderate |
| **4** | 1st-order + 3D + selected 2D + 2.5D | 351              | Comprehensive spatial analysis | High |
| **5** | All features + Moment invariants | 497              | Maximum feature extraction | Very high |
| **6** | 1st-order + 2D + 2.5D (alternative) | 351              | Secondary 2D analysis | Moderate |
| **7** | 1st-order + 2.5D only | 133              | Quick volumetric analysis | Fast |
| **8** | 1st-order statistics only | 79               | Rapid preliminary analysis | Very fast |
| **9** | 2D texture features only | 164              | Pure 2D texture studies | Moderate |
| **10** | 2.5D transitional features only | 54               | Volumetric transition analysis | Moderate |
| **11** | 3D texture features only | 82               | Pure volumetric analysis | High |
| **12** | Moment invariants only | 10               | Shape characterization | Fast |

**Selection Guidelines**:
- For **clinical workflows**: Modes 2 or 8 (balance of speed/features)
- For **research studies**: Modes 1 or 5 (maximum features)
- For **2D analysis**: Modes 3 or 9
- For **3D analysis**: Modes 11 or 4
- For **quick results**: Modes 7 or 8

## 📁Supported File Formats

### Image Files
- **NIfTI**: `.nii`, `.nii.gz`
- **DICOM**: `.dcm`, `.dicom`, directories with DICOM files
- **NRRD**: `.nrrd`, `.nhdr`
- **NumPy**: `.npy` arrays
- **Multi-DICOM**: Directory structure with patient subdirectories
- **RTSTRUCT**: DICOM-RT Structure Set files for contour-based images.
- **Other**: Any format readable by SimpleITK (e.g., CT, MRI, PET medical images).

### Mask Files
- Same formats as image files: NIfTI, DICOM, NRRD, NumPy, RTSTRUCT.
   - **Type**: Binary or labeled segmentation masks.

   - **Requirements**:
     - Must have the **same dimensions and geometry** as the corresponding image.
     - When loading folders containing images and masks, mask file names must **exactly match** the corresponding image file names.

## 🎯Library Examples

See the [`library_examples`](https://github.com/MohammadRSalmanpour/PySERATest/tree/main/library_examples) directory for comprehensive usage examples:

```bash
# Run library_examples
cd library_examples
python basic_usage.py
```

Example use cases:
- Basic single-file processing
- Batch processing with multiprocessing
- High-performance processing
- Custom parameter configuration
- Single-core processing
- Comprehensive analysis with full logging

## ⚡Performance Tips

1. **Use All Cores**: Set `num_workers=auto` to use all CPU cores
2. **Optimize ROI Volume**: Use `min_roi_volume` to filter small regions
3. **Choose Right Mode**: Use `feats2out=8` for speed, `feats2out=5` for comprehensive analysis
4. **Enable Preprocessing**: `apply_preprocessing=True` improves results
5. **Batch Processing**: Process multiple files in one call for efficiency

## ❓Troubleshooting

### Quick Guide

1. **Import Error**: `pip install -e .` in the project directory
2. **Missing Dependencies**: `pip install -r requirements-library.txt`
3. **File Not Found**: Check file paths and formats
4. **Memory Issues**: Reduce `num_workers` or increase `min_roi_volume`

### Get Help

- **Installation Issues**: See [INSTALL.md](INSTALL.md)
- **Examples**: Run `python examples/basic_usage.py`

## 🕒Version History

For detailed release notes, explanations of updates, and technical changes, please see the  
👉 [Development Report](https://github.com/MohammadRSalmanpour/PySERATest/blob/main/development_report.md)

    v1
    ├── v1.0
    │   ├── v1.0.2 - 2025-08-20
    │   │   - 🛠️change PySera name to pysera
    │   │
    │   ├── v1.0.1 - 2025-08-20
    │   │   - 🐛fixing bug in numpy array file processing in in-memory mode
    │   │
    │   └── v1.0.0 - 2025-08-19
    │       - 🛠️Structural modifications
    │       - ⚡Improved image loader 
    │       - ✨Added two strategies for feature value mode (real vs. approximate)
    │       - 🔢New parameter for number of ROIs to select
    │       - ✨Synthetic generation for ROI lesions smaller than 10 voxels
    │       - ✨New strategy for ROI selection (image-based vs. region-based)
    │       - 💾Disk-based processing to prevent RAM overflow
    │       - 🐛Fixed NaN value bug in some features
    │       - ✨Added support for processing NumPy array files in addition to file paths
    │       - ✅IBSI compliance validation
    │       - 📊New output structure including parameter set, error log, and warning report
    │       - 📦Updated package dependencies
    v0
    ├── v0.0
    │   └── v0.0.0 - 2025-08-13
    │       - 🔧IBSI Standardization 
    │       - 🐛Some Bug fix
    │
    └── initial version - 2022-02-12
       - 🎉Initial implementation  
       - 🛠️Core radiomics pipeline  
       - 📄Support for some types of files

## 📬Contact
SERA is available **free of charge**.
For access, questions, or feedback:

**Dr. Mohammad R. Salmanpour (Team Lead)**  
📧[msalman@bccrc.ca](mailto:msalman@bccrc.ca) | [m.salmanpoor66@gmail.com](mailto:m.salmanpoor66@gmail.com), [m.salmanpour@ubc.ca](mailto:m.salmanpour@ubc.ca)

---

## 🛠️Maintenance
For technical support and maintenance inquiries, please contact:

**Dr. Mohammad R. Salmanpour (Team Lead)**  
 msalman@bccrc.ca – m.salmanpoor66@gmail.com – m.salmanpour@ubc.ca

**Amir Hossein Pouria**  
amirporia99.1378@gmail.com  

## 👥Authors
- **Dr. Mohammad R. Salmanpour (Team Lead, Fund Provider, Evaluator, Medical Imaging Expert, Backend Development, Code Refactoring, Debugging, Library Management, IBSI Standardization, and Activation of the PySERA Library, and GUI Development in 3D Slicer)** – [msalman@bccrc.ca](mailto:msalman@bccrc.ca), [m.salmanpoor66@gmail.com](mailto:m.salmanpoor66@gmail.com), [m.salmanpour@ubc.ca](mailto:m.salmanpour@ubc.ca)
- **Amir Hossein Pouria (Assistant Team Lead; Backend Development, Code Refactoring, Debugging, and Library Management)** – [amirporia99.1378@gmail.com](mailto:amirporia99.1378@gmail.com)
- **Sirwan Barichin (IBSI Standardization, Debugging, and Activation of the PySERA Library, and GUI Development in 3D Slicer)** – [sirwanbarichin@gmail.com](mailto:sirwanbarichin@gmail.com)
- **Yasaman Salehi (Backend Development, Code Refactoring, and Debugging)** – [y.salehi7698@gmail.com](mailto:y.salehi7698@gmail.com)
- **Sonya Falahati (Tesing and Data prepration)** – [falahati.sonya@gmail.com](mailto:falahati.sonya@gmail.com)
- **Dr. Mehrdad Oveisi (Evaluator, Software Engineer, and Advisor)** – [moveisi@cs.ubc.ca](mailto:moveisi@cs.ubc.ca)
- **Dr. Arman Rahmim (Fund Provider, Medical Imaging Expert, Evaluator, and Advisor)** – [arman.rahmim@ubc.ca](mailto:arman.rahmim@ubc.ca), [arahmim@bccrc.ca ](mailto:arahmim@bccrc.ca)

## 📚Citation

```bibtex
@software{pysera2025,
  title={pysera: A Simple Python Library for Radiomics Feature Extraction},
  author={pysera Team},
  year={2025},
  url={https://github.com/MohammadRSalmanpour/PySERA}
}
```
## 📜License

This open-source software is released under the **MIT License**, which grants permission to use, modify, and distribute it for any purpose, including research or commercial use, without requiring modified versions to be shared as open source. See the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/MohammadRSalmanpour/PySERA/issues)
- **Documentation**: This README and the included guides
- **Examples**: See `examples/basic_usage.py`

## Acknowledgment

This study was supported by:  

- [🔬 **Qu**antitative **R**adiomolecular **I**maging and **T**herapy (Qurit) Lab, University of British Columbia, Vancouver, BC, Canada](https://www.qurit.ca)  
- [🏥 BC Cancer Research Institute, Department of Basic and Translational Research, Vancouver, BC, Canada](https://www.bccrc.ca/)  
- [💻 **Vir**tual **Collaboration (VirCollab) Group, Vancouver, BC, Canada](https://www.vircollab.com)  
- [🏭 **Tec**hnological **Vi**rtual **Co**llaboration **Corp**oration (TECVICO Corp.), Vancouver, BC, Canada](https://www.tecvico.com)  
We gratefully acknowledge funding from the💰 Natural Sciences and Engineering Research Council of Canada (**NSERC**) Idea to Innovation [**I2I**] Grant **GR034192**.
---

*PySERA - Simple, powerful radiomics in one function call. 🚀*
