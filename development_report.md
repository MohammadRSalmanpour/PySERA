# Development Report

This document provides a detailed history of the development progress, major improvements, and bug fixes made throughout the lifecycle of the project.  
It serves as both a changelog and a reference for design decisions, allowing users and developers to track the evolution of the radiomics processing pipeline.  
Each version entry documents new features, optimizations, supported formats, compliance updates, and structural refinements that have contributed to making the system more robust, efficient, and standardized.  

---

## Version History
        
        v1
        ├── v1.0
        │   └── v1.0.0 - 2025-08-19
        │       - Optimized memory management by storing large RoIs as float32 NumPy arrays on disk with memory mapping, ensuring efficient management and preventing RAM overflows.  
        │       - Enabled robust feature extraction for very small RoIs (fewer than 10 voxels) using epsilon corrections (𝜀) and synthetic augmentation, supporting cases as small as single-voxel inputs.  
        │       - Corrected mathematical operations prone to errors (e.g., log(0), division by zero, sqrt of negatives, covariance matrix for single voxel, ConvexHull requirements, percentile estimation).  
        │       - Improved data loading performance with smoother and more efficient handling of medical imaging inputs.  
        │       - Extended supported input formats to include RT-Struct data paths and direct NumPy array inputs for both images and masks.  
        │       - Enhanced Excel output with two new tabs: Parameters (record of all parameters used) and Warnings and Errors (detailed runtime issues).  
        │       - Added flexible RoI selection controls:  
        │            • roi_num: defines how many RoIs to select.  
        │            • roi_selection_mode: supports per_Img and per_region strategies.  
        │       - Exposed radiomics parameters as configurable arguments, replacing hardcoded values.  
        │       - Implemented automatic padding to resolve image and mask dimension mismatches.  
        │       - Improved RoI naming in Excel outputs by including both label and lesion information when available.  
        │       - Performed codebase restructuring: renamed variables for clarity, modularized the implementation into a standalone library, updated requirements, and verified IBSI standard compliance.  
        │
        v0
        ├── v0.0
        │   └── v0.0.0 - 2025-08-13
        │       - Implemented IBSI standardization to ensure feature extraction follows international radiomics guidelines.  
        │       - Fixed several bugs and stability issues identified in earlier development versions.  
        │
        └── initial version - 2022-02-12
            - Released the first working implementation of the project.  
            - Developed the core radiomics processing pipeline, providing the foundation for feature extraction.  
            - Added support for initial file formats, enabling users to process a limited set of medical imaging data types.  
        
