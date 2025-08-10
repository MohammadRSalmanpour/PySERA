# -*- coding: utf-8 -*-
"""
PythonCode.dicom2nifti

@author: abrys
"""
from .settings import disable_validate_slice_increment, \
    disable_validate_orientation, \
    disable_validate_orthogonal, \
    disable_validate_slicecount, \
    disable_validate_multiframe_implicit, \
    disable_resampling, \
    enable_validate_orientation, \
    enable_validate_orthogonal, \
    enable_validate_slicecount, \
    enable_validate_slice_increment, \
    enable_validate_multiframe_implicit, \
    enable_resampling
from .convert_dicom import dicom_series_to_nifti
from .convert_dir import convert_directory

from .patch_pydicom_encodings import apply
apply()

