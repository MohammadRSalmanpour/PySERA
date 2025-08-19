# -------------------------------------------------------------------------
# function [SUVpeak] = getSUVpeak(RawImg,ROI,pixelW,sliceS)
# -------------------------------------------------------------------------
# DESCRIPTION:
# This function computes the SUVpeak or local intensity peak in two ways:
# Local -> sphere around the maximum voxel
# Global -> sphere around each voxel
#
# --> This function is compatible with 2D analysis
# -------------------------------------------------------------------------
# INPUTS:
# - RawImg: This is the original ROI. This code will further trim it to the
#        ROI region. But it preserves the non-ROI values required for
#        SUVpeak calculation.
# - ROI: The whole ROI. It will be trimmed inside this code.
# - pixelW and SliceS: pixel width and slice thickness
#
# -------------------------------------------------------------------------
# OUTPUTS:
# - SUVpeak, a 2 element array consisting local and global SUVpeak.
# -------------------------------------------------------------------------
# AUTHOR(S):
# - Saeed Ashrafinia
# - Mahdi Hosseinzadeh
# -------------------------------------------------------------------------
# HISTORY:
# - Creation: December 2022
# -------------------------------------------------------------------------

import numpy as np
from scipy.signal import convolve
import gc

def getSUVpeak(RawImg2, ROI2, pixelW, sliceTh):
    RawImg = RawImg2.astype(np.float32, copy=True)
    # ROI = ROI2.astype(np.float32, copy=True)
    ROI = np.load(ROI2, mmap_mode='r')

    # Calculate kernel radius
    R = np.float_power((3 / (4 * np.pi)), (1 / 3)) * 10
    R_vox = np.divide(R, [pixelW, sliceTh])
    R_floor = np.floor(R_vox).astype(int)

    sph_shape = (2 * R_floor[0] + 1, 2 * R_floor[0] + 1, 2 * R_floor[1] + 1)
    SPH = np.zeros(sph_shape, dtype=np.float32)

    # Construct range aligned to voxel centers
    range_x = pixelW * (np.arange(-R_floor[0], R_floor[0] + 1))
    range_y = pixelW * (np.arange(-R_floor[0], R_floor[0] + 1))
    range_z = sliceTh * (np.arange(-R_floor[1], R_floor[1] + 1))
    y, x, z = np.meshgrid(range_y, range_x, range_z, indexing='ij')

    dists = np.sqrt((x - 0) ** 2 + (y - 0) ** 2 + (z - 0) ** 2)
    mask = dists <= R
    SPH[mask] = 1.0

    # Normalize kernel
    sph_sum = np.sum(SPH)
    if sph_sum == 0:
        return [0.0, 0.0]
    sph2 = SPH / sph_sum
    del SPH
    gc.collect()

    # Pad RawImg using NaNs (preserve original behavior)
    pad_wid = ((R_floor[0], R_floor[0]), (R_floor[0], R_floor[0]), (R_floor[1], R_floor[1]))
    ImgRawROIpadded = np.pad(RawImg, pad_wid, mode='constant', constant_values=np.nan)
    ImgRawROIpadded = np.nan_to_num(ImgRawROIpadded, nan=0.0)

    # Apply convolution
    try:
        C = convolve(ImgRawROIpadded, sph2, mode='valid', method='auto')
    except MemoryError:
        return [0.0, 0.0]
    del ImgRawROIpadded, sph2
    gc.collect()

    # Flatten arrays in column-major order (Fortran-like)
    T1_RawImg = RawImg.ravel(order='F')
    T1_ROI = ROI.ravel(order='F')
    T1_C = C.ravel(order='F')

    # Filter valid ROI voxels (original behavior)
    valid_mask = ~np.isnan(T1_RawImg)
    T1_RawImg = T1_RawImg[valid_mask]
    T1_ROI = T1_ROI[valid_mask]
    T1_C = T1_C[valid_mask]

    roi_mask = T1_ROI != 0
    if not np.any(roi_mask):
        return [0.0, 0.0]

    T2_RawImg = T1_RawImg[roi_mask]
    T2_C = T1_C[roi_mask]

    if T2_RawImg.size == 0:
        return [0.0, 0.0]

    maxind = np.argmax(T2_RawImg)
    SUVpeak = [float(T2_C[maxind]), float(np.max(T2_C))]

    # Clean RAM
    del ROI
    gc.collect()
    
    return SUVpeak