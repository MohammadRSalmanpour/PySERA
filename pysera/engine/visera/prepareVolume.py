import numpy as np
from ..visera.Quantization import fixedBinSizeQuantization, uniformQuantization, lloydQuantization
from ..visera.interpolation import imresize3D, imresize
import logging
import gc
from ...utils.utils import save_numpy_on_disk, remove_temp_file
# -------------------------------------------------------------------------
# function [ROIonly,levels,Maskout,ROIboxResmp,newpixelW,newsliceTh] =
# prepareVolume(volume, Mask, DataType, pixelW, sliceTh, newVoxelSize,
#               VoxInterp, ROIInterp, ROI_PV, scaleType, isIsot2D, isScale,
#               isGLround, DiscType, qntz,Bin, isReSegRng, ReSegIntrvl,
#               isOutliers)
# -------------------------------------------------------------------------
# DESCRIPTION:
# This function prepares the input volume for 3D texture analysis. The
# following operations are performed:
#
# 1. Pre-processing of the ROIbox (PET: square-root (default:off), MR:
#    Collewet normalizaton, CT: nothing).
# 2. Isotropic resampling.
# 3. Image resegmentation.
# 4. grey levels rounding.
# 5. Image resegmentation.
# 6. Quantization of intensity dynamic range.
# 7. Crop the ROI to get rid of extra zeros
# --> This function is compatible with both 2D and 3D analysis
# -------------------------------------------------------------------------
# INPUTS:
# - volume: 2D or 3D array containing the medical images to analyze
# - mask: 2D or 3D array of dimensions corresponding to 'volume'. The mask
#         contains 1's in the region of interest (ROI), and 0's elsewhere.
# - DataType: String specifying the type of scan analyzed. Either 'PET',
#             'CT' or 'MRscan'.
# - pixelW: Numerical value specifying the in-plane resolution (mm) of 'volume'.
# - sliceTh: Numerical value specifying the slice spacing (mm) of 'volume'.
#           Put a random number for 2D analysis.
# - newVoxelSize: Numerical value specifying the scale at which 'volume' is
#                 isotropically  resampled in 3D (mm).
# - VoxInterp: interpolation method for the intensity ROI
# - ROIInterp: interpolation method for the morphological ROI
# - ROI_PV: partial volume threshold for thresholding morphological ROI.
#           Used to threshold ROI after resampling:
#           i.e. ROI(ROI<ROI_PV) =0, ROI(ROI>ROI_PV) = 1.
# - scaleType: 'NoRescale' if no resampling, 'XYZscale' if resample in all
#              3 dimensions, 'XYscale' if only resampling in X and Y
#              dimensions, 'Zscale' if only scaling Z direction.
# - isIsot2D: =1 if resampling only in X and Y dimensions.
# - isScale: whether to perform resampling
# - isGLrounding: whether to perform grey level rounding
# - DiscType: discritization type. Either 'FNB' for fixed number of bins
#             or 'FBS' for fixed bin size.
# - qntz: String specifying the quantization algorithm to use on
#             'volume'. Either 'Lloyd' for Lloyd-Max quantization, or
#             'Uniform' for uniform quantization.
# - Bin: number of bins (for fixed number of bins method) or bin size
#            (for bin size method).
# - isReSeg: whether to perform resegmentation
# - ResegIntrval: a 1x2 array of values expressing resegmentation interval.
# - isOutliers: whether to perform instensity outlier filtering
# -------------------------------------------------------------------------
# OUTPUTS: ROIonly,levels,Maskout,ROIboxResmp,newpixelW,newsliceTh
# - ImgBoxResampQuntz3D: Smallest box containing the ROI, with the imaging
#           data of the ready for texture analysis computations. Voxels
#           outside the ROI are set to NaNs.
# - levels: Vector containing the quantized gray-levels in the tumor region
#           (or reconstruction levels of quantization).
# - Maskout: Smallest matrix containing morphological ROI (0 or 1)
# - ROIboxResmp: Smallest matrix containing intensity ROI. This is the
#                resampled ROI, went through resegmtation and GL rounding,
#                and just before quantization.
# - newpixelW: width of the voxel in the X (=Y) direction in resampled ROI
# - newsliceTh: Slice thickness of the voxel in the Z direction in resamped
#               ROI
# -------------------------------------------------------------------------
# REFERENCE:
# [1] Vallieres, M. et al. (2015). A radiomics model from joint FDG-PET and
#     MRI texture features for the prediction of lung metastases in soft-tissue
#     sarcomas of the extremities. Physics in Medicine and Biology, 60(14),
#     5471-5496. doi:10.1088/0031-9155/60/14/5471
# -------------------------------------------------------------------------
# AUTHOR(S):
# - Martin Vallieres <mart.vallieres@gmail.com>
# - Saeed Ashrafinia
# - Mahdi Hosseinzadeh
# -------------------------------------------------------------------------
# HISTORY:
# - Creation: December 2022
# -------------------------------------------------------------------------


def prepareVolume(volume, mask_input, DataType, pixelW, sliceTh,
                  newVoxelSize, VoxInterp, ROIInterp, ROI_PV, scaleType, isIsot2D,
                  isScale, isGLround, DiscType, qntz, Bin,
                  isReSegRng, ReSegIntrvl, isOutliers):
    """Memory-optimized volume preparation with reduced copying and efficient operations"""

    # Set up quantization function
    if DiscType == 'FBS':
        quantization = fixedBinSizeQuantization
    elif DiscType == 'FBN':
        quantization = uniformQuantization
    else:
        raise('Error with discretization type. Must either be "FBS" (Fixed Bin Size) or "FBN" (Fixed Number of Bins).')

    if qntz == 'Lloyd':
        quantization = lloydQuantization

    # Use views instead of copies where possible
    # ROIBox = Mask if Mask.flags.writeable else Mask.copy()        # toto CMed

    ROIBox_path = mask_input
    # Load mask
    Mask = np.load(mask_input, mmap_mode='r')
    ROIBox = np.load(ROIBox_path, mmap_mode='r')

    Imgbox = volume.astype(np.float32) if volume.dtype != np.float32 else volume

    # MR scan specific processing
    if DataType == 'MRscan':
        ROIonly = Imgbox.copy()
        ROIonly[ROIBox == 0] = np.nan
        temp = CollewetNorm(ROIonly)
        ROIBox[np.isnan(temp)] = 0
        del temp, ROIonly
        gc.collect()

    flagPW = 0
    if scaleType == 'NoRescale':
        flagPW = 0
    elif scaleType == 'XYZscale':
        flagPW = 1
    elif scaleType == 'XYscale':
        flagPW = 2
    elif scaleType == 'Zscale':
        flagPW = 3

    if isIsot2D == 1:
        flagPW = 2

    if isScale == 0:
        flagPW = 0

    if flagPW == 0:
        a = 1
        b = 1
        c = 1
    elif flagPW == 1:
        a = pixelW / newVoxelSize
        b = pixelW / newVoxelSize
        c = sliceTh / newVoxelSize
    elif flagPW == 2:
        a = pixelW / newVoxelSize
        b = pixelW / newVoxelSize
        c = 1
    elif flagPW == 3:
        a = 1
        b = 1
        c = sliceTh / pixelW

    # Resampling
    ImgBoxResmp = Imgbox.copy()
    ImgWholeResmp = volume.copy()
    ROIBoxResmp = ROIBox.copy()
    ROIwholeResmp = Mask.copy()

    if Imgbox.ndim == 3 and flagPW != 0:
        if (a + b + c) != 3:
            ROIBoxResmp_path = imresize3D(ROIBox_path, [pixelW, pixelW, sliceTh],
                                     [np.ceil(ROIBox.shape[0] * a), np.ceil(ROIBox.shape[1] * b),
                                      np.ceil(ROIBox.shape[2] * c)], ROIInterp, 'constant',
                                     [newVoxelSize, newVoxelSize, newVoxelSize], isIsot2D, use_disk=True)
            ROIBoxResmp = np.load(ROIBoxResmp_path, mmap_mode='r+')
            Imgbox[np.isnan(Imgbox)] = 0
            ImgBoxResmp = imresize3D(Imgbox, [pixelW, pixelW, sliceTh],
                                     [np.ceil(Imgbox.shape[0] * a), np.ceil(Imgbox.shape[1] * b),
                                      np.ceil(Imgbox.shape[2] * c)], VoxInterp, 'constant',
                                     [newVoxelSize, newVoxelSize, newVoxelSize], isIsot2D, use_disk=False)
            ROIBoxResmp[ROIBoxResmp < ROI_PV] = 0
            ROIBoxResmp[ROIBoxResmp >= ROI_PV] = 1
            ROIwholeResmp_path = imresize3D(mask_input, [pixelW, pixelW, sliceTh],
                                       [np.ceil(Mask.shape[0] * a), np.ceil(Mask.shape[1] * b),
                                        np.ceil(Mask.shape[2] * c)], ROIInterp, 'constant',
                                       [newVoxelSize, newVoxelSize, newVoxelSize], isIsot2D, use_disk=True)
            ROIwholeResmp = np.load(ROIwholeResmp_path, mmap_mode='r+')
            ImgWholeResmp = imresize3D(volume, [pixelW, pixelW, sliceTh],
                                       [np.ceil(volume.shape[0] * a), np.ceil(volume.shape[1] * b),
                                        np.ceil(volume.shape[2] * c)], VoxInterp, 'constant',
                                       [newVoxelSize, newVoxelSize, newVoxelSize], isIsot2D, use_disk=False)
            if np.max(ROIwholeResmp) < ROI_PV:
                import logging
                logging.getLogger("Dev_logger").warning("PREPARE_VOLUME- Resampled ROI has no voxels with value above ROI_PV. Cutting ROI_PV to half.")
                ROI_PV = ROI_PV / 2

    
            ROIwholeResmp[ROIwholeResmp < ROI_PV] = 0
            ROIwholeResmp[ROIwholeResmp >= ROI_PV] = 1

    elif Imgbox.ndim == 2 and flagPW != 0:
        if (a + b) != 2:
            ROIBoxResmp_path = imresize(ROIBox_path, [pixelW, pixelW],
                                   [np.ceil(ROIBox.shape[0] * a), np.ceil(ROIBox.shape[1] * b)], ROIInterp,
                                   [newVoxelSize, newVoxelSize], use_disk=True)
            ROIBoxResmp = np.load(ROIBoxResmp_path, mmap_mode='r+')
            ImgBoxResmp = imresize(Imgbox, [pixelW, pixelW],
                                   [np.ceil(Imgbox.shape[0] * a), np.ceil(Imgbox.shape[1] * b)], VoxInterp,
                                   [newVoxelSize, newVoxelSize], use_disk=False)
            ROIBoxResmp[ROIBoxResmp < ROI_PV] = 0
            ROIBoxResmp[ROIBoxResmp >= ROI_PV] = 1
            
            ROIwholeResmp_path = imresize(mask_input, [pixelW, pixelW], [np.ceil(Mask.shape[0] * a), np.ceil(Mask.shape[1] * b)],
                                     ROIInterp, [newVoxelSize, newVoxelSize], use_disk=True)
            ROIwholeResmp = np.load(ROIwholeResmp_path, mmap_mode='r+')
            ImgWholeResmp = imresize(volume, [pixelW, pixelW],
                                     [np.ceil(volume.shape[0] * a), np.ceil(volume.shape[1] * b)], VoxInterp,
                                     [newVoxelSize, newVoxelSize], use_disk=False)
            if np.max(ROIwholeResmp) < ROI_PV:
                import logging
                logging.getLogger("Dev_logger").warning("PREPARE_VOLUME- Resampled ROI has no voxels with value above ROI_PV. Cutting ROI_PV to half.")
                ROI_PV = ROI_PV / 2

            ROIwholeResmp[ROIwholeResmp < ROI_PV] = 0
            ROIwholeResmp[ROIwholeResmp >= ROI_PV] = 1
    else:
        ROIBoxResmp_path = None
        ROIwholeResmp_path = None

    IntsBoxROI = ImgBoxResmp.copy()

    ImgBoxResmp[ROIBoxResmp == 0] = np.nan

    IntsBoxROI = roundGL(ImgBoxResmp, isGLround)
    ImgWholeResmp = roundGL(ImgWholeResmp, isGLround)

    IntsBoxROItmp1 = IntsBoxROI.copy()
    ImgWholeResmptmp1 = ImgWholeResmp.copy()
    IntsBoxROItmp2 = IntsBoxROI.copy()
    ImgWholeResmptmp2 = ImgWholeResmp.copy()

    if isReSegRng == 1:
        IntsBoxROItmp1[IntsBoxROI < ReSegIntrvl[0]] = np.nan
        IntsBoxROItmp1[IntsBoxROI > ReSegIntrvl[1]] = np.nan
        ImgWholeResmptmp1[ImgWholeResmp < ReSegIntrvl[0]] = np.nan
        ImgWholeResmptmp1[ImgWholeResmp > ReSegIntrvl[1]] = np.nan

    if isOutliers == 1:
        Mu = np.nanmean(IntsBoxROI)
        Sigma = np.nanstd(IntsBoxROI)
        IntsBoxROItmp2[IntsBoxROI < (Mu - 3 * Sigma)] = np.nan
        IntsBoxROItmp2[IntsBoxROI > (Mu + 3 * Sigma)] = np.nan

        Mu = np.nanmean(ImgWholeResmp)
        Sigma = np.nanstd(ImgWholeResmp)
        ImgWholeResmptmp2[ImgWholeResmp < (Mu - 3 * Sigma)] = np.nan
        ImgWholeResmptmp2[ImgWholeResmp > (Mu + 3 * Sigma)] = np.nan

    IntsBoxROI = getMutualROI(IntsBoxROItmp1, IntsBoxROItmp2)
    ImgWholeResmp = getMutualROI(ImgWholeResmptmp1, ImgWholeResmptmp2)

    newpixelW = pixelW / a
    newsliceTh = sliceTh / c

    # Determine minimum GL value efficiently
    if DataType == 'PET':
        minGL = 0
    elif DataType == 'CT':
        minGL = ReSegIntrvl[0] if isReSegRng == 1 else np.nanmin(IntsBoxROI)
    else:
        minGL = np.nanmin(IntsBoxROI)

    # Perform quantization
    ImgBoxResampQuntz3D, levels = quantization(IntsBoxROI, Bin, minGL)

    # Calculate bounding box and crop efficiently

    boxBound = computeBoundingBox(ROIBoxResmp)
    MorphROI = ROIBoxResmp[boxBound[0][0]:boxBound[0][1], boxBound[1][0]:boxBound[1][1], boxBound[2][0]:boxBound[2][1]]
    IntsBoxROI = IntsBoxROI[boxBound[0][0]:boxBound[0][1], boxBound[1][0]:boxBound[1][1], boxBound[2][0]:boxBound[2][1]]
    ImgBoxResampQuntz3D = ImgBoxResampQuntz3D[boxBound[0][0]:boxBound[0][1], boxBound[1][0]:boxBound[1][1],
                          boxBound[2][0]:boxBound[2][1]]


    # Save numpy arrays on disk
    ROIwholeResmp_path2 = save_numpy_on_disk(ROIwholeResmp, prefix='final_ROIwholeResmp', suffix='.npy')
    MorphROI_path = save_numpy_on_disk(MorphROI, prefix='final_MorphROI', suffix='.npy')
    # Clean RAM
    del ROIBoxResmp, ROIwholeResmp, MorphROI, ROIBox, Mask
    gc.collect()
    # Clean disk
    if ROIBoxResmp_path and ROIwholeResmp_path:
        remove_temp_file(ROIBoxResmp_path)
        remove_temp_file(ROIwholeResmp_path)

    return ImgBoxResampQuntz3D, levels, MorphROI_path, IntsBoxROI, ImgWholeResmp, ROIwholeResmp_path2, newpixelW, newsliceTh


# -------------------------------------------------------------------------
# function [boxBound] = computeBoundingBox(mask)
# -------------------------------------------------------------------------
# DESCRIPTION:
# This function computes the smallest box containing the whole region of
# interest (ROI).
# -------------------------------------------------------------------------
# INPUTS:
# - mask: 3D array, with 1's inside the ROI, and 0's outside the ROI.
# -------------------------------------------------------------------------
# OUTPUTS:
# - boxBound: Bounds of the smallest box containing the ROI.
#             Format: [minRow, maxRow;
#                      minColumn, maxColumns;
#                      minSlice, maxSlice]
# -------------------------------------------------------------------------
# # AUTHOR(S):
# - Mahdi Hosseinzadeh
# -------------------------------------------------------------------------
# HISTORY:
# - Creation: December 2022
# --------------------------------------------------------------------------

def computeBoundingBox(Data_ROI_mat):
    # Load roi from disk        toto
    # Data_ROI_mat = np.load(Data_ROI_mat_path, mmap_mode='r')
    # Efficient and safe computation of bounding box for non-zero elements
    indices = np.where(Data_ROI_mat != 0)
    if len(indices[0]) == 0:
        return np.zeros((3, 2), dtype=np.uint32)

    boxBound = np.empty((3, 2), dtype=np.uint32)
    for dim in range(3):
        boxBound[dim, 0] = np.min(indices[dim])
        boxBound[dim, 1] = np.max(indices[dim]) + 1
    # # Clean RAM
    # del Data_ROI_mat
    # gc.collect()
    return boxBound


def CollewetNorm(ROIonly):
    temp = ROIonly[~np.isnan(ROIonly)]
    u = np.mean(temp)
    sigma = np.std(temp)

    ROIonlyNorm = ROIonly.copy()
    ROIonlyNorm[ROIonly > (u + 3 * sigma)] = np.nan
    ROIonlyNorm[ROIonly < (u - 3 * sigma)] = np.nan

    return ROIonlyNorm


# This function rounds image intensity voxels to the nearest integer.
def roundGL(Img, isGLrounding):
    """Memory-efficient rounding with in-place operations when possible"""
    if isGLrounding == 1:
        GLroundedImg = np.round(Img)
    else:
        GLroundedImg = Img.copy()

    return GLroundedImg


# This function receives two ROIs as a result of two resegmentation methods
# (range reseg. and outlier reseg.) and return a third ROI containing the
# mutual voxels inside both ROIs.
def getMutualROI(ROI1, ROI2):
    tmp1 = np.multiply(ROI1, ROI2)
    tmp2 = tmp1.copy()
    tmp2[~np.isnan(tmp1)] = 1
    outROI = np.multiply(tmp2, ROI1)

    return outROI


def getImgBox(volume, mask_input, isReSeg, ResegIntrval):
    # Load mask from disk
    mask = np.load(mask_input, mmap_mode='r')

    boxBound = computeBoundingBox(mask)
    # maskBox = mask[boxBound[0][0]:boxBound[0][1],boxBound[1][0]:boxBound[1][1],boxBound[2][0]:boxBound[2][1]]
    SUVbox = volume[boxBound[0][0]:boxBound[0][1], boxBound[1][0]:boxBound[1][1], boxBound[2][0]:boxBound[2][1]]

    if isReSeg == 1:
        SUVbox[SUVbox < ResegIntrval[0]] = np.nan
        SUVbox[SUVbox > ResegIntrval[1]] = np.nan

    # Clean RAM
    del mask
    gc.collect()

    return SUVbox



# -------------------------------------------------------------------------
# function [boxBound] = computeBoundingBox(mask)
# -------------------------------------------------------------------------
# DESCRIPTION:
# This function computes the smallest box containing the whole region of
# interest (ROI).
# -------------------------------------------------------------------------
# INPUTS:
# - mask: 3D array, with 1's inside the ROI, and 0's outside the ROI.
# -------------------------------------------------------------------------
# OUTPUTS:
# - boxBound: Bounds of the smallest box containing the ROI.
#             Format: [minRow, maxRow;
#                      minColumn, maxColumns;
#                      minSlice, maxSlice]
# -------------------------------------------------------------------------
# # AUTHOR(S):
# - Mahdi Hosseinzadeh
# -------------------------------------------------------------------------
# HISTORY:
# - Creation: December 2022
# --------------------------------------------------------------------------


def CollewetNorm(ROIonly):
    temp = ROIonly[~np.isnan(ROIonly)]
    u = np.mean(temp)
    sigma = np.std(temp)

    ROIonlyNorm = ROIonly.copy()
    ROIonlyNorm[ROIonly > (u + 3 * sigma)] = np.nan
    ROIonlyNorm[ROIonly < (u - 3 * sigma)] = np.nan

    return ROIonlyNorm



