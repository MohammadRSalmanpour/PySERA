from pickle import NONE
from scipy.io import savemat, loadmat
import numpy as np
from Quantization import fixedBinSizeQuantization,uniformQuantization,lloydQuantization
from interpolation import imresize3D,imresize

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

def prepareVolume(volume,Mask,DataType,pixelW,sliceTh,
                newVoxelSize,VoxInterp,ROIInterp,ROI_PV,scaleType,isIsot2D,
                isScale,isGLround,DiscType,qntz,Bin,
                isReSegRng,ReSegIntrvl,isOutliers):

    if DiscType == 'FBS':
        quantization = fixedBinSizeQuantization
    elif DiscType == 'FBN':
        quantization = uniformQuantization
    else:
        print('Error with discretization type. Must either be "FBS" (Fixed Bin Size) or "FBN" (Fixed Number of Bins).')


    if qntz == 'Lloyd':
        quantization = lloydQuantization


    ROIBox = Mask.copy()
    Imgbox = volume.copy()

    Imgbox = Imgbox.astype(np.float32)

    if DataType == 'MRscan':
        ROIonly = Imgbox.copy()
        ROIonly[ROIBox == 0] = np.nan
        temp = CollewetNorm(ROIonly)
        ROIBox[np.isnan(temp)] = 0

    flagPW = 0
    if scaleType=='NoRescale':
        flagPW = 0
    elif scaleType=='XYZscale':
        flagPW = 1
    elif scaleType=='XYscale':
        flagPW = 2
    elif scaleType=='Zscale':
        flagPW = 3
    

    if isIsot2D == 1:
        flagPW = 2
    

    if isScale == 0:
        flagPW = 0
    

    if flagPW == 0:
        a = 1
        b = 1 
        c = 1
    elif flagPW ==1:
        a = pixelW/newVoxelSize
        b = pixelW/newVoxelSize
        c = sliceTh/newVoxelSize
    elif flagPW == 2:
        a = pixelW/newVoxelSize
        b = pixelW/newVoxelSize
        c = 1
    elif flagPW == 3:
        a = 1
        b = 1
        c = sliceTh/pixelW
    
    # Resampling
    ImgBoxResmp = Imgbox.copy() 
    ImgWholeResmp = volume.copy()
    ROIBoxResmp = ROIBox.copy() 
    ROIwholeResmp = Mask.copy()


    if Imgbox.ndim == 3 and flagPW != 0:
        if (a + b + c) != 3:
            ROIBoxResmp = imresize3D(ROIBox,[pixelW, pixelW, sliceTh],[np.ceil(ROIBox.shape[0]*a),np.ceil(ROIBox.shape[1]*b),np.ceil(ROIBox.shape[2]*c)],ROIInterp ,'constant',[newVoxelSize,newVoxelSize,newVoxelSize],isIsot2D)
            Imgbox[np.isnan(Imgbox)] = 0
            ImgBoxResmp = imresize3D(Imgbox,[pixelW, pixelW, sliceTh],[np.ceil(Imgbox.shape[0]*a),np.ceil(Imgbox.shape[1]*b),np.ceil(Imgbox.shape[2]*c)],VoxInterp,'constant',[newVoxelSize,newVoxelSize,newVoxelSize],isIsot2D)
            ROIBoxResmp[ROIBoxResmp < ROI_PV] = 0
            ROIBoxResmp[ROIBoxResmp >= ROI_PV] = 1 
            ROIwholeResmp = imresize3D(Mask,[pixelW, pixelW, sliceTh],[np.ceil(Mask.shape[0]*a),np.ceil(Mask.shape[1]*b),np.ceil(Mask.shape[2]*c)],ROIInterp,'constant',[newVoxelSize,newVoxelSize,newVoxelSize],isIsot2D)
            ImgWholeResmp = imresize3D(volume,[pixelW, pixelW, sliceTh],[np.ceil(volume.shape[0]*a),np.ceil(volume.shape[1]*b),np.ceil(volume.shape[2]*c)],VoxInterp ,'constant',[newVoxelSize,newVoxelSize,newVoxelSize],isIsot2D)
            if np.max(ROIwholeResmp) < ROI_PV:
                print('Resampled ROI has no voxels with value above ROI_PV. Cutting ROI_PV to half.')
                ROI_PV = ROI_PV/2
            
            ROIwholeResmp[ROIwholeResmp<ROI_PV] = 0
            ROIwholeResmp[ROIwholeResmp>=ROI_PV] = 1

    elif Imgbox.ndim == 2 and flagPW !=0:
        if (a + b) != 2:
            ROIBoxResmp = imresize(ROIBox,[pixelW, pixelW],[np.ceil(ROIBox.shape[0]*a),np.ceil(ROIBox.shape[1]*b)],ROIInterp,[newVoxelSize,newVoxelSize])
            ImgBoxResmp = imresize(Imgbox,[pixelW, pixelW],[np.ceil(Imgbox.shape[0]*a),np.ceil(Imgbox.shape[1]*b)],VoxInterp,[newVoxelSize,newVoxelSize])
            ROIBoxResmp[ROIBoxResmp<ROI_PV] = 0
            ROIBoxResmp[ROIBoxResmp>=ROI_PV] = 1

            ROIwholeResmp = imresize(Mask,[pixelW, pixelW],[np.ceil(Mask.shape[0]*a),np.ceil(Mask.shape[1]*b)],ROIInterp,[newVoxelSize,newVoxelSize])
            ImgWholeResmp = imresize(volume,[pixelW, pixelW],[np.ceil(volume.shape[0]*a),np.ceil(volume.shape[1]*b)],VoxInterp,[newVoxelSize,newVoxelSize])
            if np.max(ROIwholeResmp) < ROI_PV:
                print('Resampled ROI has no voxels with value above ROI_PV. Cutting ROI_PV to half.')
                ROI_PV = ROI_PV/2
            
            ROIwholeResmp[ROIwholeResmp<ROI_PV] = 0
            ROIwholeResmp[ROIwholeResmp>=ROI_PV] = 1
    

    

    IntsBoxROI = ImgBoxResmp.copy()

    ImgBoxResmp[ROIBoxResmp == 0] = np.nan

    IntsBoxROI = roundGL(ImgBoxResmp , isGLround)
    ImgWholeResmp = roundGL(ImgWholeResmp , isGLround)

    
    IntsBoxROItmp1 = IntsBoxROI.copy()
    ImgWholeResmptmp1 = ImgWholeResmp.copy()
    IntsBoxROItmp2 = IntsBoxROI.copy()
    ImgWholeResmptmp2 = ImgWholeResmp.copy()

    if isReSegRng == 1:
        IntsBoxROItmp1[IntsBoxROI<ReSegIntrvl[0]] = np.nan
        IntsBoxROItmp1[IntsBoxROI>ReSegIntrvl[1]] = np.nan
        ImgWholeResmptmp1[ImgWholeResmp<ReSegIntrvl[0]] = np.nan
        ImgWholeResmptmp1[ImgWholeResmp>ReSegIntrvl[1]] = np.nan
    

    if isOutliers == 1:
        Mu = np.nanmean(IntsBoxROI)
        Sigma = np.nanstd(IntsBoxROI)
        IntsBoxROItmp2[IntsBoxROI<(Mu-3*Sigma)] = np.nan
        IntsBoxROItmp2[IntsBoxROI>(Mu+3*Sigma)] = np.nan
            
        Mu = np.nanmean(ImgWholeResmp)
        Sigma = np.nanstd(ImgWholeResmp)
        ImgWholeResmptmp2[ImgWholeResmp<(Mu-3*Sigma)] = np.nan
        ImgWholeResmptmp2[ImgWholeResmp>(Mu+3*Sigma)] = np.nan
    

    IntsBoxROI      = getMutualROI(IntsBoxROItmp1 , IntsBoxROItmp2)
    ImgWholeResmp   = getMutualROI(ImgWholeResmptmp1 , ImgWholeResmptmp2)


    newpixelW = pixelW / a
    newsliceTh = sliceTh / c


    if DataType ==  'PET':
        minGL = 0
    elif DataType == 'CT':
        if isReSegRng== 1:
            minGL = ReSegIntrvl[0]
        else:
            minGL = np.nanmin(IntsBoxROI)
        
    else:
        minGL = np.nanmin(IntsBoxROI)
    

    ImgBoxResampQuntz3D,levels = quantization(IntsBoxROI,Bin,minGL)



    boxBound = computeBoundingBox(ROIBoxResmp)
    MorphROI = ROIBoxResmp[boxBound[0][0]:boxBound[0][1],boxBound[1][0]:boxBound[1][1],boxBound[2][0]:boxBound[2][1]]
    IntsBoxROI = IntsBoxROI[boxBound[0][0]:boxBound[0][1],boxBound[1][0]:boxBound[1][1],boxBound[2][0]:boxBound[2][1]]
    ImgBoxResampQuntz3D = ImgBoxResampQuntz3D[boxBound[0][0]:boxBound[0][1],boxBound[1][0]:boxBound[1][1],boxBound[2][0]:boxBound[2][1]]
    # ImgWholeResmp = ImgWholeResmp[boxBound[0][0]:boxBound[0][1],boxBound[1][0]:boxBound[1][1],boxBound[2][0]:boxBound[2][1]]

    return ImgBoxResampQuntz3D,levels,MorphROI,IntsBoxROI,ImgWholeResmp,ROIwholeResmp,newpixelW,newsliceTh


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
#--------------------------------------------------------------------------

def computeBoundingBox(Data_ROI_mat):
    [iV,jV,kV] = np.nonzero(Data_ROI_mat)
    boxBound = np.zeros((3,2))
    boxBound[0,0] = np.min(iV)
    boxBound[0,1] = np.max(iV)+1
    boxBound[1,0] = np.min(jV)
    boxBound[1,1] = np.max(jV)+1
    boxBound[2,0] = np.min(kV)
    boxBound[2,1] = np.max(kV)+1
    boxBound = boxBound.astype(np.uint32)


    # x_ind, y_ind, z_ind = np.where(Data_ROI_mat > 0.0)

    # if len(z_ind) == 0 or len(y_ind) == 0 or len(x_ind) == 0:
    #     boxBound = [[0,0],[0,0],[0,0]]
    # else:
    #     boxBound = [[np.min(x_ind), np.max(x_ind)],
    #                 [np.min(y_ind), np.max(y_ind)],
    #                 [np.min(z_ind), np.max(z_ind)]]




    # boxBound = np.asarray(boxBound)                    
    # boxBound = boxBound.astype(np.uint32)

    return boxBound


def CollewetNorm(ROIonly):


    temp = ROIonly[~np.isnan(ROIonly)]
    u = np.mean(temp)
    sigma = np.std(temp)

    ROIonlyNorm = ROIonly.copy()
    ROIonlyNorm[ROIonly > (u + 3*sigma)] = np.nan
    ROIonlyNorm[ROIonly < (u - 3*sigma)] = np.nan

    return ROIonlyNorm


# This function rounds image intensity voxels to the nearest integer.
def roundGL(Img , isGLrounding):

    if isGLrounding == 1:
        GLroundedImg = np.round(Img)
    else:
        GLroundedImg = Img.copy()
     
    return GLroundedImg


# This function receives two ROIs as a result of two resegmentation methods
# (range reseg. and outlier reseg.) and return a third ROI containing the
# mutual voxels inside both ROIs. 
def getMutualROI(ROI1, ROI2):

    tmp1 = np.multiply(ROI1 , ROI2)
    tmp2=tmp1.copy()
    tmp2[~np.isnan(tmp1)] = 1
    outROI = np.multiply(tmp2 , ROI1)


    return outROI

def getImgBox(volume,mask,isReSeg,ResegIntrval):
    
    boxBound = computeBoundingBox(mask)
    # maskBox = mask[boxBound[0][0]:boxBound[0][1],boxBound[1][0]:boxBound[1][1],boxBound[2][0]:boxBound[2][1]]
    SUVbox = volume[boxBound[0][0]:boxBound[0][1],boxBound[1][0]:boxBound[1][1],boxBound[2][0]:boxBound[2][1]]

    if isReSeg == 1:
        SUVbox[SUVbox<ResegIntrval[0]] = np.nan
        SUVbox[SUVbox>ResegIntrval[1]] = np.nan

    return SUVbox
