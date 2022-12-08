from pickle import NONE
from scipy.io import savemat, loadmat
import numpy as np
import itertools
from getHist import getHist
from getStats import getStats
from getMorph import getMorph
from getIntVolHist import getIntVolHist
from getSUVpeak import getSUVpeak
from getGLRLMFeatures import getGLRLM2Dtex , getGLRLM3Dtex
from getGLCMFeatures import getGLCM2Dtex , getGLCM3Dtex
from getGLSZMFeatures import getGLSZMtex
from getGLDZMFeatures import getGLDZMtex
from getNGLDMFeatures import getNGLDMtex
from getNGTDMFeatures import getNGTDMtex
from getMIFeatures import getMI
from prepareVolume import prepareVolume,getImgBox
import collections.abc
import SimpleITK as sitk
import pandas as pd



# -------------------------------------------------------------------------
# [AllFeats] = SERA_FE_main_Fun(RawImg, ROI, VoxelSizeInfo, bin, DataType, 
#                               IsotVoxSize, IsotVoxSize2d, DiscType, qntzAlg,
#                               VoxInterp, ROIInterp, isScale, isGLrounding, 
#                               isReSegRng, isOutliers, isQuntzStat, isIsot2D, 
#                               ReSegIntrvl, ROI_PV, Feats2out, IVHconfig)    
# -------------------------------------------------------------------------
# DESCRIPTION: 
# This function prepares the input volume for 2D and 3D texture analysis.
# Then calculates various radiomics features, including: morphological,
# first-order: statistical, histogram, volume histogram; second-order: GLCM
# and GLRLM; and higher order: GLSZM, GLDZM, NGLDM, NGTDM.
# -------------------------------------------------------------------------
# INPUTS:
# - RawImg: the 2D or 3D matrix of intensities
# - ROI: the 2D or 3D matrix of the mask. Has to be the same size as RawImg
# - VoxelSizeInfo[0] (pixelW): Numerical value specifying the in-plane resolution(mm) of RawImg
# - VoxelSizeInfo[0] (sliceTh): Numerical value specifying the slice spacing (mm) of RawImg
#           Put a random number for 2D analysis.
# - IsotVoxSize: Numerical value specifying the new voxel size (mm) at
#                which the RIO is isotropically  resampled to in 3D.
# - IsotVoxSize2D: Numerical value specifying the new voxel size (mm) at
#                which the RIO is isotropically  resampled to in 3D. 
# - qntzAlg: String specifying the quantization algorithm to use on
#             'volume'. Either 'Lloyd' for Lloyd-Max quantization, or
#             'Uniform' for uniform quantization. 
# - bin: number of bins (for fixed number of bins method) or bin size
#        (for bin size method). It can be an array of bins, and the code
#        will loop over each bin. 
# - DiscType: discritization type. Either 'FNB' for fixed number of bins
#             or 'FBS' for fixed bin size. 
# - DataType: String specifying the type of scan analyzed. Either 'PET', 
#             'CT', or 'MRscan'.
# - VoxInterp: interpolation method for the intensity ROI
# - ROIInterp: interpolation method for the morphological ROI
# - ROI_PV: partial volume threshold for thresholding morphological ROI.
#           Used to threshold ROI after resampling: 
#           i.e. ROI(ROI<ROI_PV) =0, ROI(ROI>ROI_PV) = 1.  
# - isIsot2D: =1 if resampling only in X and Y dimensions.
# - isScale: whether to perform resampling
# - isGLrounding: whether to perform grey level rounding
# - isReSeg: whether to perform resegmentation
# - ReSegIntrvl: a 1x2 array of values expressing resegmentation interval. 
# - isQuntzStat: whether to use quantized image to calculate first order
#                statistical features. If 0, no image resample/interp for
#                calculating statistical features. (0 is preferrable for
#                PET images).
# - isOutliers: whether to perform instensity outlier filtering.
# - Feats2out: the option of which calculated radiomic features to return.
# - IVHconfig: the configuration of intensity volume histogram (IVH)
# -------------------------------------------------------------------------
# OUTPUTS: 
# - AllFeats: A vector (for a single bin) or a matrix (for multiple bins)
#             of calculated Radiomic features.
# -------------------------------------------------------------------------
# # AUTHOR(S): 
# - Saeed Ashrafinia
# - Mahdi Hosseinzadeh
# -------------------------------------------------------------------------
# HISTORY:
# - Creation: December 2022
# -------------------------------------------------------------------------

def SERA_FE_main_Fun(data_orginal,
                    Data_ROI_mat,
                    VoxelSizeInfo,
                    BinSize,
                    DataType,
                    isotVoxSize,
                    isotVoxSize2D,
                    DiscType,
                    qntz,
                    VoxInterp,
                    ROIInterp,
                    isScale,
                    isGLround,
                    isReSegRng,
                    isOutliers,
                    isQuntzStat,
                    isIsot2D,
                    ReSegIntrvl,
                    ROI_PV,
                    Feats2out,
                    IVHconfig): 

    # ROI_Box = computeBoundingBox(Data_ROI_mat)
    if isinstance(BinSize, collections.abc.Sequence):
        Lbin = BinSize.shape[0]
    else:
        Lbin = 1
        temp = BinSize
        BinSize = [temp]
    img = data_orginal.copy()

    img[Data_ROI_mat == 0] = np.nan
    

    if Feats2out == 1:
        FeaturesType = ['1st','2D','25D','3D']
    elif Feats2out == 2:
        FeaturesType = ['1st','3D','25D']
    elif Feats2out == 3:
        FeaturesType = ['1st','2D','25D']
    elif Feats2out == 4:
        FeaturesType = ['1st','3D','selected2D','25D']
    elif Feats2out == 5:
        FeaturesType = ['1st','2D','25D','3D','Moment']
    elif Feats2out == 6:
        FeaturesType = ['1st','2D','25D']
    elif Feats2out == 7:
        FeaturesType = ['1st','25D']
    elif Feats2out == 8:
        FeaturesType = ['1st']
    elif Feats2out == 9:
        FeaturesType = ['2D']
    elif Feats2out == 10:
        FeaturesType = ['25D']
    elif Feats2out == 11:
        FeaturesType = ['3D']
    elif Feats2out == 12:
        FeaturesType = ['Moment']


    MorphVect = []
    SUVpeak = []
    StatsVect  = []
    HistVect = []
    IVHvect = []

    GLCM2D_KSKD = []
    GLCM2D_KSMD = [] 
    GLCM2D_MSKD  = []
    GLCM2D_MSMD   = []
    GLCM3D_Avg  = []
    GLCM3D_Cmb  = []

    GLRLM2D_KSKD = [] 
    GLRLM2D_KSMD = [] 
    GLRLM2D_MSKD = [] 
    GLRLM2D_MSMD = [] 
    GLRLM3D_Avg = [] 
    GLRLM3D_Cmb = [] 

    GLSZM2D = [] 
    GLSZM25D = [] 
    GLSZM3D = [] 
    GLDZM2D = [] 
    GLDZM25D = [] 
    GLDZM3D = [] 

    NGTDM2D = [] 
    NGTDM25D = [] 
    NGTDM3D = [] 
    NGLDM2D = [] 
    NGLDM25D = [] 
    NGLDM3D = [] 
    MI_feats = []
    AllFeats = []

    MultiBin = 0
    pixelW = VoxelSizeInfo[0]
    sliceTh = VoxelSizeInfo[2]
    for m in range(0,Lbin):

        ROIBox3D,levels3D,ROIonlyMorph3D,IntsROI,RawInts,RawROI,newPixW,newSliceTh = prepareVolume(data_orginal,Data_ROI_mat,DataType,pixelW,sliceTh,isotVoxSize,VoxInterp,ROIInterp,ROI_PV,'XYZscale',isIsot2D,isScale,isGLround,DiscType,qntz,BinSize[m],isReSegRng,ReSegIntrvl,isOutliers)
        ImgBox = getImgBox(img,Data_ROI_mat,isReSegRng,ReSegIntrvl)
        
        if isIsot2D == 1 or pixelW == isotVoxSize or isScale == 0:
            ROIBox2D = ROIBox3D.copy()
            levels2D = levels3D.copy()
            ROIonly2D = ROIonlyMorph3D.copy()
        else:
            ROIBox2D,levels2D,ROIonly2D,_,_,_,_,_ = prepareVolume(data_orginal,Data_ROI_mat,DataType,pixelW,sliceTh,isotVoxSize2D,VoxInterp,ROIInterp,ROI_PV,'XYscale',0,isScale,isGLround,DiscType,qntz,BinSize[m],isReSegRng,ReSegIntrvl,isOutliers)

        if MultiBin == 0:
            for i in range(0,len(FeaturesType)):

                if FeaturesType[i] == '1st':
                    
                    ## Morphological features
                    MorphVect = getMorph(ROIBox3D,ROIonlyMorph3D,IntsROI, newPixW,newSliceTh)
                    ## SUVpeak (local intensity) calculation
                    SUVpeak = getSUVpeak(RawInts,RawROI,newPixW,newSliceTh)

                    ## Statistical features
                    # whether to use the above quantized image to calculate statistical
                    # features, or just use the raw image and original voxel size. 
                    # For PET images, set isQuntzStat=0.  
                    if isQuntzStat == 1:
                        StatsVect = getStats(IntsROI)                        
                    else:
                        StatsVect = getStats(ImgBox)
    
                    ## Histogram 
                    HistVect  = getHist(ROIBox3D,BinSize[m], DiscType)
                    ## Intensity Histogram features
                    IVHvect   = getIntVolHist(IntsROI,ROIBox3D,BinSize[m],isReSegRng,ReSegIntrvl,IVHconfig)


                elif FeaturesType[i] == '2D':
                    ## GLCM
                    GLCM2D_KSKD, GLCM2D_MSKD, GLCM2D_KSMD, GLCM2D_MSMD = getGLCM2Dtex(ROIBox2D,levels2D)
                    ## GLRLM
                    GLRLM2D_KSKD, GLRLM2D_KSMD, GLRLM2D_MSKD, GLRLM2D_MSMD = getGLRLM2Dtex(ROIBox2D,levels2D)
                
                elif FeaturesType[i] == 'selected2D':
                    ## GLCM
                    GLCM2D_KSKD, _, GLCM2D_KSMD, _ = getGLCM2Dtex(ROIBox2D,levels2D)
                    ## GLRLM
                    GLRLM2D_KSKD, GLRLM2D_KSMD, _, _ = getGLRLM2Dtex(ROIBox2D,levels2D)
                                        
                elif FeaturesType[i] == '3D':
                    ## GLCM
                    GLCM3D_Cmb, GLCM3D_Avg = getGLCM3Dtex(ROIBox3D,levels3D)
                    ## GLRLM             
                    GLRLM3D_Cmb, GLRLM3D_Avg = getGLRLM3Dtex(ROIBox3D,levels3D)

                elif FeaturesType[i] == '25D':
                    # GLSZM
                    GLSZM2D, GLSZM3D, GLSZM25D = getGLSZMtex(ROIBox2D,ROIBox3D,levels2D,levels3D)
                    # NGTDM
                    NGTDM2D, NGTDM3D, NGTDM25D = getNGTDMtex(ROIBox2D,ROIBox3D,levels2D,levels3D)
                    # NGLDM
                    NGLDM2D, NGLDM3D, NGLDM25D = getNGLDMtex(ROIBox2D,ROIBox3D,levels2D,levels3D)
                    # GLDZM
                    GLDZM2D, GLDZM3D, GLDZM25D = getGLDZMtex(ROIBox2D,ROIBox3D,ROIonly2D,ROIonlyMorph3D,levels2D,levels3D)


                elif FeaturesType[i] == 'Moment':
                    ## Moment Invariants
                    MI_feats = np.transpose(getMI(ImgBox))

        else:
            for i in range(0,len(FeaturesType)):
                if FeaturesType[i] == '1st':
   
                    ## Morphological features
                    MorphVect = getMorph(ROIBox3D,ROIonlyMorph3D,IntsROI, newPixW,newSliceTh)
                    ## SUVpeak (local intensity) calculation
                    SUVpeak = getSUVpeak(RawInts,RawROI,newPixW,newSliceTh)

                    ## Statistical features
                    # whether to use the above quantized image to calculate statistical
                    # features, or just use the raw image and original voxel size. 
                    # For PET images, set isQuntzStat=0.  
                    if isQuntzStat == 1:
                        StatsVect = getStats(IntsROI)                        
                    else:
                        StatsVect = getStats(ImgBox)
    
                    ## Histogram 
                    HistVect  = getHist(ROIBox3D,BinSize[m], DiscType)
                    ## Intensity Histogram features
                    IVHvect   = getIntVolHist(IntsROI,ROIBox3D,BinSize[m],isReSegRng,ReSegIntrvl,IVHconfig)


        MultiBin = 1


        

        if Feats2out == 2 or Feats2out == 11:
            GLSZM2D= []
            GLSZM25D= []
            NGTDM2D= []
            NGTDM25D= []
            NGLDM2D= []
            NGLDM25D= []
            GLDZM2D= []
            GLDZM25D= []  
        elif Feats2out == 3 or Feats2out == 6:
            GLSZM3D= []
            NGTDM3D= []
            NGLDM3D= []
            GLDZM3D= []
        elif Feats2out == 4:
            GLSZM25D= []
            NGTDM25D= []
            NGLDM25D= []
            GLDZM25D= []
        elif Feats2out == 7 or Feats2out == 10:
            GLSZM3D= []
            NGTDM3D= []
            NGLDM3D= []
            GLDZM3D= [] 
            GLSZM2D= []
            NGTDM2D= []
            NGLDM2D= []
            GLDZM2D= []    
        elif Feats2out == 9:
            GLSZM3D= []
            NGTDM3D= []
            NGLDM3D= []
            GLDZM3D= [] 
            GLSZM25D= []
            NGTDM25D= []
            NGLDM25D= []
            GLDZM25D= []


        Feats = list(itertools.chain(np.squeeze(MorphVect) , np.squeeze( SUVpeak) , np.squeeze( StatsVect) , np.squeeze( HistVect) , np.squeeze( IVHvect) , np.squeeze(
        GLCM2D_KSKD) , np.squeeze(  GLCM2D_KSMD) , np.squeeze(  GLCM2D_MSKD) , np.squeeze(  GLCM2D_MSMD) , np.squeeze(  GLCM3D_Avg) , np.squeeze(  GLCM3D_Cmb) , np.squeeze(
        GLRLM2D_KSKD) , np.squeeze( GLRLM2D_KSMD) , np.squeeze( GLRLM2D_MSKD) , np.squeeze( GLRLM2D_MSMD) , np.squeeze( GLRLM3D_Avg) , np.squeeze( GLRLM3D_Cmb) , np.squeeze(
        GLSZM2D) , np.squeeze( GLSZM25D) , np.squeeze( GLSZM3D) , np.squeeze( GLDZM2D) , np.squeeze( GLDZM25D) , np.squeeze( GLDZM3D) , np.squeeze(
        NGTDM2D) , np.squeeze( NGTDM25D) , np.squeeze( NGTDM3D) , np.squeeze( NGLDM2D) , np.squeeze( NGLDM25D) , np.squeeze( NGLDM3D) , np.squeeze(
        MI_feats)))


        AllFeats.append(Feats)

    return AllFeats
