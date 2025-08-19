import collections.abc
import collections.abc
import itertools
import gc

import numpy as np

from ..visera.getGLCMFeatures import getGLCM2Dtex, getGLCM3Dtex
from ..visera.getGLDZMFeatures import getGLDZMtex
from ..visera.getGLRLMFeatures import getGLRLM2Dtex, getGLRLM3Dtex
from ..visera.getGLSZMFeatures import getGLSZMtex
from ..visera.getHist import getHist
from ..visera.getIntVolHist import getIntVolHist
from ..visera.getMIFeatures import getMI
from ..visera.getMorph import getMorph
from ..visera.getNGLDMFeatures import getNGLDMtex
from ..visera.getNGTDMFeatures import getNGTDMtex
from ..visera.getSUVpeak import getSUVpeak
from ..visera.getStats import getStats
from ..visera.prepareVolume import prepareVolume, getImgBox
from ...utils.utils import remove_temp_file, save_numpy_on_disk


# -------------------------------------------------------------------------
# [AllFeats] = SERA_FE_main(RawImg, ROI, VoxelSizeInfo, bin, DataType,
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


def SERA_FE_main(
        data_orginal,
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
        IVHconfig,
        feature_value_mode,
        # Extra optional params for compatibility with _1 version
        Data_ROI_Name=None,
        IVH_Type=None,
        IVH_DiscCont=None,
        IVH_binSize=None,
        isROIsCombined=None
):

    if isinstance(BinSize, collections.abc.Sequence):
        Lbin = len(BinSize)
    else:
        Lbin = 1
        BinSize = [BinSize]

    # Load mask from disk
    Data_ROI_mat_disk = np.load(Data_ROI_mat, mmap_mode='r')
    img = data_orginal.copy()
    img[Data_ROI_mat_disk == 0] = np.nan

    # Clean RAM
    del Data_ROI_mat_disk
    gc.collect()

    # Determine feature types based on Feats2out
    feat_map = {
        1: ['1st', '2D', '25D', '3D'],
        2: ['1st', '3D', '25D'],
        3: ['1st', '2D', '25D'],
        4: ['1st', '3D', 'selected2D', '25D'],
        5: ['1st', '2D', '25D', '3D', 'Moment'],
        6: ['1st', '2D', '25D'],
        7: ['1st', '25D'],
        8: ['1st'],
        9: ['2D'],
        10: ['25D'],
        11: ['3D'],
        12: ['Moment']
    }
    FeaturesType = feat_map.get(Feats2out, [])

    AllFeats = []
    pixelW = VoxelSizeInfo[0]
    sliceTh = VoxelSizeInfo[2]
    MultiBin = 0

    for m in range(Lbin):
        # arr       arr         path               arr      arr     path        float       float       <= toto notes
        ROIBox3D, levels3D, ROIonlyMorph3D_path, IntsROI, RawInts, RawROI_path, newPixW, newSliceTh = prepareVolume(
            data_orginal, Data_ROI_mat, DataType, pixelW, sliceTh, isotVoxSize,
            VoxInterp, ROIInterp, ROI_PV, 'XYZscale', isIsot2D, isScale,
            isGLround, DiscType, qntz, BinSize[m], isReSegRng, ReSegIntrvl, isOutliers
        )

        ImgBox = getImgBox(img, Data_ROI_mat, isReSegRng, ReSegIntrvl)

        if isIsot2D == 1 or pixelW == isotVoxSize or isScale == 0:
            ROIBox2D = ROIBox3D.copy()
            levels2D = levels3D.copy()
            ROIonly2D_path = str(ROIonlyMorph3D_path)
        else:
            ROIBox2D, levels2D, ROIonly2D_path, *_ = prepareVolume(
                data_orginal, Data_ROI_mat, DataType, pixelW, sliceTh, isotVoxSize2D,
                VoxInterp, ROIInterp, ROI_PV, 'XYscale', 0, isScale,
                isGLround, DiscType, qntz, BinSize[m], isReSegRng, ReSegIntrvl, isOutliers
            )

        # Initialize empty feature lists
        MorphVect, SUVpeak, StatsVect, HistVect, IVHvect = [], [], [], [], []
        GLCM2D_KSKD, GLCM2D_KSMD, GLCM2D_MSKD, GLCM2D_MSMD = [], [], [], []
        GLCM3D_Avg, GLCM3D_Cmb = [], []
        GLRLM2D_KSKD, GLRLM2D_KSMD, GLRLM2D_MSKD, GLRLM2D_MSMD = [], [], [], []
        GLRLM3D_Avg, GLRLM3D_Cmb = [], []
        GLSZM2D, GLSZM25D, GLSZM3D = [], [], []
        GLDZM2D, GLDZM25D, GLDZM3D = [], [], []
        NGTDM2D, NGTDM25D, NGTDM3D = [], [], []
        NGLDM2D, NGLDM25D, NGLDM3D = [], [], []
        MI_feats = []

        if MultiBin == 0:
            for ft in FeaturesType:
                if ft == '1st':
                    MorphVect = getMorph(ROIBox3D, np.load(ROIonlyMorph3D_path), IntsROI, newPixW, newSliceTh,
                                         feature_value_mode=feature_value_mode)
                    SUVpeak = getSUVpeak(RawInts, RawROI_path, newPixW, newSliceTh)
                    StatsVect = getStats(IntsROI if isQuntzStat else ImgBox, feature_value_mode=feature_value_mode)
                    HistVect = getHist(ROIBox3D, BinSize[m], DiscType, feature_value_mode=feature_value_mode)
                    IVHvect = getIntVolHist(IntsROI, ROIBox3D, BinSize[m], isReSegRng, ReSegIntrvl, IVHconfig,
                                            feature_value_mode=feature_value_mode)

                elif ft == '2D':
                    GLCM2D_KSKD, GLCM2D_MSKD, GLCM2D_KSMD, GLCM2D_MSMD = getGLCM2Dtex(ROIBox2D, levels2D,
                                                                                      feature_value_mode=feature_value_mode)
                    GLRLM2D_KSKD, GLRLM2D_KSMD, GLRLM2D_MSKD, GLRLM2D_MSMD = getGLRLM2Dtex(ROIBox2D, levels2D,
                                                                                           feature_value_mode=feature_value_mode)

                elif ft == 'selected2D':
                    GLCM2D_KSKD, _, GLCM2D_KSMD, _ = getGLCM2Dtex(ROIBox2D, levels2D, feature_value_mode=feature_value_mode)
                    GLRLM2D_KSKD, GLRLM2D_KSMD, _, _ = getGLRLM2Dtex(ROIBox2D, levels2D, feature_value_mode=feature_value_mode)

                elif ft == '3D':
                    GLCM3D_Cmb, GLCM3D_Avg = getGLCM3Dtex(ROIBox3D, levels3D, feature_value_mode=feature_value_mode)
                    GLRLM3D_Cmb, GLRLM3D_Avg = getGLRLM3Dtex(ROIBox3D, levels3D, feature_value_mode=feature_value_mode)

                elif ft == '25D':
                    GLSZM2D, GLSZM3D, GLSZM25D = getGLSZMtex(ROIBox2D, ROIBox3D, levels2D, levels3D,
                                                             feature_value_mode=feature_value_mode)
                    NGTDM2D, NGTDM3D, NGTDM25D = getNGTDMtex(ROIBox2D, ROIBox3D, levels2D, levels3D,
                                                             feature_value_mode=feature_value_mode)
                    NGLDM2D, NGLDM3D, NGLDM25D = getNGLDMtex(ROIBox2D, ROIBox3D, levels2D, levels3D,
                                                             feature_value_mode=feature_value_mode)
                    GLDZM2D, GLDZM3D, GLDZM25D = getGLDZMtex(ROIBox2D, ROIBox3D, np.load(ROIonly2D_path),
                                                             np.load(ROIonlyMorph3D_path), levels2D, levels3D,
                                                             feature_value_mode=feature_value_mode)

                elif ft == 'Moment':
                    MI_feats = np.transpose(getMI(ImgBox, feature_value_mode=feature_value_mode))

        MultiBin = 1

        # Remove unwanted features based on Feats2out
        if Feats2out in [2, 11]:
            GLSZM2D, GLSZM25D, NGTDM2D, NGTDM25D, NGLDM2D, NGLDM25D, GLDZM2D, GLDZM25D = [], [], [], [], [], [], [], []
        elif Feats2out in [3, 6]:
            GLSZM3D, NGTDM3D, NGLDM3D, GLDZM3D = [], [], [], []
        elif Feats2out == 4:
            GLSZM25D, NGTDM25D, NGLDM25D, GLDZM25D = [], [], [], []
        elif Feats2out in [7, 10]:
            GLSZM3D, NGTDM3D, NGLDM3D, GLDZM3D = [], [], [], []
            GLSZM2D, NGTDM2D, NGLDM2D, GLDZM2D = [], [], [], []
        elif Feats2out == 9:
            GLSZM3D, NGTDM3D, NGLDM3D, GLDZM3D = [], [], [], []
            GLSZM25D, NGTDM25D, NGLDM25D, GLDZM25D = [], [], [], []

        Feats = list(itertools.chain(
            np.squeeze(MorphVect), np.squeeze(SUVpeak), np.squeeze(StatsVect), np.squeeze(HistVect), np.squeeze(IVHvect),
            np.squeeze(GLCM2D_KSKD), np.squeeze(GLCM2D_KSMD), np.squeeze(GLCM2D_MSKD), np.squeeze(GLCM2D_MSMD),
            np.squeeze(GLCM3D_Avg), np.squeeze(GLCM3D_Cmb),
            np.squeeze(GLRLM2D_KSKD), np.squeeze(GLRLM2D_KSMD), np.squeeze(GLRLM2D_MSKD), np.squeeze(GLRLM2D_MSMD),
            np.squeeze(GLRLM3D_Avg), np.squeeze(GLRLM3D_Cmb),
            np.squeeze(GLSZM2D), np.squeeze(GLSZM25D), np.squeeze(GLSZM3D),
            np.squeeze(GLDZM2D), np.squeeze(GLDZM25D), np.squeeze(GLDZM3D),
            np.squeeze(NGTDM2D), np.squeeze(NGTDM25D), np.squeeze(NGTDM3D),
            np.squeeze(NGLDM2D), np.squeeze(NGLDM25D), np.squeeze(NGLDM3D),
            np.squeeze(MI_feats)
        ))

        AllFeats.append(Feats)
    
    return AllFeats
