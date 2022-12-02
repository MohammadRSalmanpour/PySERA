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


def SERA_FE_main_Fun(data_orginal,
                    Data_ROI_mat,
                    Data_ROI_Name,
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
                    IVH_Type,
                    IVH_DiscCont,
                    IVH_binSize,
                    ROIsPerImg,
                    isROIsCombined,
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
        # print('resampling')
        # print(time.perf_counter())
        ROIBox3D,levels3D,ROIonlyMorph3D,IntsROI,RawInts,RawROI,newPixW,newSliceTh = prepareVolume(data_orginal,Data_ROI_mat,DataType,pixelW,sliceTh,isotVoxSize,VoxInterp,ROIInterp,ROI_PV,'XYZscale',isIsot2D,isScale,isGLround,DiscType,qntz,BinSize[m],isReSegRng,ReSegIntrvl,isOutliers)
        ImgBox = getImgBox(img,Data_ROI_mat,isReSegRng,ReSegIntrvl)
        
        if isIsot2D == 1 or pixelW == isotVoxSize or isScale == 0:
            ROIBox2D = ROIBox3D.copy()
            levels2D = levels3D.copy()
            ROIonly2D = ROIonlyMorph3D.copy()
        else:
            ROIBox2D,levels2D,ROIonly2D,_,_,_,_,_ = prepareVolume(data_orginal,Data_ROI_mat,DataType,pixelW,sliceTh,isotVoxSize2D,VoxInterp,ROIInterp,ROI_PV,'XYscale',0,isScale,isGLround,DiscType,qntz,BinSize[m],isReSegRng,ReSegIntrvl,isOutliers)

        # flag25D = False
        # flag2D3D = False
        if MultiBin == 0:
            for i in range(0,len(FeaturesType)):

                if FeaturesType[i] == '1st':
                    

                    MorphVect = getMorph(ROIBox3D,ROIonlyMorph3D,IntsROI, newPixW,newSliceTh)
   
                    SUVpeak = getSUVpeak(RawInts,RawROI,newPixW,newSliceTh)
    
                    if isQuntzStat == 1:
                        StatsVect = getStats(IntsROI)                        
                    else:
                        StatsVect = getStats(ImgBox)
    
                    HistVect  = getHist(ROIBox3D,BinSize[m], DiscType)

                    IVHvect   = getIntVolHist(IntsROI,ROIBox3D,BinSize[m],isReSegRng,ReSegIntrvl,IVHconfig)


                elif FeaturesType[i] == '2D':
                    # flag2D3D = True
                    # print('getGLCM2Dtex')
                    # print(time.perf_counter())
                    GLCM2D_KSKD, GLCM2D_MSKD, GLCM2D_KSMD, GLCM2D_MSMD = getGLCM2Dtex(ROIBox2D,levels2D)
                    # print('getGLRLM2Dtex')
                    # print(time.perf_counter())
                    GLRLM2D_KSKD, GLRLM2D_KSMD, GLRLM2D_MSKD, GLRLM2D_MSMD = getGLRLM2Dtex(ROIBox2D,levels2D)
                
                elif FeaturesType[i] == 'selected2D':
                    # flag2D3D = True
                    # print('getGLCM2Dtex')
                    # print(time.perf_counter())
                    GLCM2D_KSKD, _, GLCM2D_KSMD, _ = getGLCM2Dtex(ROIBox2D,levels2D)
                    
                    # print('getGLRLM2Dtex')
                    # print(time.perf_counter())
                    GLRLM2D_KSKD, GLRLM2D_KSMD, _, _ = getGLRLM2Dtex(ROIBox2D,levels2D)
                                        
                elif FeaturesType[i] == '3D':
                    
                    # flag2D3D = True
                    # print('getGLCM3Dtex')
                    # print(time.perf_counter())
                    GLCM3D_Cmb, GLCM3D_Avg = getGLCM3Dtex(ROIBox3D,levels3D)
                    
                    # print('getGLRLM3Dtex')
                    # print(time.perf_counter())                    
                    GLRLM3D_Cmb, GLRLM3D_Avg = getGLRLM3Dtex(ROIBox3D,levels3D)

                elif FeaturesType[i] == '25D':
                    # flag25D = True
                    # print('getGLSZMtex')
                    # print(time.perf_counter())                    
                    GLSZM2D, GLSZM3D, GLSZM25D = getGLSZMtex(ROIBox2D,ROIBox3D,levels2D,levels3D)
                    # print('getNGTDMtex')
                    # print(time.perf_counter())                    
                    NGTDM2D, NGTDM3D, NGTDM25D = getNGTDMtex(ROIBox2D,ROIBox3D,levels2D,levels3D)
                    # print('getNGLDMtex')
                    # print(time.perf_counter())                    
                    NGLDM2D, NGLDM3D, NGLDM25D = getNGLDMtex(ROIBox2D,ROIBox3D,levels2D,levels3D)
                    # print('getGLDZMtex')
                    # print(time.perf_counter())                    
                    GLDZM2D, GLDZM3D, GLDZM25D = getGLDZMtex(ROIBox2D,ROIBox3D,ROIonly2D,ROIonlyMorph3D,levels2D,levels3D)


                elif FeaturesType[i] == 'Moment':
                    MI_feats = np.transpose(getMI(ImgBox))

            # if flag25D == False and flag2D3D == True:
            #     # print('getGLSZMtex')
            #     # print(time.perf_counter())
            #     GLSZM2D, GLSZM3D, GLSZM25D = getGLSZMtex(ROIBox2D,ROIBox3D,levels2D,levels3D)
            #     # print('getNGTDMtex')
            #     # print(time.perf_counter())        
            #     NGTDM2D, NGTDM3D, NGTDM25D = getNGTDMtex(ROIBox2D,ROIBox3D,levels2D,levels3D)
            #     # print('getNGLDMtex')
            #     # print(time.perf_counter())                
            #     NGLDM2D, NGLDM3D, NGLDM25D = getNGLDMtex(ROIBox2D,ROIBox3D,levels2D,levels3D)
            #     # print('getGLDZMtex')
            #     # print(time.perf_counter())                
            #     GLDZM2D, GLDZM3D, GLDZM25D = getGLDZMtex(ROIBox2D,ROIBox3D,ROIonly2D,ROIonlyMorph3D,levels2D,levels3D)
            
            
        else:
            for i in range(0,len(FeaturesType)):
                if FeaturesType[i] == '1st':
   
                    MorphVect = getMorph(ROIBox3D,ROIonlyMorph3D,IntsROI, newPixW,newSliceTh)
      

                    SUVpeak = getSUVpeak(RawInts,RawROI,newPixW,newSliceTh)

                 
                    if isQuntzStat == 1:
                        StatsVect = getStats(IntsROI)
                    else:
                        StatsVect = getStats(ImgBox)
                    
                  
                    HistVect  = getHist(ROIBox3D,BinSize[m], DiscType)
                 
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



import timeit

start = timeit.default_timer()


data_orgina = sitk.ReadImage(r'..\Data\ibsi_1_ct_radiomics_phantom\nifti\image\phantom.nii.gz')
Data_RO = sitk.ReadImage(r'..\Data\ibsi_1_ct_radiomics_phantom\nifti\mask\mask.nii.gz')


VoxelSizeInfo = np.array(data_orgina.GetSpacing())
VoxelSizeInfo = VoxelSizeInfo.astype(np.float32)

# VoxelSizeInfo = [0.977,0.977,3]
# VoxelSizeInfo = [2,2,2]

data_orginal = sitk.GetArrayFromImage(data_orgina)
data_orginal = np.transpose(data_orginal,(2,1,0))
data_orginal = data_orginal.astype(np.float32)


Data_ROI = sitk.GetArrayFromImage(Data_RO)
Data_ROI = np.transpose(Data_ROI,(2,1,0))
Data_ROI = Data_ROI.astype(np.float16)
Data_ROI_mat = Data_ROI
Data_ROI_Name = '1_1'


isScale = 1
isGLround = 1
ROI_PV = 0.5
VoxInterp = 'linear'
ROIInterp = 'linear'
isotVoxSize = 2.0
isotVoxSize2D = 2.0
isIsot2D = 0

isReSegRng = 1
isOutliers = 0
ReSegIntrvl = [-1000,400]

DiscType = 'FBS'
BinSize = 25
qntz = 'Uniform'
isQuntzStat = 1

DataType = 'CT'
ROIsPerImg = 1
isROIsCombined = 0
Feats2out = 2

IVH_Type = 3
IVH_DiscCont = 1
IVH_binSize = 2.5
IVHconfig = [IVH_Type, IVH_DiscCont ,IVH_binSize]


AllFeats = SERA_FE_main_Fun(data_orginal,
                Data_ROI_mat,
                Data_ROI_Name,
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
                IVH_Type,
                IVH_DiscCont,
                IVH_binSize,
                ROIsPerImg,
                isROIsCombined,
                Feats2out,
                IVHconfig)


AllFeats = np.array(AllFeats)

FE = pd.DataFrame(AllFeats)

CSVfullpath = r"Extracted_features.xlsx"
writer = pd.ExcelWriter(CSVfullpath, engine='xlsxwriter') # pylint: disable=abstract-class-instantiated
FE.to_excel(writer, sheet_name='Extracted_features',index=None)
writer.save() 

stop = timeit.default_timer()

print('Time: ', stop - start)  