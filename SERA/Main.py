from pickle import NONE
from scipy.io import savemat, loadmat
import numpy as np
import itertools
import collections.abc
import SimpleITK as sitk
import pandas as pd
import timeit
from RF_main import SERA_FE_main_Fun


# -------------------------------------------------------------------------
# Standardized Environment for Radiomics Analysis (SERA)
# Main code for calculating radiomics features based on protocols by IBSI 
# -------------------------------------------------------------------------
# DESCRIPTION: 
# This program loads an input volume and its associated ROI for 2D and 3D
# radiomics analysis. 

# The framework first performs image pre-processing, and then calculates various
# radiomics features, including: first-order: morphological, statistical,
# histogram, volume histogram; second-order: GLCM  and GLRLM; and higher
# order: GLSZM, GLDZM, NGLDM, NGTDM, as well as moment invarient based on
# guidelines from Image Biomarker Standardization Initiative guideline
# https://arxiv.org/pdf/1612.07003.pdf 
# -------------------------------------------------------------------------
# INPUTS:
# Inputs are located in the first two sections of the code: section 1:
# "Selecting the dataset" and section 2: "Radiomics Framework Settings".
# Make sure to set and check every variable inside these two sections. 
# -------------------------------------------------------------------------
# OUTPUTS: 
# - AllFeats: A matrix of calculated Radiomic features
# - and Extracted_features.xlsx
# -------------------------------------------------------------------------
# # AUTHOR(S): 
# - Saeed Ashrafinia 
# - Mahdi Hosseinzadeh 
# - Mohammad Salmanpoor 
# - Arman Rahmim 
# -------------------------------------------------------------------------
# HISTORY:
# - Creation: December 2022
# -------------------------------------------------------------------------

start = timeit.default_timer()


# section 1: "Selecting the dataset"
data_orgina = sitk.ReadImage(r'D:\data\IBSI1 Data\ibsi_1_ct_radiomics_phantom\nifti\image\phantom.nii.gz')
Data_RO = sitk.ReadImage(r'D:\data\IBSI1 Data\ibsi_1_ct_radiomics_phantom\nifti\mask\mask.nii.gz')
# end section 1

# section 2: "Radiomics Framework Settings"
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
# end section 2



VoxelSizeInfo = np.array(data_orgina.GetSpacing())
VoxelSizeInfo = VoxelSizeInfo.astype(np.float32)

data_orginal = sitk.GetArrayFromImage(data_orgina)
data_orginal = np.transpose(data_orginal,(2,1,0))
data_orginal = data_orginal.astype(np.float32)


Data_ROI = sitk.GetArrayFromImage(Data_RO)
Data_ROI = np.transpose(Data_ROI,(2,1,0))
Data_ROI = Data_ROI.astype(np.float16)
Data_ROI_mat = Data_ROI
Data_ROI_Name = '1_1'


AllFeats = SERA_FE_main_Fun(data_orginal,
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
                IVHconfig)


AllFeats = np.array(AllFeats)

FE = pd.DataFrame(AllFeats)

CSVfullpath = r"Extracted_features.xlsx"
writer = pd.ExcelWriter(CSVfullpath, engine='xlsxwriter') # pylint: disable=abstract-class-instantiated
FE.to_excel(writer, sheet_name='Extracted_features',index=None)
writer.save() 

stop = timeit.default_timer()

print('Time: ', stop - start)  