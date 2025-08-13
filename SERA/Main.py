from pickle import NONE
from scipy.io import savemat, loadmat
import numpy as np
import itertools
import collections.abc
import SimpleITK as sitk
import pandas as pd
import timeit
from RF_main import SERA_FE_main_Fun
from Sera_ReadWrite import readimage,convert_modalities,similar

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
data_orgina_PATH = r'C:\Users\Sirwan\Desktop\PySERA-main\PySERA-main\Data\ibsi_1_digital_phantom\nifti\image\phantom.nii.gz'
Data_RO_PATH = r'C:\Users\Sirwan\Desktop\PySERA-main\PySERA-main\Data\ibsi_1_digital_phantom\nifti\mask\mask.nii.gz'
# end section 1

# section 2: "Radiomics Framework Settings"
isScale = 0    # whether to do scaling. Has to be 1 to perform any resampling. If 0, always uses the original voxel dimension. 
isGLround = 0  # whether to round voxel intensities to the nearest integer (usually =1 for CT images, =0 for PET and SPECT)
ROI_PV = 0.0    # (default 0.5) ROI partial volume threshold. Used to threshold ROI after resampling: i.e. ROI(ROI<ROI_PV) = 0, ROI(ROI>ROI_PV) = 1.
VoxInterp = 'nearest'  # Image resampling interpolation type  ('nearest', 'linear', or 'cubic'). Note: 'cubic' yeilds inconsistensies with IBSI results. 
ROIInterp = 'nearest'  # ROI resampling interpolation type  ('nearest', 'linear', or 'cubic'), default: 'linear'
isotVoxSize = 0.0    # New isotropic voxel size for resampling in 3D. This will be the new voxel size in X, Y and Z dimension. 
isotVoxSize2D = 0.0  # New voxel size for resampling slices in 2D. This maintains the Z dimension, and rescales both X and Y to this number.
isIsot2D = 0     # (default 0) whether to resample image to isotropic 2D voxels (=1, i.e.keep the original slice thickness) or resample to isotropic 3D voxels (=0). (This is for 1st order features. Higher order 2D features are always calculated with original slice thickness). 

isReSegRng = 0   # whether to perform range re-segmentation. The range is defined below in ReSegIntrvl. NOTE: Re-segmentation generally cannot be provided for arbitrary-unit modalities (MRI, SPECT)
isOutliers = 0   # whether to perform intensity outlier filtering re-segmentaion: remove outlier intensities from the intensity mask. If selected, voxels outside the range of +/- 3 standard deviation will be removed. 
ReSegIntrvl = []    # Range resegmentation interval. Intensity values outside this interval would be replaced by NaN. 

DiscType = 'FBN'  # Discretization type: either 'FBN' (fixed bin numbers) or 'FBS' (fixed bin size or fixed bin width). 
BinSize = 1      # Number of bins (for FNB) or bin size (the size of each bin for FBS). It can be an array, and the features will be calculated for each NB or BS. 
qntz = 'Uniform'   # An extra option for FBN Discretization Type: Either 'Uniform' quantization or 'Lloyd' for Max-Lloyd quantization. (defualt: Uniform)
isQuntzStat = 0   # (default 1) whether to use quantized image to calculate first order statistical features. If 0, no image resample/interp for calculating statistical features. (0 is preferrable for PET images)

DataType = 'CT'   # Type of the dataset. Choose from 'PET', 'CT' or 'MRscan'
ROIsPerImg = 1    # "Maximum" number of ROIs per image. When having multiple patients, enter the largest number of ROIs across all patients. 
isROIsCombined = 0   # Whether to combine ROIs for multiple tumors to one. 
Feats2out = 1    # Select carefully! (default 2) which set of features to return: 1: all IBSI features, 2: 1st-order+all 3D features, 3: 1st-order+only 2D features, 4: 1st-order + selected 2D + all 3D features, 5: all features + moment invarient

IVH_Type = 3     # Setting for Intensity Volume Histogram (IVH) Unit type={0: Definite(PET,CT), 1:Arbitrary(MRI,SPECT. This is FNB), 2: use 1000 bins, 3: use same discritization as histogram (for CT)} 
IVH_DiscCont = 0   # Disc/Cont = {0:Discrete(for CT), 1:Continuous(for CT,PET. This is FBS)}, 
IVH_binSize = 0   # Bin size for Intensity Volumen Histogram in case choosing setting 1 for FNB, or setting 0 and either IVH_DiscCont options.
IVHconfig = [IVH_Type, IVH_DiscCont ,IVH_binSize]
# end section 2


data_orgina = readimage(data_orgina_PATH)
Data_RO = readimage(Data_RO_PATH)


# try:
if isinstance(data_orgina[0], np.ndarray) & isinstance(Data_RO[0], np.ndarray):

    data_orginal = data_orgina[0] 
    Data_ROI = Data_RO[0]
    data_orginal = data_orginal.astype(np.float32)
    Data_ROI = Data_ROI.astype(np.float16)



    if len(data_orginal.shape) == 3 & len(Data_ROI.shape) == 3 :   

        if data_orgina[2] == 'Nifti' or data_orgina[2] == 'Dicom':
            perm = (2, 1, 0)
            data_orginal = np.transpose(data_orginal, perm)
        if Data_RO[2] == 'Nifti' or Data_RO[2] == 'Dicom':
            perm = (2, 1, 0)
            Data_ROI = np.transpose(Data_ROI, perm)  
            
            
        dim01 = np.shape(data_orginal)
        dim02 = np.shape(Data_ROI)
        if (dim01 == dim02):   

            if len(dim01) == 3 & len(dim02) == 3 : 

                    
                if data_orgina[2] == 'Nifti':  
                    VoxelSize = data_orgina[1].GetSpacing()
                if data_orgina[2] == 'Nrrd':  
                    VoxelSize = (data_orgina[1]['space directions'][0,0],data_orgina[1]['space directions'][1,1],data_orgina[1]['space directions'][2,2])
                if data_orgina[2] == 'Dicom':  
                    VoxelSize = data_orgina[1].GetSpacing()     

                if Data_RO[2] == 'Nifti':  
                    VoxelSizelabel = Data_RO[1].GetSpacing()
                if Data_RO[2] == 'Nrrd':  
                    VoxelSizelabel = (Data_RO[1]['space directions'][0,0],Data_RO[1]['space directions'][1,1],Data_RO[1]['space directions'][2,2])
                if Data_RO[2] == 'Dicom':  
                    VoxelSizelabel = Data_RO[1].GetSpacing()     
                


                VoxelSizeInfo = np.array(VoxelSizelabel).astype(np.float32)


                if VoxelSizelabel == VoxelSize:
                    
                    Data_ROI_mat = Data_ROI
                    Data_ROI_Name = '1_1'
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
                                                IVHconfig
                                            )


                    AllFeats = np.array(AllFeats)

                    FE = pd.DataFrame(AllFeats)

                    FE_T = FE.transpose()

                    CSVfullpath = r"Extracted_features.xlsx"
                    writer = pd.ExcelWriter(CSVfullpath, engine='xlsxwriter') # pylint: disable=abstract-class-instantiated
                    FE_T.to_excel(writer, sheet_name='Extracted_features',index=None)
                    writer.close() 

                    stop = timeit.default_timer()

                    print('Time: ', stop - start)  


                else:
                    print('Images voxel size must be same.')     
            else:
                print('Images must be 3D.')
        else:
            print('Dimension of original and segmented image not equal.')
    else:
        print('Dimension of original and segmented image not equal.')
else:
    print('You must use an approprate type of input.')	
# except Exception as e:
#     print('Out of Memory or the parameters of image fusion tool should be selected properly:', e)




# data_orgina = sitk.ReadImage(data_orgina_PATH)
# Data_RO = sitk.ReadImage(Data_RO_PATH)

# VoxelSizeInfo = np.array(data_orgina.GetSpacing())
# VoxelSizeInfo = VoxelSizeInfo.astype(np.float32)

# data_orginal = sitk.GetArrayFromImage(data_orgina)
# data_orginal = np.transpose(data_orginal,(2,1,0))
# data_orginal = data_orginal.astype(np.float32)


# Data_ROI = sitk.GetArrayFromImage(Data_RO)
# Data_ROI = np.transpose(Data_ROI,(2,1,0))
# Data_ROI = Data_ROI.astype(np.float16)
# Data_ROI_mat = Data_ROI
# Data_ROI_Name = '1_1'

