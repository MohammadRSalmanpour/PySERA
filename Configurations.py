

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# //													Configuration A 												  //
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# section 1: "Selecting the dataset"
data_orgina_PATH = r'C:\Users\Sirwan\Desktop\PySERA-main\PySERA-main\Data\ibsi_1_ct_radiomics_phantom\nifti\image\phantom.nii.gz'
Data_RO_PATH = r'C:\Users\Sirwan\Desktop\PySERA-main\PySERA-main\Data\ibsi_1_ct_radiomics_phantom\nifti\mask\mask.nii.gz'
# end section 1

config_A = {
    # Radiomics Framework Settings
    "isScale": 0,
    "isGLround": 1,
    "ROI_PV": 0.0,
    "VoxInterp": "linear",
    "ROIInterp": "linear",
    "isotVoxSize": 2.0,
    "isotVoxSize2D": 2.0,
    "isIsot2D": 0,

    "isReSegRng": 1,
    "isOutliers": 0,
    "ReSegIntrvl": [-500, 400],

    "DiscType": "FBS",
    "BinSize": 25,
    "qntz": "Uniform",
    "isQuntzStat": 1,

    "DataType": "CT",
    "ROIsPerImg": 1,
    "isROIsCombined": 0,
    "Feats2out": 2,

    "IVH_Type": 0,
    "IVH_DiscCont": 0,
    "IVH_binSize": 0,
    "IVHconfig": [0, 0, 0] # [IVH_Type, IVH_DiscCont ,IVH_binSize]
}


# section 2: "Radiomics Framework Settings"
isScale = 0    # whether to do scaling. Has to be 1 to perform any resampling. If 0, always uses the original voxel dimension. 
isGLround = 1  # whether to round voxel intensities to the nearest integer (usually =1 for CT images, =0 for PET and SPECT)
ROI_PV = 0.0    # (default 0.5) ROI partial volume threshold. Used to threshold ROI after resampling: i.e. ROI(ROI<ROI_PV) = 0, ROI(ROI>ROI_PV) = 1.
VoxInterp = 'linear'  # Image resampling interpolation type  ('nearest', 'linear', or 'cubic'). Note: 'cubic' yeilds inconsistensies with IBSI results. 
ROIInterp = 'linear'  # ROI resampling interpolation type  ('nearest', 'linear', or 'cubic'), default: 'linear'
isotVoxSize = 2.0    # New isotropic voxel size for resampling in 3D. This will be the new voxel size in X, Y and Z dimension. 
isotVoxSize2D = 2.0  # New voxel size for resampling slices in 2D. This maintains the Z dimension, and rescales both X and Y to this number.
isIsot2D = 0     # (default 0) whether to resample image to isotropic 2D voxels (=1, i.e.keep the original slice thickness) or resample to isotropic 3D voxels (=0). (This is for 1st order features. Higher order 2D features are always calculated with original slice thickness). 

isReSegRng = 1   # whether to perform range re-segmentation. The range is defined below in ReSegIntrvl. NOTE: Re-segmentation generally cannot be provided for arbitrary-unit modalities (MRI, SPECT)
isOutliers = 0   # whether to perform intensity outlier filtering re-segmentaion: remove outlier intensities from the intensity mask. If selected, voxels outside the range of +/- 3 standard deviation will be removed. 
ReSegIntrvl = [-500, 400]    # Range resegmentation interval. Intensity values outside this interval would be replaced by NaN. 

DiscType = 'FBS'  # Discretization type: either 'FBN' (fixed bin numbers) or 'FBS' (fixed bin size or fixed bin width). 
BinSize = 25      # Number of bins (for FNB) or bin size (the size of each bin for FBS). It can be an array, and the features will be calculated for each NB or BS. 
qntz = 'Uniform'   # An extra option for FBN Discretization Type: Either 'Uniform' quantization or 'Lloyd' for Max-Lloyd quantization. (defualt: Uniform)
isQuntzStat = 1   # (default 1) whether to use quantized image to calculate first order statistical features. If 0, no image resample/interp for calculating statistical features. (0 is preferrable for PET images)

DataType = 'CT'   # Type of the dataset. Choose from 'PET', 'CT' or 'MRscan'
ROIsPerImg = 1    # "Maximum" number of ROIs per image. When having multiple patients, enter the largest number of ROIs across all patients. 
isROIsCombined = 0   # Whether to combine ROIs for multiple tumors to one. 
Feats2out = 1    # Select carefully! (default 2) which set of features to return: 1: all IBSI features, 2: 1st-order+all 3D features, 3: 1st-order+only 2D features, 4: 1st-order + selected 2D + all 3D features, 5: all features + moment invarient

IVH_Type = 0     # Setting for Intensity Volume Histogram (IVH) Unit type={0: Definite(PET,CT), 1:Arbitrary(MRI,SPECT. This is FNB), 2: use 1000 bins, 3: use same discritization as histogram (for CT)} 
IVH_DiscCont = 0   # Disc/Cont = {0:Discrete(for CT), 1:Continuous(for CT,PET. This is FBS)}, 
IVH_binSize = 0   # Bin size for Intensity Volumen Histogram in case choosing setting 1 for FNB, or setting 0 and either IVH_DiscCont options.
IVHconfig = [IVH_Type, IVH_DiscCont ,IVH_binSize]
# end section 2


# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# //													Configuration B 												  //
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# section 1: "Selecting the dataset"
data_orgina_PATH = r'C:\Users\Sirwan\Desktop\PySERA-main\PySERA-main\Data\ibsi_1_ct_radiomics_phantom\nifti\image\phantom.nii.gz'
Data_RO_PATH = r'C:\Users\Sirwan\Desktop\PySERA-main\PySERA-main\Data\ibsi_1_ct_radiomics_phantom\nifti\mask\mask.nii.gz'
# end section 1

config_B = {
    # Radiomics Framework Settings
    "isScale": 1,
    "isGLround": 1,
    "ROI_PV": 0.5,

    "VoxInterp": "linear",
    "ROIInterp": "linear",
    "isotVoxSize": 2.0,
    "isotVoxSize2D": 2.0,
    "isIsot2D": 1,

    "isReSegRng": 1,
    "isOutliers": 0,
    "ReSegIntrvl": [-500, 400],

    "DiscType": "FBN",
    "BinSize": 32,
    "qntz": "Uniform",
    "isQuntzStat": 1,

    "DataType": "CT",
    "ROIsPerImg": 1,
    "isROIsCombined": 0,
    "Feats2out": 2,

    "IVH_Type": 0,
    "IVH_DiscCont": 0,
    "IVH_binSize": 0,
    "IVHconfig": [0, 0, 0], # [IVH_Type, IVH_DiscCont ,IVH_binSize]

}


# section 2: "Radiomics Framework Settings"
isScale = 1    # whether to do scaling. Has to be 1 to perform any resampling. If 0, always uses the original voxel dimension. 
isGLround = 1  # whether to round voxel intensities to the nearest integer (usually =1 for CT images, =0 for PET and SPECT)
ROI_PV = 0.5    # (default 0.5) ROI partial volume threshold. Used to threshold ROI after resampling: i.e. ROI(ROI<ROI_PV) = 0, ROI(ROI>ROI_PV) = 1.
VoxInterp = 'linear'  # Image resampling interpolation type  ('nearest', 'linear', or 'cubic'). Note: 'cubic' yeilds inconsistensies with IBSI results. 
ROIInterp = 'linear'  # ROI resampling interpolation type  ('nearest', 'linear', or 'cubic'), default: 'linear'
isotVoxSize = 2.0    # New isotropic voxel size for resampling in 3D. This will be the new voxel size in X, Y and Z dimension. 
isotVoxSize2D = 2.0  # New voxel size for resampling slices in 2D. This maintains the Z dimension, and rescales both X and Y to this number.
isIsot2D = 1     # (default 0) whether to resample image to isotropic 2D voxels (=1, i.e.keep the original slice thickness) or resample to isotropic 3D voxels (=0). (This is for 1st order features. Higher order 2D features are always calculated with original slice thickness). 

isReSegRng = 1   # whether to perform range re-segmentation. The range is defined below in ReSegIntrvl. NOTE: Re-segmentation generally cannot be provided for arbitrary-unit modalities (MRI, SPECT)
isOutliers = 0   # whether to perform intensity outlier filtering re-segmentaion: remove outlier intensities from the intensity mask. If selected, voxels outside the range of +/- 3 standard deviation will be removed. 
ReSegIntrvl = [-500, 400]    # Range resegmentation interval. Intensity values outside this interval would be replaced by NaN. 

DiscType = 'FBN'  # Discretization type: either 'FBN' (fixed bin numbers) or 'FBS' (fixed bin size or fixed bin width). 
BinSize = 32      # Number of bins (for FNB) or bin size (the size of each bin for FBS). It can be an array, and the features will be calculated for each NB or BS. 
qntz = 'Uniform'   # An extra option for FBN Discretization Type: Either 'Uniform' quantization or 'Lloyd' for Max-Lloyd quantization. (defualt: Uniform)
isQuntzStat = 1   # (default 1) whether to use quantized image to calculate first order statistical features. If 0, no image resample/interp for calculating statistical features. (0 is preferrable for PET images)

DataType = 'CT'   # Type of the dataset. Choose from 'PET', 'CT' or 'MRscan'
ROIsPerImg = 1    # "Maximum" number of ROIs per image. When having multiple patients, enter the largest number of ROIs across all patients. 
isROIsCombined = 0   # Whether to combine ROIs for multiple tumors to one. 
Feats2out = 1    # Select carefully! (default 2) which set of features to return: 1: all IBSI features, 2: 1st-order+all 3D features, 3: 1st-order+only 2D features, 4: 1st-order + selected 2D + all 3D features, 5: all features + moment invarient

IVH_Type = 0     # Setting for Intensity Volume Histogram (IVH) Unit type={0: Definite(PET,CT), 1:Arbitrary(MRI,SPECT. This is FNB), 2: use 1000 bins, 3: use same discritization as histogram (for CT)} 
IVH_DiscCont = 0   # Disc/Cont = {0:Discrete(for CT), 1:Continuous(for CT,PET. This is FBS)}, 
IVH_binSize = 0   # Bin size for Intensity Volumen Histogram in case choosing setting 1 for FNB, or setting 0 and either IVH_DiscCont options.
IVHconfig = [IVH_Type, IVH_DiscCont ,IVH_binSize]
# end section 2


# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# //													Configuration C 												  //
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# section 1: "Selecting the dataset"
data_orgina_PATH = r'C:\Users\Sirwan\Desktop\PySERA-main\PySERA-main\Data\ibsi_1_ct_radiomics_phantom\nifti\image\phantom.nii.gz'
Data_RO_PATH = r'C:\Users\Sirwan\Desktop\PySERA-main\PySERA-main\Data\ibsi_1_ct_radiomics_phantom\nifti\mask\mask.nii.gz'
# end section 1

config_C = {
    # Radiomics Framework Settings
    "isScale": 1,
    "isGLround": 1,
    "ROI_PV": 0.5,
    "VoxInterp": "linear",
    "ROIInterp": "linear",
    "isotVoxSize": 2.0,
    "isotVoxSize2D": 2.0,
    "isIsot2D": 0,

    "isReSegRng": 1,
    "isOutliers": 0,
    "ReSegIntrvl": [-1000, 400],

    "DiscType": "FBS",
    "BinSize": 25,
    "qntz": "Uniform",
    "isQuntzStat": 1,

    "DataType": "CT",
    "ROIsPerImg": 1,
    "isROIsCombined": 0,
    "Feats2out": 1,

    "IVH_Type": 0,
    "IVH_DiscCont": 0,
    "IVH_binSize": 2.5,
    "IVHconfig": [0, 0, 2.5] # [IVH_Type, IVH_DiscCont ,IVH_binSize]
}


# section 2: "Radiomics Framework Settings"
isScale = 1    # whether to do scaling. Has to be 1 to perform any resampling. If 0, always uses the original voxel dimension. 
isGLround = 1  # whether to round voxel intensities to the nearest integer (usually =1 for CT images, =0 for PET and SPECT)
ROI_PV = 0.5    # (default 0.5) ROI partial volume threshold. Used to threshold ROI after resampling: i.e. ROI(ROI<ROI_PV) = 0, ROI(ROI>ROI_PV) = 1.
VoxInterp = 'linear'  # Image resampling interpolation type  ('nearest', 'linear', or 'cubic'). Note: 'cubic' yeilds inconsistensies with IBSI results. 
ROIInterp = 'linear'  # ROI resampling interpolation type  ('nearest', 'linear', or 'cubic'), default: 'linear'
isotVoxSize = 2.0    # New isotropic voxel size for resampling in 3D. This will be the new voxel size in X, Y and Z dimension. 
isotVoxSize2D = 2.0  # New voxel size for resampling slices in 2D. This maintains the Z dimension, and rescales both X and Y to this number.
isIsot2D = 0     # (default 0) whether to resample image to isotropic 2D voxels (=1, i.e.keep the original slice thickness) or resample to isotropic 3D voxels (=0). (This is for 1st order features. Higher order 2D features are always calculated with original slice thickness). 

isReSegRng = 1   # whether to perform range re-segmentation. The range is defined below in ReSegIntrvl. NOTE: Re-segmentation generally cannot be provided for arbitrary-unit modalities (MRI, SPECT)
isOutliers = 0   # whether to perform intensity outlier filtering re-segmentaion: remove outlier intensities from the intensity mask. If selected, voxels outside the range of +/- 3 standard deviation will be removed. 
ReSegIntrvl = [-1000, 400]    # Range resegmentation interval. Intensity values outside this interval would be replaced by NaN. 

DiscType = 'FBS'  # Discretization type: either 'FBN' (fixed bin numbers) or 'FBS' (fixed bin size or fixed bin width). 
BinSize = 25      # Number of bins (for FNB) or bin size (the size of each bin for FBS). It can be an array, and the features will be calculated for each NB or BS. 
qntz = 'Uniform'   # An extra option for FBN Discretization Type: Either 'Uniform' quantization or 'Lloyd' for Max-Lloyd quantization. (defualt: Uniform)
isQuntzStat = 1   # (default 1) whether to use quantized image to calculate first order statistical features. If 0, no image resample/interp for calculating statistical features. (0 is preferrable for PET images)

DataType = 'CT'   # Type of the dataset. Choose from 'PET', 'CT' or 'MRscan'
ROIsPerImg = 1    # "Maximum" number of ROIs per image. When having multiple patients, enter the largest number of ROIs across all patients. 
isROIsCombined = 0   # Whether to combine ROIs for multiple tumors to one. 
Feats2out = 1    # Select carefully! (default 2) which set of features to return: 1: all IBSI features, 2: 1st-order+all 3D features, 3: 1st-order+only 2D features, 4: 1st-order + selected 2D + all 3D features, 5: all features + moment invarient

IVH_Type = 0     # Setting for Intensity Volume Histogram (IVH) Unit type={0: Definite(PET,CT), 1:Arbitrary(MRI,SPECT. This is FNB), 2: use 1000 bins, 3: use same discritization as histogram (for CT)} 
IVH_DiscCont = 0   # Disc/Cont = {0:Discrete(for CT), 1:Continuous(for CT,PET. This is FBS)}, 
IVH_binSize = 2.5   # Bin size for Intensity Volumen Histogram in case choosing setting 1 for FNB, or setting 0 and either IVH_DiscCont options.
IVHconfig = [IVH_Type, IVH_DiscCont ,IVH_binSize]
# end section 2


# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# //													Configuration D 												  //
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# section 1: "Selecting the dataset"
data_orgina_PATH = r'C:\Users\Sirwan\Desktop\PySERA-main\PySERA-main\Data\ibsi_1_ct_radiomics_phantom\nifti\image\phantom.nii.gz'
Data_RO_PATH = r'C:\Users\Sirwan\Desktop\PySERA-main\PySERA-main\Data\ibsi_1_ct_radiomics_phantom\nifti\mask\mask.nii.gz'
# end section 1

config_D = {
    # Radiomics Framework Settings
    "isScale": 1,
    "isGLround": 1,
    "ROI_PV": 0.5,
    "VoxInterp": "linear",
    "ROIInterp": "linear",
    "isotVoxSize": 2.0,
    "isotVoxSize2D": 2.0,
    "isIsot2D": 0,

    "isReSegRng": 0,
    "isOutliers": 1,
    "ReSegIntrvl": [],

    "DiscType": "FBN",
    "BinSize": 32,
    "qntz": "Uniform",
    "isQuntzStat": 1,

    "DataType": "CT",
    "ROIsPerImg": 1,
    "isROIsCombined": 0,
    "Feats2out": 1,

    "IVH_Type": 0,
    "IVH_DiscCont": 0,
    "IVH_binSize": 0,
    "IVHconfig": [0, 0, 0] # [IVH_Type, IVH_DiscCont ,IVH_binSize]
}


# section 2: "Radiomics Framework Settings"
isScale = 1    # whether to do scaling. Has to be 1 to perform any resampling. If 0, always uses the original voxel dimension. 
isGLround = 1  # whether to round voxel intensities to the nearest integer (usually =1 for CT images, =0 for PET and SPECT)
ROI_PV = 0.5    # (default 0.5) ROI partial volume threshold. Used to threshold ROI after resampling: i.e. ROI(ROI<ROI_PV) = 0, ROI(ROI>ROI_PV) = 1.
VoxInterp = 'linear'  # Image resampling interpolation type  ('nearest', 'linear', or 'cubic'). Note: 'cubic' yeilds inconsistensies with IBSI results. 
ROIInterp = 'linear'  # ROI resampling interpolation type  ('nearest', 'linear', or 'cubic'), default: 'linear'
isotVoxSize = 2.0    # New isotropic voxel size for resampling in 3D. This will be the new voxel size in X, Y and Z dimension. 
isotVoxSize2D = 2.0  # New voxel size for resampling slices in 2D. This maintains the Z dimension, and rescales both X and Y to this number.
isIsot2D = 0     # (default 0) whether to resample image to isotropic 2D voxels (=1, i.e.keep the original slice thickness) or resample to isotropic 3D voxels (=0). (This is for 1st order features. Higher order 2D features are always calculated with original slice thickness). 

isReSegRng = 0   # whether to perform range re-segmentation. The range is defined below in ReSegIntrvl. NOTE: Re-segmentation generally cannot be provided for arbitrary-unit modalities (MRI, SPECT)
isOutliers = 1   # whether to perform intensity outlier filtering re-segmentaion: remove outlier intensities from the intensity mask. If selected, voxels outside the range of +/- 3 standard deviation will be removed. 
ReSegIntrvl = []    # Range resegmentation interval. Intensity values outside this interval would be replaced by NaN. 

DiscType = 'FBN'  # Discretization type: either 'FBN' (fixed bin numbers) or 'FBS' (fixed bin size or fixed bin width). 
BinSize = 32      # Number of bins (for FNB) or bin size (the size of each bin for FBS). It can be an array, and the features will be calculated for each NB or BS. 
qntz = 'Uniform'   # An extra option for FBN Discretization Type: Either 'Uniform' quantization or 'Lloyd' for Max-Lloyd quantization. (defualt: Uniform)
isQuntzStat = 1   # (default 1) whether to use quantized image to calculate first order statistical features. If 0, no image resample/interp for calculating statistical features. (0 is preferrable for PET images)

DataType = 'CT'   # Type of the dataset. Choose from 'PET', 'CT' or 'MRscan'
ROIsPerImg = 1    # "Maximum" number of ROIs per image. When having multiple patients, enter the largest number of ROIs across all patients. 
isROIsCombined = 0   # Whether to combine ROIs for multiple tumors to one. 
Feats2out = 1    # Select carefully! (default 2) which set of features to return: 1: all IBSI features, 2: 1st-order+all 3D features, 3: 1st-order+only 2D features, 4: 1st-order + selected 2D + all 3D features, 5: all features + moment invarient

IVH_Type = 0     # Setting for Intensity Volume Histogram (IVH) Unit type={0: Definite(PET,CT), 1:Arbitrary(MRI,SPECT. This is FNB), 2: use 1000 bins, 3: use same discritization as histogram (for CT)} 
IVH_DiscCont = 0   # Disc/Cont = {0:Discrete(for CT), 1:Continuous(for CT,PET. This is FBS)}, 
IVH_binSize = 0   # Bin size for Intensity Volumen Histogram in case choosing setting 1 for FNB, or setting 0 and either IVH_DiscCont options.
IVHconfig = [IVH_Type, IVH_DiscCont ,IVH_binSize]
# end section 2


# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# //													Configuration E 												  //
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# section 1: "Selecting the dataset"
data_orgina_PATH = r'C:\Users\Sirwan\Desktop\PySERA-main\PySERA-main\Data\ibsi_1_ct_radiomics_phantom\nifti\image\phantom.nii.gz'
Data_RO_PATH = r'C:\Users\Sirwan\Desktop\PySERA-main\PySERA-main\Data\ibsi_1_ct_radiomics_phantom\nifti\mask\mask.nii.gz'
# end section 1

config_A = {
    # Radiomics Framework Settings
    "isScale": 1,
    "isGLround": 1,
    "ROI_PV": 0.5,
    "VoxInterp": "cubic",
    "ROIInterp": "linear",
    "isotVoxSize": 2.0,
    "isotVoxSize2D": 2.0,
    "isIsot2D": 0,

    "isReSegRng": 1,
    "isOutliers": 1,
    "ReSegIntrvl": [-1000, 400],

    "DiscType": "FBN",
    "BinSize": 32,
    "qntz": "Uniform",
    "isQuntzStat": 1,

    "DataType": "CT",
    "ROIsPerImg": 1,
    "isROIsCombined": 0,
    "Feats2out": 1,

    "IVH_Type": 1,
    "IVH_DiscCont": 0,
    "IVH_binSize": 1000,
    "IVHconfig": [1, 0, 1000] # [IVH_Type, IVH_DiscCont ,IVH_binSize]
}


# section 2: "Radiomics Framework Settings"
isScale = 1    # whether to do scaling. Has to be 1 to perform any resampling. If 0, always uses the original voxel dimension. 
isGLround = 1  # whether to round voxel intensities to the nearest integer (usually =1 for CT images, =0 for PET and SPECT)
ROI_PV = 0.5    # (default 0.5) ROI partial volume threshold. Used to threshold ROI after resampling: i.e. ROI(ROI<ROI_PV) = 0, ROI(ROI>ROI_PV) = 1.
VoxInterp = 'cubic'  # Image resampling interpolation type  ('nearest', 'linear', or 'cubic'). Note: 'cubic' yeilds inconsistensies with IBSI results. 
ROIInterp = 'linear'  # ROI resampling interpolation type  ('nearest', 'linear', or 'cubic'), default: 'linear'
isotVoxSize = 2.0    # New isotropic voxel size for resampling in 3D. This will be the new voxel size in X, Y and Z dimension. 
isotVoxSize2D = 2.0  # New voxel size for resampling slices in 2D. This maintains the Z dimension, and rescales both X and Y to this number.
isIsot2D = 0     # (default 0) whether to resample image to isotropic 2D voxels (=1, i.e.keep the original slice thickness) or resample to isotropic 3D voxels (=0). (This is for 1st order features. Higher order 2D features are always calculated with original slice thickness). 

isReSegRng = 1   # whether to perform range re-segmentation. The range is defined below in ReSegIntrvl. NOTE: Re-segmentation generally cannot be provided for arbitrary-unit modalities (MRI, SPECT)
isOutliers = 1   # whether to perform intensity outlier filtering re-segmentaion: remove outlier intensities from the intensity mask. If selected, voxels outside the range of +/- 3 standard deviation will be removed. 
ReSegIntrvl = [-1000, 400]    # Range resegmentation interval. Intensity values outside this interval would be replaced by NaN. 

DiscType = 'FBN'  # Discretization type: either 'FBN' (fixed bin numbers) or 'FBS' (fixed bin size or fixed bin width). 
BinSize = 32      # Number of bins (for FNB) or bin size (the size of each bin for FBS). It can be an array, and the features will be calculated for each NB or BS. 
qntz = 'Uniform'   # An extra option for FBN Discretization Type: Either 'Uniform' quantization or 'Lloyd' for Max-Lloyd quantization. (defualt: Uniform)
isQuntzStat = 1   # (default 1) whether to use quantized image to calculate first order statistical features. If 0, no image resample/interp for calculating statistical features. (0 is preferrable for PET images)

DataType = 'CT'   # Type of the dataset. Choose from 'PET', 'CT' or 'MRscan'
ROIsPerImg = 1    # "Maximum" number of ROIs per image. When having multiple patients, enter the largest number of ROIs across all patients. 
isROIsCombined = 0   # Whether to combine ROIs for multiple tumors to one. 
Feats2out = 1    # Select carefully! (default 2) which set of features to return: 1: all IBSI features, 2: 1st-order+all 3D features, 3: 1st-order+only 2D features, 4: 1st-order + selected 2D + all 3D features, 5: all features + moment invarient

IVH_Type = 1     # Setting for Intensity Volume Histogram (IVH) Unit type={0: Definite(PET,CT), 1:Arbitrary(MRI,SPECT. This is FNB), 2: use 1000 bins, 3: use same discritization as histogram (for CT)} 
IVH_DiscCont = 0   # Disc/Cont = {0:Discrete(for CT), 1:Continuous(for CT,PET. This is FBS)}, 
IVH_binSize = 1000   # Bin size for Intensity Volumen Histogram in case choosing setting 1 for FNB, or setting 0 and either IVH_DiscCont options.
IVHconfig = [IVH_Type, IVH_DiscCont ,IVH_binSize]
# end section 2
