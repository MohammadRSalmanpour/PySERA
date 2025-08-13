

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# //													Configuration A 												  //
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# "Selecting the dataset"
data_orgina_PATH = r'..\PySERA-main\PySERA-main\Data\ibsi_1_ct_radiomics_phantom\nifti\image\phantom.nii.gz'
Data_RO_PATH = r'..\PySERA-main\PySERA-main\Data\ibsi_1_ct_radiomics_phantom\nifti\mask\mask.nii.gz'

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

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# //													Configuration B 												  //
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# "Selecting the dataset"
data_orgina_PATH = r'..\PySERA-main\PySERA-main\Data\ibsi_1_ct_radiomics_phantom\nifti\image\phantom.nii.gz'
Data_RO_PATH = r'..\PySERA-main\PySERA-main\Data\ibsi_1_ct_radiomics_phantom\nifti\mask\mask.nii.gz'

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

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# //													Configuration C 												  //
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# "Selecting the dataset"
data_orgina_PATH = r'..\PySERA-main\PySERA-main\Data\ibsi_1_ct_radiomics_phantom\nifti\image\phantom.nii.gz'
Data_RO_PATH = r'..\PySERA-main\PySERA-main\Data\ibsi_1_ct_radiomics_phantom\nifti\mask\mask.nii.gz'

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

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# //													Configuration D 												  //
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# "Selecting the dataset"
data_orgina_PATH = r'..\PySERA-main\PySERA-main\Data\ibsi_1_ct_radiomics_phantom\nifti\image\phantom.nii.gz'
Data_RO_PATH = r'..\PySERA-main\PySERA-main\Data\ibsi_1_ct_radiomics_phantom\nifti\mask\mask.nii.gz'

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

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# //													Configuration E 												  //
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# "Selecting the dataset"
data_orgina_PATH = r'..\PySERA-main\PySERA-main\Data\ibsi_1_ct_radiomics_phantom\nifti\image\phantom.nii.gz'
Data_RO_PATH = r'..\PySERA-main\PySERA-main\Data\ibsi_1_ct_radiomics_phantom\nifti\mask\mask.nii.gz'

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

