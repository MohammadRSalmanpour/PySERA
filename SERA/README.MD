
1) Run Main.py.

2) Choose your Original and ROI images path in section 1: "Selecting the dataset" ( line 50 and 51 ).

for example:

data_orgina_PATH = r'D:\data\IBSI1 Data\ibsi_1_ct_radiomics_phantom\nifti\image\phantom.nii.gz'
Data_RO_PATH = r'D:\data\IBSI1 Data\ibsi_1_ct_radiomics_phantom\nifti\mask\mask.nii.gz'

3) Choose your hyperparameters in section 2: "Radiomics Framework Settings" ( line 55 to 80 ).

for example:

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

4) Run the code.

5) At the end, in current directory, you must see the excel file with name Extracted_features.xlsx.
