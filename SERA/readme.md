
1) Run RF_main.py.

2) choose your Original and ROI images as nifti file in line 310 and 311.

for example:

    data_orgina = sitk.ReadImage(r'..\Data\ibsi_1_ct_radiomics_phantom\nifti\image\phantom.nii.gz')
    Data_RO = sitk.ReadImage(r'..\Data\ibsi_1_ct_radiomics_phantom\nifti\mask\mask.nii.gz')

3) choose your hyperparameters as nifti file in line 332 to 357.

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

4) run the code.

5) at the end, in current directory, you must see the excel file with name Extracted_features.xlsx.
