import SimpleITK as sitk
from radiomics import featureextractor
import pprint

data_orgina_PATH = r'C:\Users\Sirwan\Desktop\PySERA-main\PySERA-main\Data\ibsi_1_ct_radiomics_phantom\nifti\image\phantom.nii.gz'
Data_RO_PATH = r'C:\Users\Sirwan\Desktop\PySERA-main\PySERA-main\Data\ibsi_1_ct_radiomics_phantom\nifti\mask\mask.nii.gz'

isScale = 1
BinSize = 25
isotVoxSize = 2.0

params = {
    'binWidth': BinSize,
    'resampledPixelSpacing': [isotVoxSize]*3 if isScale else None,
    'interpolator': 'sitkLinear',  # اصلاح شد
    'enableCExtensions': True,
    'featureClass': ['firstorder'],
    'normalize': False,
}

def extract_firstorder_pyradiomics(img_path, mask_path, params):
    img_sitk = sitk.ReadImage(img_path)
    mask_sitk = sitk.ReadImage(mask_path)

    mask_array = sitk.GetArrayFromImage(mask_sitk)
    print("Mask non-zero voxels:", (mask_array > 0).sum())

    img_array = sitk.GetArrayFromImage(img_sitk)
    print("Image intensity stats: min =", img_array.min(), "max =", img_array.max())

    extractor = featureextractor.RadiomicsFeatureExtractor(**params)
    features = extractor.execute(img_sitk, mask_sitk)
    first_order_feats = {k: v for k, v in features.items() if k.startswith('firstorder')}
    return first_order_feats

if __name__ == '__main__':
    feats = extract_firstorder_pyradiomics(data_orgina_PATH, Data_RO_PATH, params)
    print("Extracted first order features (pyradiomics):")
    pprint.pprint(feats)
