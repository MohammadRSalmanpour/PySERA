"""
DICOM loading and processing functionality.
"""

import os
import logging
from typing import Tuple, Optional, List, Dict, Any
import SimpleITK as sitk
import pydicom
from rt_utils import RTStructBuilder
import numpy as np
from ..utils.file_utils import detect_file_format
from ..utils.utils import save_numpy_on_disk, save_rt_on_disk, remove_temp_file, create_process_safe_tempfile
import gc
import shutil


logger = logging.getLogger("Dev_logger")

def load_dicom_series(dicom_dir: str) -> Tuple[Optional[sitk.Image], Optional[str]]:
    """
    Load a DICOM series from a directory.
    
    Args:
        dicom_dir: Directory containing DICOM files
        
    Returns:
        Tuple of (SimpleITK image, series_id) or (None, None) if failed
    """
    try:
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
        
        if not series_ids:
            logger.error(f"No DICOM series found in {dicom_dir}")
            return None, None
        
        series_data = []
        for series_id in series_ids:
            dicom_files = reader.GetGDCMSeriesFileNames(dicom_dir, series_id)
            if dicom_files:
                reader.SetFileNames(dicom_files)
                try:
                    image = reader.Execute()
                    series_data.append({
                        'series_id': series_id,
                        'image': image,
                        'files': dicom_files,
                        'num_slices': len(dicom_files)
                    })
                    logger.info(f"Loaded DICOM series {series_id} with {len(dicom_files)} slices")
                except Exception as e:
                    logger.warning(f"Failed to load series {series_id}: {e}")
                    continue
        
        if not series_data:
            logger.error("No DICOM series could be loaded")
            return None, None
        
        # Sort by number of slices and use the largest series
        series_data.sort(key=lambda x: x['num_slices'], reverse=True)
        main_series = series_data[0]
        
        logger.info(f"Using main series {main_series['series_id']} with {main_series['num_slices']} slices")
        return main_series['image'], main_series['series_id']
        
    except Exception as e:
        logger.error(f"Error loading DICOM series from {dicom_dir}: {e}")
        return None, None


def load_rtstruct_mask(rtstruct_path: str, reference_dicom_dir: str,
                       roi_names: Optional[List[str]] = None) -> Tuple[Optional[Dict[str, str]], Optional[List[str]]]:
    """Load each ROI from RT-STRUCT as a separate mask. If a ROI has multiple lesions, each lesion is named as 'ROIName_lesion_{i}'. Returns dict of ROI name to mask. For SEG/DICOM mask, returns as before."""
    logger.debug(f"rtstruct_path: {rtstruct_path}")
    def get_reference_image() -> Optional[sitk.Image]:
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(reference_dicom_dir)
        if not series_ids:
            logger.error("No DICOM series found in reference directory.")
            return None
        dicom_files = reader.GetGDCMSeriesFileNames(reference_dicom_dir, series_ids[0])
        reader.SetFileNames(dicom_files)
        return reader.Execute()

    def align_mask_shape(mask: np.ndarray, ref_image: sitk.Image, roi_name: str) -> Optional[np.ndarray]:
        image_array = sitk.GetArrayFromImage(ref_image)
        if mask.shape == image_array.shape:
            # Clear RAM
            del image_array
            return mask
        from itertools import permutations
        for perm in permutations(range(mask.ndim)):
            if mask.transpose(perm).shape == image_array.shape:
                logger.info(f"Transposed mask for ROI '{roi_name}' with {perm}")
                # Clear RAM
                del image_array
                return mask.transpose(perm)
        squeezed = np.squeeze(mask)
        if squeezed.shape == image_array.shape:
            logger.info(f"Squeezed mask for ROI '{roi_name}'")
            # Clear RAM
            del image_array
            return squeezed
        for axis in range(image_array.ndim):
            expanded = np.expand_dims(mask, axis=axis)
            if expanded.shape == image_array.shape:
                logger.info(f"Expanded mask for ROI '{roi_name}' on axis {axis}")
                # Clear RAM
                del image_array
                return expanded
        logger.error(f"Could not align shape for ROI '{roi_name}'")
        # Clear RAM
        del image_array
        return None

    def load_rtstruct(file_path: str) -> Tuple[Optional[Dict[str, str]], Optional[List[str]]]:
        try:
            rtstruct = RTStructBuilder.create_from(reference_dicom_dir, file_path)
            available_rois = rtstruct.get_roi_names()
            logger.info(f"Available ROI names: {available_rois}")
            selected_rois = roi_names or available_rois
            reference_image = get_reference_image()
            roi_masks = {}
            for roi in selected_rois:
                if roi not in available_rois:
                    logger.warning(f"ROI '{roi}' not found.")
                    continue
                try:
                    mask = rtstruct.get_roi_mask_by_name(roi).astype(np.uint8)
                    mask = align_mask_shape(mask, reference_image, roi)
                    if mask is None or np.count_nonzero(mask) == 0:
                        continue
                    # Split into connected components (lesions)
                    from scipy.ndimage import label as cc_label
                    labeled_array, num_features = cc_label(mask > 0)
                    if num_features <= 1:
                        mask_sitk = sitk.GetImageFromArray(mask)
                        mask_sitk.CopyInformation(reference_image)
                        mask_sitk_path = save_rt_on_disk(mask_sitk, prefix='load_rtstruct', suffix='.nii.gz')
                        # Clean RAM
                        del mask_sitk, mask
                        gc.collect()

                        roi_masks[f"{roi}_lesion_1"] = mask_sitk_path
                        logger.info(f"Loaded ROI '{roi}' as {roi}_lesion_1")
                    else:
                        for i in range(1, num_features + 1):
                            lesion_mask = (labeled_array == i).astype(np.uint8)
                            if np.count_nonzero(lesion_mask) == 0:
                                continue
                            mask_sitk = sitk.GetImageFromArray(lesion_mask)
                            mask_sitk.CopyInformation(reference_image)
                            mask_sitk_path = save_rt_on_disk(mask_sitk, prefix='load_rtstruct', suffix='.nii.gz')
                            # Clean RAM
                            del mask_sitk, lesion_mask
                            gc.collect()

                            roi_masks[f"{roi}_lesion_{i}"] = mask_sitk_path
                            logger.info(f"Loaded ROI '{roi}' as {roi}_lesion_{i}")
                    del labeled_array
                    gc.collect()
                except Exception as e:
                    logger.warning(f"Failed loading ROI '{roi}': {e}")
            if roi_masks:
                return roi_masks, list(roi_masks.keys())
        except Exception as e:
            logger.error(f"RTSTRUCT load error: {e}")
        return None, None

    def load_seg(file_path: str) -> Tuple[Optional[str], List[str]]:
        try:
            mask_sitk = sitk.ReadImage(file_path)
            mask_sitk_path = save_rt_on_disk(mask_sitk, prefix='load_seg', suffix='.nii.gz')
            # Clean RAM
            del mask_sitk
            gc.collect()
            return mask_sitk_path, ["SEG"]
        except Exception as e2:
            logger.error(f"Failed loading SEG image: {e2}")
        return None, None

    def load_dicom_mask(file_path: str) -> Tuple[Optional[str], List[str]]:
        try:
            mask_sitk = sitk.ReadImage(file_path)
            mask_sitk_path = save_rt_on_disk(mask_sitk, prefix='load_dicom_mask', suffix='.nii.gz')
            # Clean RAM
            del mask_sitk
            gc.collect()
            return mask_sitk_path, ["DICOM_MASK"]
        except Exception as e:
            logger.error(f"Failed loading DICOM mask: {e}")
            return None, None

    def load_from_directory(directory: str) -> Tuple[Optional[Dict[str, str]], Optional[List[str]]]:
        rtstructs, segs, dicom_masks = [], [], []
        for root, _, files in os.walk(directory):
            for file in files:
                path = os.path.join(root, file)
                try:
                    ds = pydicom.dcmread(path, stop_before_pixels=True)
                    if getattr(ds, 'SOPClassUID', None) == '1.2.840.10008.5.1.4.1.1.481.3':
                        rtstructs.append(path)
                    elif getattr(ds, 'Modality', None) == 'SEG':
                        segs.append(path)
                    elif hasattr(ds, 'PixelData'):
                        dicom_masks.append(path)
                except Exception:
                    continue
        if rtstructs:
            return load_rtstruct(rtstructs[0])
        if segs:
            return load_seg(segs[0])
        if dicom_masks:
            if len(dicom_masks) == 1:
                return load_dicom_mask(dicom_masks[0])
            try:
                reader = sitk.ImageSeriesReader()
                series_ids = reader.GetGDCMSeriesIDs(directory)     # toto list[str]
                if series_ids:
                    dicom_files = reader.GetGDCMSeriesFileNames(directory, series_ids[0])       # toto list[str]
                    reader.SetFileNames(dicom_files)
                    mask_sitk_path = save_rt_on_disk(reader.Execute(), prefix='mask_sitk_reader', suffix='.nii.gz')
                    return mask_sitk_path, ["DICOM_MASK_SERIES"]
                mask_sitk = sitk.ReadImage(dicom_masks)
                mask_sitk_path = save_rt_on_disk(mask_sitk, prefix='mask_sitk', suffix='.nii.gz')
                # Clean RAM
                del mask_sitk
                gc.collect()
                return mask_sitk_path, ["DICOM_MASK_STACK"]
            except Exception as e:
                logger.error(f"Failed to load mask series: {e}")
        return None, None

    try:
        if os.path.isfile(rtstruct_path):
            ds = pydicom.dcmread(rtstruct_path, stop_before_pixels=True)
            if getattr(ds, 'SOPClassUID', None) == '1.2.840.10008.5.1.4.1.1.481.3':
                return load_rtstruct(rtstruct_path)
            elif getattr(ds, 'Modality', None) == 'SEG':
                return load_seg(rtstruct_path)
            elif hasattr(ds, 'PixelData'):
                return load_dicom_mask(rtstruct_path)
            else:
                logger.error(f"Unrecognized file type: {rtstruct_path}")
                return None, None
        else:
            return load_from_directory(rtstruct_path)
    except Exception as e:
        logger.error(f"Error loading mask from {rtstruct_path}: {e}")
        return None, None


def align_dicom_mask_to_image(mask_dicom_files, image_sitk):
    """
    Aligns a set of DICOM mask files to the reference image volume using spatial metadata.
    Returns a mask volume (numpy array) matching the image volume shape.
    """
    import pydicom
    import numpy as np
    # Get image slice positions
    img_array = sitk.GetArrayFromImage(image_sitk)
    img_shape = img_array.shape
    img_slices = img_shape[0]
    img_positions = []
    for i in range(img_slices):
        pos = list(image_sitk.TransformIndexToPhysicalPoint((0, 0, i)))
        img_positions.append(pos)
    # Read all mask slices and their positions
    mask_slices = []
    mask_positions = []
    for f in mask_dicom_files:
        ds = pydicom.dcmread(f)
        arr = ds.pixel_array
        # If 2D, expand dims
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        # # Save mask on Disk
        # arr_path = save_numpy_on_disk(arr, prefix='rt_dicom', suffix='.npy')
        # # Clean RAM
        # del arr, ds
        # # Load array from disk
        # arr_disk = np.load(arr_path, mmap_mode=('r'))
        
        mask_slices.append(arr)
        # # Clean RAM
        # del arr_disk
        # gc.collect()
        
        pos = ds.get('ImagePositionPatient', None)
        if pos is not None:
            mask_positions.append([float(x) for x in pos])
        else:
            mask_positions.append(None)
    mask_slices = np.concatenate(mask_slices, axis=0)
    # Build output mask volume
    mask_volume = np.zeros(img_shape, dtype=mask_slices.dtype)
    used = set()
    tolerance = 1e-2  # mm
    for m_idx, m_pos in enumerate(mask_positions):
        if m_pos is None:
            continue
        # Find closest image slice
        best_idx = None
        best_dist = float('inf')
        for i_idx, i_pos in enumerate(img_positions):
            dist = np.linalg.norm(np.array(m_pos) - np.array(i_pos))
            if dist < best_dist:
                best_dist = dist
                best_idx = i_idx
        if best_dist < tolerance:
            mask_volume[best_idx, :, :] = mask_slices[m_idx, :, :]
            used.add(best_idx)
            logger.info(f"Mask slice {m_idx} mapped to image slice {best_idx} (distance {best_dist:.4f} mm)")
        else:
            logger.warning(f"Mask slice {m_idx} not mapped: closest image slice distance {best_dist:.4f} mm exceeds tolerance")
    if len(used) == 0:
        logger.error("No mask slices could be mapped to image volume using spatial alignment.")
        return None, None
    # Return shape and dtype
    type_file = mask_volume.dtype
    shape_file = mask_volume.shape
    # Save ask on disk
    mask_volume_path = save_numpy_on_disk(mask_volume, prefix='rt_dicom_msk_vol', suffix='.npy')
    # Clean RAM
    del mask_volume
    gc.collect()
    return mask_volume_path, (type_file, shape_file)


def convert_dicom_to_arrays(image_input: str, mask_input: str) -> Tuple[Optional[np.ndarray], Optional[Dict], Optional[Any], Optional[Dict]]:
    """
    Convert image and mask files to numpy arrays with metadata.
    Supports DICOM, NIfTI, NRRD, and any SimpleITK-supported format.
    For mask: supports RT-STRUCT, DICOM SEG, and single/multi-slice DICOM mask.
    If mask shape does not match image shape (for DICOM), attempts robust fallback alignment by slice index.
    For RTSTRUCT, returns a dict of ROI name to mask array.
    """
    import nrrd
    perm = (2, 1, 0)
    image_format = detect_file_format(image_input)
    mask_format = detect_file_format(mask_input)
    logger.info(f"Image format detected: {image_format}")
    logger.info(f"Mask format detected: {mask_format}")

    # Load image
    if image_format == 'dicom':
        image_sitk, series_id = load_dicom_series(image_input)
        if image_sitk is None:
            return None, None, None, None
        image_array = sitk.GetArrayFromImage(image_sitk).astype(np.float32)

        type_file = image_array.dtype
        shape_file = image_array.shape

        image_metadata = {
            'type_file': type_file,
            'shape_file': shape_file,
            'format': 'dicom',
            'file': [image_input],
            'series_id': series_id,
            'origin': image_sitk.GetOrigin(),
            'spacing': image_sitk.GetSpacing(),
            'direction': image_sitk.GetDirection()
        }
    elif image_format == 'nifti':
        image_sitk = sitk.ReadImage(image_input)
        image_array = sitk.GetArrayFromImage(image_sitk).astype(np.float32)
        image_array = np.transpose(image_array, perm)

        type_file = image_array.dtype
        shape_file = image_array.shape

        image_metadata = {
            'type_file': type_file,
            'shape_file': shape_file,
            'format': 'nifti',
            'file': [image_input],
            'origin': image_sitk.GetOrigin(),
            'spacing': image_sitk.GetSpacing(),
            'direction': image_sitk.GetDirection()
        }
    elif image_format == 'nrrd':
        try:
            image_array, header = nrrd.read(image_input)
            image_array = image_array.astype(np.float32)

            type_file = image_array.dtype
            shape_file = image_array.shape

            image_metadata = {
                'type_file': type_file,
                'shape_file': shape_file,
                'format': 'nrrd',
                'file': [image_input],
                'header': header
            }
        except Exception:
            image_sitk = sitk.ReadImage(image_input)
            image_array = sitk.GetArrayFromImage(image_sitk).astype(np.float32)

            type_file = image_array.dtype
            shape_file = image_array.shape


            image_metadata = {
                'type_file': type_file,
                'shape_file': shape_file,
                'format': 'nrrd',
                'file': [image_input],
                'origin': image_sitk.GetOrigin(),
                'spacing': image_sitk.GetSpacing(),
                'direction': image_sitk.GetDirection()
            }
    elif image_format == 'npy':
        try:
            image_array = np.load(image_input).astype(np.float32)
            # image_array = np.load(image_input, mmap_mode='r').astype(np.float32)       toto
            image_metadata = {
                'type_file': image_array.dtype,
                'shape_file': image_array.shape,
                'format': 'npy',
                'file': [image_input],
                'origin': (0.0, 0.0, 0.0),  # Default origin for numpy arrays
                'spacing': (1.0, 1.0, 1.0),  # Default spacing for numpy arrays
                'direction': (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)  # Default direction matrix
            }
            # Clean RAM
            # del image_array
            gc.collect()
        except Exception as e:
            logger.error(f"Failed to load NumPy image file: {image_input}, error: {e}")
            return None, None, None, None
    else:
        # Fallback: try SimpleITK
        try:
            image_sitk = sitk.ReadImage(image_input)
            image_array = sitk.GetArrayFromImage(image_sitk).astype(np.float32)

            type_file = image_array.dtype
            shape_file = image_array.shape

            image_metadata = {
                'type_file': type_file,
                'shape_file': shape_file,
                'format': 'other',
                'file': [image_input],
                'origin': image_sitk.GetOrigin(),
                'spacing': image_sitk.GetSpacing(),
                'direction': image_sitk.GetDirection()
            }
        except Exception as e:
            logger.error(f"Unsupported image format: {image_format}, error: {e}")
            return None, None, None, None
    # Load mask
    if mask_format == 'dicom':
        if image_format == 'dicom':
            mask_result, roi_names = load_rtstruct_mask(mask_input, image_input)          # path two itk
            # If RTSTRUCT: mask_result is a dict of {roi_name: mask_sitk}
            if isinstance(mask_result, dict):
                # RTSTRUCT: return dict of {roi_name: mask_array}
                mask_array_path = {}
                type_file = {}
                shape_file = {}
                for roi in mask_result:         # toto
                    # Load mask array
                    loaded_mask_sitk = sitk.ReadImage(mask_result[roi])
                    roi_array = sitk.GetArrayFromImage(loaded_mask_sitk).astype(np.float32)

                    # Store roi data
                    type_file[roi] = roi_array.dtype
                    shape_file[roi] = roi_array.shape

                    # Save array on disk
                    roi_array_path = save_numpy_on_disk(roi_array, prefix="rtstruct_np", suffix=".npy")
                    mask_array_path[roi] = roi_array_path
                    # CLear RAM
                    del roi_array, loaded_mask_sitk
                    gc.collect()
                    
                    #Clean disk
                    remove_temp_file(mask_result[roi])
                # mask_array_shape = len(mask_array_path)
                mask_metadata = {
                    'type_file': type_file,
                    'shape_file': shape_file,
                    'format': 'dicom-rtstruct',
                    'file': [mask_input],
                    'roi_names': list(mask_array_path.keys()),
                    'origin': image_sitk.GetOrigin(),
                    'spacing': image_sitk.GetSpacing(),
                    'direction': image_sitk.GetDirection()
                }
                return image_array, image_metadata, mask_array_path, mask_metadata
            # SEG or DICOM mask: fallback to old behavior
            mask_sitk = mask_result
        else:
            logger.error("DICOM RT-STRUCT/SEG/mask require DICOM images for reference")
            return None, None, None, None
        if mask_sitk is None:
            # Try true spatial alignment for DICOM mask fallback
            import glob
            import os
            mask_dicom_files = []
            for root, dirs, files in os.walk(mask_input):
                for file in files:
                    if file.lower().endswith('.dcm') or file.lower().endswith('.dicom'):
                        mask_dicom_files.append(os.path.join(root, file))
            if mask_dicom_files:
                logger.info(f"Attempting spatial alignment of DICOM mask files: {mask_dicom_files}")
                mask_array_path, type_file, shape_file = align_dicom_mask_to_image(mask_dicom_files, image_sitk)        # file path to path on disk
                if mask_array_path is None:
                    return None, None, None, None

                mask_metadata = {
                    'type_file': type_file,
                    'shape_file': shape_file,
                    'format': 'dicom',
                    'file': mask_dicom_files,
                    'roi_names': ['DICOM_MASK_ALIGNED'],
                    'origin': image_sitk.GetOrigin(),
                    'spacing': image_sitk.GetSpacing(),
                    'direction': image_sitk.GetDirection()
                }

                # Continue to shape check and multilabel conversion below
            else:
                return None, None, None, None
        else:
            # Load mask array from disk
            loaded_mask_sitk = sitk.ReadImage(mask_sitk)
            mask_array = sitk.GetArrayFromImage(loaded_mask_sitk).astype(np.float32)

            # If mask is 2D (single slice), align to closest image slice
            if mask_array.ndim == 2 and image_array.ndim == 3:
                import pydicom
                ds = pydicom.dcmread(mask_input)
                mask_pos = ds.get('ImagePositionPatient', None)
                if mask_pos is not None:
                    mask_pos = [float(x) for x in mask_pos]
                    img_slices = image_array.shape[0]
                    img_positions = [list(image_sitk.TransformIndexToPhysicalPoint((0, 0, i))) for i in range(img_slices)]
                    best_idx = None
                    best_dist = float('inf')
                    for i_idx, i_pos in enumerate(img_positions):
                        dist = np.linalg.norm(np.array(mask_pos) - np.array(i_pos))
                        if dist < best_dist:
                            best_dist = dist
                            best_idx = i_idx
                    tolerance = 1e-2
                    mask_volume = np.zeros_like(image_array)
                    if best_dist < tolerance:
                        mask_volume[best_idx, :, :] = mask_array
                        mask_array = mask_volume
                        logger.info(f"Single-slice mask aligned to image slice {best_idx} (distance {best_dist:.4f} mm)")
                    else:
                        logger.warning(f"Single-slice mask not aligned: closest image slice distance {best_dist:.4f} mm exceeds tolerance")
                        mask_array = mask_volume
                else:
                    # No spatial info, place in first slice
                    mask_volume = np.zeros_like(image_array)
                    mask_volume[0, :, :] = mask_array
                    mask_array = mask_volume
                    logger.warning("Single-slice mask has no spatial info; placed in first image slice.")
            
            type_file = mask_array.dtype
            shape_file = mask_array.shape

            # Save mask on disk     toto
            mask_array_path = save_numpy_on_disk(mask_array, prefix='dicom_sitk_np', suffix='.npy')

            # Clean RAM
            del mask_array
            gc.collect()

            mask_metadata = {
                'type_file': type_file,
                'shape_file': shape_file,
                'format': 'dicom',
                'file': [mask_input],
                'roi_names': roi_names,
                'origin': loaded_mask_sitk.GetOrigin(),
                'spacing': loaded_mask_sitk.GetSpacing(),
                'direction': loaded_mask_sitk.GetDirection()
            }
        # Robust fallback for shape mismatch (DICOM only)
        # Load mask array from disk
        mask_array = np.load(mask_array_path, mmap_mode='r+')
        if isinstance(mask_array, np.ndarray) and image_array.shape != shape_file:
            logger.warning(f"Shape mismatch: image {image_array.shape} vs mask {mask_array.shape}. Attempting robust fallback alignment by slice index.")
            if len(image_array.shape) == 3 and len(mask_array.shape) == 3:
                img_slices = image_array.shape[0]
                mask_slices = mask_array.shape[0]
                if mask_slices < img_slices:
                    pad_width = ((0, img_slices - mask_slices), (0, 0), (0, 0))
                    mask_array_padded = np.pad(mask_array, pad_width, mode='constant', constant_values=0)
                    mask_array = mask_array_padded[:img_slices, :, :]
                    logger.warning(f"Mask padded from {mask_slices} to {img_slices} slices.")
                elif mask_slices > img_slices:
                    mask_array = mask_array[:img_slices, :, :]
                    logger.warning(f"Mask cropped from {mask_slices} to {img_slices} slices.")
                if mask_array.shape != image_array.shape:
                    logger.error(f"Fallback failed: image {image_array.shape} vs mask {mask_array.shape}")
                    return None, None, None, None
                else:
                    logger.info(f"Mask shape after fallback: {mask_array.shape}")
                    mask_metadata['shape_file'] = mask_array.shape
            else:
                logger.error(f"Fallback not implemented for non-3D mask/image shapes: image {image_array.shape}, mask {mask_array.shape}")
                return None, None, None, None
            # Save on disk
            # mask_array_shape = mask_array.shape
            _path = save_numpy_on_disk(mask_array, prefix='padded', suffix='.npy')
            # Clean disk
            remove_temp_file(mask_array_path)
            mask_array_path = _path
        # Clear RAM
        del mask_array
        gc.collect() 
    # After loading mask (for NRRD, NIfTI, NPY), before shape check:
    # For NRRD
    elif mask_format == 'nrrd':
        try:
            mask_array, header = nrrd.read(mask_input)
            mask_array = mask_array.astype(np.float32)
            # Axis order fix: if mask shape is reverse of image shape, transpose
            if 'image_array' in locals() and mask_array.shape[::-1] == image_array.shape:
                mask_array = np.transpose(mask_array, (2, 0, 1))
                logger.info(f"Transposed NRRD mask axes from {mask_array.shape[::-1]} to {mask_array.shape} to match image.")
            type_file = mask_array.dtype
            shape_file = mask_array.shape

            # Save mask on disk     toto
            mask_array_path = save_numpy_on_disk(mask_array, prefix='nrrd_np', suffix='.npy')

            # Clean RAM
            del mask_array
            gc.collect()

            mask_metadata = {
                'type_file': type_file,
                'shape_file': shape_file,
                'format': 'nrrd',
                'file': [mask_input],
                'header': header
            }
        except Exception:
            mask_sitk = sitk.ReadImage(mask_input)
            mask_array = sitk.GetArrayFromImage(mask_sitk).astype(np.float32)

            if 'image_array' in locals() and mask_array.shape[::-1] == image_array.shape:
                mask_array = np.transpose(mask_array, (2, 0, 1))
                logger.info(f"Transposed NRRD mask axes from {mask_array.shape[::-1]} to {mask_array.shape} to match image.")
            
            type_file = mask_array.dtype
            shape_file = mask_array.shape

            # Save mask on disk     toto
            mask_array_path = save_numpy_on_disk(mask_array, prefix='nrrd_np', suffix='.npy')

            # Clean RAM
            del mask_array
            gc.collect()

            mask_metadata = {
                'type_file': type_file,
                'shape_file': shape_file,
                'format': 'nrrd',
                'file': [mask_input],
                'origin': mask_sitk.GetOrigin(),
                'spacing': mask_sitk.GetSpacing(),
                'direction': mask_sitk.GetDirection()
            }
    # For NIfTI (add similar logic if you have NIfTI support)
    elif mask_format == 'nifti':
        mask_sitk = sitk.ReadImage(mask_input)
        mask_array = sitk.GetArrayFromImage(mask_sitk).astype(np.float32)
        mask_array = np.transpose(mask_array, perm)

        if 'image_array' in locals() and mask_array.shape[::-1] == image_array.shape:
            mask_array = np.transpose(mask_array, (2, 0, 1))
            logger.info(f"Transposed NIfTI mask axes from {mask_array.shape[::-1]} to {mask_array.shape} to match image.")
        
        type_file = mask_array.dtype
        shape_file = mask_array.shape

        # Save mask on disk     toto
        mask_array_path = save_numpy_on_disk(mask_array, prefix='nifti_np', suffix='.npy')

        # Clean RAM
        del mask_array
        gc.collect()

        mask_metadata = {
            'type_file': type_file,
            'shape_file': shape_file,
            'format': 'nifti',
            'file': [mask_input],
            'origin': mask_sitk.GetOrigin(),
            'spacing': mask_sitk.GetSpacing(),
            'direction': mask_sitk.GetDirection()
        }
    # For NPY (if supported)
    elif mask_format == 'npy':
        try:
            _, mask_array_path = create_process_safe_tempfile(prefix='np_input', suffix='.npy')
            shutil.copy2(mask_input, mask_array_path)

            # mask_array = np.load(mask_input, mmap_mode='r').astype(np.float32)
            mask_array = np.load(mask_array_path, mmap_mode='r').astype(np.float32)
            if 'image_array' in locals() and mask_array.shape[::-1] == image_array.shape:
                mask_array = np.transpose(mask_array, (2, 0, 1))
                logger.info(f"Transposed NPY mask axes from {mask_array.shape[::-1]} to {mask_array.shape} to match image.")
            
            type_file = mask_array.dtype
            shape_file = mask_array.shape


            # Clean RAM
            del mask_array
            gc.collect()

            mask_metadata = {
                'type_file': type_file,
                'shape_file': shape_file,
                'format': 'npy',
                'file': [mask_input],
                'origin': (0.0, 0.0, 0.0),  # Default origin for numpy arrays
                'spacing': (1.0, 1.0, 1.0),  # Default spacing for numpy arrays
                'direction': (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)  # Default direction matrix
            }
        except Exception as e:
            logger.error(f"Failed to load NumPy mask file: {mask_input}, error: {e}")
            return None, None, None, None
    else:
        try:
            mask_sitk = sitk.ReadImage(mask_input)
            mask_array = sitk.GetArrayFromImage(mask_sitk).astype(np.float32)
            # Save mask on disk     toto
            mask_array_path = save_numpy_on_disk(mask_array, prefix='other_np', suffix='.npy')

            # Load mask from disk       toto
            mask_arrays_disk = np.load(mask_array_path, mmap_mode='r')

            type_file = mask_arrays_disk.dtype
            shape_file = mask_arrays_disk.shape

            # Clean RAM
            del mask_arrays_disk, mask_array
            gc.collect()

            mask_metadata = {
                'type_file': type_file,
                'shape_file': shape_file,
                'format': 'other',
                'file': [mask_input],
                'origin': mask_sitk.GetOrigin(),
                'spacing': mask_sitk.GetSpacing(),
                'direction': mask_sitk.GetDirection()
            }
        except Exception as e:
            logger.error(f"Unsupported mask format: {mask_format}, error: {e}")
            return None, None, None, None
    # Check shape compatibility
    if not (isinstance(mask_array_path, dict)):
        mask_array = np.load(mask_array_path, mmap_mode='r')
        if image_array.shape != mask_array.shape:
            logger.error(f"Shape mismatch after resampling: image {image_array.shape} vs mask {mask_array.shape}")
            return None, None, None, None
    # Clean ram
    del mask_array
    gc.collect()

    # For non-RTSTRUCT, convert binary to multilabel as before
    # if not isinstance(mask_array, dict):
    #     mask_array = mask_array
    return image_array, image_metadata, mask_array_path, mask_metadata