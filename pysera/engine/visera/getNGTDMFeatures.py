from ..visera.SERAutilities import *
import gc


# -------------------------------------------------------------------------
# [NGTDM2D, NGTDM3D] = getNGTDMtex(ROI2D,ROI3D,levels2D,levels3D)
# -------------------------------------------------------------------------
# DESCRIPTION:
# This function calculates the NGTDM matrix for 2D and 3D.
# In 2D, every slice is calculated separately, then features are calculated.
#
# The grey level size zone matrix (NGTDM) contains the sum of grey level
# differences of pixels/voxels with discretised grey level i and the
# average discretised grey level of neighbouring pixels/voxels within a
# distance d.
# -------------------------------------------------------------------------
# INPUTS:
# - ROI2D: Smallest box containing the 2D resampled ROI, with the imaging
#          data ready for texture analysis computations. Voxels outside the
#          ROI are set to NaNs.
# - ROI3D: Smallest box containing the 3D resampled ROI, with the imaging
#          data ready for texture analysis computations. Voxels outside the
#          ROI are set to NaNs.
# - levels2D: number of bins (for fixed number of bins method) or bin size
#           (for bin size method) for the 2D resampled ROI.
# - levels3D: number of bins (for fixed number of bins method) or bin size
#           (for bin size method) for the 3D resampled ROI.
# Note: ROIonly is the outputs of prepareVolume.m
# -------------------------------------------------------------------------
# OUTPUTS:
# - NGTDM2D: An array of 16 NGTDM features for the 2D resampled ROI.
# - NGTDM3D: An array of 16 NGTDM features for the 3D resampled ROI.
# -------------------------------------------------------------------------
# AUTHOR(S):
# - Saeed Ashrafinia
# - Mahdi Hosseinzadeh
# -------------------------------------------------------------------------
# HISTORY:
# - Creation: December 2022
# -------------------------------------------------------------------------


def getNGTDMtex(ROI2D2, ROI3D2, levels2D1, levels3D1, feature_value_mode='REAL_VALUE'):
    ROI3D = ROI3D2.copy()
    ROI2D = ROI2D2.copy()
    levels2D = levels2D1.copy()
    levels3D = levels3D1.copy()
    nZ = ROI2D.shape[2]
    # FeatTmp = []
    # NGTDM2D_all = np.zeros( (levels2D.shape[0] , nZ))
    # count_valid_all = NGTDM2D_all

    NGTDM2D_all = []
    count_valid_all = []

    for s in range(0, nZ):
        NGTDM, countValid, Aarray = getNGTDM(ROI2D[:, :, s], levels2D)

        # NGTDM2D_all[:,s] = NGTDM.flatten()
        # count_valid_all[:,s] = countValid.flatten()

        NGTDM2D_all.append(NGTDM.flatten(order='F'))
        count_valid_all.append(countValid.flatten(order='F'))

        Aarray = Aarray.flatten(order='F')
        NGTDMstr = np.array(getNGTDMtextures(NGTDM, countValid, Aarray, feature_value_mode=feature_value_mode))

        if s == 0:
            FeatTmp = NGTDMstr
        else:
            FeatTmp = np.column_stack((FeatTmp, NGTDMstr))

    if np.ndim(FeatTmp) == 1:
        FeatTmp = FeatTmp[:, np.newaxis]
    
    NGTDM2D = np.nanmean(FeatTmp, axis=1).flatten(order='F')

    NGTDM2D_all = np.dstack(NGTDM2D_all)[0]
    count_valid_all = np.dstack(count_valid_all)[0]

    NGTDM25 = np.sum(NGTDM2D_all, axis=1)
    NGTDM25D = np.transpose(
        np.array(getNGTDMtextures(NGTDM25, np.sum(count_valid_all, axis=1), Aarray, feature_value_mode=feature_value_mode)))

    NGTDM, countValid, Aarray = getNGTDM(ROI3D, levels3D)
    NGTDM3Dstr = getNGTDMtextures(NGTDM, countValid, Aarray, feature_value_mode=feature_value_mode)
    NGTDM3D = np.transpose(np.array(NGTDM3Dstr))

    for i in range(0, NGTDM3D.shape[0]):
        if hasattr(NGTDM3D[i], "__len__"):
            NGTDM3D[i] = NGTDM3D[i][0]
        if hasattr(NGTDM25D[i], "__len__"):
            NGTDM25D[i] = NGTDM25D[i][0]
        if hasattr(NGTDM2D[i], "__len__"):
            NGTDM2D[i] = NGTDM2D[i][0]

    NGTDM2D = NGTDM2D.astype(np.float32)
    NGTDM3D = NGTDM3D.astype(np.float32)
    NGTDM25D = NGTDM25D.astype(np.float32)

    return NGTDM2D, NGTDM3D, NGTDM25D


def getNGTDMtextures(NGTDM, countValid, Aarray, epsilon=1e-30, feature_value_mode='REAL_VALUE'):
    nV = np.sum(countValid)
    if nV == 0:
        if feature_value_mode=='APPROXIMATE_VALUE':
            value = 0
        else:
            # if feature_value_mode=='REAL_VALUE'
            value = np.nan
        import logging
        output = [value] * 5
        logging.getLogger("Dev_logger").warning("NGTDM No valid voxels found in NGTDM calculation (nV=0). ROI may be empty or it has only 1 voxel. Returning {value} values.")
        return output

    Pi = np.divide(countValid, nV).flatten(order='F')
    nG = len(NGTDM)
    nGp = np.count_nonzero(Pi)
    pValid = np.where(Pi > 0)[0]
    pValid = pValid + 1
    nValid = len(pValid)

    xx = np.dot(np.transpose(Pi), NGTDM)
    Coarseness = np.float_power(xx + np.finfo(float).eps, -1)
    Coarseness = float(np.squeeze(Coarseness))
    Coarseness = np.minimum(Coarseness, np.float_power(10, 6))

    val = 0
    for i in range(0, nG):
        for j in range(0, nG):
            val = val + Pi[i] * Pi[j] * np.float_power([i - j], 2)

    denominator = nGp * (nGp - 1) * nV
    if denominator == 0 and feature_value_mode == 'APPROXIMATE_VALUE':
        import logging
        logging.getLogger("Dev_logger").warning(f"NGTDM Zero denominator in NGTDM Contrast calculation. Setting Contrast to {epsilon}.")
        Contrast = val * np.sum(NGTDM) / epsilon  # to handle division by zero
    elif denominator == 0 and feature_value_mode == 'REAL_VALUE':
        import logging
        logging.getLogger("Dev_logger").warning("NGTDM Zero denominator in NGTDM Contrast calculation. Setting Contrast to NaN.")
        Contrast = np.nan
    else:
        Contrast = val * np.sum(NGTDM) / denominator

    denom = 0
    val = 0
    val_Strength = 0
    for i in range(0, nValid):
        for j in range(0, nValid):
            denom = denom + np.abs(pValid[i] * Pi[pValid[i] - 1] - pValid[j] * Pi[pValid[j] - 1])
            val = val + (np.abs(pValid[i] - pValid[j]) / (nV * (Pi[pValid[i] - 1] + Pi[pValid[j] - 1]))) * (
                        Pi[pValid[i] - 1] * NGTDM[pValid[i] - 1] + Pi[pValid[j] - 1] * NGTDM[pValid[j] - 1])
            val_Strength = val_Strength + (Pi[pValid[i] - 1] + Pi[pValid[j] - 1]) * np.float_power(
                (pValid[i] - pValid[j]), 2)

    if denom == 0 and feature_value_mode == 'APPROXIMATE_VALUE':
        import logging
        logging.getLogger("Dev_logger").warning(
            f"Zero denominator in NGTDM Busyness calculation. ROI may be too small or homogeneous. Setting Busyness to {epsilon}.")
        Busyness = (np.dot(np.transpose(Pi), NGTDM)) / epsilon  # to handle division by zero
    elif denom == 0 and feature_value_mode == 'REAL_VALUE':
        import logging
        logging.getLogger("Dev_logger").warning(
            f"Zero denominator in NGTDM Busyness calculation. ROI may be too small or homogeneous. Setting Busyness to NaN.")
        Busyness = 0
    else:
        Busyness = (np.dot(np.transpose(Pi), NGTDM)) / denom

    Complexity = val
    Strength = val_Strength / (np.finfo(float).eps + np.sum(NGTDM))

    if hasattr(Coarseness, "__len__"):
        Coarseness = Coarseness[0]
    if hasattr(Contrast, "__len__"):
        Contrast = Contrast[0]
    if hasattr(Busyness, "__len__"):
        Busyness = Busyness[0]
    if hasattr(Complexity, "__len__"):
        Complexity = Complexity[0]
    if hasattr(Strength, "__len__"):
        Strength = Strength[0]

    textures = [Coarseness, Contrast, Busyness, Complexity, Strength]

    return textures


def getNGTDM(ROIOnly2, levels):
    """Memory-optimized NGTDM calculation with reduced array operations"""
    # Use view instead of copy when possible
    ROIOnly = ROIOnly2.copy() if ROIOnly2.flags.writeable else ROIOnly2

    if ROIOnly.ndim == 2:
        twoD = 1
    else:
        twoD = 0

    nLevel = len(levels)
    adjust = 10000 if nLevel > 100 else 1000

    # Memory-efficient padding
    if twoD:
        ROIOnly = np.pad(ROIOnly, ((1, 1), (1, 1)), mode='constant', constant_values=np.nan)
    else:
        ROIOnly = np.pad(ROIOnly, ((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=np.nan)

    # Vectorized quantization to avoid loops
    uniqueVol = np.round(levels * adjust) / adjust
    ROIOnly_rounded = np.round(ROIOnly * adjust) / adjust

    # Create lookup table for more efficient quantization
    NL = len(levels)
    ROIOnly_quantized = np.full(ROIOnly.shape, np.nan)

    # Vectorized quantization using broadcasting
    for i, level in enumerate(uniqueVol):
        mask = np.isclose(ROIOnly_rounded, level, rtol=1e-10)
        ROIOnly_quantized[mask] = i + 1

    ROIOnly = ROIOnly_quantized
    del ROIOnly_rounded, ROIOnly_quantized
    gc.collect()

    # Pre-allocate arrays with appropriate size
    NGTDM = np.zeros(NL, dtype=np.float64)
    countValid = np.zeros(NL, dtype=np.int32)

    # Get valid positions more efficiently
    valid_mask = ~np.isnan(ROIOnly)
    valid_positions = np.where(valid_mask)

    if len(valid_positions[0]) == 0:
        return NGTDM.reshape(-1, 1), countValid.reshape(-1, 1), np.array([])

    if twoD:
        # Vectorized 2D neighbor processing
        i_coords, j_coords = valid_positions
        posValid = np.column_stack((i_coords, j_coords))
        nValid_temp = posValid.shape[0]

        # Pre-allocate result array
        Aarray = np.zeros((nValid_temp, 2), dtype=np.float32)

        # Process neighbors in vectorized manner where possible
        for n in range(nValid_temp):
            i, j = posValid[n]

            # Extract 3x3 neighborhood
            neighborhood = ROIOnly[i - 1:i + 2, j - 1:j + 2].ravel()
            center_value = int(neighborhood[4]) - 1

            # Remove center and calculate valid neighbors
            valid_neighbors = neighborhood[np.arange(9) != 4]  # Remove center
            valid_mask_nei = ~np.isnan(valid_neighbors)

            if np.any(valid_mask_nei):
                mean_neighbors = np.mean(valid_neighbors[valid_mask_nei])
                diff = abs(center_value + 1 - mean_neighbors)
                NGTDM[center_value] += diff
                countValid[center_value] += 1
                Aarray[n] = [center_value, diff]

    else:
        # Vectorized 3D neighbor processing
        i_coords, j_coords, k_coords = valid_positions
        posValid = np.column_stack((i_coords, j_coords, k_coords))
        nValid_temp = posValid.shape[0]

        # Pre-allocate result array
        Aarray = np.zeros((nValid_temp, 2), dtype=np.float32)

        # Process neighbors in vectorized manner where possible
        for n in range(nValid_temp):
            i, j, k = posValid[n]

            # Extract 3x3x3 neighborhood
            neighborhood = ROIOnly[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2].ravel()
            center_value = int(neighborhood[13]) - 1

            # Remove center and calculate valid neighbors
            valid_neighbors = neighborhood[np.arange(27) != 13]  # Remove center
            valid_mask_nei = ~np.isnan(valid_neighbors)

            if np.any(valid_mask_nei):
                mean_neighbors = np.mean(valid_neighbors[valid_mask_nei])
                diff = abs(center_value + 1 - mean_neighbors)
                NGTDM[center_value] += diff
                countValid[center_value] += 1
                Aarray[n] = [center_value, diff]

    return NGTDM.reshape(-1, 1), countValid.reshape(-1, 1), Aarray








