import numpy as np
import logging
from ...utils.utils import handle_math_operations
# -------------------------------------------------------------------------
# function [metrics] = getMI(ROIonlyPET)
# -------------------------------------------------------------------------
# DESCRIPTION: 
# This function computes moment invariants of an ROI.
# -------------------------------------------------------------------------
# INPUTS:
# - ROIbox: The smallest box containing the resampled 3D ROI, with the
#           imaging data ready for texture analysis computations. Voxels
#           outside the ROI are set to NaNs.
# -------------------------------------------------------------------------
# OUTPUTS:
# A list of 10 moment invariants features
# -------------------------------------------------------------------------
# AUTHOR(S): 
# - Saeed Ashrafinia
# - Mahdi Hosseinzadeh
# -------------------------------------------------------------------------
# HISTORY:
# - Creation: December 2022
# -------------------------------------------------------------------------

logger = logging.getLogger("Dev_logger")

def getMI(ROIbox, feature_value_mode):
    """
    Memory-optimized version of getMI function.
    Applies several memory-efficient tricks:
    1. In-place operations where possible
    2. Reuse of temporary arrays
    3. Strategic deletion of large arrays
    4. Chunked processing for large arrays
    5. Use of views instead of copies where safe
    """
    np_sum = np.nansum if feature_value_mode=='APPROXIMATE_VALUE' else np.sum

    # Use view instead of copy when possible, only copy if we need to modify
    if np.may_share_memory(ROIbox, ROIbox.base if ROIbox.base is not None else ROIbox):
        ROIbox = ROIbox.copy()

    # Create mask in-place to save memory
    mask = np.ones_like(ROIbox, dtype=np.float64)
    nan_mask = np.isnan(ROIbox)
    mask[nan_mask] = np.nan

    xdim, ydim, zdim = ROIbox.shape

    # Calculate min and mean before creating large meshgrids
    min_image = np.nanmin(ROIbox)
    mean_image = np.nanmean(ROIbox)
    denominator = mean_image - min_image

    # Create hold_image in-place to save memory
    # Subtract min_image in-place
    ROIbox -= min_image

    # Create a temporary mask for non-nan values
    valid_mask = ~nan_mask

    # Calculate hold_image efficiently
    hold_image = np.zeros_like(ROIbox)
    hold_image[valid_mask] = ROIbox[valid_mask] / denominator

    # Clear ROIbox as we don't need it anymore
    del ROIbox

    # Calculate initial moments
    m000 = np_sum(hold_image[valid_mask])

    # Create mask array for calculations (0 for nan, 1 for valid)
    mask_calc = np.zeros_like(mask)
    mask_calc[valid_mask] = 1.0
    om000 = np_sum(mask_calc)

    u000 = m000
    ou000 = om000

    # Memory-efficient coordinate generation using broadcasting
    # Instead of full meshgrid, use broadcasting to save memory
    def calculate_coordinate_moments():
        # Create 1D ranges
        x_range = np.arange(1, xdim + 1, dtype=np.float64)
        y_range = np.arange(1, ydim + 1, dtype=np.float64)
        z_range = np.arange(1, zdim + 1, dtype=np.float64)

        # Calculate first moments using broadcasting instead of full meshgrid
        m100 = 0.0
        m010 = 0.0
        m001 = 0.0
        om100 = 0.0
        om010 = 0.0
        om001 = 0.0

        # Process slice by slice to save memory
        for i in range(xdim):
            x_weight = x_range[i]
            slice_hold = hold_image[i, :, :]
            slice_mask = mask_calc[i, :, :]

            # X moments
            m100 += x_weight * np_sum(slice_hold)
            om100 += x_weight * np_sum(slice_mask)

            # Y moments
            for j in range(ydim):
                y_weight = y_range[j]
                line_hold = slice_hold[j, :]
                line_mask = slice_mask[j, :]

                m010 += y_weight * np_sum(line_hold)
                om010 += y_weight * np_sum(line_mask)

                # Z moments
                m001 += np_sum(z_range * line_hold)
                om001 += np_sum(z_range * line_mask)
 
        return m100, m010, m001, om100, om010, om001, x_range, y_range, z_range

    m100, m010, m001, om100, om010, om001, x_range, y_range, z_range = calculate_coordinate_moments()

    # Calculate means
    m000 = handle_math_operations(np.array(m000), feature_value_mode=feature_value_mode, mode='divide')
    om000 = handle_math_operations(np.array(om000), feature_value_mode=feature_value_mode, mode='divide')
    x_mean = m100 / m000
    y_mean = m010 / m000
    z_mean = m001 / m000
    ox_mean = om100 / om000
    oy_mean = om010 / om000
    oz_mean = om001 / om000

    # Calculate higher order moments efficiently
    def calculate_higher_moments():
        # Initialize moment accumulators
        moments = {
            'u200': 0.0, 'u020': 0.0, 'u002': 0.0,
            'u110': 0.0, 'u101': 0.0, 'u011': 0.0,
            'u300': 0.0, 'u030': 0.0, 'u003': 0.0,
            'u210': 0.0, 'u201': 0.0, 'u120': 0.0,
            'u102': 0.0, 'u021': 0.0, 'u012': 0.0,
            'u111': 0.0
        }

        omoments = {
            'ou200': 0.0, 'ou020': 0.0, 'ou002': 0.0,
            'ou110': 0.0, 'ou101': 0.0, 'ou011': 0.0,
            'ou300': 0.0, 'ou030': 0.0, 'ou003': 0.0,
            'ou210': 0.0, 'ou201': 0.0, 'ou120': 0.0,
            'ou102': 0.0, 'ou021': 0.0, 'ou012': 0.0,
            'ou111': 0.0
        }

        # Process in slices to minimize memory usage
        for i in range(xdim):
            x_diff = x_range[i] - x_mean
            ox_diff = x_range[i] - ox_mean
            x_diff2 = x_diff * x_diff
            x_diff3 = x_diff2 * x_diff
            ox_diff2 = ox_diff * ox_diff
            ox_diff3 = ox_diff2 * ox_diff

            slice_hold = hold_image[i, :, :]
            slice_mask = mask_calc[i, :, :]

            for j in range(ydim):
                y_diff = y_range[j] - y_mean
                oy_diff = y_range[j] - oy_mean
                y_diff2 = y_diff * y_diff
                y_diff3 = y_diff2 * y_diff
                oy_diff2 = oy_diff * oy_diff
                oy_diff3 = oy_diff2 * oy_diff

                line_hold = slice_hold[j, :]
                line_mask = slice_mask[j, :]

                # Calculate z differences vectorized
                z_diff = z_range - z_mean
                oz_diff = z_range - oz_mean
                z_diff2 = z_diff * z_diff
                z_diff3 = z_diff2 * z_diff
                oz_diff2 = oz_diff * oz_diff
                oz_diff3 = oz_diff2 * oz_diff

                # Accumulate moments
                moments['u200'] += x_diff2 * np_sum(line_hold)
                moments['u020'] += y_diff2 * np_sum(line_hold)
                moments['u002'] += np_sum(z_diff2 * line_hold)

                moments['u110'] += x_diff * y_diff * np_sum(line_hold)
                moments['u101'] += x_diff * np_sum(z_diff * line_hold)
                moments['u011'] += y_diff * np_sum(z_diff * line_hold)

                moments['u300'] += x_diff3 * np_sum(line_hold)
                moments['u030'] += y_diff3 * np_sum(line_hold)
                moments['u003'] += np_sum(z_diff3 * line_hold)

                moments['u210'] += x_diff2 * y_diff * np_sum(line_hold)
                moments['u201'] += x_diff2 * np_sum(z_diff * line_hold)
                moments['u120'] += x_diff * y_diff2 * np_sum(line_hold)
                moments['u102'] += x_diff * np_sum(z_diff2 * line_hold)
                moments['u021'] += y_diff2 * np_sum(z_diff * line_hold)
                moments['u012'] += y_diff * np_sum(z_diff2 * line_hold)
                moments['u111'] += x_diff * y_diff * np_sum(z_diff * line_hold)

                # Mask moments
                omoments['ou200'] += ox_diff2 * np_sum(line_mask)
                omoments['ou020'] += oy_diff2 * np_sum(line_mask)
                omoments['ou002'] += np_sum(oz_diff2 * line_mask)

                omoments['ou110'] += ox_diff * oy_diff * np_sum(line_mask)
                omoments['ou101'] += ox_diff * np_sum(oz_diff * line_mask)
                omoments['ou011'] += oy_diff * np_sum(oz_diff * line_mask)

                omoments['ou300'] += ox_diff3 * np_sum(line_mask)
                omoments['ou030'] += oy_diff3 * np_sum(line_mask)
                omoments['ou003'] += np_sum(oz_diff3 * line_mask)

                omoments['ou210'] += ox_diff2 * oy_diff * np_sum(line_mask)
                omoments['ou201'] += ox_diff2 * np_sum(oz_diff * line_mask)
                omoments['ou120'] += ox_diff * oy_diff2 * np_sum(line_mask)
                omoments['ou102'] += ox_diff * np_sum(oz_diff2 * line_mask)
                omoments['ou021'] += oy_diff2 * np_sum(oz_diff * line_mask)
                omoments['ou012'] += oy_diff * np_sum(oz_diff2 * line_mask)
                omoments['ou111'] += ox_diff * oy_diff * np_sum(oz_diff * line_mask)

        return moments, omoments

    moments, omoments = calculate_higher_moments()

    # Clean up large arrays
    del hold_image, mask_calc

    # Calculate normalized moments
    u000_power_5_3 = np.power(u000, 5.0 / 3.0)
    u000_power_2 = u000 * u000
    ou000_power_5_3 = np.power(ou000, 5.0 / 3.0)
    ou000_power_2 = ou000 * ou000

    # Calculate final metrics
    u000_power_5_3 = handle_math_operations(np.array(u000_power_5_3), feature_value_mode=feature_value_mode, mode='divide')
    n200 = moments['u200'] / u000_power_5_3
    n020 = moments['u020'] / u000_power_5_3
    n002 = moments['u002'] / u000_power_5_3
    n110 = moments['u110'] / u000_power_5_3
    n101 = moments['u101'] / u000_power_5_3
    n011 = moments['u011'] / u000_power_5_3
    u000_power_2 = handle_math_operations(np.array(u000_power_2), feature_value_mode=feature_value_mode, mode='divide')
    n300 = moments['u300'] / u000_power_2
    n030 = moments['u030'] / u000_power_2
    n003 = moments['u003'] / u000_power_2
    n210 = moments['u210'] / u000_power_2
    n201 = moments['u201'] / u000_power_2
    n120 = moments['u120'] / u000_power_2
    n102 = moments['u102'] / u000_power_2
    n021 = moments['u021'] / u000_power_2
    n012 = moments['u012'] / u000_power_2
    n111 = moments['u111'] / u000_power_2
    ou000_power_5_3 = handle_math_operations(np.array(ou000_power_5_3), feature_value_mode=feature_value_mode, mode='divide')
    on200 = omoments['ou200'] / ou000_power_5_3
    on020 = omoments['ou020'] / ou000_power_5_3
    on002 = omoments['ou002'] / ou000_power_5_3
    on110 = omoments['ou110'] / ou000_power_5_3
    on101 = omoments['ou101'] / ou000_power_5_3
    on011 = omoments['ou011'] / ou000_power_5_3
    ou000_power_2 = handle_math_operations(np.array(ou000_power_2), feature_value_mode=feature_value_mode, mode='divide')
    on300 = omoments['ou300'] / ou000_power_2
    on030 = omoments['ou030'] / ou000_power_2
    on003 = omoments['ou003'] / ou000_power_2
    on210 = omoments['ou210'] / ou000_power_2
    on201 = omoments['ou201'] / ou000_power_2
    on120 = omoments['ou120'] / ou000_power_2
    on102 = omoments['ou102'] / ou000_power_2
    on021 = omoments['ou021'] / ou000_power_2
    on012 = omoments['ou012'] / ou000_power_2
    on111 = omoments['ou111'] / ou000_power_2

    # Calculate final invariants
    J1 = n200 + n020 + n002
    Q = (n200 * n200 + n020 * n020 + n002 * n002 +
         2 * (n101 * n101 + n110 * n110 + n011 * n011))

    J2 = (n200 * n020 + n200 * n002 + n020 * n002 -
          n101 * n101 - n110 * n110 - n011 * n011)

    J3 = (n200 * n020 * n002 - n002 * n110 * n110 +
          2 * n110 * n101 * n011 - n020 * n101 * n101 - n200 * n011 * n011)

    B3 = (n300 * n300 + n030 * n030 + n003 * n003 +
          3 * (n210 * n210 + n201 * n201 + n120 * n120 + n102 * n102 + n021 * n021 + n012 * n012) +
          6 * n111 * n111)

    oJ1 = on200 + on020 + on002
    oQ = (on200 * on200 + on020 * on020 + on002 * on002 +
          2 * (on101 * on101 + on110 * on110 + on011 * on011))

    oJ2 = (on200 * on020 + on200 * on002 + on020 * on002 -
           on101 * on101 - on110 * on110 - on011 * on011)

    oJ3 = (on200 * on020 * on002 - on002 * on110 * on110 +
           2 * on110 * on101 * on011 - on020 * on101 * on101 - on200 * on011 * on011)

    oB3 = (on300 * on300 + on030 * on030 + on003 * on003 +
           3 * (on210 * on210 + on201 * on201 + on120 * on120 + on102 * on102 + on021 * on021 + on012 * on012) +
           6 * on111 * on111)

    return [J1, Q, J2, J3, B3, oJ1, oQ, oJ2, oJ3, oB3]
