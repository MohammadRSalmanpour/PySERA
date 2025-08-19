import numpy as np


# -------------------------------------------------------------------------
# function [ROIonlyUniform,levels] = fixedBinSizeQuantization(ROIonly,BinSize,minGL)
# -------------------------------------------------------------------------
# DESCRIPTION:
# This function computes fixed Bin Size quantization on the region of interest
# (ROI) of an input volume.
# -------------------------------------------------------------------------
# INPUTS:
# - ROIonly: 3D array, with voxels outside the ROI set to NaNs.
# - BinSize: Integer specifying the number of gray levels in the quantization.
# - minGL: minimum intensity of resampled ROI.
# -------------------------------------------------------------------------
# OUTPUTS:
# - ROIonlyUniform: Quantized input volume.
# - levels: Vector containing the quantized gray-level values in the ROI
#           (or reconstruction levels of quantization).
# -------------------------------------------------------------------------
# AUTHOR(S):
# - Martin Vallieres <mart.vallieres@gmail.com>
# - Saeed Ashrafinia
# - Mahdi Hosseinzadeh
# -------------------------------------------------------------------------
# HISTORY:
# - Creation: December 2022
# -------------------------------------------------------------------------


def fixedBinSizeQuantization(ROIonly, BinSize, minGL):
    ROIonlyUniform = np.floor((ROIonly - minGL) / BinSize) + 1
    ROIonlyUniform[ROIonly == minGL] = 1
    if np.nanmin(ROIonlyUniform) < 1:
        raise ValueError('Minimum discritization is below 1. Check the fixed bin size quantization.')
    levels = np.arange(1, np.nanmax(ROIonlyUniform) + 1, 1)
    return ROIonlyUniform, levels


# -------------------------------------------------------------------------
# function [ROIonlyUniform,levels] = uniformQuantization(ROIonly,BinSize,minGL)
# -------------------------------------------------------------------------
# DESCRIPTION:
# This function computes uniform quantization on the region of interest
# (ROI) of an input volume.
# -------------------------------------------------------------------------
# INPUTS:
# - ROIonly: 3D array, with voxels outside the ROI set to NaNs.
# - BinSize: Integer specifying the number of gray levels in the quantization.
# - minGL: minimum intensity of resampled ROI.
# -------------------------------------------------------------------------
# OUTPUTS:
# - ROIonlyUniform: Quantized input volume.
# - levels: Vector containing the quantized gray-level values in the ROI
#           (or reconstruction levels of quantization).
# -------------------------------------------------------------------------
# AUTHOR(S):
# - Martin Vallieres <mart.vallieres@gmail.com>
# - Saeed Ashrafinia
# - Mahdi Hosseinzadeh
# -------------------------------------------------------------------------
# HISTORY:
# - Creation: December 2022
# -------------------------------------------------------------------------

def uniformQuantization(ROIonly, Ng, minGL):
    ROIonly = ROIonly.astype(np.float32)
    maxGL = np.nanmax(ROIonly)

    ROIonlyUniform = np.ceil((Ng) * (ROIonly - minGL) / (maxGL - minGL))
    ROIonlyUniform[ROIonlyUniform == 0] = 1

    if np.min(ROIonlyUniform) < 1:
        raise ValueError('Something wrong with FNB discritization.')

    levels = np.arange(1, Ng + 1)

    return ROIonlyUniform, levels


# incomplate
def lloydQuantization(ROIonly, Ng):
    ROIonly = ROIonly.astype(np.float32)
    ROIonly = np.divide(ROIonly, np.max(ROIonly))
    qmax = np.max(ROIonly)
    qmin = np.min(ROIonly)
    ROIonly[ROIonly == 0] = qmax + 1
    ROIonlyShape = ROIonly.shape
    b = np.sort(
        ROIonly.reshape((1, ROIonlyShape[0] * ROIonlyShape[1] * ROIonlyShape[2]))
    )

    d = b(np.where(b <= 1))

    # [transitions,~] = lloyds(d,Ng)

    # temp = [qmin,transitions,qmax]
    # transitions = temp

    # sortEqual = zeros(1,size(ROIonly,1)*size(ROIonly,2)*size(ROIonly,3))

    # sortEqual(find(b <= transitions(2) & b >= transitions(1))) = 1
    # for i = 2:Ng
    #     sortEqual(find(b <= transitions(i+1) & b > transitions(i))) = i
    # end

    ROIonlyLloyd = np.zeros(1, ROIonlyShape[0] * ROIonlyShape[1] * ROIonlyShape[2])
    # ROIonlyLloyd(perm(1:end)) = sortEqual(1:end)

    # ROIonlyLloyd = reshape(ROIonlyLloyd,[size(ROIonly,1),size(ROIonly,2),size(ROIonly,3)])
    # ROIonlyLloyd(ROIonly==2) = NaN

    levels = np.arange(1, Ng)

    return ROIonlyLloyd, levels
