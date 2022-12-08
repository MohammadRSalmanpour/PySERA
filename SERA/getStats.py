

from scipy.io import savemat, loadmat
import os
import numpy as np
import sys
import skimage
import cv2
import itertools
import scipy 


# -------------------------------------------------------------------------
# function [SUVmax,SUVpeak,SUVmean,aucCSH] = getStats(ROIonlyPET)
# -------------------------------------------------------------------------
# DESCRIPTION: 
# This function computes SUVmax, SUVpeak and SUVmean, AUC-CSH and Percent 
# Inactive metrics from the region of interest (ROI) of an input PET volume.
# -------------------------------------------------------------------------
# INPUTS:
# - ROIbox: The smallest box containing the resampled 3D ROI, with the
#           imaging data ready for texture analysis computations. Voxels
#           outside the ROI are set to NaNs.
# -------------------------------------------------------------------------
# OUTPUTS:
# A list of 18 statistical features as documented by ISBI. 
# -------------------------------------------------------------------------
# AUTHOR(S): 
# - Saeed Ashrafinia
# - Mahdi Hosseinzadeh
# -------------------------------------------------------------------------
# HISTORY:
# - Creation: December 2022
# -------------------------------------------------------------------------



def getStats(ROIbox2):
    ROIbox = ROIbox2.copy()
    ROIboxPadded = np.pad(ROIbox, ((1,1),(1,1),(1,1)),mode='constant',constant_values=np.nan)


    ROIboxPadded = np.asarray(ROIboxPadded).flatten(order='F')
    ROIboxPadded = ROIboxPadded[~np.isnan(ROIboxPadded)]

    SUVmax = np.max(ROIboxPadded)

    SUVmean= np.mean(ROIboxPadded)

    # un used
    SUVstd = np.std(ROIboxPadded)

    SUVvar = np.var(ROIboxPadded,0)

    Energy = np.sum( np.float_power(ROIboxPadded,2))
    
    if SUVvar == 0.0:
        Skewness = 0.0
        Kurtosis = 0.0
        CoV = 0.0
    else:
        Skewness = scipy.stats.skew(ROIboxPadded)
        Kurtosis = scipy.stats.kurtosis(ROIboxPadded)
        CoV = np.sqrt(SUVvar) / SUVmean

    Median = np.median(ROIboxPadded)

    SUVmin = np.min(ROIboxPadded)

    Prcnt10 = np.percentile(ROIboxPadded,10)

    Prcnt90 = np.percentile(ROIboxPadded,90)

    IqntlRange = np.percentile(ROIboxPadded,75) - np.percentile(ROIboxPadded,25)

    # un used
    IqntlRangePlus = np.percentile(ROIboxPadded,75) + np.percentile(ROIboxPadded,25)

    Range = SUVmax - SUVmin

    MAD = np.mean(np.abs(ROIboxPadded - SUVmean))

    RobustSetGrater = ROIboxPadded[ROIboxPadded >= Prcnt10]     
    RobustSet = RobustSetGrater[RobustSetGrater<=Prcnt90]  
    SUVmeanR = np.mean(RobustSet)
    RMAD = np.mean(np.abs(RobustSet - SUVmeanR))

    MedAD = np.mean(np.abs(ROIboxPadded - Median))

    
    if IqntlRangePlus == 0.0:
        QCoD = 1.0E6
    else:
        QCoD = IqntlRange / IqntlRangePlus

    RMS = np.sqrt(np.mean( np.float_power(ROIboxPadded,2) ))

    # # # AUC-CSH
    # # AUC_CSH = getAUCCSH(SUVboxPadded)



    StatsVect = [SUVmean,  SUVvar,     Skewness,   Kurtosis, 
                Median,   SUVmin,     Prcnt10,    Prcnt90,
                SUVmax,   IqntlRange, Range,      MAD, 
                RMAD,     MedAD,      CoV,        QCoD, 
                Energy,   RMS,        ]

    return StatsVect
