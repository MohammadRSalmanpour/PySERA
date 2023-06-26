
from scipy.io import savemat, loadmat
import os
import numpy as np
import sys
import skimage
import cv2
import itertools
import scipy 
from mahotas.labeled import bwperim
import math


# -------------------------------------------------------------------------
# [GLDZM2D_F, GLDZM3D_F] =
# getGLDZMtex(ROI2D,ROI3D, ROIonly2D, ROIonly3D, levels2D, levels3D) 
# -------------------------------------------------------------------------
# DESCRIPTION: 
# This function calculates the GLDZM matrix for 2D and 3D.
# In 2D, every slice is calculated separately, then features are calculated.
# 
# The grey level distance zone matrix (GLDZM) counts the number of groups
# of connected voxels with a specific discretised grey level value and
# distance to ROI edge (Thibault et al., 2014). The matrix captures the
# relation between location and grey level. Two maps are required to
# calculate the GLDZM. The first is a grey level grouping map, identical
# with the one created for the grey level size zone matrix (GLSZM). The
# second is a distance map.
# Note that for GLDZM we need both the intensity and morphological ROIs.
# -------------------------------------------------------------------------
# INPUTS:
# - ROI2D: Smallest box containing the 2D resampled ROI, with the imaging
#          data ready for texture analysis computations. Voxels outside the
#          ROI are set to NaNs.   
# - ROI3D: Smallest box containing the 3D resampled ROI, with the imaging
#          data ready for texture analysis computations. Voxels outside the
#          ROI are set to NaNs.   
# - ROIonly2D: Smallest box containing the 2D resampled morphological ROI,
#              values are either 0 or 1. 
# - ROIonly3D: Smallest box containing the 3D resampled morphological ROI,
#              values are either 0 or 1. 
# - levels2D: number of bins (for fixed number of bins method) or bin size
#           (for bin size method) for the 2D resampled ROI.
# - levels3D: number of bins (for fixed number of bins method) or bin size
#           (for bin size method) for the 3D resampled ROI.
# Note: ROIonly is the outputs of prepareVolume.m
# -------------------------------------------------------------------------
# OUTPUTS:
# - GLDZM2D: An array of 16 GLDZM features for the 2D resampled ROI.
# - GLDZM3D: An array of 16 GLDZM features for the 3D resampled ROI.
# -------------------------------------------------------------------------
# AUTHOR(S): 
# - Saeed Ashrafinia
# - Mahdi Hosseinzadeh
# -------------------------------------------------------------------------
# HISTORY:
# - Creation: December 2022
# -------------------------------------------------------------------------


def getGLDZMtex(ROI2D2,ROI3D2,ROIonly2D2,ROIonly3D_2,levels2D1,levels3D1):

    ROI3D = ROI3D2.copy()
    ROI2D = ROI2D2.copy()
    ROIonly2D = ROIonly2D2.copy()
    levels2D = levels2D1.copy()
    levels3D = levels3D1.copy()
    ROIonly3D = ROIonly3D_2.copy()

    nX, nY, nZ = ROI2D.shape
    # FeatTmp = []
    # GLDZM_all = np.zeros((levels2D.ndim , np.ceil(np.max(nX,nY)/2) , nZ))
    GLDZM_all = []
    for s in range(0,nZ): 
        GLDZM  = getGLDZM(ROI2D[:,:,s],ROIonly2D[:,:,s],levels2D)
        # GLDZM_all[:,0:GLDZM.shape[1] , s] = GLDZM
        GLDZM_all.append(GLDZM)

        GLDZMstr = np.array(CalcGLDZM(GLDZM,ROI2D[:,:,s]))
        # FeatTmp = np.concatenate(FeatTmp , np.transpose(GLDZMstr) , axis=1)

        if s == 0:
            FeatTmp = GLDZMstr
            # FeatTmp = np.expand_dims(FeatTmp,axis=1)
        else:    
            FeatTmp = np.column_stack((FeatTmp , GLDZMstr))
        

    # FeatTmp_nan = FeatTmp[FeatTmp != np.nan]
    # GLDZM2D_F = np.mean(FeatTmp_nan , axis=1)
    GLDZM2D_F = np.nanmean(FeatTmp, axis=1)

    GLDZM_all = np.dstack(GLDZM_all)

    GLDZM25 = np.squeeze(np.sum(GLDZM_all, axis=2))
    GLDZM25D_F = np.transpose(CalcGLDZM(GLDZM25,ROI2D))


    GLDZM   = getGLDZM(ROI3D,ROIonly3D,levels3D)
    GLDZM3D_F = np.transpose(CalcGLDZM(GLDZM,ROI3D))



    return GLDZM2D_F, GLDZM3D_F, GLDZM25D_F

def  SmallDistEmph(Dj,nD,nS):

    J = np.arange (1,nD+1,1)
    tmp = np.divide(Dj , np.float_power(J,2))
    f_SDE = np.sum(tmp)/nS

    return f_SDE


def LargeDistEmph(Dj,nD,nS):

    J = np.arange (1,nD+1,1)
    tmp = np.multiply(Dj , np.float_power(J,2))
    f_LDE = np.sum(tmp)/nS
    return f_LDE


def LowGLCountRmph(Di,nG,nS):

    I = np.transpose(np.arange (1,nG+1,1))
    tmp = np.divide(Di , np.float_power(I,2))
    f_LGLZE = np.sum(tmp)/nS
    return f_LGLZE

def HighGLCountEmph(Di,nG,nS):

    I = np.transpose(np.arange (1,nG+1,1))
    tmp = np.multiply(Di , np.float_power(I,2))
    f_HGLZE = np.sum(tmp)/nS
    return f_HGLZE


def SmallDistLowGLEmph(GLDZM,nG,nD,nS):


    rangeX = np.arange(1, nG+1 , 1)
    rangeY = np.arange(1, nD+1 , 1)
    I , J = np.meshgrid(rangeX,rangeY)

    tmp = np.divide(GLDZM ,  np.transpose (np.multiply (  np.float_power(I,2) ,   np.float_power(J,2))))
    f_SRLGLE = np.sum(tmp)/nS

    return f_SRLGLE


def SmallDistHighRL(GLDZM,nG,nD,nS):

    rangeX = np.arange(1, nG+1 , 1)
    rangeY = np.arange(1, nD+1 , 1)
    I , J = np.meshgrid(rangeX,rangeY)

    tmp = np.multiply( np.float_power(I,2)  ,   np.divide ( np.transpose (GLDZM) ,   np.float_power(J,2)))

    f_SRHGLE = np.sum(tmp)/nS

    return f_SRHGLE


def LargeDistLowGLEmph(GLDZM,nG,nD,nS):

    rangeX = np.arange(1, nG+1 , 1)
    rangeY = np.arange(1, nD+1 , 1)
    I , J = np.meshgrid(rangeX,rangeY)

    tmp = np.multiply( np.float_power(J,2)  ,   np.divide (  np.transpose (GLDZM ),   np.float_power(I,2)))

    f_LRLGLE = np.sum(tmp)/nS

    return f_LRLGLE


def LargeDistHighGLEmph(GLDZM,nG,nD,nS):

    rangeX = np.arange(1, nG+1 , 1)
    rangeY = np.arange(1, nD+1 , 1)
    I , J = np.meshgrid(rangeX,rangeY)

    tmp = np.multiply( np.float_power(J,2)  ,   np.multiply (  np.float_power(I,2) , np.transpose (GLDZM)))

    f_LRHGLE = np.sum(tmp)/nS

    return f_LRHGLE

def GLnonUnif(Di,nS):
    f_GLNU = np.sum(np.float_power(Di,2)) / nS
    return f_GLNU


def GLnonUnifNormzd(Di,nS):

    f_GLNUN = np.sum(  np.float_power(Di,2)  )   / np.float_power(nS,2)

    return f_GLNUN


def ZoneDistNonUnif(Dj,nS):

    f_ZDNU = np.sum( np.float_power(Dj,2)) / nS

    return f_ZDNU


def ZoneDistNonUnifNormzd(Dj,nS):

    f_ZDNUN = np.sum( np.float_power(Dj,2)) / np.float_power(nS,2)

    return f_ZDNUN


def ZonePercentage(nS,nV):

    f_ZP = nS / nV
    return f_ZP


def GLVar(GLDZM,nG,nD,nS):

    Pij = GLDZM / nS


    rangeX = np.arange(1, nG+1 , 1)
    rangeY = np.arange(1, nD+1 , 1)
    I , J = np.meshgrid(rangeX,rangeY)
    
    mu = np.sum( np.multiply(np.transpose(I) , Pij))
    tmp = np.multiply(np.transpose (np.float_power((I - mu),2)) , Pij)
    f_GLV = np.sum(tmp)
    return f_GLV

def ZoneDistVar(GLDZM,nG,nD,nS):

    Pij = GLDZM / nS
    rangeX = np.arange(1, nG+1 , 1)
    rangeY = np.arange(1, nD+1 , 1)
    I , J = np.meshgrid(rangeX,rangeY)

    mu = np.sum(np.sum( np.multiply(np.transpose(J) , Pij)))
    tmp = np.multiply(np.transpose(np.float_power((J - mu),2)) , Pij)
    f_ZDV = np.sum(tmp)
    return f_ZDV


def ZoneDistEntropy(GLDZM,nS):

    Pij = GLDZM / nS
    tmp = np.multiply(Pij , np.log2(Pij+ np.finfo(float).tiny))
    f_ZD = -np.sum(tmp)

    return f_ZD











def CalcGLDZM(GLDZM,ROI):

    try:
        nf = GLDZM.shape[1]
    except:
        GLDZM = np.expand_dims(GLDZM,-1)


    sumGLDZM = np.sum(GLDZM,axis=0)

    try:
        stop = np.max(np.where(sumGLDZM > 0))
        GLDZM = GLDZM[:,:stop+2]
    except:
        GLDZM = GLDZM

    nG = GLDZM.shape[0]         
    nD = GLDZM.shape[1]           
    nS = int(np.sum(GLDZM))             
    Di = np.sum(GLDZM , axis =1).astype(np.uint32)         
    Dj = np.sum(GLDZM , axis =0).astype(np.uint32) 
    nV = np.where(~ np.isnan(ROI))[0].shape[0]


    ArrayOut = []

    ArrayOut.append(SmallDistEmph(Dj,nD,nS)) 
    ArrayOut.append(LargeDistEmph(Dj,nD,nS)) 
    ArrayOut.append(LowGLCountRmph(Di,nG,nS)) 
    ArrayOut.append(HighGLCountEmph(Di,nG,nS)) 
    ArrayOut.append(SmallDistLowGLEmph(GLDZM,nG,nD,nS)) 
    ArrayOut.append(SmallDistHighRL(GLDZM,nG,nD,nS)) 
    ArrayOut.append(LargeDistLowGLEmph(GLDZM,nG,nD,nS)) 
    ArrayOut.append(LargeDistHighGLEmph(GLDZM,nG,nD,nS)) 
    ArrayOut.append(GLnonUnif(Di,nS)) 
    ArrayOut.append(GLnonUnifNormzd(Di,nS)) 
    ArrayOut.append(ZoneDistNonUnif(Dj,nS)) 
    ArrayOut.append(ZoneDistNonUnifNormzd(Dj,nS)) 
    ArrayOut.append(ZonePercentage(nS,nV))  
    ArrayOut.append(GLVar(GLDZM,nG,nD,nS))  
    ArrayOut.append(ZoneDistVar(GLDZM,nG,nD,nS)) 
    ArrayOut.append(ZoneDistEntropy(GLDZM,nS)) 




    return ArrayOut


def bwdistNew(ROIOnly):


    tmpimg = ROIOnly.copy()
    nD = ROIOnly.ndim
    flg = 0

    if nD == 1 or ( ROIOnly.shape[0] == 1 or ROIOnly.shape[1]  == 1): 
        tmpimg= np.tile(ROIOnly,(1,int(np.sum(ROIOnly))))
        flg = 1
    

    if nD == 2: 
        con = 4 
    elif nD == 3:
        con = 6 
    else:
        raise('What is going on with the dimenstions?!!! It should be 1, 2 or 3.')
      


    D = np.zeros(tmpimg.shape) 
    for d in range(0, np.max(tmpimg.shape)):
        # edges = bwperim(tmpimg,con)

        edges = bwperim(tmpimg,con)
        
        if np.sum(edges) == 0:
            continue

        D = D + (d+1)*edges
        tmpimg = tmpimg - edges
        

    if flg:
        D = D[:, int(np.ceil( np.array (D.shape[1])/2))-1]
        if ROIOnly.shape[0] == 1:
            D = np.transpose(D)

    return D

def getGLDZM(ROIBox,ROIMask,levels):

    nLevel = len(levels)
    if nLevel > 100:
        adjust = 10000
    else:
        adjust = 1000


    levelTemp = np.max(levels)+1
    # ROIBox[ROIBox == np.nan] = levelTemp
    ROIBox = np.nan_to_num(ROIBox, nan=levelTemp)

    levels = np.append(levels, levelTemp)



    uniqueVect = np.round(levels*adjust)/adjust
    ROIBox=np.round(ROIBox*adjust)/adjust
    nGL = len(levels) - 1


    er = 0

    GLDZM = np.zeros((nGL,int(np.ceil(np.max(np.array(ROIBox.shape)/2)))))

    unqGLs = np.unique(ROIBox)

    unqGLs = unqGLs[unqGLs!=levelTemp]


    if ROIMask.ndim == 3:

        ROIOnlyP = np.pad(ROIMask,((1, 1), (1, 1), (1,1)))
        ROIBoxP = np.pad(ROIBox,((1, 1), (1, 1), (1,1)))
        ROIBoxP[ROIBoxP == 0] = levelTemp
        

        try:
            distmap = bwdistNew(ROIOnlyP)
        except:
            distmap = bwdistNew(ROIOnlyP.astype(np.float16))


        distmap[distmap==0] = np.nan
        
        for gl in range(0,unqGLs.shape[0]):

            temp = ROIOnlyP.copy()
            temp[ROIBoxP!=unqGLs[gl]] = 0
            temp[ROIBoxP==unqGLs[gl]] = 1
            temp = temp.astype(np.uint8)

            # connObjects = bwconncomp(temp,26)
            # connObjects = skimage.measure.label(temp, connectivity=26)
            if ROIBoxP.ndim == 2:
                structure = scipy.ndimage.generate_binary_structure(2, 8)
            else:
                structure = scipy.ndimage.generate_binary_structure(3, 27)

            # try:
            #    labeled_array, nConn = scipy.ndimage.label( input= temp, structure = structure)
            # except:
            #     labeled_array = temp.copy()
            #     nConn = 1
            
            # try:
            labeled_array, nConn = scipy.ndimage.label( input= temp, structure = structure)
            # nConn = len(connObjects['PixelIdxList'])
            
            # nConn = len(connObjects['PixelIdxList'])
            
            for c in range(0,nConn):
                

                arr = labeled_array == (c+1)
                arr.astype(np.uint8) 
                tmpROI = np.zeros(distmap.shape)
                tmpROI[arr == 1] = 1
                tmpROI[tmpROI == 0] = np.nan
                tmpDist = np.multiply(tmpROI , distmap)

                Dist = np.nanmin(np.nanmin(  tmpDist ,axis=0 ))

                if math.isnan(Dist) == False:
                    Dist = int(Dist)

                idx = np.where(unqGLs[gl]==uniqueVect)[0][0]


                try:
                    GLDZM[idx,Dist-1] = GLDZM[idx,Dist-1] + 1
                except:
                    # print('idx = ',str(idx))
                    # print('gl = ',str(gl))
                    # print('idx = ',str(uniqueVect[idx]))
                    # print('unqGLs(gl) = ',str(unqGLs[gl]))
                    er = 0
            # except:
            #     er = 0
        
    elif ROIMask.ndim <= 2:

        ROIOnlyP = np.pad(ROIMask,((1, 1), (1,1)))
        ROIBoxP = np.pad(ROIBox,((1, 1), (1, 1)))
        ROIBoxP[ROIBoxP == 0] = levelTemp
        

        try:
            distmap = bwdistNew(ROIOnlyP)
        except:
            distmap = bwdistNew(ROIOnlyP.astype(np.float16))
        
        distmap[distmap==0]=np.nan
        
        for gl in range(0,unqGLs.shape[0]):
            
            temp = ROIOnlyP.copy()
            temp[ROIBoxP != unqGLs[gl]] = 0
            temp[ROIBoxP == unqGLs[gl]] = 1
            temp = temp.astype(np.uint8)
            # connObjects = bwconncomp(temp,8)
            # connObjects = skimage.measure.label(temp, connectivity=8)
            if ROIBoxP.ndim == 2:
                structure = scipy.ndimage.generate_binary_structure(2, 8)
            else:
                structure = scipy.ndimage.generate_binary_structure(3, 27)

            # try:
            labeled_array, nConn = scipy.ndimage.label( input= temp, structure = structure)
            # except:
            #     if ROIBoxP.ndim == 2:
            #         # structure = scipy.ndimage.generate_binary_structure(2, 1)
            #         structure = np.zeros((3,3))
            #         structure[1, 1] = 1
            #         structure = structure.astype(dtype=bool)
            #     else:
            #         structure = scipy.ndimage.generate_binary_structure(3, 1)
            #     labeled_array, nConn = scipy.ndimage.label(input= temp,structure = structure)
            #     # labeled_array = temp.copy()
            #     # nConn = 1

            # try:
            # labeled_array, nConn = scipy.ndimage.label( input= temp, structure = structure)
            # nConn = len(connObjects['PixelIdxList'])
            
            for c in range(0,nConn):
                
                arr = labeled_array == (c+1)
                arr.astype(np.uint8) 
                tmpROI = np.zeros(distmap.shape)
                tmpROI[arr == 1] = 1
                tmpROI[tmpROI == 0] = np.nan
                tmpDist = np.multiply(tmpROI , distmap)

                Dist = np.nanmin(np.nanmin(  tmpDist ,axis=0 ))
                # Dist = np.nanmin(tmpDist ,axis=0)

                if math.isnan(Dist) == False:
                    Dist = int(Dist)

                idx = np.where(unqGLs[gl]==uniqueVect)[0][0]


                try:
                    GLDZM[idx,Dist-1] = GLDZM[idx,Dist-1] + 1
                except:
                    # print('idx = ',str(idx))
                    # print('gl = ',str(gl))
                    # print('idx = ',str(uniqueVect[idx]))
                    # print('unqGLs(gl) = ',str(unqGLs[gl]))
                    er = 0
            # except:
            #     er = 0


    # sumGLDZM = np.sum(GLDZM,axis=0)
    # stop = np.max(np.where(sumGLDZM > 0))
    # GLDZM = GLDZM[:,:stop+1]

    # stop = np.where(np.sum(GLDZM))[-1]
    # GLDZM[:,(stop+2):] = []

    return GLDZM
