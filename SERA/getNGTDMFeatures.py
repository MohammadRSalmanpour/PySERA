
from scipy.io import savemat, loadmat
import os
import numpy as np
import sys
import skimage
import cv2
import itertools
import scipy 
import math
from SERAutilities import *
import collections.abc

def getNGTDMtextures(NGTDM,countValid,Aarray):



    nV = np.sum(countValid)
    Pi = np.divide(countValid,nV).flatten(order='F')
    nG = len(NGTDM)
    nGp = np.count_nonzero(Pi)
    pValid = np.where(Pi>0)[0]
    pValid =pValid +1
    nValid = len(pValid)

    xx = np.dot(np.transpose( Pi) , NGTDM )
    Coarseness = np.float_power(((   xx    ) + np.finfo(float).eps   ),-1)
    Coarseness = np.min([Coarseness , np.float_power(10,6)])

    val = 0
    for i in range(0,nG):
        for j in range(0,nG):
            val = val + Pi[i]*Pi[j]* np.float_power([i-j],2)

    Contrast = val * np.sum(NGTDM)  / (nGp*(nGp-1)*nV)

    denom = 0
    val = 0
    val_Strength = 0
    for i in range(0,nValid):
        for j in range(0,nValid):
            denom = denom + np.abs(pValid[i]*Pi[pValid[i]-1]-pValid[j]*Pi[pValid[j]-1])
            val = val + (np.abs(pValid[i]-pValid[j])  /  (nV*(Pi[pValid[i]-1] + Pi[pValid[j]-1])))*(Pi[pValid[i]-1]*NGTDM[pValid[i]-1] + Pi[pValid[j]-1]*NGTDM[pValid[j]-1])
            val_Strength = val_Strength + (Pi[pValid[i]-1]+Pi[pValid[j]-1])* np.float_power((pValid[i]-pValid[j]),2)


    Busyness = (     np.dot(np.transpose( Pi) , NGTDM )  )/denom
    if Busyness == np.inf: 
        Busyness=0
    Complexity = val
    Strength = val_Strength/(  np.finfo(float).eps +np.sum(NGTDM))


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

    textures = [Coarseness,Contrast,Busyness,Complexity,Strength]
    
    return textures




def getNGTDM(ROIOnly2,levels):

    ROIOnly = ROIOnly2.copy()
    if ROIOnly.ndim == 2:
        twoD = 1
    else:
        twoD = 0
    
    nLevel = len(levels)
    if nLevel > 100:
        adjust = 10000
    else:
        adjust = 1000

    if twoD:
        ROIOnly = np.pad(ROIOnly,((1,1),(1,1)),mode='constant' , constant_values=np.nan)
    else:
        ROIOnly = np.pad(ROIOnly,((1,1),(1,1),(1,1)),mode='constant' , constant_values=np.nan)
    

    uniqueVol = np.round(levels*adjust)/adjust
    ROIOnly=np.round(ROIOnly*adjust)/adjust
    NL = len(levels)


    temp = ROIOnly.copy()
    for i in range(0,NL):
        ROIOnly[temp==uniqueVol[i]] = i+1
    


    NGTDM = np.zeros((NL,1))
    countValid = np.zeros((NL,1))


    if twoD:
        i,j,sl = ind2sub(ROIOnly)
        posValid = np.column_stack((i , j))
        nValid_temp = posValid.shape[0]
        weights = np.ones(9).astype(np.int32)
        # weights = weights
        Aarray = np.zeros((nValid_temp,2)).astype(np.int32)
        for n in range(0,nValid_temp):
            neighbours = np.zeros((9,1)).astype(np.int32)
            # neighbours = ROIOnly[posValid[n,0]-1:posValid[n,0]+2  ,   posValid[n,1]-1:posValid[n,1]+2].flatten(order='F')
            neighbours = ROIOnly[posValid[n,0]-1:posValid[n,0]+2  ,   posValid[n,1]-1:posValid[n,1]+2]
            neighbours = neighbours.flatten(order='F')
            neighbours = np.multiply(neighbours , weights)
            # neighbours = np.reshape(neighbours,(9,1))
            value = int(neighbours[4])-1
            neighbours[4] = np.nan
            sum_wei = 0
            for nei in range(0,neighbours.shape[0]):
                if ~np.isnan(neighbours[nei]):
                    sum_wei += weights[nei]
            if sum_wei != 0:
                neighbours = neighbours / sum_wei
            # neighbours.pop(4)
            neighbours = np.delete(neighbours,4,None)
            # neighbours[4] = []

            fin1 = np.where(~np.isnan(neighbours))
            
            if len(fin1[0]) > 0 :
            # if neighbours[ ~np.isnan(neighbours)].shape[0] > 0  or neighbours[ ~np.isnan(neighbours)].shape[1] > 0:

                sum_nei = 0
                for nei in range(0,neighbours.shape[0]):
                    if ~ np.isnan(neighbours[nei]):
                        sum_nei += neighbours[nei]
                NGTDM[value] = NGTDM[value] + float(np.abs(value+1 - sum_nei))
                countValid[value] = countValid[value] + 1

    else:
        i,j,k = ind2sub(ROIOnly)
        posValid = np.column_stack((i , j, k))
        nValid_temp = posValid.shape[0]
        weights = np.ones(27).astype(np.int32)
        # weights = weights
        Aarray = np.zeros((nValid_temp,2)).astype(np.int32)
        for n in range(0,nValid_temp):
            neighbours = np.zeros((27,1)).astype(np.int32)
            neighbours = ROIOnly[posValid[n,0]-1:posValid[n,0]+2  ,   posValid[n,1]-1:posValid[n,1]+2,   posValid[n,2]-1:posValid[n,2]+2].flatten(order='F')
            neighbours = np.multiply(neighbours , weights)

            value = int(neighbours[13])-1
            neighbours[13] = np.nan
            sum_wei = 0
            for nei in range(0,neighbours.shape[0]):
                if ~ np.isnan(neighbours[nei]):
                    sum_wei += weights[nei]

            if sum_wei != 0:
                neighbours = neighbours / sum_wei
            neighbours = np.delete(neighbours,13,None)

            fin1 = np.where(~np.isnan(neighbours))

            if len(fin1[0]) > 0:
            # if neighbours[ ~np.isnan(neighbours)].shape[0] > 0  or neighbours[ ~np.isnan(neighbours)].shape[1] > 0:
                
                sum_nei = 0
                for nei in range(0,neighbours.shape[0]):
                    if ~ np.isnan(neighbours[nei]):
                        sum_nei += neighbours[nei]
                
                Ai = np.abs(value+1-sum_nei)
                NGTDM[value] = NGTDM[value] + float(Ai)
                countValid[value] = countValid[value] + 1
                Aarray[n,:] = [value, float(Ai)]




    return NGTDM,countValid,Aarray



def getNGTDMtex(ROI2D2,ROI3D2,levels2D1,levels3D1):

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

    for s in range(0,nZ): 
        NGTDM,countValid,Aarray = getNGTDM(ROI2D[:,:,s],levels2D)

        # NGTDM2D_all[:,s] = NGTDM.flatten()
        # count_valid_all[:,s] = countValid.flatten()

        NGTDM2D_all.append(NGTDM.flatten(order='F'))
        count_valid_all.append(countValid.flatten(order='F'))

        Aarray = Aarray.flatten(order='F')
        NGTDMstr = np.array(getNGTDMtextures(NGTDM,countValid,Aarray))
        
        if s == 0:
            FeatTmp = NGTDMstr
        else:    
            FeatTmp = np.column_stack((FeatTmp , NGTDMstr))
        

    NGTDM2D = np.nanmean(FeatTmp , axis=1).flatten(order='F')

    NGTDM2D_all = np.dstack(NGTDM2D_all)[0]
    count_valid_all = np.dstack(count_valid_all)[0]

    NGTDM25 = np.sum(NGTDM2D_all, axis=1)
    NGTDM25D = np.transpose(np.array(getNGTDMtextures(NGTDM25,   np.sum(count_valid_all,axis=1),Aarray)))



    NGTDM,countValid,Aarray   = getNGTDM(ROI3D,levels3D)
    NGTDM3Dstr = getNGTDMtextures(NGTDM,countValid,Aarray)
    NGTDM3D = np.transpose(np.array(NGTDM3Dstr))

    for i in range( 0 , NGTDM3D.shape[0]):
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







