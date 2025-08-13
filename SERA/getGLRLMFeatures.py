

from scipy.io import savemat, loadmat
import os
import numpy as np
import sys
import skimage
import cv2
import itertools
import scipy 
import cv2
from scipy.sparse import spdiags
from glrlm import GLRLM
import collections.abc
from itertools import count, product
import numpy as np


# -------------------------------------------------------------------------
# function [GLRLM2D_Cmb, GLRLM2D_Avg] = getGLRLM3Dtex(ROIonly,levels)
# -------------------------------------------------------------------------
# DESCRIPTION: 
# This function calculates the GLRLM matrix for each slice of an ROI.
# Every slice is calculated separately, then features are calculated.
# 
# Like the grey level co-occurrence matrix, GLRLM also assesses the
# distribution of discretised grey levels in an image or in a stack of
# images. However, instead of assessing the combination of levels between
# neighbouring pixels or voxels, GLRLM assesses grey level run # lengths.
# Run length counts the frequency of consecutive voxels with discretised
# grey level i along  direction \Delta.
# -------------------------------------------------------------------------
# INPUTS:
# - ROIonly: Smallest box containing the ROI, with the imaging data ready 
#            for texture analysis computations. Voxels outside the ROI are 
#            set to NaNs. 
# - levels: number of bins (for fixed number of bins method) or bin size
#           (for bin size method) 
# Note: ROIonly is the outputs of prepareVolume.m
# -------------------------------------------------------------------------
# OUTPUTS:
# - GLRLM3D_Cmb: 3D GLRLM features: First merging GLRLMs for all directions
#               then calculate features for the combined GLRLM matrix.
# - GLRLM3D_Avg: 3D GLRLM features calculate GLRLM features for each
#               direction, then average over all directions.
# -------------------------------------------------------------------------
# AUTHOR(S): 
# - Saeed Ashrafinia
# - Mahdi Hosseinzadeh
# -------------------------------------------------------------------------
# HISTORY:
# - Creation: December 2022
# -------------------------------------------------------------------------


def getGLRLM3Dtex(ROIonly2,levels2):

    levels = levels2.copy()

    ROIonly = ROIonly2.copy()
    GLRLM, GLRLM_D = getGLRLM(ROIonly , levels)


    GLRLM2D_Cmb = CalcGLRLM(GLRLM, ROIonly)       
    FeatsTmp = CalcGLRLM(GLRLM_D, ROIonly)
    # FeatsTmp = FeatsTmp[FeatsTmp != np.nan]
    GLRLM2D_Avg = np.nanmean(FeatsTmp , axis=1)

    return GLRLM2D_Cmb, GLRLM2D_Avg


# -------------------------------------------------------------------------
# function [GLRLM2D_Cmb, GLRLM2D_Avg] = getGLRLM2Dtex(ROIonly,levels)
# -------------------------------------------------------------------------
# DESCRIPTION: 
# This function calculates the GLRLM matrix for each slice of an ROI.
# Every slice is calculated separately, then features are calculated.
# 
# Like the grey level co-occurrence matrix, GLRLM also assesses the
# distribution of discretised grey levels in an image or in a stack of
# images. However, instead of assessing the combination of levels between
# neighbouring pixels or voxels, GLRLM assesses grey level run # lengths.
# Run length counts the frequency of consecutive voxels with discretised
# grey level i along  direction \Delta.
# -------------------------------------------------------------------------
# INPUTS:
# - ROIonly: Smallest box containing the ROI, with the imaging data ready 
#            for texture analysis computations. Voxels outside the ROI are 
#            set to NaNs. 
# - levels: number of bins (for fixed number of bins method) or bin size
#           (for bin size method) 
# Note: RIOonly is the outputs of prepareVolume.m
# -------------------------------------------------------------------------
# OUTPUTS:
# - GLRLM2D_Cmb: 2D GLRLM features: merging GLRLM for different slice of
#                 the volume, then calculate features for the combined
#                 GLRLM matrix. 
# - GLRLM2D_Avg: 2D GLRLM features: calculate GLRLM features for each
#                slice first, then average over all slices.
# -------------------------------------------------------------------------
# AUTHOR(S): 
# - Saeed Ashrafinia
# - Mahdi Hosseinzadeh
# -------------------------------------------------------------------------
# HISTORY:
# - Creation: December 2022
# -------------------------------------------------------------------------


def getGLRLM2Dtex(ROIonly2,levels2):

    levels = levels2.copy()
    ROIonly = ROIonly2.copy()
    nx, ny, nZ = ROIonly.shape
    FeatTmp = []
    FeatTmp_D = []
    GLRLM_all = np.zeros((levels.shape[0] , np.max([nx,ny]) , nZ))
    GLRLM_D_all = np.zeros((levels.shape[0] , np.max([nx,ny]) , 4 , nZ))
    for s in range(0,nZ):
        GLRLM, GLRLM_D = getGLRLM(ROIonly[:,:,s] , levels)
        GLRLMCal =  CalcGLRLM(GLRLM, ROIonly[:,:,s])
        GLRLM_D_Cal = CalcGLRLM(GLRLM_D[:,:,0:4], ROIonly[:,:,s])

        if s == 0:
            FeatTmp = GLRLMCal
            FeatTmp_D = GLRLM_D_Cal
        else:    
            FeatTmp = np.column_stack((FeatTmp , GLRLMCal))
            FeatTmp_D = np.dstack((FeatTmp_D , GLRLM_D_Cal))

        GLRLM_all[:,0:GLRLM.shape[1],s] = GLRLM
        GLRLM_D_all[:,0:GLRLM_D.shape[1],0:4,s] = GLRLM_D[:,:,0:4]


    Feats_MSKD = np.nanmean(FeatTmp , axis=1)
    try:
        Feats_MSKD = np.nanmean(Feats_MSKD , axis=1)
    except:
        Feats_MSKD = Feats_MSKD

    Feats_MSKD[12] = Feats_MSKD[12] /4

    FeatsTmp2_D = np.reshape( a= FeatTmp_D, newshape=(16,FeatTmp_D.shape[1]*FeatTmp_D.shape[2]),order='F')
    Feats_KSKD = np.nanmean(FeatsTmp2_D,axis=1)
    try:
        Feats_KSKD = np.nanmean(Feats_KSKD , axis=1)
    except:
        Feats_KSKD = Feats_KSKD

    GLRLM_MergeSlice_KeepDirs = np.zeros((GLRLM_D_all.shape[0], GLRLM_D_all.shape[1],GLRLM_D_all.shape[2]))
    GLRLM_MergeSlice_KeepDirs[:,:,:] = np.sum(GLRLM_D_all , axis=3)
    tmp = np.sum(np.sum(GLRLM_MergeSlice_KeepDirs , axis=0),axis=0)

    GLRLMnorm_MergeSlice_KeepDirs = np.divide(GLRLM_MergeSlice_KeepDirs , np.tile(tmp ,(GLRLM_D_all.shape[0], GLRLM_D_all.shape[1],1)))
    GLRLMnorm_MergeSlice_KeepDirs = np.nan_to_num(GLRLMnorm_MergeSlice_KeepDirs, nan=0)

    Feats_KSMD = CalcGLRLM(GLRLMnorm_MergeSlice_KeepDirs, ROIonly)
    try:
        Feats_KSMD = np.nanmean(Feats_KSMD , axis=1)
    except:
        Feats_KSMD = Feats_KSMD

    GLRLM_AllMerged   = np.sum(np.sum(GLRLM_D_all,axis=2),axis=2)
    GLRLMnorm_AllMerged = GLRLM_AllMerged / np.sum(GLRLM_AllMerged)

    Feats_MSMD = CalcGLRLM(GLRLMnorm_AllMerged, ROIonly)
    try:
        Feats_MSMD = np.nanmean(Feats_MSMD , axis=1)
    except:
        Feats_MSMD = Feats_MSMD


    return Feats_KSKD, Feats_MSKD, Feats_KSMD, Feats_MSMD



def diagk(X,k):
    if X.ndim == 2:
        D = np.diag(X,k)
    else:
        is_all_zero = np.all((X == 0))
        if is_all_zero == False and 0 <= k and 1+k <= X.shape[1]:
            D = X[1+k]
        elif is_all_zero == False and k < 0 and 1-k <= X.shape[0]:
            D = X[1-k]
        else:
            # D = np.zeros((X.shape[0],1))
            D = np.array(0)
    
    return D



def spdiags(arg1, arg2 = None):

    A = arg1
    if arg2 is None:
        i,j = np.where(A)
        d = np.sort(j-i)
        index_len = np.insert(d,0,A.shape[0])
        lenn = np.diff( index_len )
        d = d [ lenn  != 0  ]
    else:
        d = arg2

    m,n = A.shape
    p = d.shape[0]
    B = np.zeros((np.min([m,n]),p))

    for k in range(0,p):
        if m >= n:
            i = np.arange(np.max([1,1+d[k]]),np.min([n,m+d[k]])+1,1)
        else:
            i = np.arange(np.max([1,1-d[k]]),np.min([m,n-d[k]])+1,1)
        
        i = list(i - 1)
        i = [int(item) for item in i]

        B[i,k] = diagk(A,d[k])
    
    res1 = B
    res2 = d

    return res1
    # return res1,res2

# def accum(accmap, a, func=None, size=None, fill_value=0, dtype=None):
#     """
#     An accumulation function similar to Matlab's `accumarray` function.

#     Parameters
#     ----------
#     accmap : ndarray
#         This is the "accumulation map".  It maps input (i.e. indices into
#         `a`) to their destination in the output array.  The first `a.ndim`
#         dimensions of `accmap` must be the same as `a.shape`.  That is,
#         `accmap.shape[:a.ndim]` must equal `a.shape`.  For example, if `a`
#         has shape (15,4), then `accmap.shape[:2]` must equal (15,4).  In this
#         case `accmap[i,j]` gives the index into the output array where
#         element (i,j) of `a` is to be accumulated.  If the output is, say,
#         a 2D, then `accmap` must have shape (15,4,2).  The value in the
#         last dimension give indices into the output array. If the output is
#         1D, then the shape of `accmap` can be either (15,4) or (15,4,1) 
#     a : ndarray
#         The input data to be accumulated.
#     func : callable or None
#         The accumulation function.  The function will be passed a list
#         of values from `a` to be accumulated.
#         If None, numpy.sum is assumed.
#     size : ndarray or None
#         The size of the output array.  If None, the size will be determined
#         from `accmap`.
#     fill_value : scalar
#         The default value for elements of the output array. 
#     dtype : numpy data type, or None
#         The data type of the output array.  If None, the data type of
#         `a` is used.

#     Returns
#     -------
#     out : ndarray
#         The accumulated results.

#         The shape of `out` is `size` if `size` is given.  Otherwise the
#         shape is determined by the (lexicographically) largest indices of
#         the output found in `accmap`.


#     Examples
#     --------
#     >>> from numpy import array, prod
#     >>> a = array([[1,2,3],[4,-1,6],[-1,8,9]])
#     >>> a
#     array([[ 1,  2,  3],
#            [ 4, -1,  6],
#            [-1,  8,  9]])
#     >>> # Sum the diagonals.
#     >>> accmap = array([[0,1,2],[2,0,1],[1,2,0]])
#     >>> s = accum(accmap, a)
#     array([9, 7, 15])
#     >>> # A 2D output, from sub-arrays with shapes and positions like this:
#     >>> # [ (2,2) (2,1)]
#     >>> # [ (1,2) (1,1)]
#     >>> accmap = array([
#             [[0,0],[0,0],[0,1]],
#             [[0,0],[0,0],[0,1]],
#             [[1,0],[1,0],[1,1]],
#         ])
#     >>> # Accumulate using a product.
#     >>> accum(accmap, a, func=prod, dtype=float)
#     array([[ -8.,  18.],
#            [ -8.,   9.]])
#     >>> # Same accmap, but create an array of lists of values.
#     >>> accum(accmap, a, func=lambda x: x, dtype='O')
#     array([[[1, 2, 4, -1], [3, 6]],
#            [[-1, 8], [9]]], dtype=object)
#     """

#     # Check for bad arguments and handle the defaults.
#     if accmap.shape[:a.ndim] != a.shape:
#         raise ValueError("The initial dimensions of accmap must be the same as a.shape")
#     if func is None:
#         func = np.sum
#     if dtype is None:
#         dtype = a.dtype
#     if accmap.shape == a.shape:
#         accmap = np.expand_dims(accmap, -1)
#     adims = tuple(range(a.ndim))
#     if size is None:
#         size = 1 + np.squeeze(np.apply_over_axes(np.max, accmap, axes=adims))
#     size = np.atleast_1d(size)

#     # Create an array of python lists of values.
#     vals = np.empty(size, dtype='O')
#     for s in product(*[range(k) for k in size]):
#         vals[s] = []
#     for s in product(*[range(k) for k in a.shape]):
#         indx = tuple(accmap[s])
#         val = a[s]
#         vals[indx].append(val)

#     # Create the output array.
#     out = np.empty(size, dtype=dtype)
#     for s in product(*[range(k) for k in size]):
#         if vals[s] == []:
#             out[s] = fill_value
#         else:
#             out[s] = func(vals[s])

#     return out



def accum(accmap = [],size=None):

    out = np.zeros((size[0],size[1]), dtype=np.int32)

    for i in accmap:
        out[i[0]][i[1]] = out[i[0]][i[1]] + 1 


    return out

def rle_0(si,NL):


    m,n=si.shape

    oneglrlm=np.zeros((NL,n))

    for i in range(0,m):
        x=si[i,:]
        a = x[:-1]
        b = x[1:]
        find = np.where( a != b )[0]
        find = find + 1
        if find.shape[0] != 0:
            index = np.insert(find,-1,x.shape[0])
            index = np.sort(index)
        else:
            index = []
            index.append(x.shape[0])
            
            
        # if  isinstance(index,collections.abc.Sequence) == False: 
        # index = [0,index]
        index_len = index.copy()
        index_len = np.insert(index_len,0,0)
        lenn = np.diff( index_len )
        # else:
        #     index = np.insert(index,0,0)
        #     lenn = np.diff( index )
        
        accmap = []

        indexmiues = list(np.array(index) - 1)
        lenn = list(np.array(lenn) - 1)
        # indexmiues = np.array(index) - 1
        # indexmiues = np.array(index) - 1

        # if lenn.shape[0] == 1:    
        val = x[indexmiues] - 1
        val = [int(item) for item in val]
            # accmap.append([val ,lenn]) 
        # else:
        for gg in range(0,len(lenn)):
            accmap.append([val[gg] ,lenn[gg]])      


        temp = accum(accmap = accmap,size=[NL, n]) 
        # # temp = accum(a=x, accmap=np.transpose([val,lenn]),func = np.sum ,size =[NL, n] )
        oneglrlm = temp + oneglrlm
    

    return oneglrlm

def FindMaxLength(lst):
    # maxList = max((x) for x in lst)
    maxLength = max(len(x) for x in lst )
 
    return maxLength

def rle_45(seq,NL):

    m = len(seq)
    n = FindMaxLength(seq)

    oneglrlm= np.zeros((NL,n))

    for i in range(0,m):
        x=seq[i]

        a = x[:-1]
        b = x[1:]
        find = np.where( a != b )[0]
        find = find + 1
        if find.shape[0] != 0:
            index = np.insert(find,-1,x.shape[0])
            index = np.sort(index)
        else:
            index = []
            index.append(x.shape[0])
            
        index_len = index.copy()
        index_len = np.insert(index_len,0,0)
        lenn = np.diff( index_len )
        
        accmap = []

        indexmiues = list(np.array(index) - 1)
        lenn = list(np.array(lenn) - 1)


        val = x[indexmiues] - 1
        val = [int(item) for item in val]

        for gg in range(0,len(lenn)):
            accmap.append([val[gg] ,lenn[gg]])      


        temp = accum(accmap = accmap,size=[NL, n]) 

        oneglrlm = temp + oneglrlm
    

    return oneglrlm


def zigzag(SI):

    seq = []

    c = 1
    r = 1

    rmin = 1
    cmin = 1

    rmax = SI.shape[0]
    cmax = SI.shape[1]

    
    i = 1
    j = 1

    sq_up_begin=1

    sq_down_begin=1


    output = np.zeros((1, rmax * cmax)).flatten(order='F')

    # rmax =  rmax - 1
    # cmax = cmax - 1


    while (r <= rmax) and (c <= cmax):


        if np.mod(c + r, 2) == 0:    
            if r == rmin:
                output[i-1] = SI[r-1, c-1]
                if c == cmax:
                    r   = r + 1
                    sq_up_end = i
                    sq_down_begin = i+1
                    seq.append(output[sq_up_begin-1:sq_up_end])
                    j = j + 1

                else:
                    c = c + 1
                    sq_up_end = i
                    sq_down_begin = i+1

                    seq.append(output[sq_up_begin-1:sq_up_end])

                    j = j + 1

                i = i + 1

            elif (c == cmax) and (r < rmax):
                output[i-1] = SI[r-1, c-1]
                r = r + 1
                
                sq_up_end = i
                seq.append(output[sq_up_begin-1:sq_up_end])
                sq_down_begin =i+1
                j=j+1
                            
                i = i + 1

            elif (r > rmin) and (c < cmax):
                output[i-1] = SI[r-1, c-1]
                r = r - 1
                c = c + 1
                i = i + 1

        else:
            if (r == rmax) and (c <= cmax):
                output[i-1] = SI[r-1, c-1]
                c = c + 1
                sq_down_end = i
                seq.append(output[sq_down_begin-1:sq_down_end])
                sq_up_begin =i+1
                j = j + 1
                i = i + 1
            elif c == cmin:
                output[i-1] = SI[r-1, c-1]
                
                if r == rmax:
                    c = c + 1
                    
                    sq_down_end = i
                    seq.append(output[sq_down_begin-1:sq_down_end])
                    sq_up_begin =i+1
                    j = j + 1

                else:
                    r = r + 1
                    sq_down_end = i
                    seq.append(output[sq_down_begin-1:sq_down_end])
                    sq_up_begin =i+1
                    j = j + 1

                

                i = i + 1
            elif (r < rmax) and (c > cmin):
                output[i-1] = SI[r-1, c-1]
                r = r + 1
                c = c - 1
                i = i + 1


        if (r == rmax) and (c == cmax):
            output[i-1] = SI[r-1, c-1]
            sq_end = i
            seq.append(np.array([output[sq_end-1]]))

            break
    

    return seq


def fliplr(x):

    x = np.flip(x,1)

    return x

def getGLRLM(ROIonly,levels2):

    levels = levels2.copy()
    nLevel = len(levels)
    if nLevel > 100:
        adjust = 10000
    else:
        adjust = 1000
    


    levelTemp = np.max(levels)+1
    # ROIonly[np.isnan(ROIonly)] = levelTemp
    ROIonly = np.nan_to_num(ROIonly,nan = levelTemp)
    levels = np.append(levels, levelTemp)



    uniqueVol = np.round(levels*adjust)/adjust
    ROIonly=np.round(ROIonly*adjust)/adjust
    NL = len(levels) - 1



    sizeV = ROIonly.shape
    numInit = int(np.ceil(np.max(sizeV)*np.sqrt(3)))
    GLRLM = np.zeros((NL+1,numInit))
    GLRLM_D = np.zeros((NL+1,numInit,13))



    if ROIonly.ndim == 3:
        nComp = sizeV[2]
    else:
        nComp = 1
        ROIonly = np.expand_dims(ROIonly,axis=-1)
    
    
    for k in range (0,nComp):
        image = ROIonly[:,:,k].copy()
        uniqueIm = np.unique(image)
        NLtemp = len(uniqueIm)
        indexRow = np.zeros((NLtemp,1))
        temp = image.copy()
        for j in range (0,NLtemp):
            indexRow[j] = np.where(uniqueIm[j]==uniqueVol)[0] + 1
            image[temp==uniqueIm[j]] = j+1
        
        GLRLMtemp = rle_0(image,NLtemp)
        # app = GLRLM()
        # glrlm = app.get_features(image,NLtemp)

        nRun = GLRLMtemp.shape[1]
        # nRun = GLRLMtemp.shape[1]

        ind = list(indexRow[0:NLtemp].flatten(order='F')  - 1)
        # ind = list(indexRow[0:NLtemp].flatten())
        ind = [int(item) for item in ind]

        GLRLM[ind,0:nRun] = GLRLM[ind,0:nRun] + GLRLMtemp[0:NLtemp,0:nRun]
        GLRLM_D[ind,0:nRun,0] = GLRLM_D[ind,0:nRun,0] + GLRLMtemp[0:NLtemp,0:nRun] 
        
        GLRLMtemp = rle_0( np.transpose(image),NLtemp)
        nRun = GLRLMtemp.shape[1]

        GLRLM[ind,0:nRun] = GLRLM[ind,0:nRun] + GLRLMtemp[0:NLtemp,0:nRun]
        GLRLM_D[ind,0:nRun,1] = GLRLM_D[ind,0:nRun,1] + GLRLMtemp[0:NLtemp,0:nRun] 
        

        seq = zigzag(image)
        GLRLMtemp = rle_45(seq,NLtemp)
        nRun = GLRLMtemp.shape[1]

        GLRLM[ind,0:nRun] = GLRLM[ind,0:nRun] + GLRLMtemp[0:NLtemp,0:nRun]
        GLRLM_D[ind,0:nRun,2] = GLRLM_D[ind,0:nRun,2] + GLRLMtemp[0:NLtemp,0:nRun] 
        

        seq = zigzag(fliplr(image))
        GLRLMtemp = rle_45(seq,NLtemp)
        nRun = GLRLMtemp.shape[1]

        GLRLM[ind,0:nRun] = GLRLM[ind,0:nRun] + GLRLMtemp[0:NLtemp,0:nRun]
        GLRLM_D[ind,0:nRun,3] = GLRLM_D[ind,0:nRun,3] + GLRLMtemp[0:NLtemp,0:nRun] 
        

    if len(np.squeeze(ROIonly).shape) == 3:

        nComp = sizeV[0]
        image = np.zeros((sizeV[2],sizeV[1]))
        # print(np.mean(ROIonly))
        # print(np.nanmean(ROIonly))

        for k in range (0,nComp):
            for j in range (0,sizeV[2]):
                image[j,:] = ROIonly[k,:,j].copy()
            
            uniqueIm = np.unique(image)
            NLtemp = len(uniqueIm)
            indexRow = np.zeros((NLtemp,1))
            temp = image.copy()
            for j in range(0,NLtemp):
                indexRow[j] = np.where(uniqueIm[j]==uniqueVol)[0] + 1
                image[temp==uniqueIm[j]] = j+1
            
            
            GLRLMtemp = rle_0( np.transpose(image),NLtemp)
            nRun = GLRLMtemp.shape[1]

            ind = list(indexRow[0:NLtemp].flatten(order='F')  - 1)
            ind = [int(item) for item in ind]
            
            GLRLM[ind,0:nRun] = GLRLM[ind,0:nRun] + GLRLMtemp[0:NLtemp,0:nRun]
            GLRLM_D[ind,0:nRun,4] = GLRLM_D[ind,0:nRun,4] + GLRLMtemp[0:NLtemp,0:nRun] 
        
            
            seq = zigzag(image)
            GLRLMtemp = rle_45(seq,NLtemp)
            nRun = GLRLMtemp.shape[1]

            GLRLM[ind,0:nRun] = GLRLM[ind,0:nRun] + GLRLMtemp[0:NLtemp,0:nRun]
            GLRLM_D[ind,0:nRun,5] = GLRLM_D[ind,0:nRun,5] + GLRLMtemp[0:NLtemp,0:nRun] 
        
            seq = zigzag(fliplr(image))
            GLRLMtemp = rle_45(seq,NLtemp)
            nRun = GLRLMtemp.shape[1]

            GLRLM[ind,0:nRun] = GLRLM[ind,0:nRun] + GLRLMtemp[0:NLtemp,0:nRun]
            GLRLM_D[ind,0:nRun,6] = GLRLM_D[ind,0:nRun,6] + GLRLMtemp[0:NLtemp,0:nRun] 


        nComp = sizeV[1]
        image = np.zeros((sizeV[0],sizeV[2]))
        for k in range (0,nComp):
            for j in range (0,sizeV[2]):
                image[:,j] = ROIonly[:,k,j].copy()
            
            uniqueIm = np.unique(image)
            NLtemp = len(uniqueIm)
            indexRow = np.zeros((NLtemp,1))
            temp = image.copy()
            for j in range (0,NLtemp):
                indexRow[j] = np.where(uniqueIm[j]==uniqueVol)[0] + 1
                image[temp==uniqueIm[j]] = j + 1
                        

            seq = zigzag(image)
            GLRLMtemp = rle_45(seq,NLtemp)
            nRun = GLRLMtemp.shape[1]

            ind = list(indexRow[0:NLtemp].flatten(order='F')  - 1)
            ind = [int(item) for item in ind]

            GLRLM[ind,0:nRun] = GLRLM[ind,0:nRun] + GLRLMtemp[0:NLtemp,0:nRun]
            GLRLM_D[ind,0:nRun,7] = GLRLM_D[ind,0:nRun,7] + GLRLMtemp[0:NLtemp,0:nRun] 
        

            seq = zigzag(fliplr(image))
            GLRLMtemp = rle_45(seq,NLtemp)
            nRun = GLRLMtemp.shape[1]
            GLRLM[ind,0:nRun] = GLRLM[ind,0:nRun] + GLRLMtemp[0:NLtemp,0:nRun]
            GLRLM_D[ind,0:nRun,8] = GLRLM_D[ind,0:nRun,8] + GLRLMtemp[0:NLtemp,0:nRun] 
        

        image = np.zeros((sizeV[2],sizeV[1]))
        temp = np.random.rand(sizeV[2],sizeV[1])
        # d = np.arange(- (temp.shape[0]-1),temp.shape[1],1)
        # d = np.arange(- (temp.shape[0]-1),1,1)
        diagTemp = spdiags(temp)
        szDiag = diagTemp.shape
        diagMat1 = np.zeros((szDiag[0],szDiag[1],sizeV[0]))
        diagMat2 = np.zeros((diagTemp.shape[0],diagTemp.shape[1],sizeV[0]))
        for k in range (0,sizeV[0]):
            for j in range (0,sizeV[2]):
                current_ROI = ROIonly[k,:,j].copy()
                image[j,:] = current_ROI
            
            try:
                # d = np.arange(- (image.shape[0]-1),image.shape[1],1)
                diagMat1[:,:,k]=spdiags(image)
                # print(diagMat1[:,:,k])
            except:
                # d = np.arange(- (image.shape[0]-1),image.shape[1],1)
                temp=spdiags(image)
                # print(temp)
                numberDiff=np.abs(temp.shape[1] - diagMat1.shape[1])
                if np.mod(numberDiff,2):
                    temp=np.pad(temp,pad_width=(  (0,0),((numberDiff+1)/2,(numberDiff+1)/2),(0,0)   ),mode='constant',constant_values = 0)
                    diagMat1[:,:,k]=temp[:,:-1].copy()
                else:
                    diagMat1[:,:,k]=np.pad(temp,pad_width=(  (0,0),(numberDiff/2,numberDiff/2),(0,0)   ),mode='constant',constant_values = 0)
                
            
            try:
                # d = np.arange(- (image.shape[0]-1),image.shape[1],1)

                diagMat2[:,:,k]=spdiags(fliplr(image))
                # print(diagMat2[:,:,k])
            except:
                # d = np.arange(- (image.shape[0]-1),image.shape[1],1)
                temp = spdiags(fliplr(image))
                # print(temp)
                numberDiff = np.abs(temp.shape[1]-diagMat2.shape[1])
                if np.mod(numberDiff,2):
                    temp = np.pad(temp,pad_width=(  (0,0),((numberDiff+1)/2,(numberDiff+1)/2),(0,0)   ),mode='constant',constant_values = 0)
                    diagMat2[:,:,k] = temp[:,:-1].copy()
                else:
                    diagMat2[:,:,k] = np.pad(temp,pad_width=(  (0,0),(numberDiff/2,numberDiff/2),(0,0)   ),mode='constant',constant_values = 0)

        for j in range (0,szDiag[1]):
            index = np.where(diagMat1[:,j,0]!=0)[0]
            nTemp = len(index)
            image1 = np.zeros((sizeV[0],nTemp))
            image2 = np.zeros((sizeV[0],nTemp))
            for k in range (0,sizeV[0]):
                image1[k,:nTemp] = np.transpose(diagMat1[index,j,k])
                image2[k,:nTemp] = np.transpose(diagMat2[index,j,k])
            
            
            uniqueIm = np.unique(image1)
            NLtemp = len(uniqueIm)
            indexRow = np.zeros((NLtemp,1))
            temp = image1.copy()
            for k in range(0,NLtemp):
                indexRow[k] = np.where(uniqueIm[k]==uniqueVol)[0] + 1
                image1[temp==uniqueIm[k]] = k + 1
            
                        

            seq = zigzag(image1)
            GLRLMtemp = rle_45(seq,NLtemp)

            nRun = GLRLMtemp.shape[1]

            ind = list(indexRow[0:NLtemp].flatten(order='F')  - 1)
            ind = [int(item) for item in ind]

            GLRLM[ind,0:nRun] = GLRLM[ind,0:nRun] + GLRLMtemp[0:NLtemp,0:nRun]
            GLRLM_D[ind,0:nRun,9] = GLRLM_D[ind,0:nRun,9] + GLRLMtemp[0:NLtemp,0:nRun] 

            seq = zigzag(fliplr(image1))
            GLRLMtemp = rle_45(seq,NLtemp)
            nRun = GLRLMtemp.shape[1]
            GLRLM[ind,0:nRun] = GLRLM[ind,0:nRun] + GLRLMtemp[0:NLtemp,0:nRun]
            GLRLM_D[ind,0:nRun,10] = GLRLM_D[ind,0:nRun,10] + GLRLMtemp[0:NLtemp,0:nRun] 


            uniqueIm = np.unique(image2)
            NLtemp = len(uniqueIm)
            indexRow = np.zeros((NLtemp,1))
            temp = image2.copy()
            for k in range(0,NLtemp):
                indexRow[k] = np.where(uniqueIm[k]==uniqueVol)[0] + 1
                image2[temp==uniqueIm[k]] = k +1 
            
            seq = zigzag(image2)
            GLRLMtemp = rle_45(seq,NLtemp)
            nRun = GLRLMtemp.shape[1]

            ind = list(indexRow[0:NLtemp].flatten(order='F')  - 1)
            ind = [int(item) for item in ind]

            GLRLM[ind,0:nRun] = GLRLM[ind,0:nRun] + GLRLMtemp[0:NLtemp,0:nRun]
            GLRLM_D[ind,0:nRun,11] = GLRLM_D[ind,0:nRun,11] + GLRLMtemp[0:NLtemp,0:nRun] 

            seq = zigzag(fliplr(image2))
            GLRLMtemp = rle_45(seq,NLtemp)
            nRun = GLRLMtemp.shape[1]
            GLRLM[ind,0:nRun] = GLRLM[ind,0:nRun] + GLRLMtemp[0:NLtemp,0:nRun]
            GLRLM_D[ind,0:nRun,12] = GLRLM_D[ind,0:nRun,12] + GLRLMtemp[0:NLtemp,0:nRun] 


    GLRLM = GLRLM[:-1,:]
    sumGLRLM = np.sum(GLRLM,axis=0)
    stop = int(np.where(sumGLRLM > 0)[0][-1])
    GLRLM = GLRLM[:,:stop+1]

    GLRLM_D = GLRLM_D[:-1,:,:]
    GLRLM_D = GLRLM_D[:,:GLRLM.shape[1],:]


    return GLRLM , GLRLM_D



def ShortRunEmph(GLRLM,nG,nR,nS):
    Rj = np.sum(GLRLM , axis=0)
    J = np.arange(0,nR) 
    J = J +1
    tmp = np.divide (Rj , np.float_power(J,2))
    f_SRE = np.sum(tmp)/nS
    return f_SRE

def LongRunEmph(GLRLM,nG,nR,nS):
    Rj =np.sum(GLRLM , axis=0)
    J = np.arange(0,nR)
    J = J +1
    tmp = np.multiply (Rj , np.float_power(J,2))
    f_LRE = np.sum(tmp)/nS
    return f_LRE

def LowGLRunRmph(GLRLM,nG,nR,nS):

    Ri = np.sum(GLRLM , axis=1)
    I = np.transpose (np.arange(0,nG))
    I = I +1
    tmp = np.divide(Ri , np.float_power(I,2))
    f_LGLRE = np.sum(tmp)/nS
    return f_LGLRE

def HighGLRunEmph(GLRLM,nG,nR,nS):
    Ri = np.sum(GLRLM , axis=1)
    I = np.transpose (np.arange(0,nG))
    I = I +1
    tmp = np.multiply(Ri , np.float_power(I,2))
    f_HGLRE = np.sum(tmp)/nS
    return f_HGLRE

def ShortRunLowGLEmph(GLRLM,nG,nR,nS):

    rangeX = np.arange(1, nG+1 , 1)
    rangeY = np.arange(1, nR+1 , 1)

    I , J = np.meshgrid(rangeX,rangeY)
    tmp = np.divide (np.transpose(GLRLM) , (  np.multiply(  np.float_power (I,2) , np.float_power (J,2))      ))
    f_SRLGLE = np.sum(tmp)/nS

    return f_SRLGLE

def ShortRunHighRL(GLRLM,nG,nR,nS):

    rangeX = np.arange(1, nG+1 , 1)
    rangeY = np.arange(1, nR+1 , 1)
    I , J = np.meshgrid(rangeX,rangeY)
    tmp = np.divide(np.multiply (np.float_power(I,2), np.transpose(GLRLM)) , np.float_power(J,2))
    f_SRHGLE = np.sum(tmp)/nS

    return f_SRHGLE

def LongRunLowGLEmph(GLRLM,nG,nR,nS):

    rangeX = np.arange(1, nG+1 , 1)
    rangeY = np.arange(1, nR+1 , 1)
    I , J = np.meshgrid(rangeX,rangeY)
        
    tmp = np.divide(np.multiply (np.float_power(J,2),np.transpose(GLRLM)) , np.float_power(I,2))
    f_LRLGLE = np.sum(tmp)/nS
    return f_LRLGLE

def LongRunHighGLEmph(GLRLM,nG,nR,nS):

    rangeX = np.arange(1, nG+1 , 1)
    rangeY = np.arange(1, nR+1 , 1)
    I , J = np.meshgrid(rangeX,rangeY)
    
    tmp = np.multiply(np.multiply (np.float_power(J,2),np.float_power(I,2)) ,np.transpose(GLRLM) )

    f_LRHGLE = np.sum(tmp)/nS

    return f_LRHGLE

def GLnonUnif(GLRLM,nS):


    Ri = np.sum(GLRLM , axis=1)
    f_GLNU = np.sum( np.float_power(Ri,2) ) / nS

    return f_GLNU


def GLnonUnifNormzd(GLRLM,nS):

    Ri = np.sum(GLRLM , axis=1)
    f_GLNUN = np.sum( np.float_power(Ri,2)) / np.float_power(nS,2)
    return f_GLNUN


def RunLengthNonUnif(GLRLM,nS):


    Ri = np.sum(GLRLM , axis=0)
    f_RLNU = np.sum(np.float_power(Ri,2)) / nS

    return f_RLNU


def RunLengthNonUnifNormzd(GLRLM,nS):

    Ri = np.sum(GLRLM , axis=0)
    f_RLNUN = np.sum( np.float_power(Ri,2)) / np.float_power(nS,2)
    return f_RLNUN


def RunPercentage(nS,nV):


    f_RP = nS / nV
    return f_RP


def GLVar(GLRLM,nG,nR,nS):

    Pij = GLRLM / nS

    rangeX = np.arange(1, nG+1 , 1)
    rangeY = np.arange(1, nR+1 , 1)
    I , J = np.meshgrid(rangeX,rangeY)

    mu = np.sum(np.sum(np.multiply(I, np.transpose(Pij))))
    tmp =  np.multiply (np.float_power((I - mu),2), np.transpose(Pij))
    f_GLV = np.sum(tmp)

    return f_GLV


def RunLengthVar(GLRLM,nG,nR,nS):

    Pij = GLRLM / nS
    rangeX = np.arange(1, nG+1 , 1)
    rangeY = np.arange(1, nR+1 , 1)
    I , J = np.meshgrid(rangeX,rangeY)
    
    mu = np.sum(np.sum(np.multiply(J, np.transpose(Pij))))
    tmp =  np.multiply (np.float_power((J - mu),2), np.transpose(Pij))
    f_RLV = np.sum(tmp)
    return f_RLV


def RunEntropy(GLRLM,nS):

    Pij = GLRLM / nS
    tmp = np.multiply(Pij , np.log2(Pij+ np.finfo(float).tiny))
    f_RE = -np.sum(tmp)

    return f_RE






















def GLRLMfeatHandler(GLRLM,nG,nR,nS,nV):

    
    try:
        nf = GLRLM.shape[1]
    except:
        GLRLM = np.expand_dims(GLRLM,-1)


    ArrayOut = []
    ArrayOut.append(ShortRunEmph(GLRLM,nG,nR,nS))
    ArrayOut.append(LongRunEmph(GLRLM,nG,nR,nS))
    ArrayOut.append(LowGLRunRmph(GLRLM,nG,nR,nS))
    ArrayOut.append(HighGLRunEmph(GLRLM,nG,nR,nS))
    ArrayOut.append(ShortRunLowGLEmph(GLRLM,nG,nR,nS))
    ArrayOut.append(ShortRunHighRL(GLRLM,nG,nR,nS))
    ArrayOut.append(LongRunLowGLEmph(GLRLM,nG,nR,nS))
    ArrayOut.append(LongRunHighGLEmph(GLRLM,nG,nR,nS))
    ArrayOut.append(GLnonUnif(GLRLM,nS))
    ArrayOut.append(GLnonUnifNormzd(GLRLM,nS))
    ArrayOut.append(RunLengthNonUnif(GLRLM,nS))
    ArrayOut.append(RunLengthNonUnifNormzd(GLRLM,nS))
    ArrayOut.append(RunPercentage(nS,nV))
    ArrayOut.append(GLVar(GLRLM,nG,nR,nS))
    ArrayOut.append(RunLengthVar(GLRLM,nG,nR,nS))
    ArrayOut.append(RunEntropy(GLRLM,nS))

    return ArrayOut


def CalcGLRLM(GLRLM,ROI):

    
    nG = GLRLM.shape[0]            
    nR = GLRLM.shape[1]               

    nV = np.where(~np.isnan(ROI))[0].shape[0]
    # nV = np.where(~np.isnan(ROI))[0]

    FeatMatrixout = []

    if GLRLM.ndim != 3:
        maxRange = 1
        tmpGLRLM = np.squeeze(GLRLM)
        nS = np.sum(tmpGLRLM)        
        
        feat = np.transpose(GLRLMfeatHandler(tmpGLRLM,nG,nR,nS,nV))
        FeatMatrixout = feat

    elif GLRLM.ndim == 3:
        maxRange = GLRLM.shape[2]
        for k in range(0,maxRange):
            tmpGLRLM = np.squeeze(GLRLM[:,:,k])
            nS = np.sum(tmpGLRLM)   
            feat = np.transpose(GLRLMfeatHandler(tmpGLRLM,nG,nR,nS,nV))  

            if k == 0:
                FeatMatrixout = feat
            else:    
                FeatMatrixout = np.column_stack((FeatMatrixout , feat))
            
    return FeatMatrixout
