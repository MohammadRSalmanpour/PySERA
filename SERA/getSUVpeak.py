
from cmath import isnan
from scipy.io import savemat, loadmat
import os
import numpy as np
import sys
import skimage
import cv2
import itertools
import scipy 




def getSUVpeak(RawImg2,ROI2,pixelW,sliceTh):

    RawImg = RawImg2.copy()
    ROI = ROI2.copy()

    R = np.divide(    np.float_power((3/(4*np.pi)),(1/3)) * 10 , [pixelW,sliceTh]     )
    SPH=np.zeros(( int(2*np.floor(R[0])+1) , int(2*np.floor(R[0])+1),  int(2*np.floor(R[1])+1)  ))

    rangeX = np.arange(pixelW  * ((-np.ceil(SPH.shape[0]/2)+0.5)- 0.5) , pixelW    * ((  np.floor(SPH.shape[0]/2))- 0.5) , pixelW)
    rangeY = np.arange(pixelW  * ((-np.ceil(SPH.shape[1]/2)+0.5)- 0.5) , pixelW   * ((   np.floor(SPH.shape[1]/2))- 0.5) , pixelW)
    rangeS = np.arange(sliceTh  * ((-np.ceil(SPH.shape[2]/2)+0.5)- 0.5) , sliceTh  * ((  np.floor(SPH.shape[2]/2))- 0.5) , sliceTh)  
    
    x,y,z = np.meshgrid(rangeY,rangeX,rangeS)

    # print(x[:,:,0])
    # print(y[:,:,0])
    # print(z[:,:,0])

    tmpsph = np.sqrt(   np.float_power( (x-x[0,int(np.ceil(x.shape[0]/2))-1,0]),2  )   +    np.float_power( (y-y[int(np.ceil(y.shape[1]/2))-1,0,0]),2 )   +   np.float_power( (z-z[0,0,int(np.ceil(z.shape[2]/2))-1]),2 )   )
    tmpsph[tmpsph > (  np.float_power ((3/(4*np.pi)) , (1/3))   *10  )] = np.nan 
    
    # zxcsac = tmpsph[:,:,1]
    # print(zxcsac)

    # print(np.mean(tmpsph))
    SPH = tmpsph.copy()
    SPH[~ np.isnan(tmpsph)] = 1 

    R = np.floor(R)
    
    pad_wid=((int(R[0]), int(R[0])), (int(R[0]), int(R[0])), (int(R[1]), int(R[1])))

    ImgRawROIpadded = np.pad(RawImg, pad_width=pad_wid,mode='constant',constant_values=np.nan) 
    ImgRawROIpadded = np.nan_to_num(ImgRawROIpadded,nan=0)
    SPH = np.nan_to_num(SPH,nan=0)

    sph2 = np.divide( SPH  ,   np.nansum(SPH) )

    # print(np.mean(ImgRawROIpadded))
    # print(np.mean(sph2))

    from scipy.signal import convolve , fftconvolve,oaconvolve
    C = convolve(ImgRawROIpadded, sph2, mode='valid', method='auto')
    # C = fftconvolve(ImgRawROIpadded, sph2, mode='valid')
    # C = oaconvolve(ImgRawROIpadded, sph2, mode='valid')
    # print(np.mean(C))
    
    # from scipy.ndimage import convolve
    # C =  convolve(ImgRawROIpadded, sph2, mode='constant', cval=np.nan)
    # print(np.mean(C))
    # print(np.nanmean(C))


    T1_RawImg=RawImg.flatten(order='F')
    T1_ROI=ROI.flatten(order='F')
    T1_C= C.flatten(order='F')
    # T1 = np.column_stack((T1_RawImg,T1_ROI,T1_C))

    # print(np.nanmean(T1_RawImg))
    # print(np.max(T1_RawImg))
    # print(np.min(T1_RawImg))
    # print(T1_RawImg.shape)
    # print('@@@@@@@@@@@@@@@@')

    # print(np.mean(T1_ROI))
    # print(np.max(T1_ROI))
    # print(np.min(T1_ROI))
    # print(T1_ROI.shape)
    # print('@@@@@@@@@@@@@@@@')
    
    # print(np.mean(T1_C))
    # print(np.max(T1_C))
    # print(np.min(T1_C))
    # print(T1_C.shape)
    # print('@@@@@@@@@@@@@@@@')

    # np.nan_to_num(T1_RawImg,nan=0)
    # np.nan_to_num(T1_ROI,nan=0)
    # np.nan_to_num(T1_C,nan=0)

    T1_RawImg2 = T1_RawImg.copy()

    T1_RawImg1 = T1_RawImg2[~np.isnan(T1_RawImg)]
    T1_ROI1 = T1_ROI[~np.isnan(T1_RawImg)]
    T1_C1 = T1_C[~np.isnan(T1_RawImg)]

    T2_RawImg = T1_RawImg1[T1_ROI1 != 0]
    T2_C = T1_C1[T1_ROI1 != 0]
    T2_ROI = T1_ROI1[T1_ROI1 != 0]

    maxind=np.argmax(T2_RawImg)
    SUVpeak = [np.max(T2_C[maxind])]   
    SUVpeak.append(np.max(T2_C)) 

    return SUVpeak