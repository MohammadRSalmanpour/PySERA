
from scipy.io import savemat, loadmat
import os
import numpy as np
import sys
import skimage
import cv2
import itertools
import scipy 


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


def getMI(ROIbox):

    ROIbox = ROIbox.copy()
    mask = ROIbox.copy()
    mask[~np.isnan(mask)] = 1

    
    xdim, ydim, zdim = ROIbox.shape


    rangeX = np.arange(1, xdim+1 , 1)
    rangeY = np.arange(1, ydim+1 , 1)
    rangeZ = np.arange(1, zdim+1 , 1)

    X,Y,Z= np.meshgrid(rangeX,rangeY,rangeZ,indexing ='ij')

    min_image=np.nanmin(ROIbox)
    mean_image=np.nanmean(ROIbox)

    # hold_image=  np.multiply ((ROIbox-min_image)     ,      (mask  /  (mean_image-min_image))    )
    hold_image=  np.multiply ((ROIbox-min_image)     ,  mask)  /  (mean_image-min_image)


    hold_image[np.isnan(hold_image)] =0


    m000=np.sum(hold_image)
    om000=np.nansum(mask)



        
    u000=m000
    ou000=om000


    m100=np.sum( np.multiply (X,hold_image))
    m010=np.sum( np.multiply (Y,hold_image))
    m001=np.sum( np.multiply (Z,hold_image))

    mask[np.isnan(mask)] = 0

    om100=np.sum( np.multiply (X,mask))
    om010=np.sum( np.multiply (Y,mask))
    om001=np.sum( np.multiply (Z,mask))


    x_mean= np.divide(m100,m000)
    y_mean= np.divide(m010,m000) 
    z_mean= np.divide(m001,m000)  

    ox_mean= np.divide(om100,om000)
    oy_mean= np.divide(om010,om000) 
    oz_mean= np.divide(om001,om000)  



    u200=np.sum(np.multiply(np.float_power((X-x_mean),2),hold_image))
    u020=np.sum(np.multiply(np.float_power((Y-y_mean),2),hold_image))
    u002=np.sum(np.multiply(np.float_power((Z-z_mean),2),hold_image))

    u110=np.sum(np.multiply ( np.multiply  ((X-x_mean),(Y-y_mean))    ,hold_image))
    u101=np.sum(np.multiply (  np.multiply ((X-x_mean),(Z-z_mean))    ,hold_image))
    u011=np.sum(np.multiply  (np.multiply  ((Y-y_mean),(Z-z_mean))    ,hold_image))

    u300=np.sum(np.multiply(np.float_power((X-x_mean),3),hold_image))
    u030=np.sum(np.multiply(np.float_power((Y-y_mean),3),hold_image))
    u003=np.sum(np.multiply(np.float_power((Z-z_mean),3),hold_image))




    u210=np.sum( np.multiply( np.multiply( np.float_power    ((X-x_mean),2)   ,(Y-y_mean))   ,hold_image))
    u201=np.sum( np.multiply( np.multiply(  np.float_power    (   (X-x_mean),2)   ,(Z-z_mean))   ,hold_image))

    u120=np.sum(np.multiply(np.multiply( (X-x_mean),    np.float_power    ((Y-y_mean),2) )  ,hold_image))
    u102=np.sum(np.multiply(np.multiply( (X-x_mean),    np.float_power    ((Z-z_mean),2) )  ,hold_image))

    u021=np.sum(  np.multiply(np.multiply( np.float_power    ((Y-y_mean),2)   ,(Z-z_mean)),hold_image))
    u012=np.sum(   np.multiply(np.multiply( (Y-y_mean), np.float_power    ((Z-z_mean),2) ), hold_image))
    u111=np.sum( np.multiply( np.multiply(np.multiply( (X-x_mean),(Y-y_mean)),(Z-z_mean)),hold_image))




    ou200=np.sum( np.multiply(np.float_power    ( (X-ox_mean),2),mask))
    ou020=np.sum( np.multiply(np.float_power    ( (Y-oy_mean),2),mask))
    ou002=np.sum(np.multiply( np.float_power    ( (Z-oz_mean),2),mask))

    ou110=np.sum(np.multiply(np.multiply((X-ox_mean),(Y-oy_mean)),mask))
    ou101=np.sum(np.multiply(np.multiply((X-ox_mean),(Z-oz_mean)),mask))
    ou011=np.sum(np.multiply(np.multiply((Y-oy_mean),(Z-oz_mean)),mask))

    ou300=np.sum(np.multiply( np.float_power    ((X-ox_mean),3),mask))
    ou030=np.sum(np.multiply( np.float_power    ((Y-oy_mean),3),mask))
    ou003=np.sum(np.multiply( np.float_power    ((Z-oz_mean),3),mask))

    ou210=np.sum(np.multiply(np.multiply(np.float_power    ((X-ox_mean),2),(Y-oy_mean)),mask))
    ou201=np.sum(np.multiply(np.multiply(np.float_power    ((X-ox_mean),2),(Z-oz_mean)),mask))
    ou120=np.sum(np.multiply(np.multiply((X-ox_mean),  np.float_power    ( (Y-oy_mean),2)),mask))
    ou102=np.sum(np.multiply(np.multiply((X-ox_mean), np.float_power    ((Z-oz_mean),2)),mask))
    ou021=np.sum(np.multiply(np.multiply(np.float_power    ((Y-oy_mean),2),(Z-oz_mean)),mask))
    ou012=np.sum(np.multiply(np.multiply((Y-oy_mean),np.float_power    ((Z-oz_mean),2)),mask))
    ou111=np.sum(np.multiply(np.multiply(np.multiply((X-ox_mean),(Y-oy_mean)),(Z-oz_mean)),mask))


        
    n200=  np.divide (u200,np.float_power    (u000,(2/3+1)))
    n020=np.divide (u020,np.float_power    (u000,(2/3+1)))
    n002=np.divide (u002,np.float_power    (u000,(2/3+1)))

    n110=np.divide (u110,np.float_power    (u000,(2/3+1)))
    n101=np.divide (u101,np.float_power    (u000,(2/3+1)))
    n011=np.divide (u011,np.float_power    (u000,(2/3+1)))

    n300=np.divide (u300,np.float_power    (u000,(2)))
    n030=np.divide (u030,np.float_power    (u000,(2)))
    n003=np.divide (u003,np.float_power    (u000,(2)))

    n210=np.divide (u210,np.float_power    (u000,(2)))
    n201=np.divide (u201,np.float_power    (u000,(2)))
    n120=np.divide (u120,np.float_power    (u000,(2)))
    n102=np.divide (u102,np.float_power    (u000,(2)))
    n021=np.divide (u021,np.float_power    (u000,(2)))
    n012=np.divide (u012,np.float_power    (u000,(2)))
    n111=np.divide (u111,np.float_power    (u000,(2)))

    J1=n200+n020+n002
    Q=  np.float_power    (n200,2) + np.float_power    (n020,2)+np.float_power    (n002,2)+  2*(  np.float_power    (n101,2)  + np.float_power    (n110,2)+ np.float_power    (n011,2))

    J2=np.multiply(n200,n020)+np.multiply(n200,n002)+np.multiply(n020,n002)-np.float_power    (n101,2)-np.float_power    (n110,2)-np.float_power    (n011,2)
    J3=np.multiply(np.multiply(n200,n020),n002)-np.multiply(n002,np.float_power    (n110,2))+2*np.multiply(np.multiply(n110,n101),n011)-np.multiply(n020,np.float_power    (n101,2))-np.multiply(n200,np.float_power    (n011,2))
    B3=np.float_power    (n300,2)+np.float_power    (n030,2)+np.float_power    (n003,2)+3*np.float_power    (n210,2)+3*np.float_power    (n201,2)+3*np.float_power    (n120,2)+6*np.float_power    (n111,2)+3*np.float_power    (n102,2)+3*np.float_power    (n021,2)+3*np.float_power    (n012,2)

    on200=np.divide (ou200, np.float_power    ( ou000,(2/3+1)))
    on020=np.divide (ou020,np.float_power    (ou000,(2/3+1)))
    on002=np.divide (ou002,np.float_power    (ou000,(2/3+1)))

    on110=np.divide (ou110,np.float_power    (ou000,(2/3+1)))
    on101=np.divide (ou101,np.float_power    (ou000,(2/3+1)))
    on011=np.divide (ou011,np.float_power    (ou000,(2/3+1)))

    on300=np.divide (ou300,np.float_power    (ou000,(2)))
    on030=np.divide (ou030,np.float_power    (ou000,(2)))
    on003=np.divide (ou003,np.float_power    (ou000,(2)))

    on210=np.divide (ou210,np.float_power    (ou000,(2)))
    on201=np.divide (ou201,np.float_power    (ou000,(2)))
    on120=np.divide (ou120,np.float_power    (ou000,(2)))
    on102=np.divide (ou102,np.float_power    (ou000,(2)))
    on021=np.divide (ou021,np.float_power    (ou000,(2)))
    on012=np.divide (ou012,np.float_power    (ou000,(2)))
    on111=np.divide (ou111,np.float_power    (ou000,(2)))

    oJ1=on200+on020+on002
    oQ=np.float_power    (on200,2)+np.float_power    (on020,2)+np.float_power    (on002,2)+2*(np.float_power    (on101,2)+np.float_power    (on110,2)+np.float_power    (on011,2))
    oJ2=np.multiply(on200,on020)+np.multiply(on200,on002)+np.multiply(on020,on002)-np.float_power    (on101,2)-np.float_power    (on110,2)-np.float_power    (on011,2)
    oJ3=np.multiply(np.multiply(on200,on020),on002)-np.multiply(on002,np.float_power    (on110,2))+2*np.multiply(np.multiply(on110,on101),on011)-np.multiply(on020,np.float_power    (on101,2))-np.multiply(on200,np.float_power    (on011,2))
    oB3=np.float_power    (on300,2)+np.float_power    (on030,2)+np.float_power    (on003,2)+3*np.float_power    (on210,2)+3*np.float_power    (on201,2)+3*np.float_power    (on120,2)+6*np.float_power    (on111,2)+3*np.float_power    (on102,2)+3*np.float_power    (on021,2)+3*np.float_power    (on012,2)

    metrics=[J1, Q, J2, J3, B3, oJ1, oQ, oJ2, oJ3, oB3]

    return metrics
