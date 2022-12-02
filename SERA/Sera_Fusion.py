# import requests
import concurrent.futures
import pywt
import cv2
# from math import *
from random import randint
# from random import *
import time
from PIL import Image
from sklearn.decomposition import PCA
import nrrd
import SimpleITK as sitk
# import pydicom as dicom
import os
import numpy as np
import nibabel as nib
import shutil
from scipy import ndimage
from scipy.io import savemat, loadmat
import pandas as pd
import subprocess
# import cherrypy
# from dicom2nifti.convert_dicom import dicom_series_to_nifti
# from difflib import SequenceMatcher
import threading
import sys
from datetime import datetime
import traceback
import pydicom
import socketserver
import datetime
from PythonCode.Sera_ReadWrite import readimage,convert_modalities,similar
import psutil


##################################  Fusion functon   ##################################
def weighted_average_fusion(img1, img2, w1, w2 , InterPolation):
    '''
    This is the simplest image fusion algorithm. 
    :param img1: The first origin image.
    :param img2: The second origin image.
    :param w1: The weight of first image.
    :param w2: The weight of second image.
    :return: The fusioned image.
    '''
    if w1<0 or w2<0:
        print('You are not allowed to use weight value lesser than zero.')
        return
    elif w1 + w2 != 1:
        w1 = w1/(w1+w2)
        w2 = w2/(w1+w2)
    shape = np.shape(img1)
    img = np.zeros(shape,dtype = np.int8)
    if np.shape(img2) != shape:
        img2 = cv2.resize(img2, np.shape(img1), interpolation = InterPolation)
    img = w1*img1+w2*img2
    return img

# main function
def weighted_Fusion(ct_image , pet_image , w1, w2, InterPolation):
    # perm = (2,1,0)
    # ct_image = np.transpose(ct_image ,perm )
    # pet_image = np.transpose(pet_image ,perm )

    ct_min = np.min(ct_image)
    ct_max = np.max(ct_image)

    pet_min = np.min(pet_image)
    pet_max = np.max(pet_image)

    ct_image = (ct_image - ct_min ) / ( ct_max - ct_min)
    pet_image = (pet_image - pet_min) / (pet_max - pet_min)

    ct_image = ct_image * 255
    pet_image = pet_image * 255

    for i in range(0 , ct_image.shape[2]):

        
        img1 = ct_image[:,:,i]
        img2 = pet_image[:,:,i]

        img1_rgb = Image.fromarray(np.uint8(img1)).convert('RGB')
        img2_rgb = Image.fromarray(np.uint8(img2)).convert('RGB')

        img1_a = np.array(img1_rgb)
        img2_a = np.array(img2_rgb)

        im = weighted_average_fusion(img1_a, img2_a, w1, w2,InterPolation)


        gray = rgb2gray(im) 
        gray = gray / 255

        if i == 0:
            Final_fusion3D = gray
        else:
            Final_fusion3D = np.dstack((Final_fusion3D ,gray))

    # perm = (2,1,0)
    # Final_fusion3D = np.transpose(Final_fusion3D ,perm )
    
    
    # Currentmin = np.min(Final_fusion3D)
    # Currentmax = np.max(Final_fusion3D)


    # Final_fusion3D = ((ct_max - ct_min )*((Final_fusion3D - Currentmin ) / (Currentmax - Currentmin))) + ct_min
    # Final_fusion3D = Final_fusion3D.astype(np.uint16())
    
    # print(np.min(Final_fusion3D))
    # print(np.max(Final_fusion3D))
    return Final_fusion3D


def HSI_Fusion(ct_image , pet_image):
    ct_min = np.min(ct_image)
    ct_max = np.max(ct_image)

    pet_min = np.min(pet_image)
    pet_max = np.max(pet_image)

    ct_image = (ct_image - ct_min ) / ( ct_max - ct_min)
    pet_image = (pet_image - pet_min) / (pet_max - pet_min)

    ct_image = ct_image * 255
    pet_image = pet_image * 255

    for i in range(0 , ct_image.shape[2]):

        img1 = ct_image[:,:,i]
        img2 = pet_image[:,:,i]

        img1_rgb = Image.fromarray(np.uint8(img1)).convert('RGB')
        img2_rgb = Image.fromarray(np.uint8(img2)).convert('RGB')

        img1_a = np.array(img1_rgb)
        img2_a = np.array(img2_rgb)

        im = HSI_image_fusion(img1_a, img2_a)

        gray = rgb2gray(im) 
        gray = gray / 255

        if i == 0:
            Final_fusion3D = gray
        else:
            Final_fusion3D = np.dstack((Final_fusion3D ,gray))
    return Final_fusion3D


def HSI_image_fusion(img1, img2):
    hsi_img1 = RGB2HSI(img1)
    hsi_img2 = RGB2HSI(img2)
    hsi_img1[:,:,2] = hsi_img2[:,:,2]
    img = HSI2RGB(hsi_img1)
    return img


def RGB2HSI( rgb_img):
    """
    RGB to HSI
    :param rgm_img: RGB image
    :return: HSI image
    """
    row = np.shape(rgb_img)[0]
    col = np.shape(rgb_img)[1]
    hsi_img = rgb_img.copy()
    B,G,R = cv2.split(rgb_img)
    [B,G,R] = [ i/ 255.0 for i in ([B,G,R])]
    H = np.zeros((row, col))    
    I = (R + G + B) / 3.0       
    S = np.zeros((row,col))      
    for i in range(row):
        den = np.sqrt((R[i]-G[i])**2+(R[i]-B[i])*(G[i]-B[i]))
        thetha = np.arccos(0.5*(R[i]-B[i]+R[i]-G[i])/den)   
        h = np.zeros(col)               
        h[B[i]<=G[i]] = thetha[B[i]<=G[i]]
        h[G[i]<B[i]] = 2*np.pi-thetha[G[i]<B[i]]
        h[den == 0] = 0
        H[i] = h/(2*np.pi)      
    for i in range(row):
        min = []
        for j in range(col):
            arr = [B[i][j],G[i][j],R[i][j]]
            min.append(np.min(arr))
        min = np.array(min)
        S[i] = 1 - min*3/(R[i]+B[i]+G[i])
        S[i][R[i]+B[i]+G[i] == 0] = 0
    hsi_img[:,:,0] = H*255
    hsi_img[:,:,1] = S*255
    hsi_img[:,:,2] = I*255
    return hsi_img


def HSI2RGB(hsi_img):
    """
    HSI image transform to RGB image
    :param hsi_img: HSI image
    :return: return RGB image
    """
    # save the shape of original image
    row = np.shape(hsi_img)[0]
    # col = np.shape(hsi_img)[1]
    #copy the origin image
    rgb_img = hsi_img.copy()
    #split the channel
    H,S,I = cv2.split(hsi_img)
    #project the channel into [0,1]
    [H,S,I] = [ i/ 255.0 for i in ([H,S,I])]
    R,G,B = H,S,I
    for i in range(row):
        h = H[i]*2*np.pi
        #H is bigger than 0 but smaller than 120
        a1 = h >=0
        a2 = h < 2*np.pi/3
        a = a1 & a2         #index in first situation
        tmp = np.cos(np.pi / 3 - h)
        b = I[i] * (1 - S[i])
        r = I[i]*(1+S[i]*np.cos(h)/tmp)
        g = 3*I[i]-r-b
        B[i][a] = b[a]
        R[i][a] = r[a]
        G[i][a] = g[a]
        #H is bigger than 120 but smaller than 240
        a1 = h >= 2*np.pi/3
        a2 = h < 4*np.pi/3
        a = a1 & a2         #index in second situation
        tmp = np.cos(np.pi - h)
        r = I[i] * (1 - S[i])
        g = I[i]*(1+S[i]*np.cos(h-2*np.pi/3)/tmp)
        b = 3 * I[i] - r - g
        R[i][a] = r[a]
        G[i][a] = g[a]
        B[i][a] = b[a]
        #H is bigger than 180 but smaller than 360
        a1 = h >= 4 * np.pi / 3
        a2 = h < 2 * np.pi
        a = a1 & a2             
        tmp = np.cos(5 * np.pi / 3 - h)
        g = I[i] * (1-S[i])
        b = I[i]*(1+S[i]*np.cos(h-4*np.pi/3)/tmp)
        r = 3 * I[i] - g - b
        B[i][a] = b[a]
        G[i][a] = g[a]
        R[i][a] = r[a]
    rgb_img[:,:,0] = B*255
    rgb_img[:,:,1] = G*255
    rgb_img[:,:,2] = R*255
    return rgb_img

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# This function does the coefficient fusing according to the fusion method
def fuseCoeff(cooef1, cooef2, method):

    if (method == 'mean'):
        cooef = (cooef1 + cooef2) / 2
    elif (method == 'min'):
        cooef = np.minimum(cooef1,cooef2)
    elif (method == 'max'):
        cooef = np.maximum(cooef1,cooef2)
    else:
        cooef = []

    return cooef

# main function
def Wavelet_Fusion(ct_image , pet_image , FUSION_METHOD, wavelet , MODE, level):
    # perm = (2,1,0)
    # ct_image = np.transpose(ct_image ,perm )
    # pet_image = np.transpose(pet_image ,perm )

    ct_min = np.min(ct_image)
    ct_max = np.max(ct_image)

    pet_min = np.min(pet_image)
    pet_max = np.max(pet_image)

    ct_image = (ct_image - ct_min ) / ( ct_max - ct_min)
    pet_image = (pet_image - pet_min) / (pet_max - pet_min)

    ct_image = ct_image * 255
    pet_image = pet_image * 255

    for jj in range(0 , ct_image.shape[2]):
        I1 = ct_image[:,:,jj]
        I2 = pet_image[:,:,jj]


        # First: Do wavelet transform on each image
        cooef1 = pywt.wavedec2(I1[:,:], wavelet , MODE, level )
        cooef2 = pywt.wavedec2(I2[:,:], wavelet , MODE, level )

        # Second: for each level in both image do the fusion according to the desire option
        fusedCooef = []
        for i in range(len(cooef1)-1):
            # The first values in each decomposition is the apprximation values of the top level
            if(i == 0):
                fusedCooef.append(fuseCoeff(cooef1[0],cooef2[0],FUSION_METHOD))
            else:
                # For the rest of the levels we have tupels with 3 coeeficents
                c1 = fuseCoeff(cooef1[i][0],cooef2[i][0],FUSION_METHOD)
                c2 = fuseCoeff(cooef1[i][1], cooef2[i][1], FUSION_METHOD)
                c3 = fuseCoeff(cooef1[i][2], cooef2[i][2], FUSION_METHOD)
                fusedCooef.append((c1,c2,c3))

        # Third: After we fused the cooefficent we nned to transfor back to get the image
        fusedImage = pywt.waverec2(fusedCooef, wavelet,MODE)

        # Forth: normmalize values to be in uint8
        fusedImage = np.multiply(np.divide(fusedImage - np.min(fusedImage),(np.max(fusedImage) - np.min(fusedImage))),255)
        fusedImage = fusedImage.astype(np.uint8)

        # cv2_imshow(fusedImage)

        shape = (I1.shape[1], I1.shape[0])
        fusedImage = cv2.resize(fusedImage,(shape[0],shape[1])) 

        if jj == 0:
            Final_fusion3D = fusedImage
        else:
            Final_fusion3D = np.dstack((Final_fusion3D ,fusedImage))


    # perm = (2,1,0)
    # Final_fusion3D = np.transpose(Final_fusion3D ,perm )


    # Currentmin = np.min(Final_fusion3D)
    # Currentmax = np.max(Final_fusion3D)


    # Final_fusion3D = ((ct_max - ct_min )*((Final_fusion3D - Currentmin ) / (Currentmax - Currentmin))) + ct_min
    # Final_fusion3D = Final_fusion3D.astype(np.uint16())
    
    # print(np.min(Final_fusion3D))
    # print(np.max(Final_fusion3D))



    return Final_fusion3D



def PCA_image_fusion(img1, img2 , nc , ss):
    '''
    This is the algorithm of image fusion based on PCA.
    :param img1: The origin image.
    :param img2: The high resolution image.
    :return: The fusioned image.
    '''


    estimator = PCA(n_components=nc ,svd_solver = ss)
    estimator.fit(img1.copy())
    estimator.fit(img2.copy())
    img_f1 = estimator.transform(img1.copy())
    img_f2 = estimator.transform(img2.copy())
    # img_f1[:,:40] = img_f2[:,:40]
    img_f1[:,:] = img_f2[:,:]
    img = estimator.inverse_transform(img_f1)
    return img

# main function
def PCA_Fusion(ct_image , pet_image, nc , ss):
    # perm = (2,1,0)
    # ct_image = np.transpose(ct_image ,perm )
    # pet_image = np.transpose(pet_image ,perm )

    ct_min = np.min(ct_image)
    ct_max = np.max(ct_image)

    pet_min = np.min(pet_image)
    pet_max = np.max(pet_image)

    ct_image = (ct_image - ct_min ) / ( ct_max - ct_min)
    pet_image = (pet_image - pet_min) / (pet_max - pet_min)

    ct_image = ct_image * 255
    pet_image = pet_image * 255

    for i in range(0 , ct_image.shape[2]):

        
        img1 = ct_image[:,:,i]
        img2 = pet_image[:,:,i]

 
        im = PCA_image_fusion(img1, img2, nc , ss)


        gray = (im - np.min(im)) / (np.max(im) - np.min(im))

   
        shape = (img1.shape[1], img1.shape[0])
        gray = cv2.resize(gray,(shape[0],shape[1])) 

        if i == 0:
            Final_fusion3D = gray
        else:
            Final_fusion3D = np.dstack((Final_fusion3D ,gray))

    # perm = (2,1,0)
    # Final_fusion3D = np.transpose(Final_fusion3D ,perm )
    
    
    
    # Currentmin = np.min(Final_fusion3D)
    # Currentmax = np.max(Final_fusion3D)


    # Final_fusion3D = ((ct_max - ct_min )*((Final_fusion3D - Currentmin ) / (Currentmax - Currentmin))) + ct_min
    # Final_fusion3D = Final_fusion3D.astype(np.uint16())
    
    # print(np.min(Final_fusion3D))
    # print(np.max(Final_fusion3D))
    
    
    
    return Final_fusion3D



def pca_fusion_folder( ct_path , pet_path , ss, nc , destfolder):

    Fixed_datas = os.listdir(ct_path)
    fixed = [i for i in Fixed_datas if (i.endswith(".nrrd") | i.endswith(".dcm") | i.endswith(".dicom") | i.endswith(".nii") | i.endswith(".nii.gz"))]

    Moved_datas = os.listdir(pet_path)
    moved = [i for i in Moved_datas if (i.endswith(".nrrd") | i.endswith(".dcm") | i.endswith(".dicom") | i.endswith(".nii") | i.endswith(".nii.gz"))]

    contin = True
    if nc.isdigit():
       nc = int(nc)
    elif nc.replace('.','',1).isdigit() and nc.count('.') < 2:
       nc = float(nc)
       if ss.lower() == 'arpack' or ss.lower() == 'randomized':
           contin = False
    elif nc == 'Mle':
       nc = 'mle'
       # if ss.lower() == 'arpack' or ss.lower() == 'randomized':
       #     contin = False
       contin = False
    elif nc == 'None':
       nc = None
    if contin:
        ss = ss.lower()
        
        # thread_list = []
        # Num_worker = os.cpu_count() + int(psutil.virtual_memory()[1]/pow(10,9)/2)
        # Num_worker = 1
        Num_worker = int(psutil.virtual_memory()[1]/pow(10,9)/2)
        if Num_worker == 0:
            Num_worker = 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=Num_worker) as executor:
            futures = []
            for co in fixed:
                if len(moved) > 0:
            
                    fixed_filename =  co.replace('.nii.gz', '').replace('.nii', '').replace('.nrrd', '').replace('.dcm', '').replace('.dicom','')        
                    
                    s_ratio = 0
                    count = 0
                    selected_index = 0
                    for co_moved in moved:
                        moved_filename =  co_moved.replace('.nii.gz', '').replace('.nii', '').replace('.nrrd', '').replace('.dcm', '').replace('.dicom','')        
                        sim_ratio = similar(fixed_filename, moved_filename)
                        
                        if s_ratio < sim_ratio:
                            selected_index = count
                            s_ratio = sim_ratio
                        count = count +1
                    
                    if s_ratio > 0 :
            
                        Fixed_fullpath = os.path.join(ct_path,co)
                        fixed_img =  readimage(Fixed_fullpath)
                        
                        
                        Moved_fullpath = os.path.join(pet_path,moved[selected_index])
                        moved_img =  readimage(Moved_fullpath)
                        
                        # print(co ,'matched with', moved[selected_index],'.')
                        try:
                            if isinstance(moved_img[0], np.ndarray) & isinstance(fixed_img[0], np.ndarray):
                    
                            # if moved_img[0] != None & fixed_img[0] != None:  
                                ct_image = fixed_img[0] 
                                pet_image = moved_img[0]
                                
                                if len(ct_image.shape) == 3 & len(pet_image.shape) == 3 :   
                        
                                    if fixed_img[2] == 'Nifti' or fixed_img[2] == 'Dicom':
                                        perm = (2, 1, 0)
                                        ct_image = np.transpose(ct_image, perm)
                                    if moved_img[2] == 'Nifti' or moved_img[2] == 'Dicom':
                                        perm = (2, 1, 0)
                                        pet_image = np.transpose(pet_image, perm)  
                                        
                                        
                                    dim01 = np.shape(ct_image)
                                    dim02 = np.shape(pet_image)
                                    if (dim01[0] == dim02[0]):   
    
    
                                        futures.append(executor.submit(PCA_fusion_folder_Thread,fixed_img,moved_img,ct_image , pet_image ,  nc , ss ,destfolder))   
                                        
                                        if(len(moved)>0):
                                            moved.pop(selected_index)
                                else:
                                    print('Images must be 3D.')
                            else:
                                print('You must use an approprate type of input.')	
                        except Exception as e:
                            print('Out of Memory or the parameters of image fusion tool should be selected properly:', e)
                    # else:
                        # print('There is no image with the same name of',co,'in another folder.')
        executor.shutdown(wait=True)
    else:
        print('Change svd solver or number of component in PCA hyperparameter.')
    # for future in concurrent.futures.as_completed(futures):
    #     cc = 0
    # for thread in thread_list:
    #     thread.join() 
            
        # we_fusion_final = PCA_Fusion(ct_image , pet_image, nc , ss)
        # _filename = fixed_img[3]
        
        # registered=we_fusion_final
        # header=fixed_img[1]
        # DatatypeFrom=fixed_img[2]
        # Datatype=fixed_img[2]
        # if (DatatypeFrom == 'Dicom'):
        #     DatatypeFrom = 'SDicom'
        # if (Datatype == 'Dicom'):
        #     Datatype = 'SDicom'
            
        
        # if DatatypeFrom == 'Nifti' or DatatypeFrom == 'SDicom':
        #     perm = (2, 1, 0)
        #     registered = np.transpose(registered, perm) 
        
        # convert_modalities(registered, header, DatatypeFrom, Datatype, OUTPUT_DIR=destfolder, filename=_filename,createfolder='False')
        
    return ""


def PCA_fusion_folder_Thread(fixed_img,moved_img,ct_image , pet_image ,  nc , ss ,destfolder):
    # ct_max_or = np.max(ct_image)
    ct_min_or = np.min(ct_image)
    we_fusion_final = PCA_Fusion(ct_image , pet_image, nc , ss)
    _filename = fixed_img[3]
    
    registered=we_fusion_final
    header=fixed_img[1]
    DatatypeFrom=fixed_img[2]
    Datatype=fixed_img[2]
    if (DatatypeFrom == 'Dicom'):
        DatatypeFrom = 'SDicom'
    if (Datatype == 'Dicom'):
        Datatype = 'SDicom'
        
    
    if DatatypeFrom == 'Nifti' or DatatypeFrom == 'SDicom':
        perm = (2, 1, 0)
        registered = np.transpose(registered, perm) 
    

    if ct_min_or >= 0 :
        if np.issubdtype(registered.dtype, np.floating) == True:
            ct_max = np.max(registered)
            
            if ct_max <= 1:
                registered = registered * 255
        
            registered = registered.astype(int)
            
    convert_modalities(registered, header, DatatypeFrom, Datatype, OUTPUT_DIR=destfolder, filename=_filename,createfolder='False')
   
    return ""



def wavelet_fusion_folder(ct_path , pet_path ,  fusion_method, wavelet , mode, level , destfolder):
    
    Fixed_datas = os.listdir(ct_path)
    fixed = [i for i in Fixed_datas if (i.endswith(".nrrd") | i.endswith(".dcm") | i.endswith(".dicom") | i.endswith(".nii") | i.endswith(".nii.gz"))]

    Moved_datas = os.listdir(pet_path)
    moved = [i for i in Moved_datas if (i.endswith(".nrrd") | i.endswith(".dcm") | i.endswith(".dicom") | i.endswith(".nii") | i.endswith(".nii.gz"))]
    # thread_list = []
    # Num_worker = os.cpu_count() + int(psutil.virtual_memory()[1]/pow(10,9)/2)
    # Num_worker = 1
    Num_worker = int(psutil.virtual_memory()[1]/pow(10,9)/2)
    if Num_worker == 0:
        Num_worker = 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=Num_worker) as executor:
        futures = []
        for co in fixed:
            if len(moved) > 0:
                fixed_filename =  co.replace('.nii.gz', '').replace('.nii', '').replace('.nrrd', '').replace('.dcm', '').replace('.dicom','')        
                
                s_ratio = 0
                count = 0
                selected_index = 0
                for co_moved in moved:
                    moved_filename =  co_moved.replace('.nii.gz', '').replace('.nii', '').replace('.nrrd', '').replace('.dcm', '').replace('.dicom','')        
                    sim_ratio = similar(fixed_filename, moved_filename)
                    
                    if s_ratio < sim_ratio:
                        selected_index = count
                        s_ratio = sim_ratio
                    count = count +1
                
                if s_ratio > 0 :
                    
                    Fixed_fullpath = os.path.join(ct_path,co)
                    fixed_img =  readimage(Fixed_fullpath)
                    
                    
                    Moved_fullpath = os.path.join(pet_path,moved[selected_index])
                    moved_img =  readimage(Moved_fullpath)
                    
                    # print(co ,'matched with', moved[selected_index],'.')
            
                    try:
                        if isinstance(moved_img[0], np.ndarray) & isinstance(fixed_img[0], np.ndarray):
                
                        # if moved_img[0] != None & fixed_img[0] != None:  
                            
                            ct_image = fixed_img[0] 
                            pet_image = moved_img[0]
                            
                            if len(ct_image.shape) == 3 & len(pet_image.shape) == 3 :   
                    
                            
                                if fixed_img[2] == 'Nifti' or fixed_img[2] == 'Dicom':
                                    perm = (2, 1, 0)
                                    ct_image = np.transpose(ct_image, perm)
                                if moved_img[2] == 'Nifti' or moved_img[2] == 'Dicom':
                                    perm = (2, 1, 0)
                                    pet_image = np.transpose(pet_image, perm)  
                                    
                                    
                                dim01 = np.shape(ct_image)
                                dim02 = np.shape(pet_image)
                                if (dim01[0] == dim02[0]):   
                  
                                    FUSION_METHOD = fusion_method.lower()
                                    MODE = mode.lower()
                                    if MODE =='antisymmrtric':
                                        MODE ='antisymmetric'
                                    level = int(level)
                                    # v = wavelet.split(' ')[0]
                                    # wavelet = v.lower()
                                    
                                    cont = True
                                    wavelet_family_list = ['gaus','mexh','morl','shan','fbsp','cmor','cgau']
                                    wavelet_family=wavelet.split(' ')[0]
                                    if wavelet_family.lower() in wavelet_family_list:
                                        cont = False
                                    
                                    if cont:
                                
                                        v = wavelet.replace(' family','')
                                        wavelet = v.lower()
                                        # wavelet = wavelet.lower()
                                        futures.append(executor.submit(wavelet_fusion_folder_Thread, fixed_img,moved_img,ct_image , pet_image , FUSION_METHOD, wavelet , MODE, level,destfolder))
                                        
                                    else:
                                        print('Change wavelet family hyperparameter.') 
                                    if(len(moved)>0):
                                        moved.pop(selected_index)
                            else:
                                print('Images must be 3D.')
                        else:
                            print('You must use an approprate type of input.')
                    except Exception as e:
                        print('Out of Memory or the parameters of image fusion tool should be selected properly:', e)
                # else:
                    # print('There is no image with the same name of',co,'in another folder.')
    executor.shutdown(wait=True)
    # for future in concurrent.futures.as_completed(futures):
    #     cc = 0





        # we_fusion_final = Wavelet_Fusion(ct_image , pet_image , FUSION_METHOD, wavelet , MODE, level)
        # _filename = fixed_img[3]
        
        # registered=we_fusion_final
        # header=fixed_img[1]
        # DatatypeFrom=fixed_img[2]
        # Datatype=fixed_img[2]
        # if (DatatypeFrom == 'Dicom'):
        #     DatatypeFrom = 'SDicom'
        # if (Datatype == 'Dicom'):
        #     Datatype = 'SDicom'
            
        # if DatatypeFrom == 'Nifti' or DatatypeFrom == 'SDicom':
        #     perm = (2, 1, 0)
        #     registered = np.transpose(registered, perm) 
        
        # convert_modalities(registered, header, DatatypeFrom, Datatype, OUTPUT_DIR=destfolder, filename=_filename,createfolder='False')
                
    return ""

def wavelet_fusion_folder_Thread(fixed_img,moved_img,ct_image , pet_image , FUSION_METHOD, wavelet , MODE, level,destfolder):
    # ct_max_or = np.max(ct_image)
    ct_min_or = np.min(ct_image)
    we_fusion_final = Wavelet_Fusion(ct_image , pet_image , FUSION_METHOD, wavelet , MODE, level)
    _filename = fixed_img[3]
    
    registered=we_fusion_final
    header=fixed_img[1]
    DatatypeFrom=fixed_img[2]
    Datatype=fixed_img[2]
    if (DatatypeFrom == 'Dicom'):
        DatatypeFrom = 'SDicom'
    if (Datatype == 'Dicom'):
        Datatype = 'SDicom'
        
    if DatatypeFrom == 'Nifti' or DatatypeFrom == 'SDicom':
        perm = (2, 1, 0)
        registered = np.transpose(registered, perm) 
    
    if ct_min_or >= 0 :
        if np.issubdtype(registered.dtype, np.floating) == True:
            ct_max = np.max(registered)
            
            if ct_max <= 1:
                registered = registered * 255
        
            registered = registered.astype(int)
        
    convert_modalities(registered, header, DatatypeFrom, Datatype, OUTPUT_DIR=destfolder, filename=_filename,createfolder='False')
     
    return ""


def weighted_fusion_folder( ct_path , pet_path , w1, w2, interpolation, destfolder):

    if (interpolation=='INTER_LINEAR'):
        interpolation = cv2.INTER_LINEAR
    elif (interpolation=='INTER_AREA'):
        interpolation = cv2.INTER_AREA
    elif (interpolation=='INTER_NEAREST'):
        interpolation = cv2.INTER_NEAREST
    elif (interpolation=='INTER_CUBIC'):
        interpolation = cv2.INTER_CUBIC
    elif (interpolation=='INTER_LANCZOS4'):
        interpolation = cv2.INTER_LANCZOS4
    elif (interpolation=='INTER_LANCZOS'):
        interpolation = cv2.INTER_LANCZOS
       
    try:
        interpolation = interpolation.upper()
        
        if (interpolation=='LINEAR'):
            interpolation = cv2.INTER_LINEAR
        elif (interpolation=='AREA'):
            interpolation = cv2.INTER_AREA
        elif (interpolation=='NEAREST'):
            interpolation = cv2.INTER_NEAREST
        elif (interpolation=='CUBIC'):
            interpolation = cv2.INTER_CUBIC
        elif (interpolation=='LANCZOS4'):
            interpolation = cv2.INTER_LANCZOS4
        elif (interpolation=='LANCZOS'):
            interpolation = cv2.INTER_LANCZOS
    except:
        # yp=0
        # print('interpolator is incorrect')
        # set default interpolator
        interpolation = cv2.INTER_LINEAR
        

    Fixed_datas = os.listdir(ct_path)
    fixed = [i for i in Fixed_datas if (i.endswith(".nrrd") | i.endswith(".dcm") | i.endswith(".dicom") | i.endswith(".nii") | i.endswith(".nii.gz"))]

    Moved_datas = os.listdir(pet_path)
    moved = [i for i in Moved_datas if (i.endswith(".nrrd") | i.endswith(".dcm") | i.endswith(".dicom") | i.endswith(".nii") | i.endswith(".nii.gz"))]
    # thread_list = []
    # Num_worker = os.cpu_count() + int(psutil.virtual_memory()[1]/pow(10,9)/2)
    # Num_worker = 1
    Num_worker = int(psutil.virtual_memory()[1]/pow(10,9)/2)
    if Num_worker == 0:
        Num_worker = 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=Num_worker) as executor:
        futures = []
        for co in fixed:
            
            if len(moved) > 0:
                fixed_filename =  co.replace('.nii.gz', '').replace('.nii', '').replace('.nrrd', '').replace('.dcm', '').replace('.dicom','')        
                
                
                s_ratio = 0
                count = 0
                selected_index = 0
                for co_moved in moved:
                    moved_filename =  co_moved.replace('.nii.gz', '').replace('.nii', '').replace('.nrrd', '').replace('.dcm', '').replace('.dicom','')        
                    sim_ratio = similar(fixed_filename, moved_filename)
                    
                    if s_ratio < sim_ratio:
                        selected_index = count
                        s_ratio = sim_ratio
                    count = count +1
                
                if s_ratio > 0 :
        
                    Fixed_fullpath = os.path.join(ct_path,co)
                    fixed_img =  readimage(Fixed_fullpath)
                    
                    
                    Moved_fullpath = os.path.join(pet_path,moved[selected_index])
                    moved_img =  readimage(Moved_fullpath)
                    
                    # print(co ,'matched with', moved[selected_index],'.')
            
                    try:
                        if isinstance(moved_img[0], np.ndarray) & isinstance(fixed_img[0], np.ndarray):
                
                        # if moved_img[0] != None & fixed_img[0] != None:  
                
                                
                            ct_image = fixed_img[0] 
                            pet_image = moved_img[0]
                            # print(ct_image.shape)
                            # print(pet_image.shape)
                    
                            if len(ct_image.shape) == 3 & len(pet_image.shape) == 3 :   
                    
                                if fixed_img[2] == 'Nifti' or fixed_img[2] == 'Dicom':
                                    perm = (2, 1, 0)
                                    ct_image = np.transpose(ct_image, perm)
                                if moved_img[2] == 'Nifti' or moved_img[2] == 'Dicom':
                                    perm = (2, 1, 0)
                                    pet_image = np.transpose(pet_image, perm)  
                                
                                    
                                dim01 = np.shape(ct_image)
                                dim02 = np.shape(pet_image)
                                if (dim01[0] == dim02[0]):   
                                    futures.append(executor.submit(weighted_fusion_folder_Thread, fixed_img,moved_img,ct_image , pet_image ,w1,w2, interpolation,destfolder))
         
                                    if(len(moved)>0):
                                        moved.pop(selected_index)
                                
                            else:
                                print('Images must be 3D.')
                        else:
                            print('You must use an approprate type of input.')
                    except Exception as e:
                        print('Out of Memory or the parameters of Image Fusion tool should be selected properly:', e)
                # else:
                    # print('There is no image with the same name of',co,'in another folder.')
    executor.shutdown(wait=True)
    # for future in concurrent.futures.as_completed(futures):
    #     cc = 0  
        
       
        # we_fusion_final = weighted_Fusion(ct_image , pet_image ,float(w1), float(w2), interpolation)
        # _filename = fixed_img[3]
        
        # registered=we_fusion_final
        # header=fixed_img[1]
        # DatatypeFrom=fixed_img[2]
        # Datatype=fixed_img[2]
        # OUTPUT_DIR=destfolder
        # filename=_filename
        # createfolder='False'
        # if (DatatypeFrom == 'Dicom'):
        #     DatatypeFrom = 'SDicom'
        # if (Datatype == 'Dicom'):
        #     Datatype = 'SDicom'
            
        # if DatatypeFrom == 'Nifti' or DatatypeFrom == 'SDicom':
        #     perm = (2, 1, 0)
        #     registered = np.transpose(registered, perm) 
        # convert_modalities(registered , header , DatatypeFrom , Datatype , OUTPUT_DIR , filename ,createfolder)

    return ""

def weighted_fusion_folder_Thread(fixed_img,moved_img,ct_image , pet_image ,w1, w2, interpolation,destfolder):
    # ct_max_or = np.max(ct_image)
    ct_min_or = np.min(ct_image)
    we_fusion_final = weighted_Fusion(ct_image , pet_image ,float(w1), float(w2), interpolation)
    _filename = fixed_img[3]
    
    registered=we_fusion_final
    header=fixed_img[1]
    DatatypeFrom=fixed_img[2]
    Datatype=fixed_img[2]
    OUTPUT_DIR=destfolder
    filename=_filename
    createfolder='False'
    if (DatatypeFrom == 'Dicom'):
        DatatypeFrom = 'SDicom'
    if (Datatype == 'Dicom'):
        Datatype = 'SDicom'
        
    if DatatypeFrom == 'Nifti' or DatatypeFrom == 'SDicom':
        perm = (2, 1, 0)
        registered = np.transpose(registered, perm) 
        
    if ct_min_or >= 0 :
   
        if np.issubdtype(registered.dtype, np.floating) == True:
            ct_max = np.max(registered)
            
            if ct_max <= 1:
                registered = registered * 255
        
            registered = registered.astype(int)
        
    convert_modalities(registered , header , DatatypeFrom , Datatype , OUTPUT_DIR , filename ,createfolder)
    return ""



def HSI_fusion_folder( ct_path , pet_path , destfolder):

    Fixed_datas = os.listdir(ct_path)
    fixed = [i for i in Fixed_datas if (i.endswith(".nrrd") | i.endswith(".dcm") | i.endswith(".dicom") | i.endswith(".nii") | i.endswith(".nii.gz"))]

    Moved_datas = os.listdir(pet_path)
    moved = [i for i in Moved_datas if (i.endswith(".nrrd") | i.endswith(".dcm") | i.endswith(".dicom") | i.endswith(".nii") | i.endswith(".nii.gz"))]
    # thread_list = []
    # Num_worker = os.cpu_count() + int(psutil.virtual_memory()[1]/pow(10,9)/2)
    # Num_worker = 1
    Num_worker = int(psutil.virtual_memory()[1]/pow(10,9)/2)
    if Num_worker == 0:
        Num_worker = 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=Num_worker) as executor:
        futures = []
        for co in fixed:
            
            if len(moved) > 0:
                fixed_filename =  co.replace('.nii.gz', '').replace('.nii', '').replace('.nrrd', '').replace('.dcm', '').replace('.dicom','')        
                
                
                s_ratio = 0
                count = 0
                selected_index = 0
                for co_moved in moved:
                    moved_filename =  co_moved.replace('.nii.gz', '').replace('.nii', '').replace('.nrrd', '').replace('.dcm', '').replace('.dicom','')        
                    sim_ratio = similar(fixed_filename, moved_filename)
                    
                    if s_ratio < sim_ratio:
                        selected_index = count
                        s_ratio = sim_ratio
                    count = count +1
                
                if s_ratio > 0 :
        
                    Fixed_fullpath = os.path.join(ct_path,co)
                    fixed_img =  readimage(Fixed_fullpath)
                    
                    
                    Moved_fullpath = os.path.join(pet_path,moved[selected_index])
                    moved_img =  readimage(Moved_fullpath)
                    
                    # print(co ,'matched with', moved[selected_index],'.')
            
                    try:
                        if isinstance(moved_img[0], np.ndarray) & isinstance(fixed_img[0], np.ndarray):
                
                        # if moved_img[0] != None & fixed_img[0] != None:  

                            ct_image = fixed_img[0] 
                            pet_image = moved_img[0]
                            # print(ct_image.shape)
                            # print(pet_image.shape)
                    
                            if len(ct_image.shape) == 3 & len(pet_image.shape) == 3 :   
                    
                                if fixed_img[2] == 'Nifti' or fixed_img[2] == 'Dicom':
                                    perm = (2, 1, 0)
                                    ct_image = np.transpose(ct_image, perm)
                                if moved_img[2] == 'Nifti' or moved_img[2] == 'Dicom':
                                    perm = (2, 1, 0)
                                    pet_image = np.transpose(pet_image, perm)  
                                
                                    
                                dim01 = np.shape(ct_image)
                                dim02 = np.shape(pet_image)
                                if (dim01[0] == dim02[0]): 

                                    futures.append(executor.submit(HSI_fusion_folder_Thread, fixed_img,moved_img,ct_image , pet_image,destfolder))
                                    
                                
                                    if(len(moved)>0):
                                        moved.pop(selected_index)
                                
                            else:
                                print('Images must be 3D.')
                        else:
                            print('You must use an approprate type of input.')
                    except Exception as e:
                        print('Out of Memory or the parameters of registration tool should be selected properly:', e)
                # else:
                    # print('There is no image with the same name of',co,'in another folder.')
    executor.shutdown(wait=True)
    # for future in concurrent.futures.as_completed(futures):
    #     cc = 0 
        
    return ""
 

def HSI_fusion_folder_Thread(fixed_img,moved_img,ct_image , pet_image ,destfolder):
    # ct_max_or = np.max(ct_image)
    ct_min_or = np.min(ct_image)
    we_fusion_final = HSI_Fusion(ct_image , pet_image)
    _filename = fixed_img[3]
    
    registered=we_fusion_final
    header=fixed_img[1]
    DatatypeFrom=fixed_img[2]
    Datatype=fixed_img[2]
    OUTPUT_DIR=destfolder
    filename=_filename
    createfolder='False'
    if (DatatypeFrom == 'Dicom'):
        DatatypeFrom = 'SDicom'
    if (Datatype == 'Dicom'):
        Datatype = 'SDicom'
        
    if DatatypeFrom == 'Nifti' or DatatypeFrom == 'SDicom':
        perm = (2, 1, 0)
        registered = np.transpose(registered, perm) 
        
    if ct_min_or >= 0 :
   
        if np.issubdtype(registered.dtype, np.floating) == True:
            ct_max = np.max(registered)
            
            if ct_max <= 1:
                registered = registered * 255
        
            registered = registered.astype(int)
        
    convert_modalities(registered , header , DatatypeFrom , Datatype , OUTPUT_DIR , filename ,createfolder)
    return ""
