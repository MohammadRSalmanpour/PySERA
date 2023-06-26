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
from SERASUVscalingObj import SUVscalingObj
from Sera_ReadWrite import readimage, convert_modalities, similar
import psutil


##################################  registeration functon   ##################################


def register_simpleITK_array_3D_folder(fixed_PATH, moved_PATH, Num_bin, Sampling_percentage, lRate, num_Iterations,
                                       interpolator, registr_method, destfolder):
    Fixed_datas = os.listdir(fixed_PATH)
    fixed = [i for i in Fixed_datas if (
                i.endswith(".nrrd") | i.endswith(".dcm") | i.endswith(".dicom") | i.endswith(".nii") | i.endswith(
            ".nii.gz"))]

    Moved_datas = os.listdir(moved_PATH)
    moved = [i for i in Moved_datas if (
                i.endswith(".nrrd") | i.endswith(".dcm") | i.endswith(".dicom") | i.endswith(".nii") | i.endswith(
            ".nii.gz"))]
    # thread_list = []
    # Num_worker = os.cpu_count() + int(psutil.virtual_memory()[1]/pow(10,9)/4)
    # Num_worker = 1
    Num_worker = int(psutil.virtual_memory()[1] / pow(10, 9) / 2)
    if Num_worker == 0:
        Num_worker = 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=Num_worker) as executor:
        futures = []
        for co in fixed:
            if len(moved) > 0:

                fixed_filename = co.replace('.nii.gz', '').replace('.nii', '').replace('.nrrd', '').replace('.dcm',
                                                                                                            '').replace(
                    '.dicom', '')

                s_ratio = 0
                count = 0
                selected_index = 0
                for co_moved in moved:
                    moved_filename = co_moved.replace('.nii.gz', '').replace('.nii', '').replace('.nrrd', '').replace(
                        '.dcm', '').replace('.dicom', '')
                    sim_ratio = similar(fixed_filename, moved_filename)

                    if s_ratio < sim_ratio:
                        selected_index = count
                        s_ratio = sim_ratio
                    count = count + 1

                if s_ratio > 0:
                    Fixed_fullpath = os.path.join(fixed_PATH, co)
                    fixed_img = readimage(Fixed_fullpath)

                    Moved_fullpath = os.path.join(moved_PATH, moved[selected_index])
                    moved_img = readimage(Moved_fullpath)

                    # print(co ,'matched with', moved[selected_index],'.')
                    try:
                        if isinstance(moved_img[0], np.ndarray) & isinstance(fixed_img[0], np.ndarray):
                            # if moved_img[0] != None & fixed_img[0] != None:
                            if len(moved_img[0].shape) == 3 & len(fixed_img[0].shape) == 3:
                                futures.append(
                                    executor.submit(register_simpleITK_array_3D_folder_Thread, fixed_img, moved_img,
                                                    Num_bin,
                                                    Sampling_percentage, lRate,
                                                    num_Iterations, interpolator, registr_method, destfolder))

                                if (len(moved) > 0):
                                    moved.pop(selected_index)

                            else:
                                raise('Images must be 3D.')
                        else:
                            raise('You must use an approprate type of input.')
                    except Exception as e:
                        raise('Out of Memory or the parameters of registration tool should be selected properly:', e)
                # else:
                # print('There is no image with the same name of',co,'in another folder.')
    executor.shutdown(wait=True)
    # for future in concurrent.futures.as_completed(futures):
    #     cc = 0            
    # for thread in thread_list:
    #     thread.join() 

    # registered = register_simpleITK_array_3D(fixed_img[0] ,fixed_img[1] , moved_img[0] ,moved_img[1], Num_bin , Sampling_percentage , lRate, num_Iterations , interpolator , registr_method)
    # if (moved_img[2] == "Dicom"):
    #     moved_img[2] = "SDicom"
    # convert_modalities(registered[0], registered[1], 'Nifti', moved_img[2], destfolder, moved_img[3],createfolder='False')

    return ""


def register_simpleITK_array_3D_folder_Thread(fixed_img, moved_img, Num_bin,
                                              Sampling_percentage, lRate,
                                              num_Iterations, interpolator, registr_method, destfolder):
    registered = register_simpleITK_array_3D(fixed_img[0], fixed_img[1], moved_img[0], moved_img[1], Num_bin,
                                             Sampling_percentage, lRate, num_Iterations, interpolator, registr_method)
    if (moved_img[2] == "Dicom"):
        moved_img[2] = "SDicom"
        
    convert_modalities(registered[0], registered[1], 'Nifti', moved_img[2], destfolder, moved_img[3],
                       createfolder='False')

    return ""


def register_simpleITK_array_3D(fixed_array, fixed_header, moved_array, moved_header, Num_bin, Sampling_percentage,
                                lRate, num_Iterations, interpolator, registr_method):
    if str(type(moved_header)) == "<class 'collections.OrderedDict'>":

        perm = (2, 1, 0)
        moved_array = np.transpose(moved_array.astype(np.float32), perm)
        perm = (0, 1, 2)
        moving_image = sitk.GetImageFromArray(np.transpose(moved_array.astype(np.float32), perm), isVector=False)

        spacing = (moved_header['space directions'][0, 0], moved_header['space directions'][1, 1],
                   moved_header['space directions'][2, 2])
        origin = (moved_header['space origin'][0], moved_header['space origin'][1], moved_header['space origin'][2])

        a1 = +1.0 if moved_header['space'].split('-')[0] == 'left' else -1.0
        a2 = +1.0 if moved_header['space'].split('-')[1] == 'posterior' else -1.0
        a3 = +1.0 if moved_header['space'].split('-')[2] == 'superior' else -1.0
        direction = (a1, 0.0, 0.0, 0.0, a2, 0.0, 0.0, 0.0, a3)

        moving_image.SetOrigin(origin)
        moving_image.SetSpacing(spacing)
        moving_image.SetDirection(direction)

    elif str(type(moved_header)) == "<class 'SimpleITK.SimpleITK.Image'>":

        moving_image = moved_header

    elif str(type(moved_header)) == "<class 'nibabel.nifti1.Nifti1Image'>":

        perm = (2, 1, 0)
        moving_image = sitk.GetImageFromArray(np.transpose(moved_array.astype(np.float32), perm), isVector=False)

        spacing = (abs(moved_header.affine[0, 0]), abs(moved_header.affine[1, 1]), abs(moved_header.affine[2, 2]))

        or1 = moved_header.affine[0, 3] if moved_header.affine[0, 0] > 0 else (moved_header.affine[0, 3] * (-1))
        or2 = moved_header.affine[1, 3] if moved_header.affine[1, 1] > 0 else (moved_header.affine[1, 3] * (-1))
        or3 = moved_header.affine[2, 3] if moved_header.affine[2, 2] > 0 else (moved_header.affine[2, 3] * (-1))
        origin = (or1, or2, or3)

        a1 = +1.0 if nib.aff2axcodes(moved_header.affine)[0] == 'L' else -1.0
        a2 = +1.0 if nib.aff2axcodes(moved_header.affine)[1] == 'P' else -1.0
        a3 = +1.0 if nib.aff2axcodes(moved_header.affine)[2] == 'S' else -1.0
        direction = (a1, 0.0, 0.0, 0.0, a2, 0.0, 0.0, 0.0, a3)

        moving_image.SetOrigin(origin)
        moving_image.SetSpacing(spacing)
        moving_image.SetDirection(direction)

    if str(type(fixed_header)) == "<class 'collections.OrderedDict'>":

        perm = (2, 1, 0)
        fixed_array = np.transpose(fixed_array.astype(np.float32), perm)
        perm = (0, 1, 2)
        fixed_image = sitk.GetImageFromArray(np.transpose(fixed_array.astype(np.float32), perm), isVector=False)

        spacing = (fixed_header['space directions'][0, 0], fixed_header['space directions'][1, 1],
                   fixed_header['space directions'][2, 2])
        origin = (fixed_header['space origin'][0], fixed_header['space origin'][1], fixed_header['space origin'][2])

        a1 = +1.0 if fixed_header['space'].split('-')[0] == 'left' else -1.0
        a2 = +1.0 if fixed_header['space'].split('-')[1] == 'posterior' else -1.0
        a3 = +1.0 if fixed_header['space'].split('-')[2] == 'superior' else -1.0
        direction = (a1, 0.0, 0.0, 0.0, a2, 0.0, 0.0, 0.0, a3)

        fixed_image.SetOrigin(origin)
        fixed_image.SetSpacing(spacing)
        fixed_image.SetDirection(direction)


    elif str(type(fixed_header)) == "<class 'SimpleITK.SimpleITK.Image'>":

        fixed_image = fixed_header

    elif str(type(fixed_header)) == "<class 'nibabel.nifti1.Nifti1Image'>":

        perm = (2, 1, 0)
        fixed_image = sitk.GetImageFromArray(np.transpose(fixed_array.astype(np.float32), perm), isVector=False)

        spacing = (abs(fixed_header.affine[0, 0]), abs(fixed_header.affine[1, 1]), abs(fixed_header.affine[2, 2]))
        or1 = fixed_header.affine[0, 3] if fixed_header.affine[0, 0] > 0 else (fixed_header.affine[0, 3] * (-1))
        or2 = fixed_header.affine[1, 3] if fixed_header.affine[1, 1] > 0 else (fixed_header.affine[1, 3] * (-1))
        or3 = fixed_header.affine[2, 3] if fixed_header.affine[2, 2] > 0 else (fixed_header.affine[2, 3] * (-1))
        origin = (or1, or2, or3)

        a1 = +1.0 if nib.aff2axcodes(fixed_header.affine)[0] == 'L' else -1.0
        a2 = +1.0 if nib.aff2axcodes(fixed_header.affine)[1] == 'P' else -1.0
        a3 = +1.0 if nib.aff2axcodes(fixed_header.affine)[2] == 'S' else -1.0
        direction = (a1, 0.0, 0.0, 0.0, a2, 0.0, 0.0, 0.0, a3)

        fixed_image.SetOrigin(origin)
        fixed_image.SetSpacing(spacing)
        fixed_image.SetDirection(direction)
    initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                          moving_image,
                                                          sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    moving_resampled = sitk.Resample(moving_image, fixed_image, initial_transform, interpolator, 0.0,
                                     moving_image.GetPixelID())

    registration_method = sitk.ImageRegistrationMethod()
    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=Num_bin)

    registr_method = registr_method.upper()
    if registr_method == 'NONE':
        registration_method.SetMetricSamplingStrategy(registration_method.NONE)
    elif registr_method == 'RANDOM':
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    elif registr_method == 'REGULAR':
        registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)

    registration_method.SetMetricSamplingPercentage(Sampling_percentage)
    registration_method.SetInterpolator(interpolator)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=lRate, numberOfIterations=num_Iterations,
                                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
                                                  sitk.Cast(moving_image, sitk.sitkFloat32))
    moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, interpolator, 0.0,
                                     moving_image.GetPixelID())
    registered_data = sitk.GetArrayFromImage(moving_resampled)

    return [registered_data, moving_resampled]
