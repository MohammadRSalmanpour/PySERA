import ntpath
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
from PythonCode.dicom2nifti.convert_dir import convert_directory
# from dicom2nifti.convert_dicom import dicom_series_to_nifti
# from difflib import SequenceMatcher
import threading
import sys
from datetime import datetime
import traceback
import pydicom
import socketserver
import datetime
from PythonCode.SERASUVscalingObj import SUVscalingObj
import psutil


##################################  similarity functons   ##################################

def similar(a, b):
    # return 

    if a == b:
        return 1
    else:
        return 0
        # a = a.replace('.nii.gz', '').replace('.nii', '').replace('.nrrd', '').replace('.dcm', '').replace('.dicom','')
        # b = b.replace('.nii.gz', '').replace('.nii', '').replace('.nrrd', '').replace('.dcm', '').replace('.dicom','')
        # if a == b :
        #     return 1
        # # elif SequenceMatcher(None, a, b).ratio() > 0.85:
        # #     return 0.85
        # else:
        #     return 0 


def ReadDicomBySITK(dcm_dir):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcm_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    orient = sitk.DICOMOrientImageFilter()
    DirectionStr = orient.GetOrientationFromDirectionCosines(image.GetDirection())

    return image, image.GetSpacing(), image.GetOrigin(), image.GetDirection(), DirectionStr


##################################  read functons   ##################################

def readimage(souecefile):
    intype = ''
    result = []
    if souecefile.endswith('.nii.gz') | souecefile.endswith('.nii'):
        result = Nifti_read(souecefile)
        intype = 'Nifti'
    elif souecefile.endswith('.nrrd'):
        result = Nrrd_read(souecefile)
        intype = 'Nrrd'
    elif souecefile.endswith('.dcm') | souecefile.endswith('.dicom'):
        intype = 'Dicom'
        file_name = souecefile.split('\\')[-1]
        data_directory = souecefile.replace(file_name, '')[:-1]
        result = Dicom_read_simpleITK(data_directory, file_name)

    _filename = ntpath.basename(souecefile).replace('.nii.gz', '').replace('.nii', '').replace('.nrrd', '').replace(
        '.dcm', '').replace('.dicom', '')

    if (result == None):
        return [None, None, intype, _filename]
    elif (len(result) == 1):
        return [result[0], None, intype, _filename]
    elif (len(result) > 1):
        return [result[0], result[1], intype, _filename]


def Dicom_read_simpleITK(data_directory, file_name):
    try:
        file_reader = sitk.ImageFileReader()
        datapath = os.path.join(data_directory, file_name)
        file_reader.SetFileName(datapath)
        file_reader.ReadImageInformation()
        series_ID = file_reader.GetMetaData('0020|000e')
        sorted_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(data_directory, series_ID)
        header = sitk.ReadImage(sorted_file_names)
        data = sitk.GetArrayFromImage(header)

        leng = len(np.shape(data))
        ShapeList = []
        perm = []
        for i in range(1, leng):
            ShapeList.append(np.shape(data)[i])
            perm.append(i - 1)

        isNormal = True
        try:
            data = data.reshape(tuple(ShapeList))
        except:
            isNormal = False

        if isNormal:
            perm = tuple(perm)
            img = sitk.GetImageFromArray(np.transpose(data.astype(np.float32), perm), isVector=False)

            direction = header.GetDirection()
            direction = np.asarray(direction)
            MatSize = np.sqrt(len(direction))
            direction = direction.reshape((int(MatSize), int(MatSize)))

            newDirection = direction[0:-1, 0:-1]
            newDirection = newDirection.flatten()

            img.SetOrigin(header.GetOrigin()[:-1])
            img.SetSpacing(header.GetSpacing()[:-1])
            img.SetDirection(tuple(newDirection))

            data = sitk.GetArrayFromImage(img)

            return [data, img]
        else:
            # print('Data and header unmatched') 
            return None
    except:
        return None


def Nifti_read(fullpath):
    try:
        img = sitk.ReadImage(fullpath)
        data = sitk.GetArrayFromImage(img)
        return [data, img]
    except:
        return None


def Nrrd_read(fullpath):
    try:
        fixed_image = nrrd.read(fullpath)
        data = fixed_image[0]
        info = fixed_image[1]
        return [data, info]
    except:
        return None


def readFolder_nii(folder, _types, createfolder, destfolder):
    name_of_slice = os.listdir(folder)
    files_nii = [i for i in name_of_slice if (i.endswith(".nii") | i.endswith(".nii.gz"))]
    # Num_worker = os.cpu_count() + int(psutil.virtual_memory()[1]/pow(10,9))
    # Num_worker = 1
    Num_worker = int(psutil.virtual_memory()[1] / pow(10, 9))
    if Num_worker == 0:
        Num_worker = 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=Num_worker) as executor:
        futures = []
        for i in files_nii:
            fullpath = os.path.join(folder, i)
            futures.append(executor.submit(Nifti_read_Thread, i, fullpath, _types, createfolder, destfolder))
    executor.shutdown(wait=True)
    return ''


def Nifti_read_Thread(name, fullpath, _types, createfolder, destfolder):
    img = Nifti_read(fullpath)
    Fullsouecefile = fullpath
    thread_list = []
    inType = 'Nifti'
    _filename = name.replace('.nii.gz', '').replace('.nii', '').replace('.nrrd', '').replace('.dcm', '').replace(
        '.dicom', '')
    for _type in _types:
        if _type != inType:
            t = threading.Thread(target=convert_modalities,
                                 args=(img[0], img[1], 'Nifti', _type, destfolder, _filename, createfolder))
            thread_list.append(t)
            t.start()
        elif _type == inType:
            filename = _filename + '_' + inType
            if createfolder == 'True':
                create_Folder_path = os.path.join(destfolder, filename)
                if os.path.isdir(create_Folder_path) == False:
                    os.mkdir(create_Folder_path)
                fullpath = os.path.join(create_Folder_path, _filename + '.nii.gz')
            else:
                fullpath = os.path.join(destfolder, _filename + '.nii.gz')
            # Fullsouecefile = os.path.join(sourcefolder,name)
            shutil.copy(Fullsouecefile, fullpath)
    for thread in thread_list:
        thread.join()


def readFolder_nrrd(folder, _types, createfolder, destfolder):
    name_of_slice = os.listdir(folder)
    files_nrrd = [i for i in name_of_slice if (i.endswith(".nrrd"))]
    # Num_worker = os.cpu_count() + int(psutil.virtual_memory()[1]/pow(10,9))
    # Num_worker = 1
    Num_worker = int(psutil.virtual_memory()[1] / pow(10, 9))
    if Num_worker == 0:
        Num_worker = 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=Num_worker) as executor:
        futures = []
        for i in files_nrrd:
            fullpath = os.path.join(folder, i)
            futures.append(executor.submit(Nrrd_read_Thread, i, fullpath, _types, createfolder, destfolder))
    executor.shutdown(wait=True)

    return ''


def Nrrd_read_Thread(name, fullpath, _types, createfolder, destfolder):
    img = Nrrd_read(fullpath)
    Fullsouecefile = fullpath

    thread_list = []
    inType = 'Nrrd'
    _filename = name.replace('.nii.gz', '').replace('.nii', '').replace('.nrrd', '').replace('.dcm', '').replace(
        '.dicom', '')
    for _type in _types:
        if _type != inType:
            t = threading.Thread(target=convert_modalities,
                                 args=(img[0], img[1], 'Nrrd', _type, destfolder, _filename, createfolder))
            thread_list.append(t)
            t.start()
        elif _type == inType:

            filename = _filename + '_' + inType
            if createfolder == 'True':
                create_Folder_path = os.path.join(destfolder, filename)
                if os.path.isdir(create_Folder_path) == False:
                    os.mkdir(create_Folder_path)
                fullpath = os.path.join(create_Folder_path, _filename + '.nrrd')
            else:
                fullpath = os.path.join(destfolder, _filename + '.nrrd')
            # Fullsouecefile = os.path.join(sourcefolder,name)
            shutil.copy(Fullsouecefile, fullpath)
    for thread in thread_list:
        thread.join()


def readFolder_dicom_2D_recurcive(folder, _types, createfolder, destfolder, suv):
    name_of_subdirectories = os.listdir(folder)
    subdirectories_dicom = []
    for i in name_of_subdirectories:
        d = os.path.join(folder, i)
        if os.path.isdir(d):
            subdirectories_dicom.append(i)
    thread_list = []
    if len(subdirectories_dicom) == 0:
        patients = convert_directory(folder, None, compression=True, reorient=True)
        Headers = patients[0]
        # filenames = patients[1]
        filepaths = patients[2]
        ImagePositionPatients = patients[4]
        ImageOrientationPatient = patients[5]
        step = patients[6]
        PixelSpacing = patients[7]

        number = len(Headers)
        for co in range(0, number):
            if Headers[co] != None:
                data = Headers[co].get_fdata()
                # inType = 'MDicom'   
                changeinType = 'Nifti'
                # _filename = filenames[co].replace('.nii.gz', '').replace('.nii', '').replace('.nrrd', '').replace('.dcm', '').replace('.dicom','')
                _filename = filepaths[co].split('\\')[-2]
                for _type in _types:
                    t = threading.Thread(target=convert_modalities, args=(
                        data, Headers[co], changeinType, _type, destfolder, _filename, createfolder,
                        ImagePositionPatients[co], ImageOrientationPatient[co], step[co], PixelSpacing[co],
                        filepaths[co]))
                    thread_list.append(t)
                    t.start()
        for thread in thread_list:
            thread.join()
        if suv == True:
            SUVresult = []
            number = len(Headers)
            cont = False
            for co in range(0, number):
                if Headers[co] != None:
                    try:
                        data = Headers[co].get_fdata()
                        _filename = filepaths[co].split('\\')[-2]
                        # returned_val = compute_suv(data,patients[3])

                        # root = folder
                        # pet_image_file_list = [os.path.join(root,x) for x in os.listdir(root) if x.endswith('.dcm')]
                        # pet_image_file_list = sort_by_instance_number(pet_image_file_list)
                        # data = imread(pet_image_file_list)
                        # f=pydicom.dcmread(pet_image_file_list[0])
                        # returned_val = compute_suv(data,f)

                        # root = folder
                        # pet_image_file_list = [os.path.join(root,x) for x in os.listdir(root) if x.endswith('.dcm')]
                        # pet_image_file_list = sort_by_instance_number(pet_image_file_list)
                        # # data = imread(pet_image_file_list)
                        # f=pydicom.dcmread(pet_image_file_list[0])
                        #      
                        f = pydicom.dcmread(filepaths[co])
                        SUV_OBJ = SUVscalingObj(f)
                        returned_val = SUV_OBJ.get_scale_factor(suv_normalisation="bw", data=data)

                        convert_modalities(returned_val[0], Headers[co], 'Nifti', 'Nifti', destfolder,
                                           _filename + '_SUV_Calculation', createfolder, ImagePositionPatients[co],
                                           ImageOrientationPatient[co], step[co], PixelSpacing[co], filepaths[co])

                        suv_max = np.max(returned_val[0])
                        suv_mean = np.mean(returned_val[0])
                        suv_min = np.min(returned_val[0])
                        suv_std = np.std(returned_val[0])

                        SUVresult.append([_filename, suv_min, suv_max, suv_mean, suv_std, returned_val[1]])
                        cont = True

                    except:
                        data = Headers[co].get_fdata()
                        _filename = filepaths[co].split('\\')[-2]
                        f = pydicom.dcmread(filepaths[co])
                        returned_val = compute_suv(data, f)

                        convert_modalities(returned_val[0], Headers[co], 'Nifti', 'Nifti', destfolder,
                                           _filename + '_SUV_Estimation', createfolder, ImagePositionPatients[co],
                                           ImageOrientationPatient[co], step[co], PixelSpacing[co], filepaths[co])

                        suv_max = np.max(returned_val[0])
                        suv_mean = np.mean(returned_val[0])
                        suv_min = np.min(returned_val[0])
                        suv_std = np.std(returned_val[0])

                        SUVresult.append([_filename, suv_min, suv_max, suv_mean, suv_std, returned_val[1]])
                        cont = True
            if cont == True:
                SUVresult_arr = np.asarray(SUVresult)
                col = ['FileName', 'suv_min', 'suv_max', 'suv_mean', 'suv_std', 'estimated']

                SUVresult_df = pd.DataFrame(SUVresult_arr)
                SUVresult_df.columns = col

                CSVFilename = "SUV_Report_Multi_Dicom.csv"
                CSVfullpath = os.path.join(destfolder, CSVFilename)
                SUVresult_df.to_csv(CSVfullpath, index=None)

        return ""
    else:
        sucsessList = []
        # Num_worker = os.cpu_count() + int(psutil.virtual_memory()[1]/pow(10,9))
        # Num_worker = 1
        Num_worker = int(psutil.virtual_memory()[1] / pow(10, 9))
        if Num_worker == 0:
            Num_worker = 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=Num_worker) as executor:
            futures = []
            for j in subdirectories_dicom:
                fullpath = os.path.join(folder, j)
                futures.append(
                    executor.submit(readFolder_dicom_2D_recurcive_Thread, j, fullpath, _types, createfolder, destfolder,
                                    suv, sucsessList))
        executor.shutdown(wait=True)

        if len(sucsessList) > 0:
            col = ['FileName', 'suv_min', 'suv_max', 'suv_mean', 'suv_std', 'estimated']
            SUVresult_arr = np.asarray(sucsessList)
            # SUVresult_arr.flatten()
            SUVresult_df = pd.DataFrame(SUVresult_arr)
            # print(col)
            SUVresult_df.columns = col

            CSVFilename = "SUV_Report_Multi_Dicom.csv"
            CSVfullpath = os.path.join(destfolder, CSVFilename)
            SUVresult_df.to_csv(CSVfullpath, index=None)

        return ''


def readFolder_dicom_2D_recurcive_Thread(name, fullpath, _types, createfolder, destfolder, suv, sucsessList):
    patients = convert_directory(fullpath, None, compression=True, reorient=True)
    # inType = 'MDicom'   
    changeinType = 'Nifti'
    Headers = patients[0]
    # filenames = patients[1]
    filepaths = patients[2]
    ImagePositionPatients = patients[4]
    ImageOrientationPatient = patients[5]
    step = patients[6]
    PixelSpacing = patients[7]

    thread_list = []
    number = len(Headers)
    for co in range(0, number):
        if Headers[co] != None:
            data = Headers[co].get_fdata()
            # _filename = filenames[co].replace('.nii.gz', '').replace('.nii', '').replace('.nrrd', '').replace('.dcm', '').replace('.dicom','')
            _filename = filepaths[co].split('\\')[-2]
            destfolder2 = os.path.join(destfolder, name)
            if os.path.isdir(destfolder2) == False:
                os.mkdir(destfolder2)
            for _type in _types:
                t = threading.Thread(target=convert_modalities, args=(
                    data, Headers[co], changeinType, _type, destfolder2, _filename, createfolder,
                    ImagePositionPatients[co],
                    ImageOrientationPatient[co], step[co], PixelSpacing[co], filepaths[co]))
                thread_list.append(t)
                t.start()
    for thread in thread_list:
        thread.join()

    if suv == True:
        # SUVresult = []
        number = len(Headers)
        for co in range(0, number):
            if Headers[co] != None:
                try:
                    data = Headers[co].get_fdata()
                    _filename = filepaths[co].split('\\')[-2]
                    # returned_val = compute_suv(data,patients[3])

                    # root = fullpath
                    # pet_image_file_list = [os.path.join(root,x) for x in os.listdir(root) if x.endswith('.dcm')]
                    # pet_image_file_list = sort_by_instance_number(pet_image_file_list)
                    # data = imread(pet_image_file_list)
                    # f=pydicom.dcmread(pet_image_file_list[0])
                    # returned_val = compute_suv(data,f)

                    # root = fullpath
                    # pet_image_file_list = [os.path.join(root,x) for x in os.listdir(root) if x.endswith('.dcm')]
                    # pet_image_file_list = sort_by_instance_number(pet_image_file_list)
                    # data = imread(pet_image_file_list)
                    # f=pydicom.dcmread(pet_image_file_list[0])  
                    # 
                    f = pydicom.dcmread(filepaths[co])
                    SUV_OBJ = SUVscalingObj(f)
                    returned_val = SUV_OBJ.get_scale_factor(suv_normalisation="bw", data=data)

                    convert_modalities(returned_val[0], Headers[co], 'Nifti', 'Nifti', destfolder2,
                                       _filename + '_SUV_Calculation', createfolder, ImagePositionPatients[co],
                                       ImageOrientationPatient[co], step[co], PixelSpacing[co], filepaths[co])

                    suv_max = np.max(returned_val[0])
                    suv_mean = np.mean(returned_val[0])
                    suv_min = np.min(returned_val[0])
                    suv_std = np.std(returned_val[0])

                    # SUVresult.append([_filename,suv_min,suv_max,suv_mean,suv_std,returned_val[1]])

                    lock = threading.Lock()
                    lock.acquire()
                    # SUVresult_arr = np.asarray(SUVresult) 
                    sucsessList.append([_filename, suv_min, suv_max, suv_mean, suv_std, returned_val[1]])
                    lock.release()

                except:

                    data = Headers[co].get_fdata()
                    _filename = filepaths[co].split('\\')[-2]
                    f = pydicom.dcmread(filepaths[co])
                    returned_val = compute_suv(data, f)

                    convert_modalities(returned_val[0], Headers[co], 'Nifti', 'Nifti', destfolder2,
                                       _filename + '_SUV_Estimation', createfolder, ImagePositionPatients[co],
                                       ImageOrientationPatient[co], step[co], PixelSpacing[co], filepaths[co])

                    suv_max = np.max(returned_val[0])
                    suv_mean = np.mean(returned_val[0])
                    suv_min = np.min(returned_val[0])
                    suv_std = np.std(returned_val[0])

                    lock = threading.Lock()
                    lock.acquire()
                    sucsessList.append([_filename, suv_min, suv_max, suv_mean, suv_std, returned_val[1]])
                    lock.release()
    return ''


def readFolder_dicom_nD(folder, _types, createfolder, destfolder, suv):
    list_of_patients = os.listdir(folder)
    files_dicom3D = [i for i in list_of_patients if (i.endswith(".dcm") | i.endswith(".dicom"))]
    sucsessList = []
    # Num_worker = os.cpu_count() + int(psutil.virtual_memory()[1]/pow(10,9))
    # Num_worker = 1
    Num_worker = int(psutil.virtual_memory()[1] / pow(10, 9))
    if Num_worker == 0:
        Num_worker = 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=Num_worker) as executor:
        futures = []
        for i in files_dicom3D:
            futures.append(
                executor.submit(Dicom_read_simpleITK_Thread, i, folder, _types, createfolder, destfolder, suv,
                                sucsessList))
    executor.shutdown(wait=True)

    if len(sucsessList) > 0:
        col = ['FileName', 'suv_min', 'suv_max', 'suv_mean', 'suv_std', 'estimated']
        SUVresult_arr = np.asarray(sucsessList)
        # SUVresult_arr.flatten()
        SUVresult_df = pd.DataFrame(SUVresult_arr)
        # print(col)
        SUVresult_df.columns = col

        CSVFilename = "SUV_Report_Single_Dicom.csv"
        CSVfullpath = os.path.join(destfolder, CSVFilename)
        SUVresult_df.to_csv(CSVfullpath, index=None)

    return ''


def Dicom_read_simpleITK_Thread(name, folder, _types, createfolder, destfolder, suv, sucsessList):
    img = Dicom_read_simpleITK(folder, name)

    inType = 'SDicom'
    changeinType = 'Nifti'
    thread_list = []

    # file_names = dicomS[3][co].split('\\')[-1]
    _filename = name.replace('.nii.gz', '').replace('.nii', '').replace('.nrrd', '').replace('.dcm', '').replace(
        '.dicom', '')
    print(_filename)
    for _type in _types:
        if _type != inType:
            t = threading.Thread(target=convert_modalities,
                                 args=(img[0], img[1], changeinType, _type, destfolder, _filename, createfolder))
            thread_list.append(t)
            t.start()

        if _type == inType:

            filename = _filename + '_' + inType
            if createfolder == 'True':
                create_Folder_path = os.path.join(destfolder, filename)
                if os.path.isdir(create_Folder_path) == False:
                    os.mkdir(create_Folder_path)
                fullpath = os.path.join(create_Folder_path, _filename + '.dcm')
            else:
                fullpath = os.path.join(destfolder, _filename + '.dcm')
            # Fullsouecefile = os.path.join(sourcefolder,d[2][co])
            shutil.copy(os.path.join(folder, name), fullpath)
    for thread in thread_list:
        thread.join()

    if suv == True:
        try:
            data = img[0]
            header = img[1]
            Dcmheader = pydicom.dcmread(os.path.join(folder, name))
            # returned_val = compute_suv(data,Dcmheader)

            # pet_image_file_list = [os.path.join(folder,name)]
            # pet_image_file_list = sort_by_instance_number(pet_image_file_list)
            # data = imread(pet_image_file_list)
            # returned_val = compute_suv(data,Dcmheader)

            # pet_image_file_list = [os.path.join(folder,name)]
            # pet_image_file_list = sort_by_instance_number(pet_image_file_list)
            # data = imread(pet_image_file_list)
            SUV_OBJ = SUVscalingObj(Dcmheader)
            returned_val = SUV_OBJ.get_scale_factor(suv_normalisation="bw", data=data)

            convert_modalities(returned_val[0], header, 'Nifti', 'Nifti', destfolder, _filename + '_SUV_Calculation',
                               createfolder)

            suv_max = np.max(returned_val[0])
            suv_mean = np.mean(returned_val[0])
            suv_min = np.min(returned_val[0])
            suv_std = np.std(returned_val[0])

            lock = threading.Lock()
            lock.acquire()
            sucsessList.append([_filename, suv_min, suv_max, suv_mean, suv_std, returned_val[1]])
            lock.release()
        except:
            data = img[0]
            header = img[1]
            Dcmheader = pydicom.dcmread(os.path.join(folder, name))
            returned_val = compute_suv(data, Dcmheader)

            convert_modalities(returned_val[0], header, 'Nifti', 'Nifti', destfolder, _filename + '_SUV_Estimation',
                               createfolder)

            suv_max = np.max(returned_val[0])
            suv_mean = np.mean(returned_val[0])
            suv_min = np.min(returned_val[0])
            suv_std = np.std(returned_val[0])

            lock = threading.Lock()
            lock.acquire()
            sucsessList.append([_filename, suv_min, suv_max, suv_mean, suv_std, returned_val[1]])
            lock.release()

        ##################################  Convertion functon   ##################################


def Convert_nifti_to_others(registered, header, Datatype, OUTPUT_DIR, filename, createfolder):
    if Datatype == 'Nifti':
        perm = (0, 1, 2)
        img = sitk.GetImageFromArray(np.transpose(registered.astype(np.float32), perm), isVector=False)

        img.SetOrigin(header.GetOrigin())
        img.SetSpacing(header.GetSpacing())
        img.SetDirection(header.GetDirection())

        if createfolder == 'True':
            if '_SUV' in filename:
                filename2 = filename
            else:
                filename2 = filename + '_Nifti'
            create_Folder_path = os.path.join(OUTPUT_DIR, filename2)
            if os.path.isdir(create_Folder_path) == False:
                os.mkdir(create_Folder_path)
            sitk.WriteImage(img, os.path.join(OUTPUT_DIR, filename2, filename + '.nii.gz'))
            return os.path.join(OUTPUT_DIR, filename2, filename + '.nii.gz')
        else:
            sitk.WriteImage(img, os.path.join(OUTPUT_DIR, filename + '.nii.gz'))
            return os.path.join(OUTPUT_DIR, filename + '.nii.gz')

    elif Datatype == 'Nrrd':

        perm = (0, 1, 2)
        img = sitk.GetImageFromArray(np.transpose(registered.astype(np.float32), perm), isVector=False)

        img.SetOrigin(header.GetOrigin())
        img.SetSpacing(header.GetSpacing())
        img.SetDirection(header.GetDirection())

        orient = sitk.DICOMOrientImageFilter()
        DirectionStr = orient.GetOrientationFromDirectionCosines(header.GetDirection())

        cont = True
        f1 = 'left' if 'L' in DirectionStr else 'right'
        f2 = 'posterior' if 'P' in DirectionStr else 'anterior'
        f3 = 'superior' if 'S' in DirectionStr else 'Inferior'
        spaceStr2 = f1 + '-' + f2 + '-' + f3

        if spaceStr2 == 'right-anterior-superior' or spaceStr2 == 'left-anterior-superior' or spaceStr2 == 'left-posterior-superior':
            _ = spaceStr2
        else:
            # spaceStr = 'right-anterior-superior'
            cont = False

        if cont:

            if createfolder == 'True':
                filename2 = filename + '_Nrrd'
                create_Folder_path = os.path.join(OUTPUT_DIR, filename2)
                if os.path.isdir(create_Folder_path) == False:
                    os.mkdir(create_Folder_path)
                sitk.WriteImage(img, os.path.join(OUTPUT_DIR, filename2, filename + '.nrrd'))
                return os.path.join(OUTPUT_DIR, filename2, filename + '.nrrd')
            else:
                sitk.WriteImage(img, os.path.join(OUTPUT_DIR, filename + '.nrrd'))
                return os.path.join(OUTPUT_DIR, filename + '.nrrd')

        else:
            print('Nrrd Format doesn''t work with', spaceStr2, 'direction')

        # perm = (2, 1, 0)
        # data = np.transpose(registered.astype(np.float32), perm)

        # orient = sitk.DICOMOrientImageFilter()
        # DirectionStr = orient.GetOrientationFromDirectionCosines(header.GetDirection())

        # # a1 = header.GetDirection()[0]
        # # # a2 = header.GetDirection()[1]
        # # # a3 = header.GetDirection()[2]
        # # # a4 = header.GetDirection()[3]
        # # a5 = header.GetDirection()[4]
        # # # a6 = header.GetDirection()[5]
        # # # a7 = header.GetDirection()[6]
        # # # a8 = header.GetDirection()[7]
        # # a9 = header.GetDirection()[8]

        # # f1 = 'left' if a1 == 1 else 'right'
        # # f2 = 'posterior' if a5 == 1  else 'anterior' 
        # # f3 = 'superior' if a9 == 1  else 'Inferior'
        # # spaceStr = f1+'-'+f2+'-'+f3

        # # f1 = 'left' if a1 > 0 else 'right'
        # # f2 = 'posterior' if a5 > 0  else 'anterior' 
        # # f3 = 'superior' if a9 > 0  else 'Inferior'
        # # spaceStr = f1+'-'+f2+'-'+f3

        # cont = True
        # f1 = 'left' if 'L' in DirectionStr else 'right'
        # f2 = 'posterior' if 'P' in DirectionStr  else 'anterior' 
        # f3 = 'superior' if 'S' in DirectionStr  else 'Inferior'
        # spaceStr2 = f1+'-'+f2+'-'+f3           

        # if spaceStr2 == 'right-anterior-superior' or spaceStr2 == 'left-anterior-superior' or spaceStr2 == 'left-posterior-superior':
        #     spaceStr = spaceStr2
        # else:
        #     # spaceStr = 'right-anterior-superior'
        #     cont = False

        # if cont:
        #     fileheader = { 
        #         'dimension' : header.GetDimension(),
        #         'sizes' : np.array([header.GetWidth() ,header.GetHeight() ,header.GetDepth()]),
        #         'space': spaceStr,  
        #         # 'spacings': [header.GetSpacing()[0], header.GetSpacing()[1], header.GetSpacing()[2]],
        #         # 'thicknesses': [header.GetSpacing()[0], header.GetSpacing()[1], header.GetSpacing()[2]],
        #         # 'space directions': np.array([[a1,a2,a3],[a4,a5,a6],[a7,a8,a9]]), 
        #         # 'space directions': np.array([[header.GetSpacing()[0],a2,a3],[a4,header.GetSpacing()[1],a6],[a7,a8,header.GetSpacing()[2]]]), 
        #         'space directions': np.array([[header.GetSpacing()[0],0,0],[0,header.GetSpacing()[1],0],[0,0,header.GetSpacing()[2]]]), 
        #         'space origin': np.array([header.GetOrigin()[0] ,header.GetOrigin()[1] ,header.GetOrigin()[2]])
        #     }

        #     if createfolder == 'True':
        #         filename2 =  filename + '_NRRD'

        #         create_Folder_path = os.path.join(OUTPUT_DIR,filename2)
        #         if os.path.isdir(create_Folder_path) == False:
        #             os.mkdir(create_Folder_path)
        #         nrrd.write(os.path.join(OUTPUT_DIR,filename2,filename+'.nrrd' ), data, fileheader)
        #     else:
        #         nrrd.write(os.path.join(OUTPUT_DIR,filename+'.nrrd' ), data, fileheader)
        # else:
        #     print('Nrrd Format doesn''t work with',spaceStr2,'direction')

    elif Datatype == 'MDicom':

        un_precision = False
        registered_max = np.max(registered)
        if registered_max > pow(2, 16):
            un_precision = True

        registered_min = np.min(registered)
        if registered_min < 0:
            registered = registered.astype(np.int16)  # best ct
        else:
            registered = registered.astype(np.uint16)  # best pet

        new_img = sitk.GetImageFromArray(registered)

        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()

        modification_time = time.strftime("%H%M%S")
        modification_date = time.strftime("%Y%m%d")

        direction = header.GetDirection()
        origin = header.GetOrigin()
        spaceiing = header.GetSpacing()

        rand_number_Mdicom = randint(1, 10000) * 19
        series_tag_values = [
            ("0008|0031", modification_time),
            ("0008|0021", modification_date),
            ("0008|0008", "DERIVED\\SECONDARY"),
            ("0020|000e", "1.2.826.0.1.3680043.2.1125." + str(rand_number_Mdicom) + "."
             + modification_date + ".1" + modification_time),
            # ("0020|0037", '\\'.join(map(str, (direction[0], direction[1],direction[2],
            #                                   direction[3],direction[4],
            #                                   direction[5])))),# Image Orientation
            ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],  # Image Orientation (Patient)
                                              direction[1], direction[4], direction[7])))),
            ("0020|0032", '\\'.join(map(str, (origin[0], origin[1], origin[2])))),
            ("0028|0030", '\\'.join(map(str, (spaceiing[0], spaceiing[1])))),
            ("0018|0050", str(spaceiing[2])),
            ("0008|103e", filename)  # Series Description
            # ("0020,0011", rand_number_Mdicom )      # Series Number
        ]

        new_img.SetDirection(header.GetDirection())
        new_img.SetSpacing(header.GetSpacing())
        new_img.SetOrigin(header.GetOrigin())

        if createfolder == 'True':
            filename2 = filename + '_MDicom'
            create_Folder_path = os.path.join(OUTPUT_DIR, filename2)
            if os.path.isdir(create_Folder_path) == False:
                os.mkdir(create_Folder_path)

        for i in range(new_img.GetDepth()):
            image_slice = new_img[:, :, i]
            list(map(lambda tag_value: image_slice.SetMetaData(tag_value[0], tag_value[1]), series_tag_values))
            for tag, value in series_tag_values:
                image_slice.SetMetaData(tag, value)
            image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))  # Instance Creation Date
            image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))  # Instance Creation Time
            image_slice.SetMetaData("0008|0060", "CT")  # set the type to CT so the thickness is carried over
            image_slice.SetMetaData("0020|0032", '\\'.join(
                map(str, new_img.TransformIndexToPhysicalPoint((0, 0, i)))))  # Image Position (Patient)
            image_slice.SetMetaData("0020,0013", str(i))  # Instance Number

            # if createfolder == 'True':
            #     outputpathWrite = os.path.join(create_Folder_path, filename)
            # else:
            #     outputpathWrite = os.path.join(OUTPUT_DIR, filename)

            # if os.path.isdir(outputpathWrite) == False:     
            #     os.mkdir(outputpathWrite)

            # writer.SetFileName(os.path.join(outputpathWrite , filename + '_' + str(i) + '.dcm'))

            if createfolder == 'True':
                outputpathWrite = os.path.join(create_Folder_path, filename + '_' + str(i) + '.dcm')
            else:
                outputpathWrite = os.path.join(OUTPUT_DIR, filename + '_' + str(i) + '.dcm')

            writer.SetFileName(outputpathWrite)
            writer.Execute(image_slice)

        if un_precision:
            print(
                'voxel intensity in the converted image was limited to a 16-bit integer, so the converted image is not precise.')
        return outputpathWrite

    elif Datatype == 'SDicom':

        un_precision = False
        registered_max = np.max(registered)
        if registered_max > pow(2, 16):
            un_precision = True

        registered_min = np.min(registered)
        if registered_min < 0:
            registered = registered.astype(np.int16)  # best ct
        else:
            registered = registered.astype(np.uint16)  # best pet

        new_img = sitk.GetImageFromArray(registered)

        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()

        modification_time = time.strftime("%H%M%S")
        modification_date = time.strftime("%Y%m%d")

        direction = header.GetDirection()
        origin = header.GetOrigin()
        spaceiing = header.GetSpacing()

        rand_number_Sdicom = randint(200000, 210000) * 13

        series_tag_values = [
            ("0008|0031", modification_time),
            ("0008|0021", modification_date),
            ("0008|0008", "DERIVED\\SECONDARY"),
            ("0020|000e", "1.2.826.0.1.3680043.2.1125." + str(rand_number_Sdicom) + "."
             + modification_date + ".1" + modification_time),
            # ("0020|0037", '\\'.join(map(str, (direction[0], direction[1],direction[2],
            #                                   direction[3],direction[4],
            #                                   direction[5])))),# Image Orientation
            ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],  # Image Orientation (Patient)
                                              direction[1], direction[4], direction[7])))),
            ("0020|0032", '\\'.join(map(str, (origin[0], origin[1], origin[2])))),
            ("0028|0030", '\\'.join(map(str, (spaceiing[0], spaceiing[1])))),
            ("0018|0050", str(spaceiing[2])),
            ("0008|103e", filename)  # Series Description
            # ("0020,0011", rand_number_Sdicom )      # Series Number

        ]

        new_img.SetDirection(header.GetDirection())
        new_img.SetSpacing(header.GetSpacing())
        new_img.SetOrigin(header.GetOrigin())

        if createfolder == 'True':
            filename2 = filename + '_SDicom'
            create_Folder_path = os.path.join(OUTPUT_DIR, filename2)
            if os.path.isdir(create_Folder_path) == False:
                os.mkdir(create_Folder_path)

        image_slice = new_img[:, :, :]
        list(map(lambda tag_value: image_slice.SetMetaData(tag_value[0], tag_value[1]), series_tag_values))
        for tag, value in series_tag_values:
            image_slice.SetMetaData(tag, value)
        image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))  # Instance Creation Date
        image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))  # Instance Creation Time
        image_slice.SetMetaData("0008|0060", "CT")  # set the type to CT so the thickness is carried over
        image_slice.SetMetaData("0020|0032", '\\'.join(
            map(str, new_img.TransformIndexToPhysicalPoint((0, 0, 0)))))  # Image Position (Patient)
        # image_slice.SetMetaData("0020,0013", str(i)) # Instance Number
        # image_slice.SetMetaData("0020,0011", str(i)) # Series Number

        # if createfolder == 'True':
        #     outputpathWrite = os.path.join(create_Folder_path, filename)
        # else:
        #     outputpathWrite = os.path.join(OUTPUT_DIR, filename)

        # if os.path.isdir(outputpathWrite) == False:     
        #     os.mkdir(outputpathWrite)
        # writer.SetFileName(os.path.join(outputpathWrite , filename + '.dcm'))

        if createfolder == 'True':
            outputpathWrite = os.path.join(create_Folder_path, filename + '.dcm')
        else:
            outputpathWrite = os.path.join(OUTPUT_DIR, filename + '.dcm')

        writer.SetFileName(outputpathWrite)
        writer.Execute(image_slice)
        if un_precision:
            print(
                'voxel intensity in the converted image was limited to a 16-bit integer, so the converted image is not precise.')
        return outputpathWrite
    else:
        print('The software just supports 4 formats such as Nifti, Nrrd, Single and multi Dicom images.')


def Convert_nifti_nibabel_to_others(registered, header, Datatype,
                                    OUTPUT_DIR, filename, createfolder,
                                    ImagePositionPatients,
                                    ImageOrientationPatient, step, PixelSpacing, filepaths):
    if Datatype == 'Nifti':

        spacing = (abs(header.affine[0, 0]), abs(header.affine[1, 1]), abs(header.affine[2, 2]))
        if 0 in spacing:
            file_name__ = filepaths.split('\\')[-1]
            data_directory = filepaths.replace(file_name__, '')[:-1]
            img, spacing, origin, direction, directionStr = ReadDicomBySITK(data_directory)
        else:
            perm = (0, 1, 2)
            img = sitk.GetImageFromArray(np.transpose(registered.astype(np.float32), perm), isVector=False)

            # or1 = header.affine[0,3] if header.affine[0,0] > 0 else (header.affine[0,3] * (-1))
            # or2 = header.affine[1,3] if header.affine[1,1] > 0 else (header.affine[1,3] * (-1))
            # or3 = header.affine[2,3] if header.affine[2,2] > 0 else (header.affine[2,3] * (-1))
            # origin = (or1, or2, or3)

            # aff2axcodestupple = nib.aff2axcodes(header.affine)
            # # print(aff2axcodestupple)

            # a1 = +1.0 if 'L' in aff2axcodestupple else -1.0  
            # a2 = +1.0 if 'P' in aff2axcodestupple else -1.0  
            # a3 = +1.0 if 'S' in aff2axcodestupple else -1.0  

            # direction = (a1, 0.0, 0.0, 0.0, a2, 0.0, 0.0, 0.0, a3)
            file_name__ = filepaths.split('\\')[-1]
            data_directory = filepaths.replace(file_name__, '')[:-1]
            _, spacing, origin, direction, directionStr = ReadDicomBySITK(data_directory)

        img.SetOrigin(origin)
        img.SetSpacing(spacing)
        img.SetDirection(direction)

        if createfolder == 'True':
            filename2 = filename + '_Nifti'

            create_Folder_path = os.path.join(OUTPUT_DIR, filename2)
            if os.path.isdir(create_Folder_path) == False:
                os.mkdir(create_Folder_path)
            sitk.WriteImage(img, os.path.join(OUTPUT_DIR, filename2, filename + '.nii.gz'))
            return os.path.join(OUTPUT_DIR, filename2, filename + '.nii.gz')
        else:
            sitk.WriteImage(img, os.path.join(OUTPUT_DIR, filename + '.nii.gz'))
            return os.path.join(OUTPUT_DIR, filename + '.nii.gz')

    elif Datatype == 'Nrrd':

        cont = True
        spacing = (abs(header.affine[0, 0]), abs(header.affine[1, 1]), abs(header.affine[2, 2]))
        if 0 in spacing:
            file_name = filepaths.split('\\')[-1]
            data_directory = filepaths.replace(file_name, '')[:-1]
            img, spacing, origin, direction, directionStr = ReadDicomBySITK(data_directory)

            f1 = 'left' if 'L' in directionStr else 'right'
            f2 = 'posterior' if 'P' in directionStr else 'anterior'
            f3 = 'superior' if 'S' in directionStr else 'Inferior'
            spaceStr2 = f1 + '-' + f2 + '-' + f3

            if spaceStr2 == 'right-anterior-superior' or spaceStr2 == 'left-anterior-superior' or spaceStr2 == 'left-posterior-superior':
                _ = spaceStr2
            else:
                # spaceStr = 'right-anterior-superior'
                cont = False

        else:
            perm = (0, 1, 2)
            img = sitk.GetImageFromArray(np.transpose(registered.astype(np.float32), perm), isVector=False)

            file_name = filepaths.split('\\')[-1]
            data_directory = filepaths.replace(file_name, '')[:-1]
            _, spacing, origin, direction, directionStr = ReadDicomBySITK(data_directory)

            f1 = 'left' if 'L' in directionStr else 'right'
            f2 = 'posterior' if 'P' in directionStr else 'anterior'
            f3 = 'superior' if 'S' in directionStr else 'Inferior'
            spaceStr2 = f1 + '-' + f2 + '-' + f3

            if spaceStr2 == 'right-anterior-superior' or spaceStr2 == 'left-anterior-superior' or spaceStr2 == 'left-posterior-superior':
                _ = spaceStr2
            else:
                # spaceStr = 'right-anterior-superior'
                cont = False

        img.SetOrigin(origin)
        img.SetSpacing(spacing)
        img.SetDirection(direction)

        # print (spaceStr)
        if cont:
            if createfolder == 'True':
                filename2 = filename + '_Nrrd'

                create_Folder_path = os.path.join(OUTPUT_DIR, filename2)
                if os.path.isdir(create_Folder_path) == False:
                    os.mkdir(create_Folder_path)
                sitk.WriteImage(img, os.path.join(OUTPUT_DIR, filename2, filename + '.nrrd'))
                return os.path.join(OUTPUT_DIR, filename2, filename + '.nrrd')
            else:
                sitk.WriteImage(img, os.path.join(OUTPUT_DIR, filename + '.nrrd'))
                return os.path.join(OUTPUT_DIR, filename + '.nrrd')
        else:
            print('Nrrd Format doesn''t work with', spaceStr2, 'direction')

        # cont = True
        # spacing = (abs(header.affine[0,0]),abs(header.affine[1,1]),abs(header.affine[2,2]))
        # if 0 in spacing:
        #     file_name = filepaths.split('\\')[-1]
        #     data_directory = filepaths.replace(file_name , '')[:-1]
        #     img,spacing , origin,direction,directionStr = ReadDicomBySITK(data_directory)

        #     f1 = 'left' if 'L' in directionStr else 'right'
        #     f2 = 'posterior' if 'P' in directionStr  else 'anterior' 
        #     f3 = 'superior' if 'S' in directionStr  else 'Inferior'
        #     spaceStr2 = f1+'-'+f2+'-'+f3           

        #     if spaceStr2 == 'right-anterior-superior' or spaceStr2 == 'left-anterior-superior' or spaceStr2 == 'left-posterior-superior':
        #         spaceStr = spaceStr2
        #     else:
        #         # spaceStr = 'right-anterior-superior'
        #         cont = False

        #     perm = (2, 1, 0)
        #     data = np.transpose(registered.astype(np.float32), perm)

        # else:   
        #     perm = (2, 1, 0)
        #     data = np.transpose(registered.astype(np.float32), perm)

        #     # or1 = header.affine[0,3] if header.affine[0,0] > 0 else (header.affine[0,3] * (-1))
        #     # or2 = header.affine[1,3] if header.affine[1,1] > 0 else (header.affine[1,3] * (-1))
        #     # or3 = header.affine[2,3] if header.affine[2,2] > 0 else (header.affine[2,3] * (-1))
        #     # origin = (or1, or2, or3)

        #     # a1 = +1.0 if nib.aff2axcodes(header.affine)[0] == 'L' else -1.0  
        #     # a2 = +1.0 if nib.aff2axcodes(header.affine)[1] == 'P' else -1.0  
        #     # a3 = +1.0 if nib.aff2axcodes(header.affine)[2] == 'S' else -1.0  
        #     # direction = (a1, 0.0, 0.0, 0.0, a2, 0.0, 0.0, 0.0, a3)

        #     # f1 = 'left' if a1 == 1 else 'right'
        #     # f2 = 'posterior' if a2 == 1  else 'anterior' 
        #     # f3 = 'superior' if a3 == 1  else 'Inferior'

        #     # spaceStr2 = f1+'-'+f2+'-'+f3
        #     file_name = filepaths.split('\\')[-1]
        #     data_directory = filepaths.replace(file_name , '')[:-1]
        #     newimg,spacing , origin,direction,directionStr = ReadDicomBySITK(data_directory)           

        #     f1 = 'left' if 'L' in directionStr else 'right'
        #     f2 = 'posterior' if 'P' in directionStr  else 'anterior' 
        #     f3 = 'superior' if 'S' in directionStr  else 'Inferior'
        #     spaceStr2 = f1+'-'+f2+'-'+f3           

        #     if spaceStr2 == 'right-anterior-superior' or spaceStr2 == 'left-anterior-superior' or spaceStr2 == 'left-posterior-superior':
        #         spaceStr = spaceStr2
        #     else:
        #         # spaceStr = 'right-anterior-superior'
        #         cont = False

        # # print (spaceStr)
        # if cont:
        #     fileheader = { 
        #         'dimension' : len(registered.shape) ,
        #         'sizes' : np.array([registered.shape[2] ,registered.shape[1] ,registered.shape[0]]),
        #         'space': spaceStr,    
        #         # 'spacings': [spacing[0],spacing[1],spacing[2]],
        #         'space directions': np.array([[spacing[0],0,0],[0,spacing[1],0],[0,0,spacing[2]]]), 
        #         'space origin': np.array([origin[0] ,origin[1] ,origin[2]])
        #     }

        #     if createfolder == 'True':
        #         filename2 =  filename + '_NRRD'

        #         create_Folder_path = os.path.join(OUTPUT_DIR,filename2)
        #         if os.path.isdir(create_Folder_path) == False:
        #             os.mkdir(create_Folder_path)
        #         nrrd.write(os.path.join(OUTPUT_DIR,filename2,filename+'.nrrd' ), data, fileheader)
        #     else:
        #         nrrd.write(os.path.join(OUTPUT_DIR,filename+'.nrrd' ), data, fileheader)
        # else:
        #     print('Nrrd Format doesn''t work with',spaceStr2,'direction')

    elif Datatype == 'MDicom':

        un_precision = False
        registered_max = np.max(registered)
        if registered_max > pow(2, 16):
            un_precision = True

        registered_min = np.min(registered)

        if registered_min < 0:
            registered = registered.astype(np.int16)  # best ct
        else:
            registered = registered.astype(np.uint16)  # best pet

        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()

        modification_time = time.strftime("%H%M%S")
        modification_date = time.strftime("%Y%m%d")

        spaceiing = (abs(header.affine[0, 0]), abs(header.affine[1, 1]), abs(header.affine[2, 2]))
        if 0 in spaceiing:
            file_name = filepaths.split('\\')[-1]
            data_directory = filepaths.replace(file_name, '')[:-1]
            new_img, spaceiing, origin, direction, directionStr = ReadDicomBySITK(data_directory)

            # img_Arr_tr = sitk.GetArrayFromImage(img)
            # perm = (2, 1, 0)
            # img_Arr_tr = np.transpose(img_Arr.astype(np.float32),perm)
            # print(img_Arr_tr.shape)
            # print(registered.shape)
            # new_img = sitk.GetImageFromArray(img_Arr_tr)
            # new_img = sitk.GetImageFromArray(registered)

        else:
            # perm = (0, 1, 2)
            # img = sitk.GetImageFromArray(np.transpose(registered.astype(np.float32), perm), isVector=False)
            file_name = filepaths.split('\\')[-1]
            data_directory = filepaths.replace(file_name, '')[:-1]
            img, spaceiing, origin, direction, directionStr = ReadDicomBySITK(data_directory)
            new_img = sitk.GetImageFromArray(registered)

            # or1 = header.affine[0,3] if header.affine[0,0] > 0 else (header.affine[0,3] * (-1))
            # or2 = header.affine[1,3] if header.affine[1,1] > 0 else (header.affine[1,3] * (-1))
            # or3 = header.affine[2,3] if header.affine[2,2] > 0 else (header.affine[2,3] * (-1))
            # origin = (or1, or2, or3)

            # aff2axcodestupple = nib.aff2axcodes(header.affine)
            # # print(aff2axcodestupple)

            # a1 = +1.0 if 'L' in aff2axcodestupple else -1.0  
            # a2 = +1.0 if 'P' in aff2axcodestupple else -1.0  
            # a3 = +1.0 if 'S' in aff2axcodestupple else -1.0  

            # direction = (a1, 0.0, 0.0, 0.0, a2, 0.0, 0.0, 0.0, a3)

            # a1 = +1.0 if nib.aff2axcodes(header.affine)[0] == 'L' else -1.0  
            # a2 = +1.0 if nib.aff2axcodes(header.affine)[1] == 'P' else -1.0  
            # a3 = +1.0 if nib.aff2axcodes(header.affine)[2] == 'S' else -1.0  
            # direction = (a1, 0.0, 0.0, 0.0, a2, 0.0, 0.0, 0.0, a3)

        rand_number_Mdicom = randint(1, 10000) * 19

        series_tag_values = [
            ("0008|0031", modification_time),  # Series Time
            ("0008|0021", modification_date),  # Series Date
            ("0008|0008", "DERIVED\\SECONDARY"),  # Image Type
            ("0020|000e", "1.2.826.0.1.3680043.2.1125." + str(rand_number_Mdicom) + "."
             + modification_date + ".1" + modification_time),  # Series Instance UID
            # ("0020|0037", '\\'.join(map(str, (direction[0], direction[1],direction[2],
            #                                   direction[3],direction[4],
            #                                   direction[5])))),# Image Orientation
            ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],  # Image Orientation (Patient)
                                              direction[1], direction[4], direction[7])))),
            ("0020|0032", '\\'.join(map(str, (origin[0], origin[1], origin[2])))),
            ("0028|0030", '\\'.join(map(str, (spaceiing[0], spaceiing[1])))),
            ("0018|0050", str(spaceiing[2])),
            ("0008|103e", filename)  # Series Description
        ]

        new_img.SetDirection(direction)
        new_img.SetSpacing(spaceiing)
        new_img.SetOrigin(origin)

        if createfolder == 'True':
            filename2 = filename + '_MDicom'
            create_Folder_path = os.path.join(OUTPUT_DIR, filename2)
            if os.path.isdir(create_Folder_path) == False:
                os.mkdir(create_Folder_path)

        for i in range(new_img.GetDepth()):
            image_slice = new_img[:, :, i]
            list(map(lambda tag_value: image_slice.SetMetaData(tag_value[0], tag_value[1]), series_tag_values))
            for tag, value in series_tag_values:
                image_slice.SetMetaData(tag, value)
            image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))  # Instance Creation Date
            image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))  # Instance Creation Time
            image_slice.SetMetaData("0008|0060", "CT")  # set the type to CT so the thickness is carried over
            image_slice.SetMetaData("0020|0032", '\\'.join(
                map(str, new_img.TransformIndexToPhysicalPoint((0, 0, i)))))  # Image Position (Patient)
            image_slice.SetMetaData("0020,0013", str(i))  # Instance Number

            # if createfolder == 'True':
            #     outputpathWrite = os.path.join(create_Folder_path, filename)
            # else:
            #     outputpathWrite = os.path.join(OUTPUT_DIR, filename)

            # if os.path.isdir(outputpathWrite) == False:     
            #     os.mkdir(outputpathWrite)
            # writer.SetFileName(os.path.join(outputpathWrite , filename + '_' + str(i) + '.dcm'))
            # writer.Execute(image_slice)

            if createfolder == 'True':
                outputpathWrite = os.path.join(create_Folder_path, filename + '_' + str(i) + '.dcm')
            else:
                outputpathWrite = os.path.join(OUTPUT_DIR, filename + '_' + str(i) + '.dcm')

            writer.SetFileName(outputpathWrite)
            writer.Execute(image_slice)
        if un_precision:
            print(
                'voxel intensity in the converted image was limited to a 16-bit integer, so the converted image is not precise.')
        return outputpathWrite
    elif Datatype == 'SDicom':

        un_precision = False
        registered_max = np.max(registered)
        if registered_max > pow(2, 16):
            un_precision = True

        registered_min = np.min(registered)

        if registered_min < 0:
            registered = registered.astype(np.int16)  # best ct
            # registered = np.rint(registered)
        else:
            registered = registered.astype(np.uint16)  # best pet
            # registered = np.rint(registered)

        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()

        modification_time = time.strftime("%H%M%S")
        modification_date = time.strftime("%Y%m%d")

        spaceiing = (abs(header.affine[0, 0]), abs(header.affine[1, 1]), abs(header.affine[2, 2]))
        if 0 in spaceiing:
            file_name = filepaths.split('\\')[-1]
            data_directory = filepaths.replace(file_name, '')[:-1]
            new_img, spaceiing, origin, direction, directionStr = ReadDicomBySITK(data_directory)
        else:
            # perm = (0, 1, 2)
            # img = sitk.GetImageFromArray(np.transpose(registered.astype(np.float32), perm), isVector=False)
            file_name = filepaths.split('\\')[-1]
            data_directory = filepaths.replace(file_name, '')[:-1]
            img, spaceiing, origin, direction, directionStr = ReadDicomBySITK(data_directory)

            new_img = sitk.GetImageFromArray(registered)

            # or1 = header.affine[0,3] if header.affine[0,0] > 0 else (header.affine[0,3] * (-1))
            # or2 = header.affine[1,3] if header.affine[1,1] > 0 else (header.affine[1,3] * (-1))
            # or3 = header.affine[2,3] if header.affine[2,2] > 0 else (header.affine[2,3] * (-1))
            # origin = (or1, or2, or3)

            # aff2axcodestupple = nib.aff2axcodes(header.affine)
            # # print(aff2axcodestupple)

            # a1 = +1.0 if 'L' in aff2axcodestupple else -1.0  
            # a2 = +1.0 if 'P' in aff2axcodestupple else -1.0  
            # a3 = +1.0 if 'S' in aff2axcodestupple else -1.0  

            # direction = (a1, 0.0, 0.0, 0.0, a2, 0.0, 0.0, 0.0, a3)

            # a1 = +1.0 if nib.aff2axcodes(header.affine)[0] == 'L' else -1.0  
            # a2 = +1.0 if nib.aff2axcodes(header.affine)[1] == 'P' else -1.0  
            # a3 = +1.0 if nib.aff2axcodes(header.affine)[2] == 'S' else -1.0  
            # direction = (a1, 0.0, 0.0, 0.0, a2, 0.0, 0.0, 0.0, a3)

        rand_number_Sdicom = randint(200000, 210000) * 13

        series_tag_values = [
            ("0008|0031", modification_time),  # Series Time
            ("0008|0021", modification_date),  # Series Date
            ("0008|0008", "DERIVED\\SECONDARY"),  # Image Type
            ("0020|000e", "1.2.826.0.1.3680043.2.1125." + str(rand_number_Sdicom) + "."
             + modification_date + ".1" + modification_time),  # Series Instance UID
            # ("0020|0037", '\\'.join(map(str, (direction[0], direction[1],direction[2],
            #                                   direction[3],direction[4],
            #                                   direction[5])))),# Image Orientation
            ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],  # Image Orientation (Patient)
                                              direction[1], direction[4], direction[7])))),
            ("0020|0032", '\\'.join(map(str, (origin[0], origin[1], origin[2])))),
            ("0028|0030", '\\'.join(map(str, (spaceiing[0], spaceiing[1])))),
            ("0018|0050", str(spaceiing[2])),
            ("0008|103e", filename)  # Series Description
        ]

        new_img.SetDirection(direction)
        new_img.SetSpacing(spaceiing)
        new_img.SetOrigin(origin)

        if createfolder == 'True':
            filename2 = filename + '_SDicom'

            create_Folder_path = os.path.join(OUTPUT_DIR, filename2)
            if os.path.isdir(create_Folder_path) == False:
                os.mkdir(create_Folder_path)

        image_slice = new_img[:, :, :]
        list(map(lambda tag_value: image_slice.SetMetaData(tag_value[0], tag_value[1]), series_tag_values))
        for tag, value in series_tag_values:
            image_slice.SetMetaData(tag, value)
        image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))  # Instance Creation Date
        image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))  # Instance Creation Time
        image_slice.SetMetaData("0008|0060", "CT")  # set the type to CT so the thickness is carried over
        image_slice.SetMetaData("0020|0032", '\\'.join(
            map(str, new_img.TransformIndexToPhysicalPoint((0, 0, 0)))))  # Image Position (Patient)
        # image_slice.SetMetaData("0020,0013", str(i)) # Instance Number

        # if createfolder == 'True':
        #     outputpathWrite = os.path.join(create_Folder_path, filename)
        # else:
        #     outputpathWrite = os.path.join(OUTPUT_DIR, filename)

        # if os.path.isdir(outputpathWrite) == False:     
        #     os.mkdir(outputpathWrite)
        # writer.SetFileName(os.path.join(outputpathWrite , filename +'.dcm'))
        # writer.Execute(image_slice)

        if createfolder == 'True':
            outputpathWrite = os.path.join(create_Folder_path, filename + '.dcm')
        else:
            outputpathWrite = os.path.join(OUTPUT_DIR, filename + '.dcm')

        writer.SetFileName(outputpathWrite)
        writer.Execute(image_slice)

        if un_precision:
            print(
                'voxel intensity in the converted image was limited to a 16-bit integer, so the converted image is not precise.')
        return outputpathWrite
    else:
        print('The software just supports 4 formats such as Nifti, Nrrd, Single and multi Dicom images.')


def Convert_nrrd_to_others(registered, header, Datatype, OUTPUT_DIR, filename, createfolder):
    if Datatype == 'Nifti':
        perm = (0, 1, 2)
        img = sitk.GetImageFromArray(np.transpose(registered.astype(np.float32), perm), isVector=False)
        spacing = (header['space directions'][0, 0], header['space directions'][1, 1], header['space directions'][2, 2])
        origin = (header['space origin'][0], header['space origin'][1], header['space origin'][2])
        a1 = +1.0 if header['space'].split('-')[0] == 'left' else -1.0
        a2 = +1.0 if header['space'].split('-')[1] == 'posterior' else -1.0
        a3 = +1.0 if header['space'].split('-')[2] == 'superior' else -1.0
        direction = (a1, 0.0, 0.0, 0.0, a2, 0.0, 0.0, 0.0, a3)

        img.SetOrigin(origin)
        img.SetSpacing(spacing)
        img.SetDirection(direction)
        if createfolder == 'True':
            filename2 = filename + '_Nifti'

            create_Folder_path = os.path.join(OUTPUT_DIR, filename2)
            if os.path.isdir(create_Folder_path) == False:
                os.mkdir(create_Folder_path)
            sitk.WriteImage(img, os.path.join(OUTPUT_DIR, filename2, filename + '.nii.gz'))
            return os.path.join(OUTPUT_DIR, filename2, filename + '.nii.gz')
        else:
            sitk.WriteImage(img, os.path.join(OUTPUT_DIR, filename + '.nii.gz'))
            return os.path.join(OUTPUT_DIR, filename + '.nii.gz')


    elif Datatype == 'Nrrd':

        perm = (2, 1, 0)
        data = np.transpose(registered.astype(np.float32), perm)

        if createfolder == 'True':
            filename2 = filename + '_Nrrd'

            create_Folder_path = os.path.join(OUTPUT_DIR, filename2)
            if os.path.isdir(create_Folder_path) == False:
                os.mkdir(create_Folder_path)
            nrrd.write(os.path.join(OUTPUT_DIR, filename2, filename + '.nrrd'), data, header)
            return os.path.join(OUTPUT_DIR, filename2, filename + '.nrrd')
        else:
            nrrd.write(os.path.join(OUTPUT_DIR, filename + '.nrrd'), data, header)
            return os.path.join(OUTPUT_DIR, filename + '.nrrd')


    elif Datatype == 'MDicom':

        un_precision = False
        registered_max = np.max(registered)
        if registered_max > pow(2, 16):
            un_precision = True

        registered_min = np.min(registered)
        if registered_min < 0:
            registered = registered.astype(np.int16)  # best ct
        else:
            registered = registered.astype(np.uint16)  # best pet

        new_img = sitk.GetImageFromArray(registered)

        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()

        modification_time = time.strftime("%H%M%S")
        modification_date = time.strftime("%Y%m%d")

        spaceiing = (
            header['space directions'][0, 0], header['space directions'][1, 1], header['space directions'][2, 2])
        origin = (header['space origin'][0], header['space origin'][1], header['space origin'][2])

        a1 = +1.0 if header['space'].split('-')[0] == 'left' else -1.0
        a2 = +1.0 if header['space'].split('-')[1] == 'posterior' else -1.0
        a3 = +1.0 if header['space'].split('-')[2] == 'superior' else -1.0
        direction = (a1, 0.0, 0.0, 0.0, a2, 0.0, 0.0, 0.0, a3)

        rand_number_Mdicom = randint(1, 10000) * 19

        series_tag_values = [
            ("0008|0031", modification_time),  # Series Time
            ("0008|0021", modification_date),  # Series Date
            ("0008|0008", "DERIVED\\SECONDARY"),  # Image Type
            ("0020|000e", "1.2.826.0.1.3680043.2.1125." + str(rand_number_Mdicom) + "."
             + modification_date + ".1" + modification_time),  # Series Instance UID
            # ("0020|0037", '\\'.join(map(str, (direction[0], direction[1],direction[2],
            #                                   direction[3],direction[4],
            #                                   direction[5])))),# Image Orientation
            ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],  # Image Orientation (Patient)
                                              direction[1], direction[4], direction[7])))),
            ("0020|0032", '\\'.join(map(str, (origin[0], origin[1], origin[2])))),
            ("0028|0030", '\\'.join(map(str, (spaceiing[0], spaceiing[1])))),
            ("0018|0050", str(spaceiing[2])),
            ("0008|103e", filename)  # Series Description
        ]

        new_img.SetDirection(direction)
        new_img.SetSpacing(spaceiing)
        new_img.SetOrigin(origin)

        if createfolder == 'True':
            filename2 = filename + '_MDicom'
            create_Folder_path = os.path.join(OUTPUT_DIR, filename2)
            if os.path.isdir(create_Folder_path) == False:
                os.mkdir(create_Folder_path)

        for i in range(new_img.GetDepth()):
            image_slice = new_img[:, :, i]
            list(map(lambda tag_value: image_slice.SetMetaData(tag_value[0], tag_value[1]), series_tag_values))
            for tag, value in series_tag_values:
                image_slice.SetMetaData(tag, value)
            image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))  # Instance Creation Date
            image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))  # Instance Creation Time
            image_slice.SetMetaData("0008|0060", "CT")  # set the type to CT so the thickness is carried over
            image_slice.SetMetaData("0020|0032", '\\'.join(
                map(str, new_img.TransformIndexToPhysicalPoint((0, 0, i)))))  # Image Position (Patient)
            image_slice.SetMetaData("0020,0013", str(i))  # Instance Number
            outputpathWrite = os.path.join(OUTPUT_DIR, filename)

            # if createfolder == 'True':
            #     outputpathWrite = os.path.join(create_Folder_path, filename)
            # else:
            #     outputpathWrite = os.path.join(OUTPUT_DIR, filename)

            # if os.path.isdir(outputpathWrite) == False:     
            #     os.mkdir(outputpathWrite)
            # writer.SetFileName(os.path.join(outputpathWrite , filename + '_' + str(i) + '.dcm'))
            # writer.Execute(image_slice)

            if createfolder == 'True':
                outputpathWrite = os.path.join(create_Folder_path, filename + '_' + str(i) + '.dcm')
            else:
                outputpathWrite = os.path.join(OUTPUT_DIR, filename + '_' + str(i) + '.dcm')

            writer.SetFileName(outputpathWrite)
            writer.Execute(image_slice)

        if un_precision:
            print(
                'voxel intensity in the converted image was limited to a 16-bit integer, so the converted image is not precise.')
        return outputpathWrite
    elif Datatype == 'SDicom':

        un_precision = False
        registered_max = np.max(registered)
        if registered_max > pow(2, 16):
            un_precision = True

        registered_min = np.min(registered)
        if registered_min < 0:
            registered = registered.astype(np.int16)  # best ct
        else:
            registered = registered.astype(np.uint16)  # best pet

        new_img = sitk.GetImageFromArray(registered)

        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()

        modification_time = time.strftime("%H%M%S")
        modification_date = time.strftime("%Y%m%d")

        spaceiing = (
            header['space directions'][0, 0], header['space directions'][1, 1], header['space directions'][2, 2])
        origin = (header['space origin'][0], header['space origin'][1], header['space origin'][2])

        a1 = +1.0 if header['space'].split('-')[0] == 'left' else -1.0
        a2 = +1.0 if header['space'].split('-')[1] == 'posterior' else -1.0
        a3 = +1.0 if header['space'].split('-')[2] == 'superior' else -1.0
        direction = (a1, 0.0, 0.0, 0.0, a2, 0.0, 0.0, 0.0, a3)

        rand_number_Sdicom = randint(200000, 210000) * 13

        series_tag_values = [
            ("0008|0031", modification_time),  # Series Time
            ("0008|0021", modification_date),  # Series Date
            ("0008|0008", "DERIVED\\SECONDARY"),  # Image Type
            ("0020|000e", "1.2.826.0.1.3680043.2.1125." + str(rand_number_Sdicom) + "."
             + modification_date + ".1" + modification_time),  # Series Instance UID
            # ("0020|0037", '\\'.join(map(str, (direction[0], direction[1],direction[2],
            #                                   direction[3],direction[4],
            #                                   direction[5])))),# Image Orientation
            ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],  # Image Orientation (Patient)
                                              direction[1], direction[4], direction[7])))),
            ("0020|0032", '\\'.join(map(str, (origin[0], origin[1], origin[2])))),
            ("0028|0030", '\\'.join(map(str, (spaceiing[0], spaceiing[1])))),
            ("0018|0050", str(spaceiing[2])),
            ("0008|103e", filename)  # Series Description
        ]

        new_img.SetDirection(direction)
        new_img.SetSpacing(spaceiing)
        new_img.SetOrigin(origin)

        if createfolder == 'True':
            filename2 = filename + '_SDicom'

            create_Folder_path = os.path.join(OUTPUT_DIR, filename2)
            if os.path.isdir(create_Folder_path) == False:
                os.mkdir(create_Folder_path)

        image_slice = new_img[:, :, :]
        list(map(lambda tag_value: image_slice.SetMetaData(tag_value[0], tag_value[1]), series_tag_values))
        for tag, value in series_tag_values:
            image_slice.SetMetaData(tag, value)
        image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))  # Instance Creation Date
        image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))  # Instance Creation Time
        image_slice.SetMetaData("0008|0060", "CT")  # set the type to CT so the thickness is carried over
        image_slice.SetMetaData("0020|0032", '\\'.join(
            map(str, new_img.TransformIndexToPhysicalPoint((0, 0, 0)))))  # Image Position (Patient)
        # image_slice.SetMetaData("0020,0013", str(i)) # Instance Number
        outputpathWrite = os.path.join(OUTPUT_DIR, filename)

        # if createfolder == 'True':
        #     outputpathWrite = os.path.join(create_Folder_path, filename)
        # else:
        #     outputpathWrite = os.path.join(OUTPUT_DIR, filename)

        # if os.path.isdir(outputpathWrite) == False:     
        #     os.mkdir(outputpathWrite)
        # writer.SetFileName(os.path.join(outputpathWrite , filename +'.dcm'))
        # writer.Execute(image_slice)  

        if createfolder == 'True':
            outputpathWrite = os.path.join(create_Folder_path, filename + '.dcm')
        else:
            outputpathWrite = os.path.join(OUTPUT_DIR, filename + '.dcm')

        writer.SetFileName(outputpathWrite)
        writer.Execute(image_slice)
        if un_precision:
            print(
                'voxel intensity in the converted image was limited to a 16-bit integer, so the converted image is not precise.')
        return outputpathWrite
    else:
        print('The software just supports 4 formats such as Nifti, Nrrd, Single and multi Dicom images.')


def Convert_dicom_to_others(registered, header, Datatype, OUTPUT_DIR, filename, createfolder):
    if len(registered.shape) == 3:
        if Datatype == 'Nifti':
            perm = (0, 1, 2)
            img = sitk.GetImageFromArray(np.transpose(registered.astype(np.float32), perm), isVector=False)

            img.SetOrigin(header.GetOrigin())
            img.SetSpacing(header.GetSpacing())
            img.SetDirection(header.GetDirection())

            if createfolder == 'True':
                filename2 = filename + '_Nifti'

                create_Folder_path = os.path.join(OUTPUT_DIR, filename2)
                if os.path.isdir(create_Folder_path) == False:
                    os.mkdir(create_Folder_path)
                sitk.WriteImage(img, os.path.join(OUTPUT_DIR, filename2, filename + '.nii.gz'))
                return os.path.join(OUTPUT_DIR, filename2, filename + '.nii.gz')
            else:
                sitk.WriteImage(img, os.path.join(OUTPUT_DIR, filename + '.nii.gz'))
                return os.path.join(OUTPUT_DIR, filename + '.nii.gz')

        elif Datatype == 'Nrrd':

            perm = (0, 1, 2)
            img = sitk.GetImageFromArray(np.transpose(registered.astype(np.float32), perm), isVector=False)

            img.SetOrigin(header.GetOrigin())
            img.SetSpacing(header.GetSpacing())
            img.SetDirection(header.GetDirection())

            orient = sitk.DICOMOrientImageFilter()
            DirectionStr = orient.GetOrientationFromDirectionCosines(header.GetDirection())

            cont = True
            f1 = 'left' if 'L' in DirectionStr else 'right'
            f2 = 'posterior' if 'P' in DirectionStr else 'anterior'
            f3 = 'superior' if 'S' in DirectionStr else 'Inferior'
            spaceStr2 = f1 + '-' + f2 + '-' + f3

            if spaceStr2 == 'right-anterior-superior' or spaceStr2 == 'left-anterior-superior' or spaceStr2 == 'left-posterior-superior':
                _ = spaceStr2
            else:
                # spaceStr = 'right-anterior-superior'
                cont = False

            if cont:
                if createfolder == 'True':
                    filename2 = filename + '_Nrrd'

                    create_Folder_path = os.path.join(OUTPUT_DIR, filename2)
                    if os.path.isdir(create_Folder_path) == False:
                        os.mkdir(create_Folder_path)
                    sitk.WriteImage(img, os.path.join(OUTPUT_DIR, filename2, filename + '.nrrd'))
                    return os.path.join(OUTPUT_DIR, filename2, filename + '.nrrd')
                else:
                    sitk.WriteImage(img, os.path.join(OUTPUT_DIR, filename + '.nrrd'))
                    return os.path.join(OUTPUT_DIR, filename + '.nrrd')

            else:
                print('Nrrd Format doesn''t work with', spaceStr2, 'direction')

            # perm = (2, 1, 0)
            # data = np.transpose(registered.astype(np.float32), perm)

            # orient = sitk.DICOMOrientImageFilter()
            # DirectionStr = orient.GetOrientationFromDirectionCosines(header.GetDirection())

            # # a1 = header.GetDirection()[0]
            # # # a2 = header.GetDirection()[1]
            # # # a3 = header.GetDirection()[2]
            # # # a4 = header.GetDirection()[3]
            # # a5 = header.GetDirection()[4]
            # # # a6 = header.GetDirection()[5]
            # # # a7 = header.GetDirection()[6]
            # # # a8 = header.GetDirection()[7]
            # # a9 = header.GetDirection()[8]

            # # f1 = 'left' if a1 == 1 else 'right'
            # # f2 = 'posterior' if a5 == 1  else 'anterior' 
            # # f3 = 'superior' if a9 == 1  else 'Inferior'
            # # spaceStr = f1+'-'+f2+'-'+f3

            # # f1 = 'left' if a1 > 0 else 'right'
            # # f2 = 'posterior' if a5 > 0  else 'anterior' 
            # # f3 = 'superior' if a9 > 0  else 'Inferior'
            # # spaceStr = f1+'-'+f2+'-'+f3

            # cont = True
            # f1 = 'left' if 'L' in DirectionStr else 'right'
            # f2 = 'posterior' if 'P' in DirectionStr  else 'anterior' 
            # f3 = 'superior' if 'S' in DirectionStr  else 'Inferior'
            # spaceStr2 = f1+'-'+f2+'-'+f3           

            # if spaceStr2 == 'right-anterior-superior' or spaceStr2 == 'left-anterior-superior' or spaceStr2 == 'left-posterior-superior':
            #     spaceStr = spaceStr2
            # else:
            #     # spaceStr = 'right-anterior-superior'
            #     cont = False
            # if cont:
            #     fileheader = { 
            #         'dimension' : header.GetDimension(),
            #         'sizes' : np.array([header.GetWidth() ,header.GetHeight() ,header.GetDepth()]),
            #         'space': spaceStr,  
            #         # 'spacings': [header.GetSpacing()[0], header.GetSpacing()[1], header.GetSpacing()[2]],
            #         # 'thicknesses': [header.GetSpacing()[0], header.GetSpacing()[1], header.GetSpacing()[2]],
            #         # 'space directions': np.array([[a1,a2,a3],[a4,a5,a6],[a7,a8,a9]]), 
            #         # 'space directions': np.array([[header.GetSpacing()[0],a2,a3],[a4,header.GetSpacing()[1],a6],[a7,a8,header.GetSpacing()[2]]]), 
            #         'space directions': np.array([[header.GetSpacing()[0],0,0],[0,header.GetSpacing()[1],0],[0,0,header.GetSpacing()[2]]]), 
            #         'space origin': np.array([header.GetOrigin()[0] ,header.GetOrigin()[1] ,header.GetOrigin()[2]])
            #     }
            #     if createfolder == 'True':
            #         filename2 =  filename + '_NRRD'

            #         create_Folder_path = os.path.join(OUTPUT_DIR,filename2)
            #         if os.path.isdir(create_Folder_path) == False:
            #             os.mkdir(create_Folder_path)
            #         nrrd.write(os.path.join(OUTPUT_DIR,filename2,filename+'.nrrd' ), data, fileheader)
            #     else:
            #         nrrd.write(os.path.join(OUTPUT_DIR,filename+'.nrrd' ), data, fileheader)
            # else:
            #     print('Nrrd Format doesn''t work with',spaceStr2,'direction')

        elif Datatype == 'SDicom':

            un_precision = False
            registered_max = np.max(registered)
            if registered_max > pow(2, 16):
                un_precision = True
            registered_min = np.min(registered)
            if registered_min < 0:
                registered = registered.astype(np.int16)  # best ct
            else:
                registered = registered.astype(np.uint16)  # best pet

            new_img = sitk.GetImageFromArray(registered)

            writer = sitk.ImageFileWriter()
            writer.KeepOriginalImageUIDOn()

            modification_time = time.strftime("%H%M%S")
            modification_date = time.strftime("%Y%m%d")

            direction = header.GetDirection()
            origin = header.GetOrigin()
            spaceiing = header.GetSpacing()

            rand_number_Sdicom = randint(200000, 210000) * 13

            series_tag_values = [
                ("0008|0031", modification_time),
                ("0008|0021", modification_date),
                ("0008|0008", "DERIVED\\SECONDARY"),
                ("0020|000e", "1.2.826.0.1.3680043.2.1125." + str(rand_number_Sdicom) + "."
                 + modification_date + ".1" + modification_time),
                # ("0020|0037", '\\'.join(map(str, (direction[0], direction[1],direction[2],
                #                                   direction[3],direction[4],
                #                                   direction[5])))),# Image Orientation
                ("0020|0037",
                 '\\'.join(map(str, (direction[0], direction[3], direction[6],  # Image Orientation (Patient)
                                     direction[1], direction[4], direction[7])))),
                ("0020|0032", '\\'.join(map(str, (origin[0], origin[1], origin[2])))),
                ("0028|0030", '\\'.join(map(str, (spaceiing[0], spaceiing[1])))),
                ("0018|0050", str(spaceiing[2])),
                ("0008|103e", filename)  # Series Description
            ]

            new_img.SetDirection(header.GetDirection())
            new_img.SetSpacing(header.GetSpacing())
            new_img.SetOrigin(header.GetOrigin())

            if createfolder == 'True':
                filename2 = filename + '_SDicom'

                create_Folder_path = os.path.join(OUTPUT_DIR, filename2)
                if os.path.isdir(create_Folder_path) == False:
                    os.mkdir(create_Folder_path)

            image_slice = new_img[:, :, :]
            list(map(lambda tag_value: image_slice.SetMetaData(tag_value[0], tag_value[1]), series_tag_values))
            for tag, value in series_tag_values:
                image_slice.SetMetaData(tag, value)
            image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))  # Instance Creation Date
            image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))  # Instance Creation Time
            image_slice.SetMetaData("0008|0060", "CT")  # set the type to CT so the thickness is carried over
            image_slice.SetMetaData("0020|0032", '\\'.join(
                map(str, new_img.TransformIndexToPhysicalPoint((0, 0, 0)))))  # Image Position (Patient)
            # image_slice.SetMetaData("0020,0013", str(i)) # Instance Number

            # if createfolder == 'True':
            #     outputpathWrite = os.path.join(create_Folder_path, filename)
            # else:
            #     outputpathWrite = os.path.join(OUTPUT_DIR, filename)

            # if os.path.isdir(outputpathWrite) == False:     
            #     os.mkdir(outputpathWrite)
            # writer.SetFileName(os.path.join(outputpathWrite , filename + '.dcm'))

            if createfolder == 'True':
                outputpathWrite = os.path.join(create_Folder_path, filename + '.dcm')
            else:
                outputpathWrite = os.path.join(OUTPUT_DIR, filename + '.dcm')

            writer.SetFileName(outputpathWrite)
            writer.Execute(image_slice)
            if un_precision:
                print(
                    'voxel intensity in the converted image was limited to a 16-bit integer, so the converted image is not precise.')
            return outputpathWrite

        elif Datatype == 'MDicom':
            print('The software just supports 4 formats such as Nifti, Nrrd, Single and multi Dicom images.')
        else:
            print('The software just supports 4 formats such as Nifti, Nrrd, Single and multi Dicom images.')

    elif len(registered.shape) == 2:
        if Datatype == 'Nifti':
            perm = (0, 1)
            img = sitk.GetImageFromArray(np.transpose(registered.astype(np.float32), perm), isVector=False)

            img.SetOrigin(header.GetOrigin())
            img.SetSpacing(header.GetSpacing())
            img.SetDirection(header.GetDirection())

            if createfolder == 'True':
                filename2 = filename + '_Nifti'

                create_Folder_path = os.path.join(OUTPUT_DIR, filename2)
                if os.path.isdir(create_Folder_path) == False:
                    os.mkdir(create_Folder_path)
                sitk.WriteImage(img, os.path.join(OUTPUT_DIR, filename2, filename + '.nii.gz'))
                return os.path.join(OUTPUT_DIR, filename2, filename + '.nii.gz')
            else:
                sitk.WriteImage(img, os.path.join(OUTPUT_DIR, filename + '.nii.gz'))
                return os.path.join(OUTPUT_DIR, filename + '.nii.gz')

        elif Datatype == 'Nrrd':

            perm = (0, 1)
            img = sitk.GetImageFromArray(np.transpose(registered.astype(np.float32), perm), isVector=False)

            img.SetOrigin(header.GetOrigin())
            img.SetSpacing(header.GetSpacing())
            img.SetDirection(header.GetDirection())

            orient = sitk.DICOMOrientImageFilter()
            DirectionStr = orient.GetOrientationFromDirectionCosines(header.GetDirection())

            cont = True
            f1 = 'left' if 'L' in DirectionStr else 'right'
            f2 = 'posterior' if 'P' in DirectionStr else 'anterior'
            # f3 = 'superior' if 'S' in DirectionStr  else 'Inferior'
            # spaceStr2 = f1+'-'+f2+'-'+f3    
            spaceStr2 = f1 + '-' + f2

            if spaceStr2 == 'right-anterior' or spaceStr2 == 'left-anterior' or spaceStr2 == 'left-posterior':
                _ = spaceStr2
            else:
                cont = False

            if cont:

                if createfolder == 'True':
                    filename2 = filename + '_Nrrd'

                    create_Folder_path = os.path.join(OUTPUT_DIR, filename2)
                    if os.path.isdir(create_Folder_path) == False:
                        os.mkdir(create_Folder_path)
                    sitk.WriteImage(img, os.path.join(OUTPUT_DIR, filename2, filename + '.nrrd'))
                    return os.path.join(OUTPUT_DIR, filename2, filename + '.nrrd')
                else:
                    sitk.WriteImage(img, os.path.join(OUTPUT_DIR, filename + '.nrrd'))
                    return os.path.join(OUTPUT_DIR, filename + '.nrrd')

            else:
                print('Nrrd Format doesn''t work with', spaceStr2, 'direction')

            # perm = (1, 0)
            # data = np.transpose(registered.astype(np.float32), perm)

            # orient = sitk.DICOMOrientImageFilter()
            # DirectionStr = orient.GetOrientationFromDirectionCosines(header.GetDirection())

            # # a1 = header.GetDirection()[0]
            # # # a2 = header.GetDirection()[1]
            # # # a3 = header.GetDirection()[2]
            # # # a4 = header.GetDirection()[3]
            # # a5 = header.GetDirection()[4]
            # # # a6 = header.GetDirection()[5]
            # # # a7 = header.GetDirection()[6]
            # # # a8 = header.GetDirection()[7]
            # # # a9 = header.GetDirection()[8]

            # # # f1 = 'left' if a1 == 1 else 'right'
            # # # f2 = 'posterior' if a5 == 1  else 'anterior' 
            # # # f3 = 'superior' if a9 == 1  else 'Inferior'
            # # # spaceStr = f1+'-'+f2+'-'+f3

            # # f1 = 'left' if a1 > 0 else 'right'
            # # f2 = 'posterior' if a5 > 0  else 'anterior' 
            # # # f3 = 'superior' if a9 > 0  else 'Inferior'
            # # # spaceStr = f1+'-'+f2+'-'+f3
            # # spaceStr = f1+'-'+f2

            # cont = True
            # f1 = 'left' if 'L' in DirectionStr else 'right'
            # f2 = 'posterior' if 'P' in DirectionStr  else 'anterior' 
            # # f3 = 'superior' if 'S' in DirectionStr  else 'Inferior'
            # # spaceStr2 = f1+'-'+f2+'-'+f3    
            # spaceStr2 = f1+'-'+f2     

            # if spaceStr2 == 'right-anterior' or spaceStr2 == 'left-anterior' or spaceStr2 == 'left-posterior':
            #     spaceStr = spaceStr2
            # else:
            #     cont = False

            # if cont:
            #     fileheader = { 
            #         'dimension' : header.GetDimension(),
            #         'sizes' : np.array([header.GetWidth() ,header.GetHeight()]),
            #         'space': spaceStr,  
            #         # 'spacings': [header.GetSpacing()[0], header.GetSpacing()[1], header.GetSpacing()[2]],
            #         # 'thicknesses': [header.GetSpacing()[0], header.GetSpacing()[1], header.GetSpacing()[2]],
            #         # 'space directions': np.array([[a1,a2,a3],[a4,a5,a6],[a7,a8,a9]]), 
            #         # 'space directions': np.array([[header.GetSpacing()[0],a2,a3],[a4,header.GetSpacing()[1],a6],[a7,a8,header.GetSpacing()[2]]]), 
            #         'space directions': np.array([[header.GetSpacing()[0],0],[0,header.GetSpacing()[1]]]), 
            #         'space origin': np.array([header.GetOrigin()[0] ,header.GetOrigin()[1]])
            #     }
            #     if createfolder == 'True':
            #         filename2 =  filename + '_NRRD'

            #         create_Folder_path = os.path.join(OUTPUT_DIR,filename2)
            #         if os.path.isdir(create_Folder_path) == False:
            #             os.mkdir(create_Folder_path)
            #         nrrd.write(os.path.join(OUTPUT_DIR,filename2,filename+'.nrrd' ), data, fileheader)
            #     else:
            #         nrrd.write(os.path.join(OUTPUT_DIR,filename+'.nrrd' ), data, fileheader)
            # else:
            #     print('Nrrd Format doesn''t work with',spaceStr2,'direction')

        elif Datatype == 'SDicom':
            print('The software just supports 4 formats such as Nifti, Nrrd, Single and multi Dicom images.')
        elif Datatype == 'MDicom':
            print('The software just supports 4 formats such as Nifti, Nrrd, Single and multi Dicom images.')
        else:
            print('The software just supports 4 formats such as Nifti, Nrrd, Single and multi Dicom images.')


def convert_modalities(registered, header, DatatypeFrom, Datatype, OUTPUT_DIR, filename, createfolder,
                       ImagePositionPatients=None, ImageOrientationPatient=None, step=None, PixelSpacing=None,
                       filepaths=None):
    if DatatypeFrom == 'Nifti':
        if str(type(header)) == "<class 'SimpleITK.SimpleITK.Image'>":
            return Convert_nifti_to_others(registered, header, Datatype, OUTPUT_DIR, filename, createfolder)
        elif str(type(header)) == "<class 'nibabel.nifti1.Nifti1Image'>":
            perm = (2, 1, 0)
            registered = np.transpose(registered.astype(np.float32), perm)
            return Convert_nifti_nibabel_to_others(registered, header, Datatype, OUTPUT_DIR, filename, createfolder,
                                                   ImagePositionPatients, ImageOrientationPatient, step, PixelSpacing,
                                                   filepaths)
        else:
            print('This kind of header does''nt support.')
    elif DatatypeFrom == 'Nrrd':
        perm = (2, 1, 0)
        registered = np.transpose(registered.astype(np.float32), perm)
        return Convert_nrrd_to_others(registered, header, Datatype, OUTPUT_DIR, filename, createfolder)
    elif DatatypeFrom == 'SDicom':
        # perm = (1, 0)
        # registered = np.transpose(registered.astype(np.float32), perm)
        # Convert_nifti_nibabel_to_others(registered , header , Datatype , OUTPUT_DIR , filename)
        return Convert_dicom_to_others(registered, header, Datatype, OUTPUT_DIR, filename, createfolder)
    # elif DatatypeFrom == 'MDicom':
    #     # perm = (1, 0)
    #     # registered = np.transpose(registered.astype(np.float32), perm)
    #     # Convert_nifti_nibabel_to_others(registered , header , Datatype , OUTPUT_DIR , filename)
    #     Convert_dicom_to_others(registered , header , Datatype , OUTPUT_DIR , filename,createfolder)
    else:
        print('The software just supports 4 formats such as Nifti, Nrrd, Single and multi Dicom images.')
        return ""


def convertimage(souecefile, types='SDicom;Nifti;Nrrd', createfolder='True', destfolder='', suv=False):
    intype = ''
    if souecefile.endswith('.nii.gz') | souecefile.endswith('.nii'):
        nii = Nifti_read(souecefile)
        intype = 'Nifti'
    elif souecefile.endswith('.nrrd'):
        nrrd = Nrrd_read(souecefile)
        intype = 'Nrrd'
    elif souecefile.endswith('.dcm') | souecefile.endswith('.dicom'):
        file_name = souecefile.split('\\')[-1]
        data_directory = souecefile.replace(file_name, '')[:-1]
        # file_name = souecefile
        # data_directory = destfolder
        dicom = Dicom_read_simpleITK(data_directory, file_name)
        intype = 'SDicom'

    if '\\' in types:
        _types = types.split('\\')
    else:
        _types = types.split(';')

    _filename = ntpath.basename(souecefile).replace('.nii.gz', '').replace('.nii', '').replace('.nrrd', '').replace(
        '.dcm', '').replace('.dicom', '')

    mpath = []
    if (intype == 'Nifti'):
        for _type in _types:
            if _type != intype:
                try:
                    if isinstance(nii[0], np.ndarray):
                        # if nii != None:
                        path = convert_modalities(registered=nii[0], header=nii[1], DatatypeFrom='Nifti',
                                                  Datatype=_type, OUTPUT_DIR=destfolder, filename=_filename,
                                                  createfolder=createfolder)
                        if path is not None:
                            mpath.append(path)
                    else:
                        print('The image converter tool can''t read the input.')
                except Exception as e:
                    print('The image converter tool can''t read the input:', e)
            elif _type == intype:
                if createfolder == 'True':
                    create_Folder_path = os.path.join(destfolder, _filename)
                    if os.path.isdir(create_Folder_path) == False:
                        os.mkdir(create_Folder_path)
                if createfolder == 'True':
                    fullpath = os.path.join(create_Folder_path, _filename + '.nii.gz')
                else:
                    fullpath = os.path.join(destfolder, _filename + '.nii.gz')

                shutil.copy(souecefile, fullpath)
                if fullpath is not None:
                    mpath.append(fullpath)
    elif (intype == 'Nrrd'):
        for _type in _types:
            if _type != intype:
                try:
                    if isinstance(nrrd[0], np.ndarray):
                        # if nrrd != None:
                        path = convert_modalities(nrrd[0], nrrd[1], 'Nrrd', _type, destfolder, _filename, createfolder)
                        if path is not None:
                            mpath.append(path)
                    else:
                        print('The image converter tool can''t read the input.')
                except Exception as e:
                    print('The image converter tool can''t read the input:', e)
            elif _type == intype:
                if createfolder == 'True':
                    create_Folder_path = os.path.join(destfolder, _filename)
                    if os.path.isdir(create_Folder_path) == False:
                        os.mkdir(create_Folder_path)

                if createfolder == 'True':
                    fullpath = os.path.join(create_Folder_path, _filename + '.nrrd')
                else:
                    fullpath = os.path.join(destfolder, _filename + '.nrrd')

                shutil.copy(souecefile, fullpath)
                if fullpath is not None:
                    mpath.append(fullpath)

    elif (intype == 'SDicom'):
        cont = False
        for _type in _types:
            if _type != intype:
                try:
                    if isinstance(dicom[0], np.ndarray):
                        # if dicom != None:
                        fullpath = convert_modalities(dicom[0], dicom[1], 'SDicom', _type, destfolder, _filename,
                                                      createfolder)
                        cont = True
                    else:
                        print('The image converter tool can''t read the input.')
                except Exception as e:
                    print('The image converter tool can''t read the input:', e)
            elif _type == intype:
                if createfolder == 'True':
                    create_Folder_path = os.path.join(destfolder, _filename)
                    if os.path.isdir(create_Folder_path) == False:
                        os.mkdir(create_Folder_path)
                if createfolder == 'True':
                    fullpath = os.path.join(create_Folder_path, _filename + '.dcm')
                else:
                    fullpath = os.path.join(destfolder, _filename + '.dcm')

                shutil.copy(souecefile, fullpath)
                cont = True

        if suv == True:
            if cont == True:
                SUVresult = []
                try:
                    if isinstance(dicom[0], np.ndarray):
                        data = dicom[0]
                        header = dicom[1]
                        # if len(data.shape) == 3:
                        Dcmheader = pydicom.dcmread(os.path.join(data_directory, file_name))
                        # returned_val = compute_suv(data,Dcmheader)

                        # pet_image_file_list = [os.path.join(data_directory , file_name)]
                        # pet_image_file_list = sort_by_instance_number(pet_image_file_list)
                        # data = imread(pet_image_file_list)
                        # returned_val = compute_suv(data,Dcmheader)

                        # pet_image_file_list = [os.path.join(data_directory , file_name)]
                        # pet_image_file_list = sort_by_instance_number(pet_image_file_list)
                        # data = imread(pet_image_file_list)
                        SUV_OBJ = SUVscalingObj(Dcmheader)
                        returned_val = SUV_OBJ.get_scale_factor(suv_normalisation="bw", data=data)

                        convert_modalities(returned_val[0], header, 'SDicom', 'Nifti', destfolder,
                                           _filename + '_SUV_Calculation', createfolder)

                        suv_max = np.max(returned_val[0])
                        suv_mean = np.mean(returned_val[0])
                        suv_min = np.min(returned_val[0])
                        suv_std = np.std(returned_val[0])

                        SUVresult.append([_filename, suv_min, suv_max, suv_mean, suv_std, returned_val[1]])

                        SUVresult_arr = np.asarray(SUVresult)
                        col = ['FileName', 'suv_min', 'suv_max', 'suv_mean', 'suv_std', 'estimated']

                        SUVresult_df = pd.DataFrame(SUVresult_arr)
                        SUVresult_df.columns = col

                        CSVFilename = "SUV_Report_Single_Dicom.csv"
                        CSVfullpath = os.path.join(destfolder, CSVFilename)
                        SUVresult_df.to_csv(CSVfullpath, index=None)
                    else:
                        print('The image converter tool can''t read the input.')
                except:
                    # print('The image converter tool can''t read the input.')
                    data = dicom[0]
                    header = dicom[1]
                    Dcmheader = pydicom.dcmread(os.path.join(data_directory, file_name))
                    returned_val = compute_suv(data, Dcmheader)

                    convert_modalities(returned_val[0], header, 'SDicom', 'Nifti', destfolder,
                                       _filename + '_SUV_Estimation', createfolder)

                    suv_max = np.max(returned_val[0])
                    suv_mean = np.mean(returned_val[0])
                    suv_min = np.min(returned_val[0])
                    suv_std = np.std(returned_val[0])

                    SUVresult.append([_filename, suv_min, suv_max, suv_mean, suv_std, returned_val[1]])

                    SUVresult_arr = np.asarray(SUVresult)
                    col = ['FileName', 'suv_min', 'suv_max', 'suv_mean', 'suv_std', 'estimated']

                    SUVresult_df = pd.DataFrame(SUVresult_arr)
                    SUVresult_df.columns = col

                    CSVFilename = "SUV_Report_Single_Dicom.csv"
                    CSVfullpath = os.path.join(destfolder, CSVFilename)
                    SUVresult_df.to_csv(CSVfullpath, index=None)
                if fullpath is not None:
                    mpath.append(fullpath)
    else:
        print('The image converter tool can''t read the input.')
    if len(mpath) == 1:
        return mpath[0]
    elif len(mpath) == 0:
        return ""
    return mpath[0]


def convertimage_folder(sourcefolder, types='SDicom;MDicom;Nifti;Nrrd', createfolder='True', destfolder='', suv=False):
    if '\\' in types:
        _types = types.split('\\')
    else:
        _types = types.split(';')

    readFolder_nii(sourcefolder, _types, createfolder, destfolder)
    readFolder_nrrd(sourcefolder, _types, createfolder, destfolder)
    readFolder_dicom_nD(sourcefolder, _types, createfolder, destfolder, suv)
    readFolder_dicom_2D_recurcive(sourcefolder, _types, createfolder, destfolder, suv)

    files = os.listdir(sourcefolder)
    for i in range(len(files)):
        if "nii.gz" in files[i]:
            files[i] = destfolder + "/" + ".".join(files[i].split(".")[:-2]) + "_" + types.split(";")[0]
        else:
            files[i] = destfolder + "/" + ".".join(files[i].split(".")[:-1]) + "_" + types.split(";")[0]
    return files


def sort_by_instance_number(image_file_list):
    data = []
    for row in image_file_list:
        f = pydicom.dcmread(row)
        data.append({'f': row, 'n': f.InstanceNumber})
    data = sorted(data, key=lambda x: x['n'])
    return [x['f'] for x in data]


def imread(fpath):
    if isinstance(fpath, list):
        image_file_list = fpath
        image_file_list = sort_by_instance_number(image_file_list)
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(image_file_list)
    elif fpath.endswith('.list'):
        with open(fpath, 'r') as f:
            dicom_names = [x for x in f.read().split('\n') if len(x) > 0]
        if not os.path.exists(dicom_names[0]):
            image_file_list = [os.path.join(os.path.dirname(fpath), x) for x in dicom_names]
            image_file_list = sort_by_instance_number(image_file_list)
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(image_file_list)
    else:
        reader = sitk.ImageFileReader()
        reader.SetFileName(fpath)
    img = reader.Execute()
    arr = sitk.GetArrayFromImage(img)
    # spacing = img.GetSpacing()
    # origin = img.GetOrigin()
    # direction = img.GetDirection()    
    return arr


# import platform
# import dateutil
# def compute_suv(data,dcm_file):
#     '''
#     Calculates the conversion factor from Bq/mL to SUV bw [g/mL] using 
#     the dicom header information in one of the images from a dicom series
#     '''
#     # TODO: You can access these attributes in a more user friendly way rather
#     # than using the codes...change this at some point
#     nuclide_dose = dcm_file[0x054, 0x0016][0][0x0018, 0x1074].value  # Total injected dose (Bq)
#     weight = dcm_file[0x0010, 0x1030].value  # Patient weight (Kg)
#     half_life = float(dcm_file[0x054, 0x0016][0][0x0018, 0x1075].value)  # Radionuclide half life (s)

#     parse = lambda x: dateutil.parser.parse(x)

#     series_time = str(dcm_file[0x0008, 0x00031].value)  # Series start time (hh:mm:ss)
#     series_date = str(dcm_file[0x0008, 0x00021].value)  # Series start date (yyy:mm:dd)
#     series_datetime_str = series_date + ' ' + series_time
#     series_dt = parse(series_datetime_str)

#     nuclide_time = str(dcm_file[0x054, 0x0016][0][0x0018, 0x1072].value)  # Radionuclide time of injection (hh:mm:ss)
#     nuclide_datetime_str = series_date + ' ' + nuclide_time
#     nuclide_dt = parse(nuclide_datetime_str)

#     delta_time = (series_dt - nuclide_dt).total_seconds()
#     decay_correction = 2 ** (-1 * delta_time/half_life)
#     suv_factor = (weight * 1000) / (decay_correction * nuclide_dose)
#     estimated = False
#     ImageArray = sitk.GetImageFromArray(data, isVector=False)
#     image = sitk.Multiply(ImageArray, suv_factor)
#     SUV = sitk.GetArrayFromImage(image)
#     return SUV ,estimated


# our code
def compute_suv(raw, f):
    estimated = False

    # pet_image_file_list = [os.path.join(path,x) for x in os.listdir(path) if x.endswith('.dcm')]
    # f=pydicom.dcmread(pet_image_file_list[0])

    try:
        weight_grams = float(f.PatientWeight) * 1000
    except:
        traceback.print_exc()
        weight_grams = 75000
        estimated = True

    try:
        # Get Scan time
        scantime = datetime.datetime.strptime(str(int(float(f.AcquisitionTime))), '%H%M%S')
        # Start Time for the Radiopharmaceutical Injection
        injection_time = datetime.datetime.strptime(
            f.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime, '%H%M%S.%f')
        # Half Life for Radionuclide # seconds
        half_life = float(f.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife)
        # Total dose injected for Radionuclide
        injected_dose = float(f.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)

        # Calculate decay
        decay = np.exp(-np.log(2) * ((scantime - injection_time).seconds) / half_life)
        # Calculate the dose decayed during procedure
        injected_dose_decay = injected_dose * decay  # in Bq
    except:
        traceback.print_exc()
        decay = np.exp(-np.log(2) * (1.75 * 3600) / 6588)  # 90 min waiting time, 15 min preparation
        injected_dose_decay = 420000000 * decay  # 420 MBq
        estimated = True

    # Calculate SUV # g/ml
    suv = raw * weight_grams / injected_dose_decay

    return suv, estimated

# Carlos
# def compute_suv(bw,hdr):

#     if ('ATTN' in hdr.CorrectedImage and 'DECY' in hdr.CorrectedImage) and hdr.DecayCorrection == 'START':
#         if hdr.Units == 'BQML':
#             # seconds (0018,1075)
#             half_life = hdr.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife
#             scan_time = hdr.SeriesTime  # (0008,0031)
#             # (0018,1072)
#             start_time = hdr.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime
#             # convert tref and injtime from strings to datetime
#             scan_time = strptime(scan_time.split('.')[0], '%H%M%S')
#             start_time = strptime(start_time.split('.')[0], '%H%M%S')
#             decay_time = scan_time - start_time
#             # (18,1074)
#             inj_act = hdr.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose
#             decayed_act = inj_act * 2**(-decay_time.total_seconds()/half_life)
#             SUVbw_scale_factor = bw/decayed_act
#             suvbw_img = (hdr.pixel_array*hdr.RescaleSlope +
#                          hdr.RescaleIntercept)*SUVbw_scale_factor
#             estimated = False

#             return suvbw_img,estimated
