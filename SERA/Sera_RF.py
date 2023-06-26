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
from Sera_ReadWrite import readimage,convert_modalities,similar
import psutil
import timeit
from RF_main import SERA_FE_main_Fun



#===============================SERA functions==========================================
# def SERA_main_process(da_original, VoxelSize, da_label,
#                       BinSize,
#                       isotVoxSize,
#                       isotVoxSize2D,
#                       DataType,
#                       DiscType,
#                       qntz,
#                       VoxInterp,
#                       ROIInterp,
#                       isScale,
#                       isGLround,
#                       isReSegRng,
#                       isOutliers,
#                       isQuntzStat,
#                       isIsot2D,
#                       ReSegIntrvl,
#                       ROI_PV,
#                       IVH_Type,
#                       IVH_DiscCont,
#                       IVH_binSize,
#                       ROIsPerImg,
#                       isROIsCombined,
#                       Feats2out,
#                       IVHconfig,
#                       destfolder,rand_number,sucsessList):

    
    
#     # print(type(da_original[0]))
#     # print('fffff = ',np.issubdtype(da_original[0].dtype, np.floating))
    
#     if np.issubdtype(da_original[0].dtype, np.floating) == True:
#         # ct_min = np.min(da_original[0])
#         ct_max = np.max(da_original[0])
        
#         if ct_max <= 1:
#             da_original[0] = da_original[0] * 255
    
#         da_original[0] = da_original[0].astype(int)


#     Data_inputs = {"Data_Orginal": da_original[0]}
#     # fullpath = os.path.join(destfolder, "Data_Orginal.mat")
#     fullpath = "Data_Orginal_"+da_original[3]+".mat"
#     # fullpath = "Data_Orginal.mat"
    
#     savemat(fullpath, Data_inputs)

#     voxel_inputs = {"VoxelSizeInfo": VoxelSize}
#     # fullpath = os.path.join(destfolder, "VoxelSizeInfo.mat")
    
#     fullpath = "VoxelSizeInfo_"+da_original[3]+".mat"
#     # fullpath =  "VoxelSizeInfo.mat"
#     savemat(fullpath, voxel_inputs)

#     Data_ROI_np = da_label[0]
#     AllROIs = []
#     ROIMax = np.max(Data_ROI_np)
#     number_of_ROI = ROIsPerImg if ROIsPerImg < ROIMax else ROIMax
#     for ijk in range(0, number_of_ROI):
#         Data_ROI_aa = Data_ROI_np == (ijk+1)
#         Data_ROI_aa = Data_ROI_aa.astype(int)
#         labeled_array, num_features = ndimage.label(Data_ROI_aa)

#         ROIs = []
#         number_of_ROI_per_ROI = num_features
#         # number_of_ROI_per_ROI = ROIsPerImg if ROIsPerImg < num_features else num_features
#         for jk in range(0, number_of_ROI_per_ROI):
#             arr = labeled_array == (jk+1)
#             arr = arr.astype(int)
#             volume = ndimage.sum(arr)
#             ROIs.append((arr, volume))

#         ROIs.sort(key=lambda x: x[1], reverse=True)
#         ROIs = ROIs[:ROIsPerImg]
#         AllROIs.append(ROIs)

#     current_List = []
#     count = 1
#     for mn in AllROIs:
#         count2 = 1
#         for mn2 in mn:
#             mn3 = mn2[0].astype(np.uint8)
#             ROI_inputs = {"ROI": mn3, "name": str(count)+"_"+str(count2)}
#             current_List.append(ROI_inputs)
#             count2 = count2+1
#         count = count+1

#     current_Tuple = tuple(current_List)
#     data = {'Data_ROI': current_Tuple}
#     # fullpath = os.path.join(destfolder, "Data_ROI.mat")
    
#     fullpath = "Data_ROI_"+da_original[3]+".mat"
#     # fullpath = "Data_ROI.mat"
#     savemat(fullpath, data, oned_as='column')

#     if DataType == 'CT / SPECT':
#         DataType = 'CT'
#     # if DataType == 'MR':
#     #     DataType = 'MRscan'
    
    
#     user_inputs = {"BinSize": BinSize, "isotVoxSize": isotVoxSize, "isotVoxSize2D": isotVoxSize2D,
#                    "DataType": DataType,
#                    "DiscType": DiscType,
#                    "qntz": qntz,
#                    "VoxInterp": VoxInterp,
#                    "ROIInterp": ROIInterp,
#                    "isScale": isScale,
#                    "isGLround": isGLround,
#                    "isReSegRng": isReSegRng,
#                    "isOutliers": isOutliers,
#                    "isQuntzStat": isQuntzStat,
#                    "isIsot2D": isIsot2D,
#                    "ReSegIntrvl": ReSegIntrvl,
#                    "ROI_PV": ROI_PV,
#                    "IVH_Type": IVH_Type,
#                    "IVH_DiscCont": IVH_DiscCont,
#                    "IVH_binSize": IVH_binSize,
#                    "ROIsPerImg": ROIsPerImg,
#                    "isROIsCombined": isROIsCombined,
#                    "Feats2out": Feats2out,
#                    "IVHconfig": IVHconfig
#                    }
#     # fullpath = os.path.join(destfolder, "user_inputs.mat")
#     fullpath = "user_inputs_"+da_original[3]+".mat"
#     # fullpath = "user_inputs.mat"
#     savemat(fullpath, user_inputs)

#     # winView = r"SERA\for_redistribution_files_only"
#     # timestamp1 = time.time()
#     # os.startfile('SERA_Feature_Extraction.exe')
#     os.system('SERA_Feature_Extraction.exe '+da_original[3]+'')

#     # timestamp2 = time.time()
#     # print("This took %.2f seconds", (timestamp2 - timestamp1))
    
#     # matfilesss = os.listdir('')
#     # print(matfilesss)
    
#     os.remove("Data_Orginal_"+ da_original[3]+".mat")
#     os.remove("Data_ROI_"+ da_original[3]+".mat")
#     os.remove("user_inputs_"+ da_original[3]+".mat")
#     os.remove("VoxelSizeInfo_"+ da_original[3]+".mat")
#     matFilename = "All_extracted_features_"+da_original[3]+".mat"
#     matFilename_path = "All_extracted_features_"+ da_original[3]+"_"+str(rand_number)+".mat"
#     # cwd = os.getcwd()
    
#     if  os.path.exists(matFilename) == True:  
#         if rand_number >= 0:
#             # matFilename = "All_extracted_features_"+ da_original[3]+".mat"
#             # matFilename_path = "All_extracted_features_"+ da_original[3]+"-"+str(rand_number)+".mat"
        
#             fullpath = os.path.join(destfolder, matFilename_path)
#             shutil.move(matFilename, fullpath)
        
#         lock = threading.Lock()
#         lock.acquire()
#         sucsessList.append((matFilename_path,True))
#         lock.release()
#         return True    
#     else:  
#         print('Out of Memory or the radiomics feature generator tool can not work on',da_original[3],', check your input or hyperparameter.')
#         lock = threading.Lock()
#         lock.acquire()
#         sucsessList.append((matFilename_path,False))
#         lock.release()  
#         return False




def SERA_main_process(da_original, VoxelSize, da_label,
                      BinSize,
                      isotVoxSize,
                      isotVoxSize2D,
                      DataType,
                      DiscType,
                      qntz,
                      VoxInterp,
                      ROIInterp,
                      isScale,
                      isGLround,
                      isReSegRng,
                      isOutliers,
                      isQuntzStat,
                      isIsot2D,
                      ReSegIntrvl,
                      ROI_PV,
                      IVH_Type,
                      IVH_DiscCont,
                      IVH_binSize,
                      ROIsPerImg,
                      isROIsCombined,
                      Feats2out,
                      IVHconfig,
                      destfolder,rand_number,sucsessList,roi_segments,obj):

    
    if np.issubdtype(da_original[0].dtype, np.floating) == True:
        ct_max = np.max(da_original[0])
        
        if ct_max <= 1:
            da_original[0] = da_original[0] * 255
    
        da_original[0] = da_original[0].astype(int)


    data_orginal = np.squeeze(da_original[0]).astype(np.float32) 

    VoxelSizeInfo = np.squeeze(VoxelSize).astype(np.float32)

    if roi_segments == None:

        Data_ROI_np = da_label[0]
        AllROIs = []
        ROIMax = np.max(Data_ROI_np)
        number_of_ROI = ROIsPerImg if ROIsPerImg < ROIMax else ROIMax
        for ijk in range(0, number_of_ROI):
            Data_ROI_aa = Data_ROI_np == (ijk+1)
            Data_ROI_aa = Data_ROI_aa.astype(int)
            labeled_array, num_features = ndimage.label(Data_ROI_aa)

            ROIs = []
            number_of_ROI_per_ROI = num_features

            for jk in range(0, number_of_ROI_per_ROI):
                arr = labeled_array == (jk+1)
                arr = arr.astype(int)
                volume = ndimage.sum(arr)
                ROIs.append((arr, volume))

            ROIs.sort(key=lambda x: x[1], reverse=True)
            if len(ROIs) > ROIsPerImg:
                ROIs = ROIs[:ROIsPerImg]
            AllROIs.append(ROIs)

        current_List = []
        current_Name = []
        count = 1
        for mn in AllROIs:
            # count2 = 1
            for mn2 in mn:
                # label_name = 'label-'+str(count)+'_'+str(count2)
                label_name = 'label '+str(count)
                mn3 = mn2[0].astype(np.uint8)
                current_List.append(mn3)
                current_Name.append(label_name)
                # count2 = count2+1
                # count = count+1
            count = count+1

    else:
        # Data_ROI_np = da_label[0]
        # AllROIs = []
        # ROIMax = np.max(Data_ROI_np)
        # number_of_ROI = ROIsPerImg if ROIsPerImg < ROIMax else ROIMax
        # for ijk in range(0, number_of_ROI):
        #     Data_ROI_aa = Data_ROI_np == (ijk+1)
        #     Data_ROI_aa = Data_ROI_aa.astype(int)
        #     labeled_array, num_features = ndimage.label(Data_ROI_aa)

        #     ROIs = []
        #     number_of_ROI_per_ROI = num_features

        #     for jk in range(0, number_of_ROI_per_ROI):
        #         arr = labeled_array == (jk+1)
        #         arr = arr.astype(int)
        #         volume = ndimage.sum(arr)
        #         ROIs.append((arr, volume))

        #     ROIs.sort(key=lambda x: x[1], reverse=True)
        #     ROIs = ROIs[:ROIsPerImg]
        #     AllROIs.append(ROIs)
        # np.save('roi_segments', roi_segments)
        # import pickle
        # with open('roi_segments.pkl', 'wb') as file:
        #     pickle.dump(roi_segments, file)

        # current_List = []
        # current_Name = []
        # # count = 1
        # for mn in roi_segments[obj]:
        #     # count2 = 1
        #     for mn2 in mn:

        #         for mn3 in mn[mn2]:
                        
        #             mn4 = mn3[0].astype(np.uint8)
        #             current_List.append(mn4)
        #             # current_Name.append(mn2[2])
        #             # count2 = count2+1
        #         # count = count+1

        current_List = []
        current_Name = []
        current_Size = []

        for mn in roi_segments[1]:
            for mn2 in mn:
                for mn3 in mn[mn2]:        
                    mn4 = mn3[0].astype(np.uint8)
                    current_List.append(mn4)
                    current_Name.append(mn2)
                    current_Size.append(mn3[1])


        # current_List = []
        # current_Name = []
        # # count = 1
        # for mn in roi_segments[obj]:
        #     # count2 = 1
        #     for mn2 in mn:
        #         mn3 = mn2[0].astype(np.uint8)
        #         current_List.append(mn3)
        #         current_Name.append(mn2[2])
        #         # count2 = count2+1
        #     # count = count+1

    # import pickle

    # # save dictionary to person_data.pkl file
    # with open('roi_segments_data.pkl', 'wb') as fp:
    #     pickle.dump(roi_segments, fp)
    #     # print('dictionary saved successfully to file')

    # with open('current_List.pkl', 'wb') as fp:
    #     pickle.dump(current_List, fp)
    #     # print('dictionary saved successfully to file')

    if DataType == 'CT / SPECT':
        DataType = 'CT'
    # if DataType == 'MR':
    #     DataType = 'MRscan'

    Co = 0
    Co_n = 0
    lastCo = 0
    pd_ROIS = []
    con = False
    for curROI in current_List:

        try:
            lastCo = Co
            Co = Co +1
            Co_n = Co_n + 1
            Data_ROI_mat = np.squeeze(curROI).astype(np.float32)
            Data_ROI_Name = str(Co)

            AllFeat = SERA_FE_main_Fun(data_orginal,
                            Data_ROI_mat,
                            Data_ROI_Name,
                            VoxelSizeInfo,
                            BinSize,
                            DataType,
                            isotVoxSize,
                            isotVoxSize2D,
                            DiscType,
                            qntz,
                            VoxInterp,
                            ROIInterp,
                            isScale,
                            isGLround,
                            isReSegRng,
                            isOutliers,
                            isQuntzStat,
                            isIsot2D,
                            ReSegIntrvl,
                            ROI_PV,
                            IVH_Type,
                            IVH_DiscCont,
                            IVH_binSize,
                            ROIsPerImg,
                            isROIsCombined,
                            Feats2out,
                            IVHconfig)
            
            if Co == 1:
                AllFeats = AllFeat
                con = True
            else:
                AllFeats += AllFeat
                con = True

            for i in range(len(AllFeat)) :
                # pd_ROIS += [Co]  
                pd_ROIS += [current_Name[Co_n - 1]]  
            # print(AllFeats)
        except:
            Co = lastCo

    if con:
        AllFeats = np.array(AllFeats)

        matFilename = "All_extracted_features_"+da_original[3]+".xlsx"
        matFilename_path = "Patient_"+ da_original[3]+"_"+str(rand_number)+".xlsx"

        features = pd.read_excel("SERA Features Name and Tags.xlsx" , sheet_name= str(Feats2out) , engine='openpyxl' ,header=0)
        fe2 = features.iloc[:,-1]
        column1 = list(fe2)
        column = column1


        ExtendedBinSize = []
        try:
            for i in range(int(AllFeats.shape[0]/len(BinSize))) :
                ExtendedBinSize += [j for j in BinSize]
        except:
            for i in range(int(AllFeats.shape[0])) :
                ExtendedBinSize += [BinSize]
        
        FE = pd.DataFrame(AllFeats)
        FE.columns = column

                
        FE.insert(0, "ROI", pd_ROIS , True)
        FE.insert(0, "Bin Size", ExtendedBinSize , True)
        pd_id = da_original[3]

        pd_ids = []
        for i in range(int(AllFeats.shape[0])) :
            pd_ids += [pd_id]


        FE.insert(0, "PatientID", pd_ids , True)
                
        para = {
                'Parameter':['Bin Size/Width', '3D Isotropic Voxel Size Flag', '2D Isotropic Voxel Size Flag', 'Image Modality Type',
                                'Discretization Type', 'Quantization Type', 'Image Resampling Interpolation Type', 'ROI Resampling Interpolation Type',
                                'Scale (Resampling) Flag', 'Round Voxel Intensity Values Flag', 'Range Re-Segmentation Flag', 
                                'Intensity Outlier Re-Segmentation Flag','Image Quantization Flag','Isotropic 2D Voxels Flag',
                                'Re-Segmentation Interval Range','ROI Partial Volume Threshold','Intensity Volume Histogram (IVH) Type','Intensity Volume Histogram (IVH) Discretization Type',
                                'Intensity Volume Histogram (IVH) Discretization Binning Option/Size',
                                'Max Number Of ROIs per Image',
                                'Combine Multiple ROIs To ONE Flag',
                                'Type of Ouyput Data'
                                ],
                'Value':[BinSize,
                        isotVoxSize,
                        isotVoxSize2D,
                        DataType,
                        DiscType,
                        qntz,
                        VoxInterp,
                        ROIInterp,
                        isScale,
                        isGLround,
                        isReSegRng,
                        isOutliers,
                        isQuntzStat,
                        isIsot2D,
                        ReSegIntrvl,
                        ROI_PV,
                        IVH_Type,
                        IVH_DiscCont,
                        IVH_binSize,
                        ROIsPerImg,
                        isROIsCombined,
                        Feats2out]
            }
        

        now = datetime.datetime.now()
        

        Currentdate = now.strftime("%m-%d-%Y_%H%M%S")

                
        result = pd.DataFrame(para)
                        
        if rand_number >= 0:    
            CSVFilename = matFilename_path.split(".")[0]+".xlsx"
        else:
            CSVFilename = matFilename.split(".")[0]+"_"+Currentdate+".xlsx"

        CSVfullpath = os.path.join(destfolder, CSVFilename)

        writer = pd.ExcelWriter(CSVfullpath, engine='xlsxwriter') # pylint: disable=abstract-class-instantiated
        FE.to_excel(writer, sheet_name='Extracted_features',index=None)
        result.to_excel(writer, sheet_name='Parameters',index=None)
        writer.save() 
            
        sucsessList.append((matFilename,True))


    return True   

                        
        

def FE_export_excels(destfolder , str_rand_number):
    List_files = os.listdir(destfolder)
    FirstFile = True  
    for i in List_files:
        postfix = str_rand_number + '.xlsx'
        if i.endswith(postfix):
            fullpath = os.path.join(destfolder,i)
            data = pd.read_excel(fullpath,sheet_name='Extracted_features',header=0,index_col=None)
            Parameters = pd.read_excel(fullpath,sheet_name='Parameters',header=0,index_col=None)
            # extracted_features = extracted_features.astype(np.float32)			                

            if FirstFile == True :
                FirstFile = False
                merged_df = data
            else:
                frames = [merged_df, data]
                merged_df = pd.concat(frames)
            
            
            fullpathDes = os.path.join(destfolder,'Radiomics features for single patient')
            if os.path.exists(fullpathDes) == False:
                os.mkdir(fullpathDes)

            fullpathDesFilename = os.path.join(fullpathDes,i)
            shutil.move(fullpath,fullpathDesFilename)
            
            # os.remove(fullpath)
    if FirstFile == False:    

        now = datetime.datetime.now() 
        Currentdate = now.strftime("%m-%d-%Y_%H%M%S")                                
        CSVFilename = "All_extracted_features_"+Currentdate+".xlsx"
        CSVfullpath = os.path.join(destfolder, CSVFilename)
        writer = pd.ExcelWriter(CSVfullpath, engine='xlsxwriter') # pylint: disable=abstract-class-instantiated
        merged_df.to_excel(writer, sheet_name='Extracted_features',index=None)
        Parameters.to_excel(writer, sheet_name='Parameters',index=None)
        writer.save()




def FE_export(destfolder , str_rand_number,Feats2out,result,BinSize):
    List_files = os.listdir(destfolder)
    FirstFile = True  
    for i in List_files:
        postfix = str_rand_number + '.mat'
        if i.endswith(postfix):
            fullpath = os.path.join(destfolder,i)
            data = loadmat(fullpath)
            extracted_features = data['All_extracted_features']
            extracted_features = extracted_features.astype(np.float32)
            
            
            col = ['ROI']
            features = pd.read_excel("SERA Features Name and Tags.xlsx" , sheet_name= str(Feats2out) , engine='openpyxl' ,header=0)
            fe2 = features.iloc[:,-1]
            column1 = list(fe2)
            column = col + column1

            
            # col = ['ROI']
            # column1 = ['img_'+str(i) for i in range(1, extracted_features.shape[1])]
            # column = col + column1

            FE = pd.DataFrame(extracted_features)
            FE.columns = column

            rep = "_"+str_rand_number
            pd_id = ((i.replace(rep,"")).replace("All_extracted_features_","")).split(".")[0]
			                
            
            ExtendedBinSize = []
            try:
                for i in range(int(extracted_features.shape[0]/len(BinSize))) :
                    ExtendedBinSize += [j for j in BinSize]
            except:
                for i in range(int(extracted_features.shape[0])) :
                    ExtendedBinSize += [BinSize]
            FE.insert(0, "Bin Size", ExtendedBinSize , True)


            pd_ids = []
            for i in range(int(extracted_features.shape[0])) :
                pd_ids += [pd_id]
            FE.insert(0, "PatientID", pd_ids , True)



            # CSVFilename = i.split(".")[0]+".csv"
            # CSVfullpath = os.path.join(destfolder, CSVFilename)
            # FE.to_csv(CSVfullpath ,index=None)        
            
            if FirstFile == True :
                FirstFile = False
                merged_df = FE
            else:
                frames = [merged_df, FE]
                merged_df = pd.concat(frames)
            
            os.remove(fullpath)
    if FirstFile == False:    
        # CSVFilename = "All_extracted_feature.csv"
        # CSVfullpath = os.path.join(destfolder, CSVFilename)
        # merged_df.to_csv(CSVfullpath ,index=None)  
        
        
        now = datetime.datetime.now() # current date and time
                        
                        
                    
                        
        Currentdate = now.strftime("%m-%d-%Y_%H%M%S")

                                                        
        CSVFilename = "All_extracted_feature_"+Currentdate+".xlsx"
        
        # CSVFilename = "All_extracted_feature.xlsx"
        CSVfullpath = os.path.join(destfolder, CSVFilename)
        
        writer = pd.ExcelWriter(CSVfullpath, engine='xlsxwriter') # pylint: disable=abstract-class-instantiated
        merged_df.to_excel(writer, sheet_name='Extracted_features',index=None)
        result.to_excel(writer, sheet_name='Parameters',index=None)
        writer.save()
                                    
        

def CheckGTV(x):
    Result = True
    x = x.astype(np.uint8)
    unique, counts = np.unique(x, return_counts=True)
    unique = list(unique)
    counts = list(counts)
    if len(unique) > 1:
        unique2 = [i for i in range(len(unique))]
        if unique == unique2:
            for i in counts:
                if i != 0:
                    Result = True
                else:
                    Result = False
                    break
        else:
            Result = False
    else:
        Result = False

    return Result


def SERA(da_original, da_label,
                BinSize,
                isotVoxSize,
                isotVoxSize2D,
                DataType,
                DiscType,
                qntz,
                VoxInterp,
                ROIInterp,
                isScale,
                isGLround,
                isReSegRng,
                isOutliers,
                isQuntzStat,
                isIsot2D,
                ReSegIntrvl01,
                ReSegIntrvl02,
                ROI_PV,
                IVH_Type,
                IVH_DiscCont,
                IVH_binSize,
                ROIsPerImg,
                isROIsCombined,
                Feats2out,
                destfolder,roi_segments,obj):
        
    da_original = readimage(da_original)
    da_label = readimage(da_label)
    sucsessList = []
    try:
        if isinstance(da_original[0], np.ndarray) & isinstance(da_label[0], np.ndarray):

        # if da_original[0] != None & da_label[0] != None:  

                            
            if da_original[2] == 'Nifti' or da_original[2] == 'Dicom':
                perm = (2, 1, 0)
                da_original[0] = np.transpose(da_original[0], perm)
            if da_label[2] == 'Nifti' or da_label[2] == 'Dicom':
                perm = (2, 1, 0)
                da_label[0] = np.transpose(da_label[0], perm)  

            dim01 = np.shape(da_original[0])
            dim02 = np.shape(da_label[0])
            if (dim01 == dim02):
                
                if len(dim01) == 3 & len(dim02) == 3 :   
            
                    if da_original[2] == 'Nifti':  
                        VoxelSize = da_original[1].GetSpacing()
                    if da_original[2] == 'Nrrd':  
                        VoxelSize = (da_original[1]['space directions'][0,0],da_original[1]['space directions'][1,1],da_original[1]['space directions'][2,2])
                    if da_original[2] == 'Dicom':  
                        VoxelSize = da_original[1].GetSpacing()     

                    if da_label[2] == 'Nifti':  
                        VoxelSizelabel = da_label[1].GetSpacing()
                    if da_label[2] == 'Nrrd':  
                        VoxelSizelabel = (da_label[1]['space directions'][0,0],da_label[1]['space directions'][1,1],da_label[1]['space directions'][2,2])
                    if da_label[2] == 'Dicom':  
                        VoxelSizelabel = da_label[1].GetSpacing()     
                    
                    if np.all(np.around(VoxelSizelabel, 3) == np.around(VoxelSize, 3)):
                        
                        if CheckGTV(da_label[0]):
                            # BinSize = int(BinSize)
                            # isotVoxSize = int(isotVoxSize)
                            # isotVoxSize2D = int(isotVoxSize2D)
                            # isScale = int(isScale)
                            # isGLround = int(isGLround)
                            # isReSegRng = int(isReSegRng)
                            # isOutliers = int(isOutliers)
                            # isQuntzStat = int(isQuntzStat)
                            # isIsot2D = int(isIsot2D)
                            # ReSegIntrvl = [int(ReSegIntrvl01), int(ReSegIntrvl02)]
                            # ROI_PV = float(ROI_PV)
                            # IVH_Type = int(IVH_Type)
                            # IVH_DiscCont = int(IVH_DiscCont)
                            # IVH_binSize = int(IVH_binSize)
                            # ROIsPerImg = int(ROIsPerImg)
                            # isROIsCombined = int(isROIsCombined)
                            # Feats2out = int(Feats2out)
                            # IVHconfig = [IVH_Type, IVH_DiscCont, IVH_binSize]
                            
                            contin = True 
                            try:
                                # BinSize_values = BinSize.split(',')
                                # BinSize_values = [int(mi) for mi in BinSize_values]
                                # # print(BinSize_values)
                                
                                # if len(BinSize_values) == 1:
                                #     BinSize = BinSize_values[0]
                                #     BinSize = int(BinSize)
                                # else:
                                #     BinSize = BinSize_values
                                #     BinSize = [int(mi) for mi in BinSize]
                            
                                # contin = True        
                                # print(BinSize)  
                                # 
                                if BinSize != '':
                                    BinSize_values = BinSize.split(',')
                                    BinSize_values2 = []
                                    for mi in BinSize_values:
                                        if mi != '':
                                            BinSize_values2.append(int(mi))
                                    # print(BinSize_values2)

                                    if len(BinSize_values2) == 1:
                                        BinSize = BinSize_values2[0]
                                        BinSize = int(BinSize)
                                    elif len(BinSize_values2) == 0:
                                        BinSize = int(32) 
                                    else:   
                                        BinSize = []
                                        for mi in BinSize_values2:
                                            if mi != '':
                                                BinSize.append(int(mi)) 
                                    # if len(BinSize) == 0 :
                                    #     BinSize = int(32)            
                                else:
                                    BinSize = int(32)
                                
                            except:
                                # print('Bin size input is invalid.')
                                BinSize = int(32)
                                # exit()
                            
                            if contin == True:                  
                                isotVoxSize = float(isotVoxSize)
                                isotVoxSize2D = float(isotVoxSize2D)
                                isScale = int(isScale)
                                isGLround = int(isGLround)
                                isReSegRng = int(isReSegRng)
                                isOutliers = int(isOutliers)
                                isQuntzStat = int(isQuntzStat)
                                isIsot2D = int(isIsot2D)
                                ReSegIntrvl = [int(ReSegIntrvl01), int(ReSegIntrvl02)]
                                ROI_PV = float(ROI_PV)
                                IVH_Type = int(IVH_Type)
                                IVH_DiscCont = int(IVH_DiscCont)
                                IVH_binSize = float(IVH_binSize)
                                ROIsPerImg = int(ROIsPerImg)
                                isROIsCombined = int(isROIsCombined)
                                Feats2out = int(Feats2out)
                                IVHconfig = [IVH_Type, IVH_DiscCont, IVH_binSize]
                                
                                SERA_main_process(da_original, VoxelSize, da_label,
                                                    BinSize,
                                                    isotVoxSize,
                                                    isotVoxSize2D,
                                                    DataType,
                                                    DiscType,
                                                    qntz,
                                                    VoxInterp,
                                                    ROIInterp,
                                                    isScale,
                                                    isGLround,
                                                    isReSegRng,
                                                    isOutliers,
                                                    isQuntzStat,
                                                    isIsot2D,
                                                    ReSegIntrvl,
                                                    ROI_PV,
                                                    IVH_Type,
                                                    IVH_DiscCont,
                                                    IVH_binSize,
                                                    ROIsPerImg,
                                                    isROIsCombined,
                                                    Feats2out,
                                                    IVHconfig,
                                                    destfolder,-1,sucsessList,roi_segments,obj)       
                                

                                
                        else:
                            raise('Label Image is Corrupted or Empty.') 
                                    

                        # else:
                        #     print ('Failed in feature extraction')
                    else:
                        raise('Images voxel size must be same.')    
                else:
                    raise('Images must be 3D.')
            else:
                raise('Dimension of original and segmented image not equal.')
        else:
            raise('You must use an approprate type of input.')
    except Exception as e:
        raise('Out of Memory or the parameters of radiomics feature generator tool should be selected properly:', e)
		
		
    return ""



def SERA_folder(da_original_path, da_label_path,
                BinSize,
                isotVoxSize,
                isotVoxSize2D,
                DataType,
                DiscType,
                qntz,
                VoxInterp,
                ROIInterp,
                isScale,
                isGLround,
                isReSegRng,
                isOutliers,
                isQuntzStat,
                isIsot2D,
                ReSegIntrvl01,
                ReSegIntrvl02,
                ROI_PV,
                IVH_Type,
                IVH_DiscCont,
                IVH_binSize,
                ROIsPerImg,
                isROIsCombined,
                Feats2out,
                destfolder,roi_segments,obj):
    
    Fixed_datas = os.listdir(da_original_path)
    fixed = [i for i in Fixed_datas if (i.endswith(".nrrd") | i.endswith(".dcm") | i.endswith(".dicom") | i.endswith(".nii") | i.endswith(".nii.gz"))]
    
    Moved_datas = os.listdir(da_label_path)
    moved = [i for i in Moved_datas if (i.endswith(".nrrd") | i.endswith(".dcm") | i.endswith(".dicom") | i.endswith(".nii") | i.endswith(".nii.gz"))]
    
    rand_number = randint(1, 10000) * 19
    # thread_list = []
    
    contin = True 
    try:
        # BinSize_values = BinSize.split(',')
        # BinSize_values = [int(mi) for mi in BinSize_values]
        # # raise(BinSize_values)
        
        # if len(BinSize_values) == 1:
        #     BinSize = BinSize_values[0]
        #     BinSize = int(BinSize)
        # else:
        #     BinSize = BinSize_values
        #     BinSize = [int(mi) for mi in BinSize]
    
        # contin = True        
        # print(BinSize)  
        # 
        if BinSize != '':
            BinSize_values = BinSize.split(',')
            BinSize_values2 = []
            for mi in BinSize_values:
                if mi != '':
                    BinSize_values2.append(int(mi))
            # print(BinSize_values2)

            if len(BinSize_values2) == 1:
                BinSize = BinSize_values2[0]
                BinSize = int(BinSize)
            elif len(BinSize_values2) == 0:
                BinSize = int(32) 
            else:   
                BinSize = []
                for mi in BinSize_values2:
                    if mi != '':
                        BinSize.append(int(mi)) 
            # if len(BinSize) == 0 :
            #     BinSize = int(32)            
        else:
            BinSize = int(32)
         
    except:
        # print('Bin size input is invalid.')
        BinSize = int(32)
        
    
    if contin == True: 
                    # exit()
        sucsessList = []
        result = pd.DataFrame()
        # Num_worker = 1 + int(psutil.virtual_memory()[1]/pow(10,9)/8)
        # Num_worker = 1
        Num_worker = int(psutil.virtual_memory()[1]/pow(10,9)/8)
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
            
                        futures.append(
                            executor.submit(
                                SERA_folder_Thread, da_original_path, da_label_path,
                                            BinSize,
                                            isotVoxSize,
                                            isotVoxSize2D,
                                            DataType,
                                            DiscType,
                                            qntz,
                                            VoxInterp,
                                            ROIInterp,
                                            isScale,
                                            isGLround,
                                            isReSegRng,
                                            isOutliers,
                                            isQuntzStat,
                                            isIsot2D,
                                            ReSegIntrvl01,
                                            ReSegIntrvl02,                                            
                                            ROI_PV,
                                            IVH_Type,
                                            IVH_DiscCont,
                                            IVH_binSize,
                                            ROIsPerImg,
                                            isROIsCombined,
                                            Feats2out,
                                            destfolder,rand_number,sucsessList,co,moved[selected_index],result
                                            ,roi_segments,obj
                            )
                        )

                        if(len(moved)>0):
                            moved.pop(selected_index)

                    # else:
                        # print('There is no image with the same name of',co,'in another folder.')
 
    
        executor.shutdown(wait=True)
            # print(len(futures))    
            # for future in concurrent.futures.as_completed(futures):
            #     cc = 0 
                # if future.result() == False:
                #     print(future.result())    
        FE_export_excels(destfolder , str(rand_number))


def SERA_folder_Thread(da_original_path, da_label_path,
                BinSize,
                isotVoxSize,
                isotVoxSize2D,
                DataType,
                DiscType,
                qntz,
                VoxInterp,
                ROIInterp,
                isScale,
                isGLround,
                isReSegRng,
                isOutliers,
                isQuntzStat,
                isIsot2D,
                ReSegIntrvl01,
                ReSegIntrvl02,
                ROI_PV,
                IVH_Type,
                IVH_DiscCont,
                IVH_binSize,
                ROIsPerImg,
                isROIsCombined,
                Feats2out,
                destfolder,rand_number,sucsessList,co,co2,result,roi_segments,obj):

    Fixed_fullpath = os.path.join(da_original_path,co)
    da_original =  readimage(Fixed_fullpath)
    
    
    
    Moved_fullpath = os.path.join(da_label_path,co2)
    da_label =  readimage(Moved_fullpath)
    
    try:
        if isinstance(da_original[0], np.ndarray) & isinstance(da_label[0], np.ndarray):
        
        # if da_original[0] != None & da_label[0] != None:  

            if da_original[2] == 'Nifti' or da_original[2] == 'Dicom':
                perm = (2, 1, 0)
                da_original[0] = np.transpose(da_original[0], perm)
            if da_label[2] == 'Nifti' or da_label[2] == 'Dicom':
                perm = (2, 1, 0)
                da_label[0] = np.transpose(da_label[0], perm)  

    
            dim01 = np.shape(da_original[0])
            dim02 = np.shape(da_label[0])
            if (dim01 == dim02):
                    
                    
                
                if len(dim01) == 3 & len(dim02) == 3 :   
    
                    if da_original[2] == 'Nifti':  
                        VoxelSize = da_original[1].GetSpacing()
    
                    if da_original[2] == 'Nrrd':  
                        VoxelSize = (da_original[1]['space directions'][0,0],da_original[1]['space directions'][1,1],da_original[1]['space directions'][2,2])
    
                    if da_original[2] == 'Dicom':  
                        VoxelSize = da_original[1].GetSpacing()     
                

                    if da_label[2] == 'Nifti':  
                        VoxelSizelabel = da_label[1].GetSpacing()
                    if da_label[2] == 'Nrrd':  
                        VoxelSizelabel = (da_label[1]['space directions'][0,0],da_label[1]['space directions'][1,1],da_label[1]['space directions'][2,2])
                    if da_label[2] == 'Dicom':  
                        VoxelSizelabel = da_label[1].GetSpacing()     
                    
                    if np.all(np.around(VoxelSizelabel, 3) == np.around(VoxelSize, 3)):

                        if CheckGTV(da_label[0]):           
                            
                            isotVoxSize = float(isotVoxSize)
                            isotVoxSize2D = float(isotVoxSize2D)
                            isScale = int(isScale)
                            isGLround = int(isGLround)
                            isReSegRng = int(isReSegRng)
                            isOutliers = int(isOutliers)
                            isQuntzStat = int(isQuntzStat)
                            isIsot2D = int(isIsot2D)
                            ReSegIntrvl = [int(ReSegIntrvl01), int(ReSegIntrvl02)]
                            ROI_PV = float(ROI_PV)
                            IVH_Type = int(IVH_Type)
                            IVH_DiscCont = int(IVH_DiscCont)
                            IVH_binSize = float(IVH_binSize)
                            ROIsPerImg = int(ROIsPerImg)
                            isROIsCombined = int(isROIsCombined)
                            Feats2out = int(Feats2out)
                            IVHconfig = [IVH_Type, IVH_DiscCont, IVH_binSize]
                    
                    
                            
                            
                            SERA_main_process(da_original, VoxelSize, da_label,
                                        BinSize,
                                        isotVoxSize,
                                        isotVoxSize2D,
                                        DataType,
                                        DiscType,
                                        qntz,
                                        VoxInterp,
                                        ROIInterp,
                                        isScale,
                                        isGLround,
                                        isReSegRng,
                                        isOutliers,
                                        isQuntzStat,
                                        isIsot2D,
                                        ReSegIntrvl,
                                        ROI_PV,
                                        IVH_Type,
                                        IVH_DiscCont,
                                        IVH_binSize,
                                        ROIsPerImg,
                                        isROIsCombined,
                                        Feats2out,
                                        IVHconfig,
                                        destfolder,rand_number,sucsessList,roi_segments,obj)
                        else:
                            raise('Label Image is Corrupted or Empty.') 
                    else:
                        raise('Images voxel size must be same.') 
                else:
                    raise('Images must be 3D.')
            else:
                raise('Dimension of original and segmented image not equal.')
        else:
            raise('You must use an approprate type of input.')
    except Exception as e:
        raise('Out of Memory or the parameters of radiomic feature generator tool should be selected properly:', e)



# def Save_Output(BinSize,
#                       isotVoxSize,
#                       isotVoxSize2D,
#                       DataType,
#                       DiscType,
#                       qntz,
#                       VoxInterp,
#                       ROIInterp,
#                       isScale,
#                       isGLround,
#                       isReSegRng,
#                       isOutliers,
#                       isQuntzStat,
#                       isIsot2D,
#                       ReSegIntrvl,
#                       ROI_PV,
#                       IVH_Type,
#                       IVH_DiscCont,
#                       IVH_binSize,
#                       ROIsPerImg,
#                       isROIsCombined,
#                       Feats2out,
#                       IVHconfig,rand_number):

#     if DataType == 'CT / SPECT':
#         DataType = 'CT'
#     # if DataType == 'MR':
#     #     DataType = 'MRscan'
    
    
#     user_inputs = {"BinSize": BinSize, "isotVoxSize": isotVoxSize, "isotVoxSize2D": isotVoxSize2D,
#                    "DataType": DataType,
#                    "DiscType": DiscType,
#                    "qntz": qntz,
#                    "VoxInterp": VoxInterp,
#                    "ROIInterp": ROIInterp,
#                    "isScale": isScale,
#                    "isGLround": isGLround,
#                    "isReSegRng": isReSegRng,
#                    "isOutliers": isOutliers,
#                    "isQuntzStat": isQuntzStat,
#                    "isIsot2D": isIsot2D,
#                    "ReSegIntrvl": ReSegIntrvl,
#                    "ROI_PV": ROI_PV,
#                    "IVH_Type": IVH_Type,
#                    "IVH_DiscCont": IVH_DiscCont,
#                    "IVH_binSize": IVH_binSize,
#                    "ROIsPerImg": ROIsPerImg,
#                    "isROIsCombined": isROIsCombined,
#                    "Feats2out": Feats2out,
#                    "IVHconfig": IVHconfig
#                    }
#     # fullpath = os.path.join(destfolder, "user_inputs.mat")
#     fullpath = "user_inputs_"+str(rand_number)+".mat"
#     # fullpath = "user_inputs.mat"
#     savemat(fullpath, user_inputs)




def SERA_main_process_Once(da_original, VoxelSize, da_label,
                      BinSize,
                      isotVoxSize,
                      isotVoxSize2D,
                      DataType,
                      DiscType,
                      qntz,
                      VoxInterp,
                      ROIInterp,
                      isScale,
                      isGLround,
                      isReSegRng,
                      isOutliers,
                      isQuntzStat,
                      isIsot2D,
                      ReSegIntrvl,
                      ROI_PV,
                      IVH_Type,
                      IVH_DiscCont,
                      IVH_binSize,
                      ROIsPerImg,
                      isROIsCombined,
                      Feats2out,
                      IVHconfig,
                      destfolder,rand_number):

    
    
    # print(type(da_original[0]))
    # print('fffff = ',np.issubdtype(da_original[0].dtype, np.floating))
    
    if np.issubdtype(da_original[0].dtype, np.floating) == True:
        # ct_min = np.min(da_original[0])
        ct_max = np.max(da_original[0])
        
        if ct_max <= 1:
            da_original[0] = da_original[0] * 255
    
        da_original[0] = da_original[0].astype(int)


    Data_inputs = {"Data_Orginal": da_original[0]}
    # fullpath = os.path.join(destfolder, "Data_Orginal.mat")
    fullpath = "Data_Orginal_"+da_original[3]+"_"+str(rand_number)+".mat"
    # fullpath = "Data_Orginal.mat"
    
    savemat(fullpath, Data_inputs)

    voxel_inputs = {"VoxelSizeInfo": VoxelSize}
    # fullpath = os.path.join(destfolder, "VoxelSizeInfo.mat")
    
    fullpath = "VoxelSizeInfo_"+da_original[3]+"_"+str(rand_number)+".mat"
    # fullpath =  "VoxelSizeInfo.mat"
    savemat(fullpath, voxel_inputs)

    Data_ROI_np = da_label[0]
    AllROIs = []
    ROIMax = np.max(Data_ROI_np)
    number_of_ROI = ROIsPerImg if ROIsPerImg < ROIMax else ROIMax
    for ijk in range(0, number_of_ROI):
        Data_ROI_aa = Data_ROI_np == (ijk+1)
        Data_ROI_aa = Data_ROI_aa.astype(int)
        labeled_array, num_features = ndimage.label(Data_ROI_aa)

        ROIs = []
        number_of_ROI_per_ROI = num_features
        # number_of_ROI_per_ROI = ROIsPerImg if ROIsPerImg < num_features else num_features
        for jk in range(0, number_of_ROI_per_ROI):
            arr = labeled_array == (jk+1)
            arr = arr.astype(int)
            volume = ndimage.sum(arr)
            ROIs.append((arr, volume))

        ROIs.sort(key=lambda x: x[1], reverse=True)
        ROIs = ROIs[:ROIsPerImg]
        AllROIs.append(ROIs)

    current_List = []
    count = 1
    for mn in AllROIs:
        count2 = 1
        for mn2 in mn:
            mn3 = mn2[0].astype(np.uint8)
            ROI_inputs = {"ROI": mn3, "name": str(count)+"_"+str(count2)}
            current_List.append(ROI_inputs)
            count2 = count2+1
        count = count+1

    current_Tuple = tuple(current_List)
    data = {'Data_ROI': current_Tuple}
    # fullpath = os.path.join(destfolder, "Data_ROI.mat")
    
    fullpath = "Data_ROI_"+da_original[3]+"_"+str(rand_number)+".mat"
    # fullpath = "Data_ROI.mat"
    savemat(fullpath, data, oned_as='column')


    if DataType == 'CT / SPECT':
        DataType = 'CT'
    # if DataType == 'MR':
    #     DataType = 'MRscan'
    
    
    user_inputs = {"BinSize": BinSize, "isotVoxSize": isotVoxSize, "isotVoxSize2D": isotVoxSize2D,
                   "DataType": DataType,
                   "DiscType": DiscType,
                   "qntz": qntz,
                   "VoxInterp": VoxInterp,
                   "ROIInterp": ROIInterp,
                   "isScale": isScale,
                   "isGLround": isGLround,
                   "isReSegRng": isReSegRng,
                   "isOutliers": isOutliers,
                   "isQuntzStat": isQuntzStat,
                   "isIsot2D": isIsot2D,
                   "ReSegIntrvl": ReSegIntrvl,
                   "ROI_PV": ROI_PV,
                   "IVH_Type": IVH_Type,
                   "IVH_DiscCont": IVH_DiscCont,
                   "IVH_binSize": IVH_binSize,
                   "ROIsPerImg": ROIsPerImg,
                   "isROIsCombined": isROIsCombined,
                   "Feats2out": Feats2out,
                   "IVHconfig": IVHconfig
                   }
    # fullpath = os.path.join(destfolder, "user_inputs.mat")
    fullpath = "user_inputs_"+da_original[3]+"_"+str(rand_number)+".mat"
    # fullpath = "user_inputs.mat"
    savemat(fullpath, user_inputs)

    # winView = r"SERA\for_redistribution_files_only"
    # timestamp1 = time.time()
    # os.startfile('SERA_Feature_Extraction.exe')
    # os.system('SERA_Feature_Extraction.exe '+da_original[3]+'')

    # timestamp2 = time.time()
    # print("This took %.2f seconds", (timestamp2 - timestamp1))
    
    # matfilesss = os.listdir('')
    # print(matfilesss)
    
    # os.remove("Data_Orginal_"+ da_original[3]+".mat")
    # os.remove("Data_ROI_"+ da_original[3]+".mat")
    # os.remove("user_inputs_"+ da_original[3]+".mat")
    # os.remove("VoxelSizeInfo_"+ da_original[3]+".mat")
    # matFilename = "All_extracted_features_"+da_original[3]+".mat"
    # matFilename_path = "All_extracted_features_"+ da_original[3]+"_"+str(rand_number)+".mat"
    # cwd = os.getcwd()
    
    # if  os.path.exists(matFilename) == True:  
    #     if rand_number >= 0:
    #         # matFilename = "All_extracted_features_"+ da_original[3]+".mat"
    #         # matFilename_path = "All_extracted_features_"+ da_original[3]+"-"+str(rand_number)+".mat"
        
    #         fullpath = os.path.join(destfolder, matFilename_path)
    #         shutil.move(matFilename, fullpath)
        
    #     lock = threading.Lock()
    #     lock.acquire()
    #     sucsessList.append((matFilename_path,True))
    #     lock.release()
    #     return True    
    # else:  
    #     print('Out of Memory or the radiomics feature generator tool can not work on',da_original[3],', check your input or hyperparameter.')
    #     lock = threading.Lock()
    #     lock.acquire()
    #     sucsessList.append((matFilename_path,False))
    #     lock.release()  
    #     return False



# def SERA_folder(da_original_path, da_label_path,
#                 BinSize,
#                 isotVoxSize,
#                 isotVoxSize2D,
#                 DataType,
#                 DiscType,
#                 qntz,
#                 VoxInterp,
#                 ROIInterp,
#                 isScale,
#                 isGLround,
#                 isReSegRng,
#                 isOutliers,
#                 isQuntzStat,
#                 isIsot2D,
#                 ReSegIntrvl01,
#                 ReSegIntrvl02,
#                 ROI_PV,
#                 IVH_Type,
#                 IVH_DiscCont,
#                 IVH_binSize,
#                 ROIsPerImg,
#                 isROIsCombined,
#                 Feats2out,
#                 destfolder):
    
#     Fixed_datas = os.listdir(da_original_path)
#     fixed = [i for i in Fixed_datas if (i.endswith(".nrrd") | i.endswith(".dcm") | i.endswith(".dicom") | i.endswith(".nii") | i.endswith(".nii.gz"))]
    
#     Moved_datas = os.listdir(da_label_path)
#     moved = [i for i in Moved_datas if (i.endswith(".nrrd") | i.endswith(".dcm") | i.endswith(".dicom") | i.endswith(".nii") | i.endswith(".nii.gz"))]
    
#     rand_number = randint(1, 10000) * 19
#     # thread_list = []
    
#     contin = True 
#     try:
#         # BinSize_values = BinSize.split(',')
#         # BinSize_values = [int(mi) for mi in BinSize_values]
#         # # print(BinSize_values)
        
#         # if len(BinSize_values) == 1:
#         #     BinSize = BinSize_values[0]
#         #     BinSize = int(BinSize)
#         # else:
#         #     BinSize = BinSize_values
#         #     BinSize = [int(mi) for mi in BinSize]
    
#         # contin = True        
#         # print(BinSize)  
#         # 
#         if BinSize != '':
#             BinSize_values = BinSize.split(',')
#             BinSize_values2 = []
#             for mi in BinSize_values:
#                 if mi != '':
#                     BinSize_values2.append(int(mi))
#             # print(BinSize_values2)

#             if len(BinSize_values2) == 1:
#                 BinSize = BinSize_values2[0]
#                 BinSize = int(BinSize)
#             elif len(BinSize_values2) == 0:
#                 BinSize = int(32) 
#             else:   
#                 BinSize = []
#                 for mi in BinSize_values2:
#                     if mi != '':
#                         BinSize.append(int(mi)) 
#             # if len(BinSize) == 0 :
#             #     BinSize = int(32)            
#         else:
#             BinSize = int(32)
         
#     except:
#         # print('Bin size input is invalid.')
#         BinSize = int(32)
        
        
#     if contin == True: 
#                     # exit()
#         sucsessList = []
#         # Num_worker = 1 + int(psutil.virtual_memory()[1]/pow(10,9)/8)
#         # Num_worker = 1
#         Num_worker = int(psutil.virtual_memory()[1]/pow(10,9)/8)
#         if Num_worker == 0:
#             Num_worker = 1
#         with concurrent.futures.ThreadPoolExecutor(max_workers=Num_worker) as executor:
#             futures = []
#             for co in fixed:
#                 if len(moved) > 0:
                        
#                     fixed_filename =  co.replace('.nii.gz', '').replace('.nii', '').replace('.nrrd', '').replace('.dcm', '').replace('.dicom','')        
                    
#                     s_ratio = 0
#                     count = 0
#                     selected_index = 0
#                     for co_moved in moved:
#                         moved_filename =  co_moved.replace('.nii.gz', '').replace('.nii', '').replace('.nrrd', '').replace('.dcm', '').replace('.dicom','')        
#                         sim_ratio = similar(fixed_filename, moved_filename)
                        
#                         if s_ratio < sim_ratio:
#                             selected_index = count
#                             s_ratio = sim_ratio
#                         count = count +1
                    
#                     if s_ratio > 0 :
            
                        
#                         Fixed_fullpath = os.path.join(da_original_path,co)
#                         da_original =  readimage(Fixed_fullpath)
                        
                        
                        
#                         Moved_fullpath = os.path.join(da_label_path,moved[selected_index])
#                         da_label =  readimage(Moved_fullpath)
                        
#                         try:
#                             if isinstance(da_original[0], np.ndarray) & isinstance(da_label[0], np.ndarray):
                            
#                             # if da_original[0] != None & da_label[0] != None:  
            
#                                 if da_original[2] == 'Nifti' or da_original[2] == 'Dicom':
#                                     perm = (2, 1, 0)
#                                     da_original[0] = np.transpose(da_original[0], perm)
#                                 if da_label[2] == 'Nifti' or da_label[2] == 'Dicom':
#                                     perm = (2, 1, 0)
#                                     da_label[0] = np.transpose(da_label[0], perm)  

                        
#                                 dim01 = np.shape(da_original[0])
#                                 dim02 = np.shape(da_label[0])
#                                 if (dim01 == dim02):
                                        
                                        
                                    
#                                     if len(dim01) == 3 & len(dim02) == 3 :   
                        
#                                         if da_original[2] == 'Nifti':  
#                                             VoxelSize = da_original[1].GetSpacing()
                        
#                                         if da_original[2] == 'Nrrd':  
#                                             VoxelSize = (da_original[1]['space directions'][0,0],da_original[1]['space directions'][1,1],da_original[1]['space directions'][2,2])
                        
#                                         if da_original[2] == 'Dicom':  
#                                             VoxelSize = da_original[1].GetSpacing()     
                                    

#                                         if da_label[2] == 'Nifti':  
#                                             VoxelSizelabel = da_label[1].GetSpacing()
#                                         if da_label[2] == 'Nrrd':  
#                                             VoxelSizelabel = (da_label[1]['space directions'][0,0],da_label[1]['space directions'][1,1],da_label[1]['space directions'][2,2])
#                                         if da_label[2] == 'Dicom':  
#                                             VoxelSizelabel = da_label[1].GetSpacing()     
                                        
#                                         if VoxelSizelabel == VoxelSize:

#                                             if CheckGTV(da_label[0]):           
#                                                 # BinSize = int(BinSize)
#                                                 # isotVoxSize = int(isotVoxSize)
#                                                 # isotVoxSize2D = int(isotVoxSize2D)
#                                                 # isScale = int(isScale)
#                                                 # isGLround = int(isGLround)
#                                                 # isReSegRng = int(isReSegRng)
#                                                 # isOutliers = int(isOutliers)
#                                                 # isQuntzStat = int(isQuntzStat)
#                                                 # isIsot2D = int(isIsot2D)
#                                                 # ReSegIntrvl = [int(ReSegIntrvl01), int(ReSegIntrvl02)]
#                                                 # ROI_PV = float(ROI_PV)
#                                                 # IVH_Type = int(IVH_Type)
#                                                 # IVH_DiscCont = int(IVH_DiscCont)
#                                                 # IVH_binSize = int(IVH_binSize)
#                                                 # ROIsPerImg = int(ROIsPerImg)
#                                                 # isROIsCombined = int(isROIsCombined)
#                                                 # Feats2out = int(Feats2out)
#                                                 # IVHconfig = [IVH_Type, IVH_DiscCont, IVH_binSize]
                                        
                                                
                                                    
                                                
#                                                 # BinSize = int(BinSize)
#                                                 isotVoxSize = float(isotVoxSize)
#                                                 isotVoxSize2D = float(isotVoxSize2D)
#                                                 isScale = int(isScale)
#                                                 isGLround = int(isGLround)
#                                                 isReSegRng = int(isReSegRng)
#                                                 isOutliers = int(isOutliers)
#                                                 isQuntzStat = int(isQuntzStat)
#                                                 isIsot2D = int(isIsot2D)
#                                                 ReSegIntrvl = [int(ReSegIntrvl01), int(ReSegIntrvl02)]
#                                                 ROI_PV = float(ROI_PV)
#                                                 IVH_Type = int(IVH_Type)
#                                                 IVH_DiscCont = int(IVH_DiscCont)
#                                                 IVH_binSize = float(IVH_binSize)
#                                                 ROIsPerImg = int(ROIsPerImg)
#                                                 isROIsCombined = int(isROIsCombined)
#                                                 Feats2out = int(Feats2out)
#                                                 IVHconfig = [IVH_Type, IVH_DiscCont, IVH_binSize]
                                        
#                                                 para = {
#                                                 'Parameter':['Bin Size/Width', '3D Isotropic Voxel Size Flag', '2D Isotropic Voxel Size Flag', 'Image Modality Type',
#                                                         'Discretization Type', 'Quantization Type', 'Image Resampling Interpolation Type', 'ROI Resampling Interpolation Type',
#                                                         'Scale (Resampling) Flag', 'Round Voxel Intensity Values Flag', 'Range Re-Segmentation Flag', 
#                                                         'Intensity Outlier Re-Segmentation Flag','Image Quantization Flag','Isotropic 2D Voxels Flag',
#                                                         'Re-Segmentation Interval Range','ROI Partial Volume Threshold','Intensity Volume Histogram (IVH) Type','Intensity Volume Histogram (IVH) Discretization Type',
#                                                         'Intensity Volume Histogram (IVH) Discretization Binning Option/Size',
#                                                         'Max Number Of ROIs per Image',
#                                                         'Combine Multiple ROIs To ONE Flag',
#                                                         'Type of Ouyput Data'
#                                                         ],
#                                                 'Value':[BinSize,
#                                                     isotVoxSize,
#                                                     isotVoxSize2D,
#                                                     DataType,
#                                                     DiscType,
#                                                     qntz,
#                                                     VoxInterp,
#                                                     ROIInterp,
#                                                     isScale,
#                                                     isGLround,
#                                                     isReSegRng,
#                                                     isOutliers,
#                                                     isQuntzStat,
#                                                     isIsot2D,
#                                                     ReSegIntrvl,
#                                                     ROI_PV,
#                                                     IVH_Type,
#                                                     IVH_DiscCont,
#                                                     IVH_binSize,
#                                                     ROIsPerImg,
#                                                     isROIsCombined,
#                                                     Feats2out]
#                                                 }
#                                                 result = pd.DataFrame(para)
                                                    
                                                
#                                                 futures.append(
#                                                     executor.submit(
#                                                         SERA_main_process, da_original, VoxelSize, da_label,
#                                                                     BinSize,
#                                                                     isotVoxSize,
#                                                                     isotVoxSize2D,
#                                                                     DataType,
#                                                                     DiscType,
#                                                                     qntz,
#                                                                     VoxInterp,
#                                                                     ROIInterp,
#                                                                     isScale,
#                                                                     isGLround,
#                                                                     isReSegRng,
#                                                                     isOutliers,
#                                                                     isQuntzStat,
#                                                                     isIsot2D,
#                                                                     ReSegIntrvl,
#                                                                     ROI_PV,
#                                                                     IVH_Type,
#                                                                     IVH_DiscCont,
#                                                                     IVH_binSize,
#                                                                     ROIsPerImg,
#                                                                     isROIsCombined,
#                                                                     Feats2out,
#                                                                     IVHconfig,
#                                                                     destfolder,rand_number,sucsessList
#                                                     )
#                                                 )
                                                
#                                                 if(len(moved)>0):
#                                                     moved.pop(selected_index)
#                                             else:
#                                                 print('Label Image is Corrupted or Empty.') 
#                                         else:
#                                             print('Images voxel size must be same.') 
#                                     else:
#                                         print('Images must be 3D.')
#                                 else:
#                                     print('Dimension of original and segmented image not equal.')
#                             else:
#                                 print('You must use an approprate type of input.')
#                         except:
#                             print('Out of Memory or the parameters of radiomics feature generator tool should be selected properly.')
#                     # else:
#                         # print('There is no image with the same name of',co,'in another folder.')
#             # for thread in thread_list:
#             #     thread.join()    
    
#             executor.shutdown(wait=True)
#             # print(len(futures))    
#             # for future in concurrent.futures.as_completed(futures):
#             #     cc = 0 
#                 # if future.result() == False:
#                 #     print(future.result())    
#             FE_export(destfolder , str(rand_number),Feats2out,result,BinSize)




def SERA_folder_Once(da_original_path, da_label_path,
                BinSize,
                isotVoxSize,
                isotVoxSize2D,
                DataType,
                DiscType,
                qntz,
                VoxInterp,
                ROIInterp,
                isScale,
                isGLround,
                isReSegRng,
                isOutliers,
                isQuntzStat,
                isIsot2D,
                ReSegIntrvl01,
                ReSegIntrvl02,
                ROI_PV,
                IVH_Type,
                IVH_DiscCont,
                IVH_binSize,
                ROIsPerImg,
                isROIsCombined,
                Feats2out,
                destfolder):
    
    Fixed_datas = os.listdir(da_original_path)
    fixed = [i for i in Fixed_datas if (i.endswith(".nrrd") | i.endswith(".dcm") | i.endswith(".dicom") | i.endswith(".nii") | i.endswith(".nii.gz"))]
    
    Moved_datas = os.listdir(da_label_path)
    moved = [i for i in Moved_datas if (i.endswith(".nrrd") | i.endswith(".dcm") | i.endswith(".dicom") | i.endswith(".nii") | i.endswith(".nii.gz"))]
    
    rand_number = randint(1, 10000) * 19
    
    contin = True 
    try:

        if BinSize != '':
            BinSize_values = BinSize.split(',')
            BinSize_values2 = []
            for mi in BinSize_values:
                if mi != '':
                    BinSize_values2.append(int(mi))

            if len(BinSize_values2) == 1:
                BinSize = BinSize_values2[0]
                BinSize = int(BinSize)
            elif len(BinSize_values2) == 0:
                BinSize = int(32) 
            else:   
                BinSize = []
                for mi in BinSize_values2:
                    if mi != '':
                        BinSize.append(int(mi))            
        else:
            BinSize = int(32)
         
    except:
        # print('Bin size input is invalid.')
        BinSize = int(32)
    
    # BinSize = int(BinSize)
    isotVoxSize = float(isotVoxSize)
    isotVoxSize2D = float(isotVoxSize2D)
    isScale = int(isScale)
    isGLround = int(isGLround)
    isReSegRng = int(isReSegRng)
    isOutliers = int(isOutliers)
    isQuntzStat = int(isQuntzStat)
    isIsot2D = int(isIsot2D)
    ReSegIntrvl = [int(ReSegIntrvl01), int(ReSegIntrvl02)]
    ROI_PV = float(ROI_PV)
    IVH_Type = int(IVH_Type)
    IVH_DiscCont = int(IVH_DiscCont)
    IVH_binSize = float(IVH_binSize)
    ROIsPerImg = int(ROIsPerImg)
    isROIsCombined = int(isROIsCombined)
    Feats2out = int(Feats2out)
    IVHconfig = [IVH_Type, IVH_DiscCont, IVH_binSize]

    para = {
    'Parameter':['Bin Size/Width', '3D Isotropic Voxel Size Flag', '2D Isotropic Voxel Size Flag', 'Image Modality Type',
            'Discretization Type', 'Quantization Type', 'Image Resampling Interpolation Type', 'ROI Resampling Interpolation Type',
            'Scale (Resampling) Flag', 'Round Voxel Intensity Values Flag', 'Range Re-Segmentation Flag', 
            'Intensity Outlier Re-Segmentation Flag','Image Quantization Flag','Isotropic 2D Voxels Flag',
            'Re-Segmentation Interval Range','ROI Partial Volume Threshold','Intensity Volume Histogram (IVH) Type','Intensity Volume Histogram (IVH) Discretization Type',
            'Intensity Volume Histogram (IVH) Discretization Binning Option/Size',
            'Max Number Of ROIs per Image',
            'Combine Multiple ROIs To ONE Flag',
            'Type of Ouyput Data'
            ],
    'Value':[BinSize,
        isotVoxSize,
        isotVoxSize2D,
        DataType,
        DiscType,
        qntz,
        VoxInterp,
        ROIInterp,
        isScale,
        isGLround,
        isReSegRng,
        isOutliers,
        isQuntzStat,
        isIsot2D,
        ReSegIntrvl,
        ROI_PV,
        IVH_Type,
        IVH_DiscCont,
        IVH_binSize,
        ROIsPerImg,
        isROIsCombined,
        Feats2out]
    }
    result = pd.DataFrame(para)
    

    # Save_Output(BinSize,
    #             isotVoxSize,
    #             isotVoxSize2D,
    #             DataType,
    #             DiscType,
    #             qntz,
    #             VoxInterp,
    #             ROIInterp,
    #             isScale,
    #             isGLround,
    #             isReSegRng,
    #             isOutliers,
    #             isQuntzStat,
    #             isIsot2D,
    #             ReSegIntrvl,
    #             ROI_PV,
    #             IVH_Type,
    #             IVH_DiscCont,
    #             IVH_binSize,
    #             ROIsPerImg,
    #             isROIsCombined,
    #             Feats2out,
    #             IVHconfig,rand_number)


    if contin == True: 
        dataList = []
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
                    dataList.append((co,moved[selected_index]))
                    if(len(moved)>0):
                        moved.pop(selected_index)

        # Num_worker = 1 + int(psutil.virtual_memory()[1]/pow(10,9)/8)
        # Num_worker = 1
        Num_worker = int(psutil.virtual_memory()[1]/pow(10,9))
        if Num_worker == 0:
            Num_worker = 1
        # HelthdataList = []

        if len(dataList)>0:
            # os.system('SERA_Feature_Extraction_once.exe '+str(rand_number)+'')
            import subprocess
            p = subprocess.Popen('SERA_Feature_Extraction_once.exe '+str(rand_number)+'', shell=True)  
                
        with concurrent.futures.ThreadPoolExecutor(max_workers=Num_worker) as executor:
            futures = []
            for da in dataList:
    
                futures.append(
                    executor.submit(
                        SERA_folder_Once_Folder, 
                        da_original_path, da_label_path,
                        BinSize,
                        isotVoxSize,
                        isotVoxSize2D,
                        DataType,
                        DiscType,
                        qntz,
                        VoxInterp,
                        ROIInterp,
                        isScale,
                        isGLround,
                        isReSegRng,
                        isOutliers,
                        isQuntzStat,
                        isIsot2D,
                        ReSegIntrvl,
                        ROI_PV,
                        IVH_Type,
                        IVH_DiscCont,
                        IVH_binSize,
                        ROIsPerImg,
                        isROIsCombined,
                        Feats2out,
                        IVHconfig,
                        destfolder,rand_number,da
                    )
                )
                                        

      
        executor.shutdown(wait=True)
        p.communicate()
        import re
        files = [f for f in os.listdir('.') if f.endswith('_'+str(rand_number)+'.mat')]
        for fi in files:
            matFilename_path = fi
            fullpath = os.path.join(destfolder, matFilename_path)
            shutil.move(matFilename_path, fullpath)

        FE_export(destfolder , str(rand_number),Feats2out,result,BinSize)


def SERA_folder_Once_Folder(da_original_path, da_label_path,
                BinSize,
                isotVoxSize,
                isotVoxSize2D,
                DataType,
                DiscType,
                qntz,
                VoxInterp,
                ROIInterp,
                isScale,
                isGLround,
                isReSegRng,
                isOutliers,
                isQuntzStat,
                isIsot2D,
                ReSegIntrvl,
                ROI_PV,
                IVH_Type,
                IVH_DiscCont,
                IVH_binSize,
                ROIsPerImg,
                isROIsCombined,
                Feats2out,
                IVHconfig,
                destfolder,rand_number,da):

    Fixed_fullpath = os.path.join(da_original_path,da[0])
    da_original =  readimage(Fixed_fullpath)
    
    
    Moved_fullpath = os.path.join(da_label_path,da[1])
    da_label =  readimage(Moved_fullpath)
    
    try:
        if isinstance(da_original[0], np.ndarray) & isinstance(da_label[0], np.ndarray):
                
            if da_original[2] == 'Nifti' or da_original[2] == 'Dicom':
                perm = (2, 1, 0)
                da_original[0] = np.transpose(da_original[0], perm)
            if da_label[2] == 'Nifti' or da_label[2] == 'Dicom':
                perm = (2, 1, 0)
                da_label[0] = np.transpose(da_label[0], perm)  

    
            dim01 = np.shape(da_original[0])
            dim02 = np.shape(da_label[0])
            if (dim01 == dim02):
            
                if len(dim01) == 3 & len(dim02) == 3 :   
    
                    if da_original[2] == 'Nifti':  
                        VoxelSize = da_original[1].GetSpacing()
    
                    if da_original[2] == 'Nrrd':  
                        VoxelSize = (da_original[1]['space directions'][0,0],da_original[1]['space directions'][1,1],da_original[1]['space directions'][2,2])
    
                    if da_original[2] == 'Dicom':  
                        VoxelSize = da_original[1].GetSpacing()     
                

                    if da_label[2] == 'Nifti':  
                        VoxelSizelabel = da_label[1].GetSpacing()
                    if da_label[2] == 'Nrrd':  
                        VoxelSizelabel = (da_label[1]['space directions'][0,0],da_label[1]['space directions'][1,1],da_label[1]['space directions'][2,2])
                    if da_label[2] == 'Dicom':  
                        VoxelSizelabel = da_label[1].GetSpacing()     
                    
                    if VoxelSizelabel == VoxelSize:

                        if CheckGTV(da_label[0]):           
                            
                            
                            SERA_main_process_Once(
                                da_original, VoxelSize, da_label,
                                BinSize,
                                isotVoxSize,
                                isotVoxSize2D,
                                DataType,
                                DiscType,
                                qntz,
                                VoxInterp,
                                ROIInterp,
                                isScale,
                                isGLround,
                                isReSegRng,
                                isOutliers,
                                isQuntzStat,
                                isIsot2D,
                                ReSegIntrvl,
                                ROI_PV,
                                IVH_Type,
                                IVH_DiscCont,
                                IVH_binSize,
                                ROIsPerImg,
                                isROIsCombined,
                                Feats2out,
                                IVHconfig,
                                destfolder,rand_number
                            )
                        else:
                            raise('Label Image is Corrupted or Empty.') 
                    else:
                        raise('Images voxel size must be same.') 
                else:
                    raise('Images must be 3D.')
            else:
                raise('Dimension of original and segmented image not equal.')
        else:
            raise('You must use an approprate type of input.')
    except Exception as e:
        raise('Out of Memory or the parameters of radiomic feature generator tool should be selected properly:', e)


        