# -*- coding: utf-8 -*-
"""
this module houses all the code to just convert a directory of random dicom files

@author: abrys
"""

# import compressed_dicom as compressed_dicom
# import compressed_dicom as compressed_dicom

from .compressed_dicom  import read_file, is_dicom_file 


import gc
import os
import re
import traceback
import unicodedata

from pydicom.tag import Tag

import logging
from .common import is_philips, is_multiframe_dicom 
from .convert_dicom import dicom_array_to_nifti

from .settings import *

logger = logging.getLogger(__name__)



def convert_directory(dicom_directory, output_folder, compression=True, reorient=True):
    """
    This function will order all dicom files by series and order them one by one

    :param compression: enable or disable gzip compression
    :param reorient: reorient the dicoms according to LAS orientation
    :param output_folder: folder to write the nifti files to
    :param dicom_directory: directory with dicom files
    """
    # sort dicom files by series uid
    dicom_series = {}
    filename_series = {}
    filename_path = {}

    for root, _, files in os.walk(dicom_directory):
        for dicom_file in files:
            file_path = os.path.join(root, dicom_file)
            # noinspection PyBroadException
            try:
                if is_dicom_file(file_path):
                    # read the dicom as fast as possible
                    # (max length for SeriesInstanceUID is 64 so defer_size 100 should be ok)

                    dicom_headers = read_file(file_path,
                                                    defer_size="1 KB",
                                                    stop_before_pixels=False,
                                                    force=pydicom_read_force)
                    
                    if not _is_valid_imaging_dicom(dicom_headers):
                        logger.info("Skipping: %s" % file_path)
                        continue
                    logger.info("Organizing: %s" % file_path)
                    
                    id_dicom_series = str(dicom_headers.SeriesInstanceUID)

                    if 'SequenceName' in dicom_headers:
                        id_dicom_series = id_dicom_series +"___"+ str(dicom_headers.SequenceName)
                    if 'ProtocolName' in dicom_headers:
                        id_dicom_series = id_dicom_series +"___"+ str(dicom_headers.ProtocolName)
                    # if 'InstanceNumber' in dicom_headers:
                    #     id_dicom_series = id_dicom_series +"___"+ str(dicom_headers.InstanceNumber)

                    if id_dicom_series not in dicom_series:
                        dicom_series[id_dicom_series] = []
                        filename_series[id_dicom_series] = []
                        filename_path[id_dicom_series] = []

                    dicom_series[id_dicom_series].append(dicom_headers)                  
                    filename_series[id_dicom_series].append(dicom_file)
                    filename_path[id_dicom_series].append(file_path)



            except:  # Explicitly capturing all errors here to be able to continue processing all the rest
                logger.warning("Unable to read: %s" % file_path)
                traceback.print_exc()

    listOfResults = []
    listOfResults_filename = []
    listOfResults_filepath = []
    listOfFirstHeader = []
    listOfImagePositionPatient = []
    listOfImageOrientationPatient = []
    listOfStep = []
    listOfPixelSpacing = []

    # start converting one by one
    for series_id, dicom_input in dicom_series.items():
        base_filename = ""
        # noinspection PyBroadException
        try:
            # construct the filename for the nifti
            base_filename = ""
            if 'SeriesNumber' in dicom_input[0]:
                base_filename = _remove_accents('%s' % dicom_input[0].SeriesNumber)
                if 'SeriesDescription' in dicom_input[0]:
                    base_filename = _remove_accents('%s_%s' % (base_filename,
                                                               dicom_input[0].SeriesDescription))
                elif 'SequenceName' in dicom_input[0]:
                    base_filename = _remove_accents('%s_%s' % (base_filename,
                                                               dicom_input[0].SequenceName))
                elif 'ProtocolName' in dicom_input[0]:
                    base_filename = _remove_accents('%s_%s' % (base_filename,
                                                               dicom_input[0].ProtocolName))
            else:
                base_filename = _remove_accents(dicom_input[0].SeriesInstanceUID)
                
                
            logger.info('--------------------------------------------')
            logger.info('Start converting %s' % base_filename)
            if compression:
                # nifti_file = os.path.join(output_folder, base_filename + '.nii.gz')
                nifti_file = base_filename + '.nii.gz'
            else:
                # nifti_file = os.path.join(output_folder, base_filename + '.nii')
                nifti_file = base_filename + '.nii' 

            if dicom_input[0].Rows > 0 and dicom_input[0].Columns > 0:     
                results = dicom_array_to_nifti(dicom_input, nifti_file, reorient)

                seq_name = ""
                if 'SequenceName' in dicom_input[0]:
                    seq_name = _remove_accents('%s_%s' % (seq_name, dicom_input[0].SequenceName))
                # if 'InstanceNumber' in dicom_input[0]:
                #     seq_name = _remove_accents('%s_%s' % (seq_name, dicom_input[0].InstanceNumber))
                if 'ProtocolName' in dicom_input[0]:
                    seq_name = _remove_accents('%s_%s' % (seq_name, dicom_input[0].ProtocolName))

                if seq_name == "":
                    results['filename'] = base_filename
                else:
                    results['filename'] = base_filename + "___"+ seq_name


                results['filepath'] = filename_path[series_id][0]
                gc.collect()
                listOfResults.append(results['NII'])  
                listOfResults_filename.append(results['filename'])  
                listOfResults_filepath.append(results['filepath'])  
                listOfFirstHeader.append(dicom_input[0])
                listOfImagePositionPatient.append(results['ImagePositionPatient'])  
                listOfImageOrientationPatient.append(results['ImageOrientationPatient']) 
                listOfStep.append(results['step'])  
                listOfPixelSpacing.append(results['PixelSpacing'])  
            else:
                logger.info("Unable to convert: %s" % base_filename)
                traceback.print_exc()
                # return None
                listOfResults.append(None)   
                listOfResults_filename.append(None)  
                listOfResults_filepath.append(None)  
                listOfFirstHeader.append(None)
                listOfImagePositionPatient.append(None)  
                listOfImageOrientationPatient.append(None)  
                listOfStep.append(None)  
                listOfPixelSpacing.append(None) 
            # print(results)
            # return results['NII']

        except:  # Explicitly capturing app exceptions here to be able to continue processing
            logger.info("Unable to convert: %s" % base_filename)
            traceback.print_exc()
            # return None
            listOfResults.append(None)   
            listOfResults_filename.append(None)  
            listOfResults_filepath.append(None)  
            listOfFirstHeader.append(None)
            listOfImagePositionPatient.append(None)  
            listOfImageOrientationPatient.append(None)  
            listOfStep.append(None)  
            listOfPixelSpacing.append(None)  


    return listOfResults,listOfResults_filename,listOfResults_filepath,listOfFirstHeader,listOfImagePositionPatient,listOfImageOrientationPatient,listOfStep,listOfPixelSpacing

def _is_valid_imaging_dicom(dicom_header):
    """
    Function will do some basic checks to see if this is a valid imaging dicom
    """
    # if it is philips and multiframe dicom then we assume it is ok
    try:
        if is_philips([dicom_header]):
            if is_multiframe_dicom([dicom_header]):
                return True

        if "SeriesInstanceUID" not in dicom_header:
            return False

        if "InstanceNumber" not in dicom_header:
            return False

        if "ImageOrientationPatient" not in dicom_header or len(dicom_header.ImageOrientationPatient) < 6:
            return False

        if "ImagePositionPatient" not in dicom_header or len(dicom_header.ImagePositionPatient) < 3:
            return False

        # for all others if there is image position patient we assume it is ok
        if Tag(0x0020, 0x0037) not in dicom_header:
            return False

        return True
    except (KeyError, AttributeError):
        return False


def _remove_accents(unicode_filename):
    """
    Function that will try to remove accents from a unicode string to be used in a filename.
    input filename should be either an ascii or unicode string
    """
    # noinspection PyBroadException
    try:
        unicode_filename = unicode_filename.replace(" ", "_")
        cleaned_filename = unicodedata.normalize('NFKD', unicode_filename).encode('ASCII', 'ignore').decode('ASCII')

        cleaned_filename = re.sub(r'[^\w\s-]', '', cleaned_filename.strip().lower())
        cleaned_filename = re.sub(r'[-\s]+', '-', cleaned_filename)

        return cleaned_filename
    except:
        traceback.print_exc()
        return unicode_filename


def _remove_accents_(unicode_filename):
    """
    Function that will try to remove accents from a unicode string to be used in a filename.
    input filename should be either an ascii or unicode string
    """
    valid_characters = bytes(b'-_.() 1234567890abcdefghijklmnopqrstuvwxyz')
    cleaned_filename = unicodedata.normalize('NFKD', unicode_filename).encode('ASCII', 'ignore')

    new_filename = ""

    for char_int in bytes(cleaned_filename):
        char_byte = bytes([char_int])
        if char_byte in valid_characters:
            new_filename += char_byte.decode()

    return new_filename
