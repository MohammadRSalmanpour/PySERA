# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 16:10:56 2013

@author: vterzopoulos, abrys
"""
# To ignore numpy errors:
#     pylint: disable=E1101
import nibabel
import numpy

from dicom2nifti.image_volume import load, SliceType, ImageVolume


def reorient_image(input_image, output_image):
    """
    Change the orientation of the Image data in order to be in LAS space
    x will represent the coronal plane, y the sagittal and z the axial plane.
    x increases from Right (R) to Left (L), y from Posterior (P) to Anterior (A) and z from Inferior (I) to Superior (S)

    :returns: The output image in nibabel form
    :param output_image: filepath to the nibabel image
    :param input_image: filepath to the nibabel image
    """
    # Use the imageVolume module to find which coordinate corresponds to each plane
    # and get the image data in RAS orientation
    # print 'Reading nifti'
    if isinstance(input_image, nibabel.Nifti1Image):
        image = ImageVolume(input_image)
    else:
        image = load(input_image)

    # 4d have a different conversion to 3d
    # print 'Reorganizing data'
    if image.nifti_data.squeeze().ndim == 4:
        new_image = _reorient_4d(image)
    elif image.nifti_data.squeeze().ndim == 3 or image.nifti_data.ndim == 3 or image.nifti_data.squeeze().ndim == 2:
        new_image = _reorient_3d(image)
    else:
        raise Exception('Only 3d and 4d images are supported')

    # print 'Recreating affine'
    affine = image.nifti.affine
    # Based on VolumeImage.py where slice orientation 1 represents the axial plane
    # Flipping on the data may be needed based on x_inverted, y_inverted, ZInverted

    # Create new affine header by changing the order of the columns of the input image header
    # the last column with the origin depends on the origin of the original image, the size and the direction of x,y,z

    new_affine = numpy.eye(4)
    new_affine[:, 0] = affine[:, image.sagittal_orientation.normal_component]
    new_affine[:, 1] = affine[:, image.coronal_orientation.normal_component]
    new_affine[:, 2] = affine[:, image.axial_orientation.normal_component]
    point = [0, 0, 0, 1]

    # If the orientation of coordinates is inverted, then the origin of the "new" image
    # would correspond to the last voxel of the original image
    # First we need to find which point is the origin point in image coordinates
    # and then transform it in world coordinates
    if not image.axial_orientation.x_inverted:
        new_affine[:, 0] = - new_affine[:, 0]
        point[image.sagittal_orientation.normal_component] = image.dimensions[
                                                                 image.sagittal_orientation.normal_component] - 1
        # new_affine[0, 3] = - new_affine[0, 3]
    if image.axial_orientation.y_inverted:
        new_affine[:, 1] = - new_affine[:, 1]
        point[image.coronal_orientation.normal_component] = image.dimensions[
                                                                image.coronal_orientation.normal_component] - 1
        # new_affine[1, 3] = - new_affine[1, 3]
    if image.coronal_orientation.y_inverted:
        new_affine[:, 2] = - new_affine[:, 2]
        point[image.axial_orientation.normal_component] = image.dimensions[image.axial_orientation.normal_component] - 1
        # new_affine[2, 3] = - new_affine[2, 3]

    new_affine[:, 3] = numpy.dot(affine, point)

    # DONE: Needs to update new_affine, so that there is no translation difference between the original
    # and created image (now there is 1-2 voxels translation)
    # print 'Creating new nifti image'
    if new_image.ndim > 3:  # do not squeeze single slice data
        new_image = new_image.squeeze()
    output = nibabel.nifti1.Nifti1Image(new_image, new_affine)
    output.header.set_slope_inter(1, 0)
    output.header.set_xyzt_units(2)  # set units for xyz (leave t as unknown)
    # output.to_filename(output_image)
    return output


def _reorient_4d(image):
    """
    Reorganize the data for a 4d nifti
    """
    # print 'converting 4d image'
    # Create empty array where x,y,z correspond to LR (sagittal), PA (coronal), IS (axial) directions and the size
    # of the array in each direction is the same with the corresponding direction of the input image.
    new_image = numpy.zeros([image.dimensions[image.sagittal_orientation.normal_component],
                             image.dimensions[image.coronal_orientation.normal_component],
                             image.dimensions[image.axial_orientation.normal_component],
                             image.dimensions[3]],
                            dtype=image.nifti_data.dtype)

    # loop over all timepoints
    for timepoint in range(0, image.dimensions[3]):
        # Fill the new image with the values of the input image but with mathicng the orientation with x,y,z
        if image.coronal_orientation.y_inverted:
            for i in range(new_image.shape[2]):
                new_image[:, :, i, timepoint] = numpy.fliplr(numpy.squeeze(image.get_slice(SliceType.AXIAL,
                                                                                           new_image.shape[2] - 1 - i,
                                                                                           timepoint).original_data))
        else:
            for i in range(new_image.shape[2]):
                new_image[:, :, i, timepoint] = numpy.fliplr(numpy.squeeze(image.get_slice(SliceType.AXIAL,
                                                                                           i, timepoint).original_data))

    return new_image


def _reorient_3d(image):
    """
    Reorganize the data for a 3d nifti
    """
    # Create empty array where x,y,z correspond to LR (sagittal), PA (coronal), IS (axial) directions and the size
    # of the array in each direction is the same with the corresponding direction of the input image.
    new_image = numpy.zeros([image.dimensions[image.sagittal_orientation.normal_component],
                             image.dimensions[image.coronal_orientation.normal_component],
                             image.dimensions[image.axial_orientation.normal_component]],
                            dtype=image.nifti_data.dtype)

    # Fill the new image with the values of the input image but with matching the orientation with x,y,z
    if image.coronal_orientation.y_inverted:
        for i in range(new_image.shape[2]):
            new_image[:, :, i] = numpy.fliplr(image.get_slice(SliceType.AXIAL,
                                                              new_image.shape[2] - 1 - i).original_data)
    else:
        for i in range(new_image.shape[2]):
            new_image[:, :, i] = numpy.fliplr(image.get_slice(SliceType.AXIAL,
                                                              i).original_data)

    return new_image
