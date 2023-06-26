
from pickle import NONE
from scipy.io import savemat, loadmat
import numpy as np
import scipy.ndimage as ndi
import SimpleITK as sitk

# -------------------------------------------------------------------------
# function [Vnew] = imresize3D(V,scale,tsize,ntype,npad,new_spacing,isIsot2D)
# -------------------------------------------------------------------------
# DESCRIPTION:
# This function resizes a 3D input volume to new dimensions.
# -------------------------------------------------------------------------
# INPUTS:
# - V: The input volume (3D array).
# - scale: scaling factor, when used set tsize to [].
# - nsize: new dimensions, when used set scale to [].
# - ntype: Type of interpolation ('nearest', 'linear', or 'cubic')
# - npad: Boundary condition ('replicate', 'symmetric', 'circular', 'fill',
#         or 'bound')
# -------------------------------------------------------------------------
# OUTPUTS:
# - Vnew: Resized volume.
# -------------------------------------------------------------------------
# AUTHOR(S): 
# - Mahdi Hosseinzadeh
# -------------------------------------------------------------------------
# HISTORY:
# - Creation: December 2022
#--------------------------------------------------------------------------



def imresize3D(V,scale,tsize,ntype,npad,new_spacing,isIsot2D):

    # I = np.eye(4)
    # print(I)s
    # I[0,0] = scale[0]
    # I[1,1] = scale[1]
    # I[2,2] = scale[2]
    # print(I)
    # Vnew = imwarp(V,I,ntype)
    # Vnew = skimage.transform.warp(V,I,ntype)
    # Read order of multidimensional spline filter (0=nearest neighbours, 1=linear, 3=cubic)
    # Vnew = translate_image(V, I, ntype=ntype)

    new_spacing = np.array(new_spacing)
    # scale = np.array(scale , dtype=np.float32).flatten(order='F')
    new_spacing = new_spacing.astype(np.float32)
    # scale = scale.astype(np.float32)
    ntype = ntype.lower()
    if ntype == 'nearest':
        order = 0
    elif ntype == 'linear':
        order = 1
    elif ntype == 'cubic':
        order = 3
    else:
        order = 0


    
    if isIsot2D == 1:
        new_spacing[2] = scale[2]

    Vnew = interpolate_to_new_grid(
                            orig_dim=np.asarray(V.shape),
                            orig_spacing=np.asarray(scale).flatten(order='F'),
                            orig_vox=V,
                            grid_origin = None,
                            sample_spacing = np.asarray(new_spacing).flatten(order='F'),
                            order=order,
                            mode=npad,
                            align_to_center=True
                            )



    # orientation = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    # m_affine = np.zeros((3, 3), dtype=float)
    # m_affine[:, 0] = scale[0] * np.array([orientation[0], orientation[1], orientation[2]])
    # m_affine[:, 1] = scale[1] * np.array([orientation[3], orientation[4], orientation[5]])
    # m_affine[:, 2] = scale[2] * np.array([orientation[6], orientation[7], orientation[8]])
    # m_affine_inv = np.linalg.inv(m_affine)


    # interpolate_to_new_grid(
    #                     # sample_dim=img_obj.size,
    #                     grid_origin=np.dot(m_affine_inv, np.transpose(img_obj.origin - self.roi.origin)),
    #                     align_to_center=False
    #                     )

    # Vnew = interpolate(
    #                 vol_dim=np.asarray(V.shape),
    #                 vol_spacing=np.asarray(scale).flatten(),
    #                 vol=V,
    #                 grid_origin = None,
    #                 new_dim = np.asarray(tsize).flatten(),
    #                 new_spacing = np.asarray(new_spacing).flatten(),
    #                 order=order,
    #                 npad=npad)
    
    

    return Vnew

# -------------------------------------------------------------------------
# function [Vnew] = imresize(V,scale,tsize,ntype,npad,new_spacing,isIsot2D)
# -------------------------------------------------------------------------
# DESCRIPTION:
# This function resizes a 2D input volume to new dimensions.
# -------------------------------------------------------------------------
# INPUTS:
# - V: The input volume (2D array).
# - scale: scaling factor, when used set tsize to [].
# - nsize: new dimensions, when used set scale to [].
# - ntype: Type of interpolation ('nearest', 'linear', or 'cubic')
# - npad: Boundary condition ('replicate', 'symmetric', 'circular', 'fill',
#         or 'bound')
# -------------------------------------------------------------------------
# OUTPUTS:
# - Vnew: Resized volume.
# -------------------------------------------------------------------------
# AUTHOR(S): 
# - Mahdi Hosseinzadeh
# -------------------------------------------------------------------------
# HISTORY:
# - Creation: December 2022
#--------------------------------------------------------------------------

def imresize(V,scale,tsize,ntype,new_spacing):

    new_spacing[2] = scale[2]
    new_spacing = np.array(new_spacing)
    # scale = np.array(scale,dtype=np.float32)
    new_spacing = new_spacing.astype(np.float32)
    # scale = scale.astype(np.float32)

    ntype = ntype.lower()
    if ntype == 'nearest':
        order = 0
    elif ntype == 'linear':
        order = 1
    elif ntype == 'cubic':
        order = 2
    else:
        order = 0
    

    Vnew = interpolate_to_new_grid(
                            orig_dim=np.asarray(V.shape),
                            orig_spacing=np.asarray(scale).flatten(order='F'),
                            orig_vox=V,
                            grid_origin = None,
                            sample_spacing = np.asarray(new_spacing).flatten(order='F'),
                            order=order)

    # Vnew = interpolate(
    #                 vol_dim=np.asarray(V.shape),
    #                 vol_spacing=np.asarray(scale).flatten(),
    #                 vol=V,
    #                 new_dim = np.asarray(tsize).flatten(),
    #                 new_spacing = new_spacing,
    #                 order=order)

    return Vnew


def interpolate_to_new_grid(orig_dim,
                            orig_spacing,
                            orig_vox,
                            sample_dim = None ,
                            sample_spacing = np.array([0.0, 0.0, 0.0]),
                            grid_origin = None,
                            translation = np.array([0.0, 0.0, 0.0]),
                            order = 0,
                            mode='nearest',
                            align_to_center=True,
                            processor='scipy'):



    sample_spacing = sample_spacing.astype(np.float32)
    orig_spacing = orig_spacing.astype(np.float32)
    orig_dim = orig_dim.astype(np.float32)

    if sample_dim is None:
        sample_dim = np.ceil(np.multiply(orig_dim, orig_spacing / sample_spacing))
    #     sample_dim = np.array(sample_dim[0])

    grid_spacing = sample_spacing / orig_spacing

    if grid_origin is None:
        if align_to_center:
            grid_origin = 0.5 * (np.array(orig_dim) - 1.0) - 0.5 * (np.array(sample_dim) - 1.0) * grid_spacing
        else:
            grid_origin = np.array([0.0, 0.0, 0.0])

        grid_origin += translation * grid_spacing


    if processor == 'scipy':
        map_x, map_y, map_z = np.mgrid[:sample_dim[0], :sample_dim[1], :sample_dim[2]]

        map_z = map_z * grid_spacing[2] + grid_origin[2]
        map_z = map_z.astype(np.float32)
        map_y = map_y * grid_spacing[1] + grid_origin[1]
        map_y = map_y.astype(np.float32)
        map_x = map_x * grid_spacing[0] + grid_origin[0]
        map_x = map_x.astype(np.float32)

        map_vox = ndi.map_coordinates(input=orig_vox.astype(np.float32),
                                        coordinates=np.array([map_x, map_y, map_z], dtype=np.float32),
                                        order=order,
                                        mode=mode)

    elif processor == 'sitk':
        sitk_orig_img = sitk.GetImageFromArray(orig_vox.astype(np.float32), isVector=False)
        sitk_orig_img.SetOrigin(np.array([0.0, 0.0, 0.0]))
        sitk_orig_img.SetSpacing(np.array(orig_spacing,dtype=np.float32).tolist())

        interpolator = sitk.ResampleImageFilter()

        if order == 0:
            interpolator.SetInterpolator(sitk.sitkNearestNeighbor)
        elif order == 1:
            interpolator.SetInterpolator(sitk.sitkLinear)
        elif order == 2:
            interpolator.SetInterpolator(sitk.sitkBSplineResamplerOrder2)
        elif order == 3:
            interpolator.SetInterpolator(sitk.sitkBSpline)


        interpolator.SetOutputOrigin(np.array(grid_origin[::-1],dtype=np.float32).tolist())
        interpolator.SetOutputSpacing(np.array(grid_spacing[::-1],dtype=np.float32).tolist())
        interpolator.SetSize(sample_dim[::-1].astype(int).tolist())

        map_vox = sitk.GetArrayFromImage(interpolator.Execute(sitk_orig_img))

    return map_vox


