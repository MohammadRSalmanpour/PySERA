import numpy as np
import scipy
import cmath
import gc
# from skimage.measure import marching_cubes
from ..visera.SERAutilities import *

from pysera.utils.utils import handle_math_operations
from ...processing.synthesize_RoIs import synthesis_small_RoI

import logging

logger = logging.getLogger("Dev_logger")

# -------------------------------------------------------------------------
# function [MorphVect] = getMorph(MorphBox,MorphROI,ROIints, pixelW,sliceS)
# -------------------------------------------------------------------------
# DESCRIPTION:
# This function calculates morphological features.
# -------------------------------------------------------------------------
# INPUTS:
# - MorphBox: The smallest box containing the resampled 3D ROI, with the
#             imaging data ready for texture analysis computations. Voxels
#             outside the ROI are set to NaNs.
# - MorphROI: The smallest box containing the 3D morphological ROI. Voxels
#             outside the ROI are set to 0. It has only 0 and 1.
# - ROIints: The smallest box containing the 3D intensity ROI with
#             their intensities. Voxels outside the ROI are set to NaNs.
# - pixelW: width of the voxel in the X (=Y) direction
# - sliceS: Slice thickness of the voxel in the Z direction
#
# Note: The first 3 parameters are outputs of prepareVolume function.
# -------------------------------------------------------------------------
# OUTPUTS:
# - MorphVect: An array of calculated morphological features
# -------------------------------------------------------------------------
# AUTHOR(S):
# - Saeed Ashrafinia
# - Mahdi Hosseinzadeh
# -------------------------------------------------------------------------
# HISTORY:
# - Creation: December 2022
# -------------------------------------------------------------------------


def getMorph(MorphBox2, _MorphROI2, ROIints2, pixelW, sliceS, feature_value_mode='REAL_VALUE'):
    MorphVect = []
    
    # Load roi from disk
    MorphROI = _MorphROI2.copy()
    ROIints = ROIints2.copy()
    # MorphBox = MorphBox2.copy()
    # MorphROI[MorphROI == 0] = np.nan
    # MorphROI_disk = np.load(MorphROI2_path, mmap_mode='c')
    morph_count = np.count_nonzero(MorphROI)
    RoI_count = np.count_nonzero(ROIints2)

    if morph_count < 2 and feature_value_mode=='APPROXIMATE_VALUE':             # Synthesize 1 voxels
            synth_MorphROI = synthesis_small_RoI(morph_count, MorphROI.copy(), background=0, target_num=2)
            synth_ROIints = synthesis_small_RoI(RoI_count, ROIints.copy(), background='NaN', target_num=2)
            synth_flag = True
    else:
        # if feature_value_mode=='REAL_VALUE' and NOT morph_count < 2
        synth_MorphROI = None
        synth_ROIints = None
        synth_flag = False

    # Clean RAM
    del ROIints2
    gc.collect()

    ROIboxF, ROIboxV, _ = getMesh(MorphROI, pixelW, pixelW, sliceS)
    MV, Surface = getVolSurfaceMesh(ROIboxF, ROIboxV)

    # from skimage.measure import mesh_surface_area
    # ROIboxV = ROIboxV - 1
    # ROIboxF = ROIboxF - 1
    # area = mesh_surface_area(verts=ROIboxV.astype(np.uint32), faces=ROIboxF.astype(np.uint32))

    MorphROI[MorphROI == 0] = np.nan
    if synth_flag:
        MVappx = morph_count
    else:
        MVappx = getVolume(MorphROI, pixelW, sliceS)

    xM, yM, zM = ind2sub(MorphROI)
    xI, yI, zI = ind2sub(ROIints)
    try:
        concat = np.column_stack((xM * pixelW, yM * pixelW, zM * sliceS)) + 1
        concat2 = np.column_stack((xI * pixelW, yI * pixelW, zI * sliceS)) + 1
        aa = np.nanmean(concat, axis=0)
        ROIints_flatten = ROIints.flatten(order='F')
        notnanval = np.where(~np.isnan(ROIints_flatten))
        arr = np.expand_dims(ROIints_flatten[notnanval[0]], -1)
        tile = np.tile(arr, reps=(1, 3))
        bb = np.nansum(np.multiply(concat2, tile), axis=0)
        cc = np.tile(np.nansum(ROIints), (1, 3))
        dd = aa - np.divide(bb, cc)

        CoM_shift = np.linalg.norm(dd)

    except:
        print('**** ROI is 1D. Brace for problems.....')
        CoM_shift = np.abs(np.mean(np.squeeze(MorphROI) - np.squeeze(ROIints)))

    Shape = getShape(MV, Surface)
    compactness1 = Shape['compactness1']
    compactness2 = Shape['compactness2']
    sphericity = Shape['sphericity']
    SVratio = Shape['SVratio']
    Irregularity = Shape['Irregularity']
    sphericalDisp = Shape['sphericalDisp']
    Asphericity = Shape['Asphericity']

    # Volume Densities
    VDs, ConvHullV = getVDs(MorphROI, synth_MorphROI, synth_flag, ROIboxV, pixelW, sliceS, MV, Surface, feature_value_mode=feature_value_mode)
    Mean = np.nanmean(ROIints)
    TLG = Mean * MV

    ## 1111111
    ROIints = ROIints.flatten(order='F')
    MorphROI3 = MorphROI.flatten(order='F')

    MorphROI3 = MorphROI3[~ np.isnan(ROIints)]
    ROIints = ROIints[~ np.isnan(ROIints)]
    ROIints = ROIints[~ np.isnan(MorphROI3)]

    xI = xI[~ np.isnan(MorphROI3)]
    yI = yI[~ np.isnan(MorphROI3)]
    zI = zI[~ np.isnan(MorphROI3)]


    ## 22222
    # ROIints_flatten = ROIints.flatten(order='F')
    # MorphROI3 = MorphROI.flatten(order='F')

    # print(np.mean(ROIints))
    # ROIints = ROIints[~np.isnan(MorphROI)]
    # MorphROI3 = MorphROI2.flatten(order='F')
    # xI = xI[MorphROI3 != 0]
    # yI = yI[MorphROI3 != 0]
    # zI = zI[MorphROI3 != 0]

    # # MoranI, GearyC = getAutoCorrs(ROIints, xI,yI,zI , pixelW,sliceS , Mean)
    # if MoranI == None and Max3Ddiam == None and feature_value_mode=='REAL_VALUE':
    #     Max3Ddiam = np.nan
    #     MoranI, GearyC = np.nan, np.nan
    # else:
    Max3Ddiam = getMax3Ddiam(ROIboxV, ConvHullV)
    MoranI, GearyC = getAutoCorrs(ROIints, xI, yI, zI, pixelW, sliceS)
    # MoranI, GearyC = getAutoCorrs(ROIints_flatten, xI,yI,zI,pixelW,sliceS )

    threshold = 2
    # GT2 = getVolume_gt(MorphBox,threshold,pixelW,sliceS)
    MorphVect = [MV, MVappx, Surface, SVratio, compactness1, compactness2,
                 sphericalDisp, sphericity, Asphericity, CoM_shift, Max3Ddiam]
    for v in VDs:
        MorphVect.append(v)

    MorphVect.append(TLG)
    MorphVect.append(MoranI)
    MorphVect.append(GearyC)

    return MorphVect


def getMesh(ROIbox, pixelWx, pixelWy, sliceS):
    dx, dy, dz = ROIbox.shape

    morphology_mask = np.pad(ROIbox, pad_width=((1, 1), (1, 1), (1, 1)), mode="constant", constant_values=0.0)

    # Ensure coordinate arrays match the exact dimensions of the morphology_mask
    mask_dx, mask_dy, mask_dz = morphology_mask.shape

    rangeX = np.linspace(0, pixelWx * (mask_dx - 1), mask_dx)
    rangeY = np.linspace(0, pixelWy * (mask_dy - 1), mask_dy)
    rangeS = np.linspace(0, sliceS * (mask_dz - 1), mask_dz)

    X, Y, Z = np.meshgrid(rangeX, rangeY, rangeS, indexing='ij')

    faces, vertices, norms = MarchingCubes(X, Y, Z, morphology_mask, 0.5, None)

    # from skimage.measure import marching_cubes

    # # Get ROI and pad with empty voxels
    # morphology_mask = np.pad(ROIbox, pad_width=1, mode="constant", constant_values=0.0)

    # # Use marching cubes to generate a mesh grid for the ROI
    # vertices, faces, norms, values = marching_cubes(volume=morphology_mask, level=0.5, spacing=(float(pixelWx),float(pixelWy),float(sliceS)))

    return faces, vertices, norms


def getVolSurfaceMesh(F, V):
    nFaces = F.shape[0]
    vols = np.zeros((nFaces, 1), dtype=np.float32)
    meanMat = np.mean(V, axis=0)
    vertices = V - meanMat
    F = F.astype(np.int64)
    for i in range(0, nFaces):
        ind = list(F[i, :] - 1)
        tetra = vertices[ind, :]

        vols[i] = np.linalg.det(tetra) / 6

    vol_out = np.abs(np.sum(vols))

    a = V[list(F[:, 1] - 1), :] - V[list(F[:, 0] - 1), :]
    b = V[list(F[:, 2] - 1), :] - V[list(F[:, 0] - 1), :]
    c = np.cross(a, b)
    area_out = 1 / 2 * np.sum(np.sqrt(np.sum(np.float_power(c, 2), axis=1)))

    return vol_out, area_out


def getVolume(ROIonly, pixelW, sliceS):
    # mask = ROIonly[~isnan(ROIonly)]
    numberVoxel = np.nansum(ROIonly.astype(np.float64))
    volume = numberVoxel * pixelW * pixelW * sliceS

    if hasattr(volume, '__len__') == True:
        volume = np.squeeze(volume)

    return volume


def getShape(V, S):
    compactness1 = V / np.sqrt(np.pi) / np.sqrt(np.float_power(S, 3))
    compactness2 = 36 * np.pi * np.float_power(V, 2) / np.float_power(S, 3)
    sphericity = np.float_power((36 * np.pi * np.float_power(V, 2)), (1 / 3)) / S
    sphericalDisp = S / np.float_power((36 * np.pi * np.float_power(V, 2)), (1 / 3))
    Asphericity = np.float_power((np.float_power(S, 3) / np.float_power(V, 2) / 36 / np.pi), (1 / 3)) - 1
    SVratio = S / V
    Scircle = np.float_power((3 * V / (4 * np.pi)), (2 / 3)) * 4 * np.pi
    Irregularity = S / Scircle

    Shape = {}
    Shape['compactness1'] = compactness1
    Shape['compactness2'] = compactness2
    Shape['sphericity'] = sphericity
    Shape['SVratio'] = SVratio
    Shape['Irregularity'] = Irregularity
    Shape['sphericalDisp'] = sphericalDisp
    Shape['Asphericity'] = Asphericity

    return Shape


def getVDs(ROIbox, synth_ROIbox, synth_flag, ROIboxV, pixelW, sliceS, MV, Surface, epsilon=1e-30, feature_value_mode='REAL_VALUE'):
    if synth_flag:
        x, y, z = ind2sub(synth_ROIbox)
    else:
        x, y, z = ind2sub(ROIbox)

    concat = np.column_stack((x * pixelW, y * pixelW, z * sliceS)) + 1

    covariance = np.cov(np.transpose(concat))
    MaxV = 20
    v0 = np.arange(0, MaxV + 1, 1)

    # Get eighenvalues
    # print(covariance, np.all(np.isnan(covariance)), 54678437685478365)
    if np.all(np.isnan(covariance)):
        MALs = np.full((3,), np.nan)
        Elong, Fltns = np.nan, np.nan
        VDaee, ADaee = np.nan, np.nan
        if feature_value_mode=='APPROXIMATE_VALUE':
            logger.error(f"Issue with synthesis. Returning NaN for MALs, Elong, Fltns, VDaee, ADaee.")
    else:
        EIGs = np.linalg.eigh(covariance)[0]  # EIGs =  scipy.linalg.eigh(covariance)[0]

        EIGs = np.transpose(np.flip(EIGs))

        EIGs = handle_math_operations(EIGs, feature_value_mode, 'sqrt', epsilon)

        if EIGs.shape[0] == 1:
            EIGs = np.tile(EIGs, (1, 3))

        L1 = 2 * np.sqrt(EIGs[0])
        L2 = 2 * np.sqrt(EIGs[1])
        L3 = 2 * np.sqrt(EIGs[2])

        MALs = 4 * np.sqrt(EIGs)
        # EIGs[0] = handle_math_operations(np.array(EIGs[0]), feature_value_mode, 'divide', epsilon)
        if feature_value_mode=='APPROXIMATE_VALUE' and EIGs[0] == 0.:
            Elong = np.sqrt(EIGs[1] / epsilon)
            Fltns = np.sqrt(EIGs[2] / epsilon)
        else:
            Elong = np.sqrt(EIGs[1] / EIGs[0])
            Fltns = np.sqrt(EIGs[2] / EIGs[0])

        if feature_value_mode=='APPROXIMATE_VALUE' and np.prod(np.sqrt(EIGs)) == 0.:
            VDaee = 3 * MV / (4 * np.pi * np.float_power(2, 3) * np.prod(np.sqrt(handle_math_operations(EIGs, feature_value_mode, 'divide', epsilon))))
        else:
            VDaee = 3 * MV / (4 * np.pi * np.float_power(2, 3) * np.prod(np.sqrt(EIGs)))

        alpha = np.sqrt(1 - np.float_power(L2, 2) / np.float_power(L1, 2))
        beta = np.sqrt(1 - np.float_power(L3, 2) / np.float_power(L1, 2))


        sumtmp = []
        aa = (np.float_power(alpha, 2) + np.float_power(beta, 2)) / (2 * alpha * beta)
        mm = np.min([1, aa])
        for k in range(0, MaxV + 1):

            bb = np.transpose(np.array(legendre(k, np.array(mm))))
            cc = (1 - 4 * np.float_power(k, 2))
            ee = np.float_power((alpha * beta), k)
            ff = np.divide(ee, cc)
            dd = np.multiply(ff, bb)
            if np.isscalar(dd):
                tem = dd
                dd = np.array([tem])
            else:
                dd = dd[0]
            sumtmp.append(dd)

        Final_sumtmp = []
        for i in range(0, len(sumtmp)):  # toto: use extend instead ???
            for j in sumtmp[i]:
                Final_sumtmp.append(j)

        sumtmp = np.array(Final_sumtmp)
        aee = 4 * np.pi * L1 * L2 * np.sum(sumtmp)
        aee = handle_math_operations(np.array(aee), feature_value_mode, 'divide', epsilon)
        ADaee = Surface / aee

    # if synth_flag:
    #     nx, ny, nz = synth_ROIbox.shape
    # else:
    nx, ny, nz = ROIbox.shape

    VDaabb = MV / np.prod(np.multiply(np.array([nx, ny, nz]), np.transpose(np.array([pixelW, pixelW, sliceS]))))
    ADaabb = (Surface / (2 * nx * pixelW * ny * pixelW + 2 * ny * pixelW * nz * sliceS + 2 * nx * pixelW * nz * sliceS))

    if hasattr(ADaabb, '__len__') == True:
        ADaabb = np.squeeze(ADaabb)

    covariance = np.cov(np.transpose(ROIboxV))
    EIGs = np.linalg.eigh(covariance)[0]
    # EIGs = np.sum(EIGs , axis = 0)
    L1 = 2 * np.sqrt(EIGs[0])
    L2 = 2 * np.sqrt(EIGs[1])
    L3 = 2 * np.sqrt(EIGs[2])
    VDmvee = MV / (4 * np.pi * L1 * L2 * L3 / 3)

    aa = (np.float_power(L2, 2) / np.float_power(L1, 2))
    bb = (np.float_power(L3, 2) / np.float_power(L1, 2))
    alpha = cmath.sqrt(1 - aa)
    beta = cmath.sqrt(1 - bb)

    try:
        yy = 4 * np.pi * L1 * L2

        aa = (np.float_power(alpha, 2) + np.float_power(beta, 2)) / (2 * alpha * beta)
        mm = np.min([1, aa])
        bb = np.transpose(np.array(legendre(MaxV, np.array(mm))))
        cc = (1 - 4 * np.float_power(v0, 2))
        ee = np.float_power((alpha * beta), v0)
        ff = np.divide(ee, cc)
        dd = np.multiply(ff, bb)

        if np.isscalar(dd):
            tem = dd
            dd = np.array([tem])
        else:
            dd = dd[0]

        aee = yy * np.sum(dd)

    except:
        EIGs = np.transpose(np.flip(EIGs))

        L1 = 2 * np.sqrt(EIGs[0])
        L2 = 2 * np.sqrt(EIGs[1])
        L3 = 2 * np.sqrt(EIGs[2])

        VDmvee = MV / (4 * np.pi * L1 * L2 * L3 / 3)

        alpha = cmath.sqrt(1 - (np.float_power(L2, 2) / np.float_power(L1, 2)))
        beta = cmath.sqrt(1 - (np.float_power(L3, 2) / np.float_power(L1, 2)))

        yy = 4 * np.pi * L1 * L2

        aa = (np.float_power(alpha, 2) + np.float_power(beta, 2)) / (2 * alpha * beta)
        mm = np.min([1, aa])
        bb = np.transpose(np.array(legendre(MaxV, np.array(mm))))
        cc = (1 - 4 * np.float_power(v0, 2))
        ee = np.float_power((alpha * beta), v0)
        ff = np.divide(ee, cc)
        dd = np.multiply(ff, bb)

        if np.isscalar(dd):
            tem = dd
            dd = np.array([tem])
        else:
            dd = dd[0]
        aee = yy * np.sum(dd)
    aee = np.real(aee)

    if aee == 0. and feature_value_mode == 'APPROXIMATE_VALUE':
        ADmvee = Surface / epsilon
    elif aee == 0. and feature_value_mode == 'REAL_VALUE':
        ADmvee = np.nan
    else:
        ADmvee = Surface / aee

    try:
        _, _, Vombb, Aombb, _ = minboundbox(x * pixelW, y * pixelW, z * sliceS, 'volume', 1, feature_value_mode=feature_value_mode)
    except:
        print('This is a 2D or 1D ROI. Switch to 2D Convex Hull and Bounding Box calculation.')
        if synth_flag:
            newROIbox = np.squeeze(synth_ROIbox)
        else:
            newROIbox = np.squeeze(ROIbox)

        xtmp, ytmp, _ = ind2sub(newROIbox)
        try:
            if synth_flag:
                RoIbox_shape = synth_ROIbox.shape
            else:
                RoIbox_shape = ROIbox.shape

            if RoIbox_shape[2] == 1:
                _, Aombb, Vombb = minBoundingBox2D(xtmp, ytmp, pixelW, pixelW, sliceS, feature_value_mode=feature_value_mode)
            elif RoIbox_shape[1] == 1:
                _, Aombb, Vombb = minBoundingBox2D(xtmp, ytmp, pixelW, sliceS, pixelW, feature_value_mode=feature_value_mode)
            elif RoIbox_shape[0] == 1:
                _, Aombb, Vombb = minBoundingBox2D(xtmp, ytmp, sliceS, pixelW, pixelW, feature_value_mode=feature_value_mode)
            else:
                print('Min bounding box does not respond (no Convex Hull available). Set OMBB = AABB....')
                arr1 = np.column_stack((nx, ny, nz))
                arr2 = np.column_stack((pixelW, pixelW, sliceS))

                Vombb = np.prod(np.multiply(arr1, arr2))
                Aombb = 2 * nx * pixelW * ny * pixelW + 2 * ny * pixelW * nz * sliceS + 2 * nx * pixelW * nz * sliceS


        except:
            print('Min bounding box does not respond (probably a 1D ROI). Set OMBB = AABB....')
            arr1 = np.column_stack((nx, ny, nz))
            arr2 = np.column_stack((pixelW, pixelW, sliceS))
            Vombb = np.prod(np.multiply(arr1, arr2))
            Aombb = 2 * nx * pixelW * ny * pixelW + 2 * ny * pixelW * nz * sliceS + 2 * nx * pixelW * nz * sliceS

    VDombb = MV / Vombb
    ADombb = Surface / Aombb

    # approache 1
    K2 = scipy.spatial.ConvexHull(ROIboxV)
    Vch = K2.volume
    K = K2.simplices

    # approache 2
    # hull = pyhull.convex_hull.ConvexHull(ROIboxV)
    # K = [x for x in hull.vertices if len(x) == 3]
    # K = np.array(K)

    Solidity = MV / Vch

    CHarea = np.sum(np.sqrt(np.sum(
        np.float_power(
            (
                    np.column_stack((
                        np.multiply(ROIboxV[K[:, 0], 1], ROIboxV[K[:, 1], 2]) - np.multiply(ROIboxV[K[:, 0], 2],
                                                                                            ROIboxV[K[:, 1], 1]),
                        np.multiply(ROIboxV[K[:, 0], 2], ROIboxV[K[:, 1], 0]) - np.multiply(ROIboxV[K[:, 0], 0],
                                                                                            ROIboxV[K[:, 1], 2]),
                        np.multiply(ROIboxV[K[:, 0], 0], ROIboxV[K[:, 1], 1]) - np.multiply(ROIboxV[K[:, 0], 1],
                                                                                            ROIboxV[K[:, 1], 0])))
                    +
                    np.column_stack((
                        np.multiply(ROIboxV[K[:, 1], 1], ROIboxV[K[:, 2], 2]) - np.multiply(ROIboxV[K[:, 1], 2],
                                                                                            ROIboxV[K[:, 2], 1]),
                        np.multiply(ROIboxV[K[:, 1], 2], ROIboxV[K[:, 2], 0]) - np.multiply(ROIboxV[K[:, 1], 0],
                                                                                            ROIboxV[K[:, 2], 2]),
                        np.multiply(ROIboxV[K[:, 1], 0], ROIboxV[K[:, 2], 1]) - np.multiply(ROIboxV[K[:, 1], 1],
                                                                                            ROIboxV[K[:, 2], 0])))
                    +
                    np.column_stack((
                        np.multiply(ROIboxV[K[:, 2], 1], ROIboxV[K[:, 0], 2]) - np.multiply(ROIboxV[K[:, 2], 2],
                                                                                            ROIboxV[K[:, 0], 1]),
                        np.multiply(ROIboxV[K[:, 2], 2], ROIboxV[K[:, 0], 0]) - np.multiply(ROIboxV[K[:, 2], 0],
                                                                                            ROIboxV[K[:, 0], 2]),
                        np.multiply(ROIboxV[K[:, 2], 0], ROIboxV[K[:, 0], 1]) - np.multiply(ROIboxV[K[:, 2], 1],
                                                                                            ROIboxV[K[:, 0], 0])))
            ), 2), axis=1
    ))) / 2

    CH_AD = Surface / CHarea

    VDs = np.transpose(
        [MALs[0], MALs[1], MALs[2], Elong, Fltns, VDaabb, ADaabb, VDombb, ADombb, VDaee, ADaee, VDmvee, ADmvee,
         Solidity, CH_AD])

    return VDs, K


def getMax3Ddiam(ROIboxV, ConvHullV):
    n_v_int = ROIboxV.shape[0]

    if 1 <= n_v_int < 1000:
        ROIboxV = ROIboxV.astype(np.float32)
        NumVer = ROIboxV.shape[0]

        # Equivalent to tiling + reshaping with order='F'
        diff = ROIboxV[:, np.newaxis, :] - ROIboxV[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        Max3Ddiam = np.sqrt(np.max(dist_sq))

    elif n_v_int >= 1000:
        idx0 = ConvHullV[:, 0].astype(int)
        idx1 = ConvHullV[:, 1].astype(int)
        idx2 = ConvHullV[:, 2].astype(int)

        r1 = np.column_stack((
            ROIboxV[idx0, 0],
            ROIboxV[idx1, 1],
            ROIboxV[idx2, 2]
        )).astype(np.float32)

        NumVer = r1.shape[0]
        diff = r1[:, np.newaxis, :] - r1[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        Max3Ddiam = np.sqrt(np.max(dist_sq))

    else:
        Max3Ddiam = 0.0

    return Max3Ddiam


def geospatial(df_int, spacing, xI, yI, zI):
    n_v = df_int.shape[0]
    pos_mat = np.column_stack((zI * spacing[0], yI * spacing[1], xI * spacing[2]))

    gl_dev = df_int - np.mean(df_int)
    gl_denom = np.sum(gl_dev ** 2.0)

    if n_v < 2000:
        # Match original indexing logic using tile and repeat
        comb_iter = np.array([np.tile(np.arange(n_v), n_v), np.repeat(np.arange(n_v), n_v)])
        comb_iter = comb_iter[:, comb_iter[0, :] > comb_iter[1, :]]

        i = comb_iter[0]
        j = comb_iter[1]

        diffs = pos_mat[i] - pos_mat[j]
        dist = np.sqrt(np.sum(diffs ** 2, axis=1))
        w_ij = 1.0 / dist

        # Moran's I
        moran_nom = n_v * np.sum(w_ij * gl_dev[i] * gl_dev[j])
        w_sum = np.sum(w_ij)

        moran_i = moran_nom / (w_sum * gl_denom) if gl_denom > 0.0 else 1.0

        # Geary's C
        geary_nom = (n_v - 1.0) * np.sum(w_ij * (gl_dev[i] - gl_dev[j]) ** 2)
        geary_c = geary_nom / (2 * w_sum * gl_denom) if gl_denom > 0.0 else 1.0

    else:
        moran_nom = 0.0
        geary_nom = 0.0
        w_denom = 0.0

        for ii in range(n_v - 1):
            jj = np.arange(ii + 1, n_v)
            diffs = pos_mat[ii] - pos_mat[jj]
            w_iijj = 1.0 / np.sqrt(np.sum(diffs ** 2, axis=1))

            gl_ii = gl_dev[ii]
            gl_jj = gl_dev[jj]

            moran_nom += np.sum(w_iijj * gl_ii * gl_jj)
            geary_nom += np.sum(w_iijj * (gl_ii - gl_jj) ** 2)
            w_denom += np.sum(w_iijj)

        moran_i = n_v * moran_nom / (w_denom * gl_denom) if gl_denom > 0.0 else 1.0
        geary_c = (n_v - 1.0) * geary_nom / (2 * w_denom * gl_denom) if gl_denom > 0.0 else 1.0

    return moran_i, geary_c


def getAutoCorrs(ROIints, xI, yI, zI, pixelW, sliceS):
    n_v_int = ROIints.size
    if (1000 > n_v_int >= 1):

        MoranI, GearyC = geospatial(df_int=ROIints, spacing=[pixelW, pixelW, sliceS], xI=xI, yI=yI, zI=zI)

    elif n_v_int >= 1000:
        moran_list, geary_list = [], []
        iter_nr = 1
        tol_aim = 0.002
        tol_sem = 1.000
        while tol_sem > tol_aim:
            curr_points = np.random.choice(n_v_int, size=100, replace=False)
            moran_i, geary_c = geospatial(df_int=ROIints[curr_points], spacing=[pixelW, pixelW, sliceS], xI=xI, yI=yI,
                                          zI=zI)

            moran_list.append(moran_i)
            geary_list.append(geary_c)

            if iter_nr > 10:
                tol_sem = np.max([np.std(moran_list), np.std(geary_list)]) / np.sqrt(iter_nr)
            iter_nr += 1

            del curr_points, moran_i, geary_c

        MoranI = np.mean(moran_list)
        GearyC = np.mean(geary_list)

        del iter_nr

    return MoranI, GearyC


