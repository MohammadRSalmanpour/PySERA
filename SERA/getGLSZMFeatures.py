
import numpy as np
import scipy 



# -------------------------------------------------------------------------
# [GLSZM2D, GLSZM3D] = getGLSZMtex(ROI2D,ROI3D,levels2D,levels3D)
# -------------------------------------------------------------------------
# DESCRIPTION: 
# This function calculates the GLSZM matrix for 2D and 3D.
# In 2D, every slice is calculated separately, then features are calculated.
# 
# The grey level size zone matrix (GLSZM) counts the number of groups of
# connected voxels witha specific discretised grey level value and size
# (Thibault et al., 2014). Voxels are connected ifthe neighbouring voxel
# has the same discretised grey level value.   
# -------------------------------------------------------------------------
# INPUTS:
# - ROI2D: Smallest box containing the 2D resampled ROI, with the imaging
#          data ready for texture analysis computations. Voxels outside the
#          ROI are set to NaNs.   
# - ROI3D: Smallest box containing the 3D resampled ROI, with the imaging
#          data ready for texture analysis computations. Voxels outside the
#          ROI are set to NaNs.   
# - levels2D: number of bins (for fixed number of bins method) or bin size
#           (for bin size method) for the 2D resampled ROI.
# - levels3D: number of bins (for fixed number of bins method) or bin size
#           (for bin size method) for the 3D resampled ROI.
# Note: ROIonly is the outputs of prepareVolume.m
# -------------------------------------------------------------------------
# OUTPUTS:
# - GLSZM2D: An array of 16 GLSZM features for the 2D resampled ROI.
# - GLSZM3D: An array of 16 GLSZM features for the 3D resampled ROI.
# -------------------------------------------------------------------------
# AUTHOR(S): 
# - Saeed Ashrafinia
# - Mahdi Hosseinzadeh
# -------------------------------------------------------------------------
# HISTORY:
# - Creation: December 2022
# -------------------------------------------------------------------------


def getGLSZMtex(ROI2D2,ROI3D2,levels2D1,levels3D1):

    ROI3D = ROI3D2.copy()
    ROI2D = ROI2D2.copy()
    levels2D = levels2D1.copy()
    levels3D = levels3D1.copy()

    nX, nY, nZ = ROI2D.shape
    # FeatTmp = []
    # GLSZM2D_all = np.zeros(  (levels2D.shape[0] , int(np.ceil(np.max([nX,nY])/2)) , nZ  ))
    GLSZM2D_all = []
    for s in range(0,nZ): 
        GLSZM   = getGLSZM(ROI2D[:,:,s],levels2D)
        # GLSZM2D_all[:,0:GLSZM.shape[1],s] = GLSZM
        GLSZM2D_all.append(GLSZM) 

        GLSZMstr = np.array(getGLSZMtextures(GLSZM))
        # FeatTmp = np.concatenate(FeatTmp , list(GLSZMstr) , axis=1)

        if s == 0:
            FeatTmp = GLSZMstr
            # FeatTmp = np.expand_dims(FeatTmp,axis=1)
        else:    
            FeatTmp = np.column_stack((FeatTmp , GLSZMstr))
        
    # FeatTmp_nan = FeatTmp[FeatTmp != np.nan]
    Feats_2D = np.nanmean(FeatTmp, axis=1)

    GLSZM2D_all = np.dstack(GLSZM2D_all)

    GLSZM25 = np.sum(GLSZM2D_all, axis =2)
    if np.sum(GLSZM25)!=0:
        Feats_25D = np.transpose(np.array(getGLSZMtextures(GLSZM25)))
    else:
        Feats_25D = Feats_2D



    GLSZM  = getGLSZM(ROI3D,levels3D)
    GLSZM3Dstr = getGLSZMtextures(GLSZM)
    Feats_3D = np.transpose(np.array(GLSZM3Dstr))


    return Feats_2D, Feats_3D, Feats_25D 


def getGLSZMtextures(GLSZM):


    sz = GLSZM.shape
    nRuns = int(np.sum(GLSZM))
    cVect = np.arange(1,sz[1]+1,1) 
    rVect = np.arange(1,sz[0]+1,1) 
    cMat,rMat = np.meshgrid(cVect,rVect)
    pg = np.transpose(np.expand_dims(np.sum(GLSZM,axis=1),axis=1))
    pr = np.transpose(np.expand_dims(np.sum(GLSZM,axis=0),axis=1))

    

    SZE = (np.dot(pr,np.expand_dims(np.float_power(cVect ,-2),axis=1)) / nRuns)[0][0]

    LZE = (np.dot(pr,np.expand_dims(np.float_power(cVect,2),axis=1))/nRuns)[0][0]

    LGZE = (np.dot(pg,np.expand_dims(  np.float_power(rVect,-2)  ,axis=1   ))/nRuns)[0][0]

    HGZE = (np.dot(pg,np.expand_dims(   np.float_power(rVect,2) ,axis=1    ))/nRuns)[0][0]


    

    SZLGE = np.sum(np.sum(  np.multiply(GLSZM  ,  np.multiply(   np.float_power(rMat,-2), np.float_power(cMat,-2)     ))  ))/nRuns

    SZHGE = np.sum(np.sum(  np.multiply(GLSZM  ,  np.multiply(   np.float_power(rMat,2), np.float_power(cMat,-2)     ))   ))/nRuns

    LZLGE = np.sum(np.sum(  np.multiply(GLSZM  ,  np.multiply(   np.float_power(rMat,-2), np.float_power(cMat,2)     ))   ))/nRuns

    LZHGE = np.sum(np.sum(  np.multiply(GLSZM  ,  np.multiply(   np.float_power(rMat,2), np.float_power(cMat,2)     ))  ))/nRuns


    GLN = np.sum(np.float_power(pg,2))/nRuns

    GLNN = np.sum(np.float_power(pg,2))/ np.float_power(nRuns,2)

    ZSN = np.sum(np.float_power(pr,2))/nRuns

    ZSNN = np.sum(np.float_power(pr,2))/ np.float_power(nRuns,2)


    ZP = nRuns/ np.dot(pr,  np.expand_dims(cVect,axis=1))[0][0]

    mu = np.sum( np.dot(np.expand_dims(rVect,axis=0) , GLSZM))/nRuns
    GLV = np.sum(np.dot(np.expand_dims(np.float_power((rVect-mu),2),axis=0) ,GLSZM))/nRuns

    mu = np.sum(   np.dot(GLSZM ,  np.expand_dims(cVect,axis =1))    )/nRuns
    ZSV = np.sum(   np.dot( GLSZM ,  np.expand_dims(  np.float_power((cVect-mu),2 ) ,axis=1 )      )   )/nRuns

    Entropy = -np.sum(  np.multiply((GLSZM/nRuns) ,np.log2(   (GLSZM/nRuns) + np.finfo(float).tiny) ))

    textures = [SZE,LZE,  LGZE,  HGZE, SZLGE, SZHGE, LZLGE,  LZHGE,
                GLN, GLNN, ZSN, ZSNN, ZP, GLV, ZSV, Entropy]


    return textures


def getGLSZM(ROIOnly2,levels):
    
    ROIOnly = ROIOnly2.copy()
    nLevel = len(levels)
    if nLevel > 100:
        adjust = 10000
    else:
        adjust = 1000
    

    levelTemp = np.max(levels)+1

    ROIOnly = np.nan_to_num(ROIOnly, nan=levelTemp)
    # ROIOnly[ROIOnly == np.nan] = levelTemp

    levels = np.append(levels, levelTemp)


    uniqueVect = np.round(levels*adjust)/adjust
    ROIOnly=np.round(ROIOnly*adjust)/adjust
    NL = len(levels) - 1


    nInit = np.size(ROIOnly)
    GLSZM = np.zeros((NL,nInit))

    temp = ROIOnly.copy()
    for i in range (0,NL):

        temp[ROIOnly != uniqueVect[i]] = 0
        temp[ROIOnly == uniqueVect[i]] = 1
        # connObjects = bwconncomp(temp,26)
        # connObjects = skimage.measure.label(temp, connectivity=None)
        if ROIOnly.ndim == 2:
            structure = scipy.ndimage.generate_binary_structure(2, 8)
        else:
            structure = scipy.ndimage.generate_binary_structure(3, 27)
        labeled_array, nZone = scipy.ndimage.label( input= temp, structure = structure)
        # nZone = len(connObjects['PixelIdxList'])
        for j in range (0,nZone):
            # col = len(connObjects['PixelIdxList'][j])
            # col = int((labeled_array)) - 1
            arr = labeled_array == (j+1)
            col = np.count_nonzero(arr) - 1
            GLSZM[i,col] = GLSZM[i,col] + 1

    # sumGLSZM = np.sum(GLSZM,axis=0)
    # stop = np.max(np.where(sumGLSZM > 0))
    # GLSZM = GLSZM[:,:stop+1]

    # GLSZM = GLSZM[:,~np.all(GLSZM == 0, axis=1)]
    # Result = OriginMat[:,~np.all(OriginMat == 0, axis = 0)]
    return GLSZM


