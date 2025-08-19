from ..visera.SERAutilities import *



# -------------------------------------------------------------------------
# [NGLDM2D_F, NGLDM3D_F] = getNGLDMtex(ROI2D,ROI3D,levels2D,levels3D)
# -------------------------------------------------------------------------
# DESCRIPTION: 
# This function calculates the NGLDM matrix for 2D and 3D.
# In 2D, every slice is calculated separately, then features are calculated.
# 
# The grey level size zone matrix (NGLDM) counts the number of groups of
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
# - NGLDM2D: An array of 16 NGLDM features for the 2D resampled ROI.
# - NGLDM3D: An array of 16 NGLDM features for the 3D resampled ROI.
# -------------------------------------------------------------------------
# AUTHOR(S): 
# - Saeed Ashrafinia
# - Mahdi Hosseinzadeh
# -------------------------------------------------------------------------
# HISTORY:
# - Creation: December 2022
# -------------------------------------------------------------------------

def getNGLDMtex(ROI2D2,ROI3D2,levels2D1,levels3D1, feature_value_mode='REAL_VALUE'):

    ROI3D = ROI3D2.copy()
    ROI2D = ROI2D2.copy()
    levels2D = levels2D1.copy()
    levels3D = levels3D1.copy()
    nX, nY, nZ = ROI2D.shape

    # FeatTmp = []


    NGLDM_all = np.zeros((levels2D.shape[0] , 9 , nZ), dtype=np.float32)

    for s in range(0,nZ): 
        NGLDM   = getNGLDM(ROI2D[:,:,s],levels2D)
        if NGLDM.ndim == 1:
            NGLDM = np.expand_dims(NGLDM, 1)
        NGLDM_all[:,0:NGLDM.shape[1], s] = NGLDM 
        NGLDMstr = CalcNGLDM(NGLDM,ROI2D[:,:,s], feature_value_mode=feature_value_mode)
        if s == 0:
            FeatTmp = np.array(NGLDMstr)

        else:    
            FeatTmp = np.column_stack((FeatTmp , NGLDMstr))
        

    if np.ndim(FeatTmp) == 1:
        FeatTmp = FeatTmp[:, np.newaxis]
    NGLDM2D_F = np.nanmean(FeatTmp , axis=1)

    NGLDM25 = np.sum(NGLDM_all, axis=1)
    if np.size(NGLDM25) > 1:
        NGLDM25 = np.squeeze(NGLDM25)
    elif np.size(NGLDM25) == 1:
        NGLDM25 = np.reshape(NGLDM25, (1,))

    NGLDM25D = np.transpose(CalcNGLDM(NGLDM25,ROI2D, feature_value_mode=feature_value_mode))


    NGLDM   = getNGLDM(ROI3D,levels3D)
    NGLDM3D_F = np.transpose(CalcNGLDM(NGLDM,ROI3D, feature_value_mode=feature_value_mode))



    return NGLDM2D_F, NGLDM3D_F, NGLDM25D



def LowDepEmph(Sj,nN,nS):

    J = np.arange(1,nN+1,1)
    tmp = np.divide (Sj , np.float_power(J,2))
    f_LDE = np.sum(tmp)/nS
    return f_LDE

def HighDepEmph(Sj,nN,nS):

    J = np.arange(1,nN+1,1)
    tmp = np.multiply (Sj , np.float_power(J,2))
    f_HDE = np.sum(tmp)/nS

    return f_HDE


def LowGLCountRmph(Si,nG,nS):

    I = np.arange(1,nG+1,1)
    tmp = np.divide (Si , np.float_power(I,2))
    f_LGLRE = np.sum(tmp)/nS
    return f_LGLRE

def HighGLCountEmph(Si,nG,nS):

    I = np.arange(1,nG+1,1)
    tmp = np.multiply (Si , np.float_power(I,2))
    f_HGLRE = np.sum(tmp)/nS
    return f_HGLRE

def LowDepLowGLEmph(NGLDM,nG,nN,nS):

    rangeX = np.arange(1, nG+1 , 1)
    rangeY = np.arange(1, nN+1 , 1)
    I , J = np.meshgrid(rangeX,rangeY)

    tmp = np.divide (  np.transpose(NGLDM) ,np.multiply(  np.float_power(I,2) , np.float_power(J,2)    ))
    f_SRLGLE = np.sum(tmp)/nS
    return f_SRLGLE

def LowDepHighRL(NGLDM,nG,nN,nS):

    rangeX = np.arange(1, nG+1 , 1)
    rangeY = np.arange(1, nN+1 , 1)
    I , J = np.meshgrid(rangeX,rangeY)
    
    tmp = np.multiply ( np.float_power(I,2) ,np.divide( np.transpose(NGLDM)  , np.float_power(J,2)    ))
    f_SRHGLE = np.sum(tmp)/nS

    return f_SRHGLE


def HighDepLowGLEmph(NGLDM,nG,nN,nS):

    rangeX = np.arange(1, nG+1 , 1)
    rangeY = np.arange(1, nN+1 , 1)
    I , J = np.meshgrid(rangeX,rangeY)

    tmp = np.multiply ( np.float_power(J,2) ,np.divide( np.transpose(NGLDM)  , np.float_power(I,2)    ))
    f_LRLGLE = np.sum(tmp)/nS

    return f_LRLGLE

def  HighDepHighGLEmph(NGLDM,nG,nN,nS):

    rangeX = np.arange(1, nG+1 , 1)
    rangeY = np.arange(1, nN+1 , 1)
    I , J = np.meshgrid(rangeX,rangeY)

    tmp = np.multiply ( np.float_power(J,2) ,np.multiply( np.float_power(I,2) ,np.transpose( NGLDM )   ))

    f_LRHGLE = np.sum(tmp)/nS
    return f_LRHGLE


def GLnonUnif(Si,nS):

    pow_Si = np.float_power(Si,2)
    res = np.sum(pow_Si)
    f_GLNU =  res / nS

    return f_GLNU


def GLnonUnifNormzd(Si,nS):

    f_GLNUN = np.sum(np.float_power(Si,2)) / np.float_power(nS,2)

    return f_GLNUN


def DepCountNonUnif(Sj,nS):


    f_RLNU = np.sum(np.float_power(Sj,2)) / nS

    return f_RLNU


def DepCountNonUnifNormzd(Sj,nS):

    f_RLNUN = np.sum(np.float_power(Sj,2)) / np.float_power(nS,2)

    return f_RLNUN

def DepCountPercentage(nS,nV):

    f_RP = nS / nV
    return f_RP


def GLVar(NGLDM,nG,nN,nS):

    Pij = NGLDM / nS

    rangeX = np.arange(1, nG+1 , 1)
    rangeY = np.arange(1, nN+1 , 1)
    I , J = np.meshgrid(rangeX,rangeY)

    mu = np.sum(np.sum( np.multiply(I , np.transpose(Pij))))
    tmp = np.multiply(np.float_power((I - mu),2) , np.transpose(Pij))
    f_GLV = np.sum(tmp)

    return f_GLV

def DepCountVar(NGLDM,nG,nN,nS):

    Pij = NGLDM / nS

    rangeX = np.arange(1, nG+1 , 1)
    rangeY = np.arange(1, nN+1 , 1)
    I , J = np.meshgrid(rangeX,rangeY)

    mu = np.sum(np.sum( np.multiply(J , np.transpose(Pij))))
    tmp = np.multiply(np.float_power((J - mu),2) , np.transpose(Pij))
    f_RLV = np.sum(tmp)

    return f_RLV


def DepCountEntropy(NGLDM,nS, feature_value_mode='REAL_VALUE', epsilon=1e-30):

    Pij = NGLDM / nS
    tmp = np.multiply(Pij , np.log2(Pij + epsilon))

    f_RE = -np.sum(tmp)

    return f_RE


def DepCountEnergy(NGLDM,nS):

    Pij = NGLDM / nS
    f_Enrg = np.sum(  np.float_power(Pij,2))

    return f_Enrg





def CalcNGLDM(NGLDM,ROI, feature_value_mode='REAL_VALUE'):

    ArrayOut = []

    try:
        nN = NGLDM.shape[1] 
    except:
        NGLDM = np.expand_dims(NGLDM,1)
        nN = NGLDM.shape[1] 
        
    nG = NGLDM.shape[0]        
    nN = NGLDM.shape[1]           
    nS = int(np.sum(NGLDM))  
               
    Si = np.sum(NGLDM , axis= 1).astype(np.int32)          
    Sj = np.sum(NGLDM , axis= 0).astype(np.int32)               
    nV = np.where( ~np.isnan(ROI))[0].shape[0]

    if nS == 0:
        if feature_value_mode=='APPROXIMATE_VALUE':
            value = 0
        else:
            # if feature_value_mode=='REAL_VALUE'
            value = np.nan
        import logging
        output = [value] * 17
        logging.getLogger("Dev_logger").warning("NGLDM Empty NGLDM matrix (nS=0). ROI may be too small or empty or it has only 1 voxel. Returning {value} values.")
        return output

    ArrayOut.append(LowDepEmph(Sj,nN,nS))
    ArrayOut.append(HighDepEmph(Sj,nN,nS))
    ArrayOut.append(LowGLCountRmph(Si,nG,nS))
    ArrayOut.append(HighGLCountEmph(Si,nG,nS))
    ArrayOut.append(LowDepLowGLEmph(NGLDM,nG,nN,nS))
    ArrayOut.append(LowDepHighRL(NGLDM,nG,nN,nS))
    ArrayOut.append(HighDepLowGLEmph(NGLDM,nG,nN,nS))
    ArrayOut.append(HighDepHighGLEmph(NGLDM,nG,nN,nS))
    ArrayOut.append(GLnonUnif(Si,nS))
    ArrayOut.append(GLnonUnifNormzd(Si,nS))
    ArrayOut.append(DepCountNonUnif(Sj,nS))
    ArrayOut.append(DepCountNonUnifNormzd(Sj,nS))
    ArrayOut.append(DepCountPercentage(nS,nV))
    ArrayOut.append(GLVar(NGLDM,nG,nN,nS))
    ArrayOut.append(DepCountVar(NGLDM,nG,nN,nS))
    ArrayOut.append(DepCountEntropy(NGLDM,nS, feature_value_mode=feature_value_mode))
    ArrayOut.append(DepCountEnergy(NGLDM,nS))


    return ArrayOut





def getNGLDM(ROIonly2,levels):

    ROIonly = ROIonly2.copy()

    a = 0
    d = 1


    nLevel = len(levels)
    if nLevel > 100:
        adjust = 10000
    else:
        adjust = 1000
    
    ROIonly = np.round(ROIonly*adjust)/adjust
    sizeV = ROIonly.shape


    if ROIonly.ndim == 3:
        if sizeV[2] != 1:
            nComp = sizeV[2]
        else:
            nComp = 2
        NnInit = int(np.float_power((2*d+1),3))
        ROIonly = np.pad(ROIonly,((1,1),(1,1),(1,1)),mode='constant' , constant_values=np.nan)
    else:
        nComp = 1
        NnInit = int(np.float_power((2*d+1),2))
        ROIonly = np.pad(ROIonly,((1,1),(1,1)),mode='constant' , constant_values=np.nan)

    NGLDM = np.zeros((nLevel,NnInit), dtype=np.float32)


    if nComp == 1:
        I,J,sl = ind2sub(ROIonly)
        for n in range (0,I.shape[0]):
            XglC = int(ROIonly[I[n],J[n]])

            newROIonly = ROIonly[ I[n]-1:I[n]+2 , J[n]-1:J[n]+2]
            k = np.where(np.abs(XglC - newROIonly) <= a)[0].shape[0]
            NGLDM[XglC-1,k-1] = NGLDM[XglC-1,k-1] + 1
        
    else:
        I,J,K = ind2sub(ROIonly)
        for n in range (0,I.shape[0]):
            XglC = int(ROIonly[I[n],J[n],K[n]])
            newROIonly = ROIonly[I[n]-1:I[n]+2 , J[n]-1:J[n]+2 , K[n]-1:K[n]+2 ]
            k = np.where(np.abs(XglC - newROIonly ) <= a)[0].shape[0]
            NGLDM[XglC-1,k-1] = NGLDM[XglC-1,k-1] + 1


    fin = np.where(np.sum(NGLDM,axis=0) > 0)[0]
    if len(fin) > 0:
        NGLDM = NGLDM[:,0:fin[-1]+1]
    else:    
        NGLDM = NGLDM[:,0]



    return NGLDM


