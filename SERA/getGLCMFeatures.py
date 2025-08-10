
import numpy as np
from SERAutilities import ind2sub

# -------------------------------------------------------------------------
# function [GLCM3D_Cmb, GLCM3D_Avg] = getGLCM3Dtex(ROIonly,levels)
# -------------------------------------------------------------------------
# DESCRIPTION: 
# This function calculates 2D GLCM and calculates texture features. 
# The beginning of this code is from Martin Valleries GLCM code. 
# Then a code by Carl Philips and Daniel Li (2008) was used to calculate
# GLCM in 3D (http://facweb.cs.depaul.edu/research/vc/contact.htm)
# Then we call the function CalcGLCM to calculate the features. 
# -------------------------------------------------------------------------
# INPUTS:
# - ROIonly: Smallest box containing the ROI, with the imaging data ready 
#            for texture analysis computations. Voxels outside the ROI are 
#            set to NaNs. 
# - levels: number of bins (for fixed number of bins method) or bin size
#           (for bin size method) 
# Note: ROIonly is the outputs of prepareVolume.m
# -------------------------------------------------------------------------
# OUTPUTS:
# - GLCM3D_Cmb: 3D GLCM features: First merging GLCMs for all directions,
#               then calculate features for the combined GLCM matrix.
# - GLCM3D_Avg: 3D GLCM features calculate GLCM features for each
#               direction, then average over all directions.
# -------------------------------------------------------------------------
# AUTHOR(S): 
# - Saeed Ashrafinia
# - Mahdi Hosseinzadeh
# -------------------------------------------------------------------------
# HISTORY:
# - Creation: December 2022
# -------------------------------------------------------------------------

def getGLCM3Dtex(ROIonly2,levels2):

    ROIonly = ROIonly2.copy()
    levels = levels2.copy()

    GLCMs = getGLCM(ROIonly , levels)

    if GLCMs.ndim == 3:
        tmp = np.squeeze(np.sum(GLCMs,axis=2))
    tmp = np.squeeze(GLCMs)
    GLCMsNormalized =  tmp / np.sum(tmp)

    GLCM3D_Cmb = CalcGLCM(GLCMsNormalized)

    ROIonly = np.nan_to_num(ROIonly,nan=0)
    new_ROIonly = ROIonly>0
    new_ROIonly = np.multiply(new_ROIonly,1)

    GLCMs = GLCM_3D(ROIonly , 
                    new_ROIonly , 
                    [np.min(levels), np.max(levels)],     
                    DIRECTION=None, 
                    DISTANCE = 1 ,
                    NUMGRAY = len(levels) , 
                    COMBINE=None, 
                    ACCUMULATE = None)


    tmp1 = np.sum(GLCMs , axis=0)
    tmp = np.sum(tmp1, axis=0)
    tmp = np.expand_dims(tmp,-1)
    tmp = np.moveaxis(tmp, 0, -1)

    tile1 = np.tile(tmp ,(GLCMs.shape[0],GLCMs.shape[1],1))
    GLCMs = np.squeeze(GLCMs)
    GLCMsNormalized = np.divide(GLCMs , tile1)
    GLCMsNormalized = np.squeeze(GLCMsNormalized)

    try:
        nff = GLCMsNormalized.shape[1]
    except:
        GLCMsNormalized = np.expand_dims(GLCMsNormalized,0)
        GLCMsNormalized = np.expand_dims(GLCMsNormalized,0)

    tmp = CalcGLCM(GLCMsNormalized)
    # new_tmp = tmp[tmp != np.nan]
    GLCM3D_Avg = np.nanmean(tmp , axis=0)


    return GLCM3D_Cmb, GLCM3D_Avg


# -------------------------------------------------------------------------
# function [SM_f, SS_f] = getGLCM2Dtex(ROIonly,levels)
# -------------------------------------------------------------------------
# DESCRIPTION: 
# This function calculates 2D GLCM and calculates texture features. 
# The beginning of this code is from Martin Valleries GLCM code 
# -------------------------------------------------------------------------
# INPUTS:
# - ROIonly: Smallest box containing the ROI, with the imaging data ready 
#            for texture analysis computations. Voxels outside the ROI are 
#            set to NaNs. 
# - levels: number of bins (for fixed number of bins method) or bin size
#           (for bin size method) 
# Note: RIOonly is the outputs of prepareVolume.m
# -------------------------------------------------------------------------
# OUTPUTS:
# - SM_f: 2D GLCM features merging GLCMs of slices, then calculate features
# - SS_f: 2D GLCM features calculate GLCM features for each slice, then
#         average over slices
# -------------------------------------------------------------------------
# AUTHOR(S): 
# - Saeed Ashrafinia
# - Mahdi Hosseinzadeh
# -------------------------------------------------------------------------
# HISTORY:
# - Creation: December 2022
# -------------------------------------------------------------------------



def getGLCM2Dtex(ROIonly2,levels2):

    ROIonly = ROIonly2.copy()
    levels = levels2.copy()

    levels = list(levels)
    nLevel = len(levels)
    if nLevel > 100:
        adjust = 10000
    else:
        adjust = 1000

    levelTemp = np.max(levels)+1
    ROIonly[ np.isnan(ROIonly)] = levelTemp
    levels.append(levelTemp)

    dim = ROIonly.shape
    if ROIonly.ndim == 2:
        dim[2] = 1
    d = np.prod(dim)
    q2 = np.reshape(ROIonly,(1,d),order='F')



    qs = np.round(  np.multiply(adjust ,levels))/adjust
    q2 = np.round(np.multiply(adjust ,q2))/adjust

    q3 = q2*0
    for k in range(0,len(qs)):
        q3[q2==qs[k]] = k+1
    
    ROInanReplaced = np.reshape(q3,dim,order='F')
    ROInanReplaced = ROInanReplaced.astype(np.uint32)

    DIST = [[1, 0],[1, 1],[0, 1],[-1, 1]]
    DIST = np.array(DIST)
    # DIST = [[0 1];[1 0];[1 1];[0 -1];[0 -1];[-1 0];[-1 1];[1 -1]]

    nZ = ROInanReplaced.shape[2]
    nlevels = len(levels)-1 
    GLCM_KeepSlice_KeepDirs = np.zeros((nlevels,nlevels,DIST.shape[0],nZ))

















    # un comment plz
    # if len(np.where(ROInanReplaced==0)[0]) == 0: 
    #     print('Need to change the minimum GrayLimits of GLCM to zeros.')
     















    for s in range(0,nZ):

        # tmpGLCM = graycomatrix(ROInanReplaced[:,:,s],'Offset',DIST,'NumLevels',nlevels+1,'GrayLimits',[1, nlevels+1],'Symmetric', True)
            # 270,315,0,45
        tmpGLCM,_ =  graycomatrix(ROInanReplaced[:,:,s],[1, nlevels+1],nlevels+1 ,DIST,True)
        # tmpGLCM = feature.graycomatrix(ROInanReplaced[:,:,s],distances=[1,0,1,2], angles=[0]  ,levels=nlevels+1,symmetric=True ,normed=False)
        # newDime = tmpGLCM.shape
        # tmpGLCM = tmpGLCM[0:-1 , 0:-1,:]
        GLCM_KeepSlice_KeepDirs[:,:,:,s] =  tmpGLCM[0:-1 , 0:-1,:]
        # try:
        #     if s == 1: 
        #         print('off','last')
        # except:
        #     print('s == 1 error')
    
    # print(np.mean(GLCM_KeepSlice_KeepDirs))
    GLCM_AllMerged   = np.sum(np.sum(GLCM_KeepSlice_KeepDirs,axis=3),axis=2)
    GLCMnorm_AllMerged = GLCM_AllMerged / np.sum(GLCM_AllMerged) 

    GLCM_KeepSlice_MergeDirs = np.zeros((GLCM_AllMerged.shape[0], GLCM_AllMerged.shape[1],GLCM_KeepSlice_KeepDirs.shape[3] ))
    GLCM_KeepSlice_MergeDirs[:,:,:] = np.sum(GLCM_KeepSlice_KeepDirs , axis = 2)

    ab = np.sum(GLCM_KeepSlice_MergeDirs , axis=0)
    tmp = np.sum(ab,axis=0)
    tile1 = np.tile(tmp ,(GLCM_KeepSlice_MergeDirs.shape[0],GLCM_KeepSlice_MergeDirs.shape[1],1))
    GLCMnorm_KeepSlice_MergeDirs = np.divide(   GLCM_KeepSlice_MergeDirs , tile1)
    GLCMnorm_KeepSlice_MergeDirs[ np.isnan( GLCMnorm_KeepSlice_MergeDirs)] = 0

    tmp = np.sum(np.sum(GLCM_KeepSlice_KeepDirs , axis=0),axis=0)
    tile1 = np.tile(tmp ,(GLCM_KeepSlice_MergeDirs.shape[0],GLCM_KeepSlice_MergeDirs.shape[1],1,1))
    GLCMnorm_KeepSlice_KeepDirs = np.divide(GLCM_KeepSlice_KeepDirs , tile1 )
    GLCMnorm_KeepSlice_KeepDirs[ np.isnan(GLCMnorm_KeepSlice_KeepDirs )] = 0

    GLCM_MergeSlice_KeepDirs = np.zeros((GLCM_AllMerged.shape[0], GLCM_AllMerged.shape[1],GLCM_KeepSlice_KeepDirs.shape[2] ))
    GLCM_MergeSlice_KeepDirs[:,:,:] = np.sum(GLCM_KeepSlice_KeepDirs , axis = 3)

    tmp = np.sum(np.sum(GLCM_MergeSlice_KeepDirs , axis=0),axis=0)
    tile1 = np.tile(tmp ,(GLCM_KeepSlice_KeepDirs.shape[0],GLCM_KeepSlice_KeepDirs.shape[1],1))
    GLCMnorm_MergeSlice_KeepDirs = np.divide(GLCM_MergeSlice_KeepDirs , tile1)
    GLCMnorm_MergeSlice_KeepDirs[np.isnan(GLCMnorm_MergeSlice_KeepDirs )] = 0



    Feats_MSMD = CalcGLCM(GLCMnorm_AllMerged)
    Feats_MSMD = np.dstack(Feats_MSMD)
    Feats_MSMD = np.squeeze(Feats_MSMD)
    nFeats = Feats_MSMD.shape[0]

    tmp = CalcGLCM(GLCMnorm_KeepSlice_MergeDirs)
    tmp = np.dstack(tmp)
    tmp = np.squeeze(tmp)

    if tmp.ndim > 2:
        raise('GLCM features might having an extra dimension')
    Feats_KSMD = np.nanmean(tmp , axis=1)

    tmp = CalcGLCM(GLCMnorm_MergeSlice_KeepDirs)
    tmp = np.dstack(tmp)
    tmp = np.squeeze(tmp)
    
    if tmp.ndim >2:
        raise('GLCM features might having an extra dimension')

    Feats_MSKD = np.nanmean(tmp ,axis=1)

    tmp = np.zeros((nFeats , DIST.shape[0], nZ))
    for d in range(0,DIST.shape[0]):
        for s in range(0,nZ): 
            tm = CalcGLCM(GLCMnorm_KeepSlice_KeepDirs[:,:,d,s])
            tm = np.dstack(tm)
            tm = np.squeeze(tm)
            tmp[:,d,s] = tm
    
    fea = np.squeeze(np.nanmean(tmp , axis=2))
    Feats_KSKD = np.nanmean(fea,axis=1)
    

    return Feats_KSKD, Feats_MSKD, Feats_KSMD, Feats_MSMD


def MaxProb(GLCM):

    f_joint_max = np.max(GLCM)

    return f_joint_max


def JointAvg(GLCM,nG):

    aa = np.arange(1,nG+1,1)
    tmp = np.multiply( GLCM ,np.transpose(np.tile(  aa   ,(nG,1))) )
    f_joint_avg = np.sum(tmp)

    return f_joint_avg


def JointVar(GLCM,nG , mu):

    aa = np.float_power((np.transpose(np.arange(1,nG+1,1)) - mu ) , 2 )
    bb = np.tile(  aa  , (nG,1))
    tmp = np.multiply( GLCM , bb )
    f_joint_var = np.sum(tmp)

    return f_joint_var


def Entropy(GLCM):

    tmp = np.multiply(GLCM , np.log2(GLCM + np.finfo(float).tiny))
    f_entropy = -np.sum(tmp)

    return f_entropy


def DiffAvg(GLCM,nG):

    pDiag = DiagProb(GLCM)
    tmp =  np.multiply(  np.arange(0,(nG))    , np.squeeze(pDiag))
    f_diffavg = np.sum(tmp)

    return f_diffavg




def DiffVar(GLCM,nG,muDiffAvg):

    pDiag = DiagProb(GLCM)

    aa = np.float_power(( np.arange(0,nG) - muDiffAvg ),2)

    tmp =  np.multiply(   aa  , np.squeeze(pDiag))
    f_diffvar = np.sum(tmp)

    return f_diffvar


def DiffEnt(GLCM):

    pDiag = DiagProb(GLCM)
    tmp = np.multiply(pDiag , np.log2(pDiag+ np.finfo(float).tiny))
    f_diffent = -np.sum(tmp)
    return f_diffent


def  SumAvg(GLCM,nG):

    pCross = CrossProb(GLCM)
    tmp =   np.multiply( np.arange(2,(2*nG)+1,1)  , pCross)
    f_sumavg = np.sum(tmp)

    return f_sumavg


def SumVar(GLCM,nG,muSumAvg):

    pCross = CrossProb(GLCM)
    tmp =   np.multiply(  np.float_power(  (np.arange(2,(2*nG)+1,1) -muSumAvg) ,2), pCross)
    f_sumvar = np.sum(tmp)

    return f_sumvar





def SumEnt(GLCM):
    pCross = CrossProb(GLCM)
    tmp =  np.multiply(pCross , np.log2(pCross+ np.finfo(float).tiny))
    f_sument = -np.sum(tmp)


    return f_sument


def Energy(GLCM):

    f_nrg = np.sum(  np.float_power (GLCM , 2))
    return f_nrg

def Contrast(GLCM,nG):

    I , J  = np.meshgrid(  np.arange (0,nG) , np.arange(0,nG))
    T = I-J
    tmp = np.multiply(np.float_power(T , 2) , GLCM)
    f_cont = np.sum(tmp)

    return f_cont


def Dissimilarity(GLCM,nG):

    I , J = np.meshgrid(np.arange (0,nG) , np.arange(0,nG))
    T = I-J
    tmp = np.multiply(np.abs(T) , GLCM)
    f_diss = np.sum(tmp)

    return f_diss


def InvDiff(GLCM,nG):

    I , J = np.meshgrid(np.arange (1,nG+1) , np.arange(1,nG+1))
    T = I-J
    tmp =  np.divide(GLCM , (1+np.abs(T)))
    f_invdiff = np.sum(tmp)


    return f_invdiff


def InvDiffNorm(GLCM,nG):

    I , J = np.meshgrid(np.arange (0,nG) , np.arange(0,nG))
    T = I-J
    tmp = np.divide( GLCM , (1+np.abs(T)/nG))
    f_invdiffN = np.sum(tmp)

    return f_invdiffN


def InvDiffMom(GLCM,nG):


    I , J = np.meshgrid(np.arange (0,nG) , np.arange(0,nG))
    T = I-J
    tmp = np.divide( GLCM , (1+  np.float_power( T,2)  ))
    f_invdiffM = np.sum(tmp)


    return f_invdiffM


def InvDiffMomNorm(GLCM,nG):

    I , J = np.meshgrid(np.arange (0,nG) , np.arange(0,nG))
    T = I-J
    tmp = np.divide(GLCM , (1+    np.float_power(T,2)   /   np.float_power(nG,2)    ))
    f_invdiffMN = np.sum(tmp)


    return f_invdiffMN


def InvVar(GLCM,nG):

    I , J = np.meshgrid(np.arange (0,nG) , np.arange(0,nG))
    T = I-J
    aa = np.float_power(T,2)
    tmp = np.divide (GLCM , aa )
    tmp2 = np.triu(tmp,1)
    f_invVar = 2 * np.sum(tmp2)

    return f_invVar


def Correlation(GLCM,nG):

    Pi = np.sum(GLCM,1) 
    Pj = np.sum(GLCM,0)
    Ui = np.sum(   np.multiply(np.transpose(np.arange(1,nG+1)), Pi))
    Uj = np.sum(   np.multiply(np.arange(1,nG+1), Pj) )
    Si = np.sqrt(np.sum( np.multiply( np.float_power((  np.transpose(np.arange(1,nG+1))  -Ui) ,2), Pi)))

    I , J = np.meshgrid(np.arange (1,nG+1) , np.arange(1,nG+1))
    tmp = np.multiply( np.multiply( (I - Ui) , (J - Uj)) , GLCM)
    f_corr = np.sum(tmp) / np.float_power(Si,2)
    f_corr = np.nan_to_num(f_corr,nan=0)
    # f_corr[f_corr == np.nan] = 0 

    return f_corr


def AutoCorr(GLCM,nG):

    I , J = np.meshgrid(np.arange (1,nG+1) , np.arange(1,nG+1))
    tmp = np.multiply( np.multiply( I , J) , GLCM)
    f_autocorr = np.sum(tmp)

    return f_autocorr


def ClusterTend(GLCM,nG):

    Pi = np.sum(GLCM,1) 
    Ui = np.sum(  np.multiply(np.transpose(np.arange(1,nG+1)), Pi))
    I , J = np.meshgrid(np.arange (1,nG+1) , np.arange(1,nG+1))
    tmp = np.multiply(np.float_power ((I + J - 2*Ui),2) , GLCM)
    f_clstnd = np.sum(tmp)

    return f_clstnd


def ClusterShade(GLCM,nG):

    Pi = np.sum(GLCM,1) 
    Ui = np.sum(  np.multiply(np.transpose(np.arange(1,nG+1)), Pi))
    I , J = np.meshgrid(np.arange (1,nG+1) , np.arange(1,nG+1))
    tmp = np.multiply(np.float_power ( (I + J - 2*Ui),3 ), GLCM)
    f_clsshd = np.sum(tmp)

    return f_clsshd


def ClusterProm(GLCM,nG):

    Pi = np.sum(GLCM,axis = 1) 
    Ui = np.sum(  np.multiply(np.transpose(np.arange(1,nG+1)), Pi))
    I , J = np.meshgrid(np.arange (1,nG+1) , np.arange(1,nG+1))
    tmp = np.multiply(np.float_power ((I + J - 2*Ui),4) , GLCM)
    f_clsprm = np.sum(tmp)

    return f_clsprm


def InfoCorr1(GLCM,Ent,nG):

    Pi = np.sum(GLCM,1) 
    Pj = np.sum(GLCM,0)
    tmp =  np.multiply(Pi , np.log2(Pi+ np.finfo(float).tiny))
    HX = -np.sum(tmp)
    tmp = np.multiply( Pj , np.log2(Pj+ np.finfo(float).tiny))
    HY = -np.sum(tmp)


    aa = np.multiply( np.tile(Pi,(nG,1)),np.tile(Pj,(nG,1)) ) 
    tmp =  np.multiply( GLCM , np.log2( aa + np.finfo(float).tiny  ))
    HXY1 = -np.sum(tmp)
    f_ic1 = (Ent - HXY1) / np.max([HX,HY])
    # f_ic1[f_ic1 == np.nan] = 0
    f_ic1 = np.nan_to_num(f_ic1,nan=0)



    return f_ic1


def InfoCorr2(GLCM,Ent,nG):

    Pi = np.sum(GLCM,1) 
    Pj = np.sum(GLCM,0)
    tile1 = np.transpose(np.tile(Pi,(nG,1)))
    tile2 = np.tile(Pj,(nG,1))
    aa = np.multiply(tile1,tile2)  
    tmp = np.multiply(aa , np.log2(   aa  + np.finfo(float).tiny  ))
    HXY2 = -np.sum(tmp)
    if Ent > HXY2:
        f_ic2 = 0
    else:
        f_ic2 = np.sqrt(1-np.exp(-2*(HXY2-Ent)))

    return f_ic2

def Agreement(GLCM):

    Po = np.sum(np.diag(GLCM))
    Pe = np.sum(np.diag(GLCM*GLCM))
    f_agr = (Po-Pe) / (1-Pe)
    f_agr[ np.isnan(f_agr)]= 0 

    return f_agr



def SumAvg2(GLCM,nG):

    I , J = np.meshgrid(np.arange (0,nG) , np.arange(0,nG))
    tmp = np.multiply (I , GLCM )  +   np.multiply (J , GLCM)
    f_sumavg2 = np.sum(tmp / np.float_power(nG,2))


    return f_sumavg2



def DiagProb(GLCM):

    try:
        nf = GLCM.shape[0]
    except:
        nf = 1

    pDiag = np.zeros((1,nf))
    for k in range (0,nf):
        if k==0:
            pDiag[0,k] = np.sum(np.diag(GLCM,k))
        else:
            pDiag[0,k] = np.sum(np.diag(GLCM,k)) + np.sum(np.diag(GLCM,-k))

    return pDiag


def fliplr(x):

    x = np.flip(x,1)

    return x
    
def CrossProb(GLCM):

    try:
        nG = GLCM.shape[0]
    except:
        nG = 1

    pCross = np.zeros((1,nG*2-1))
    GF = fliplr(GLCM)
    for k in range (0,nG*2-1):
        pCross[0,k] = np.sum(np.diag(GF,nG - k - 1))


    return pCross




def GLCMfeatHandler(GLCM,nG):

    try:
        nf = GLCM.shape[0]
    except:
        GLCM2 = np.zeros((1,1))
        GLCM2[0,0]= float(GLCM)
        GLCM = GLCM2.copy()

    ArrayOut = []
    ArrayOut.append(MaxProb(GLCM))
    ArrayOut.append(JointAvg(GLCM,nG))
    ArrayOut.append(JointVar(GLCM,nG,ArrayOut[1]))
    ArrayOut.append(Entropy(GLCM))
    ArrayOut.append(DiffAvg(GLCM,nG))
    ArrayOut.append(DiffVar(GLCM,nG,ArrayOut[4]))
    ArrayOut.append(DiffEnt(GLCM))
    ArrayOut.append(SumAvg(GLCM,nG))
    ArrayOut.append(SumVar(GLCM,nG,ArrayOut[7]))
    ArrayOut.append(SumEnt(GLCM))
    ArrayOut.append(Energy(GLCM))
    ArrayOut.append(Contrast(GLCM,nG))
    ArrayOut.append(Dissimilarity(GLCM,nG))
    ArrayOut.append(InvDiff(GLCM,nG))
    ArrayOut.append(InvDiffNorm(GLCM,nG))
    ArrayOut.append(InvDiffMom(GLCM,nG))
    ArrayOut.append(InvDiffMomNorm(GLCM,nG))
    ArrayOut.append(InvVar(GLCM,nG))
    ArrayOut.append(Correlation(GLCM,nG))
    ArrayOut.append(AutoCorr(GLCM,nG))
    ArrayOut.append(ClusterTend(GLCM,nG))
    ArrayOut.append(ClusterShade(GLCM,nG))
    ArrayOut.append(ClusterProm(GLCM,nG))
    ArrayOut.append(InfoCorr1(GLCM,ArrayOut[3],nG))
    ArrayOut.append(InfoCorr2(GLCM,ArrayOut[3],nG))
    # ArrayOut.append(Agreement(GLCM))
    # ArrayOut.append(SumAvg2(GLCM,nG))


    return ArrayOut





def CalcGLCM(GLCM):


    try:
        nG = GLCM.shape[0]
    except:
        nG = 1
        GLCM2 = np.zeros((1,1))
        GLCM2[0,0]= float(GLCM)
        GLCM = GLCM2.copy()
        
    if GLCM.ndim == 2:
        tmpGLCM = GLCM
        FeatMatrixout =  list(np.transpose (GLCMfeatHandler(tmpGLCM,nG)))
    else:
        FeatMatrixout = []
        for k in range (0,GLCM.shape[2]):
            tmpGLCM = np.squeeze(GLCM[:,:,k])
            FeatMatrixout.append(list((GLCMfeatHandler(tmpGLCM,nG))))
    


    return FeatMatrixout



def getrangefromclass(I):

    if isinstance(I, (int)):
        # classType = iGetClass(I)
        range = [np.min(I), np.max(I)]
    else:
        range = [0, 1]

    return range



def ParseInputs(I,gl = None ,nl = 2 ,offset = [0, 1] , sym = None):



    
    # gl = getrangefromclass(I)
    if sym is None:
        sym = False


    if offset.ndim != 2:
        raise('images:graycomatrix:invalidOffsetSize')
    
    offset = offset.astype(np.float32)


    # if isinstance(I, (int, float)):
    #     nl = 2
    # else:
    #     nl = 8


    # if nl.ndim > 1:
    #     print('images:graycomatrix:invalidNumLevels')

    if isinstance(I, (int, float)) and nl != 2:
        raise('images:graycomatrix:invalidNumLevelsForBinary')
    

    # nl = nl.astype(np.float32)



    if gl is None:
        gl = [np.min(I), np.max(I)]
    
    gl = np.array(gl)
            
    if len(gl) != 2:
        raise('images:graycomatrix:invalidGrayLimitsSize')
    
    gl = gl.astype(np.float32)
    

    return I, offset, nl, gl, sym


def accum(accmap = [],size=None):

    out = np.zeros((size[0],size[1]), dtype=np.int32)

    for i in accmap:
        out[i[0]][i[1]] = out[i[0]][i[1]] + 1 


    return out

def Union(lst1, lst2, lst3,lst4):
    final_list = list(set().union(lst1, lst2, lst3,lst4))
    return final_list

def computeGLCM(r,c,offset,si,nl):


    r2 = r + offset[0]
    c2 = c + offset[1]

    nRow, nCol = si.shape
    
    a_find = np.where(c2 < 0)[0]
    b_find = np.where(c2 > nCol - 1) [0]
    c_find = np.where(r2 < 0)[0]
    d_find = np.where(r2 > nRow - 1)[0]


    outsideBounds =  Union(a_find , b_find , c_find  , d_find)

    v1 = np.moveaxis(si, 0, 1)
    # v1 = shiftdim(si,1)
    v1 = v1.flatten(order='F')


    v1 = np.delete(v1,outsideBounds,None).astype(np.int32)
    r2 = np.delete(r2,outsideBounds,None).astype(np.int32)
    c2 = np.delete(c2,outsideBounds,None).astype(np.int32)

    Index = r2 + np.multiply(nRow,(c2)).astype(np.int32)

    si_flatten = si.flatten(order='F')


    v2 = si_flatten[Index]


    bad1 = v1 == np.nan
    bad2 = v2 == np.nan
    bad = np.logical_or(bad1, bad2) 
    if np.any(bad):
        raise('images:graycomatrix:scaledImageContainsNan')
    

    Ind = np.column_stack((v1, v2))
    bad = np.multiply(bad,1)
    bad = np.where(bad == 1)[0]
    Ind = np.delete(Ind,bad,axis=0).astype(np.int32)
    
    if len(Ind) == 0:
        oneGLCM = np.zeros((nl))
    else:
        # Ind, 1, [nl, nl]
        Ind = Ind - 1
        oneGLCM = accum(accmap = Ind ,size=[nl, nl] )
    
    
    return oneGLCM


def graycomatrix(I,GrayLimits,NumLevels ,Offset,Symmetric):

    I, Offset, NL, GL, makeSymmetric = ParseInputs(I,GrayLimits,NumLevels ,Offset,Symmetric)


    if GL[1] == GL[0]:
        SI = np.ones((I.shape))
    else :
        slope = NL / (GL[1] - GL[0])
        intercept = 1 - (slope*(GL[0]))
        
        result = np.multiply(slope , I).astype(np.float32) + intercept
        SI = np.floor(result)
        # aa = imlincomb(slope,I,intercept)
    


    SI[SI > NL] = NL
    SI[SI < 1] = 1

    numOffsets = Offset.shape[0]



    if NL != 0:

        s = I.shape

        r,c = np.meshgrid( np.arange (0,s[0]) , np.arange(0,s[1]))
        r = r.flatten(order='F')
        c = c.flatten(order='F')

        GLCMS = np.zeros((NL,NL,numOffsets))
        for k in range(0 , numOffsets):
            GLCMS[:,:,k] = computeGLCM(r,c,Offset[k,:],SI,NL)
            
            if makeSymmetric: 
                glcmTranspose =  np.transpose(GLCMS[:,:,k])
                GLCMS[:,:,k] = GLCMS[:,:,k] + glcmTranspose

    else:
        GLCMS = np.zeros((0,0,numOffsets))

    # print(np.mean(GLCMS))
    return GLCMS,SI




def getGLCM(ROIonly,levels):


    nLevel = len(levels)
    if nLevel > 100:
        adjust = 10000
    else:
        adjust = 1000
    
    levelTemp = np.max(levels)+1
    ROIonly = np.nan_to_num(ROIonly, nan=levelTemp)
    levels = np.append(levels, levelTemp)

    dim = ROIonly.shape
    if ROIonly.ndim == 2 :
        dim[2] = 1
    
    q2 = np.reshape(ROIonly,(1,np.prod(dim)),order='F')

    qs = np.round(np.multiply(adjust ,levels))/adjust
    q2 = np.round(np.multiply(adjust ,q2))/adjust

    q3 = q2*0
    for k in range(0,len(qs)):
        q3[q2==qs[k]] = k+1


    q3 = np.reshape(q3,dim,order='F')
    lqs = len(qs)
    GLCM = np.zeros((lqs,lqs))
    for i in range(0,dim[0]):
        for j in range(0,dim[1]):
            for k in range(0,dim[2]):
                val_q3 = int(q3[i,j,k])
                i_min = np.max([1,i])
                i_max = np.min([i+2,dim[0]])
                j_min = np.max([1,j])
                j_max = np.min([j+2,dim[1]])
                k_min = np.max([1,k])
                k_max = np.min([k+2,dim[2]])

                for I2 in range(i_min-1,i_max): 
                    for J2 in range(j_min-1,j_max): 
                        for K2 in range(k_min-1,k_max): 
                            if I2 == i and J2 == j and K2 == k:
                                continue
                            else:
                                val_neighbor = int(q3[I2,J2,K2])
                                new_Value = GLCM[val_q3-1,val_neighbor-1] + 1
                                GLCM[val_q3-1,val_neighbor-1] = new_Value


    GLCM = GLCM[0:-1,0:-1]

    return GLCM




def grayLevelMatrixSingle(img, numLevels, offset, symmetric):

    glcmSingle = np.zeros((numLevels, numLevels))
    
    d1N = img.shape[0]
    d2N = img.shape[1]
    d3N = img.shape[2]

    for d1 in range(0,d1N):
        for d2 in range( 0,d2N):
            for d3 in range( 0,d3N):
                v1 = int(img[d1,d2,d3])

                if v1 != 0:

                    t1 = d1 + offset[0]
                    t2 = d2 + offset[1]
                    t3 = d3 + offset[2]

                    if t1>-1 and t1<d1N and t2>-1 and t2<d2N and t3>-1 and t3<d3N:
                        v2 = int(img[t1, t2, t3])

                        if v2 != 0:
                            glcmSingle[v1-1,v2-1] = glcmSingle[v1-1,v2-1] + 1
                            if symmetric:
                                glcmSingle[v2-1,v1-1] = glcmSingle[v2-1,v1-1] + 1


    return glcmSingle


def graycooc3d_bis(I,ROI,distance,numLevels,offSet,normflag):

        
    noDirections = offSet.shape[0]
    coMat = np.zeros((numLevels,numLevels,noDirections))

    if normflag.ndim == 1:
        if normflag.shape[0]==2:
            minImage=normflag[0]
            maxImage=normflag[1] - minImage

            I2 = I[ROI == 1]
            if minImage > np.min(I2):
                raise('min too large in graycooc3d_bis.m')
            
            if normflag[1] < np.max(I2) :
                raise('max too small in graycooc3d_bis.m')
            
            I=I-minImage
        else:
            I2 = I[ROI == 1]
            minImage = np.min(I2)
            I=I-minImage
            maxImage = np.max(I2)
    else:
        I2 = I[ROI == 1]
        minImage = np.min(I2)
        I=I-minImage
        maxImage = np.max(I2)
        
    tempShift = maxImage / numLevels
    if tempShift==0:
        I[ROI == 1]=1
    else:
        I = np.ceil( I.astype(np.float32)  / tempShift )
        I[I == 0]=1
    
    # newROI = ROI[ROI > 0]
    # newROI = np.reshape(newROI,I.shape,order='F')
    I =  np.multiply(I,ROI)

    newI = I[ROI>0]
    if np.max(newI) > numLevels:
        raise('Error in graylevel resizing.')
        # print('graycooc3d_bis.m')

    # fin = np.where(ROI > 0)
    
    # np.save('ROI.npy', ROI)
    # np.save('I.npy', I)
    # np.save('offSet.npy', offSet)

    d1,d2,d3 = ind2sub(ROI)

    ROIflatten = ROI.flatten(order='F')


    d1 = d1[ROIflatten>0]
    d2 = d2[ROIflatten>0]
    d3 = d3[ROIflatten>0]


    Icropped = I[np.min(d1):np.max(d1)+1, np.min(d2):np.max(d2)+1, np.min(d3):np.max(d3)+1]


    for direction in range(0,noDirections):
        newoffSet = distance * offSet[direction,:]
        coMat[:,:,direction] = grayLevelMatrixSingle(Icropped, numLevels, newoffSet , True)


    return coMat

def GLCM_3D(ROIonly,new_ROIonly=None , normflag = [1,1] , DIRECTION=None, DISTANCE = None ,NUMGRAY = None , COMBINE=None, ACCUMULATE = None):


    coocMat = np.nan
    # DISTANCE = np.array([1,2,4,8])
    numLevels = 16


    offSet = [[0, 1, 0], [-1, 1, 0], [-1, 0, 0], [-1, -1, 0]]
    dimension3 = [[0, 1, -1], [0, 0, -1], [0, -1, -1], [-1, 0, -1], [1, 0, -1], [-1, 1, -1],[1, -1, -1],[-1, -1, -1], [1, 1, -1]]
    offSet = offSet + dimension3
    offSet = np.array(offSet).astype(np.int32)


    data = ROIonly.copy()
    temp = data.shape
    if len(temp) < 3:
        raise('Error: This program is designed for 3 dimensional data')

    if new_ROIonly is not None:
        # new_ROIonly = np.reshape(new_ROIonly,data.shape,order='F')
        ROI = new_ROIonly.copy()
        temp = ROI.shape
        if len(temp)<3:
            raise('Error: This program is designed for 3 dimensional data (ROI)')
    else:
        ROI= np.ones(data.shape)
    
    # new_Roi = ROI[ROI>0]
    new_Roi = ROI.copy()
    data= np.multiply (data , new_Roi)

    normflag = np.array(normflag).astype(np.float32)


    if DIRECTION is not None:
        temp2 = np.array(DIRECTION).astype(np.int32)
        if len(temp2.shape) != 2:
            raise('Error: Direction input is formatted poorly')

        if temp2.shape[1] != 3:
            raise('Error: Incorrect number of columns in direction variable')

        if np.max(temp2) > 1 or np.min(temp2) < -1 :
            raise('Error: Direction values can only be {-1,0,1}')

        offSet = temp2.copy()

    if DISTANCE is not None:
        temp2 = np.array(DISTANCE).astype(np.int32)
        # if len(temp2.shape) != 2:
        #     print('Error: Incorrect formatting of distance variable')
        
        # if np.sum(temp2.shape) != (np.max(temp2.shape) +1 ):
        #     print('Error: Distance variable is to be a one dimensional array')
        distance = temp2.copy()
        d = 1

    if NUMGRAY is not None:
        temp2 = NUMGRAY
        if temp2<1:
            raise('The number of graylevels must be positive')
        # numLevels = temp2.astype(np.uint16)
        numLevels = temp2

    if COMBINE is not None:
        temp2 = COMBINE.copy()
        combine_all_coMat=temp2.copy()

    if ACCUMULATE is not None:
        temp2 = ACCUMULATE.copy()
        accumulate_distance=temp2.copy()


    noDirections = offSet.shape[0]
    coocMat = np.zeros((numLevels, numLevels, noDirections, d))

    # for dist in range(0,d):
    coocMat[:,:,:,0] = graycooc3d_bis(data,ROI,distance,numLevels,offSet,normflag)

    return coocMat




