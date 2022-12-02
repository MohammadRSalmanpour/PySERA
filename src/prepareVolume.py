from pickle import NONE
from scipy.io import savemat, loadmat
import numpy as np
from Quantization import fixedBinSizeQuantization,uniformQuantization,lloydQuantization
from interpolation import imresize3D,imresize




def computeBoundingBox(Data_ROI_mat):
    [iV,jV,kV] = np.nonzero(Data_ROI_mat)
    boxBound = np.zeros((3,2))
    boxBound[0,0] = np.min(iV)
    boxBound[0,1] = np.max(iV)+1
    boxBound[1,0] = np.min(jV)
    boxBound[1,1] = np.max(jV)+1
    boxBound[2,0] = np.min(kV)
    boxBound[2,1] = np.max(kV)+1
    boxBound = boxBound.astype(np.uint32)


    # x_ind, y_ind, z_ind = np.where(Data_ROI_mat > 0.0)

    # if len(z_ind) == 0 or len(y_ind) == 0 or len(x_ind) == 0:
    #     boxBound = [[0,0],[0,0],[0,0]]
    # else:
    #     boxBound = [[np.min(x_ind), np.max(x_ind)],
    #                 [np.min(y_ind), np.max(y_ind)],
    #                 [np.min(z_ind), np.max(z_ind)]]




    # boxBound = np.asarray(boxBound)                    
    # boxBound = boxBound.astype(np.uint32)

    return boxBound


def CollewetNorm(ROIonly):


    temp = ROIonly[~np.isnan(ROIonly)]
    u = np.mean(temp)
    sigma = np.std(temp)

    ROIonlyNorm = ROIonly.copy()
    ROIonlyNorm[ROIonly > (u + 3*sigma)] = np.nan
    ROIonlyNorm[ROIonly < (u - 3*sigma)] = np.nan

    return ROIonlyNorm


def roundGL(Img , isGLrounding):

    if isGLrounding == 1:
        GLroundedImg = np.round(Img)
    else:
        GLroundedImg = Img.copy()
     
    return GLroundedImg

def getMutualROI(ROI1, ROI2):

    tmp1 = np.multiply(ROI1 , ROI2)
    tmp2=tmp1.copy()
    tmp2[~np.isnan(tmp1)] = 1
    outROI = np.multiply(tmp2 , ROI1)


    return outROI

def prepareVolume(volume,Mask,DataType,pixelW,sliceTh,
                newVoxelSize,VoxInterp,ROIInterp,ROI_PV,scaleType,isIsot2D,
                isScale,isGLround,DiscType,qntz,Bin,
                isReSegRng,ReSegIntrvl,isOutliers):

    if DiscType == 'FBS':
        quantization = fixedBinSizeQuantization
    elif DiscType == 'FBN':
        quantization = uniformQuantization
    else:
        print('Error with discretization type. Must either be "FBS" (Fixed Bin Size) or "FBN" (Fixed Number of Bins).')


    if qntz == 'Lloyd':
        quantization = lloydQuantization


    ROIBox = Mask.copy()
    Imgbox = volume.copy()

    Imgbox = Imgbox.astype(np.float32)

    if DataType == 'MRscan':
        ROIonly = Imgbox.copy()
        ROIonly[ROIBox == 0] = np.nan
        temp = CollewetNorm(ROIonly)
        ROIBox[np.isnan(temp)] = 0

    flagPW = 0
    if scaleType=='NoRescale':
        flagPW = 0
    elif scaleType=='XYZscale':
        flagPW = 1
    elif scaleType=='XYscale':
        flagPW = 2
    elif scaleType=='Zscale':
        flagPW = 3
    

    if isIsot2D == 1:
        flagPW = 2
    

    if isScale == 0:
        flagPW = 0
    

    if flagPW == 0:
        a = 1
        b = 1 
        c = 1
    elif flagPW ==1:
        a = pixelW/newVoxelSize
        b = pixelW/newVoxelSize
        c = sliceTh/newVoxelSize
    elif flagPW == 2:
        a = pixelW/newVoxelSize
        b = pixelW/newVoxelSize
        c = 1
    elif flagPW == 3:
        a = 1
        b = 1
        c = sliceTh/pixelW
    
    # Resampling
    ImgBoxResmp = Imgbox.copy() 
    ImgWholeResmp = volume.copy()
    ROIBoxResmp = ROIBox.copy() 
    ROIwholeResmp = Mask.copy()


    if Imgbox.ndim == 3 and flagPW != 0:
        if (a + b + c) != 3:
            ROIBoxResmp = imresize3D(ROIBox,[pixelW, pixelW, sliceTh],[np.ceil(ROIBox.shape[0]*a),np.ceil(ROIBox.shape[1]*b),np.ceil(ROIBox.shape[2]*c)],ROIInterp ,'constant',[newVoxelSize,newVoxelSize,newVoxelSize],isIsot2D)
            Imgbox[np.isnan(Imgbox)] = 0
            ImgBoxResmp = imresize3D(Imgbox,[pixelW, pixelW, sliceTh],[np.ceil(Imgbox.shape[0]*a),np.ceil(Imgbox.shape[1]*b),np.ceil(Imgbox.shape[2]*c)],VoxInterp,'constant',[newVoxelSize,newVoxelSize,newVoxelSize],isIsot2D)
            ROIBoxResmp[ROIBoxResmp < ROI_PV] = 0
            ROIBoxResmp[ROIBoxResmp >= ROI_PV] = 1 
            ROIwholeResmp = imresize3D(Mask,[pixelW, pixelW, sliceTh],[np.ceil(Mask.shape[0]*a),np.ceil(Mask.shape[1]*b),np.ceil(Mask.shape[2]*c)],ROIInterp,'constant',[newVoxelSize,newVoxelSize,newVoxelSize],isIsot2D)
            ImgWholeResmp = imresize3D(volume,[pixelW, pixelW, sliceTh],[np.ceil(volume.shape[0]*a),np.ceil(volume.shape[1]*b),np.ceil(volume.shape[2]*c)],VoxInterp ,'constant',[newVoxelSize,newVoxelSize,newVoxelSize],isIsot2D)
            if np.max(ROIwholeResmp) < ROI_PV:
                print('Resampled ROI has no voxels with value above ROI_PV. Cutting ROI_PV to half.')
                ROI_PV = ROI_PV/2
            
            ROIwholeResmp[ROIwholeResmp<ROI_PV] = 0
            ROIwholeResmp[ROIwholeResmp>=ROI_PV] = 1

    elif Imgbox.ndim == 2 and flagPW !=0:
        if (a + b) != 2:
            ROIBoxResmp = imresize(ROIBox,[pixelW, pixelW],[np.ceil(ROIBox.shape[0]*a),np.ceil(ROIBox.shape[1]*b)],ROIInterp,[newVoxelSize,newVoxelSize])
            ImgBoxResmp = imresize(Imgbox,[pixelW, pixelW],[np.ceil(Imgbox.shape[0]*a),np.ceil(Imgbox.shape[1]*b)],VoxInterp,[newVoxelSize,newVoxelSize])
            ROIBoxResmp[ROIBoxResmp<ROI_PV] = 0
            ROIBoxResmp[ROIBoxResmp>=ROI_PV] = 1

            ROIwholeResmp = imresize(Mask,[pixelW, pixelW],[np.ceil(Mask.shape[0]*a),np.ceil(Mask.shape[1]*b)],ROIInterp,[newVoxelSize,newVoxelSize])
            ImgWholeResmp = imresize(volume,[pixelW, pixelW],[np.ceil(volume.shape[0]*a),np.ceil(volume.shape[1]*b)],VoxInterp,[newVoxelSize,newVoxelSize])
            if np.max(ROIwholeResmp) < ROI_PV:
                print('Resampled ROI has no voxels with value above ROI_PV. Cutting ROI_PV to half.')
                ROI_PV = ROI_PV/2
            
            ROIwholeResmp[ROIwholeResmp<ROI_PV] = 0
            ROIwholeResmp[ROIwholeResmp>=ROI_PV] = 1
    

    

    IntsBoxROI = ImgBoxResmp.copy()

    ImgBoxResmp[ROIBoxResmp == 0] = np.nan

    IntsBoxROI = roundGL(ImgBoxResmp , isGLround)
    ImgWholeResmp = roundGL(ImgWholeResmp , isGLround)

    
    IntsBoxROItmp1 = IntsBoxROI.copy()
    ImgWholeResmptmp1 = ImgWholeResmp.copy()
    IntsBoxROItmp2 = IntsBoxROI.copy()
    ImgWholeResmptmp2 = ImgWholeResmp.copy()

    if isReSegRng == 1:
        IntsBoxROItmp1[IntsBoxROI<ReSegIntrvl[0]] = np.nan
        IntsBoxROItmp1[IntsBoxROI>ReSegIntrvl[1]] = np.nan
        ImgWholeResmptmp1[ImgWholeResmp<ReSegIntrvl[0]] = np.nan
        ImgWholeResmptmp1[ImgWholeResmp>ReSegIntrvl[1]] = np.nan
    

    if isOutliers == 1:
        Mu = np.nanmean(IntsBoxROI)
        Sigma = np.nanstd(IntsBoxROI)
        IntsBoxROItmp2[IntsBoxROI<(Mu-3*Sigma)] = np.nan
        IntsBoxROItmp2[IntsBoxROI>(Mu+3*Sigma)] = np.nan
            
        Mu = np.nanmean(ImgWholeResmp)
        Sigma = np.nanstd(ImgWholeResmp)
        ImgWholeResmptmp2[ImgWholeResmp<(Mu-3*Sigma)] = np.nan
        ImgWholeResmptmp2[ImgWholeResmp>(Mu+3*Sigma)] = np.nan
    

    IntsBoxROI      = getMutualROI(IntsBoxROItmp1 , IntsBoxROItmp2)
    ImgWholeResmp   = getMutualROI(ImgWholeResmptmp1 , ImgWholeResmptmp2)


    newpixelW = pixelW / a
    newsliceTh = sliceTh / c


    if DataType ==  'PET':
        minGL = 0
    elif DataType == 'CT':
        if isReSegRng== 1:
            minGL = ReSegIntrvl[0]
        else:
            minGL = np.nanmin(IntsBoxROI)
        
    else:
        minGL = np.nanmin(IntsBoxROI)
    

    ImgBoxResampQuntz3D,levels = quantization(IntsBoxROI,Bin,minGL)



    boxBound = computeBoundingBox(ROIBoxResmp)
    MorphROI = ROIBoxResmp[boxBound[0][0]:boxBound[0][1],boxBound[1][0]:boxBound[1][1],boxBound[2][0]:boxBound[2][1]]
    IntsBoxROI = IntsBoxROI[boxBound[0][0]:boxBound[0][1],boxBound[1][0]:boxBound[1][1],boxBound[2][0]:boxBound[2][1]]
    ImgBoxResampQuntz3D = ImgBoxResampQuntz3D[boxBound[0][0]:boxBound[0][1],boxBound[1][0]:boxBound[1][1],boxBound[2][0]:boxBound[2][1]]
    # ImgWholeResmp = ImgWholeResmp[boxBound[0][0]:boxBound[0][1],boxBound[1][0]:boxBound[1][1],boxBound[2][0]:boxBound[2][1]]

    return ImgBoxResampQuntz3D,levels,MorphROI,IntsBoxROI,ImgWholeResmp,ROIwholeResmp,newpixelW,newsliceTh

def getImgBox(volume,mask,isReSeg,ResegIntrval):
    
    boxBound = computeBoundingBox(mask)
    # maskBox = mask[boxBound[0][0]:boxBound[0][1],boxBound[1][0]:boxBound[1][1],boxBound[2][0]:boxBound[2][1]]
    SUVbox = volume[boxBound[0][0]:boxBound[0][1],boxBound[1][0]:boxBound[1][1],boxBound[2][0]:boxBound[2][1]]

    if isReSeg == 1:
        SUVbox[SUVbox<ResegIntrval[0]] = np.nan
        SUVbox[SUVbox>ResegIntrval[1]] = np.nan

    return SUVbox
