
import scipy 
import cmath

from SERAutilities import *

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


def getMorph(MorphBox2,MorphROI2,ROIints2, pixelW,sliceS):
    
    MorphVect = []
    MorphROI = MorphROI2.copy()
    ROIints = ROIints2.copy()
    MorphBox = MorphBox2.copy()
    # MorphROI[MorphROI == 0] = np.nan
    ROIboxF,ROIboxV , _ = getMesh(MorphROI,pixelW,pixelW,sliceS)

    MV, Surface = getVolSurfaceMesh(ROIboxF,ROIboxV)
    # from skimage.measure import mesh_surface_area
    # ROIboxV = ROIboxV - 1
    # ROIboxF = ROIboxF - 1
    # area = mesh_surface_area(verts=ROIboxV.astype(np.uint32), faces=ROIboxF.astype(np.uint32))

    MorphROI[MorphROI==0] = np.nan
    MVappx = getVolume(MorphROI,pixelW,sliceS)

    xM,yM,zM = ind2sub(MorphROI)
    xI,yI,zI = ind2sub(ROIints) 
    try:
        concat = np.column_stack((xM*pixelW, yM*pixelW, zM*sliceS)) + 1
        concat2 = np.column_stack((xI*pixelW, yI*pixelW, zI*sliceS)) + 1
        aa = np.nanmean( concat , axis=0 )
        ROIints_flatten = ROIints.flatten(order='F')
        notnanval = np.where(~np.isnan(ROIints_flatten))
        arr = np.expand_dims(ROIints_flatten[notnanval[0]],-1)
        tile = np.tile(arr,reps=(1,3))
        bb = np.nansum( np.multiply( concat2 , tile)  , axis= 0  )
        cc = np.tile(np.nansum(ROIints),(1,3))
        dd = aa - np.divide(bb,cc)

        CoM_shift = np.linalg.norm(dd)

    except:
        raise('**** ROI is 1D. Brace for problems.....')
        CoM_shift = np.abs(np.mean(np.squeeze(MorphROI)-np.squeeze(ROIints))) 
    

    Shape = getShape(MV,Surface)
    compactness1 = Shape['compactness1'] 
    compactness2 = Shape['compactness2']
    sphericity = Shape['sphericity']
    SVratio = Shape['SVratio']
    Irregularity = Shape['Irregularity']
    sphericalDisp = Shape['sphericalDisp']
    Asphericity = Shape['Asphericity']


    # Volume Densities 
    VDs, ConvHullV = getVDs(MorphROI,ROIboxV,pixelW,sliceS,MV,Surface )
    Mean = np.nanmean(ROIints)
    TLG = Mean * MV
    Max3Ddiam = getMax3Ddiam(ROIboxV, ConvHullV)



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
    MoranI = np.nan
    GearyC = np.nan 
    # # MoranI, GearyC = getAutoCorrs(ROIints, xI,yI,zI , pixelW,sliceS , Mean) 
    MoranI, GearyC = getAutoCorrs(ROIints, xI,yI,zI,pixelW,sliceS ) 
    # MoranI, GearyC = getAutoCorrs(ROIints_flatten, xI,yI,zI,pixelW,sliceS ) 

    threshold=2
    # GT2 = getVolume_gt(MorphBox,threshold,pixelW,sliceS)

    MorphVect = [MV, MVappx, Surface, SVratio, compactness1, compactness2, 
            sphericalDisp, sphericity, Asphericity, CoM_shift, Max3Ddiam]
    for v in VDs:
        MorphVect.append(v)

    MorphVect.append(TLG)
    MorphVect.append(MoranI)
    MorphVect.append(GearyC)
    
    return MorphVect



def getMesh(ROIbox,pixelWx,pixelWy,sliceS):


    dx,dy,dz= ROIbox.shape

    morphology_mask = np.pad(ROIbox, pad_width= ((1,1),(1,1),(1,1)), mode="constant", constant_values=0.0)

    rangeX = np.arange(0, np.ceil(pixelWx*(dx+2)) , pixelWx)
    rangeY = np.arange(0, np.ceil(pixelWy*(dy+2)) , pixelWy)
    rangeS = np.arange(0, np.ceil(sliceS*(dz+2)) , sliceS)

    X,Y,Z= np.meshgrid(rangeX,rangeY,rangeS,indexing='ij')

    if X.shape != Y.shape or Y.shape != Z.shape or Z.shape != morphology_mask.shape:
        rangeX = np.arange(0, np.ceil(pixelWx*(dx+2)-1) , pixelWx)
        rangeY = np.arange(0, np.ceil(pixelWy*(dy+2)-1) , pixelWy)
        rangeS = np.arange(0, np.ceil(sliceS*(dz+2)-1) , sliceS)

        X,Y,Z= np.meshgrid(rangeX,rangeY,rangeS,indexing='ij')

    faces , vertices, norms = MarchingCubes(X,Y,Z,morphology_mask,0.5,None)


    # from skimage.measure import marching_cubes

    # # Get ROI and pad with empty voxels
    # morphology_mask = np.pad(ROIbox, pad_width=1, mode="constant", constant_values=0.0)

    # # Use marching cubes to generate a mesh grid for the ROI
    # vertices, faces, norms, values = marching_cubes(volume=morphology_mask, level=0.5, spacing=(float(pixelWx),float(pixelWy),float(sliceS)))


    return faces ,vertices, norms

    


def getVolSurfaceMesh(F,V):


    nFaces = F.shape[0]
    vols = np.zeros((nFaces, 1))
    meanMat = np.mean(V,axis=0)
    vertices = V - meanMat
    F = F.astype(np.int64)
    for i in range(0,nFaces):
        ind = list(F[i,:] - 1)
        tetra = vertices[ind,:]
        
        vols[i] = np.linalg.det(tetra) / 6

    vol_out = np.abs(np.sum(vols))


    a = V[list(F[:, 1]-1), :] - V[list(F[:, 0]-1), :]
    b = V[list(F[:, 2]-1), :] - V[list(F[:, 0]-1), :]
    c = np.cross(a,b)
    area_out = 1/2 * np.sum(np.sqrt(np.sum(np.float_power(c,2), axis=1)))


    return vol_out, area_out


def getVolume(ROIonly,pixelW,sliceS):


    # mask = ROIonly[~isnan(ROIonly)] 
    numberVoxel = np.nansum(ROIonly)
    volume = numberVoxel * pixelW * pixelW * sliceS

    if hasattr(volume, '__len__') == True:
        volume = np.squeeze(volume)

    return volume


def getShape(V,S):

        
    compactness1 = V / np.sqrt(np.pi) / np.sqrt(np.float_power(S,3))
    compactness2 = 36 * np.pi * np.float_power(V,2) / np.float_power(S,3)
    sphericity = np.float_power(( 36*np.pi * np.float_power(V,2)) , (1/3)) / S
    sphericalDisp = S / np.float_power(( 36*np.pi * np.float_power(V,2)) , (1/3))
    Asphericity =  np.float_power( (np.float_power(S,3) / np.float_power(V,2) / 36 / np.pi),(1/3)) - 1 
    SVratio = S/V
    Scircle = np.float_power((3  *  V /   (4*np.pi))  ,(2/3))  * 4 * np.pi
    Irregularity = S/Scircle

    Shape = {}
    Shape['compactness1'] = compactness1
    Shape['compactness2'] = compactness2
    Shape['sphericity'] = sphericity
    Shape['SVratio'] = SVratio
    Shape['Irregularity'] = Irregularity
    Shape['sphericalDisp'] = sphericalDisp
    Shape['Asphericity'] = Asphericity


    return Shape


def getVDs(ROIbox,ROIboxV,pixelW,sliceS,MV,Surface ):

    x,y,z=ind2sub(ROIbox)
    
    concat = np.column_stack((x*pixelW, y*pixelW, z*sliceS)) + 1

    covariance = np.cov(np.transpose(concat))
    EIGs =  np.linalg.eigh(covariance)[0]  # EIGs =  scipy.linalg.eigh(covariance)[0]
    

    EIGs = np.transpose(np.flip(EIGs))

    if EIGs.shape[0] == 1:
        EIGs = np.tile(EIGs,(1,3))
    
    L1 = 2* np.sqrt(EIGs[0]) 
    L2 = 2* np.sqrt(EIGs[1]) 
    L3 = 2* np.sqrt(EIGs[2])


    MALs = 4*np.sqrt(EIGs)                
    Elong = np.sqrt(EIGs[1] / EIGs[0])        
    Fltns = np.sqrt(EIGs[2] / EIGs[0])       

    nx, ny, nz = ROIbox.shape

    VDaabb = MV / np.prod( np.multiply( np.array([nx, ny ,nz]), np.transpose(np.array([pixelW,pixelW,sliceS]))  ))    
    ADaabb = (Surface / (2*nx*pixelW*ny*pixelW + 2*ny*pixelW*nz*sliceS + 2*nx*pixelW*nz*sliceS))

    if hasattr(ADaabb, '__len__') == True:
        ADaabb = np.squeeze(ADaabb)
    
    VDaee  = 3*MV / (4*np.pi*  np.float_power(2,3)*np.prod(np.sqrt(EIGs)))


    alpha = np.sqrt(1 -  np.float_power(L2,2)  /   np.float_power(L1,2))
    beta = np.sqrt(1-np.float_power(L3,2)/np.float_power(L1,2))
    MaxV = 20

    v0 = np.arange(0,MaxV+1,1)
    sumtmp = []
    aa = (  np.float_power (alpha,2)+ np.float_power(beta,2)  ) / (2*alpha*beta)
    mm = np.min([1,aa])
    for k in range(0, MaxV+1):

        bb = np.transpose (np.array(legendre(k , np.array(mm)  )) )
        cc = (1-4*np.float_power(k,2))
        ee = np.float_power((alpha*beta),k)
        ff = np.divide (ee , cc)
        dd = np.multiply(ff , bb)
        if np.isscalar(dd):
            tem = dd
            dd = np.array([tem])
        else:
            dd = dd[0]    
        sumtmp.append(dd)

    Final_sumtmp = []
    for i in range (0 , len(sumtmp)):
        for j in sumtmp[i]:
            Final_sumtmp.append(j)


    sumtmp = np.array(Final_sumtmp)
    aee = 4*np.pi*L1*L2 * np.sum(sumtmp)
    ADaee = Surface / aee

    covariance = np.cov(np.transpose(ROIboxV))
    EIGs = np.linalg.eigh(covariance)[0]
    # EIGs = np.sum(EIGs , axis = 0)
    L1 = 2*np.sqrt(EIGs[0])
    L2 = 2*np.sqrt(EIGs[1])  
    L3 = 2*np.sqrt(EIGs[2]) 
    VDmvee = MV / (4*np.pi*L1*L2*L3/3)    
    
    
    aa = (np.float_power(L2,2)/np.float_power(L1,2))
    bb = (np.float_power(L3,2)/np.float_power(L1,2))
    alpha = cmath.sqrt(1-aa)
    beta = cmath.sqrt(1-bb)

    try:
        yy = 4*np.pi*L1*L2

        aa = (  np.float_power (alpha,2)+ np.float_power(beta,2)  ) / (2*alpha*beta)
        mm = np.min([1,aa])
        bb = np.transpose (np.array(legendre(MaxV , np.array(mm)  )) )
        cc = (1-4*np.float_power(v0,2))
        ee = np.float_power((alpha*beta),v0)
        ff = np.divide (ee , cc)
        dd = np.multiply(ff , bb)

        if np.isscalar(dd):
            tem = dd
            dd = np.array([tem])
        else:
            dd = dd[0]    

        aee = yy * np.sum(dd)

    except:
        EIGs = np.transpose(np.flip(EIGs))
        
        L1 = 2* np.sqrt(EIGs[0]) 
        L2 = 2* np.sqrt(EIGs[1]) 
        L3 = 2* np.sqrt(EIGs[2])

        VDmvee = MV / (4*np.pi*L1*L2*L3/3)

        alpha = cmath.sqrt(1-(np.float_power(L2,2)/np.float_power(L1,2)))
        beta = cmath.sqrt(1-(np.float_power(L3,2)/np.float_power(L1,2)))


        yy = 4*np.pi*L1*L2

        aa = (  np.float_power (alpha,2)+ np.float_power(beta,2)  ) / (2*alpha*beta)
        mm = np.min([1,aa])
        bb = np.transpose (np.array(legendre(MaxV , np.array(mm)  )) )
        cc = (1-4*np.float_power(v0,2))
        ee = np.float_power((alpha*beta),v0)
        ff = np.divide (ee , cc)
        dd = np.multiply(ff , bb)

        if np.isscalar(dd):
            tem = dd
            dd = np.array([tem])
        else:
            dd = dd[0]    

        aee = yy * np.sum(dd)

    aee = np.real(aee)
    ADmvee = Surface / aee


    try:
        _,_,Vombb,Aombb,_ = minboundbox(x*pixelW, y*pixelW, z*sliceS,'volume',1)
    except:
        raise('This is a 2D or 1D ROI. Switch to 2D Convex Hull and Bounding Box calculation.')
        newROIbox = np.squeeze(ROIbox)
        xtmp,ytmp,_=ind2sub(newROIbox)
        try:
            if ROIbox.shape[2] == 1:
                _, Aombb, Vombb = minBoundingBox2D(xtmp , ytmp,pixelW,pixelW,sliceS)
            elif ROIbox.shape[1] == 1:
                _, Aombb, Vombb = minBoundingBox2D(xtmp , ytmp,pixelW,sliceS,pixelW)
            elif ROIbox.shape[0] == 1:
                _, Aombb, Vombb = minBoundingBox2D(xtmp , ytmp,sliceS,pixelW,pixelW)
            else:
                raise('Min bounding box does not respond (no Convex Hull available). Set OMBB = AABB....')
                arr1 = np.column_stack((nx, ny, nz))
                arr2 = np.column_stack((pixelW,pixelW,sliceS))

                Vombb = np.prod(np.multiply(arr1,arr2))
                Aombb = 2*nx*pixelW*ny*pixelW + 2*ny*pixelW*nz*sliceS + 2*nx*pixelW*nz*sliceS
                
            
        except:
            raise('Min bounding box does not respond (probably a 1D ROI). Set OMBB = AABB....')
            arr1 = np.column_stack((nx, ny, nz))
            arr2 = np.column_stack((pixelW,pixelW,sliceS))
            Vombb = np.prod(np.multiply(arr1,arr2))
            Aombb = 2*nx*pixelW*ny*pixelW + 2*ny*pixelW*nz*sliceS + 2*nx*pixelW*nz*sliceS
        
            
    
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
                    np.multiply(ROIboxV[K[:,0],1],ROIboxV[K[:,1],2]) - np.multiply(ROIboxV[K[:,0],2],ROIboxV[K[:,1],1]),
                    np.multiply(ROIboxV[K[:,0],2],ROIboxV[K[:,1],0]) - np.multiply(ROIboxV[K[:,0],0],ROIboxV[K[:,1],2]),
                    np.multiply(ROIboxV[K[:,0],0],ROIboxV[K[:,1],1]) - np.multiply(ROIboxV[K[:,0],1],ROIboxV[K[:,1],0])))
                + 
                np.column_stack((
                    np.multiply(ROIboxV[K[:,1],1],ROIboxV[K[:,2],2]) - np.multiply(ROIboxV[K[:,1],2],ROIboxV[K[:,2],1]),
                    np.multiply(ROIboxV[K[:,1],2],ROIboxV[K[:,2],0]) - np.multiply(ROIboxV[K[:,1],0],ROIboxV[K[:,2],2]),
                    np.multiply(ROIboxV[K[:,1],0],ROIboxV[K[:,2],1]) - np.multiply(ROIboxV[K[:,1],1],ROIboxV[K[:,2],0])))
                + 
                np.column_stack((
                    np.multiply(ROIboxV[K[:,2],1],ROIboxV[K[:,0],2]) - np.multiply(ROIboxV[K[:,2],2],ROIboxV[K[:,0],1]),
                    np.multiply(ROIboxV[K[:,2],2],ROIboxV[K[:,0],0] ) - np.multiply(ROIboxV[K[:,2],0],ROIboxV[K[:,0],2]),
                    np.multiply(ROIboxV[K[:,2],0],ROIboxV[K[:,0],1] ) - np.multiply(ROIboxV[K[:,2],1],ROIboxV[K[:,0],0])))
            ),2),axis=1
    ))) /2

    CH_AD = Surface/CHarea


    VDs = np.transpose([MALs[0],MALs[1],MALs[2], Elong ,Fltns ,VDaabb ,ADaabb ,VDombb, ADombb, VDaee ,ADaee, VDmvee, ADmvee, Solidity, CH_AD])


    return VDs, K


def getMax3Ddiam(ROIboxV, ConvHullV):

    n_v_int = ROIboxV.shape[0]
    
    if (1000 > n_v_int > 1):
    
        NumVer          = ROIboxV.shape[0]
        ROIboxV_single  = ROIboxV.astype(np.float16)
        r2              = np.transpose (ROIboxV_single)

        tile1 = np.tile(ROIboxV_single,(1,NumVer))
        tile2 = np.tile(np.transpose(r2.flatten(order='F')),(NumVer,1))
        # r2_array = np.float_power( (tile1  -  tile2) ,2 )
        r2_array = (tile1  -  tile2) ** 2
        r2              = np.reshape(r2_array, (NumVer,3,NumVer),order='F')

        sum_r2 = np.sum(r2,axis=1)
        rss             = np.sqrt(sum_r2)
        Max3Ddiam       = np.max(rss)

    elif n_v_int >= 1000:

        # except:
        # print('Problem with calculating Max 3D diameter, most probably issue with "low memory". An alternative "approximate" way using Convex Hull will be selected.')
        
        NumVer = ConvHullV.shape[0]
        r1=  np.column_stack(( ROIboxV[list(ConvHullV[:,0]),0]  ,   ROIboxV[list(ConvHullV[:,1]),1]  ,   ROIboxV[list(ConvHullV[:,2]),2]     ))
        r1m = np.tile(r1,(1,NumVer))
        r2= np.transpose(r1)
        r2m= np.tile(np.transpose(r2.flatten(order='F')),(NumVer,1))
        rm= np.float_power((r1m-r2m),2)
        rmSize = rm.shape
        rs=np.reshape(rm,(int((rmSize[0]*rmSize[1])/3),3),order='F')
        rss=np.sqrt(np.sum(rs,axis=1))
        Max3Ddiam = np.max(rss)
    



    return Max3Ddiam


# def getAutoCorrsThread(r , MNum , GNum , sumW,Mean,Points,av2,Xgl):

#     W = np.divide(1,np.sqrt(   np.float_power((Points[:,0]-Points[r,0]),2) + np.float_power((Points[:,1]-Points[r,1]),2) + np.float_power((Points[:,2]-Points[r,2]),2)    ))

#     W[r] = 0
#     av = Xgl[r] - Mean
#     mul_W = np.multiply(av , av2)

#     mn2 = Xgl[r] - Xgl
#     mn = np.float_power( mn2 , 2)
#     aa = np.multiply( W ,  mn )   
#     bb = np.multiply(W , mul_W)

#     lock = threading.Lock()
#     lock.acquire()
#     MNum = MNum + np.sum(bb)
#     GNum = GNum + np.sum(aa)
#     sumW = sumW + np.sum(W)
#     lock.release()  

#     return ''

# def getAutoCorrs(ROIints , xI,yI,zI , pixelW,sliceS , Mean):

#     Xgl = ROIints.copy()
#     Xgl = Xgl[ROIints != np.nan]
#     # Xgl = Xgl.flatten(order='F')
#     # Xgl = Xgl.astype(np.float32)
#     Points = np.column_stack((xI*pixelW,yI*pixelW,zI*sliceS))
#     Points = np.asarray(Points)
    
#     if Xgl.shape[0] == 1:
#         try:
#             Xgl = np.transpose(Xgl) 
#         except:
#             Xgl = np.squeeze(Xgl)
        
#     if Points.shape[0] == 1:
#         Points = np.column_stack((np.transpose(xI) * pixelW, np.transpose(yI) *pixelW ,np.transpose(zI)*sliceS))
#         Points = np.asarray(Points)
    

#     nV = Xgl.shape[0]
#     MNum = 0
#     GNum = 0 
#     sumW = 0
    
#     Mean = float(Mean)
#     av2 = Xgl - Mean

#     # Num_worker = 1000
#     # with concurrent.futures.ThreadPoolExecutor(max_workers=Num_worker) as executor:
#     #     futures = []
#     #     for r in range(0,nV):
#     #         futures.append(executor.submit(getAutoCorrsThread, r , MNum , GNum , sumW ,Mean,Points,av2,Xgl))
#     #         # if r # Num_worker == 0:
#     #         print(r)

#     # executor.shutdown(wait=True)

#     for r in range(0,nV):

#         sq = np.sqrt(   np.float_power((Points[:,0]-Points[r,0]),2) + np.float_power((Points[:,1]-Points[r,1]),2) + np.float_power((Points[:,2]-Points[r,2]),2)    )
#         W = np.divide(1,sq)
#         # W = np.float_power(sq,-1)

#         W[r] = 0
#         av = Xgl[r] - Mean
#         mul_W = np.multiply(av , av2)
#         bb = np.dot(W , mul_W)
#         bb2 = np.sum(np.multiply(W , mul_W))
#         bb4 = np.matmul(W , mul_W)

#         MNum = MNum + bb

#         print(np.mean(W))
#         print(np.mean(av))
#         print(np.mean(av2))
#         print(np.mean(mul_W))
#         print(np.mean(bb))
#         print(np.mean(Xgl))


#         mn2 = Xgl[r] - Xgl
#         mn = np.float_power( mn2 , 2)
#         aa = np.multiply( W ,  mn )   

#         GNum = GNum + np.sum(aa , dtype=np.float32)
#         sumW = sumW + np.sum(W)
#         # print(r+1  ,  MNum , GNum , sumW )

#     denom = np.sum(  np.float_power(av2,2))
#     MoranI = (nV  / sumW) * (MNum / denom)
#     GearyC = (nV-1) / (2*sumW) * GNum / denom
        

#     return MoranI, GearyC


def geospatial(df_int,spacing,xI,yI,zI):

    # Define constants
    n_v = df_int.shape[0]

    pos_mat = np.column_stack((zI*spacing[0],yI*spacing[1],xI*spacing[2]))

    if n_v < 2000:
        # Determine all interactions between voxels
        comb_iter = np.array([   np.tile(np.arange(0, n_v), n_v)  , np.repeat(np.arange(0, n_v), n_v)   ])
        comb_iter = comb_iter[:, comb_iter[0, :] > comb_iter[1, :]]

        # Determine weighting for all interactions (inverse weighting with distance)
        w_ij = 1.0 / np.array(list(
            map(lambda i: np.sqrt(np.sum((pos_mat[comb_iter[0, i], :] - pos_mat[comb_iter[1, i], :]) ** 2.0)),
                np.arange(np.shape(comb_iter)[1]))))

        # Create array of mean-corrected grey level intensities
        gl_dev = df_int - np.mean(df_int)

        # Moran's I
        nom = n_v * np.sum(np.multiply(np.multiply(w_ij, gl_dev[comb_iter[0, :]]), gl_dev[comb_iter[1, :]]))
        denom = np.sum(w_ij) * np.sum(gl_dev ** 2.0)
        if denom > 0.0:
            moran_i = nom / denom
        else:
            # If the denominator is 0.0, this basically means only one intensity is present in the volume, which indicates ideal spatial correlation.
            moran_i = 1.0

        # Geary's C
        nom = (n_v - 1.0) * np.sum(np.multiply(w_ij, (gl_dev[comb_iter[0, :]] - gl_dev[comb_iter[1, :]]) ** 2.0))
        if denom > 0.0:
            geary_c = nom / (2.0 * denom)
        else:
            # If the denominator is 0.0, this basically means only one intensity is present in the volume.
            geary_c = 1.0
    else:
        # In practice, this code variant is only used if the ROI is too large to perform all distance calculations in one go.

        # Create array of mean-corrected grey level intensities
        gl_dev = df_int - np.mean(df_int)

        moran_nom = 0.0
        geary_nom = 0.0
        w_denom = 0.0

        # Iterate over voxels
        for ii in np.arange(n_v-1):
            # Get all jj > ii voxels
            jj = np.arange(start=ii+1, stop=n_v)

            # Get distance weights
            w_iijj = 1.0 / np.sqrt(np.sum(np.power(pos_mat[ii, :] - pos_mat[jj, :], 2.0), axis=1))

            moran_nom += np.sum(np.multiply(np.multiply(w_iijj, gl_dev[ii]), gl_dev[jj]))
            geary_nom += np.sum(np.multiply(w_iijj, (gl_dev[ii] - gl_dev[jj]) ** 2.0))
            w_denom += np.sum(w_iijj)

        gl_denom = np.sum(gl_dev ** 2.0)

        # Moran's I index
        if gl_denom > 0.0:
            moran_i = n_v * moran_nom / (w_denom * gl_denom)
        else:
            moran_i = 1.0

        # Geary's C measure
        if gl_denom > 0.0:
            geary_c = (n_v - 1.0) * geary_nom / (2*w_denom*gl_denom)
        else:
            geary_c = 1.0

    return moran_i, geary_c


def getAutoCorrs(ROIints , xI,yI,zI,pixelW,sliceS ):
    
    n_v_int = ROIints.size
    if (1000 > n_v_int > 1):
    
        MoranI, GearyC = geospatial(df_int = ROIints, spacing = [pixelW,pixelW,sliceS], xI=xI, yI=yI, zI=zI)

    elif n_v_int >= 1000:
        moran_list, geary_list = [], []
        iter_nr = 1
        tol_aim = 0.002
        tol_sem = 1.000
        while tol_sem > tol_aim:
            curr_points = np.random.choice(n_v_int, size=100, replace=False)
            moran_i, geary_c = geospatial(df_int=ROIints[curr_points], spacing = [pixelW,pixelW,sliceS] , xI=xI, yI=yI, zI=zI)

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

