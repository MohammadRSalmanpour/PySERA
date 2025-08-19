import numpy as np
import logging

# -------------------------------------------------------------------------
# function [textures] = getGlobalTextures(ROIonly,Nbins)
# -------------------------------------------------------------------------
# DESCRIPTION:
# This function computes features related to the Intensity Volume Histogram
# ====> Make sure to specify the DataType. If not specified, Binsize will
# be considered as the 1000 bins for histogramming as directed by ISBI.
# -------------------------------------------------------------------------
# INPUTS:
# - ROIonly: Smallest box containing the ROI, with the imaging data ready
#            for texture analysis computations. Voxels outside the ROI are
#            set to NaNs. This should be the intesity ROI.
# - BinSize: scalar indicating the number of discretized bins
#           (or reconstruction levels of quantization).
# - DataType: 'PET': uses 1000 bins by default as suggested by ISBI.
#             'CT': uses the CT HU numbers as bins.
#             Otherwise: uses 1000 bins as suggested by ISBI.
# - minGL: minimum intensity within the ROI
#
# ** 'ROIonly' and 'levels' should be outputs from 'prepareVolume.m' **
# -------------------------------------------------------------------------
# OUTPUTS:
# - textures: Struture specifying the values of different IVH features as
# defined below.
# -------------------------------------------------------------------------
# AUTHOR(S): 
# - Saeed Ashrafinia
# - Mahdi Hosseinzadeh
# -------------------------------------------------------------------------
# HISTORY:
# - Creation: December 2022
# -------------------------------------------------------------------------

logger = logging.getLogger("Dev_logger")

def getIntVolHist(IntsROI2, ROIBox3D2, BinSize, isReSeg, ResegIntrval, IVHconfig, feature_value_mode):

    IntsROI = IntsROI2.copy()
    ROIBox3D = ROIBox3D2.copy()


    if IVHconfig[0] == 0:
        ROIarrayValid = np.squeeze(IntsROI[~np.isnan(IntsROI)])
        if IVHconfig[1] == 0 :
            
            if isReSeg == 1:
                minGL = ResegIntrval[0]  
                maxGL = np.min([np.max(ROIarrayValid), ResegIntrval[1]])
            else:
                minGL = np.min(ROIarrayValid)
                maxGL = np.max(ROIarrayValid)
            
            BinSize = 1
            G = np.arange(minGL,maxGL+1,BinSize)
            
            G_minues = G - np.min(G)
            gamma = G_minues.astype(np.uint32) /  (np.max(G) - np.min(G))
            Hist = np.histogram(ROIarrayValid,G)
            HistAc = np.cumsum(Hist[0])
            BinsCenters = G
            V = [1]
            V2 = 1 - (  np.divide(HistAc, HistAc[-1]))
            V = V + list(V2)
            V = np.asarray(V)

        elif IVHconfig[1] == 1 :
            
            if isReSeg == 1:
                minGL = ResegIntrval[0]  
                maxGL = np.min([np.max(ROIarrayValid), ResegIntrval[1]])
            else:
                minGL = np.min(ROIarrayValid)
                maxGL = np.max(ROIarrayValid)

            BinSize = IVHconfig[2]

            G = np.arange(minGL,(np.ceil(maxGL/BinSize)+1)*BinSize,BinSize)            
            G_minues = G - np.min(G)
            maxrng = np.max(G) - np.min(G)
            gamma = G_minues / maxrng 

            try:
                G2 = np.insert(G,0,np.min(G)-1)
                # G2 = G.copy()
                Hist = np.histogram(ROIarrayValid,G2)  
                # n, bins, patches = plt.hist(ROIarrayValid,G2)
            except:
                raise Exception('--Problem with IVH. Dividing the bin size by 10 to fix.')
                BinSize = BinSize/10

                G = np.arange(minGL,(np.ceil(maxGL/BinSize)+1)*BinSize,BinSize)
                G_minues = G - np.min(G)
                maxrng = np.max(G) - np.min(G)
                gamma = G_minues / maxrng 
                
                try:
                    G2 = np.insert(G,0,np.min(G)-1)
                    Hist = np.histogram(ROIarrayValid,G2)
                except:
                    raise Exception('---It is even worse. Lets add 0.0001 to see if it fixes the histogram.')
                    G2 = np.insert(G,0,np.min(G)-1)
                    Hist = np.histogram(ROIarrayValid, [G2, G2 + 0.0001])
                
            # Hist2 = np.insert(Hist[0],0,0)
            HistAc = np.cumsum(Hist[0])
            av = np.arange(1,Hist[0].shape[0]+2)-0.5
            BinsCenters = minGL + BinSize  * av
            V = 1 - (  np.divide(HistAc, HistAc[-1]))

            # HistAc = np.cumsum(Hist[0])
            # HistAc = HistAc[1:]
            # HistAc = np.insert(HistAc,-1,HistAc[-1])
            # av = np.arange(1,Hist[0].shape[0]+1)-0.5
            # BinsCenters = minGL + BinSize  * av
            # V = 1 - (  np.divide(HistAc, HistAc[-1]))
              
        else:
            raise Exception('Wrong IVH Config parameter!')
        
        
    elif IVHconfig[0] == 1:

        ROIarrayValid = np.squeeze(IntsROI[~np.isnan(IntsROI)])
        if IVHconfig[1] == 0: 

            if isReSeg == 1:
                minGL = ResegIntrval[0]  
                maxGL = np.min([np.max(ROIarrayValid), ResegIntrval[1]])
            else:
                minGL = np.min(ROIarrayValid)
                maxGL = np.max(ROIarrayValid)


            Ng = int(IVHconfig[2])
            Hist = np.histogram(ROIarrayValid,Ng)
            HistAc = np.cumsum(Hist[0]) 
            BinsCenters =  np.arange(0,Hist[0].shape[0])
            BinSize = (maxGL-minGL)/Ng 
            V = 1 - (  np.divide(HistAc, HistAc[-1]))
            gamma = ((BinsCenters - BinsCenters[0]) / (BinsCenters[-1]-BinsCenters[0]))
            
        else:
            raise Exception('Wrong IVH Config parameter!')
        
    elif IVHconfig[0] == 2:
        raise Exception('Not a standard IVH Setting!! Make sure you know what you are doing.')
        ROIarrayValid = np.squeeze(IntsROI[~np.isnan(IntsROI)])
        BinSize = 1000
        Hist = np.histogram(ROIarrayValid,BinSize)
        HistAc = np.cumsum(Hist[0])
        V = 1 - (  np.divide(HistAc, HistAc[-1]))
        BinsCenters =  np.arange(1,BinSize+1)
        gamma = ((BinsCenters - BinsCenters[0]) / (BinsCenters[-1]-BinsCenters[0]))

    elif IVHconfig[0] == 3:

        ROIarrayValid = np.squeeze(IntsROI[~np.isnan(IntsROI)])

        if isReSeg == 1:
            minGL = np.max([np.nanmin(IntsROI), ResegIntrval[0]]) 
            maxGL = np.min([np.nanmax(IntsROI), ResegIntrval[1]])
        else:
            minGL = np.min(ROIarrayValid)
            maxGL = np.max(ROIarrayValid)


        BinsCenters =  np.arange(minGL,np.ceil(maxGL)+1,1)

        # BinsCenters = np.insert(BinsCenters,0,0)

        BinsCenters_list = list(BinsCenters)
        BinsCenters_list.append(BinsCenters[-1]+1)

        histraw = np.histogram(ROIarrayValid,BinsCenters_list)
        HistAc = np.cumsum(histraw[0])
        
        VV = list(1 - (  np.divide(HistAc, HistAc[-1])))
        V = [1]
        V = V + VV[:-1]
        V = np.asarray(V)
        gamma = ((BinsCenters - BinsCenters[0]) / (BinsCenters[-1]-BinsCenters[0]))
        
    else:
        raise Exception('Wrong IVH Config parameter!')
    
    if np.all(np.isnan(gamma)):
        if feature_value_mode=='APPROXIMATE_VALUE':
            value = 0
        else:
            # if feature_value_mode=='REAL_VALUE'
            value = np.nan

        output = [value] * 7
        logger.warning(f"In Intensity Volume Histogram calculations, 'gamma' is NaN. Returning {value} for V_10, V_90, I_10, I_90, V_10_90, I_10_90, AUC.")
        return output

    V_10 = V[np.where(gamma>=0.1)[0][0]]
    V_90 = V[ np.where(gamma>=0.9)[0][0]]

    V_10_idx = np.where(V<=0.1) # Points where <10% of the volume remains above a certain intensity     toto
    # Handle small RoIs ( <10) which lead to empty V_10_idx     toto
    if np.size(V_10_idx[0]) > 0:
        I_10 = BinsCenters[np.min([BinsCenters.shape[0],int(V_10_idx[0][0]+1)])-1]
    elif feature_value_mode == "APPROXIMATE_VALUE":
        synth_V = np.append(V, 0.05)
        I_10 = BinsCenters[np.min([BinsCenters.shape[0],int(np.where(synth_V<=0.1)[0][0]+1)])-1]
        logger.warning(f"Insufficient volume to extract features. Sunthesizing RoI with appending 0.05.")
    elif feature_value_mode == "REAL_VALUE":
        I_10 = np.nan
        logger.warning(f"Insufficient volume to extract features. Returning NaN for 'ivh_i10' and, 'ivh_diff_i10_i90'.")

    I_90 = BinsCenters[np.min([BinsCenters.shape[0],int(np.where(V<=0.9)[0][0]+1)])-1]

    V_10_90 = V_10 - V_90

    I_10_90 = I_10 - I_90

    # AUC = (-1) * np.trapz(gamma,V)
    # AUC = (-1) *  scipy.integrate.trapz(gamma,V)
    # AUC = scipy.integrate.simps(gamma,V)
    AUC = np.trapz(V,gamma)

    textures = [V_10, V_90, I_10, I_90, V_10_90, I_10_90, AUC ]


    return textures