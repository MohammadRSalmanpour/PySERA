
from scipy.io import savemat, loadmat
import os
import numpy as np
import sys
import skimage
import cv2
import itertools
import scipy 
from scipy import integrate
import matplotlib.pyplot as plt

def getIntVolHist(IntsROI2,ROIBox3D2,BinSize,isReSeg,ResegIntrval,IVHconfig):

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

            G2 = np.insert(G,0,0)

            Hist = np.histogram(ROIarrayValid,G2)
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
                # print('--Problem with IVH. Dividing the bin size by 10 to fix.')
                BinSize = BinSize/10

                G = np.arange(minGL,(np.ceil(maxGL/BinSize)+1)*BinSize,BinSize)
                G_minues = G - np.min(G)
                maxrng = np.max(G) - np.min(G)
                gamma = G_minues / maxrng 
                
                try:
                    G2 = np.insert(G,0,np.min(G)-1)
                    Hist = np.histogram(ROIarrayValid,G2)
                except:
                    # print('---It is even worse. Lets add 0.0001 to see if it fixes the histogram.')
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
            print('Wrong IVH Config parameter!')  
        
        
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
            print('Wrong IVH Config parameter!')
        
    elif IVHconfig[0] == 2:
        # print('Not a standard IVH Setting!! Make sure you know what you are doing.')
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
        print('Wrong IVH Config parameter!')
    

    V_10 = V[np.where(gamma>=0.1)[0][0]]

    V_90 = V[ np.where(gamma>=0.9)[0][0]]

    I_10 = BinsCenters[np.min([BinsCenters.shape[0],int(np.where(V<=0.1)[0][0]+1)])-1]

    I_90 = BinsCenters[np.min([BinsCenters.shape[0],int(np.where(V<=0.9)[0][0]+1)])-1]

    V_10_90 = V_10 - V_90

    I_10_90 = I_10 - I_90

    # AUC = (-1) * np.trapz(gamma,V)
    # AUC = (-1) *  scipy.integrate.trapz(gamma,V)
    # AUC = scipy.integrate.simps(gamma,V)
    AUC = np.trapz(V,gamma)

    textures = [V_10, V_90, I_10, I_90, V_10_90, I_10_90, AUC ]


    return textures