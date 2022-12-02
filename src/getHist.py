

from scipy.io import savemat, loadmat
import os
import numpy as np
import sys
import skimage
import cv2
import itertools
import scipy 

def getHist(ROIonly2,BinSize, DiscType):
    
    ROIonly = ROIonly2.copy()
    ROIarrayValid = ROIonly[~np.isnan(ROIonly2)].flatten(order='F')

    if DiscType == 'FBN':
        histraw = np.histogram(ROIarrayValid,BinSize)
        histo = np.divide(histraw[0],np.sum(histraw[0]))
        Binslist = np.arange(1,BinSize+1)
        Mean = np.sum(histo * Binslist)

    elif DiscType == 'FBS':
        
        BinsCntrVect = np.arange(1,np.max(ROIarrayValid)+2)
        histraw = np.histogram(ROIarrayValid ,BinsCntrVect)
        histo = np.divide(histraw[0],np.sum(histraw[0]))
        BinsCntrVect = np.delete(BinsCntrVect,-1)
        # BinSize = BinsCntrVect.ndim
        BinSize = BinsCntrVect.shape[0]
        Binslist = BinsCntrVect
        Mean = np.sum(histo * Binslist)


    Mean = Mean

    Var = 0
    skw = 0
    krt = 0
    entropy = 0
    for i in range(0,BinSize):
        Var = Var + histo[i] *  np.float_power(((i+1)-Mean),2)
        skw = skw + histo[i] *  np.float_power(((i+1)-Mean),3)
        krt = krt + histo[i] *  np.float_power(((i+1)-Mean),4)
        entropy = entropy-histo[i] *   np.log2(   histo[i] + np.finfo(float).tiny)
    
    Entropy = entropy

    sigma = np.sqrt(Var)
    Variance = Var
    skw = skw / np.float_power(sigma,3)
    if np.isnan(skw) == True:
        skw = 0
    Skewness = skw
    krt = (krt   /  np.float_power(sigma,4)) - 3

    if np.isnan(krt) == True:
        krt = 0
    Kurtosis = krt

    SUVmax = np.max(Binslist)

    cumHisto = 0
    Prcnt10 = float('inf')
    Prcnt90 = float('inf')
    Median = float('inf')
    q75 = float('inf')
    q25 = float('inf')
    for i in range(0,BinSize):
        cumHisto += histo[i]
        if cumHisto >= 0.9:
            if (i+1) <= Prcnt90:
                Prcnt90 = i+1
        if cumHisto >= 0.75:
            if (i+1) <= q75:
                q75 = i+1
        if cumHisto >= 0.5:
            if (i+1) <= Median:
                Median = i+1
        if cumHisto >= 0.25:
            if (i+1) <= q25:
                q25 = i+1
        if cumHisto >= 0.1:
            if (i+1) <= Prcnt10:
                Prcnt10 = i+1

    # SUVmin = np.min(Binslist)
    SUVmin = np.min(np.where(histraw[0]>0)[0])+1


    ModeIdx = np.array(np.argmax(histo))
    if ModeIdx.size == 1:
        Mode = ModeIdx + 1
    else:
        Idx = np.argsort(np.abs(ModeIdx-Mean))
        Mode = int(ModeIdx[np.argmin(Idx)]) + 1

    IqntlRange = q75-q25
    
    Range = SUVmax - SUVmin

    nV = ROIarrayValid.shape[0]
    MAD = np.sum(np.abs(    np.multiply( histraw[0],(Binslist - Mean) )    )) / nV

    RobustSet = list(np.arange(Prcnt10-1,Prcnt90))
    nVr = np.sum(histraw[0][RobustSet])

    aaa = np.divide(histraw[0][RobustSet] , nVr)
    bbb = np.transpose(Binslist[RobustSet])
    SUVmeanR =  np.dot (aaa,bbb)
    RMAD = np.sum(np.abs(    np.multiply(histraw[0][RobustSet],(Binslist[RobustSet] - SUVmeanR) )  )) / nVr


    MedAD = np.sum(np.abs(    np.multiply( histraw[0],(Binslist - Median) )   ))  / nV


    CoV = np.sqrt(Var) / Mean


    QCoD = (q75 - q25) / (q75 + q25)

    Energy = np.sum(  np.float_power(histo,2))

    
    
    try:
        HH = []
        H = histraw[0][1]-histraw[0][0]
        HH.append(H)
        H2 = list(np.divide( np.subtract(histraw[0][2:] , histraw[0][0:-2] ) , 2))
        HH = HH + H2
        H3 = histraw[0][-1] - histraw[0][-2] 
        HH.append(H3)
        HH = np.array(HH)
    except:
        HH = histraw[0][0]

    MaxGrad, MaxGradGL = np.max(HH),np.argmax(HH)+1

    MinGrad, MinGradGL = np.min(HH),np.argmin(HH)+1



    HistVect = [Mean, Variance, Skewness, Kurtosis, Median,
            SUVmin, Prcnt10, Prcnt90, SUVmax, Mode, 
            IqntlRange, Range, MAD, RMAD, MedAD, 
            CoV, QCoD, Entropy, Energy, 
            MaxGrad, MaxGradGL, MinGrad, MinGradGL]

    return HistVect