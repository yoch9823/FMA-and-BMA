
# import data
import os
print(os.getcwd())

import numpy as np 
import numpy.matlib as ml
import matplotlib.pyplot as plt 
import pandas as pd
from pandas import DataFrame
from random import seed
from random import randint

from im_c1 import im_c1
from im_c2 import im_c2
# from gets import gets
from mapc_sim import mapc

import linear_averaging_sim as linear_averaging
import bma_sim

NOBS = 100
MODEL_NUM = 10
PRIOR =1/5

def mf_criteria(y,X,E,d):
    """ This is the criteria function for the movie project. 
    Use it to estiamte the MSFE and MAE for open box office. 

    Parameters
    ----------
    X : np.ndarray R^(nobs x ndim)
        independent variables
    y : np.array R^nobs
        dependent variable
    IM : int 
        model index for other estimators 
    E : vector
        exercise sample vector
    d : int
        number of simulation

    Return
    ------
    R : dictionary 
        MSFE, MAE
    """

    # initial variables 
    nobs = X.shape[0]
    ndim = X.shape[1]
    
    MSFE_fma = []
    MAE_fma = []
    MSFE_bma = []
    MAE_bma = []


    for j in E:

        print('-----------')
        print(j)

        ne = j
        MSFEt_fma = []
        MAEt_fma = []
        MSFEt_bma = []
        MAEt_bma = []
        for i in range(d):
                    
            IN = np.random.randint(nobs, size=nobs)    
            xt = X[IN[0:nobs-ne],:]
            yt = y[IN[0:nobs-ne]]
        
            # solve short rank problem 
            while xt.shape[1]<X.shape[1]:
                IN = np.random.randint(nobs, size=nobs)    
                xt = X[IN[0:nobs-ne],:]
                yt = y[IN[0:nobs-ne]]
            
            xe = X[IN[nobs-ne:nobs],:]
            ye = y[IN[nobs-ne:nobs]]
            # b3 = gets(yt,xt)
            # b5, w = mapc(yt,xt)
            fma, w = mapc(yt, xt)
            # print(fma)
            # # for i in range(len(w)):
            # #     print(w[i])
           


            # full enumeration 
            ## there is enumerator function in the bma_0128 (core.py) code script 
            enumerator = linear_averaging.LinearEnumerator(xt[:,1:], yt, MODEL_NUM**2, PRIOR)
            enumerator.select()
            enumerator.estimate()

            # mcmc approximation
            mc3 = linear_averaging.LinearMC3(xt[:,1:], yt, MODEL_NUM**2, PRIOR)
            mc3.select(niter=10000, method="random")
            mc3.estimate()
            r= mc3.estimates
            bma = r.get('coefficients')

            # # add Hansen's group estimators
            # M0 = np.array([1,4,8,18,19,29]) #18,29 included
            # M1 = np.array(range(25,29,1))
            # M2 = np.array(range(20,25,1))
            # M3 = np.array(range(14,18,1))
            # M4 = np.array([2,3,6,9,12,13])
            # M5 = np.array([5,7,10,11])
            # ##MM = np.hstack((M0,M1,M2,M3,M4,M5))
            # MM = [M0,M1,M2,M3,M4,M5]

            # # create group indicies
            # IM1 = im_c1(MM)
            # t = 5
            # IM2 = im_c2(y,X,t)

            # ------------------------
            # FMA result computation
            BM = np.array((fma))
        
            tmp = BM.shape[0]
            p = BM.shape[1]
            yeRepmat = ml.repmat(np.array([ye]).T,1,p) # ye: k x 1 vector
            xeBm = np.matmul(xe,BM)
            U = yeRepmat - xeBm
            # compute criterion values
            MSFEt_fma = np.append(MSFEt_fma,np.sum((U**2))/(ne))
            MAEt_fma = np.append(MAEt_fma,np.sum((abs(U)))/(ne))
            # ------------------------
            #BMA result computation
            BMA = np.array([bma]).T
        
            # BM = np.array([rBeta]).T # k X 1 vector
            tmp = BMA.shape[0]
            p = BMA.shape[1]
            yeRepmat = ml.repmat(np.array([ye]).T,1,p) # ye: k x 1 vector
            xeBm = np.matmul(xe,BMA)
            U = yeRepmat - xeBm
            # compute criterion values
             
            MSFEt_bma = np.append(MSFEt_bma,np.sum((U**2))/(ne))
            MAEt_bma = np.append(MAEt_bma,np.sum((abs(U)))/(ne))
            # ------------------------
            
            print(i)

        MSFE_fma = np.array([np.append(MSFE_fma, np.median(MSFEt_fma))]).T
        MAE_fma = np.array([np.append(MAE_fma, np.median(MAEt_fma))]).T

        MSFE_bma = np.array([np.append(MSFE_bma, np.median(MSFEt_bma))]).T
        MAE_bma = np.array([np.append(MAE_bma, np.median(MAEt_bma))]).T
        

    return MSFE_fma, MAE_fma, MSFE_bma, MAE_bma














