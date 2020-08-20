# Applying Hansen's simulated data set 

### n = 50
### k = 10, 15, 20 

# Provides testing utilities 
import os
print(os.getcwd())

import numpy as np 
import matplotlib.pyplot as plt 
import linear_averaging_sim as linear_averaging
from mf_criteria_sim import mf_criteria
import csv 

NOBS = 50 # mapc_sim file need to be change when default value changed. 
# constant term would be included underneath in BMA and during the process. 

# Following Hansen (ECMT, 2007) simulated dataset 

MODEL_NUM = int(3*NOBS**(1/3)) - 1
PRIOR =1/5
ALPHA = 1.5
C = np.arange(0.1,1,0.1)


def simulate_data(nobs,weights, cor =0.5):
    """ Simulate a simple dataset where 10 predictors are irrelevant and 5 increasingly relevant. 

    Predictors are jointly gaussian with 0 mean, given correlation and variance 1. 

    Residuals are gaussian with 0 mean and variance 1. 

    Parameters
    ----------
    nobs : int{1,..., inf}
        number of samples to be drawn 
    weights : np.ndarray
        model weights
    cor : float (-1,1)
        correlation of the predictors
    
    Returns
    -------
    tup
        with [0] feature matrix and [1] response
    """

    ndim = len(weights)
    cov = (1-cor) * np.identity(ndim) + cor * np.ones((ndim,ndim))
    mean = np.zeros(ndim)

    X = np.random.multivariate_normal(mean, cov, nobs)
    e = np.random.normal(0, 1, nobs)
    y = np.dot(X,weights) + e

    return (X, y)


def replicate_trial(trial, n):
    """ Repeat a random trial n times. 

    Parameters
    ----------
    trial : func
        call to the generating function
    n : int {1,...,inf}
        number of trials 

    Returns
    -------
    np.ndarray 
        where rows are tirals and columns are variables
    """

    return np.narray([trial() for i in range(n)])

# name of csv file  
filename = "hansenSim.csv"
msfeFma = []  
msfeBma = []
maeFma = []
maeBma =[]

for c in C:
    print(c)

    # set coefficients
    # weights = np.hstack((np.zeros((2,)), 0.5*np.ones((MODEL_NUM-2,))))
    weights = []
    for j in range(1,MODEL_NUM+1):
        weightsTemp = c * np.sqrt(ALPHA)*j**(-ALPHA-(1/2))
        weights.append(weightsTemp)

    # simulate data
    np.random.seed(2015)
    X, y = simulate_data(NOBS, weights, 0.5)


    # run the forecasting experiment 
    # E = np.array([10, 20, 30, 40, 50])
    E = np.array([25])
    d = 2


    # make y as 1-D
    y = np.reshape(y,y.shape[0]) # convert from 2-D to 1-D
    # add const vector
    cons = np.ones((y.shape[0],1))
    X = np.hstack((cons, X))

    MSFE_fma, MAE_fma, MSFE_bma, MAE_bma = mf_criteria(y,X,E,d,MODEL_NUM,PRIOR)
    print('MSFE-FMA')
    print(MSFE_fma)    

    print('MSFE-BMA')
    print(MSFE_bma)
    
    print('MAE-FMA')
    print(MAE_fma)

    print('MAE-BMA')
    print(MAE_bma)
    msfeFma.append(float(MSFE_fma))
    msfeBma.append(float(MSFE_bma))
    maeFma.append(float(MAE_fma))
    maeBma.append(float(MAE_bma))

# open csv file to write
f = open(filename, "w")

# make MSFE/MAE label
risk_labels = ['c,','MSFE-FMA,','MSFE-BMA,','MAE-FMA,','MAE-BMA']
for risk_label in risk_labels:
    f.write(risk_label)
f.write('\n')

# write data
for i in range(len(msfeFma)):
    f.write(str(C[i]) + ',' + str(msfeFma[i]) + ',' + str(msfeBma[i]) + ',' + str(maeFma[i]) + ',' + str(maeBma[i]) + '\n')

f.close()

print('Simulation is terminated.')
