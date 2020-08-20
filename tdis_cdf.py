import os
print(os.getcwd())

import numpy as np 
from itertools import product
from scipy.optimize import minimize, Bounds 
from collections import Counter 
import pandas as pd 

from fdis_cdf import *

# define function gets
def tdis_cdf(X,n):
    """ To return CDF at x of the t(n) distribution. 
    
    Parameters
    ----------
    X :  np.ndarray 
        independent variable 
    n : int
        a parameter with dof 

    Returns
    -------
        a vector of CDF at each element of x of the t(n) distribution
    """

    nobs = X.shape[0]
    junk = X.shape[1]
    neg = X<0
    F = fdis_cdf(X**2,1,n)
    iota = np.ones((nobs,1))
    out = iota-(iota-F)/2
    F = out + (iota-2*out)*neg

    return F

    


