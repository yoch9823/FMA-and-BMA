import os
print(os.getcwd())

import numpy as np 
from scipy.special import betainc

def beta_cdf(X,a,b):
    """ CDF of the beta distribution. 

    Parameters
    ----------
    X : np.array
        Prob[beta(a,b)<=x]
    a : int 
        beta distribution parameter
    b : int
        beta distribution parameter

    Returns
    -------
    cdf : 
        cdf of each element of x of the beta distribution
    """

    Xshape = X.shape
    # to make X in [0,1]
    for i in range(Xshape[0]):
        if X[i,0] < 0:
            X[i,0] = 0
        elif X[i,0] > 1:
            X[i,0] = 1


    if X.size > 0:
        cdf = betainc(a,b,X) # default function betainc : Incomplete beta integral.
        # It computes the incomplete beta integral of the arguments evaluated from 0 to x
        # gamma(a+b) / (gamma(a)*gamma(b)) * integral(t**(a-1) (1-t)**(b-1), t=0..x)

    return cdf
