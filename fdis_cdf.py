import os
print(os.getcwd())

import numpy as np 
from scipy import stats
from beta_cdf import beta_cdf

def fdis_cdf(X,a,b):
    """ Returns CDF at x of the F(a,b) distribution. 
    Parameters
    ----------
    X : np.ndarray 
    a : int
        numerator dof 
    b : int
        denominator dof 
    """

    X = X/(X+b/a)
    # cdf = stats.beta.cdf(X,a,b,loc=0,scale=1)
    cdf = beta_cdf(X,a/2,b/2)

    return cdf
