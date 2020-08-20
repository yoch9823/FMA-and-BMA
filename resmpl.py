import numpy as np 
def resmpl(u1):
    """
    Returns
    -------
    u2 : int 
    """
    n = u1.shape[0]
    a = np.ceil(n*np.random.random((1,n)))
    a = np.reshape(a,a.shape[1]) # convert from 2-D to 1-D 
    u2 = [u1[j] for j in a]

    return u2
