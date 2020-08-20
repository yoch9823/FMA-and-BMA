import os
print(os.getcwd())

import numpy as np 
from im_c1 import im_c1

def im_c2(y,X,t):
    """ Create index Matrix : case 2, pure statistical 
    
    Parameter
    ---------
    y : np.ndarray 
        dependent variable
    X : np.ndarray 
        independent variable 
    t : np.ndarray
        top N variables are key variables

    Return
    ------
    IM2 : np.ndarray 
        model matrix 
    """
    
    y = np.array([y]).T # make 2-D array

    nobs = X.shape[0]
    ndim = X.shape[1]
    b = np.linalg.lstsq(X,y,rcond=-1)
    b=b[0] # k x 1 dimension 
    XSquared = np.matmul(X.T,X) # k x k dimension
    XInv = np.linalg.inv(XSquared) 
    ee = np.sum((y-np.matmul(X,b)) ** 2) # scalar 
    V = ee * np.diag(XInv)
    V = np.array([V]).T # k x 1 matrix 
    tstat = b/np.sqrt(V) # k x 1 matrix 
    tmp = abs(tstat) # k x 1 matrix 
    TMP = sorted(range(len(tmp)), key = lambda x:tmp[x], reverse = True)
    IN = sorted(range(len(tmp)), key = lambda x:TMP[x])
    # print(tmp)
    # print(TMP)
    # print(IN)
    IN = np.array([IN]) # 1 x k matrix
    IN.astype(int) 
    MM = []
    kt = (X.shape[1]-t)/4
    kt = int(np.floor(kt))
    for i in range(kt):
        if i == 1:
            MM = np.append(MM,IN[0,0:t])
        elif i == kt:
            MM = np.append(MM,IN[0,4*(i-1)+t:-1])
        else :
            MM = np.append(MM,IN[0,4*(i-1)+t:4*i+t])
        
    IM2 = im_c1(MM.astype(int))

    return IM2


