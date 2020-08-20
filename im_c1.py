import numpy as np 

def im_c1(MM):
    """ Create index Matrix: case 1
    
    Parameter
    ---------
    MM : np.array 
        model matrix 
    
    Return
    ------
    IM1 : np.ndarray 
        model matrix 
    """
    M = len(MM)

    k = -1
    for m in MM:
        k = np.max((np.max(m),k))

    IM1 = []
    IMt = np.zeros((1,k))
    for m in MM:   
        IMt[0,m-1] = 1
        
        IM1.append(IMt)
        
    return IM1



