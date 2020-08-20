# this program estimates averaged coefficients
# and associated variances using model averaging prediction criterion (MAPC, PMA) estimator

# import data
import os
print(os.getcwd())

import numpy as np 
from itertools import product
from scipy.optimize import minimize, Bounds 
from collections import Counter 
import pandas as pd 

#  *args is used when we don't know how many how many arguments
#  that will be passed into your function, add a * before the parameter name in the function definition.
def mapc(y,*args):
    """ To estimate average coefficients and associated variances 
    using model averaging prediction criterion (MAPC, PMA) estimators. 

    Parameters
    ----------
    y : np.ndarray in R^nobs
        dependent variable 

    Returns
    -------
    b : np.ndarray in R^ndim 
        averaged coefficients
    w : np.ndarray in R^model 
        weighting vector
    V : np.ndarray in R^(ndim x ndim )
        covariance matrix for the averaged coefficients  
    """

    if len(args) == 1: 
        # make yt as 2-D array for using mapc
        y = np.array([y]).T 

        X = args[0] # allocate first variable of args vector variables (not sure..)
        Xshape = X.shape
        yshape = y.shape
        if Xshape[0] != yshape[0]: 
            print('x and y must have the same number of observation')
            return
        m = 2**Xshape[1] - 1

        im = product((0,1), repeat = Xshape[1]) 

    elif len(args) == 2:
        # make yt as 2-D array for using mapc
        y = np.array([y]).T 

        X = args[0]
        im = args[1]
        imshape = im.shape
        im = im == np.ones(imshape)
        im.astype(int)
        Xshape = X.shape
        yshape = y.shape

        m = imshape[0]
        
        # if Xshape[0] != yshape[0]:
        #     print('x and y must have the same number of observation')
        #     return
        # if Xshape[1] != imshape[1]:
        #     print('im should be m by k, where m is the number of models and k is the number of total regressors')
        #     return
    
    else:
        print('Wrong # of arguments to mapc')
        return


    # Compute the OLS coefficients for each approximation model and locate them 
    # in a (k x m) beta_matrix 

    bm = np.zeros((X.shape[1],m))
    iter = 0
    for model in im:
        if iter == 0:
            pass
        else:
            xm = []
            model = np.array(model)
        
            # in matlab it is x(:,~im(i,:))
            # take indices that indicate model elements to be used
            idx = [i for i,x in enumerate(model) if x == 1]
            xm = [X[:,j] for j in idx]

            # make array type to be available in np lib.
            xm = np.array(xm) # xm dim: m(1) X 94
            xmSquared = np.matmul(xm,xm.T)
            xmInv = np.linalg.inv(xmSquared)
            xmY = np.matmul(xm,y) 
            b = np.matmul(xmInv,xmY) # b dim: m(1) X 1

            # store element of b in the each model (bm)
            # in Matlab, it is bm(~im(i,:),i)=b

            jdx = 0
            for j in idx:
                bm[idx,iter-1] = b[jdx] # beware of index (iter): exclude iter = 0 (zero model vector)
                jdx += 1
        
            del xm, idx, xmSquared, xmInv, xmY, b, model


    # Compute the averaged coefficients and weights 
    ym = np.matmul( y, np.ones((1,m)) ) 
    em = (ym - np.matmul(X,bm )) # generate residual matrix 
    w0 = np.ones((m,1))/m # set the initial weight 

    # re-declare ims
    im = product((0,1), repeat = Xshape[1]) 
    kv = np.array([ 
        np.sum(np.array(list(model)))
        for model in im
        ])

    kv = kv[1:] # remove zero model

    #-----------------------
    # fmincon of MATLAB
    #-----------------------

    # declare function
    n = X.shape[0]
    fun = lambda w: np.matmul( np.matmul( em, w ).T, np.matmul( em, w ) ) * ( n + np.matmul(kv, w) ) / ( n - np.matmul(kv, w) )
    # boundary condition
    bnds = ( ((0,1),) * m)
    # constraints
    cons = ({'type': 'ineq', 'fun': lambda w: -np.matmul( np.zeros((1,m)), w )},
            {'type': 'eq',   'fun': lambda w: np.matmul( np.ones((1,m)), w ) })
    
    # print('optimization starts!')
    # optimize
    res = minimize(fun, w0, method='SLSQP', bounds=bnds, constraints=cons)
    # if res.success:
    #     print('optimization terminated.')
    # else:
    #     print('optimization failed.')

    # take w from x-value of res
    w = res.x
    # normalization
    w = w/np.sum(w)

    b = np.matmul(bm, w)
    b.resize(b.shape[0],1)
    # print('--------------------')
    # print('average weighted coefficient')
    # print(b)
    
    # # compute the covariance matrix for the averaged coefficients
    # sigs = np.matmul( (y - np.matmul(X,b)).T, (y - np.matmul(X,b)) ) / (n - np.matmul(kv,w) )
    # dm = bm - np.matmul( b, np.ones((1,m)) )
    # V = 0
    
    # jm = im
    
    # iter = 0
    # for model in im:
    #     if iter == 0:
    #         pass
    #     else:
    #         xm = []
    #         model = np.array(model)
        
    #         # in matlab it is x(:,~im(i,:))
    #         # take indices that indicate model elements to be used
    #         idx = [i for i,x in enumerate(model) if x == 1]
    #         xm = [X[:,j] for j in idx]

    #         # make array type to be available in np lib.
    #         xm = np.array(xm) # xm dim: m(1) X 94
    #         xmSquared = np.matmul(xm,xm.T)
    #         xmInv = np.linalg.inv(xmSquared)
    #         xSquared = np.matmul(X.T,X)

    #         if np.linalg.det(xSquared) != 0:
    #             xInv = np.linalg.inv(xSquared)
    #         else:
    #             # cannot compute inverse: singular
    #             return b, w

    #         # gm = np.round( np.matmul(xInv, np.matmul(X.T,xm.T) ) )
    #         gm = np.round( np.matmul(xInv, np.matmul(X.T,xm.T) ) )

    #         jter = 0
    #         for model_j in jm:
    #             if jter == 0:
    #                 pass
    #             else:
    #                 xs = []
    #                 model_j = np.array(model_j)
                
    #                 # in matlab it is x(:,~im(j,:))
    #                 # take indices that indicate model elements to be used
    #                 idx = [i for i,x in enumerate(model_j) if x == 1]
    #                 xs = [X[:,j] for j in idx]

    #                 # make array type to be available in np lib.
    #                 xs = np.array(xs)
    #                 xmxs = np.matmul( xm, xs.T )
    #                 xsSquared = np.matmul(xs,xs.T)
    #                 xsInv = np.linalg.inv(xsSquared)
    #                 gs = np.round( np.matmul(xInv, np.matmul(X.T,xs.T) ) )

    #                 gmInv = np.matmul( gm, xmInv )
    #                 gmXmXs = np.matmul( gmInv, xmxs )
    #                 gmXs = np.matmul( gmXmXs, xsInv )
    #                 gmGs = np.matmul( gmXs, gs.T )

    #                 # take -1 from removing zero model vector
    #                 V += w[iter-1]*w[jter-1]* ( sigs * gmGs ) + np.matmul( dm[:,iter-1],dm[:,jter-1].T ) 

    #                 del model_j  

    #             jter += 1                     

    #         del xm, idx, xmSquared, xmInv, model

    #     iter += 1

    # # print('--------------------')
    # # print('covariance matrix')
    # # print(V)

    return b, w



