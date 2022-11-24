import numpy as np
from GPy.kern.src.stationary import Stationary

from GPy.kern.src.linear import Linear

from GPy.kern.src.standard_periodic import StdPeriodic

from GPy.util.linalg import jitchol, dpotri 
from scipy.linalg import lapack



"""
Different helper functions for GPs.
Inparticular, it implements the derivative for the kernels, which are used in the optimization.
"""

def dK_dX(kern, X, X2=None, dK_dR=None):
    # in place
    # derivative wrt to the second argument!!!!!!!!!!!!!!!!!!!!!!!!
    # stored sparsely in each column

    X = X[:,kern.active_dims]




    B, D = X.shape                  # batch size, input dimension

    if X2 is None:
        X2 = X
    else:
        X2 = X2[:,kern.active_dims]

    J = X2.shape[0]

    if dK_dR is None:
        dK_dR = np.zeros((B,J,D))

    _Temp = -kern.dK_dr_via_X(X,X2) * kern._inv_dist(X,X2)

    if kern.ARD:
        for d in range(0,D):
            dK_dR[:,:,d] =  _Temp * (X[:,d][:,None]-X2[:,d][None,:]) / (kern.lengthscale[d]**2)
    else:
        for d in range(0,D):
            dK_dR[:,:,d] =  _Temp * (X[:,d][:,None]-X2[:,d][None,:]) / (kern.lengthscale**2)

    return dK_dR


setattr(Stationary, "dK_dX", dK_dX)


def dK_dσ02(kern, X, X2=None):
    # note: dK wrt variance σ0^2, transformation: dK_d0 = dK_dσ02 * 2 * σ02

    if X2 is None:
        X2 = X

    return kern.K(X,X2) / kern.variance


setattr(Stationary, "dK_dσ02", dK_dσ02)


def dK_dσ02_diag(kern, X):

    return np.ones(X.shape[0])


setattr(Stationary, "dK_dσ02_diag", dK_dσ02_diag)




def dK_dl(kern, X, X2=None):
    # note: dK wrt lengthscale ls, transformation: dK_dl = dK_dl * l

    X = X[:,kern.active_dims]

    #print(kern.active_dims)

    if X2 is None:
        X2 = X
    else:
        X2 = X2[:,kern.active_dims]

    if kern.ARD:
        B, D = X.shape
        J = X2.shape[0]
        _Temp = -kern.dK_dr_via_X(X,X2) * kern._inv_dist(X,X2)
        dK_dl = np.zeros((B,J,D))
        for d in range(0,D):
            dK_dl[:,:,d] = _Temp * np.square( X[:,d:d+1] - X2[:,d:d+1].T ) / (kern.lengthscale[d]**3)
    else:
        dK_dl = (-kern.dK_dr_via_X(X,X2) * kern._scaled_dist(X,X2) / kern.lengthscale )[:,:,None]

    return dK_dl

setattr(Stationary, "dK_dl", dK_dl)

def dK_dl_diag(kern, X):

    X = X[:,kern.active_dims]

    if kern.ARD:
        res = np.zeros(X.shape)
    else:
        res = np.zeros(X.shape[0])

    return res

setattr(Stationary, "dK_dl_diag", dK_dl_diag)


### for linear kernel
def dK_dvs(kern,  X, X2=None):

    X = X[:,kern.active_dims]

    if X2 is None:
        X2 = X
    else:
        X2 = X2[:,kern.active_dims]

    if kern.ARD:
        B, D = X.shape
        J = X2.shape[0]
        dK_dvs = np.zeros((B,J,D))
        for d in range(0,D):
            dK_dvs[:,:,d] = np.outer(X[:,d], X2[:,d])
    else:
        dK_dvs = np.dot(X,X2.T)


    return dK_dvs

setattr(Linear, "dK_dvs", dK_dvs)

def dK_dvs_diag(kern,  X):

    X = X[:,kern.active_dims]
    

    if kern.ARD:
        dK_dvs_diag = X**2
    else:
        dK_dvs_diag = np.sum(X**2,1)

    return dK_dvs_diag

setattr(Linear, "dK_dvs_diag", dK_dvs_diag)





# for StdPeriodic


def dK_dv_per(kern, X, X2=None):
        """derivative of the covariance matrix with respect to the variance."""

        X = X[:,kern.active_dims]

        if X2 is None:
            X2 = X
        else:
            X2 = X2[:,kern.active_dims]

        base = np.pi * (X[:, None, :] - X2[None, :, :]) / kern.period

        sin_base = np.sin( base )
        exp_dist = np.exp( -0.5* np.sum( np.square(  sin_base / kern.lengthscale ), axis = -1 ) )

        dK_dv_per = exp_dist

        return dK_dv_per
     

setattr(StdPeriodic, "dK_dv", dK_dv_per)


def dK_dv_diag_per(kern, X):

    return np.ones(X.shape[0])

setattr(StdPeriodic, "dK_dv_diag", dK_dv_diag_per)

def dK_dl_per(kern, X, X2=None):
        """derivative of the covariance matrix with respect to the lengthscales."""

        X = X[:,kern.active_dims]

        if X2 is None:
            X2 = X
        else:
            X2 = X2[:,kern.active_dims]

        base = np.pi * (X[:, None, :] - X2[None, :, :]) / kern.period

        sin_base = np.sin( base )
        exp_dist = np.exp( -0.5* np.sum( np.square(  sin_base / kern.lengthscale ), axis = -1 ) )


        dl = kern.variance * np.square( sin_base) / np.power( kern.lengthscale, 3)


        if kern.ARD2: # different lengthscales
            dK_dl = dl * exp_dist[:,:,None] 
        else: # same lengthscales
            dK_dl = dl.sum(-1) * exp_dist 

        return dK_dl

setattr(StdPeriodic, "dK_dl", dK_dl_per)


def dK_dl_diag_per(kern, X):

    X = X[:,kern.active_dims]

    if kern.ARD2:
        return np.zeros(X.shape)
    else:
        return np.zeros(X.shape[0])


setattr(StdPeriodic, "dK_dl_diag", dK_dl_diag_per)


def dK_dp_per(kern, X, X2=None):
        """derivative of the covariance matrix with respect to the period."""
        X = X[:,kern.active_dims]

        if X2 is None:
            X2 = X
        else:
            X2 = X2[:,kern.active_dims]

        base = np.pi * (X[:, None, :] - X2[None, :, :]) / kern.period

        sin_base = np.sin( base )
        exp_dist = np.exp( -0.5* np.sum( np.square(  sin_base / kern.lengthscale ), axis = -1 ) )

        dwl = kern.variance * (1.0/np.square(kern.lengthscale)) * sin_base*np.cos(base) * (base / kern.period)


        if kern.ARD1: # different periods
            dK_dp = dwl * exp_dist[:,:,None] 
        else:  # same period
            dK_dp = dwl.sum(-1) * exp_dist 

        return dK_dp

   
setattr(StdPeriodic, "dK_dp", dK_dp_per)


def dK_dp_diag_per(kern, X):

    X = X[:,kern.active_dims]

    if kern.ARD1:
        return np.zeros(X.shape)
    else:
        return np.zeros(X.shape[0])


setattr(StdPeriodic, "dK_dp_diag", dK_dp_diag_per)



def inv_logDet(M):
    #M = np.ascontiguousarray(M)
    #L_M = np.linalg.cholesky(M)            #############################################################

    A = np.ascontiguousarray(M)
    L_M, info = lapack.dpotrf(A, lower=1)

    iM, _ = dpotri(L_M)
    logDetM = 2*sum(np.log(np.diag(L_M)))


    return iM, logDetM


def inv_logDet_jit(M, jit=0):
    #M = np.ascontiguousarray(M)
    #L_M = np.linalg.cholesky(M)            #############################################################

    A = np.ascontiguousarray(M+np.eye(M.shape[0])*jit)
    L_M, info = lapack.dpotrf(A, lower=1)

    iM, _ = dpotri(L_M)
    logDetM = 2*sum(np.log(np.diag(L_M)))


    return iM, logDetM





def inv_c(M):
    A = np.ascontiguousarray(M)
    L_M, info = lapack.dpotrf(A, lower=1)

    #L_M = np.linalg.cholesky(M)            
    iM, _ = dpotri(L_M)

    return iM

def inv_jit(M, jit):

    return inv_c(M + np.eye(M.shape[0])*jit )


def dot3lr(A,B,C):
    return np.dot(np.dot(A,B),C)


def dot3rl(A,B,C):
    return np.dot(A,np.dot(B,C))

# computes the diag of H.T K H
def diag_HtKH(H,K):
    return np.sum( np.dot(K,H) * H, 0)

# computes the diag of A.T K B
def diag_AtKB(A,K,B):
    # K symmetric
    # A,B same outer dimension such that product is square
    return np.sum( np.dot(K,A) * B, 0)

def r(arr, rn=1):
    return np.round(arr,rn)

def r1(arr):
    return r(arr,1)
def r2(arr):
    return r(arr,2)
def r3(arr):
    return r(arr,3)



from numpy.linalg import inv as inv_np

def invEE(M, jit=1e-15):

    return inv_np(M + np.eye(M.shape[0])*jit)





