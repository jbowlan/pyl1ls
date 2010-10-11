# python implementation of least angle regression LARS
# this solves the l1 regularized least squares problem for 
# linear regression.

# Author: John Bowlan

import scipy
import scipy.linalg

from scipy import array, log, c_, r_, shape, pi, sqrt, zeros, mean, linspace, \
    exp, diff, sign, arange, average, interp, floor, iterable, argmin, std, \
    var, ones, dot, argmax

from scipy.linalg import solve, cho_solve, norm

"""
LEAST angle regression is a "homotopy" method for solving a l1
regularized least squares problem.  It is

Summary of LARS algorithm from Hastie, Tibshirani, et. al. p 74

1. standardize the predictors to have mean zero and unit norm. Start
with the residual r = y - ybar, beta_1 = ... beta_p = 0

2. Find the predictor x_j most correlated with r

3. Move beta_j from 0 towards its least-squares coefficient
dot(X[j,:], r) until some other competitor X[k,:] has as much
correlation with the current residual as does X[j,:]

4. Move beta[j] and beta[k] in the direction defined by the joint
least squares coefficient of the current residual on (X[j,:],X[k,:])
until some other competitor X[l] has as much correlation with the
current resuidual

5. Continue in this way until all p predictors have been
entered. After min(N-1,p) steps, we arrive at the full least-squares
solution
"""

# useful utility functions updates R as a cholesky factorization
# dot(R.T,R) = dot(X.T,X) R is current cholesky matrix to be updated,
# x is the column vector representing the variable to be added and X
# is the data matrix with the currently active variables other than x
#
def cholinsert(R, x, X):
    diag_k = dot(x.T,x)
    if R.shape == (0,0):
        R = array([[sqrt(diag_k)]])
    else:
        col_k = dot(x.T,X)
                
        R_k = solve(R,col_k)
        R_kk = sqrt(diag_k - dot(R_k.T,R_k))
        R = r_[c_[R,R_k],c_[zeros((1,R.shape[0])),R_kk]]
    
    return R

# LASSO requires implementation of choldelete which removes a variable
# from the cholesky factorization

def lars(X, y):
    # n is the number of variables, p is the number of "predictors" or
    # basis vectors

    # the predictors are assumed to be standardized and y is centered.

    # in the example of the prostate data n would be the number 
    n,p = X.shape
    
    mu = zeros(n)
    act_set = []
    inact_set = range(p)

    k = 0
    vs = 0
    nvs = min(n-1,p)

    beta = zeros((2*nvs,p))

    maxiter = nvs * 8

    # initial cholesky decomposition of the gram matrix
    R = zeros((0,0))

    while vs < nvs and k < maxiter:      
        print "new iteration: vs = ", vs, " nvs = ", nvs, " k = ", k
        print "mu.shape = ", mu.shape
        #print "mu = ", mu

        # compute correlation with inactive set
        # and element that has the maximum correlation
        c = dot(X.T, y - mu)
        #c = c.reshape(1,len(c))
        jia = argmax(abs(c[inact_set]))
        j = inact_set[jia]
        C = c[j]
        
        print "predictor ", j, " max corr with w/ current residual: ", C
        print "adding ", j, " to active set"

        print "R shape before insert: ", R.shape

        # add the most correlated predictor to the active set
        R = cholinsert(R,X[:,j],X[:,act_set])
        act_set.append(j)
        inact_set.remove(j)
        vs += 1

        print "R shape after insert ", R.shape
        
        print "active set = ", act_set
        print "inactive set = ", inact_set 

        # get the signs of the correlations
        s = sign(c[act_set])
        s = s.reshape(len(s),1)
        #print "R.shape = ", R.shape
        #print "s.shape = ", s.shape

        # move in the direction of the least squares solution
        
        GA1 = solve(R,solve(R.T, s))
        AA = 1/sqrt(sum(GA1 * s))
        w = AA * GA1

        # equiangular direction - this should be a unit vector
        print "X[:,act_set].shape = ",X[:,act_set].shape
        #print "w.shape = ",w.shape

        u = dot(X[:,act_set], w).reshape(-1)

        #print "norm of u = ", norm(u)
        #print "u.shape = ", u.shape
    
        # if this is the last iteration i.e. all variables are in the
        # active set, then set the step toward the full least squares
        # solution
        if vs == nvs:
            print "last variable going all the way to least squares solution"
            gamma = C / AA
        else:
            a = dot(X.T,u)
            a = a.reshape((len(a),))
            tmp = r_[(C - c[inact_set])/(AA - a[inact_set]), 
                     (C + c[inact_set])/(AA + a[inact_set])]
            gamma = min(r_[tmp[tmp > 0], array([C/AA]).reshape(-1)])
        
        mu = mu + gamma * u
        
        if beta.shape[0] < k:        
            beta = c_[beta, zeros((beta.shape[0],))]
        beta[k+1,act_set] = beta[k,act_set] + gamma*w.T.reshape(-1)
                
        k += 1

    return beta

def load_pros_data():
    # load the prostate data
    dat = scipy.loadtxt('diabetes.data', skiprows=1)
    
    # now normalize and center the data
    X = dat[:,1:9]
    y = dat[:,9]
    train = dat[:,10] == 1
    
    X = (X - mean(X, axis=0))/std(X, axis=0)
    y = (y - mean(y))/std(y)

    return X,y

