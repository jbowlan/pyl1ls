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

import pylab

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
    # n = number of data points, p = number of predictors
    n,p = X.shape
    
    # mu = regressed version of y sice there are no predictors it is initially the 
    # zero vector
    mu = zeros(n)
    
    # active set and inactive set - they should invariably be complements
    act_set = []
    inact_set = range(p)

    # current regression coefficients and correlation with residual
    beta = zeros((p+1,p))
    corr = zeros((p+1,p))
    
    # initial cholesky decomposition of the gram matrix
    # since the active set is empty this is the empty matrix
    R = zeros((0,0))

    # add the variables one at a time
    for k in xrange(p):
        print "NEW ITERATION k = ", k, " active_set = ", act_set
        
        # compute the current correlation
        c = dot(X.T, y - mu)
        
        print "current correlation = ", c
        
        # store the result
        corr[k,:] = c
        
        # choose the predictor with the maximum correlation and add it to the active
        # set
        jmax = inact_set[argmax(abs(c[inact_set]))]
        C = c[jmax]
        
        print "iteration = ", k, " jmax = ", jmax, " C = ", C
        
        # add the most correlated predictor to the active set
        R = cholinsert(R,X[:,jmax],X[:,act_set])
        act_set.append(jmax)
        inact_set.remove(jmax)
        
        # get the signs of the correlations
        s = sign(c[act_set])
        s = s.reshape(len(s),1)
        print "sign = ", s
        
        # move in the direction of the least squares solution restricted to the active
        # set
         
        GA1 = solve(R,solve(R.T, s))
        AA = 1/sqrt(sum(GA1 * s))
        w = AA * GA1
        
        print "AA = ", AA
        print "w = ", w
        
        u = dot(X[:,act_set], w).reshape(-1)

        print "norm of u = ", norm(u)
        print "u.shape = ", u.shape
    
        # if this is the last iteration i.e. all variables are in the
        # active set, then set the step toward the full least squares
        # solution
        if k == p:
            print "last variable going all the way to least squares solution"
            gamma = C / AA
        else:
            a = dot(X.T,u)
            a = a.reshape((len(a),))
            
            tmp = r_[(C - c[inact_set])/(AA - a[inact_set]), 
                     (C + c[inact_set])/(AA + a[inact_set])]
            
            gamma = min(r_[tmp[tmp > 0], array([C/AA]).reshape(-1)])

        print "ITER k = ", k, ", gamma = ", gamma
        
        mu = mu + gamma * u
        
        if beta.shape[0] < k:        
            beta = c_[beta, zeros((beta.shape[0],))]
        beta[k+1,act_set] = beta[k,act_set] + gamma*w.T.reshape(-1)
    
    return beta, corr

def plot_lars(X,y):
    # make two plots of the correlations with time
    beta, corr = lars(X,y)
    l1B = scipy.sum(abs(beta), axis=1)
    
    pylab.figure()
    for j in xrange(corr.shape[1]):
        pylab.plot(arange(corr.shape[0]), abs(corr[:,j]), label='%d'%j)
    pylab.legend()


    
def load_data(name='diabetes.data'):
    print "loading ", name
    dat = scipy.loadtxt(name, skiprows=1)
    print "dat.shape = ", dat.shape
    cs = dat.shape[1]

    X = dat[:,:cs-1]
    y = dat[:,cs-1]
    
    X = (X - mean(X, axis=0))/std(X, axis=0)
    y = (y - mean(y))/std(y)

    return X,y


