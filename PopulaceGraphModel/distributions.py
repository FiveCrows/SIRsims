# Distributions useful for Covid modeling

import numpy as np

# Experiment with creating a negative binomial of mean R0 and variance
# R0 * (1 + k*R0)

def gamma(R0, k, n):
    # This is correct
    alpha = k 
    beta = R0 / k
    print("dist.gamma: alpha, beta= %f, %f" % (alpha, beta))
    gammas = np.random.gamma(alpha, beta, n)
    return gammas

def poisson(lmbda, n):
    # This is correct
    pois = np.random.poisson(lmbda, n)
    return(pois)

def exponential(lmbda, n):
    expos = np.random.exponential(lmbda, n)
    return expos

def negativeBinomial(R0, k, n):
    # average mu=R0, var=sigma**2=mu*(1+k*mu)
    # k: Dispersion
    # R0: reproduction number at early times
    p = 1 / (1+R0/k)
    #n = 1. / k  # do not know which is correct. This or next line
    r = k 
    neg_bins = np.random.negative_binomial(r, p, n)
    return neg_bins
