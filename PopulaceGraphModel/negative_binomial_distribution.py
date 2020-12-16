import networkx as nx
import EoN as eon
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random

# Experiment with creating a negative binomial of mean R0 and variance
# R0 * (1 + k*R0)

def gamma(R0, k, n):
    # This is correct
    alpha = 1 / k 
    beta = R0 * k
    gammas = np.random.gamma(alpha, beta, n)
    return gammas

def poisson(lmbda, n):
    # This is correct
    pois = np.random.poisson(lmbda, n)
    #print("Poisson: mean/var= ", np.mean(pois), np.var(pois))
    return(pois)

def expon(lmbda, n):
    #expo = np.random.exponential(lmbda, n)
    expo = np.random.exponential(lmbda, n)
    # Expect mean,std = lmbda, lmbda
    #print("mean/var/std= ", np.mean(expo), np.var(expo), np.std(expo))
    return expo


#gams = gamma(5., .045, 100000)
#lmbda = 3.
#lmbda = np.random.uniform(0.,1.,50000)

R0 = 2.5
dispersion = k = 0.75
print("R0= ", R0)
print("k= ", k)

N = 500000
lmbda = gamma(R0, k, N)
print("lmbda= ", lmbda)

# the results should be equivalent to a properly scaled Negative Binomial
print("Composite Poisson")
expo = poisson(lmbda, N)

"""
negative_binomial(n, p, size=None)

    Draw samples from a negative binomial distribution.

    Samples are drawn from a negative binomial distribution with specified
    parameters, `n` successes and `p` probability of success where `n`
    is > 0 and `p` is in the interval [0, 1].
"""

p = 1. - k*R0/(1+k*R0)
#p = (1+k*R0-k*R0)/(1+k*R0) = 1/(1+k*R0)
n = 1./k
neg_bin = np.random.negative_binomial(n, p, N)
mean = np.mean(neg_bin)
var = np.var(neg_bin)
print("Computed Statistics of negative binomial")
print("np.mean NB: ", mean)
print("np.var NB: ", var)
print("mean*(1+k*mean)= ", mean*(1+k*mean))
print("dispersion=(var-mean)/mean ", (var-mean)/mean**2) # var = mean*(1+k*mean)
print("p= ", p, ",   n= ", n)
print("R0= ", R0)
print("k= ", k)
print("R0*(1+k*R0)= ", R0*(1+k*R0))

p = 1. / (1. + R0 / k)
r = p / (1.-p) / R0

# The composite gives Gamma(r+k)/[k!*Gamma(r)] p**k (1-p)**r
# This generates the negative binomial f(k; r,p) (Wikipedia)
# f(k; r, p) = Pr(X=k) = C((i+r-1), r-1)) (1-p)**k p**r
# r is number of successes
# k is number of failures
# p is probability of success
# Mean: p*r/(1-p), 
# var = p*r/(1-p)**2 = mean/(1-p)
# Mean = mu = R0 = p*r/(1-p)
# var = R0 + k*R0**2 = mean/(1-p)
# Compute p and r
# mu*(1-p) = R0*(1-p) = p*r ==> 
# R0 + k*R0**2 = R0 / (1-p) ==> (1-p) = R0 / (R0 + k*R0**2) = 1/(1+k*R0)
#   ==> p = 1 - 1/(1+k*R0) = k*R0 / (1+k*R0)
#   ==> r = R0 * (1-p) / p

#p = k*R0 / (1+k*R0)
#p = 1. - k*R0 / (1+k*R0)

strg = """
=============================================
NEGATIVE BINOMIAL DISTRIBUTION
p = 1 / (1+R0/k)
r =  k
nb = np.random.negative_binomial(r, p, N)
=============================================
"""
print(strg)

N = 200000
p = 1. / (1+R0/k)
r =  k

samples = np.random.negative_binomial(r, p, N)
muhat = np.mean(samples)
varhat = np.var(samples)
stdhat = np.sqrt(varhat)

print("p= ", p, ",  r= ", r)
print("R0= ", R0, ",  R0*(1.+R0/k)= ", R0*(1.+R0/k))
print("muhat= %f, varhat= %f, stdhat= %f\n" % (muhat, varhat, stdhat))
print("muhat= ", muhat)
print("k= ", k)
print("muhat*(1.+muhat/k)= %f" % (muhat*(1.+muhat/k)))

strg = """
=============================================
GEOMETRIC DISTRIBUTION
p = 1 / (1+R0)
nb = np.random.geometric(p, N)
=============================================
"""
print(strg)

print("N: %d samples" % N)
p = 1 / R0
samples = np.random.geometric(p, N)
muhat = np.mean(samples)
varhat = np.var(samples)
stdhat = np.sqrt(varhat)
print("R0= ", R0, ",  R0*(1.+R0)= ", R0*(1.+R0*k))
print("mean=1/p= %f" % (1/p))
print("var=(1-p)/p**2= %f" % ((1-p)/p**2))
print("muhat= %f, varhat= %f, stdhat= %f\n" % (muhat, varhat, stdhat))
print("(varhat/muhat-1.)= ", varhat/muhat-1.)

strg = """
==============================================
POISSON MIXTURE WITH POISSON is a GEOMETRIC DISTRIBUTION
Choose R according to Poisson with mean R0
Sample Poission with this R

My formulas for mu, var=sigma^2 are not correct 
==============================================
"""
print(strg)

samples = poisson(R0, N)
samples = poisson(samples, N)
muhat = np.mean(samples)
varhat = np.var(samples)
stdhat = np.sqrt(varhat)
p = 1. / (1.+R0)  # p of resulting geometric distribution
print("muhat= %f, varhat= %f, stdhat= %f\n" % (muhat, varhat, stdhat))
print("p = ", p)
print("mean=1/p= %f" % (1/p))
print("var=(1-p)/p**2= %f" % ((1-p)/p**2))

strg = """
==============================================
POISSON MIXTURE WITH GAMMA is a NEGATIVE BINOMIAL
Choose R according to Poisson with mean R0
Sample Poission with this R

My formulas for mu, var=sigma^2 are not correct 
==============================================
"""
print(strg)

samples =  gamma(R0, k, N)
print("mean gamma: ", np.mean(samples))
print("var gamma: ", np.var(samples))
samples = poisson(samples, N)
print("k= ", k)
print("muhat= %f, varhat= %f, stdhat= %f\n" % (muhat, varhat, stdhat))
print("var/mean= ", np.var(samples)/np.mean(samples))
# Once I figure this out. 
quit()

