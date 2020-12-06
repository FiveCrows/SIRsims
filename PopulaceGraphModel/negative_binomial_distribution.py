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
    print("Poisson: mean/var= ", np.mean(pois), np.var(pois))
    return(pois)

def expon(lmbda, n):
    #expo = np.random.exponential(lmbda, n)
    expo = np.random.exponential(lmbda, n)
    # Expect mean,std = lmbda, lmbda
    print("mean/var/std= ", np.mean(expo), np.var(expo), np.std(expo))
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

p = k*R0/(1+k*R0)
p = 1-p
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
quit()

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

p = k*R0 / (1+k*R0)
p = 1. - k*R0 / (1+k*R0)
r =  1 / k
print("p= ", p, ",  r= ", r)

nb = np.random.negative_binomial(r, p, N)
print("R0= ", R0, ",  R0*(1.+R0*k)= ", R0*(1.+R0*k))
print("mean/var/std(nb)= ", np.mean(nb), np.var(nb), np.std(nb))
print("mu=p*r/(1-p)= ", p*r/(1-p))
print("var=mean/p= ", p*r/(1-p)**2)  

# This seems the same as the calculated mu/var. So p is prob of failure
# (I interchanged p and 1-p in the formula above)
print("mu=(1-p)*r/p= ", (1-p)*r/p)
print("var=mean/p= ", (1-p)*r/p**2)  
# This means that
# 1-p = k*R0 / (1 + k*R0)
# p = 1 / (1 + k*R0) 
# r = R0 * p / (1-p)

#print(help(np.random.negative_binomial))

# Once I figure this out. 
quit()

