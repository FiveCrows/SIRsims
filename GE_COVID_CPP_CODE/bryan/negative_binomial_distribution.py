import networkx as nx
import EoN as eon
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import distributions as dist

# Experiment with creating a negative binomial of mean R0 and variance
# R0 * (1 + k*R0)


R0 = 2.5
dispersion = k = 0.75
print("R0= ", R0)
print("k= ", k)

N = 500000
lmbda = dist.gamma(R0, k, N)
print("lmbda= ", lmbda)

# the results should be equivalent to a properly scaled Negative Binomial
print("Composite Poisson")
expo = dist.poisson(lmbda, N)

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

#samples = np.random.negative_binomial(r, p, N)
samples = dist.negativeBinomial(R0, k, N)
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
muhat   = np.mean(samples)
varhat  = np.var(samples)
stdhat  = np.sqrt(varhat)
print("R0= ", R0, ",  R0*(1.+R0)= ", R0*(1.+R0*k))
print("mean=1/p= %f" % (1/p))
print("var=(1-p)/p**2= %f" % ((1-p)/p**2))
print("muhat= %f, varhat= %f, stdhat= %f\n" % (muhat, varhat, stdhat))
print("(varhat/muhat-1.)= ", varhat/muhat-1.)

strg = """
==============================================
EXPONENTIAL MIXTURE WITH POISSON is a GEOMETRIC DISTRIBUTION
interpreted as the number of FAILURES before the first SUCCESS
Choose R according to Poisson with mean R0
Sample Poisson with this R

My formulas for mu, var=sigma^2 are not correct 
==============================================
"""
print(strg)

samples = dist.exponential(R0, N)
muhat   = np.mean(samples)
varhat  = np.var(samples)
print("dist.exponential: muhat= %f, varhat= %f" % (muhat, varhat))
samples = dist.poisson(samples, N)
muhat   = np.mean(samples)
varhat  = np.var(samples)

p = 1. / (1.+R0)  # p of resulting geometric distribution
print("exponential-poisson: muhat= %f, varhat= %f" % (muhat, varhat))
print("p=1./(1.+R0)= ", p)
print("Geometric mean=1/p= %f" % (1/p))
print("Geometric var=(1-p)/p**2= %f" % ((1-p)/p**2))

# Equivalent geometric
# p* = p/(1-p) ==> p* (1-p) = p ==> p* = p(p*+1) => p = p* / (p* + 1)
samples = np.random.geometric(p, N)  # number of trials to achieve success
muhat   = np.mean(samples)    # average number of trials to achieve success
muhat   = np.mean(samples-1)  # average number of failures to achieve success
varhat  = np.var(samples)
print("geometric(p=%f): muhat=      %f" % (p, muhat))
print("geometric(p=%f): varhat=     %f" % (p, varhat))
print("geometric(p=%f): mu=(1-p)/p= %f" % (p, (1-p)/p))
print("geometric(p=%f): var= %f"        % (p, (1-p)/p**2))


strg = """
==============================================
POISSON MIXTURE WITH GAMMA is a NEGATIVE BINOMIAL
Choose R according to Poisson with mean R0
Sample Poisson with this R
==============================================
"""
print(strg)

samples = dist.gamma(R0, k, N)
muhat   = np.mean(samples)
varhat  = np.var(samples)
print("dist.gamma mean: ", muhat)
print("dist.gamma var: ", varhat)
samples = dist.poisson(samples, N)
muhat   = np.mean(samples)
varhat  = np.var(samples)
print("k= ", k)
print("gamma-poisson: muhat= %f" % (muhat))
print("gamma-poisson: varhat= %f" % (varhat))
print("R0*(1.+R0/k)= ", R0*(1.+R0/k))
# I must figure this out. 
quit()

