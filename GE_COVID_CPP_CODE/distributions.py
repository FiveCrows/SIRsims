# Distributions useful for Covid modeling

import numpy as np
import scipy.special as sp

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

def weibull(shape, scale, n):
    wei = scale * np.random.weibull(shape, size=n)
    return wei

def lognormal(mean, std, n):
    # mean, std of underlying normal distribution
    lognorm = np.random.lognormal(mean, std, n)
    return lognorm

def weibullPDF(shape, scale, a, b, n):
    # the range of the x-axis is a,b
    x = np.linspace(a, b, n)
    pdfs = (shape/scale)*(x/scale)**(shape-1.) * np.exp(-(x/scale)**shape)
    mean = np.mean(np.dot(x,pdfs)) * (b-a)/(n-1.)
    #for i in zip(x, pdfs):
        #print(i)
    #print(f"mean= {mean}")
    return x, pdfs



if __name__ == "__main__":
    shape = 2.826
    scale = 5.665
    wei = weibull(shape, scale, 100000)
    mean = np.mean(wei)
    var = np.var(wei)
    std = np.std(wei)
    print("mean,var,std= ", mean, var, std)
    # Compute theoretical mean
    mean = scale * sp.gamma(1.+1/shape)
    var = scale**2 *(sp.gamma(1.+2./shape) - (sp.gamma(1.+1./shape))**2)
    print("theor mean: ", mean)
    print("theor var: ", var)

    print("\n=== lognormal ===")
    e_mean = 1.644
    e_std = 0.363
    logn= lognormal(e_mean, e_std, 100000)
    mean = np.exp(e_mean + e_std**2/2.)
    var = (np.exp(e_std**2) -1) * np.exp(2*e_mean+e_std**2)
    std = var**(0.5)
    print(f"log(x), mean/std/var= {mean}, {std}, {std**2}")
    a_mean = np.mean(logn)
    a_std = np.std(logn)
    print(f"log(x), actual mean/std/var= {a_mean}, {a_std}, {a_std**2}")


