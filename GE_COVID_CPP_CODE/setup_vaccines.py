
import numpy as np
import pandas as pd

# Author: Gordon Erlebacher
# Date: 2021-01-01

# How to take vaccine reliability into account. Through edge weights, I suppose.
# Output of this script: a file with three columns: 
#   1. node number
#   2. vaccinated or not (0/1)
#   3. effectiveness of the vaccine  [0, 1]
#      If the vaccine is 100% effective, the value is 1

# generate list of people to vaccinate
# These people will be Recovered at initial time
# Create weights in [0,1]: 0.7 will signify that the vaccine is 70% effective in preventing 
# the latent stage. 

def setupVaccines(N, N_vacc, nb_doses, perc_vacc, filenm):
    nodes = list(range(0,N))
    vaccines     = np.random.choice(N, size=N_vacc//nb_doses, replace=False)
    vaccinated   = np.zeros(N, dtype='int')
    vacc_weights = np.ones(N, dtype='float')
    vaccinated[vaccines] = 1
    vacc_weights[vaccines] = 1.0

    output = np.vstack([nodes, vaccinated, vacc_weights]).transpose()
    with open(filenm, "w") as fd:
        fd.write("%d\n" % N)
        np.savetxt(fd, output, fmt="%6d %1d %5.2f")
#----------------------------------------------------------------------
if __name__ == "__main__":
    N = 260542
    N_vacc = 100000   # number of vaccine doses
    nb_doses = 1     # number of doses per person
    perc_vacc = 0.5  # percent vaccinated
    
    setupVaccines(N, N_vacc, nb_doses, perc_vacc, filenm="vaccines.csv")
#----------------------------------------------------------------------
