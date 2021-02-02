# Read parameter files and put variables and their values in a glob_dict = {}

# This approach makes it hard to change a parameter in the parameter file. 
# It would be better to have a parameter file written as a Python dictionary
# but then how to read it into a C++ file. 

import pandas as pd
from scanf import *

def readParamFile(filenm, glob_dict):
    with open(filenm, 'r') as fd:
        while True:
            line = fd.readline().rstrip()
            if line[0:3] == "EOF":
                break
            strg, val = scanf("%s %f", line)
            glob_dict[strg] = val

if __name__ == "__main__":
    readParamFile("p", glob_dict)
    print(glob_dict)

