# Read parameter files and put variables and their values in a glob_dict = {}

import pandas as pd
from scanf import *

glob_dict = {}

def readParamFile(filenm, glob_dict):
    with open(filenm, 'r') as fd:
        while True:
            line = fd.readline().rstrip()
            if line[0:3] == "EOF":
                break
            strg, val = scanf("%s %f", line)
            glob_dict[strg] = val

readParamFile("p", glob_dict)
print(glob_dict)


