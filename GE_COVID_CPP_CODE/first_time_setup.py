import os, sys

# creates the full path
cmd = "mkdir -p data_ge/results"
os.system(cmd)

os.system("cd data_ge; gunzip network.txt.gz nodes.txt.gz")


