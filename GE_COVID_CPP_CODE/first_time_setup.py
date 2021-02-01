import os, sys

# creates the full path
cmd = "mkdir -p data_ge/results"
os.system(cmd)

if os.path.exists("data_ge/bak_gz/network.txt.gz"):
    os.system("cd data_ge; gunzip -c network.txt.gz > network.txt")

if os.path.exists("data_ge/bak_gz/bak_gz/nodes.txt.gz"):
    os.system("cd data_ge; gunzip -c nodes.txt.gz > nodes.txt")



