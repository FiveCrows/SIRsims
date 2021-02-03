#!/bin/sh 

# To run: ./installation_in_new_folder.x folder_name

mkdir $1
folder=$1
mkdir -p $1/data_ge/results
\cp Makefile *.py *.h *.hpp *.cpp $1/
\cp BA_Makefile  $1/
\cp data_ge/network.txt.gz $1/data_ge/
\cp data_ge/nodes.txt.gz $1/data_ge/
\cp data_ge/parameters_0.txt $1/data_ge/
