GCC= g++
# For Gordon. Added extern "C" in header.h. Might not work for Bryan
#CPPFLAGS= -fgnu-runtime -std=c++14 -Wall -lm -lgsl -lgslcblas -O2 -Wno-unused-result 
CPPFLAGS= -std=c++14 -Wall -lm -lgsl -lgslcblas -O2 -Wno-unused-result 
#CPPFLAGS= -g -std=c++14 -Wall -lm -lgsl -lgslcblas -O2 -Wno-unused-result 
SOURCES_ORIG=  main.c read.c utilities.c spread.c
SOURCES_SIR=  main.c read_SIR.c utilities.c spread_SIR_clean.c
SOURCES_SEPIR= main.c read_SIR.c utilities.c spread_SEPIR_clean.c
SOURCES_SEIR= main.cpp G.cpp 

SOURCES_DIST= distributions.cpp

.PHONY: seir sir orig sepir distribution
seir:
	$(GCC) $(CPPFLAGS) $(SOURCES_SEIR) -o seir

distribution: 
	$(GCC) -DBASELINE $(SOURCES_DIST) $(CPPFLAGS) -o dist
