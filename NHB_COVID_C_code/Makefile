CC= gcc
GCC= g++
CPPFLAGS= -Wall -lm -lgsl -lgslcblas -O2 -Wno-unused-result
CFLAGS= -Wall -std=c99 -lm -lgsl -lgslcblas -O2 -Wno-unused-result
#CFLAGS= -g -Wall -std=c99 -lm -lgsl -lgslcblas -O0 -Wno-unused-result
#SOURCES=  main.c read.c utilities.c spread.c
#SOURCES=  main.c read_SIR.c utilities.c spread.c
#SOURCES=  main.c read_SIR.c utilities.c spread_SIR.c
SOURCES_ORIG=  main.c read.c utilities.c spread.c
SOURCES_SIR=  main.c read_SIR.c utilities.c spread_SIR_clean.c
SOURCES_SEIR= main.c read_SIR.c utilities.c spread_SEIR_clean.c
SOURCES_SEPIR= main.c read_SIR.c utilities.c spread_SEPIR_clean.c
SOURCES_DIST= distributions.cpp

.PHONY: seir sir orig sepir distribution
orig:
	$(CC) -DBASELINE $(SOURCES_ORIG) $(CFLAGS) -o orig

sir:
	$(CC) -DBASELINE $(SOURCES_SIR) $(CFLAGS) -o sir

seir:
	$(CC) -DBASELINE $(SOURCES_SEIR) $(CFLAGS) -o seir

sepir:
	$(CC) -DBASELINE $(SOURCES_SEPIR) $(CFLAGS) -o sepir

distribution: 
	$(GCC) -DBASELINE $(SOURCES_DIST) $(CPPFLAGS) -o dist
