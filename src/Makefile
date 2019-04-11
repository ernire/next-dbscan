CC=g++
CCFLAGS=-fopenmp -std=c++14
CCDEBUG=-O1 -pg -g
CCRUN=-O3

SRC=nextdbscan.cpp
OUT=dbscan
OBJ=dbscan.o

all:
        rm -f dbscan.exe
        $(CC) $(CCRUN) $(CCFLAGS) $(SRC) -o $(OUT)

debug:
        $(CC) $(CCDEBUG) $(CCFLAGS) $(SRC) -o $(OUT)
