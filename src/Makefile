CC=icc
MPI=mpicxx
GCC=g++
CCFLAGS=-fopenmp -std=c++14 -fno-exceptions
CCMPI_FLAG=-DMPI_ON=1
CCDEBUG=-O1 -pg -g
CCRUN=-O3

SRC=main.cpp nextdbscan_new.cpp deep_io.cpp
OUT=ndbscan
OUT_MPI=ndbscan_mpi
OUT_BIN=cvs2bin
OBJ=dbscan.o

all:
	rm -f $(OUT)
	$(CC) $(CCRUN) $(CCFLAGS) $(SRC) -o $(OUT)

mpi:
	rm -f $(OUT_MPI)
	$(MPI) $(CCRUN) $(CCFLAGS) $(CCMPI_FLAG) $(SRC) -o $(OUT_MPI)

gcc:
	rm -f $(OUT)
	$(GCC) $(CCRUN) $(CCFLAGS) $(SRC) -o $(OUT)

cvs2bin:
	rm -f $(OUT_BIN)
	$(CC) $(CCRUN) $(CCFLAGS) csv2bin.cpp -o $(OUT_BIN)

debug:
	$(CC) $(CCDEBUG) $(CCFLAGS) $(SRC) -o $(OUT)
