CC=icc
MPI=mpicxx
GCC=g++
NVCC=nvcc
CCFLAGS=-fopenmp -std=c++14 -fno-exceptions
CUDAFLAGS=-Xcompiler -fopenmp -std=c++14
CCMPI_FLAG=-DMPI_ON=1
CCUDA_FLAG=-DCUDA_ON=1
CCDEBUG=-O1 -pg -g
CCRUN=-O3

SRC=main.cpp nextdbscan_new.cpp deep_io.cpp
OUT=ndbscan
OUT_BIN=cvs2bin
OBJ=dbscan.o

all:
	rm -f $(OUT)-icc
	$(CC) $(CCRUN) $(CCFLAGS) $(SRC) -o $(OUT)-icc

mpi:
	rm -f $(OUT)-mpi
	$(MPI) $(CCRUN) $(CCFLAGS) $(CCMPI_FLAG) $(SRC) -o $(OUT)-mpi

gcc:
	rm -f $(OUT)-gcc
	$(GCC) $(CCRUN) $(CCFLAGS) $(SRC) -o $(OUT)-gcc

cu:
	rm -f $(OUT)-cu
	$(NVCC) $(CCRUN) $(CUDAFLAGS) $(CCUDA_FLAG) -x cu $(SRC) -o $(OUT)-cu

cvs2bin:
	rm -f $(OUT_BIN)
	$(CC) $(CCRUN) $(CCFLAGS) csv2bin.cpp -o $(OUT_BIN)

debug:
	$(CC) $(CCDEBUG) $(CCFLAGS) $(SRC) -o $(OUT)
