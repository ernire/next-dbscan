CC=icc
MPI=mpicxx
GCC=g++
NVCC=nvcc
CCFLAGS=-fopenmp -std=c++14 -fno-exceptions
CUDAFLAGS=-Xcompiler -fopenmp -std=c++14 --expt-extended-lambda
CCMPI_FLAG=-DMPI_ON=1
CCUDA_FLAG=-DCUDA_ON=1
CCDEBUG=-O1 -pg -g
CCRUN=-O3

SRC=main.cpp nextdbscan_new.cpp deep_io.cpp
SRCCUDA=nextdbscan_cuda.cpp
SRCMPI=nextdbscan_mpi.cpp
SRCOMP=nextdbscan_omp.cpp
OUT=ndbscan
OUT_BIN=cvs2bin
OBJ=dbscan.o

all:
	rm -f $(OUT)-icc
	$(CC) $(CCRUN) $(CCFLAGS) $(SRC) $(SRCOMP) -o $(OUT)-icc

mpi:
	rm -f $(OUT)-mpi
	$(MPI) $(CCRUN) $(CCFLAGS) $(CCMPI_FLAG) $(SRC) $(SRCOMP) $(SRCMPI) -o $(OUT)-mpi

gcc:
	rm -f $(OUT)-gcc
	$(GCC) $(CCRUN) $(CCFLAGS) $(SRC) $(SRCOMP) -o $(OUT)-gcc

cu:
	rm -f $(OUT)-cu
	$(NVCC) $(CCRUN) $(CUDAFLAGS) $(CCUDA_FLAG) -x cu $(SRC) $(SRCCUDA) -o $(OUT)-cu

cvs2bin:
	rm -f $(OUT_BIN)
	$(CC) $(CCRUN) $(CCFLAGS) csv2bin.cpp -o $(OUT_BIN)

debug:
	$(CC) $(CCDEBUG) $(CCFLAGS) $(SRC) -o $(OUT)
