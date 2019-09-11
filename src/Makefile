CC=icc
MPI=mpicxx
GCC=g++
CCFLAGS=-fopenmp -std=c++14 -fno-exceptions
CCFLAGS2=-DMPI_ON=1
CCDEBUG=-O1 -pg -g
CCRUN=-O3

SRC=main.cpp nextdbscan_new.cpp deep_io.cpp
OUT=dbscan
OBJ=dbscan.o

all:
	rm -f dbscan.exe
	$(MPI) $(CCRUN) $(CCFLAGS) $(CCFLAGS2) $(SRC) -o $(OUT)

omp:
	rm -f dbscan.exe
	$(CC) $(CCRUN) $(CCFLAGS) $(SRC) -o $(OUT)

gcc:
	rm -f dbscan.exe
	$(GCC) $(CCRUN) $(CCFLAGS) $(SRC) -o $(OUT)

debug:
	$(CC) $(CCDEBUG) $(CCFLAGS) $(SRC) -o $(OUT)
