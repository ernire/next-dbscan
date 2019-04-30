cmake_minimum_required(VERSION 3.12)
project(next-dbscan)

FIND_PACKAGE(OpenMP)
set(CMAKE_CXX_STANDARD 14)
set(GCC_COVERAGE_COMPILE_FLAGS "-Wall -Wextra -O3 -fopenmp -fno-exceptions")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )
add_executable(odbscan src/nextdbscan.cpp)