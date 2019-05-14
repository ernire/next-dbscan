cmake_minimum_required(VERSION 3.12)
project(next-dbscan)

enable_testing()

find_package(OpenMP)
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 14)
set(GCC_COVERAGE_COMPILE_FLAGS "-Wall -Wextra -O3 -fopenmp -fno-exceptions")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )

add_executable(ndbscan src/nextdbscan.cpp)

add_executable(ndbscan_tests src/nextdbscan_test.cpp)
target_link_libraries(ndbscan_tests ${GTEST_BOTH_LIBRARIES})