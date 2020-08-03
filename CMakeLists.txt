cmake_minimum_required(VERSION 3.16)
#set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/bin/nvcc.exe")
project(next-dbscan LANGUAGES CXX)

set(GTEST OFF)
set(IS_MPI ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(GTEST)
    configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
    # Prevent overriding the parent project's compiler/linker
    # settings on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    # Add googletest directly to our build. This defines
    # the gtest and gtest_main targets.
    add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
            ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
            EXCLUDE_FROM_ALL)
    enable_testing()
    include_directories(${GTEST_INCLUDE_DIRS})
    add_executable(ndbscan_tests src/test.cpp ${NDB_FILES})
    target_link_libraries(ndbscan_tests gtest_main)
endif()

set(CMAKE_CXX_STANDARD 17)
set(NDB_FILES src/nextdbscan.cpp src/deep_io.cpp src/nextdbscan_omp.cpp src/nc_tree.cpp
        src/cell_processor.cpp)

if(IS_MPI)
    find_package(MPI REQUIRED)
#    find_package(OpenMP REQUIRED)
    # Include MPI includes
    include_directories(${MPI_INCLUDE_PATH})
    SET(CMAKE_CXX_COMPILER "${MPI_CXX_COMPILER}")
    SET(CMAKE_VERBOSE_MAKEFILE ON)

    add_definitions(-DMPI_ON)

    set(CMAKE_CXX_FLAGS "-Wall")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 ")
    add_executable(next-dbscan src/main.cpp ${NDB_FILES})
    if(MPI_COMPILE_FLAGS)
        set_target_properties(next-dbscan PROPERTIES
                COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
    endif()
    if(MPI_LINK_FLAGS)
        set_target_properties(next-dbscan PROPERTIES
                LINK_FLAGS "${MPI_LINK_FLAGS}")
    endif()
    target_link_libraries(next-dbscan ${MPI_LIBRARIES})
    else()
        set(GCC_COVERAGE_COMPILE_FLAGS "-std=c++17 -Wall -Wextra -O3 -fopenmp -fno-exceptions")
        set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )
        set(NDB_FILES src/nextdbscan.cpp src/deep_io.cpp src/nextdbscan_omp.cpp src/nc_tree.cpp
                src/cell_processor.cpp)
        add_executable(ndbscan src/main.cpp ${NDB_FILES})
        add_executable(normalize src/normalizer.cpp ${NDB_FILES})
        add_executable(csv2bin src/csv2bin.cpp src/deep_io.cpp)
endif()

#add_executable(ndbscan-cu src/main.cpp src/nextdbscan_cu.cu ${NDB_FILES})
#set_target_properties( ndbscan-cu PROPERTIES COMPILE_FLAGS "-DCUDA_ON=1" )

#set_target_properties(
#        ndbscan-cu
#        PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON)