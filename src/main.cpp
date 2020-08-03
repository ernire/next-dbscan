/*
Copyright (c) 2019, Ernir Erlingsson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include "nextdbscan.h"
#include "deep_io.h"
#include "next_util.h"
//#define MPI_ON
//#define HDF5_ON
#ifdef MPI_ON
#include <mpi.h>
#endif
#include <omp.h>

void usage() {
    std::cout << "NextDBSCAN compiled for OpenMP";
#ifdef MPI_ON
    std::cout << ", MPI";
#endif
#ifdef HDF5_ON
    std::cout << ", HDF5";
#endif
#ifdef CUDA_ON
    std::cout << ", CUDA (V100)";
#endif
    std::cout << std::endl << std::endl;
    std::cout << "Usage: [executable] -m minPoints -e epsilon -t threads [input file]" << std::endl;
    std::cout << "    -m minPoints : DBSCAN parameter, minimum number of points required to form a cluster, postive integer, required" << std::endl;
    std::cout << "    -e epsilon   : DBSCAN parameter, maximum neighborhood search radius for cluster, positive floating point, required" << std::endl;
    std::cout << "    -t threads   : Processing parameter, the number of threads to use, positive integer, defaults to number of cores" << std::endl;
    std::cout << "    -o output    : Output file containing the cluster ids in the same order as the input" << std::endl;
    std::cout << "    -h help      : Show this help message" << std::endl << std::endl;
    std::cout << "Supported Input Types:" << std::endl;

    std::cout << ".csv: Text file with one sample/point per line and features/dimensions separated by a space delimiter, i.e. ' '" << std::endl;
    std::cout << ".bin: Custom binary format for faster file reads. Use cvs2bin executable to transform csv into bin files." << std::endl;
#ifdef HDF5_ON
    std::cout << ".hdf5: The best I/O performance when using multiple nodes." << std::endl;
#endif
}

int main(int argc, char** argv) {
    char option;
    long m = -1;
    float e = -1;
    long n_threads = -1;
    int errors = 0;
    std::string input_file;
    std::string output_file = "";

    /*
    while ((option = getopt(argc, argv, "hm:e:o:t:d:")) != -1) {
        switch (option) {
            case 'm': {
                ssize_t minPoints = std::stoll(optarg);
                if (minPoints <= 0L) {
                    std::cerr << "minPoints must be a positive integer number, but was " << optarg << std::endl;
                    ++errors;
                } else {
                    m = static_cast<size_t>(minPoints);
                }
                break;
            }
            case 'e': {
                float epsilon = std::stof(optarg);
                if (epsilon <= 0.0f) {
                    std::cerr << "epsilon must be a positive floating struct_label number, but was " << optarg << std::endl;
                    ++errors;
                }
                else {
                    e = epsilon;
                }
                break;
            }
            case 't': {
                ssize_t threads = std::stoll(optarg);
                if (threads <= 0L) {
                    std::cerr << "thread count must be a positive integer number, but was " << optarg << std::endl;
                    ++errors;
                } else {
                    n_threads = static_cast<size_t>(threads);
                }
                break;
            }
            case 'o': {
                output_file = optarg;
                std::cout << "output file: " << output_file << std::endl;
            }
            default:
                break;
        }
    }
    if (argc - optind <= 0) {
        input_file = "../input/aloi-hsb-2x2x2_trimmed.csv";
    }
    else if (argc - optind > 1) {
        std::cerr << "Please provide only one data file" << std::endl;
        ++errors;
    }
    else {
        input_file = argv[optind];
    }
     */
    m = static_cast<long>(std::stoll(argv[2]));
    e = std::stof(argv[4]);
    n_threads = static_cast<long>(std::stoll(argv[6]));
    input_file = argv[7];
    std::cout << "input file: " << input_file << " : " << m << " : " << e << " n_threads: " << n_threads << std::endl;

    if (errors || m == -1 || e == -1) {
        std::cout << "Input Error: Please specify the m and e parameters" << std::endl << std::endl;
        usage();
        std::exit(EXIT_FAILURE);
    }
//    std::vector<uint> prime_cnt;
//    if (!next_util::small_prime_factor(prime_cnt, n_threads)) {
//        std::cerr << "ERROR: t must be a multiple of at least one of these primes: 2,3,5,7 (t=" << n_threads
//            << ")" << std::endl;
//        std::exit(EXIT_FAILURE);
//    }
    if (n_threads == -1) {
        n_threads = 1;
    }

    uint block_index = 0;
    uint blocks_no = 1;
#ifdef MPI_ON
    MPI_Init(&argc, &argv);
    int mpi_size;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    block_index = mpi_rank;
    blocks_no = mpi_size;
    std::cout << "rank: " << mpi_rank << " of " << mpi_size << std::endl;
#endif

//    omp_set_num_threads(4);
//    #pragma omp parallel
//    {
//        std::cout << "Hello from OpenMP tid " << omp_get_thread_num()  << " : " << omp_get_num_threads() << std::endl;
//    }

    omp_set_dynamic(0);
    omp_set_num_threads((int) n_threads);

    if (block_index == 0)
        std::cout << "Starting NextDBSCAN with m: " << m << ", e: " << e << ", t: "
                  << n_threads << " file:" << input_file << std::endl;
    nextdbscan::result results = nextdbscan::start(m, e, n_threads, input_file, block_index, blocks_no);
    if (block_index == 0) {
        std::cout << std::endl;
        std::cout << "Estimated clusters: " << results.clusters << std::endl;
        std::cout << "Core Points: " << results.core_count << std::endl;
        std::cout << "Noise Points: " << results.noise << std::endl;

        if (output_file.length() > 0) {
            std::cout << "Writing output to " << output_file << std::endl;
            std::ofstream os(output_file);
            // TODO
            for (int i = 0; i < results.n; ++i) {
                os << results.point_clusters[i] << std::endl;
            }
//            for (auto &c : results.point_clusters) {
//                os << c << '\n';
//            }
            os.flush();
            os.close();
            std::cout << "Done!" << std::endl;
        }
    }

#ifdef MPI_ON
    MPI_Finalize();
#endif

}
