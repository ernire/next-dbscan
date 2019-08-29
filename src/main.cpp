//
// Created by Ernir Erlingsson (ernire@gmail.com, ernire.org) on 20.2.2019.
//
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
#include <getopt.h>
#include <fstream>

#include "nextdbscan.h"

void usage() {
    std::cout << "Usage: [executable] -m minPoints -e epsilon -t threads [input file]" << std::endl
              << "    Format : One data point per line, whereby each line contains the space-seperated values for each dimension '<dim 1> <dim 2> ... <dim n>'" << std::endl
              << "    -m minPoints : DBSCAN parameter, minimum number of points required to form a cluster, postive integer, required" << std::endl
              << "    -e epsilon   : DBSCAN parameter, maximum neighborhood search radius for cluster, positive floating point, required" << std::endl
              << "    -t threads   : Processing parameter, the number of threads to use, positive integer, defaults to number of cores" << std::endl
              << "    -o output    : Output file containing the cluster ids in the same order as the input" << std::endl
              << "    -h help      : Show this help message" << std::endl;
}

int main(int argc, char* const* argv) {
    char option;
    int m = -1, max_d = -1;
    float e = -1;
    int n_threads = -1;
    int errors = 0;
    std::string input_file;
    std::string output_file = "";

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
            case 'd': {
                ssize_t d = std::stoll(optarg);
                if (d <= 0L) {
                    std::cerr << "max dim must be a positive integer number, but was " << optarg << std::endl;
                    ++errors;
                } else {
                    max_d = d;
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

    if (errors || m == -1 || e == -1) {
        std::cout << "Input Error: Please specify the m, e" << std::endl;
        usage();
        std::exit(EXIT_FAILURE);
    }
    if (n_threads > 1 && n_threads % 2 == 1) {
        std::cerr << "The number of threads must be a multiple of 2 (2^0 also permitted)." << std::endl;
        std::exit(EXIT_FAILURE);
    } else if (n_threads == -1) {
        n_threads = 1;
    }
    std::cout << "Starting NextDBSCAN with m: " << m << ", e: " << e << ", t: "
        << n_threads << " file:" << input_file << std::endl;

    nextdbscan::result results = nextdbscan::start(m, e, n_threads, input_file);
    std::cout << std::endl;
    std::cout << "Estimated clusters: " << results.clusters << std::endl;
    std::cout << "Core Points: " << results.core_count << std::endl;
    std::cout << "Noise Points: " << results.noise << std::endl;

    if (output_file.length() > 0) {
        std::cout << "Writing output to " << output_file << std::endl;
        std::ofstream os(output_file);
        for (auto &c : *results.point_clusters) {
            os << c << '\n';
        }
        os.flush();
        os.close();
        std::cout << "Done!" << std::endl;
    }

}
