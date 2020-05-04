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
#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <omp.h>
#include <functional>
#include <numeric>
#include <random>

#ifdef MPI_ON
#include <mpi.h>
#endif
#ifdef HDF5_ON
#include <hdf5.h>
#endif
#include "nextdbscan.h"
#include "nc_tree.h"
#include "deep_io.h"
#include "next_util.h"

namespace nextdbscan {

    static bool g_quiet = false;

    void measure_duration(const std::string &name, const bool is_out, const std::function<void()> &callback) noexcept {
        auto start_timestamp = std::chrono::high_resolution_clock::now();
        std::cout << name << std::flush;
        callback();
        auto end_timestamp = std::chrono::high_resolution_clock::now();
        if (!g_quiet && is_out) {
            std::cout
                << std::chrono::duration_cast<std::chrono::milliseconds>(end_timestamp - start_timestamp).count()
                << " milliseconds\n";
        }
    }

    result collect_results(nc_tree &nc) noexcept {
        result res{0, 0, 0, nc.n_coords, new int[nc.n_coords]};
        res.core_count = nc.get_no_of_cores();
        res.clusters = nc.get_no_of_clusters();
        res.noise = nc.get_no_of_noise();
        return res;
    }

    void read_input_csv(const std::string &in_file, s_vec<float> &v_points, int max_d) noexcept {
        std::ifstream is(in_file);
        std::string line, buf;
        std::stringstream ss;
        int index = 0;
        while (std::getline(is, line)) {
            ss.str(std::string());
            ss.clear();
            ss << line;
            for (int j = 0; j < max_d; j++) {
                ss >> buf;
                v_points[index++] = atof(buf.c_str());
            }
        }
        is.close();
    }

    uint read_input_hdf5(const std::string &in_file, s_vec<float> &v_points, uint &max_d,
            const uint n_nodes, const uint node_index) noexcept {
        uint n = 0;
#ifdef HDF5_ON
        // TODO H5F_ACC_RDONLY ?
        hid_t file = H5Fopen(in_file.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        hid_t dset = H5Dopen1(file, "DBSCAN");
        hid_t fileSpace= H5Dget_space(dset);

        // Read dataset size and calculate chunk size
        hsize_t count[2];
        H5Sget_simple_extent_dims(fileSpace, count,NULL);
        n = count[0];
        max_d = count[1];
        std::cout << "HDF5 total size: " << n << std::endl;

//        hsize_t chunkSize =(this->m_totalSize / this->m_mpiSize) + 1;
//        hsize_t offset[2] = {this->m_mpiRank * chunkSize, 0};
//        count[0] = std::min(chunkSize, this->m_totalSize - offset[0]);
//        uint deep_io::get_block_size(const uint block_index, const uint number_of_samples, const uint number_of_blocks) {

        hsize_t block_size =  deep_io::get_block_size(node_index, n, n_nodes);
        hsize_t block_offset =  deep_io::get_block_start_offset(node_index, n, n_nodes);
        hsize_t offset[2] = {block_offset, 0};
        count[0] = block_size;
        v_points.resize(block_size * max_d);

        hid_t memSpace = H5Screate_simple(2, count, NULL);
        H5Sselect_hyperslab(fileSpace,H5S_SELECT_SET,offset, NULL, count, NULL);
        H5Dread(dset, H5T_IEEE_F32LE, memSpace, fileSpace,H5P_DEFAULT, &v_points[0]);

        H5Dclose(dset);
        H5Fclose(file);
#endif
#ifndef HDF5_ON
        std::cerr << "Error: HDF5 is not supported by this executable. "
                     "Use the cu-hdf5 flag when building from source to support HDF5." << std::endl;
        exit(-1);
#endif
        return n;
    }

    inline bool is_equal(const std::string &in_file, const std::string &s_cmp) noexcept {
        return in_file.compare(in_file.size() - s_cmp.size(), s_cmp.size(), s_cmp) == 0;
    }

    uint read_input(const std::string &in_file, s_vec<float> &v_points, uint &n, uint &max_d,
            const uint n_nodes, const uint node_index) noexcept {
        std::string s_cmp = ".bin";
        std::string s_cmp_hdf5_1 = ".h5";
        std::string s_cmp_hdf5_2 = ".hdf5";
        int total_samples = 0;
        if (is_equal(in_file, s_cmp)) {
            char c[in_file.size() + 1];
            strcpy(c, in_file.c_str());
            auto *data = new deep_io(c, n_nodes, node_index);
            int read_bytes = data->load_next_samples(v_points);
            if (read_bytes == -1) {
                std::cerr << "Critical Error: Failed to read input" << std::endl;
                exit(-1);
            }
            n = data->sample_read_no;
            max_d = data->feature_no;
            return data->sample_no;
        } else if (is_equal(in_file, s_cmp_hdf5_1) || is_equal(in_file, s_cmp_hdf5_2)) {
            n = read_input_hdf5(in_file, v_points, max_d, n_nodes, node_index);
            total_samples = n;
        } else {
            deep_io::count_lines_and_dimensions(in_file, n, max_d);
            v_points.resize(n * max_d);
            std::cout << "WARNING: USING SLOW CSV I/O." << std::endl;
            read_input_csv(in_file, v_points, max_d);
            total_samples = n;
        }
        return total_samples;
    }


    inline bool dist_leq(const float *coord1, const float *coord2, const int max_d, const float e2) noexcept {
        float tmp = 0;
        #pragma omp simd
        for (int d = 0; d < max_d; d++) {
            float tmp2 = coord1[d] - coord2[d];
            tmp += tmp2 * tmp2;
        }
        return tmp <= e2;
    }

    result start(const uint m, const float e, const uint n_threads, const std::string &in_file,
            const uint node_index, const uint n_nodes) noexcept {


//        #pragma omp parallel
//        {
//            int thread_num = omp_get_thread_num();
//            int cpu_num = sched_getcpu();
//            printf("Thread %3d is running on CPU %3d\n", thread_num, cpu_num);
//        }

        auto time_start = std::chrono::high_resolution_clock::now();
        omp_set_dynamic(0);
        omp_set_num_threads(n_threads);
        uint n, n_dim, total_samples;
        s_vec<float> v_coords;
        if (node_index == 0) {
            std::cout << "Total of " << (n_threads * n_nodes) << " cores used on " << n_nodes << " node(s)." << std::endl;
        }
        measure_duration("Input Read: ", node_index == 0, [&]() -> void {
            total_samples = read_input(in_file, v_coords, n, n_dim, n_nodes, node_index);
        });
        auto time_data_read = std::chrono::high_resolution_clock::now();
        if (node_index == 0) {
            std::cout << "Found " << n << " points in " << n_dim << " dimensions" << " and read " << n <<
                      " of " << total_samples << " samples." << std::endl;
        }
        /*
        std::vector<uint> v_sample(n);
        std::iota(v_sample.begin(), v_sample.end(), 0);
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::shuffle(v_sample.begin(), v_sample.end(), std::default_random_engine(seed));

        std::vector<std::vector<uint>> vv_sorted_dim(n_dim);
        measure_duration("Sort Dimensions: ", node_index == 0, [&]() -> void {
            for (uint d = 0; d < n_dim; ++d) {
                vv_sorted_dim[d].resize(n);
                std::iota(vv_sorted_dim[d].begin(), vv_sorted_dim[d].end(), 0);
                std::sort(vv_sorted_dim[d].begin(), vv_sorted_dim[d].end(), [&v_coords, &d, &n_dim]
                (const auto &i1, const auto &i2) -> bool {
                    // TODO optimize the n_dim mult (not necessary)
                    return v_coords[i1*n_dim+d] < v_coords[i2*n_dim+d];
                });
            }
        });

        std::vector<std::vector<uint>> vv_sorted_index(n_dim);
        measure_duration("Build index table: ", node_index == 0, [&]() -> void {
            for (uint d = 0; d < n_dim; ++d) {
                vv_sorted_index[d].resize(vv_sorted_dim[d].size());
                for (uint i = 0; i < vv_sorted_dim[d].size(); ++i) {
                    vv_sorted_index[d][vv_sorted_dim[d][i]] = i;
                }
            }
        });


        std::vector<uint> v_lower_bound(n*n_dim);
        std::vector<uint> v_higher_bound(n*n_dim);
        measure_duration("Build bounds table: ", node_index == 0, [&]() -> void {
            #pragma omp parallel for
            for (uint i = 0; i < v_sample.size(); ++i) {
                uint bound_index = i*n_dim;
                for (uint d = 0; d < n_dim; ++d) {
                    float lower = v_coords[v_sample[i]*n_dim+d] - e;
                    float higher = v_coords[v_sample[i]*n_dim+d] + e;
                    int sorted_index = vv_sorted_index[d][v_sample[i]];
                    for (int j = sorted_index+1; j < n; ++j) {
                        if (v_coords[vv_sorted_index[d][j]*n_dim+d] > higher) {
                            v_lower_bound[bound_index+d] = j;
                            j = n;
                        }
                    }
                    for (int j = sorted_index-1; j >= 0; --j) {
                        if (v_coords[vv_sorted_index[d][j]*n_dim+d] < lower) {
                            v_higher_bound[bound_index+d] = j;
                            j = -1;
                        }
                    }
                }
            }
        });


        std::vector<uint> v_cnt(n, 1);
        measure_duration("Count neighbours: ", node_index == 0, [&]() -> void {
            float e2 = e * e;
            #pragma omp parallel for
            for (uint i = 0; i < v_sample.size(); ++i) {
                for (uint j = i+1; j < v_sample.size(); ++j) {
                    bool is_conn = true;
                    for (uint d = 0; d < n_dim; ++d) {
//                        uint lower = ;
                        if (vv_sorted_index[d][v_sample[j]] < v_lower_bound[i*n_dim+d]
                            || vv_sorted_index[d][v_sample[j]] > v_higher_bound[i*n_dim+d] ) {
                            is_conn = false;
                            d = n_dim;
                        }
                    }
                    if (is_conn && dist_leq(&v_coords[v_sample[i]*n_dim], &v_coords[v_sample[j]*n_dim], n_dim, e2)) {
                        #pragma omp atomic
                        ++v_cnt[v_sample[i]];
                        #pragma omp atomic
                        ++v_cnt[v_sample[j]];
                    }
                }

            }
        });

        uint cores = 0;
        for (uint i = 0; i < v_cnt.size(); ++i) {
            if (v_cnt[i] >= m) {
                ++cores;
            }
        }
        std::cout << "cores: " << cores << std::endl;
         */

        nc_tree nc(&v_coords[0], n_dim, n, e, m, n_threads);
        nc.init();

        measure_duration("MPI partition " + std::to_string(node_index) + ": ", true/*node_index == 0*/, [&]() -> void {
            // TODO handle prime number of partitions
            nc.partition_data(n_threads);
        });


/*node_index == 0*/
/*
        measure_duration("Build tree: " + std::to_string(node_index), true, [&]() -> void {
            nc.build_tree();
        });
//        next_util::print_tree_meta_data(nc);
        measure_duration("Collect Proximity Queries: ", node_index == 0, [&]() -> void {
            nc.collect_proximity_queries();
        });
        std::cout << "Number of edges: " << nc.get_no_of_edges() << std::endl;
        measure_duration("Process Proximity Queries: ", node_index == 0, [&]() -> void {
            nc.process_proximity_queries();
        });
        measure_duration("Infer types: ", node_index == 0, [&]() -> void {
            nc.infer_types();
        });

        measure_duration("Determine Labels: ", node_index == 0, [&]() -> void {
            nc.determine_cell_labels();
        });
        auto time_end = std::chrono::high_resolution_clock::now();
        if (!g_quiet && node_index == 0) {
            std::cout << "Total Execution Time: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count()
                      << " milliseconds\n";
            std::cout << "Total Execution Time (without I/O): "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_data_read).count()
                      << " milliseconds\n";
        }
        */
        return collect_results(nc);
    }

}