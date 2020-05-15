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
#include "nc_tree_new.h"
#include "deep_io.h"
#include "next_util.h"
// TODO CUDA
#include "next_data_omp.h"
#include "cell_processor.h"


namespace nextdbscan {

    static bool g_quiet = false;

    void measure_duration(std::string const &name, bool const is_out, std::function<void()> const &callback) noexcept {
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

    float get_lowest_e(float const e, long const n_dim) {
        // TODO find a less wasteful formula to maintain precision
        if (n_dim <= 3) {
            return e / sqrtf(3);
        } else if (n_dim <= 8) {
            return e / sqrtf(3.5);
        } else if (n_dim <= 30) {
            return e / sqrtf(4);
        } else if (n_dim <= 80) {
            return e / sqrtf(5);
        } else {
            return e / sqrtf(6);
        }
    }

    result collect_results(cell_processor &cp, long const n_coords) noexcept {
        result res{0, 0, 0, n_coords, new long[n_coords]};
        cp.get_result_meta(res.core_count, res.noise, res.clusters);
        return res;
    }

    void read_input_csv(const std::string &in_file, s_vec<float> &v_points, long const max_d) noexcept {
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
                v_points[index++] = static_cast<float>(atof(buf.c_str()));
            }
        }
        is.close();
    }

    uint read_input_hdf5(const std::string &in_file, s_vec<float> &v_points, unsigned long &max_d,
            unsigned long const n_nodes, unsigned long const node_index) noexcept {
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

    unsigned long read_input(const std::string &in_file, s_vec<float> &v_points, unsigned long &n, unsigned long &max_d,
            unsigned long const n_nodes, unsigned long const node_index) noexcept {
        std::string s_cmp = ".bin";
        std::string s_cmp_hdf5_1 = ".h5";
        std::string s_cmp_hdf5_2 = ".hdf5";
        unsigned long total_samples = 0;
        if (is_equal(in_file, s_cmp)) {
            char c[in_file.size() + 1];
            strcpy(c, in_file.c_str());
            auto *data = new deep_io(c, n_nodes, node_index);
            int read_bytes = data->load_next_samples(v_points);
            if (read_bytes == -1) {
                std::cerr << "Critical Error: Failed to read input" << std::endl;
                exit(-1);
            }
            n = static_cast<unsigned long>(data->sample_read_no);
            max_d = static_cast<unsigned long>(data->feature_no);
            return static_cast<unsigned long>(data->sample_no);
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

    result start(unsigned long const m, const float e, unsigned long const n_threads, const std::string &in_file,
            unsigned long const node_index, unsigned long const n_nodes) noexcept {


//        #pragma omp parallel
//        {
//            int thread_num = omp_get_thread_num();
//            int cpu_num = sched_getcpu();
//            printf("Thread %3d is running on CPU %3d\n", thread_num, cpu_num);
//        }

        auto time_start = std::chrono::high_resolution_clock::now();
        omp_set_dynamic(0);
        omp_set_num_threads((int) n_threads);
        unsigned long n, n_dim, total_samples;
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
        s_vec<float> v_min_bounds(n_dim);
        s_vec<float> v_max_bounds(n_dim);
        auto e_lowest = get_lowest_e(e, n_dim);
        auto n_level = next_data::determine_data_boundaries(&v_min_bounds[0], &v_max_bounds[0], &v_coords[0],
                n_dim, n, e_lowest);
        std::cout << "Max Level: " << n_level << std::endl;

        s_vec<unsigned long> v_part_coord;
        s_vec<unsigned long> v_part_offset;
        s_vec<unsigned long> v_part_size;

        cell_processor cp(n_threads);
        auto nc = nc_tree_new(v_coords, v_min_bounds, v_max_bounds, e, e_lowest, n_dim, n_level, n, m);
        if (n_threads > 1) {
            measure_duration("Partition Data: ", node_index == 0, [&]() -> void {
                nc.partition_data(n_threads, n_threads);
            });
        }

        measure_duration("Build NC Tree: ", node_index == 0, [&]() -> void {
            if (n_threads > 1) {
                nc.build_tree_parallel(n_threads);
            } else {
                nc.build_tree();
            }
        });

        s_vec<long> v_edges;
        measure_duration("Collect Edges: ", node_index == 0, [&]() -> void {
            if (n_threads > 1) {
                nc.collect_edges_parallel(v_edges, n_threads);
            } else {
                nc.collect_edges(v_edges);
            }
        });

        std::cout << "Edges size: " << v_edges.size()/2 << std::endl;

        measure_duration("Process Edges: ", node_index == 0, [&]() -> void {
            cp.process_edges(v_coords, v_edges, nc);
        });
        measure_duration("Infer Types: ", node_index == 0, [&]() -> void {
            cp.infer_types(nc);
        });

        measure_duration("Determine Labels: ", node_index == 0, [&]() -> void {
            cp.determine_cell_labels(v_coords, v_edges, nc);
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

        return collect_results(cp, total_samples);
    }

}