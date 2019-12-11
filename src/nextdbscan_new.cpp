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

    /*
    inline bool dist_leq(const float *coord1, const float *coord2, const int max_d, const float e2) noexcept {
        float tmp = 0;
        #pragma unroll
        for (int d = 0; d < max_d; d++) {
            float tmp2 = coord1[d] - coord2[d];
            tmp += tmp2 * tmp2;
        }
        return tmp <= e2;
    }

    inline bool is_in_reach(const float *min1, const float *max1, const float *min2, const float *max2,
            const uint max_d, const float e) noexcept {
        for (uint d = 0; d < max_d; ++d) {
            if ((min2[d] > (max1[d] + e) || min2[d] < (min1[d] - e)) &&
                (min1[d] > (max2[d] + e) || min1[d] < (min2[d] - e)) &&
                (max2[d] > (max1[d] + e) || max2[d] < (min1[d] - e)) &&
                (max1[d] > (max2[d] + e) || max1[d] < (min2[d] - e))) {
                return false;
            }
        }
        return true;
    }

    inline void update_to_ac(s_vec<uint> &v_index_maps, s_vec<uint> &v_cell_ns,
            s_vec<uint> &v_cell_begin, std::vector<uint8_t> &is_core, std::vector<uint8_t> &v_types,
            const uint c) noexcept {
        v_types[c] = AC;
        uint begin = v_cell_begin[c];
        for (uint j = 0; j < v_cell_ns[c]; ++j) {
            is_core[v_index_maps[begin + j]] = 1;
        }
    }

    void update_type(s_vec<uint> &v_index_maps, s_vec<uint> &v_cell_ns,
            s_vec<uint> &v_cell_begin, std::vector<uint> &v_cell_nps, std::vector<uint> &v_point_nps,
            std::vector<uint8_t> &is_core, std::vector<uint8_t> &v_types, const uint c, const uint m) noexcept {
        if (v_types[c] == AC) {
            return;
        }
        if (v_cell_nps[c] >= m) {
            update_to_ac(v_index_maps, v_cell_ns, v_cell_begin, is_core, v_types, c);
        }
        bool all_cores = true;
        bool some_cores = false;
        uint begin = v_cell_begin[c];
        for (uint j = 0; j < v_cell_ns[c]; ++j) {
            uint p = v_index_maps[begin + j];
            if (is_core[p])
                continue;
            if (v_cell_nps[c] + v_point_nps[p] >= m) {
                is_core[p] = 1;
                some_cores = true;
            } else {
                all_cores = false;
            }
        }
        if (all_cores) {
            v_types[c] = AC;
        } else if (some_cores) {
            v_types[c] = SC;
        }
    }

    void update_points(s_vec<uint> &v_index_map_level, std::vector<uint> &v_cell_nps,
            std::vector<uint> &v_point_nps, uint *v_range_cnt, const uint size, const uint begin,
            const uint c) noexcept {
        uint min_change = INT32_MAX;
        for (uint k = 0; k < size; ++k) {
            if (v_range_cnt[k] < min_change)
                min_change = v_range_cnt[k];
        }
        if (min_change > 0) {
            #pragma omp atomic
            v_cell_nps[c] += min_change;
        }
        for (uint k = 0; k < size; ++k) {
            if (min_change > 0)
                v_range_cnt[k] -= min_change;
            if (v_range_cnt[k] > 0) {
                uint p = v_index_map_level[begin + k];
                #pragma omp atomic
                v_point_nps[p] += v_range_cnt[k];
            }
        }
    }
     */

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

    result start(const uint m, const float e, const uint n_threads, const std::string &in_file,
            const uint node_index, const uint n_nodes) noexcept {
        auto time_start = std::chrono::high_resolution_clock::now();
        omp_set_dynamic(0);
        omp_set_num_threads(n_threads);
        uint n, max_d, total_samples;
        s_vec<float> v_coords;
        if (node_index == 0) {
            std::cout << "Total of " << (n_threads * n_nodes) << " cores used on " << n_nodes << " nodes." << std::endl;
        }
        measure_duration("Input Read: ", node_index == 0, [&]() -> void {
            total_samples = read_input(in_file, v_coords, n, max_d, n_nodes, node_index);
        });
        auto time_data_read = std::chrono::high_resolution_clock::now();
        if (node_index == 0) {
            std::cout << "Found " << n << " points in " << max_d << " dimensions" << " and read " << n <<
                      " of " << total_samples << " samples." << std::endl;
        }

        nc_tree nc(&v_coords[0], max_d, n, e, m, n_threads);
        nc.init();
        measure_duration("Build tree: ", node_index == 0, [&]() -> void {
            nc.build_tree();
        });
        next_util::print_tree_meta_data(nc);
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
//        std::cout << "All Cores cell cnt: " << nc.cnt_leaf_cells_of_type(ALL_CORES) << std::endl;
//        std::cout << "No Cores cell cnt: " << nc.cnt_leaf_cells_of_type(NO_CORES) << std::endl;

        auto time_end = std::chrono::high_resolution_clock::now();
        if (!g_quiet && node_index == 0) {
            std::cout << "Total Execution Time: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count()
                      << " milliseconds\n";
            std::cout << "Total Execution Time (without I/O): "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_data_read).count()
                      << " milliseconds\n";
        }
        return collect_results(nc);
    }

}