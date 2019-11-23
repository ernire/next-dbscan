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
#include <numeric>
#include <functional>
//#define MPI_ON
//#define CUDA_ON
//#define HDF5_ON
#ifdef MPI_ON
#include <mpi.h>
#endif
#ifdef HDF5_ON
#include <hdf5.h>
#endif
#include "nextdbscan.h"
#ifdef CUDA_ON
#include "nextdbscan_cuda.h"
#endif
#ifndef CUDA_ON
#include "nextdbscan_omp.h"
#endif
#include "deep_io.h"

namespace nextdbscan {

    static bool g_quiet = false;

    struct cell_meta {
        uint l, c;

        cell_meta(uint l, uint c) : l(l), c(c) {}
    };

    struct cell_meta_2 {
        uint c1, c2;

        cell_meta_2(uint c1, uint c2) : c1(c1), c2(c2) {}
    };

    struct cell_meta_3 {
        uint l, c1, c2;

        cell_meta_3(uint l, uint c1, uint c2) : l(l), c1(c1), c2(c2) {}
    };

    struct cell_meta_5 {
        uint l, c1, c2, n1, n2;

        cell_meta_5(uint l, uint c1, uint c2, uint n1, uint n2) : l(l), c1(c1), c2(c2), n1(n1), n2(n2) {}
    };

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

    void calc_bounds(s_vec<float> &v_coords, uint n, float *min_bounds,
            float *max_bounds, const uint max_d) noexcept {
        for (uint d = 0; d < max_d; d++) {
            min_bounds[d] = INT32_MAX;
            max_bounds[d] = INT32_MIN;
        }
        #pragma omp parallel for reduction(max:max_bounds[:max_d]) reduction(min:min_bounds[:max_d])
        for (uint i = 0; i < n; i++) {
            size_t index = i * max_d;
            for (uint d = 0; d < max_d; d++) {
                if (v_coords[index + d] > max_bounds[d]) {
                    max_bounds[d] = v_coords[index + d];
                }
                if (v_coords[index + d] < min_bounds[d]) {
                    min_bounds[d] = v_coords[index + d];
                }
            }
        }
    }

    void calc_dims_mult(ull *dims_mult, const uint max_d, s_vec<float> &min_bounds,
            s_vec<float> &max_bounds, const float e_inner) noexcept {
        std::vector<uint> dims(max_d);
        dims_mult[0] = 1;
        for (uint d = 0; d < max_d; d++) {
            dims[d] = ((max_bounds[d] - min_bounds[d]) / e_inner) + 1;
            if (d > 0) {
                dims_mult[d] = dims_mult[d - 1] * dims[d - 1];
                if (dims_mult[d] < dims_mult[d-1]) {
                    std::cerr << "Error: Index Overflow Detected" << std::endl;
                    std::cout << "Number of possible cells exceeds 2^64 (not yet supported). "
                        << "Try using a larger epsilon value." << std::endl;
                    exit(-1);
                }
            }
        }
    }

    inline bool dist_leq(const float *coord1, const float *coord2, const int max_d, const float e2) noexcept {
        float tmp = 0;
        #pragma unroll
        for (int d = 0; d < max_d; d++) {
            float tmp2 = coord1[d] - coord2[d];
            tmp += tmp2 * tmp2;
        }
        return tmp <= e2;
    }

    void vector_min(std::vector<int> &omp_in, std::vector<int> &omp_out) noexcept {
        for (int i = 0; i < omp_out.size(); ++i) {
            omp_out[i] = std::min(omp_in[i], omp_out[i]);
        }
    }

#pragma omp declare reduction(vec_min: std::vector<int>: vector_min(omp_in, omp_out)) initializer(omp_priv=omp_orig)
#pragma omp declare reduction(vec_merge_int: std::vector<int>: omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end())) initializer(omp_priv=omp_orig)
#pragma omp declare reduction(vec_merge_cell_3: std::vector<cell_meta_3>: omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end())) initializer(omp_priv=omp_orig)

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

    uint fill_range_table(s_vec<float> &v_coords, s_vec<uint> &v_index_map_level,
            const uint size1, const uint size2, std::vector<bool> &v_range_table,
            const uint begin1, const uint begin2, const uint max_d, const float e2) noexcept {
        uint hits = 0;
        uint index = 0;
        uint total_size = size1 * size2;
        std::fill(v_range_table.begin(), v_range_table.begin() + total_size, false);
        for (uint k1 = 0; k1 < size1; ++k1) {
            uint p1 = v_index_map_level[begin1 + k1];
            for (uint k2 = 0; k2 < size2; ++k2, ++index) {
                uint p2 = v_index_map_level[begin2 + k2];
                if (dist_leq(&v_coords[p1 * max_d], &v_coords[p2 * max_d], max_d, e2)) {
                    v_range_table[index] = true;
                    ++hits;
                }
            }
        }
        return hits;
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

    void update_cell_pair_nn(s_vec<uint> &v_index_map_level, const uint size1, const uint size2,
            std::vector<uint> &v_cell_nps, std::vector<uint> &v_point_nps, std::vector<bool> &v_range_table,
            std::vector<uint> &v_range_count,
            const uint c1, const uint begin1, const uint c2, const uint begin2,
            const bool is_update1, const bool is_update2) noexcept {
        std::fill(v_range_count.begin(), std::next(v_range_count.begin() + (size1 + size2)), 0);
        uint index = 0;
        for (uint k1 = 0; k1 < size1; ++k1) {
            for (uint k2 = 0; k2 < size2; ++k2, ++index) {
                if (v_range_table[index]) {
                    if (is_update1)
                        ++v_range_count[k1];
                    if (is_update2)
                        ++v_range_count[size1 + k2];
                }
            }
        }
        if (is_update1) {
            update_points(v_index_map_level, v_cell_nps, v_point_nps, &v_range_count[0], size1, begin1, c1);
        }
        if (is_update2) {
            update_points(v_index_map_level, v_cell_nps, v_point_nps, &v_range_count[size1], size2, begin2, c2);
        }
    }

    void process_pair_proximity(s_vec<float> &v_coords,
            s_vec<uint> &v_index_maps,
            std::vector<uint> &v_point_nps,
            s_vec<uint> &v_cell_ns,
            std::vector<bool> &v_range_table,
            std::vector<uint> &v_range_cnt,
            std::vector<uint> &v_cell_nps,
            const uint max_d, const float e2, const uint m,
            const uint c1, const uint begin1, const uint c2, const uint begin2) noexcept {

        uint size1 = v_cell_ns[c1];
        uint size2 = v_cell_ns[c2];
        uint hits = fill_range_table(v_coords, v_index_maps, size1, size2,
                v_range_table, begin1, begin2, max_d, e2);
        if (hits == 0) {
            return;
        }
        if (hits == size1*size2) {
            if (v_cell_nps[c1] < m) {
                #pragma omp atomic
                v_cell_nps[c1] += v_cell_ns[c2];
            }
            if (v_cell_nps[c2] < m) {
                #pragma omp atomic
                v_cell_nps[c2] += v_cell_ns[c1];
            }
        } else {
            update_cell_pair_nn(v_index_maps, size1, size2, v_cell_nps, v_point_nps, v_range_table,
                    v_range_cnt, c1, begin1, c2, begin2, v_cell_nps[c1] < m,
                    v_cell_nps[c2] < m);
        }
    }

    result collect_results(std::vector<uint8_t> &v_is_core,std::vector<int> &v_cluster_label, uint n) noexcept {
        result res{0, 0, 0, n, new int[n]};

        uint sum = 0;
        #pragma omp parallel for reduction(+:sum)
        for (uint i = 0; i < v_is_core.size(); ++i) {
            if (v_is_core[i])
                ++sum;
        }
        res.core_count = sum;
        sum = 0;
        #pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < v_cluster_label.size(); ++i) {
            if (v_cluster_label[i] == i)
                ++sum;
        }
        res.clusters = sum;

        uint &noise = res.noise;
        #pragma omp parallel for reduction(+: noise)
        for (int i = 0; i < n; i++) {
            if (v_cluster_label[i] == UNASSIGNED) {
                ++noise;
            }
        }
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

    int determine_data_boundaries(s_vec<float> &v_coords, s_vec<float> &v_min_bounds,
            s_vec<float> &v_max_bounds, const uint n, const uint max_d,
            const float e_inner) noexcept {
        float max_limit = INT32_MIN;
        calc_bounds(v_coords, n, &v_min_bounds[0], &v_max_bounds[0], max_d);
#ifdef MPI_ON
        auto v_global_min_bounds = std::make_unique<float[]>(max_d);
        auto v_global_max_bounds = std::make_unique<float[]>(max_d);
        MPI_Allreduce(&v_min_bounds[0], &v_global_min_bounds[0], max_d, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&v_max_bounds[0], &v_global_max_bounds[0], max_d, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
        std::copy(&v_global_min_bounds[0], &v_global_min_bounds[max_d], &v_min_bounds[0]);
        std::copy(&v_global_max_bounds[0], &v_global_max_bounds[max_d], &v_max_bounds[0]);
#endif
#pragma omp parallel for reduction(max: max_limit)
        for (uint d = 0; d < max_d; d++) {
            if (v_max_bounds[d] - v_min_bounds[d] > max_limit)
                max_limit = v_max_bounds[d] - v_min_bounds[d];
        }
        return static_cast<int>(ceilf(logf(max_limit / e_inner) / logf(2))) + 1;
    }


    inline int get_label(std::vector<int> &v_c_labels, uint p) noexcept {
        int label = v_c_labels[p];
        bool flatten = false;
        while (label != v_c_labels[label]) {
            label = v_c_labels[label];
            flatten = true;
        }
        if (flatten) {
            v_c_labels[p] = label;
        }
        return label;
    }

    void process_pair_labels(s_vec<float> &v_coords,
            std::vector<int> &v_c_labels,
            d_vec<uint> &vv_cell_ns,
            d_vec<uint> &vv_index_maps,
            std::vector<uint8_t> &v_cell_types,
            std::vector<uint8_t> &v_is_core,
            const uint c1, const uint c2, const uint l, const uint begin1, const uint begin2,
            const uint max_d, const float e2) noexcept {
        // Do both cells have cores ?
        if (v_cell_types[c1] != NC && v_cell_types[c2] != NC) {
            for (uint k1 = 0; k1 < vv_cell_ns[l][c1]; ++k1) {
                uint p1 = vv_index_maps[l][begin1 + k1];
                if (!v_is_core[p1]) {
                    continue;
                }
                int label1 = get_label(v_c_labels, p1);
                for (uint k2 = 0; k2 < vv_cell_ns[l][c2]; ++k2) {
                    uint p2 = vv_index_maps[l][begin2 + k2];
                    if (!v_is_core[p2]) {
                        continue;
                    }
                    int label2 = get_label(v_c_labels, p2);
                    if (label1 != label2) {
                        if (dist_leq(&v_coords[p1 * max_d],
                                &v_coords[p2 * max_d], max_d, e2)) {
                            if (label1 < label2)
                                v_c_labels[label2] = label1;
                            else
                                v_c_labels[label1] = label2;
                        }
                    }
                }
            }
        } else {
            // one NC one SC or AC
            for (uint k1 = 0; k1 < vv_cell_ns[l][c1]; ++k1) {
                uint p1 = vv_index_maps[l][begin1 + k1];
                if (!v_is_core[p1] && v_c_labels[p1] != UNASSIGNED)
                    continue;
                for (uint k2 = 0; k2 < vv_cell_ns[l][c2]; ++k2) {
                    uint p2 = vv_index_maps[l][begin2 + k2];
                    if (!v_is_core[p2] && v_c_labels[p2] != UNASSIGNED)
                        continue;
                    if (v_is_core[p1]) {
                        if (dist_leq(&v_coords[p1 * max_d], &v_coords[p2 * max_d], max_d, e2)) {
                            v_c_labels[p2] = v_c_labels[p1];
                        }
                    } else if (v_is_core[p2]) {
                        if (dist_leq(&v_coords[p1 * max_d], &v_coords[p2 * max_d], max_d, e2)) {
                            v_c_labels[p1] = v_c_labels[p2];
                            k2 = vv_cell_ns[l][c2];
                        }
                    }
                }
            }

        }
    }

    void collect_sub_tree_edges(d_vec<uint> &vv_index_map,
            d_vec<uint> &vv_cell_begin,
            d_vec<uint> &vv_cell_ns,
            d_vec<float> &vv_min_cell_dim,
            d_vec<float> &vv_max_cell_dim,
            s_vec<cell_meta_2> &v_cell_pairs,
            std::vector<cell_meta_3> &v_stack, const uint max_d, const float e) {
        while (!v_stack.empty()) {
            uint l = v_stack.back().l;
            uint c1 = v_stack.back().c1;
            uint c2 = v_stack.back().c2;
            v_stack.pop_back();
            uint begin1 = vv_cell_begin[l][c1];
            uint begin2 = vv_cell_begin[l][c2];
            if (l == 0) {
//                v_cell_pairs.emplace_back(c1, c2);
            } else {
                for (uint k1 = 0; k1 < vv_cell_ns[l][c1]; ++k1) {
                    uint c1_next = vv_index_map[l][begin1 + k1];
                    for (uint k2 = 0; k2 < vv_cell_ns[l][c2]; ++k2) {
                        uint c2_next = vv_index_map[l][begin2 + k2];
                        if (is_in_reach(&vv_min_cell_dim[l - 1][c1_next * max_d],
                                &vv_max_cell_dim[l - 1][c1_next * max_d],
                                &vv_min_cell_dim[l - 1][c2_next * max_d],
                                &vv_max_cell_dim[l - 1][c2_next * max_d], max_d, e)) {
                            v_stack.emplace_back(l - 1, c1_next, c2_next);
                        }
                    }
                }
            }
        }
    }

    void collect_task_edges(cell_meta &task,
            d_vec<uint> &vv_index_map,
            d_vec<uint> &vv_cell_begin,
            d_vec<uint> &vv_cell_ns,
            d_vec<float> &vv_min_cell_dim,
            d_vec<float> &vv_max_cell_dim,
            s_vec<cell_meta_2> &v_cell_pairs,
            std::vector<cell_meta_3> &v_stack, const uint max_d, const float e
            ) {


        uint l = task.l;
        uint c = task.c;
        uint begin = vv_cell_begin[l][c];
        for (uint c1 = 0; c1 < vv_cell_ns[l][c]; ++c1) {
            uint c1_index = vv_index_map[l][begin + c1];
            for (uint c2 = c1 + 1; c2 < vv_cell_ns[l][c]; ++c2) {
                uint c2_index = vv_index_map[l][begin + c2];
                if (is_in_reach(&vv_min_cell_dim[l - 1][c1_index * max_d],
                        &vv_max_cell_dim[l - 1][c1_index * max_d],
                        &vv_min_cell_dim[l - 1][c2_index * max_d],
                        &vv_max_cell_dim[l - 1][c2_index * max_d], max_d, e)) {

                }
            }
        }
            /*
            for (uint c2 = c1 + 1; c2 < vv_cell_ns[l][c]; ++c2) {
                uint c2_index = vv_index_map[l][begin + c2];
                if (is_in_reach(&vv_min_cell_dim[l - 1][c1_index * max_d],
                        &vv_max_cell_dim[l - 1][c1_index * max_d],
                        &vv_min_cell_dim[l - 1][c2_index * max_d],
                        &vv_max_cell_dim[l - 1][c2_index * max_d], max_d, e)) {
                    vv_stacks3[tid].emplace_back(l - 1, c1_index, c2_index);
                    */
    }

    void process_pair_stack(s_vec<float> &v_coords,
            d_vec<uint> &vv_index_map,
            d_vec<uint> &vv_cell_begin,
            d_vec<uint> &vv_cell_ns,
            d_vec<float> &vv_min_cell_dim,
            d_vec<float> &vv_max_cell_dim,
            std::vector<uint> &v_leaf_cell_np,
            std::vector<uint> &v_point_np,
            std::vector<cell_meta_3> &v_stack,
            std::vector<bool> &v_range_table,
            std::vector<uint> &v_range_counts,
            std::vector<uint8_t> &v_cell_types,
            std::vector<uint8_t> &v_is_core,
            std::vector<int> &v_c_labels,
            const uint m, const uint max_d, const float e, const float e2, const bool is_proximity_cnt) noexcept {
        while (!v_stack.empty()) {
            uint l = v_stack.back().l;
            uint c1 = v_stack.back().c1;
            uint c2 = v_stack.back().c2;
            v_stack.pop_back();
            uint begin1 = vv_cell_begin[l][c1];
            uint begin2 = vv_cell_begin[l][c2];
            if (l == 0) {
                if (is_proximity_cnt) {
                    if (v_leaf_cell_np[c1] < m || v_leaf_cell_np[c2] < m) {
                        process_pair_proximity(v_coords, vv_index_map[0], v_point_np,
                                vv_cell_ns[0], v_range_table, v_range_counts, v_leaf_cell_np,
                                max_d, e2, m, c1, begin1, c2, begin2);
                    }
                } else {
                    if (v_cell_types[c1] != NC || v_cell_types[c2] != NC) {
                        process_pair_labels(v_coords,v_c_labels, vv_cell_ns,
                                vv_index_map, v_cell_types, v_is_core, c1, c2, l,
                                begin1, begin2, max_d, e2);
                    }
                }
            } else {
                for (uint k1 = 0; k1 < vv_cell_ns[l][c1]; ++k1) {
                    uint c1_next = vv_index_map[l][begin1 + k1];
                    for (uint k2 = 0; k2 < vv_cell_ns[l][c2]; ++k2) {
                        uint c2_next = vv_index_map[l][begin2 + k2];
                        if (is_in_reach(&vv_min_cell_dim[l - 1][c1_next * max_d],
                                &vv_max_cell_dim[l - 1][c1_next * max_d],
                                &vv_min_cell_dim[l - 1][c2_next * max_d],
                                &vv_max_cell_dim[l - 1][c2_next * max_d], max_d, e)) {
                            v_stack.emplace_back(l - 1, c1_next, c2_next);
                        }
                    }
                }
            }
        }
    }

    uint infer_local_types_and_init_clusters(s_vec<uint> &v_index_map,
            s_vec<uint> &v_cell_begin,
            s_vec<uint> &v_cell_ns,
            std::vector<uint> &v_leaf_cell_np,
            std::vector<uint> &v_point_np,
            std::vector<uint8_t> &v_cell_types,
            std::vector<uint8_t> &v_is_core,
            std::vector<int> &v_c_labels,
            const uint m) noexcept {
        uint max_clusters = 0;
        #pragma omp parallel for reduction(+: max_clusters)
        for (uint i = 0; i < v_cell_ns.size(); ++i) {
            update_type(v_index_map, v_cell_ns, v_cell_begin,
                    v_leaf_cell_np, v_point_np, v_is_core, v_cell_types, i, m);
            if (v_cell_types[i] != NC) {
                ++max_clusters;
                uint begin = v_cell_begin[i];
                int core_p = UNASSIGNED;
                for (uint j = 0; j < v_cell_ns[i]; ++j) {
                    uint p = v_index_map[begin + j];
                    if (core_p != UNASSIGNED) {
                        v_c_labels[p] = core_p;
                    } else if (v_is_core[p]) {
                        core_p = p;
                        v_c_labels[core_p] = core_p;
                        for (uint k = 0; k < j; ++k) {
                            p = v_index_map[begin + k];
                            v_c_labels[p] = core_p;
                        }
                    }
                }
            }
        }
        return max_clusters;
    }

    void init_stacks(d_vec<uint> &vv_cell_ns,
            std::vector<uint> &v_leaf_cell_np,
            std::vector<std::vector<cell_meta_3>> &vv_stacks3,
            std::vector<std::vector<bool>> &vv_range_table,
            std::vector<std::vector<uint>> &vv_range_counts,
            const uint max_d, const uint n_threads) noexcept {
        uint max_points_in_leaf_cell = 0;
        #pragma omp parallel for reduction(max: max_points_in_leaf_cell)
        for (uint i = 0; i < vv_cell_ns[0].size(); ++i) {
            v_leaf_cell_np[i] = vv_cell_ns[0][i];
            if (vv_cell_ns[0][i] > max_points_in_leaf_cell) {
                max_points_in_leaf_cell = vv_cell_ns[0][i];
            }
        }
        #pragma omp parallel for
        for (uint t = 0; t < n_threads; ++t) {
            vv_stacks3[t].reserve(vv_cell_ns[0].size() * (uint) std::max((int) logf(max_d), 1));
            vv_range_table[t].resize(max_points_in_leaf_cell * max_points_in_leaf_cell);
            vv_range_counts[t].resize(max_points_in_leaf_cell * 2);
        }
    }

    void index_points(s_vec<float> &v_coords,
            s_vec<float> &v_eps_levels,
            s_vec<ull> &v_dims_mult,
            s_vec<float> &v_min_bounds,
            d_vec<uint> &vv_index_map,
            d_vec<uint> &vv_cell_begin,
            d_vec<uint> &vv_cell_ns,
            d_vec<float> &vv_min_cell_dim,
            d_vec<float> &vv_max_cell_dim,
            const uint max_d, const uint n_threads,
            const uint max_levels, const uint n) noexcept {
        uint size = n;
#ifdef CUDA_ON
        nextdbscan_cuda::index_points(v_coords, v_eps_levels, v_dims_mult, v_min_bounds, vv_index_map, vv_cell_begin,
                vv_cell_ns, vv_min_cell_dim, vv_max_cell_dim, max_d, n_threads, max_levels, size);
#endif
#ifndef CUDA_ON
        nextdbscan_omp::index_points(v_coords, v_eps_levels, v_dims_mult, v_min_bounds, vv_index_map,
                vv_cell_begin,vv_cell_ns, vv_min_cell_dim, vv_max_cell_dim, max_d, n_threads, max_levels, size);
#endif
    }

    void populate_tasks(d_vec<uint> &vv_cell_begin,
            std::vector<cell_meta> &v_tasks,
            const uint max_level) noexcept {

        uint size = 0;
        for (uint l = 1; l < max_level; ++l) {
            size += vv_cell_begin[l].size();
        }
        v_tasks.reserve(size);
        for (uint l = 1; l < max_level; ++l) {
            for (uint i = 0; i < vv_cell_begin[l].size(); ++i) {
                v_tasks.emplace_back(l, i);
            }
        }
    }

    result start(const uint m, const float e, const uint n_threads, const std::string &in_file,
            const uint node_index, const uint n_nodes) noexcept {
        auto time_start = std::chrono::high_resolution_clock::now();
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
        const auto e_inner = (e / sqrtf(3));
        const float e2 = e*e;
        s_vec<float> v_min_bounds(max_d);
        s_vec<float> v_max_bounds(max_d);
        int max_level;
        measure_duration("Determine Data Boundaries: ", node_index == 0, [&]() -> void {
            max_level = determine_data_boundaries(v_coords, v_min_bounds, v_max_bounds, n,
                    max_d, e_inner);
        });
        s_vec<float> v_eps_levels(max_level);
        s_vec<ull> v_dims_mult(max_level * max_d);
        measure_duration("Initialize Index Space: ", node_index == 0, [&]() -> void {
            #pragma omp parallel for
            for (uint l = 0; l < max_level; l++) {
                v_eps_levels[l] = (e_inner * pow(2, l));
                calc_dims_mult(&v_dims_mult[l * max_d], max_d, v_min_bounds, v_max_bounds, v_eps_levels[l]);
            }
        });

        d_vec<uint> vv_index_map(max_level);
        d_vec<uint> vv_cell_begin(max_level);
        d_vec<uint> vv_cell_ns(max_level);
        d_vec<float> vv_min_cell_dim(max_level);
        d_vec<float> vv_max_cell_dim(max_level);
        measure_duration("Index and Bounds: ", node_index == 0, [&]() -> void {
            index_points(v_coords, v_eps_levels, v_dims_mult, v_min_bounds, vv_index_map,
                    vv_cell_begin, vv_cell_ns, vv_min_cell_dim,
                    vv_max_cell_dim, max_d, n_threads, max_level, n);
        });
        std::vector<std::vector<cell_meta_3>> vv_stacks3(n_threads);
        std::vector<std::vector<bool>> vv_range_table(n_threads);
        std::vector<std::vector<uint>> vv_range_counts(n_threads);
        std::vector<uint> v_leaf_cell_np(vv_cell_ns[0].size(), 0);
        std::vector<uint> v_point_np(n, 0);
        std::vector<uint8_t> v_cell_type(vv_cell_ns[0].size(), NC);
        std::vector<uint8_t> v_is_core(n, 0);

        std::vector<cell_meta> v_tasks;
        measure_duration("Stacks and Tasks: ", node_index == 0, [&]() -> void {
            init_stacks(vv_cell_ns, v_leaf_cell_np, vv_stacks3, vv_range_table, vv_range_counts,
                    max_d, n_threads);
            populate_tasks(vv_cell_begin, v_tasks, max_level);
        });
        std::vector<int> v_c_labels(n, UNASSIGNED);

        measure_duration("Local Tree Proximity: ", node_index == 0, [&]() -> void {
//            uint task_cnt = 0;
//            uint empty_task_cnt = 0;


            #pragma omp parallel
            {
                uint tid = omp_get_thread_num();
                #pragma omp for schedule(dynamic)
                for (uint i = 0; i < v_tasks.size(); ++i) {
                    uint l = v_tasks[i].l;
                    uint c = v_tasks[i].c;
                    uint begin = vv_cell_begin[l][c];
                    for (uint c1 = 0; c1 < vv_cell_ns[l][c]; ++c1) {
                        uint c1_index = vv_index_map[l][begin + c1];
                        for (uint c2 = c1 + 1; c2 < vv_cell_ns[l][c]; ++c2) {
                            uint c2_index = vv_index_map[l][begin + c2];
                            if (is_in_reach(&vv_min_cell_dim[l - 1][c1_index * max_d],
                                    &vv_max_cell_dim[l - 1][c1_index * max_d],
                                    &vv_min_cell_dim[l - 1][c2_index * max_d],
                                    &vv_max_cell_dim[l - 1][c2_index * max_d], max_d, e)) {
                                vv_stacks3[tid].emplace_back(l - 1, c1_index, c2_index);
                                process_pair_stack(v_coords, vv_index_map, vv_cell_begin,
                                        vv_cell_ns, vv_min_cell_dim, vv_max_cell_dim,
                                        v_leaf_cell_np, v_point_np, vv_stacks3[tid], vv_range_table[tid],
                                        vv_range_counts[tid], v_cell_type, v_is_core, v_c_labels,
                                        m, max_d, e, e2, true);
                            }
                        }
                    }
                }
            }
        });

        uint max_local_clusters;
        measure_duration("Infer Cores and Init Local Clusters: ", node_index == 0, [&]() -> void {
            max_local_clusters = infer_local_types_and_init_clusters(vv_index_map[0], vv_cell_begin[0],
                    vv_cell_ns[0], v_leaf_cell_np, v_point_np, v_cell_type, v_is_core, v_c_labels,
                    m);
        });
        if (node_index == 0)
            std::cout << "Maximum number of local clusters: " << max_local_clusters << std::endl;

        measure_duration("Local Tree Labels: ", node_index == 0, [&]() -> void {
            #pragma omp parallel
            {
                uint tid = omp_get_thread_num();
                #pragma omp for schedule(dynamic)
                for (uint i = 0; i < v_tasks.size(); ++i) {
                    uint l = v_tasks[i].l;
                    uint c = v_tasks[i].c;
                    uint begin = vv_cell_begin[l][c];
                    for (uint c1 = 0; c1 < vv_cell_ns[l][c]; ++c1) {
                        uint c1_index = vv_index_map[l][begin + c1];
                        for (uint c2 = c1 + 1; c2 < vv_cell_ns[l][c]; ++c2) {
                            uint c2_index = vv_index_map[l][begin + c2];
                            if (is_in_reach(&vv_min_cell_dim[l - 1][c1_index * max_d],
                                    &vv_max_cell_dim[l - 1][c1_index * max_d],
                                    &vv_min_cell_dim[l - 1][c2_index * max_d],
                                    &vv_max_cell_dim[l - 1][c2_index * max_d], max_d, e)) {
                                vv_stacks3[tid].emplace_back(l - 1, c1_index, c2_index);
                                process_pair_stack(v_coords, vv_index_map, vv_cell_begin,
                                        vv_cell_ns, vv_min_cell_dim, vv_max_cell_dim,
                                        v_leaf_cell_np, v_point_np, vv_stacks3[tid], vv_range_table[tid],
                                        vv_range_counts[tid], v_cell_type, v_is_core, v_c_labels,
                                        m, max_d, e, e2, false);
                            }
                        }
                    }
                }
            }
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
        return collect_results(v_is_core, v_c_labels, total_samples);
    }

}