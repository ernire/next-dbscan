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
#include <fstream>
#include <cstdint>
#include <iterator>
#include <omp.h>
#include <numeric>
#include <functional>
//#define MPI_ON
//#define CUDA_ON
#ifdef MPI_ON
#include <mpi.h>
#endif
#ifdef CUDA_ON
#include <thrust/device_vector.h>
#endif
#include "nextdbscan.h"
#include "deep_io.h"

namespace nextdbscan {

    static const int UNASSIGNED = -1;

    static const uint8_t NC = 0;
    static const uint8_t AC = 1;
    static const uint8_t SC = 2;

    typedef unsigned long long ull;




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
        callback();
        auto end_timestamp = std::chrono::high_resolution_clock::now();
        if (!g_quiet && is_out) {
            std::cout << name
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

    void calc_dims_mult(ull *dims_mult, const uint max_d, const std::unique_ptr<float[]> &min_bounds,
            const std::unique_ptr<float[]> &max_bounds, const float e_inner) noexcept {
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
        float tmp = 0, tmp2;

        for (int d = 0; d < max_d; d++) {
            tmp2 = coord1[d] - coord2[d];
            tmp += tmp2 * tmp2;
        }
        return tmp <= e2;
    }

    inline ull get_cell_index(const float *dv, const std::unique_ptr<float[]> &mv, const ull *dm, const uint max_d,
            const float size) noexcept {
        ull cell_index = 0;
        for (uint d = 0; d < max_d; d++) {
            cell_index += (ull)((dv[d] - mv[d]) / size) * dm[d];
        }
        return cell_index;
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

    void read_input_txt(const std::string &in_file, s_vec<float> &v_points, int max_d) noexcept {
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

    void count_lines_and_dimensions(const std::string &in_file, uint &lines, uint &dimensions) noexcept {
        std::ifstream is(in_file);
        std::string line, buf;
        int cnt = 0;
        dimensions = 0;
        while (std::getline(is, line)) {
            if (dimensions == 0) {
                std::istringstream iss(line);
                std::vector<std::string> results(std::istream_iterator<std::string>{iss},
                        std::istream_iterator<std::string>());
                dimensions = results.size();
            }
            ++cnt;
        }
        lines = cnt;
        is.close();
    }

    uint load_input(const std::string &in_file, s_vec<float> &v_points, uint &n, uint &max_d,
            const uint blocks_no, const uint block_index) noexcept {
        std::string s_cmp = ".bin";
        int total_samples = 0;
        if (in_file.compare(in_file.size() - s_cmp.size(), s_cmp.size(), s_cmp) == 0) {
            char c[in_file.size() + 1];
            strcpy(c, in_file.c_str());
            auto *data = new deep_io(c, blocks_no, block_index);
            int read_bytes = data->load_next_samples(v_points);
            if (read_bytes == -1) {
                std::cerr << "Critical Error: Failed to read input" << std::endl;
                exit(-1);
            }
            n = data->sample_read_no;
            max_d = data->feature_no;
            return data->sample_no;
        } else {
            count_lines_and_dimensions(in_file, n, max_d);
            v_points.resize(n * max_d);
            std::cout << "WARNING: USING VERY SLOW NON-PARALLEL I/O." << std::endl;
            read_input_txt(in_file, v_points, max_d);
            total_samples = n;
        }
        return total_samples;
    }

    void calculate_level_cell_bounds(float *v_coords, s_vec<uint> &v_cell_begins,
            s_vec<uint> &v_cell_ns, s_vec<uint> &v_index_maps,
            std::vector<std::vector<float>> &vv_min_cell_dims,
            std::vector<std::vector<float>> &vv_max_cell_dims, uint max_d, uint l) noexcept {
        vv_min_cell_dims[l].resize(v_cell_begins.size() * max_d);
        vv_max_cell_dims[l].resize(vv_min_cell_dims[l].size());
        float *coord_min = nullptr, *coord_max = nullptr;

        #pragma omp parallel for private(coord_min, coord_max)
        for (uint i = 0; i < v_cell_begins.size(); i++) {
            uint begin = v_cell_begins[i];
            uint coord_offset = 0;
            if (l == 0) {
                coord_offset = v_index_maps[begin] * max_d;
                coord_min = &v_coords[coord_offset];
                coord_max = &v_coords[coord_offset];
            } else {
                coord_min = &vv_min_cell_dims[l - 1][v_index_maps[begin] * max_d];
                coord_max = &vv_max_cell_dims[l - 1][v_index_maps[begin] * max_d];
            }
            std::copy(coord_min, coord_min + max_d, &vv_min_cell_dims[l][i * max_d]);
            std::copy(coord_max, coord_max + max_d, &vv_max_cell_dims[l][i * max_d]);

            for (uint j = 1; j < v_cell_ns[i]; j++) {
                uint coord_offset_inner = 0;
                if (l == 0) {
                    coord_offset_inner = v_index_maps[begin + j] * max_d;
                    coord_min = &v_coords[coord_offset_inner];
                    coord_max = &v_coords[coord_offset_inner];
                } else {
                    coord_min = &vv_min_cell_dims[l - 1][v_index_maps[begin + j] * max_d];
                    coord_max = &vv_max_cell_dims[l - 1][v_index_maps[begin + j] * max_d];
                }
                for (uint d = 0; d < max_d; d++) {
                    if (coord_min[d] < vv_min_cell_dims[l][i * max_d + d]) {
                        vv_min_cell_dims[l][i * max_d + d] = coord_min[d];
                    }
                    if (coord_max[d] > vv_max_cell_dims[l][i * max_d + d]) {
                        vv_max_cell_dims[l][i * max_d + d] = coord_max[d];
                    }
                }
            }
        }
    }

    template<class T>
    void print_array(const std::string &name, T *arr, const uint max_d) noexcept {
        std::cout << name << ": ";
        for (int i = 0; i < max_d; ++i) {
            std::cout << arr[i] << " ";
        }
        std::cout << std::endl;
    }

    int determine_data_boundaries(s_vec<float> &v_coords, std::unique_ptr<float[]> &v_min_bounds,
            std::unique_ptr<float[]> &v_max_bounds, const uint n, const uint max_d,
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

#ifdef MPI_ON
    template<class T>
    void mpi_sum_vectors(std::vector<std::vector<T>> &vv_vector, std::vector<T> &v_payload,
            std::vector<T> &v_sink, std::vector<T> &v_additive, const int n_nodes,
            MPI_Datatype send_type, const bool is_additive) noexcept {
        int send_cnt = 0;
        int size[n_nodes];
        int offset[n_nodes];
        offset[0] = 0;
        for (int n = 0; n < n_nodes; ++n) {
            size[n] = vv_vector[n].size();
            send_cnt += size[n];
            if (n > 0) {
                offset[n] = offset[n - 1] + size[n - 1];
            }
        }
        if (v_payload.empty()) {
            v_payload.resize(send_cnt);
        }
        if (v_sink.empty()) {
            v_sink.resize(send_cnt);
        }
        int index = 0;
        for (int n = 0; n < n_nodes; ++n) {
            #pragma omp parallel for
            for (uint i = 0; i < vv_vector[n].size(); ++i) {
                v_payload[index+i] = is_additive ? vv_vector[n][i] - v_additive[index+i] : vv_vector[n][i];
            }
            index += vv_vector[n].size();
        }
        MPI_Allreduce(&v_payload[0], &v_sink[0], send_cnt, send_type, MPI_SUM, MPI_COMM_WORLD);
        index = 0;
        for (int n = 0; n < n_nodes; ++n) {
            #pragma omp parallel for
            for (uint i = 0; i < vv_vector[n].size(); ++i) {
                if (is_additive) {
                    vv_vector[n][i] = v_additive[i+index] + v_sink[i+index];
                } else {
                    vv_vector[n][i] = v_sink[i+index];
                }
            }
            index += vv_vector[n].size();
        }
    }

#ifdef MPI_ON
    template<class T>
    void mpi_gather_cell_tree(std::vector<std::vector<std::vector<T>>> &vvv_cell_tree,
            const int max_levels, const int n_nodes, const int node_index, std::vector<T> &v_buffer,
            MPI_Datatype send_type) noexcept {
        int size[n_nodes];
        int offset[n_nodes];
        for (int n = 0; n < n_nodes; ++n) {
            size[n] = 0;
            offset[n] = 0;
        }
        for (int n = 0; n < n_nodes; ++n) {
            for (int l = 0; l < max_levels; ++l) {
                size[n] += vvv_cell_tree[n][l].size();
            }
        }
        offset[0] = 0;
        for (int n = 1; n < n_nodes; ++n) {
            offset[n] = offset[n - 1] + size[n - 1];
        }
        int total_size = 0;
        for (int n = 0; n < n_nodes; ++n) {
            total_size += size[n];
        }
        v_buffer.resize(total_size, INT32_MAX);
//        print_array("Transmit size: ", size, n_nodes);
//        print_array("Transmit offset: ", offset, n_nodes);
        int index = 0;
        // TODO make smarter
        for (int n = 0; n < n_nodes; ++n) {
            for (int l = 0; l < max_levels; ++l) {
                for (auto &val : vvv_cell_tree[n][l]) {
                    assert(index < v_buffer.size());
                    if (n == node_index) {
                        v_buffer[index] = val;
                    }
                    ++index;
                }
            }
        }
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &v_buffer[0], size,
                offset, send_type, MPI_COMM_WORLD);
        index = 0;
        for (int n = 0; n < n_nodes; ++n) {
            for (int l = 0; l < max_levels; ++l) {
                // TODO skip node index
                for (int i = 0; i < vvv_cell_tree[n][l].size(); ++i) {
                    assert(index < v_buffer.size());
                    assert(v_buffer[index] != (T) INT32_MAX);
                    vvv_cell_tree[n][l][i] = v_buffer[index];
                    ++index;
                }
            }
        }
        assert(index == v_buffer.size());
    }

#endif

    void mpi_merge_cell_trees(std::vector<std::vector<std::vector<uint>>> &vvv_index_maps,
            std::vector<std::vector<std::vector<uint>>> &vvv_cell_begins,
            std::vector<std::vector<std::vector<uint>>> &vvv_cell_ns,
            std::vector<std::vector<std::vector<float>>> &vvv_min_cell_dims,
            std::vector<std::vector<std::vector<float>>> &vvv_max_cell_dims,
            const int node_index, const int n_nodes, const int max_levels,
            const uint max_d) noexcept {
        // count the number of elements and share it
        int total_levels = n_nodes * max_levels;
        auto n_node_level_elem = std::make_unique<int[]>(total_levels);
        std::fill(&n_node_level_elem[0], &n_node_level_elem[0] + total_levels, 0);
        uint index = node_index * max_levels;
        for (uint l = 0; l < max_levels; ++l, ++index) {
            n_node_level_elem[index] += vvv_index_maps[node_index][l].size();
        }
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &n_node_level_elem[0], max_levels,
                MPI_INT, MPI_COMM_WORLD);

        index = 0;
        for (uint n = 0; n < n_nodes; ++n) {
            for (uint l = 0; l < max_levels; ++l, ++index) {
                vvv_index_maps[n][l].resize(n_node_level_elem[index]);
                if (l > 0) {
                    vvv_cell_begins[n][l - 1].resize(n_node_level_elem[index]);
                    vvv_cell_ns[n][l - 1].resize(n_node_level_elem[index]);
                    vvv_min_cell_dims[n][l - 1].resize(n_node_level_elem[index] * max_d);
                    vvv_max_cell_dims[n][l - 1].resize(n_node_level_elem[index] * max_d);
                }
            }
            vvv_cell_begins[n][max_levels - 1].resize(1);
            vvv_cell_ns[n][max_levels - 1].resize(1);
            vvv_min_cell_dims[n][max_levels - 1].resize(max_d);
            vvv_max_cell_dims[n][max_levels - 1].resize(max_d);
        }

        std::vector<uint> v_uint_buffer;
        std::vector<float> v_float_buffer;
        mpi_gather_cell_tree(vvv_index_maps, max_levels, n_nodes, node_index, v_uint_buffer,
                MPI_UNSIGNED);
        mpi_gather_cell_tree(vvv_cell_begins, max_levels, n_nodes, node_index, v_uint_buffer,
                MPI_UNSIGNED);
        mpi_gather_cell_tree(vvv_cell_ns, max_levels, n_nodes, node_index, v_uint_buffer,
                MPI_UNSIGNED);
        mpi_gather_cell_tree(vvv_min_cell_dims, max_levels, n_nodes, node_index, v_float_buffer,
                MPI_FLOAT);
        mpi_gather_cell_tree(vvv_max_cell_dims, max_levels, n_nodes, node_index, v_float_buffer,
                MPI_FLOAT);
    }

#endif

    void sort_indexes_omp(std::unique_ptr<uint[]> &v_omp_sizes, std::unique_ptr<uint[]> &v_omp_offsets,
            s_vec<uint> &v_index_map,
            std::vector<ull> &v_value_map,
            std::vector<std::vector<uint>> &v_bucket,
            std::vector<ull> &v_bucket_seperator,
            std::vector<ull> &v_bucket_seperator_tmp,
            t_uint_iterator &v_iterator,
            const uint tid, const uint n_threads, const bool is_parallel_sort) noexcept {
        v_bucket[tid].clear();
        v_iterator[tid].clear();
        if (is_parallel_sort) {
            std::sort(std::next(v_index_map.begin(), v_omp_offsets[tid]),
                    std::next(v_index_map.begin(), v_omp_offsets[tid] + v_omp_sizes[tid]),
                    [&](const auto &i1, const auto &i2) -> bool {
                        return v_value_map[i1] < v_value_map[i2];
                    });
            #pragma omp barrier
            #pragma omp single
            {
                v_bucket_seperator.clear();
                v_bucket_seperator_tmp.clear();
                for (uint t = 0; t < n_threads; ++t) {
                    for (uint i = 0; i < n_threads - 1; ++i) {
                        uint index = v_omp_offsets[t] + ((v_omp_sizes[t] / n_threads) * (i + 1));
                        v_bucket_seperator_tmp.push_back(v_value_map[v_index_map[index]]);
                    }
                }
                std::sort(v_bucket_seperator_tmp.begin(), v_bucket_seperator_tmp.end());
                for (uint i = n_threads / 2; i < v_bucket_seperator_tmp.size(); i += n_threads) {
                    if (v_bucket_seperator.empty()) {
                        v_bucket_seperator.push_back(v_bucket_seperator_tmp[i]);
                    } else if (v_bucket_seperator.size() == n_threads - 2) {
                        v_bucket_seperator.push_back(v_bucket_seperator_tmp[i - 1]);
                    } else {
                        v_bucket_seperator.push_back(
                                (v_bucket_seperator_tmp[i - 1] + v_bucket_seperator_tmp[i]) / 2);
                    }
                }
            } // end single
            auto iter_begin = std::next(v_index_map.begin(), v_omp_offsets[tid]);
            auto iter_end = std::next(v_index_map.begin(), v_omp_offsets[tid] + v_omp_sizes[tid]);
            v_iterator[tid].push_back(iter_begin);
            for (auto &separator : v_bucket_seperator) {
                auto iter = std::lower_bound(
                        iter_begin,
                        iter_end,
                        separator,
                        [&v_value_map](const auto &i1, const auto &val) -> bool {
                            return v_value_map[i1] < val;
                        });
                v_iterator[tid].push_back(iter);
            }
            v_iterator[tid].push_back(std::next(v_index_map.begin(),
                    v_omp_offsets[tid] + v_omp_sizes[tid]));
            #pragma omp barrier
            for (uint t_index = 0; t_index < n_threads; ++t_index) {
                v_bucket[tid].insert(v_bucket[tid].end(), v_iterator[t_index][tid], v_iterator[t_index][tid + 1]);
            }
            #pragma omp barrier
            std::sort(v_bucket[tid].begin(), v_bucket[tid].end(), [&](const auto &i1, const auto &i2) -> bool {
                return v_value_map[i1] < v_value_map[i2];
            });
            #pragma omp barrier
            #pragma omp single
            {
                for (uint t = 1; t < n_threads; ++t) {
                    v_bucket[0].insert(v_bucket[0].end(), v_bucket[t].begin(), v_bucket[t].end());
                }
                v_index_map.clear();
                v_index_map.insert(v_index_map.end(), std::make_move_iterator(v_bucket[0].begin()),
                        std::make_move_iterator(v_bucket[0].end()));
            }
        } else if (!is_parallel_sort) {
            #pragma omp barrier
            #pragma omp single
            {
                std::sort(v_index_map.begin(), v_index_map.end(), [&](const auto &i1, const auto &i2) -> bool {
                    return v_value_map[i1] < v_value_map[i2];
                });
            }
        }
    }

#ifdef MPI_ON
    void mpi_sort_merge(std::vector<uint> &v_index_map,
            std::vector<ull> &v_value_map,
            std::vector<ull> &v_bucket_seperator,
            std::vector<ull> &v_bucket_seperator_tmp,
            std::vector<std::vector<uint>::iterator> &v_iterator,
            const uint n_nodes, const uint node_index) noexcept {

        v_bucket_seperator.clear();
//            v_bucket_seperator.resize(n_nodes);
        v_bucket_seperator_tmp.clear();
        v_bucket_seperator_tmp.resize(n_nodes * (n_nodes - 1), 0);
        std::vector<int> v_block_sizes(n_nodes, 0);
        std::vector<int> v_block_offsets(n_nodes, 0);
        int block_offset = 0;
        for (uint n = 0; n < n_nodes; ++n) {
            if (n < n_nodes - 1) {
                uint index = (node_index * (n_nodes - 1)) + n;
                uint map_index = (n + 1) * (v_index_map.size() / n_nodes);
                v_bucket_seperator_tmp[index] = v_value_map[v_index_map[map_index]];
            }
            v_block_sizes[n] = n_nodes - 1;
            v_block_offsets[n] = block_offset;
            block_offset += v_block_sizes[n];
        }
//            std::cout << "v_block_sizes_in_bytes[0]: " << v_block_sizes_in_bytes[0] << std::endl;
//        print_array("block sizes: ", &v_block_sizes[0], v_block_sizes.size());
//        print_array("block offsets: ", &v_block_offsets[0], v_block_offsets.size());
//        print_array("Pre: ", &v_bucket_seperator_tmp[0], v_bucket_seperator_tmp.size());
//        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &v_bucket_seperator_tmp[0],
                &v_block_sizes[0], &v_block_offsets[0], MPI_UNSIGNED_LONG_LONG,
                MPI_COMM_WORLD);
//        MPI_Barrier(MPI_COMM_WORLD);
//        print_array("Post: ", &v_bucket_seperator_tmp[0], v_bucket_seperator_tmp.size());
//        MPI_Barrier(MPI_COMM_WORLD);
        std::sort(v_bucket_seperator_tmp.begin(), v_bucket_seperator_tmp.end());
        for (uint n = 0; n < n_nodes - 1; ++n) {
            uint index = (n * n_nodes) + (n_nodes / 2);
            v_bucket_seperator.push_back((v_bucket_seperator_tmp[index] + v_bucket_seperator_tmp[index - 1]) / 2);
        }
//        MPI_Barrier(MPI_COMM_WORLD);
//        print_array("Selected: ", &v_bucket_seperator[0], v_bucket_seperator.size());
//        MPI_Barrier(MPI_COMM_WORLD);
//        std::vector<std::vector<uint>::iterator> v_iterator;
//        std::vector<std::vector<uint>> v_node_bucket(n_nodes);
        v_iterator.push_back(v_index_map.begin());
        // TODO parallelize
        for (auto &separator : v_bucket_seperator) {
            auto iter = std::lower_bound(
                    v_index_map.begin(),
                    v_index_map.end(),
                    separator,
                    [&v_value_map](const auto &i1, const auto &val) -> bool {
                        return v_value_map[i1] < val;
                    });
            v_iterator.push_back(iter);
        }
        v_iterator.push_back(v_index_map.end());
    }

#endif

    void determine_index_values(s_vec<float> &v_coords,
            std::unique_ptr<float[]> &v_min_bounds,
            d_vec<uint> &vv_index_map,
            d_vec<uint> &vv_cell_begin,
            std::vector<ull> &v_value_map,
            const ull *dims_mult,
            const int l, const uint size, const uint offset, const uint max_d, const float level_eps,
            const uint node_offset) noexcept {
        for (uint i = 0; i < size; ++i) {
            uint p_index = i + offset;
            int level_mod = 1;
            while (l - level_mod >= 0) {
                p_index = vv_index_map[l - level_mod][vv_cell_begin[l - level_mod][p_index]];
                ++level_mod;
            }
            uint coord_index = (p_index + node_offset) * max_d;
            v_value_map[offset + i] = get_cell_index(&v_coords[coord_index], v_min_bounds,
                    dims_mult, max_d, level_eps);
        }
    }

#ifdef CUDA_ON
    uint cu_index_level_and_get_cells(thrust::device_vector<float> &v_coords,
            thrust::device_vector<thrust::device_vector<uint>> &vv_index_map,
            thrust::device_vector<ull> &v_value_map,
            const uint size, const float level_eps, const ull *dims_mult, const uint l) {
        v_value_map.resize(size);
//        vv_index_map[l].resize(size);
//        thrust::sequence(vv_index_map[l].begin(), vv_index_map[l].end());
//        thrust::transform(vv_index_map[l].begin(), vv_index_map[l].end(), v_value_map.begin(),
//                thrust::negate<uint>());

        /*
                for (uint i = 0; i < size; ++i) {
            uint p_index = i + offset;
            int level_mod = 1;
            while (l - level_mod >= 0) {
                p_index = vv_index_map[l - level_mod][vv_cell_begin[l - level_mod][p_index]];
                ++level_mod;
            }
            uint coord_index = (p_index + node_offset) * max_d;
            v_value_map[offset + i] = get_cell_index(&v_coords[coord_index], v_min_bounds,
                    dims_mult, max_d, level_eps);
        }
         */
        return 0;
    }
#endif

    uint index_level_and_get_cells(s_vec<float> &v_coords,
            std::unique_ptr<float[]> &v_min_bounds,
            d_vec<uint> &vv_index_map,
            d_vec<uint> &vv_cell_begin,
            s_vec<uint> &v_cell_ns,
            std::vector<ull> &v_value_map,
            std::vector<std::vector<uint>> &v_bucket,
            std::vector<ull> &v_bucket_separator,
            std::vector<ull> &v_bucket_separator_tmp,
            t_uint_iterator &v_iterator,
            const uint size, const int l, const uint max_d, const uint node_offset, const float level_eps,
            const ull *dims_mult, const uint n_threads) noexcept {
        vv_index_map[l].resize(size);
        v_value_map.resize(size);
        uint unique_new_cells = 0;
        uint no_of_cells[n_threads];
        auto v_omp_sizes = std::make_unique<uint[]>(n_threads);
        auto v_omp_offsets = std::make_unique<uint[]>(n_threads);
        bool is_parallel_sort = true;
        deep_io::get_blocks_meta(v_omp_sizes, v_omp_offsets, size, n_threads);
        for (uint t = 0; t < n_threads; ++t) {
            no_of_cells[t] = 0;
            if (v_omp_sizes[t] == 0)
                is_parallel_sort = false;
        }
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            if (l == 0) {
                v_bucket[tid].reserve(v_omp_sizes[tid]);
            }
            std::iota(std::next(vv_index_map[l].begin(), v_omp_offsets[tid]),
                    std::next(vv_index_map[l].begin(), v_omp_offsets[tid] + v_omp_sizes[tid]),
                    v_omp_offsets[tid]);
            #pragma omp barrier
            determine_index_values(v_coords, v_min_bounds, vv_index_map, vv_cell_begin, v_value_map,
                    dims_mult, l, v_omp_sizes[tid], v_omp_offsets[tid], max_d, level_eps, node_offset);
            sort_indexes_omp(v_omp_sizes, v_omp_offsets, vv_index_map[l], v_value_map, v_bucket,
                    v_bucket_separator, v_bucket_separator_tmp, v_iterator, tid, n_threads, is_parallel_sort);
            #pragma omp barrier
            if (v_omp_sizes[tid] > 0) {
                uint new_cells = 1;
                uint index = vv_index_map[l][v_omp_offsets[tid]];
                ull last_value = v_value_map[index];
                // boundary correction
                if (tid > 0) {
                    index = vv_index_map[l][v_omp_offsets[tid] - 1];
                    if (v_value_map[index] == last_value)
                        --new_cells;
                }
                for (uint i = 1; i < v_omp_sizes[tid]; ++i) {
                    index = vv_index_map[l][v_omp_offsets[tid] + i];
                    if (v_value_map[index] != last_value) {
                        last_value = v_value_map[index];
                        ++new_cells;
                    }
                }
                no_of_cells[tid] = new_cells;
                #pragma omp atomic
                unique_new_cells += new_cells;
            }
            #pragma omp barrier
            #pragma omp single
            {
                vv_cell_begin[l].resize(unique_new_cells);
                v_cell_ns.resize(unique_new_cells);
            }

            if (no_of_cells[tid] > 0) {
                uint cell_offset = 0;
                for (uint t = 0; t < tid; ++t) {
                    cell_offset += no_of_cells[t];
                }
                uint index_map_offset = v_omp_offsets[tid];
                ull last_value = v_value_map[vv_index_map[l][index_map_offset]];
                // boundary corrections
                if (index_map_offset > 0) {
                    if (v_value_map[vv_index_map[l][index_map_offset - 1]] == last_value) {
                        while (v_value_map[vv_index_map[l][index_map_offset]] == last_value
                               && index_map_offset < v_value_map.size()) {
                            ++index_map_offset;
                        }
                        last_value = v_value_map[vv_index_map[l][index_map_offset]];
                    }
                }
                vv_cell_begin[l][cell_offset] = index_map_offset;
                uint cell_cnt = 1;
                for (uint i = index_map_offset; cell_cnt < no_of_cells[tid]; ++i) {
                    if (v_value_map[vv_index_map[l][i]] != last_value) {
                        last_value = v_value_map[vv_index_map[l][i]];
                        vv_cell_begin[l][cell_offset + cell_cnt] = i;
                        ++cell_cnt;
                    }
                }
            }
            #pragma omp barrier
            #pragma omp for
            for (uint i = 0; i < unique_new_cells - 1; ++i) {
                v_cell_ns[i] = vv_cell_begin[l][i + 1] - vv_cell_begin[l][i];
            }

        } // end parallel
        v_cell_ns[unique_new_cells - 1] = v_value_map.size() - vv_cell_begin[l][unique_new_cells - 1];
        return unique_new_cells;
    }

    bool process_pair_stack(s_vec<float> &v_coords,
            d_vec<uint> &vv_index_map,
            d_vec<uint> &vv_cell_begin,
            d_vec<uint> &vv_cell_ns,
            std::vector<std::vector<float>> &vv_min_cell_dim,
            std::vector<std::vector<float>> &vv_max_cell_dim,
            std::vector<uint> &v_leaf_cell_np,
            std::vector<uint> &v_point_np,
            std::vector<cell_meta_3> &v_stack,
            std::vector<bool> &v_range_table,
            std::vector<uint> &v_range_counts,
            std::vector<uint8_t> &v_cell_types,
            std::vector<uint8_t> &v_is_core,
            std::vector<int> &v_c_labels,
            const uint m, const uint max_d, const float e, const float e2, const bool is_proximity_cnt) noexcept {
        bool ret = false;
        while (!v_stack.empty()) {
            uint l = v_stack.back().l;
            uint c1 = v_stack.back().c1;
            uint c2 = v_stack.back().c2;
            v_stack.pop_back();
            uint begin1 = vv_cell_begin[l][c1];
            uint begin2 = vv_cell_begin[l][c2];
            if (l == 0) {
                if (is_proximity_cnt) {
                    ret = true;
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
        return ret;
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
            std::unique_ptr<float[]> &v_eps_levels,
            std::unique_ptr<ull[]> &v_dims_mult,
            std::unique_ptr<float[]> &v_min_bounds,
            d_vec<uint> &vv_index_map,
            d_vec<uint> &vv_cell_begin,
            d_vec<uint> &vv_cell_ns,
            std::vector<std::vector<float>> &vv_min_cell_dim,
            std::vector<std::vector<float>> &vv_max_cell_dim,
            const uint max_d, const uint n_threads,
            const uint max_levels, const uint n) noexcept {
        uint size = n;
        for (int l = 0; l < max_levels; ++l) {
#ifdef CUDA_ON
            thrust::device_vector<float> v_device_coords(v_coords);
            thrust::device_vector<ull> v_device_value_map;
            thrust::device_vector<thrust::device_vector<uint>> vv_device_index_map;
            uint cuda_size = cu_index_level_and_get_cells(v_device_coords, vv_device_index_map,
                    v_device_value_map, size,
                    v_eps_levels[l], &v_dims_mult[l * max_d], l);
            std::cout << "Level: " << l << " cuda size: " << cuda_size << std::endl;
#endif
//#ifndef CUDA_ON
            std::vector<ull> v_value_map;
            std::vector<std::vector<uint>> v_bucket(n_threads);
            std::vector<ull> v_bucket_separator;
            v_bucket_separator.reserve(n_threads);
            std::vector<ull> v_bucket_separator_tmp;
            v_bucket_separator_tmp.reserve(n_threads * n_threads);
            t_uint_iterator v_iterator(n_threads);
            size = index_level_and_get_cells(v_coords, v_min_bounds, vv_index_map, vv_cell_begin,
                    vv_cell_ns[l], v_value_map, v_bucket, v_bucket_separator, v_bucket_separator_tmp,
                    v_iterator, size, l, max_d, 0, v_eps_levels[l],
                    &v_dims_mult[l * max_d], n_threads);
//#endif
            calculate_level_cell_bounds(&v_coords[0], vv_cell_begin[l], vv_cell_ns[l],
                    vv_index_map[l], vv_min_cell_dim, vv_max_cell_dim, max_d, l);
        }
    }

    /*
#ifdef MPI_ON
    void process_nodes_nearest_neighbour(std::unique_ptr<float[]> &v_coords,
            std::unique_ptr<uint[]> &v_node_offset,
            std::vector<std::vector<std::vector<uint>>> &vvv_index_map,
            std::vector<std::vector<std::vector<uint>>> &vvv_cell_begin,
            std::vector<std::vector<std::vector<uint>>> &vvv_cell_ns,
            std::vector<std::vector<cell_meta_3>> &vv_stacks3,
            std::vector<std::vector<bool>> &vv_range_table,
            // TODO implement the range counts
            std::vector<std::vector<uint>> &vv_range_counts,
            std::vector<std::vector<uint>> &vv_leaf_cell_nn,
            std::vector<std::vector<uint>> &vv_point_nn,
            std::vector<std::vector<uint8_t>> &vv_cell_type,
            std::vector<std::vector<uint8_t>> &vv_is_core,
            std::vector<std::vector<std::vector<float>>> &vvv_min_cell_dim,
            std::vector<std::vector<std::vector<float>>> &vvv_max_cell_dim,
            const uint n_nodes, const uint max_d, const uint m, const float e, const uint max_levels,
            const uint node_index) noexcept {
        const float e2 = e * e;
        std::vector<uint> v_payload;
        std::vector<uint> v_sink_cells;
        std::vector<uint> v_sink_points;
        mpi_sum_vectors(vv_point_nn, v_payload, v_sink_points, v_sink_points, n_nodes,
                MPI_UNSIGNED, false);
        mpi_sum_vectors(vv_leaf_cell_nn, v_payload, v_sink_cells, v_sink_cells, n_nodes,
                MPI_UNSIGNED, false);
        int mod = n_nodes == 2 ? 5 : 4;
        int level = std::max((int)max_levels - mod, 0);
        std::vector<cell_meta_5> v_pair_task;
        v_pair_task.reserve(vvv_cell_ns[0][level].size() * n_nodes);
        int cnt = 0;
        for (uint n1 = 0; n1 < n_nodes; ++n1) {
            for (uint i = 0; i < vvv_cell_ns[n1][level].size(); ++i) {
                for (uint n2 = n1 + 1; n2 < n_nodes; ++n2) {
                    for (uint j = 0; j < vvv_cell_ns[n2][level].size(); ++j) {
                        if (++cnt % n_nodes == node_index) {
                            v_pair_task.emplace_back(level, i, j, n1, n2);
                        }
                    }
                }
            }
        }
        if (node_index == 0)
            std::cout << "pair tasks: " << v_pair_task.size() << std::endl;
        #pragma omp parallel for schedule(dynamic)
        for (uint i = 0; i < v_pair_task.size(); ++i) {
            uint tid = omp_get_thread_num();
            uint n1 = v_pair_task[i].n1;
            uint n2 = v_pair_task[i].n2;
            vv_stacks3[tid].emplace_back(v_pair_task[i].l, v_pair_task[i].c1, v_pair_task[i].c2);
            process_cell_pair(&v_coords[v_node_offset[n1] * max_d], &v_coords[v_node_offset[n2] * max_d],
                    vvv_index_map[n1], vvv_index_map[n2],
                    vvv_cell_ns[n1], vvv_cell_ns[n2], vvv_cell_begin[n1], vvv_cell_begin[n2],
                    vvv_min_cell_dim[n1], vvv_max_cell_dim[n1], vvv_min_cell_dim[n2], vvv_max_cell_dim[n2],
                    vv_leaf_cell_nn[n1], vv_leaf_cell_nn[n2], vv_point_nn[n1], vv_point_nn[n2],
                    vv_stacks3[tid], vv_range_table[tid], max_d, m, e, e2, true);
        }
        std::vector<uint> v_sink;
        mpi_sum_vectors(vv_point_nn, v_payload, v_sink, v_sink_points, n_nodes,
                MPI_UNSIGNED, true);
        mpi_sum_vectors(vv_leaf_cell_nn, v_payload, v_sink, v_sink_cells, n_nodes,
                MPI_UNSIGNED, true);
    }

#endif
     */

#ifdef MPI_ON
    void coord_partition_and_merge(std::unique_ptr<float[]> &v_coords,
            std::unique_ptr<float[]> &v_min_bounds,
            const ull* dims_mult, const float eps_level,
            const uint node_size, const uint node_offset, const uint max_d, const uint n_threads,
            const uint node_index, const uint n_nodes, const uint total_samples) {
        int sizes[n_nodes];
        int offsets[n_nodes];
        std::vector<uint> v_index_map(node_size);
        std::iota(v_index_map.begin(), v_index_map.end(), 0);
        std::vector<ull> v_value_map(node_size);
        uint index = 0;
        for (uint i = node_offset; i < node_size + node_offset; ++i, ++index) {
            v_value_map[index] = get_cell_index(&v_coords[i * max_d], v_min_bounds,
                    dims_mult, max_d, eps_level);
        }
        std::sort(v_index_map.begin(), v_index_map.end(), [&](const auto &i1, const auto &i2) -> bool {
            return v_value_map[i1] < v_value_map[i2];
        });
        std::vector<ull> v_bucket_seperator;
        v_bucket_seperator.reserve(n_threads);
        std::vector<ull> v_bucket_seperator_tmp;
        v_bucket_seperator_tmp.reserve(n_threads * n_threads);
        std::vector<std::vector<uint>::iterator> v_iterator;
        mpi_sort_merge(v_index_map, v_value_map, v_bucket_seperator, v_bucket_seperator_tmp, v_iterator,
                n_nodes, node_index);
        int block_sizes[n_nodes * n_nodes];
//         TODO tmp
//        for (uint i = 0; i < n_nodes * n_nodes; ++i) {
//            block_sizes[i] = 0;
//        }
        int block_offsets[n_nodes * n_nodes];
        index = node_index * n_nodes;
        offsets[0] = 0;
        for (uint n = 0; n < n_nodes; ++n) {
            block_sizes[index + n] = v_iterator[n + 1] - v_iterator[n];
            sizes[n] = n_nodes;
            if (n > 0)
                offsets[n] = offsets[n - 1] + sizes[n - 1];
        }
//        print_array("pre block sizes: ", &block_sizes[0], n_nodes * n_nodes);

        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &block_sizes[0], sizes,
                offsets, MPI_FLOAT, MPI_COMM_WORLD);
//        if (node_index == 0) {
//            print_array("post block sizes: ", &block_sizes[0], n_nodes * n_nodes);
//        }

        int last_size = 0;
        int last_val = 0;
        int val = 0;
        for (int n1 = 0; n1 < n_nodes; ++n1) {
            for (int n2 = 0; n2 < n_nodes; ++n2) {
                index = n2 * n_nodes + n1;
                val = last_val + last_size;
                block_offsets[index] = val;
                last_val = val;
                last_size = block_sizes[index];
            }
        }
//        if (node_index == 0) {
//            print_array("post block offsets: ", &block_offsets[0], n_nodes * n_nodes);
//        }
        std::vector<float> v_coord_copy(total_samples * max_d/*, INT32_MAX*/);

        index = node_index * n_nodes;
        for (uint n = 0; n < n_nodes; ++n) {
            uint begin_coord = block_offsets[index + n] * max_d;
            uint begin_block = (v_iterator[n] - v_iterator[0]);
//            std::cout << "node: " << node_index << " begin_block: " << begin_block << std::endl;
            for (uint i = 0; i < block_sizes[index + n]; ++i) {
                assert(begin_coord + i < v_coord_copy.size());
                for (uint j = 0; j < max_d; ++j) {
                    assert(begin_coord + (i * max_d) + j < v_coord_copy.size());
                    v_coord_copy[begin_coord + (i * max_d) + j] = v_coords[
                            (node_offset + v_index_map[begin_block + i]) * max_d + j];
                }

            }
        }
        last_size = 0;
        last_val = 0;
        val = 0;
        for (uint n1 = 0; n1 < n_nodes; ++n1) {
            index = 0;
            for (uint n2 = 0; n2 < n_nodes; ++n2) {
                sizes[index] = block_sizes[n2 * n_nodes + n1];
                val = last_val + last_size;
                offsets[index] = val;
                last_val = val;
                last_size = sizes[index];
                sizes[index] *= max_d;
                offsets[index] *= max_d;
                ++index;
            }
//            if (node_index == 1) {
//                print_array("transmit sizes: ", &sizes[0], n_nodes);
//                print_array("transmit offsets: ", &offsets[0], n_nodes);
//            }
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &v_coord_copy[0],
                    sizes, offsets, MPI_FLOAT, MPI_COMM_WORLD);
        }
//        for (auto &elem : v_coord_copy) {
//            assert(elem != INT32_MAX);
//        }
        std::copy(v_coord_copy.begin(), v_coord_copy.end(), &v_coords[0]);
    }
#endif

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
            total_samples = load_input(in_file, v_coords, n, max_d, n_nodes, node_index);
        });
        auto time_data_read = std::chrono::high_resolution_clock::now();
        if (node_index == 0) {
            std::cout << "Found " << n << " points in " << max_d << " dimensions" << " and read " << n <<
                      " of " << total_samples << " samples." << std::endl;
        }
        const auto e_inner = (e / sqrtf(3));
        const float e2 = e*e;
        auto v_min_bounds = std::make_unique<float[]>(max_d);
        auto v_max_bounds = std::make_unique<float[]>(max_d);
        int max_level;
        measure_duration("Determine Data Boundaries: ", node_index == 0, [&]() -> void {
            max_level = determine_data_boundaries(v_coords, v_min_bounds, v_max_bounds, n,
                    max_d, e_inner);
        });
        auto v_eps_levels = std::make_unique<float[]>(max_level);
        auto v_dims_mult = std::make_unique<ull[]>(max_level * max_d);
        measure_duration("Initialize Index Space: ", node_index == 0, [&]() -> void {
            #pragma omp parallel for
            for (uint l = 0; l < max_level; l++) {
                v_eps_levels[l] = (e_inner * pow(2, l));
                calc_dims_mult(&v_dims_mult[l * max_d], max_d, v_min_bounds, v_max_bounds, v_eps_levels[l]);
            }
        });
#ifdef MPI_ON
        if (n_nodes > 1) {
            measure_duration("Node Group Points: ", node_index == 0, [&]() -> void {

            });
        }
#endif
        /*
#ifdef MPI_ON
        // Share coordinates
        auto time_mpi1 = std::chrono::high_resolution_clock::now();
        int sizes[n_nodes];
        int offsets[n_nodes];
        for (int n = 0; n < n_nodes; ++n) {
            sizes[n] = v_node_sizes[n] * max_d;
            offsets[n] = v_node_offsets[n] * max_d;
        }
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &v_coords[0], sizes,
                offsets, MPI_FLOAT, MPI_COMM_WORLD);

        auto time_mpi2 = std::chrono::high_resolution_clock::now();
        if (!g_quiet && node_index == 0) {
            std::cout << "MPI Point Merge: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(time_mpi2 - time_mpi1).count()
                      << " milliseconds\n";
        }
        if (n_nodes > 1) {
            uint node_offset = v_node_offsets[node_index];
            uint node_size = v_node_sizes[node_index];
            coord_partition_and_merge(v_coords, v_min_bounds, &v_dims_mult[0], v_eps_levels[0],
                    node_size, node_offset, max_d, n_threads, node_index, n_nodes, total_samples);
        }
        auto time_mpi3 = std::chrono::high_resolution_clock::now();
        if (!g_quiet && node_index == 0) {
            std::cout << "MPI Point Sort: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(time_mpi3 - time_mpi2).count()
                      << " milliseconds\n";
        }
#endif
         */
//        std::vector<std::vector<uint>> vv_index_map(max_level);
//        std::vector<std::vector<uint>> vv_cell_begin(max_level);
//        std::vector<std::vector<uint>> vv_cell_ns(max_level);
        d_vec<uint> vv_index_map(max_level);
        d_vec<uint> vv_cell_begin(max_level);
        d_vec<uint> vv_cell_ns(max_level);
        std::vector<std::vector<float>> vv_min_cell_dim(max_level);
        std::vector<std::vector<float>> vv_max_cell_dim(max_level);
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
        /*
#ifdef MPI_ON
        if (n_nodes > 1) {
            measure_duration("MPI Grid-cell Tree Merge: ", node_index == 0, [&]() -> void {
                mpi_merge_cell_trees(vvv_index_map, vvv_cell_begin, vvv_cell_ns, vvv_min_cell_dim,
                        vvv_max_cell_dim, node_index, n_nodes, max_levels, max_d);
            });
        }
#endif
         */
        std::vector<cell_meta> v_tasks;
        measure_duration("Stacks and Tasks: ", node_index == 0, [&]() -> void {
            init_stacks(vv_cell_ns, v_leaf_cell_np, vv_stacks3, vv_range_table, vv_range_counts,
                    max_d, n_threads);
            populate_tasks(vv_cell_begin, v_tasks, max_level);
        });
        std::vector<int> v_c_labels(n, UNASSIGNED);

        measure_duration("Local Tree Proximity: ", node_index == 0, [&]() -> void {
            uint task_cnt = 0;
            uint empty_task_cnt = 0;

            #pragma omp parallel
            {
                uint tid = omp_get_thread_num();
                #pragma omp for schedule(dynamic)
                for (uint i = 0; i < v_tasks.size(); ++i) {
                    uint l = v_tasks[i].l;
                    uint c = v_tasks[i].c;
                    uint begin = vv_cell_begin[l][c];
                    bool check = false;
                    for (uint c1 = 0; c1 < vv_cell_ns[l][c]; ++c1) {
                        uint c1_index = vv_index_map[l][begin + c1];
                        for (uint c2 = c1 + 1; c2 < vv_cell_ns[l][c]; ++c2) {
                            uint c2_index = vv_index_map[l][begin + c2];
                            if (is_in_reach(&vv_min_cell_dim[l - 1][c1_index * max_d],
                                    &vv_max_cell_dim[l - 1][c1_index * max_d],
                                    &vv_min_cell_dim[l - 1][c2_index * max_d],
                                    &vv_max_cell_dim[l - 1][c2_index * max_d], max_d, e)) {
                                #pragma omp atomic
                                ++task_cnt;
                                vv_stacks3[tid].emplace_back(l - 1, c1_index, c2_index);
                                bool ret = process_pair_stack(v_coords, vv_index_map, vv_cell_begin,
                                        vv_cell_ns, vv_min_cell_dim, vv_max_cell_dim,
                                        v_leaf_cell_np, v_point_np, vv_stacks3[tid], vv_range_table[tid],
                                        vv_range_counts[tid], v_cell_type, v_is_core, v_c_labels,
                                        m, max_d, e, e2, true);
                                if (ret) {
                                    check = true;
                                }
                            }
                        }
                    }
                    if (!check) {
                        #pragma omp atomic
                        ++empty_task_cnt;
                    }
                }
            }
            std::cout << "empty tasks: " << empty_task_cnt << " of " << task_cnt << " tasks." << std::endl;
        });

#ifdef MPI_ON
        if (n_nodes > 1) {
            measure_duration("Node Trees Proximity: ", node_index == 0, [&]() -> void {
                std::vector<float> v_min_send_buf;
                std::vector<float> v_max_send_buf;
                auto v_gather_buf = std::make_unique<int[]>(n_nodes * n_nodes);
                // TODO reserve
                auto v_send_offsets = std::make_unique<int[]>(n_nodes);
                auto v_send_cnts = std::make_unique<int[]>(n_nodes);
//                auto v_gather_cnts = std::make_unique<int[]>(n_nodes);
//                auto v_gather_offsets = std::make_unique<int[]>(n_nodes);
//                for (uint i = 0; i < n_nodes; ++i) {
//                    v_gather_cnts[i] =
//                }
                // While stack not empty code pattern
//                vv_stacks3[0].push_back()
                uint gather_index = node_index * n_nodes;
                int offset = 0;
                for (int n = 0; n < n_nodes; ++n) {
                    uint cnt = 0;
                    int l = max_level-1;
                    for (uint i = 0; i < vv_cell_begin[l].size(); ++i) {
                        for (uint c = 0; c < vv_cell_ns[l][i]; ++c) {
                            uint begin = vv_cell_begin[l][c];
                            uint c_index = vv_index_map[l][begin + c];
                            // Insert instead of iteration
                            for (uint j = 0; j < max_d; ++j) {
                                v_min_send_buf.push_back(vv_min_cell_dim[l-1][c_index * max_d]);
                                v_max_send_buf.push_back(vv_max_cell_dim[l-1][c_index * max_d]);
                                ++cnt;
                            }
                        }
                    }
                    v_send_cnts[n] = cnt;
                    v_send_offsets[n] = offset;
                    v_gather_buf[gather_index + n] = cnt;
                }
                MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &v_gather_buf[0],
                        n_nodes, MPI_INT, MPI_COMM_WORLD);

//                MPI_Alltoallv(&v_min_send_buf[0], &v_send_cnts[0], &v_send_offsets[0],
//                        MPI_FLOAT,nullptr, nullptr,
//                        nullptr, MPI_FLOAT, MPI_COMM_WORLD);
            });
        }
#endif
        uint max_local_clusters;
        measure_duration("Infer Cores and Init Local Clusters: ", node_index == 0, [&]() -> void {
            max_local_clusters = infer_local_types_and_init_clusters(vv_index_map[0], vv_cell_begin[0],
                    vv_cell_ns[0], v_leaf_cell_np, v_point_np, v_cell_type, v_is_core, v_c_labels,
                    m);
            if (node_index == 0)
                std::cout << "Maximum number of local clusters: " << max_local_clusters << std::endl;
        });

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
#ifdef MPI_ON
        if (n_nodes > 1) {
            measure_duration("Node Trees Labels: ", node_index == 0, [&]() -> void {

            });
        }
#endif
        /*
#ifdef MPI_ON
        measure_duration("Update Shared Labels: ", node_index == 0, [&]() -> void {
            for (uint n = 0; n < n_nodes; ++n) {
                if (n == node_index)
                    continue;
                #pragma omp parallel for
                for (uint i = 0; i < vvv_cell_ns[n][0].size(); ++i) {
                    update_type(vvv_index_map[n][0], vvv_cell_ns[n][0], vvv_cell_begin[n][0],
                            vv_leaf_cell_nn[n], vv_point_nn[n], vv_is_core[n], vv_cell_type[n], i, m);
                }
            }
        });
#endif
         */
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