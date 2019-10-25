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
#define MPI_ON
#ifdef MPI_ON
#include <mpi.h>
#endif
#include "nextdbscan.h"
#include "deep_io.h"

namespace nextdbscan {

    static const int UNASSIGNED = -1;

    static const uint8_t NC = 0;
    static const uint8_t AC = 1;
    static const uint8_t SC = 2;

    static const int LABEL_CELL = INT32_MAX;

    typedef unsigned long long ull;
// TODO Detect when this is necessary during indexing
//    typedef unsigned __int128 ull;
    static bool g_quiet = false;

    struct cell_meta_3 {
        uint l, c1, c2;

        cell_meta_3(uint l, uint c1, uint c2) : l(l), c1(c1), c2(c2) {}
    };

    struct cell_meta_5 {
        uint l, c1, c2, n1, n2;

        cell_meta_5(uint l, uint c1, uint c2, uint n1, uint n2) : l(l), c1(c1), c2(c2), n1(n1), n2(n2) {}
    };

    void calc_bounds(std::unique_ptr<float[]> &v_coords, uint n, float *min_bounds,
            float *max_bounds, const uint max_d, const uint node_offset) noexcept {
        for (uint d = 0; d < max_d; d++) {
            min_bounds[d] = INT32_MAX;
            max_bounds[d] = INT32_MIN;
        }
#pragma omp parallel for reduction(max:max_bounds[:max_d]) reduction(min:min_bounds[:max_d])
        for (uint i = 0; i < n; i++) {
            size_t index = (i + node_offset) * max_d;
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
        std::vector<uint> dims;
        dims.resize(max_d);
        dims_mult[0] = 1;
        for (uint d = 0; d < max_d; d++) {
            dims[d] = static_cast<uint>((max_bounds[d] - min_bounds[d]) / e_inner) + 1;
            if (d > 0)
                dims_mult[d] = dims_mult[d - 1] * dims[d - 1];
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
        uint local_index;
        for (uint d = 0; d < max_d; d++) {
            local_index = static_cast<uint>((dv[d] - mv[d]) / size);
            cell_index += local_index * dm[d];
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
//    #pragma omp declare reduction(vec_merge_uint: std::vector<uint>: omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end())) initializer(omp_priv=omp_orig)

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

    inline void update_to_ac(std::vector<uint> &v_index_maps, std::vector<uint> &v_cell_ns,
            std::vector<uint> &v_cell_begin, std::vector<uint8_t> &is_core, std::vector<uint8_t> &v_types,
            const uint c) noexcept {
        v_types[c] = AC;
        uint begin = v_cell_begin[c];
        for (uint j = 0; j < v_cell_ns[c]; ++j) {
            is_core[v_index_maps[begin + j]] = 1;
        }
    }

    void update_type(std::vector<uint> &v_index_maps, std::vector<uint> &v_cell_ns,
            std::vector<uint> &v_cell_begin, std::vector<uint> &v_cell_nps, std::vector<uint> &v_point_nps,
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

    void update_point_nn(std::vector<uint> &v_index_maps_1, std::vector<uint> &v_index_maps_2,
            std::vector<uint> &v_point_nps_1, std::vector<uint> &v_point_nps_2,
            std::vector<bool> &v_range_table, const uint begin1, const uint begin2,
            const uint size1, const uint size2) noexcept {
        uint index = 0;
        for (uint k1 = 0; k1 < size1; ++k1) {
            uint p1 = v_index_maps_1[begin1 + k1];
            for (uint k2 = 0; k2 < size2; ++k2, ++index) {
                uint p2 = v_index_maps_2[begin2 + k2];
                if (v_range_table[index]) {
#pragma omp atomic
                    ++v_point_nps_1[p1];
#pragma omp atomic
                    ++v_point_nps_2[p2];
                }
            }
        }
    }

    bool fill_range_table(const float *v_coords, std::vector<uint> &v_index_map_level,
            std::vector<uint> &v_cell_ns_level, std::vector<bool> &v_range_table, const uint c1,
            const uint begin1, const uint c2, const uint begin2, const uint max_d, const float e2) noexcept {
        uint size1 = v_cell_ns_level[c1];
        uint size2 = v_cell_ns_level[c2];
        uint index = 0;
        uint total_size = size1 * size2;
        std::fill(v_range_table.begin(), v_range_table.begin() + total_size, false);
        for (uint k1 = 0; k1 < size1; ++k1) {
            uint p1 = v_index_map_level[begin1 + k1];
            for (uint k2 = 0; k2 < size2; ++k2, ++index) {
                uint p2 = v_index_map_level[begin2 + k2];
                if (dist_leq(&v_coords[p1 * max_d], &v_coords[p2 * max_d], max_d, e2)) {
                    v_range_table[index] = true;
                }
            }
        }
        for (uint i = 0; i < total_size; ++i) {
            if (!v_range_table[i])
                return false;
        }
        return true;
    }

    void update_points(std::vector<uint> &v_index_map_level, std::vector<uint> &v_cell_nps,
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

    void update_cell_pair_nn(std::vector<uint> &v_index_map_level, std::vector<uint> &v_cell_ns_level,
            std::vector<uint> &v_cell_nps, std::vector<uint> &v_point_nps, std::vector<bool> &v_range_table,
            std::vector<uint> &v_range_count,
            const uint c1, const uint begin1, const uint c2, const uint begin2,
            const bool is_update1, const bool is_update2) noexcept {
        uint size1 = v_cell_ns_level[c1];
        uint size2 = v_cell_ns_level[c2];
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

    void process_pair_nn(const float *v_coords, std::vector<uint> &v_index_maps,
            std::vector<uint> &v_point_nps,
            std::vector<uint> &v_cell_ns,
            std::vector<bool> &v_range_table,
            std::vector<uint> &v_range_cnt,
            std::vector<uint> &v_cell_nps,
            const uint max_d, const float e2, const uint m,
            const uint c1, const uint begin1, const uint c2, const uint begin2) noexcept {
        // TODO use an int value instead to rule out zero quickly
        bool all_range_check = fill_range_table(v_coords, v_index_maps, v_cell_ns,
                v_range_table, c1, begin1, c2, begin2,
                max_d, e2);
        if (all_range_check) {
            if (v_cell_nps[c1] < m) {
#pragma omp atomic
                v_cell_nps[c1] += v_cell_ns[c2];
            }
            if (v_cell_nps[c2] < m) {
#pragma omp atomic
                v_cell_nps[c2] += v_cell_ns[c1];
            }
        } else {
            update_cell_pair_nn(v_index_maps, v_cell_ns, v_cell_nps, v_point_nps, v_range_table,
                    v_range_cnt, c1, begin1, c2, begin2, v_cell_nps[c1] < m,
                    v_cell_nps[c2] < m);
        }
    }

    void read_input_txt(const std::string &in_file, std::unique_ptr<float[]> &v_points, int max_d) noexcept {
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


    result calculate_results(std::vector<std::vector<uint8_t>> &vv_is_core, std::vector<int> &v_cluster_label,
            std::vector<int> &v_labels, uint n) noexcept {
        result res{0, 0, 0, n, new int[n]};

        uint sum = 0;
#pragma omp parallel for reduction(+:sum)
        for (uint n = 0; n < vv_is_core.size(); ++n) {
            for (auto &val : vv_is_core[n]) {
                if (val)
                    ++sum;
            }
        }
        res.core_count = sum;
        sum = 0;
#pragma omp parallel for reduction(+:sum)
        for (uint i = 0; i < v_cluster_label.size(); ++i) {
            if (v_cluster_label[i] == LABEL_CELL)
                ++sum;

        }
        res.clusters = sum;

        uint &noise = res.noise;
#pragma omp parallel for reduction(+: noise)
        for (int i = 0; i < n; i++) {
            if (v_labels[i] == UNASSIGNED) {
                ++noise;
            }
        }

        /*
        bool_vector labels(n);
        labels.fill(n, false);
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            int label = get_label(&ps[i])->label;
            (*res.point_clusters)[i] = label;
            if (label != UNASSIGNED) {
                labels[label] = true;
            }
        }
        uint &clusters = res.clusters;
        #pragma omp parallel for reduction(+: clusters)
        for (int i = 0; i < n; i++) {
            if (labels[i]) {
                ++clusters;
            }
        }

        std::unordered_map<int, int> map;
        std::unordered_map<int, int>::iterator iter;
        int id = 0;
        for (int i = 0; i < n; i++) {
            int val = (*res.point_clusters)[i];
            if (val == UNASSIGNED)
                continue;
            iter = map.find(val);
            if (iter == map.end()) {
                (*res.point_clusters)[i] = id;
                map.insert(std::make_pair(val, id++));
            } else {
                (*res.point_clusters)[i] = iter->second;
            }
        }
        uint &noise = res.noise;
        #pragma omp parallel for reduction(+: noise)
        for (int i = 0; i < n; i++) {
            if (get_label(&ps[i])->label == UNASSIGNED) {
                ++noise;
            }
        }
         */
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

    uint process_input(const std::string &in_file, std::unique_ptr<float[]> &v_points, uint &n, uint &max_d,
            const uint blocks_no, const uint block_index) noexcept {
        std::string s_cmp = ".bin";
        int total_samples = 0;
        if (in_file.compare(in_file.size() - s_cmp.size(), s_cmp.size(), s_cmp) == 0) {
            char c[in_file.size() + 1];
            strcpy(c, in_file.c_str());
            auto *data = new deep_io(c, blocks_no, block_index);
            int read_bytes = data->load_next_samples(v_points);
//            std::cout << "read data bytes: " << read_bytes << std::endl;
            n = data->sample_read_no;
            max_d = data->feature_no;
            return data->sample_no;
        } else {
            count_lines_and_dimensions(in_file, n, max_d);
            v_points = std::make_unique<float[]>(n * max_d);
            std::cout << "WARNING: USING VERY SLOW NON-PARALLEL I/O." << std::endl;
            read_input_txt(in_file, v_points, max_d);
            total_samples = n;
        }
        return total_samples;
    }

    void calculate_level_cell_bounds(float *v_coords, std::vector<uint> &v_cell_begins,
            std::vector<uint> &v_cell_ns, std::vector<uint> &v_index_maps,
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

    int determine_data_boundaries(std::unique_ptr<float[]> &v_coords, std::unique_ptr<float[]> &v_min_bounds,
            std::unique_ptr<float[]> &v_max_bounds, const uint n, const uint node_offset, const uint max_d,
            const float e_inner) noexcept {
        float max_limit = INT32_MIN;
        calc_bounds(v_coords, n, &v_min_bounds[0], &v_max_bounds[0], max_d, node_offset);
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

    void process_pair_labels(const float *v_coords,
            std::vector<int> &v_t_c_cores,
            std::vector<int> &v_c_index,
            std::vector<std::vector<uint>> &vv_cell_ns, std::vector<std::vector<uint>> &vv_index_maps,
            std::vector<uint8_t> &v_cell_types, std::vector<uint8_t> &v_is_core,
            const uint c1, const uint c2, const uint l, const uint begin1, const uint begin2, const uint max_d,
            const float e2) noexcept {
        // Do both cells have cores ?
        if (v_cell_types[c1] != NC && v_cell_types[c2] != NC) {
            for (uint k1 = 0; k1 < vv_cell_ns[l][c1]; ++k1) {
                uint p1 = vv_index_maps[l][begin1 + k1];
                if (!v_is_core[p1]) {
                    continue;
                }
                int label1 = v_c_index[p1];
                bool flatten = false;
                while (v_t_c_cores[label1] != LABEL_CELL) {
                    label1 = v_t_c_cores[label1];
                    flatten = true;
                }
                if (flatten) {
                    v_c_index[p1] = label1;
                }
                for (uint k2 = 0; k2 < vv_cell_ns[l][c2]; ++k2) {
                    uint p2 = vv_index_maps[l][begin2 + k2];
                    if (v_is_core[p2]) {
                        int label2 = v_c_index[p2];

                        flatten = false;
                        while (v_t_c_cores[label2] != LABEL_CELL) {
                            label2 = v_t_c_cores[label2];
                            flatten = true;
                        }
                        if (flatten) {
                            v_c_index[p2] = label2;
                        }
                        if (label1 != label2) {
                            if (dist_leq(&v_coords[p1 * max_d],
                                    &v_coords[p2 * max_d], max_d, e2)) {
                                if (label1 < label2)
                                    v_t_c_cores[label2] = label1;
                                else
                                    v_t_c_cores[label1] = label2;
                                return;
                            }
                        } else {
                            return;
                        }
                    }
                }
            }
        } else {
            // one NC one not
            for (uint k1 = 0; k1 < vv_cell_ns[l][c1]; ++k1) {
                uint p1 = vv_index_maps[l][begin1 + k1];
                if (!v_is_core[p1] && v_c_index[p1] != UNASSIGNED)
                    continue;
                for (uint k2 = 0; k2 < vv_cell_ns[l][c2]; ++k2) {
                    uint p2 = vv_index_maps[l][begin2 + k2];
                    if (!v_is_core[p2] && v_c_index[p2] != UNASSIGNED)
                        continue;
                    if (v_is_core[p1]) {
                        if (dist_leq(&v_coords[p1 * max_d], &v_coords[p2 * max_d], max_d, e2)) {
                            v_c_index[p2] = v_c_index[p1];
                        }
                    } else if (v_is_core[p2]) {
                        if (dist_leq(&v_coords[p1 * max_d], &v_coords[p2 * max_d], max_d, e2)) {
                            v_c_index[p1] = v_c_index[p2];
                            k2 = vv_cell_ns[l][c2];
                        }
                    }
                }
            }
        }
    }

    bool determine_point_reach(const float *v_coords_1, const float *v_coords_2, std::vector<uint> &v_index_maps_1,
            std::vector<uint> &v_index_maps_2, std::vector<bool> &v_range_table, const uint begin1, const uint begin2,
            const uint size1, const uint size2, const uint max_d, const float e2) noexcept {
        uint index = 0;
        uint total_size = size1 * size2;
        assert(total_size < v_range_table.size());
        std::fill(v_range_table.begin(), v_range_table.begin() + total_size, false);
        for (uint k1 = 0; k1 < size1; ++k1) {
            uint p1 = v_index_maps_1[begin1 + k1];
            for (uint k2 = 0; k2 < size2; ++k2, ++index) {
                uint p2 = v_index_maps_2[begin2 + k2];
                if (dist_leq(&v_coords_1[p1 * max_d], &v_coords_2[p2 * max_d], max_d, e2)) {
                    v_range_table[index] = true;
                }
            }
        }
        for (uint i = 0; i < total_size; ++i) {
            if (!v_range_table[i])
                return false;
        }
        return true;
    }

    void process_cell_pair(const float *v_coords_1, const float *v_coords_2,
            std::vector<std::vector<uint>> &vv_index_maps_1,
            std::vector<std::vector<uint>> &vv_index_maps_2,
            std::vector<std::vector<uint>> &vv_cell_ns_1,
            std::vector<std::vector<uint>> &vv_cell_ns_2,
            std::vector<std::vector<uint>> &vv_cell_begins_1,
            std::vector<std::vector<uint>> &vv_cell_begins_2,
            std::vector<std::vector<float>> &vv_min_cell_dims_1,
            std::vector<std::vector<float>> &vv_max_cell_dims_1,
            std::vector<std::vector<float>> &vv_min_cell_dims_2,
            std::vector<std::vector<float>> &vv_max_cell_dims_2,
            std::vector<uint> &v_leaf_cell_nns_1,
            std::vector<uint> &v_leaf_cell_nns_2,
            std::vector<uint> &v_point_nns_1,
            std::vector<uint> &v_point_nns_2,
            std::vector<cell_meta_3> &stack,
            std::vector<bool> &v_range_table,
            const uint max_d, const uint m, const float e, const float e2, const bool is_nn) noexcept {
        while (!stack.empty()) {
            uint l = stack.back().l;
            uint c1 = stack.back().c1;
            uint c2 = stack.back().c2;
            stack.pop_back();
            // TODO move this back
            if (!is_in_reach(&vv_min_cell_dims_1[l][c1 * max_d],
                    &vv_max_cell_dims_1[l][c1 * max_d], &vv_min_cell_dims_2[l][c2 * max_d],
                    &vv_max_cell_dims_2[l][c2 * max_d], max_d, e)) {
                continue;
            }
            uint begin1 = vv_cell_begins_1[l][c1];
            uint begin2 = vv_cell_begins_2[l][c2];
            if (l == 0) {
                if (is_nn && (v_leaf_cell_nns_1[c1] < m || v_leaf_cell_nns_2[c2] < m)) {
                    bool all_range_check = determine_point_reach(v_coords_1, v_coords_2, vv_index_maps_1[0],
                            vv_index_maps_2[0], v_range_table, begin1, begin2, vv_cell_ns_1[0][c1],
                            vv_cell_ns_2[0][c2], max_d, e2);
                    if (all_range_check) {
                        if (v_leaf_cell_nns_1[c1] < m) {
                            #pragma omp atomic
                            v_leaf_cell_nns_1[c1] += vv_cell_ns_2[0][c2];
                        }
                        if (v_leaf_cell_nns_2[c2] < m) {
                            #pragma omp atomic
                            v_leaf_cell_nns_2[c2] += vv_cell_ns_1[0][c1];
                        }
                    } else {
                        update_point_nn(vv_index_maps_1[0], vv_index_maps_2[0], v_point_nns_1, v_point_nns_2,
                                v_range_table, begin1, begin2, vv_cell_ns_1[0][c1], vv_cell_ns_2[0][c2]);
                    }
                } else {
                    // TODO labels
                }
            } else {
                for (uint k1 = 0; k1 < vv_cell_ns_1[l][c1]; ++k1) {
                    uint c1_next = vv_index_maps_1[l][begin1 + k1];
                    for (uint j = 0; j < vv_cell_ns_2[l][c2]; ++j) {
                        uint c2_next = vv_index_maps_2[l][begin2 + j];
                        stack.emplace_back(l - 1, c1_next, c2_next);
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
        // TODO Use the offset with OMP
        int index = 0;
        for (int n = 0; n < n_nodes; ++n) {
            for (auto &val : vv_vector[n]) {
                v_payload[index] = is_additive ? val - v_additive[index] : val;
                ++index;
            }
        }
        // TODO is a barrier necessary ?
//        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(&v_payload[0], &v_sink[0], send_cnt, send_type, MPI_SUM, MPI_COMM_WORLD);
        index = 0;
        for (int n = 0; n < n_nodes; ++n) {
            for (int i = 0; i < vv_vector[n].size(); ++i, ++index) {
//                assert(v_sink[index] != (T)-1);
                if (is_additive) {
                    vv_vector[n][i] = v_additive[index] + v_sink[index];
                } else {
                    vv_vector[n][i] = v_sink[index];
                }
            }
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
            std::vector<uint> &v_index_map,
            std::vector<ull> &v_value_map,
            std::vector<std::vector<uint>> &v_bucket,
            std::vector<ull> &v_bucket_seperator,
            std::vector<ull> &v_bucket_seperator_tmp,
            std::vector<std::vector<std::vector<uint>::iterator>> &v_iterator,
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

//                for (uint t = 0; t < n_threads; ++t) {
//                    uint index = v_omp_offsets[t] + (v_omp_sizes[t]/2);
//                    v_bucket_seperator_tmp.push_back((v_value_map[v_index_map[index]] + v_value_map[v_index_map[index+1]]) / 2);
//                    v_bucket_seperator_tmp.push_back(v_value_map[v_index_map[index+1]]);
//                }
                std::sort(v_bucket_seperator_tmp.begin(), v_bucket_seperator_tmp.end());
//                if (l == 0) {
//                    print_array("aggregated bucket seperators: ", &v_bucket_seperator_tmp[0],
//                            v_bucket_seperator_tmp.size());
//                }

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

//                v_bucket_seperator.push_back(v_bucket_seperator_tmp[1]);
//                if (l == 0) {
//                    print_array("Selected bucket seperators: ", &v_bucket_seperator[0],
//                            v_bucket_seperator.size());
//                }
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
//            #pragma omp barrier
//            if (tid == 0) {
//                std::cout << "level " << l << " bucket sizes: ";
//                for (uint t = 0; t < n_threads; ++t) {
//                    std::cout << v_bucket[t].size() << " ";
//                }
//                std::cout << std::endl;
//            }
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

    void determine_index_values(std::unique_ptr<float[]> &v_coords,
            std::unique_ptr<float[]> &v_min_bounds,
            std::vector<std::vector<uint>> &vv_index_map,
            std::vector<std::vector<uint>> &vv_cell_begin,
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

    uint index_level_and_get_cells(std::unique_ptr<float[]> &v_coords,
            std::unique_ptr<float[]> &v_min_bounds,
            std::vector<std::vector<uint>> &vv_index_map,
            std::vector<std::vector<uint>> &vv_cell_begin,
            std::vector<uint> &v_cell_ns,
            std::vector<ull> &v_value_map,
            std::vector<std::vector<uint>> &v_bucket,
            std::vector<ull> &v_bucket_seperator,
            std::vector<ull> &v_bucket_seperator_tmp,
            std::vector<std::vector<std::vector<uint>::iterator>> &v_iterator,
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
//            if (tid == 0 && l == 0) {
//                std::cout << vv_index_map[l][0] << " : " << vv_index_map[l][1] << std::endl;
//            }
#pragma omp barrier
            determine_index_values(v_coords, v_min_bounds, vv_index_map, vv_cell_begin, v_value_map,
                    dims_mult, l, v_omp_sizes[tid], v_omp_offsets[tid], max_d, level_eps, node_offset);
            sort_indexes_omp(v_omp_sizes, v_omp_offsets, vv_index_map[l], v_value_map, v_bucket,
                    v_bucket_seperator, v_bucket_seperator_tmp, v_iterator, tid, n_threads, is_parallel_sort);
#ifdef MPI_ON
//        if (l == 0 && n_nodes > 1) {
//            #pragma omp barrier
//            #pragma omp single
//            {
//                mpi_sort_merge(vv_index_map[0], v_value_map, v_bucket_seperator, v_bucket_seperator_tmp,
//                        n_nodes, node_index, node_offset);
//            }
//            #pragma omp barrier
//            determine_index_values(v_coords, v_min_bounds, vv_index_map, vv_cell_begin, v_value_map,
//                    dims_mult, l, v_omp_sizes[tid], v_omp_offsets[tid], max_d, level_eps, node_index);
//            sort_indexes_omp(v_omp_sizes, v_omp_offsets, vv_index_map[l], v_value_map, v_bucket,
//                    v_bucket_seperator, v_bucket_seperator_tmp, v_iterator, tid, n_threads, is_parallel_sort);
//        }
#endif
#pragma omp barrier
            if (v_omp_sizes[tid] > 0) {
                uint new_cells = 1;
//                assert(v_omp_offsets[tid] < vv_index_map[l].size());
                uint index = vv_index_map[l][v_omp_offsets[tid]];
//                assert(index < v_value_map.size());
                ull last_value = v_value_map[index];
                // boundary correction
                if (tid > 0) {
//                    assert(v_omp_offsets[tid] > 0);
                    index = vv_index_map[l][v_omp_offsets[tid] - 1];
                    if (v_value_map[index] == last_value)
                        --new_cells;
                }
                for (uint i = 1; i < v_omp_sizes[tid]; ++i) {
//                    assert(v_omp_offsets[tid] + i < vv_index_map[l].size());
                    index = vv_index_map[l][v_omp_offsets[tid] + i];
//                    assert(index < v_value_map.size());
                    if (v_value_map[index] != last_value) {
                        last_value = v_value_map[index];
                        ++new_cells;
                    }
                }
                no_of_cells[tid] = new_cells;
#pragma omp atomic
                unique_new_cells += new_cells;
            }
//            #pragma omp barrier
//            if (tid == 0) {
//                std::cout << "new cells: " << unique_new_cells << std::endl;
//            }
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
//                    assert(v_omp_offsets[tid] > 0);
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

        /*
        vv_cell_begin[l][0] = 0;
        uint cell_cnt = 1;
        ull last_value = v_value_map[vv_index_map[l][0]];
        for (uint i = 1; i < v_value_map.size(); ++i) {
            if (v_value_map[vv_index_map[l][i]] != last_value) {
                last_value = v_value_map[vv_index_map[l][i]];
                vv_cell_begin[l][cell_cnt] = i;
                v_cell_ns[cell_cnt-1] = vv_cell_begin[l][cell_cnt] - vv_cell_begin[l][cell_cnt-1];
                ++cell_cnt;
            }
        }
        assert(cell_cnt == unique_new_cells);
        v_cell_ns[cell_cnt-1] = v_value_map.size() - vv_cell_begin[l][cell_cnt-1];
         */

        return unique_new_cells;
    }

    void process_pair_stack(const float *v_coords, std::vector<int> &v_t_c_cores, std::vector<int> &v_c_index,
            std::vector<std::vector<uint>> &vv_index_map,
            std::vector<std::vector<uint>> &vv_cell_begin,
            std::vector<std::vector<uint>> &vv_cell_ns,
            std::vector<std::vector<float>> &vv_min_cell_dim,
            std::vector<std::vector<float>> &vv_max_cell_dim,
            std::vector<uint> &v_leaf_cell_nns,
            std::vector<uint> &v_point_nns,
            std::vector<cell_meta_3> &v_stacks3,
            std::vector<bool> &v_range_table,
            std::vector<uint> &v_range_counts,
            std::vector<uint8_t> &v_cell_types,
            std::vector<uint8_t> &v_is_core,
            const uint m, const uint max_d, const float e, const float e2, const bool is_nn) noexcept {
        while (!v_stacks3.empty()) {
            uint l = v_stacks3.back().l;
            uint c1 = v_stacks3.back().c1;
            uint c2 = v_stacks3.back().c2;
            v_stacks3.pop_back();
            uint begin1 = vv_cell_begin[l][c1];
            uint begin2 = vv_cell_begin[l][c2];
            // TODO Throw meaningless tasks out at NN
            if (l == 0) {
                if (is_nn) {
                    if (v_leaf_cell_nns[c1] < m || v_leaf_cell_nns[c2] < m) {
                        process_pair_nn(v_coords, vv_index_map[0], v_point_nns,
                                vv_cell_ns[0], v_range_table, v_range_counts, v_leaf_cell_nns,
                                max_d, e2, m, c1, begin1, c2, begin2);
                    }
                } else {
                    if (v_cell_types[c1] != NC || v_cell_types[c2] != NC) {
                        process_pair_labels(v_coords, v_t_c_cores, v_c_index, vv_cell_ns,
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
                            v_stacks3.emplace_back(l - 1, c1_next, c2_next);
                        }
                    }
                }
            }
        }
    }

    uint infer_types_and_init_clusters_omp(std::vector<std::vector<std::vector<uint>>> &vvv_index_map,
            std::vector<std::vector<std::vector<uint>>> &vvv_cell_begin,
            std::vector<std::vector<std::vector<uint>>> &vvv_cell_ns,
            std::vector<std::vector<uint>> &vv_leaf_cell_nn,
            std::vector<std::vector<uint>> &vv_point_nn,
            std::vector<std::vector<uint8_t>> &vv_cell_types,
            std::vector<std::vector<uint8_t>> &vv_is_core,
            std::vector<int> &v_c_index,
            const uint m, const uint n_threads, const uint n_nodes, const uint node_index) noexcept {
        std::vector<uint> v_cluster_cells[n_threads];
        uint max_clusters = 0;

        for (uint n = 0; n < vvv_cell_ns.size(); ++n) {
            for (uint i = 0; i < vvv_cell_ns[n][0].size(); ++i) {
                update_type(vvv_index_map[n][0], vvv_cell_ns[n][0], vvv_cell_begin[n][0],
                        vv_leaf_cell_nn[n], vv_point_nn[n], vv_is_core[n], vv_cell_types[n], i, m);
            }
        }
        // TODO fix the cluster issue
        return 0;
        /*
        auto v_task_size = std::make_unique<uint[]>(n_threads);
        auto v_task_offset = std::make_unique<uint[]>(n_threads);
        deep_io::get_blocks_meta(v_task_size, v_task_offset, v_cell_types.size(), n_threads);
        #pragma omp parallel
        {
            uint tid = omp_get_thread_num();
            #pragma omp for
            for (uint i = 0; i < vv_cell_ns[0].size(); ++i) {
                update_type(vv_index_map[0], vv_cell_ns[0], vv_cell_begin[0],
                        v_leaf_cell_nns, v_point_nns, v_is_core, v_cell_types, i, m);
            }
            #pragma omp barrier
            for (uint i = v_task_offset[tid]; i < v_task_offset[tid] + v_task_size[tid]; ++i) {
                if (v_cell_types[i] == NC) {
                    continue;
                }
                v_cluster_cells[tid].push_back(i);
            }
            #pragma omp atomic
            max_clusters += v_cluster_cells[tid].size();
            #pragma omp barrier
            uint label = 0;
            for (uint t = 0; t < n_threads; ++t) {
                if (t < tid) {
                    label += v_cluster_cells[t].size();
                }
            }
            for (auto &i : v_cluster_cells[tid]) {
                uint begin = vv_cell_begin[0][i];
                for (uint j = 0; j < vv_cell_ns[0][i]; ++j) {
                    uint p = vv_index_map[0][begin + j];
                    v_c_index[p] = label;
                }
                ++label;
            }
        }
         */
        return max_clusters;
    }

    void determine_tasks(std::vector<std::vector<uint>> &vv_index_map,
            std::vector<std::vector<uint>> &vv_cell_begin,
            std::vector<std::vector<uint>> &vv_cell_ns,
            std::vector<cell_meta_3> &v_tasks,
            std::vector<std::vector<float>> &vv_min_cell_dim,
            std::vector<std::vector<float>> &vv_max_cell_dim,
            const uint max_levels, const uint max_d, const float e, const uint n_threads) noexcept {

        std::vector<cell_meta_3> v_tmp;
        v_tmp.reserve(vv_cell_begin[0].size());
        std::vector<std::vector<cell_meta_3>> v_tasks_t(n_threads);
        for (uint t = 0; t < n_threads; ++t) {
            v_tasks_t[t].reserve(vv_cell_begin[0].size() / n_threads);
        }
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            for (uint l = 1; l < max_levels; ++l) {
                v_tasks_t[tid].clear();
#pragma omp for
                for (uint i = 0; i < vv_cell_begin[l].size(); ++i) {
                    uint begin = vv_cell_begin[l][i];
                    for (uint c1 = 0; c1 < vv_cell_ns[l][i]; ++c1) {
                        uint c1_index = vv_index_map[l][begin + c1];
                        for (uint c2 = c1 + 1; c2 < vv_cell_ns[l][i]; ++c2) {
                            uint c2_index = vv_index_map[l][begin + c2];
                            v_tasks_t[tid].emplace_back(l - 1, c1_index, c2_index);
                        }
                    }
                }
#pragma omp barrier
#pragma omp single
                {
                    v_tmp.clear();
                    for (uint t = 0; t < n_threads; ++t) {
                        v_tmp.insert(v_tmp.end(), v_tasks_t[t].begin(), v_tasks_t[t].end());
                    }
                }
                v_tasks_t[tid].clear();
#pragma omp for
                for (uint i = 0; i < v_tmp.size(); ++i) {
                    uint c1_index = v_tmp[i].c1;
                    uint c2_index = v_tmp[i].c2;
                    if (is_in_reach(&vv_min_cell_dim[l - 1][c1_index * max_d],
                            &vv_max_cell_dim[l - 1][c1_index * max_d],
                            &vv_min_cell_dim[l - 1][c2_index * max_d],
                            &vv_max_cell_dim[l - 1][c2_index * max_d], max_d, e)) {
                        v_tasks_t[tid].push_back(v_tmp[i]);
                    }
                }
#pragma omp barrier
#pragma omp single
                {
                    for (uint t = 0; t < n_threads; ++t) {
                        v_tasks.insert(v_tasks.end(), v_tasks_t[t].begin(), v_tasks_t[t].end());
                    }
                }

            }
        } // end parallel region
    }

    void init_stacks(std::vector<std::vector<std::vector<uint>>> &vvv_cell_ns,
            std::vector<std::vector<uint>> &vv_leaf_cell_nn,
            std::vector<std::vector<cell_meta_3>> &vv_stacks3,
            std::vector<std::vector<bool>> &vv_range_table,
            std::vector<std::vector<uint>> &vv_range_counts,
            const uint max_d, const uint n_nodes, const uint n_threads, const uint node_index) noexcept {
        uint max_points_in_leaf_cell = 0;
        for (uint n = 0; n < n_nodes; ++n) {
#pragma omp parallel for reduction(max: max_points_in_leaf_cell)
            for (uint i = 0; i < vvv_cell_ns[n][0].size(); ++i) {
                if (n == node_index)
                    vv_leaf_cell_nn[n][i] = vvv_cell_ns[n][0][i];
                if (vvv_cell_ns[n][0][i] > max_points_in_leaf_cell) {
                    max_points_in_leaf_cell = vvv_cell_ns[n][0][i];
                }
            }
        }
#pragma omp parallel for
        for (uint t = 0; t < n_threads; ++t) {
            vv_stacks3[t].reserve(vvv_cell_ns[0][0].size() * (uint) std::max((int) logf(max_d), 1));
            vv_range_table[t].resize(max_points_in_leaf_cell * max_points_in_leaf_cell);
            vv_range_counts[t].resize(max_points_in_leaf_cell * 2);
        }
    }

    void process_labels(std::vector<std::vector<int>> &v_t_c_labels,
            std::vector<int> &v_labels,
            std::vector<int> &v_cluster_label,
            const uint n_threads, const uint max_clusters) noexcept {
        std::vector<std::vector<int>> v_index_stack(n_threads);
        uint flatten_cnt = 0;
        // Flatten label trees
#pragma omp parallel for reduction(+:flatten_cnt)
        for (uint t = 0; t < n_threads; ++t) {
            for (uint i = 0; i < max_clusters; ++i) {
                if (v_t_c_labels[t][i] == LABEL_CELL) {
                    continue;
                }
                int label = v_t_c_labels[t][i];
                while (v_t_c_labels[t][label] != LABEL_CELL) {
                    label = v_t_c_labels[t][label];
                }
                v_t_c_labels[t][i] = label;
                ++flatten_cnt;
            }
        }
        std::cout << "Flatten cnt: " << flatten_cnt << std::endl;

//        for (uint t = 0; t < n_threads; ++t) {
//            for (uint i = 0; i < max_clusters; ++i) {
//                if (v_t_c_labels[t][i] != LABEL_CELL) {
//                    assert(v_t_c_labels[t][v_t_c_labels[t][i]] == LABEL_CELL);
//                }
//            }
//        }
#pragma omp parallel for
        for (uint i = 0; i < max_clusters; ++i) {
            for (uint t = 0; t < n_threads; ++t) {
                if (v_t_c_labels[t][i] != LABEL_CELL && v_t_c_labels[t][i] < v_labels[i]) {
                    v_labels[i] = v_t_c_labels[t][i];
                }
            }
        }
        std::vector<int> v_cluster_index;
#pragma omp parallel for reduction(vec_merge_int: v_cluster_index)
        for (uint i = 0; i < max_clusters; ++i) {
            if (v_labels[i] == LABEL_CELL) {
                v_cluster_index.push_back(i);
            }
        }
        // TODO sort v_cluster_index
        v_cluster_label.resize(v_cluster_index.size(), LABEL_CELL);

//        #pragma omp parallel for/* reduction(vec_min: v_cluster_label)*/ schedule(dynamic)
        for (uint i = 0; i < max_clusters; ++i) {
            if (v_labels[i] != LABEL_CELL) {
                for (uint t = 0; t < n_threads; ++t) {
                    if (v_t_c_labels[t][i] != LABEL_CELL && v_t_c_labels[t][i] != v_labels[i]) {
                        int label1 = v_labels[i];
                        while (v_labels[label1] != LABEL_CELL) {
                            label1 = v_labels[label1];
                        }
                        int label2 = v_t_c_labels[t][i];
                        while (v_labels[label2] != LABEL_CELL) {
                            label2 = v_labels[label2];
                        }
                        if (label1 != label2) {
                            int index1 = UNASSIGNED, index2 = UNASSIGNED;
                            for (int index = 0; index < v_cluster_index.size(); ++index) {
                                if (v_cluster_index[index] == label1)
                                    index1 = index;
                                if (v_cluster_index[index] == label2)
                                    index2 = index;
                            }
//                            assert(index1 != UNASSIGNED && index2 != UNASSIGNED);
//                            assert(index1 < v_cluster_label.size());
//                            assert(index2 < v_cluster_label.size());
                            while (v_cluster_label[index1] != LABEL_CELL) {
                                index1 = v_cluster_label[index1];
                            }
                            while (v_cluster_label[index2] != LABEL_CELL) {
                                index2 = v_cluster_label[index2];
                            }
                            if (index1 != index2) {
//                                ++cnt;
                                if (index1 < index2) {
                                    v_cluster_label[index1] = index2;
                                } else if (index2 < index1) {
                                    v_cluster_label[index2] = index1;
                                }
                            }
                        }
                    }
                }
            }
        }
//        std::cout << "label pair cnt: " << cnt << std::endl;
    }

    void index_points(std::unique_ptr<float[]> &v_coords, std::unique_ptr<float[]> &v_eps_levels,
            std::unique_ptr<ull[]> &v_dims_mult,
            std::unique_ptr<float[]> &v_min_bounds,
            std::vector<std::vector<uint>> &vv_index_map,
            std::vector<std::vector<uint>> &vv_cell_begin,
            std::vector<std::vector<uint>> &vv_cell_ns,
            std::vector<std::vector<float>> &vv_min_cell_dim,
            std::vector<std::vector<float>> &vv_max_cell_dim,
            std::unique_ptr<uint[]> &v_node_sizes,
            std::unique_ptr<uint[]> &v_node_offsets,
            const uint node_index, const uint max_d, const uint n_threads,
            const uint max_levels) noexcept {
        std::vector<ull> v_value_map;
        std::vector<std::vector<uint>> v_bucket(n_threads);
        std::vector<ull> v_bucket_seperator;
        v_bucket_seperator.reserve(n_threads);
        std::vector<ull> v_bucket_seperator_tmp;
        v_bucket_seperator_tmp.reserve(n_threads * n_threads);
        std::vector<std::vector<std::vector<uint>::iterator>> v_iterator(n_threads);
        uint size = v_node_sizes[node_index];
        for (int l = 0; l < max_levels; ++l) {
            size = index_level_and_get_cells(v_coords, v_min_bounds, vv_index_map, vv_cell_begin,
                    vv_cell_ns[l], v_value_map, v_bucket, v_bucket_seperator, v_bucket_seperator_tmp,
                    v_iterator, size, l, max_d, v_node_offsets[node_index], v_eps_levels[l],
                    &v_dims_mult[l * max_d], n_threads);
            calculate_level_cell_bounds(&v_coords[v_node_offsets[node_index] * max_d], vv_cell_begin[l], vv_cell_ns[l],
                    vv_index_map[l], vv_min_cell_dim, vv_max_cell_dim, max_d, l);
        }
    }

#ifdef MPI_ON

    void process_nodes_nearest_neighbour(std::unique_ptr<float[]> &v_coords,
            std::unique_ptr<uint[]> &v_node_offset,
            std::vector<std::vector<std::vector<uint>>> &vvv_index_map,
            std::vector<std::vector<std::vector<uint>>> &vvv_cell_begin,
            std::vector<std::vector<std::vector<uint>>> &vvv_cell_ns,
            std::vector<std::vector<cell_meta_3>> &vv_stacks3,
            std::vector<std::vector<bool>> &vv_range_table,
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

        // TODO dynamic level and parallelize
        uint level = max_levels - 5;
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
        std::vector<uint> v_sink_cells2;
        std::vector<uint> v_sink_points2;
        mpi_sum_vectors(vv_point_nn, v_payload, v_sink_points2, v_sink_points, n_nodes,
                MPI_UNSIGNED, true);
        mpi_sum_vectors(vv_leaf_cell_nn, v_payload, v_sink_cells2, v_sink_cells, n_nodes,
                MPI_UNSIGNED, true);
    }

#endif

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
//                    &v_dims_mult[0], max_d, v_eps_levels[0]);
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

    result start(const uint m, const float e, const uint n_threads, const std::string &in_file,
            const uint node_index, const uint n_nodes) noexcept {
        // *** READ DATA ***
        auto time_start = std::chrono::high_resolution_clock::now();
        omp_set_num_threads(n_threads);
        uint n, max_d;
        if (node_index == 0)
            std::cout << "Total of " << (n_threads * n_nodes) << " cores used on " << n_nodes << " nodes." << std::endl;
        std::unique_ptr<float[]> v_coords;
        uint total_samples = process_input(in_file, v_coords, n, max_d, n_nodes, node_index);
        auto time_data_read = std::chrono::high_resolution_clock::now();
        if (!g_quiet && node_index == 0) {
            std::cout << "Input Read: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(time_data_read - time_start).count()
                      << " milliseconds\n";
        }
        if (node_index == 0)
            std::cout << "Found " << n << " points in " << max_d << " dimensions" << " and read " << n <<
                      " of " << total_samples << " samples." << std::endl;
//        const auto e_inner = (e / 2);
        const auto e_inner = (e / 1.8);
        const float e2 = e*e;
        // *** INITIALIZE ***
        auto v_node_sizes = std::make_unique<uint[]>(n_nodes);
        auto v_node_offsets = std::make_unique<uint[]>(n_nodes);
        deep_io::get_blocks_meta(v_node_sizes, v_node_offsets, total_samples, n_nodes);
//        if (node_index == 1) {
//            print_array("node sizes: ", &v_node_sizes[0], n_nodes);
//            print_array("node offsets: ", &v_node_offsets[0], n_nodes);
//        }
        auto v_min_bounds = std::make_unique<float[]>(max_d);
        auto v_max_bounds = std::make_unique<float[]>(max_d);
        const int max_levels = determine_data_boundaries(v_coords, v_min_bounds, v_max_bounds, n,
                v_node_offsets[node_index], max_d, e_inner);
        auto v_eps_levels = std::make_unique<float[]>(max_levels);
        auto v_dims_mult = std::make_unique<ull[]>(max_levels * max_d);
        #pragma omp parallel for
        for (uint l = 0; l < max_levels; l++) {
            v_eps_levels[l] = (e_inner * powf(2, l));
            calc_dims_mult(&v_dims_mult[l*max_d], max_d, v_min_bounds, v_max_bounds, v_eps_levels[l]);
        }
        auto time_initialized = std::chrono::high_resolution_clock::now();
        if (!g_quiet && node_index == 0) {
            std::cout << "Initialize: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(time_initialized - time_data_read).count()
                      << " milliseconds\n";
        }
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
        auto time_index_start = std::chrono::high_resolution_clock::now();
        // *** INDEX POINTS ***
        std::vector<std::vector<std::vector<uint>>> vvv_index_map(n_nodes);
        std::vector<std::vector<std::vector<uint>>> vvv_cell_begin(n_nodes);
        std::vector<std::vector<std::vector<uint>>> vvv_cell_ns(n_nodes);
        std::vector<std::vector<std::vector<float>>> vvv_min_cell_dim(n_nodes);
        std::vector<std::vector<std::vector<float>>> vvv_max_cell_dim(n_nodes);
        for (uint n = 0; n < n_nodes; ++n) {
            vvv_index_map[n].resize(max_levels);
            vvv_cell_begin[n].resize(max_levels);
            vvv_cell_ns[n].resize(max_levels);
            vvv_min_cell_dim[n].resize(max_levels);
            vvv_max_cell_dim[n].resize(max_levels);
        }
        index_points(v_coords, v_eps_levels, v_dims_mult, v_min_bounds, vvv_index_map[node_index],
                vvv_cell_begin[node_index],vvv_cell_ns[node_index], vvv_min_cell_dim[node_index],
                vvv_max_cell_dim[node_index], v_node_sizes,v_node_offsets, node_index, max_d,
                n_threads, max_levels);
        auto time_index_end = std::chrono::high_resolution_clock::now();
        if (!g_quiet && node_index == 0) {
            std::cout << "Index and Bounds: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(time_index_end - time_index_start).count()
                      << " milliseconds\n";
        }
        std::vector<cell_meta_3> v_tasks;
        determine_tasks(vvv_index_map[node_index], vvv_cell_begin[node_index], vvv_cell_ns[node_index], v_tasks,
                vvv_min_cell_dim[node_index], vvv_max_cell_dim[node_index], max_levels, max_d, e, n_threads);

        std::vector<std::vector<cell_meta_3>> vv_stacks3(n_threads);
        std::vector<std::vector<bool>> vv_range_table(n_threads);
        std::vector<std::vector<uint>> vv_range_counts(n_threads);
        std::vector<std::vector<uint>> vv_leaf_cell_nn(n_nodes);
        std::vector<std::vector<uint>> vv_point_nn(n_nodes);
        std::vector<std::vector<uint8_t>> vv_cell_type(n_nodes);
        std::vector<std::vector<uint8_t>> vv_is_core(n_nodes);

        auto time_tasks = std::chrono::high_resolution_clock::now();
        if (!g_quiet && node_index == 0) {
            std::cout << "Tasks and Memory Initialize: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(time_tasks - time_index_end).count()
                      << " milliseconds\n";
        }

#ifdef MPI_ON
        auto time_mpi_merge1 = std::chrono::high_resolution_clock::now();
        if (n_nodes > 1) {
            mpi_merge_cell_trees(vvv_index_map, vvv_cell_begin, vvv_cell_ns, vvv_min_cell_dim,
                    vvv_max_cell_dim, node_index, n_nodes, max_levels, max_d);
//        MPI_Barrier(MPI_COMM_WORLD);
        }
        auto time_mpi_merge2 = std::chrono::high_resolution_clock::now();
        if (!g_quiet && node_index == 0) {
            std::cout << "MPI Grid-cell Tree Merge: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(time_mpi_merge2 - time_mpi_merge1).count()
                      << " milliseconds\n";
        }
#endif
        auto time_stacks_start = std::chrono::high_resolution_clock::now();
        for (uint n = 0; n < n_nodes; ++n) {
            vv_leaf_cell_nn[n].resize(vvv_cell_ns[n][0].size(), 0);
            vv_point_nn[n].resize(v_node_sizes[n], 0);
            vv_cell_type[n].resize(vvv_cell_ns[n][0].size(), NC);
            vv_is_core[n].resize(v_node_sizes[n], 0);
        }
        init_stacks(vvv_cell_ns, vv_leaf_cell_nn, vv_stacks3, vv_range_table, vv_range_counts, max_d,
                n_nodes, n_threads, node_index);
        auto time_stacks_end = std::chrono::high_resolution_clock::now();
        if (!g_quiet && node_index == 0) {
            std::cout << "Stacks and Counters: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(
                              time_stacks_end - time_stacks_start).count()
                      << " milliseconds\n";
        }
        std::vector<int> v_c_index(v_node_sizes[node_index], UNASSIGNED);
        std::vector<std::vector<int>> v_t_c_labels(n_threads);
        #pragma omp parallel for schedule(dynamic)
        for (uint i = 0; i < v_tasks.size(); ++i) {
            uint tid = omp_get_thread_num();
            vv_stacks3[tid].push_back(v_tasks[i]);
            process_pair_stack(&v_coords[v_node_offsets[node_index]*max_d], v_t_c_labels[tid], v_c_index,
                    vvv_index_map[node_index], vvv_cell_begin[node_index], vvv_cell_ns[node_index],
                    vvv_min_cell_dim[node_index], vvv_max_cell_dim[node_index], vv_leaf_cell_nn[node_index],
                    vv_point_nn[node_index], vv_stacks3[tid],vv_range_table[tid], vv_range_counts[tid],
                    vv_cell_type[node_index], vv_is_core[node_index], m, max_d, e, e2, true);
        }
        auto time_local_tree = std::chrono::high_resolution_clock::now();
        if (!g_quiet && node_index == 0) {
            std::cout << "Local Tree NN: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(
                              time_local_tree - time_stacks_end).count()
                      << " milliseconds\n";
        }
#ifdef MPI_ON
        auto time_mpi_trees_start = std::chrono::high_resolution_clock::now();
        if (n_nodes > 1) {
            process_nodes_nearest_neighbour(v_coords, v_node_offsets, vvv_index_map, vvv_cell_begin,
                    vvv_cell_ns, vv_stacks3, vv_range_table, vv_range_counts, vv_leaf_cell_nn,
                    vv_point_nn, vv_cell_type, vv_is_core, vvv_min_cell_dim, vvv_max_cell_dim,
                    n_nodes, max_d, m, e, max_levels, node_index);
        }
        auto time_mpi_trees_end = std::chrono::high_resolution_clock::now();
        if (!g_quiet && node_index == 0) {
            std::cout << "Node Trees NN: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(
                              time_mpi_trees_end - time_mpi_trees_start).count()
                      << " milliseconds\n";
        }
#endif
        auto time_infer_start = std::chrono::high_resolution_clock::now();
        const uint max_clusters = infer_types_and_init_clusters_omp(vvv_index_map, vvv_cell_begin,
                vvv_cell_ns, vv_leaf_cell_nn, vv_point_nn, vv_cell_type, vv_is_core,
                v_c_index, m, n_threads, n_nodes, node_index);
        if (node_index == 0)
            std::cout << "Maximum number of clusters: " << max_clusters << std::endl;
        auto time_infer_end = std::chrono::high_resolution_clock::now();
        if (!g_quiet && node_index == 0) {
            std::cout << "Infer Cores and Init Clusters: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(
                              time_infer_end - time_infer_start).count()
                      << " milliseconds\n";
        }

        /*
        #pragma omp parallel for
        for (uint t = 0; t < n_threads; ++t) {
            v_t_c_labels[t].resize(max_clusters, LABEL_CELL);
        }
        #pragma omp parallel for schedule(dynamic)
        for (uint i = 0; i < v_tasks.size(); ++i) {
            uint tid = omp_get_thread_num();
            vv_stacks3[tid].push_back(v_tasks[i]);
            process_pair_stack(&v_coords[v_node_offsets[node_index]], v_t_c_labels[tid],v_c_index,
                    vvv_index_map[node_index], vvv_cell_begin[node_index], vvv_cell_ns[node_index],
                    vvv_min_cell_dim[node_index], vvv_max_cell_dim[node_index],
                    vv_leaf_cell_nn[node_index], vv_point_nn[node_index], vv_stacks3[tid],vv_range_table[tid],
                    vv_range_counts[tid], vv_cell_type[node_index], vv_is_core[node_index], m, max_d, e, e2, false);
        }
        auto time9 = std::chrono::high_resolution_clock::now();
        if (!g_quiet && node_index == 0) {
            std::cout << "Local Tree Labels: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(
                              time9 - time8).count()
                      << " milliseconds\n";
        }
         */
        std::vector<int> v_labels(max_clusters, LABEL_CELL);
        std::vector<int> v_cluster_label;
        /*
        process_labels(v_t_c_labels, v_labels, v_cluster_label, n_threads, max_clusters);
        auto time10 = std::chrono::high_resolution_clock::now();
        if (!g_quiet && node_index == 0) {
            std::cout << "Parse Labels: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(
                              time10 - time9).count()
                      << " milliseconds\n";
        }
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
        return calculate_results(vv_is_core, v_cluster_label, v_c_index, total_samples);
    }

}