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
#include <sstream>
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
//#define MPI_ON
#ifdef MPI_ON
#include <mpi.h>
#endif
#include "nextdbscan.h"
#include "deep_io.h"

namespace nextdbscan {

    static const int UNASSIGNED = -1;
    static const int TYPE_NC = 0;
    static const int TYPE_AC = 1;
    static const int TYPE_SC = 2;

    static const uint8_t NC = 0;
    static const uint8_t AC = 1;
    static const uint8_t SC = 2;

    static const int RET_NONE = -100;
    static const int RET_FULL = 100;
    static const int RET_PARTIAL = 200;

    typedef unsigned long long ull;
//    typedef unsigned int ull;
// TODO Detect when this is necessary during indexing
//    typedef unsigned __int128 ull;
    typedef unsigned int uint;

    static bool g_quiet = false;

    class struct_label {
    public:
        int label;
        struct_label *label_p;

        struct_label() noexcept {
            label = UNASSIGNED;
            label_p = nullptr;
        }
    };

    struct cell_meta_3 {
        uint l, c1, c2;

        cell_meta_3(uint l, uint c1, uint c2) : l(l), c1(c1), c2(c2) {}
    };

    struct cell_meta_5 {
        uint l, c1, c2, t1, t2;

        cell_meta_5(uint l, uint c1, uint c2, uint t1, uint t2) : l(l), c1(c1), c2(c2), t1(t1), t2(t2) {}
    };

    struct_label *get_label(struct_label *p) noexcept {
        struct_label *p_origin = p;
        while (p->label_p != nullptr) {
            p = p->label_p;
        }
        if (p_origin->label_p != nullptr && p_origin->label_p != p) {
            p_origin->label_p = p;
        }
        return p;
    }

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

    inline void calc_dims_mult(ull *dims_mult, const uint max_d, const std::unique_ptr<float[]> &min_bounds,
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

    inline void set_lower_label(struct_label *c1_label, struct_label *c2_label) noexcept {
        if (c1_label->label < c2_label->label) {
            c2_label->label_p = c1_label;
        } else {
            c1_label->label_p = c2_label;
        }
    }

    /*
    void print_cell_types(type_vector &cell_types, std::vector<uint> &v_no_of_cells) {
        uint types[3]{0};
        for (uint i = 0; i < v_no_of_cells[0]; i++) {
            if (cell_types[i] == TYPE_NC)
                ++types[TYPE_NC];
            else if (cell_types[i] == TYPE_SC)
                ++types[TYPE_SC];
            else if (cell_types[i] == TYPE_AC)
                ++types[TYPE_AC];
        }
        std::cout << "TYPE NC: " << types[TYPE_NC] << std::endl;
        std::cout << "TYPE SC: " << types[TYPE_SC] << std::endl;
        std::cout << "TYPE AC: " << types[TYPE_AC] << std::endl;
    }
     */

    inline void update_to_ac(std::vector<uint> &v_index_maps, std::vector<uint> &v_cell_ns,
            std::vector<uint> &v_cell_begin, std::vector<bool> &is_core, std::vector<uint8_t> &v_types, const uint c) {
        v_types[c] = AC;
        uint begin = v_cell_begin[c];
        for (uint j = 0; j < v_cell_ns[c]; ++j) {
            is_core[v_index_maps[begin + j]] = true;
        }
    }

    void update_type(std::vector<uint> &v_index_maps, std::vector<uint> &v_cell_ns,
            std::vector<uint> &v_cell_begin, std::vector<uint> &v_cell_nps, std::vector<uint> &v_point_nps,
            std::vector<bool> &is_core, std::vector<uint8_t> &v_types, const uint c, const uint m) {
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
                is_core[p] = true;
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

    bool fill_range_table_mult(std::unique_ptr<float*[]> &v_p_coords,
            std::vector<std::vector<std::vector<uint>>> &vvv_index_map,
            std::vector<std::vector<std::vector<uint>>> &vvv_cell_ns_level, std::vector<bool> &v_range_table,
            const uint c1, const uint begin1, const uint c2, const uint begin2, const uint max_d, const float e2,
            const uint t1, const uint t2) noexcept {
        uint size1 = vvv_cell_ns_level[t1][0][c1];
        uint size2 = vvv_cell_ns_level[t2][0][c2];
        bool all_in_range = true;
        uint index = 0;
        std::fill(v_range_table.begin(), v_range_table.begin()+(size1*size2), false);
        for (uint k1 = 0; k1 < size1; ++k1) {
            uint p1 = vvv_index_map[t1][0][begin1 + k1];
            for (uint k2 = 0; k2 < size2; ++k2, ++index) {
                uint p2 = vvv_index_map[t2][0][begin2 + k2];
                if (dist_leq(&v_p_coords[t1][p1 * max_d], &v_p_coords[t2][p2 * max_d], max_d, e2)) {
                    v_range_table[index] = true;
                } else {
                    all_in_range = false;
                }
            }
        }
        return all_in_range;
    }

    void update_cell_pair_nn_multi(std::vector<std::vector<std::vector<uint>>> &vvv_index_map,
            std::vector<std::vector<std::vector<uint>>> &vvv_cell_ns,
            std::vector<std::vector<uint>> &v_point_nps, std::vector<bool> &v_range_table,
            std::vector<std::vector<bool>> &vv_is_core, const uint c1,
            const uint begin1, const uint c2, const uint begin2, const uint t1, const uint t2, bool &update_1,
            bool &update_2) {
        uint size1 = vvv_cell_ns[t1][0][c1];
        uint size2 = vvv_cell_ns[t2][0][c2];
        uint index = 0;
        for (uint k1 = 0; k1 < size1; ++k1) {
            uint p1 = vvv_index_map[t1][0][begin1 + k1];
            for (uint k2 = 0; k2 < size2; ++k2, ++index) {
                uint p2 = vvv_index_map[t2][0][begin2 + k2];
                if (vv_is_core[t1][p1] && vv_is_core[t2][p2])
                    continue;
                if (v_range_table[index]) {
                    if (!vv_is_core[t1][p1]) {
                        if (!update_1)
                            update_1 = true;
                        #pragma omp atomic
                        ++v_point_nps[t1][p1];
                    }
                    if (!vv_is_core[t2][p2]) {
                        if (!update_2)
                            update_2 = true;
                        #pragma omp atomic
                        ++v_point_nps[t2][p2];
                    }
                }
            }
        }
    }

    bool fill_range_table(float *v_coords, std::vector<uint> &v_index_map_level,
            std::vector<uint> &v_cell_ns_level, std::vector<bool> &v_range_table, const uint c1,
            const uint begin1, const uint c2, const uint begin2, const uint max_d, const float e2) noexcept {
        uint size1 = v_cell_ns_level[c1];
        uint size2 = v_cell_ns_level[c2];
        bool all_in_range = true;
        uint index = 0;
        std::fill(v_range_table.begin(), v_range_table.begin()+(size1*size2), false);
        for (uint k1 = 0; k1 < size1; ++k1) {
            uint p1 = v_index_map_level[begin1 + k1];
            for (uint k2 = 0; k2 < size2; ++k2, ++index) {
                uint p2 = v_index_map_level[begin2 + k2];
                if (dist_leq(&v_coords[p1 * max_d], &v_coords[p2 * max_d], max_d, e2)) {
                    v_range_table[index] = true;
                } else {
                    all_in_range = false;
                }
            }
        }
        return all_in_range;
    }

    void update_cell_pair_nn(std::vector<uint> &v_index_map_level, std::vector<uint> &v_cell_ns_level,
            std::vector<uint> &v_point_nps, std::vector<bool> &v_range_table, std::vector<bool> &v_is_core,
            const uint c1, const uint begin1, const uint c2, const uint begin2, bool &update_1, bool &update_2) {
        uint size1 = v_cell_ns_level[c1];
        uint size2 = v_cell_ns_level[c2];
        uint index = 0;
        for (uint k1 = 0; k1 < size1; ++k1) {
            uint p1 = v_index_map_level[begin1 + k1];
            for (uint k2 = 0; k2 < size2; ++k2, ++index) {
                uint p2 = v_index_map_level[begin2 + k2];
                if (v_is_core[p1] && v_is_core[p2])
                    continue;
                if (v_range_table[index]) {
                    if (!v_is_core[p1]) {
                        if (!update_1)
                            update_1 = true;
                        ++v_point_nps[p1];
                    }
                    if (!v_is_core[p2]) {
                        if (!update_2)
                            update_2 = true;
                        ++v_point_nps[p2];
                    }
                }
            }
        }
    }

    void process_pair_nn(float *v_coords, std::vector<std::vector<uint>> &vv_index_maps,
            std::vector<uint> &v_point_nps, std::vector<std::vector<uint>> &vv_cell_begin,
            std::vector<std::vector<uint>> &vv_cell_ns,
            std::vector<bool> &v_range_table, std::vector<uint> &v_cell_nps, std::vector<bool> &v_is_core,
            std::vector<uint8_t> &v_types, const uint max_d, const float e2, const uint m, const uint l,
            const uint c1, const uint begin1, const uint c2, const uint begin2) {
        bool all_range_check = fill_range_table(v_coords, vv_index_maps[l], vv_cell_ns[l],
                v_range_table, c1, begin1, c2, begin2,
                max_d, e2);
        bool update_1 = false;
        bool update_2 = false;
        if (all_range_check) {
            if (v_types[c1] != AC) {
                v_cell_nps[c1] += vv_cell_ns[0][c2];
                update_1 = true;
            }
            if (v_types[c2] != AC) {
                v_cell_nps[c2] += vv_cell_ns[0][c1];
                update_2 = true;
            }
        } else {
            update_cell_pair_nn(vv_index_maps[l], vv_cell_ns[l], v_point_nps, v_range_table,
                    v_is_core, c1, begin1, c2, begin2, update_1, update_2);
        }
        if (update_1) {
            update_type(vv_index_maps[0], vv_cell_ns[0], vv_cell_begin[0], v_cell_nps,
                    v_point_nps, v_is_core, v_types, c1, m);
        }
        if (update_2) {
            update_type(vv_index_maps[0], vv_cell_ns[0], vv_cell_begin[0], v_cell_nps,
                    v_point_nps, v_is_core, v_types, c2, m);
        }
    }

    void read_input_txt(const std::string &in_file, float *v_points, int max_d) noexcept {
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


    result calculate_output(std::vector<std::vector<bool>> &is_core, std::vector<struct_label> &ps, int n) noexcept {
        result res{0, 0, 0, new std::vector<int>(n)};

        uint sum = 0;

        #pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < is_core.size(); ++i) {
            for (auto is : is_core[i]) {
                if (is) ++sum;
            }
        }
        res.core_count = sum;

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
    }

    uint process_input(const std::string &in_file, std::unique_ptr<float[]> &v_points, uint &n, uint &max_d,
            const uint blocks_no, const uint block_index) {
        std::string s_cmp = ".bin";
        if (in_file.compare(in_file.size() - s_cmp.size(), s_cmp.size(), s_cmp) == 0) {
            char c[in_file.size() + 1];
            strcpy(c, in_file.c_str());
            auto *data = new deep_io(c, blocks_no, block_index);
            int read_bytes = data->load_next_samples(v_points);
//            std::cout << "read data bytes: " << read_bytes << std::endl;
            n = data->sample_read_no;
            max_d = data->feature_no;
            return data->sample_no;
//            std::cout << "Found " << n << " points in " << max_d << " dimensions" << std::endl;
//            std::cout << "block data offset: " << data->block_sample_offset << std::endl;
//            return data->sample_read_no;
//            v_points = &data->features[data->block_sample_offset];
        } else {
            count_lines_and_dimensions(in_file, n, max_d);
            // TODO
            std::cerr << "ERROR Only supports binary" << std::endl;
            exit(-1);
//            v_points.reserve(n * max_d);
//            v_points = new float[n * max_d];
//            read_input_txt(in_file, v_points, max_d);
        }
        return 0;
    }

    void calculate_level_cell_bounds(float *v_coords, std::vector<uint> &v_cell_begins,
            std::vector<uint> &v_cell_ns, std::vector<uint> &v_index_maps, std::vector<std::vector<float>> &vv_min_cell_dims,
            std::vector<std::vector<float>> &vv_max_cell_dims, uint max_d, uint l) noexcept {
        vv_min_cell_dims[l].resize(v_cell_begins.size()*max_d);
        vv_max_cell_dims[l].resize(vv_min_cell_dims[l].size());
        float *coord_min = nullptr, *coord_max = nullptr;

        for (uint i = 0; i < v_cell_begins.size(); i++) {
            uint begin = v_cell_begins[i];
            uint coord_offset = 0;
            if (l == 0) {
                coord_offset = v_index_maps[begin] * max_d;
                coord_min = &v_coords[coord_offset];
                coord_max = &v_coords[coord_offset];
            } else {
                coord_min = &vv_min_cell_dims[l-1][v_index_maps[begin]*max_d];
                coord_max = &vv_max_cell_dims[l-1][v_index_maps[begin]*max_d];
            }
            std::copy(coord_min, coord_min + max_d, &vv_min_cell_dims[l][i*max_d]);
            std::copy(coord_max, coord_max + max_d, &vv_max_cell_dims[l][i*max_d]);

            for (uint j = 1; j < v_cell_ns[i]; j++) {
                uint coord_offset_inner = 0;
                if (l == 0) {
                    coord_offset_inner = v_index_maps[begin+j] * max_d;
                    coord_min = &v_coords[coord_offset_inner];
                    coord_max = &v_coords[coord_offset_inner];
                } else {
                    coord_min = &vv_min_cell_dims[l-1][v_index_maps[begin+j]*max_d];
                    coord_max = &vv_max_cell_dims[l-1][v_index_maps[begin+j]*max_d];
                }
                for (uint d = 0; d < max_d; d++) {
                    if (coord_min[d] < vv_min_cell_dims[l][i*max_d+d]) {
                        vv_min_cell_dims[l][i*max_d+d] = coord_min[d];
                    }
                    if (coord_max[d] > vv_max_cell_dims[l][i*max_d+d]) {
                        vv_max_cell_dims[l][i*max_d+d] = coord_max[d];
                    }
                }
            }
        }
    }

    void print_array(const std::string name, uint *arr, const uint max_d) {
        std::cout << name << ": ";
        for (int i = 0; i < max_d; ++i) {
            std::cout << arr[i] << " ";
        }
        std::cout << std::endl;
    }

    void print_array(const std::string name, int *arr, const uint max_d) {
        std::cout << name << ": ";
        for (int i = 0; i < max_d; ++i) {
            std::cout << arr[i] << " ";
        }
        std::cout << std::endl;
    }

    void print_array(const std::string name, float *arr, const uint max_d) {
        std::cout << name << ": ";
        for (int i = 0; i < max_d; ++i) {
            std::cout << arr[i] << " ";
        }
        std::cout << std::endl;
    }

    int determine_data_boundaries(std::unique_ptr<float[]> &v_coords, std::unique_ptr<float[]> &v_min_bounds,
            std::unique_ptr<float[]> &v_max_bounds, const uint n, const uint node_offset, const uint max_d, const float e_inner) {
        float max_limit = INT32_MIN;
        calc_bounds(v_coords, n, &v_min_bounds[0], &v_max_bounds[0], max_d, node_offset);
#ifdef MPI_ON
        auto v_global_min_bounds = std::make_unique<float[]>(max_d);
        auto v_global_max_bounds = std::make_unique<float[]>(max_d);
        MPI_Allreduce(&v_min_bounds[0], &v_global_min_bounds[0], max_d, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&v_max_bounds[0], &v_global_max_bounds[0], max_d, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
        std::copy(&v_global_min_bounds[0], &v_global_min_bounds[max_d], &v_min_bounds[0]);
        std::copy(&v_global_max_bounds[0], &v_global_max_bounds[max_d], &v_max_bounds[0]);
        v_global_min_bounds.reset();
        v_global_max_bounds.reset();
#endif
        for (uint d = 0; d < max_d; d++) {
            if (v_max_bounds[d] - v_min_bounds[d] > max_limit)
                max_limit = v_max_bounds[d] - v_min_bounds[d];
        }
        return static_cast<int>(ceilf(logf(max_limit / e_inner) / logf(2))) + 1;
    }

    int index_level(const float *v_coords, std::vector<uint> &v_index_map, std::vector<ull> &v_value_map,
            std::vector<std::vector<uint>> &vv_index_lookup, std::vector<std::vector<uint>> &vv_cell_begin,
            std::vector<uint> &v_cell_ns, std::unique_ptr<float[]> &v_min_bounds, const ull *v_dims_mult,
            const int level, const uint max_d, const float eps_level) noexcept {
        std::iota(v_index_map.begin(), v_index_map.end(), 0);
//        assert(v_index_map[0] == 0 && v_index_map[1] == 1);
        for (uint i = 0; i < v_index_map.size(); ++i) {
            int level_mod = 1;
            uint p_index = i;
            while (level - level_mod >= 0) {
                p_index = vv_index_lookup[level-level_mod][vv_cell_begin[level-level_mod][p_index]];
                ++level_mod;
            }
            uint coord_index = p_index*max_d;
            v_value_map[i] = get_cell_index(&v_coords[coord_index], v_min_bounds, v_dims_mult, max_d, eps_level);
        }
        std::sort(v_index_map.begin(), v_index_map.end(), [&] (const auto &i1, const auto &i2) -> bool {
            return v_value_map[i1] < v_value_map[i2];
        });
        uint new_cells = 1;
        ull last_value = v_value_map[v_index_map[0]];
        for (uint i = 1; i < v_value_map.size(); ++i) {
            if (v_value_map[v_index_map[i]] != last_value) {
                last_value = v_value_map[v_index_map[i]];
                ++new_cells;
            }
        }
        vv_cell_begin[level].resize(new_cells);
        v_cell_ns.resize(new_cells);
        vv_cell_begin[level][0] = 0;
        uint cell_cnt = 1;
        last_value = v_value_map[v_index_map[0]];
        for (uint i = 1; i < v_value_map.size(); ++i) {
            if (v_value_map[v_index_map[i]] != last_value) {
                last_value = v_value_map[v_index_map[i]];
                vv_cell_begin[level][cell_cnt] = i;
                v_cell_ns[cell_cnt-1] = vv_cell_begin[level][cell_cnt] - vv_cell_begin[level][cell_cnt-1];
                ++cell_cnt;
            }
        }
//        assert(cell_cnt == new_cells);
        v_cell_ns[cell_cnt-1] = v_value_map.size() - vv_cell_begin[level][cell_cnt-1];
//        assert(cell_cnt == v_cell_ns.size());
        uint sum = 0;
        for (auto &val : v_cell_ns) {
            sum += val;
        }
//        assert(sum == v_index_map.size());
        return new_cells;
    }

    void process_cell_tree_level(float *v_coords, std::vector<std::vector<uint>> &vv_cell_begins,
            std::vector<std::vector<uint>> &vv_cell_ns, std::vector<std::vector<uint>> &vv_index_maps,
            std::vector<uint> &v_leaf_cell_nns, std::vector<uint8_t > &v_cell_types, std::vector<bool> &v_is_core,
            std::vector<uint> &v_point_nns, std::vector<cell_meta_3> &stack3, std::vector<bool> &v_range_table,
            std::vector<std::vector<float>> &vv_min_cell_dims, std::vector<std::vector<float>> &vv_max_cell_dims,
            uint level, const uint m, const uint max_d, const float e, const float e2, const bool is_nn) noexcept {
        if (level == 0) {
            for (uint i = 0; i < vv_cell_ns[0].size(); ++i) {
                uint begin = vv_cell_begins[0][i];
                uint size = vv_cell_ns[0][i];
                v_leaf_cell_nns[i] = size;
                if (size >= m) {
                    v_cell_types[i] = AC;
                    for (uint j = 0; j < size; ++j) {
                        uint index = vv_index_maps[0][begin + j];
                        v_is_core[index] = true;
                    }
                }
            }
            return;
        }
        for (uint i = 0; i < vv_cell_begins[level].size(); ++i) {
            uint begin = vv_cell_begins[level][i];
            for (uint c1 = 0; c1 < vv_cell_ns[level][i]; ++c1) {
                uint c1_index = vv_index_maps[level][begin + c1];
                for (uint c2 = c1 + 1; c2 < vv_cell_ns[level][i]; ++c2) {
                    stack3.emplace_back(level-1, c1_index, vv_index_maps[level][begin+c2]);
                }
            }
            while (!stack3.empty()) {
                uint l = stack3.back().l;
                uint c1 = stack3.back().c1;
                uint c2 = stack3.back().c2;
                stack3.pop_back();
                if (!is_in_reach(&vv_min_cell_dims[l][c1 * max_d], &vv_max_cell_dims[l][c1 * max_d],
                        &vv_min_cell_dims[l][c2 * max_d], &vv_max_cell_dims[l][c2 * max_d], max_d, e)) {
                    continue;
                }
                uint begin1 = vv_cell_begins[l][c1];
                uint begin2 = vv_cell_begins[l][c2];
                if (l == 0) {
                    if (is_nn) {
                        if (v_cell_types[c1] != AC || v_cell_types[c2] != AC) {
                            process_pair_nn(v_coords, vv_index_maps, v_point_nns, vv_cell_begins, vv_cell_ns,
                                    v_range_table, v_leaf_cell_nns, v_is_core, v_cell_types, max_d, e2, m, l,
                                    c1, begin1, c2, begin2);
                        }
                    }

                } else {
                    for (uint k1 = 0; k1 < vv_cell_ns[l][c1]; ++k1) {
                        uint c1_next = vv_index_maps[l][begin1 + k1];
                        for (uint k2 = 0; k2 < vv_cell_ns[l][c2]; ++k2) {
                            uint c2_next = vv_index_maps[l][begin2 + k2];
                            stack3.emplace_back(l - 1, c1_next, c2_next);
                        }
                    }
                }
            }
        }
    }

    void process_cell_tree_pairs(std::unique_ptr<float*[]> &v_p_coords,
            std::vector<std::vector<std::vector<uint>>> &vvv_index_maps,
            std::vector<std::vector<std::vector<ull>>> &vvv_value_maps,
            std::vector<std::vector<std::vector<uint>>> &vvv_cell_begins,
            std::vector<std::vector<std::vector<float>>> &vvv_min_cell_dims,
            std::vector<std::vector<std::vector<float>>> &vvv_max_cell_dims,
            std::vector<std::vector<uint>> &vv_leaf_cell_nns, std::vector<std::vector<uint8_t>> &vv_cell_types,
            std::vector<bool> v_range_table,
            std::vector<std::vector<bool>> &vv_is_core, std::vector<std::vector<uint>> &vv_point_nns,
            std::vector<std::vector<std::vector<uint>>> &vvv_cell_ns, std::vector<cell_meta_5> &stack5,
            const uint max_d, const uint m, const float e, const float e2, const bool is_nn) noexcept {
        while (!stack5.empty()) {
            uint l = stack5.back().l;
            uint t1 = stack5.back().t1;
            uint c1 = stack5.back().c1;
            uint t2 = stack5.back().t2;
            uint c2 = stack5.back().c2;
//            std::cout << l << " : " << t1 << " : " << c1 << " : " << " : " << t2 << " : " << c2 << std::endl;
            stack5.pop_back();
            ull c1_val = vvv_value_maps[t1][l][vvv_index_maps[t1][l][c1]];
            ull c2_val = vvv_value_maps[t2][l][vvv_index_maps[t2][l][c2]];
            if (c1_val != c2_val && !is_in_reach(&vvv_min_cell_dims[t1][l][c1*max_d],
                    &vvv_max_cell_dims[t1][l][c1*max_d], &vvv_min_cell_dims[t2][l][c2*max_d],
                    &vvv_max_cell_dims[t2][l][c2*max_d], max_d, e)) {
                continue;
            }
            uint begin1 = vvv_cell_begins[t1][l][c1];
            uint begin2 = vvv_cell_begins[t2][l][c2];
            if (l == 0) {
                if (is_nn) {
                    if (vv_cell_types[t1][c1] != AC || vv_cell_types[t2][c2] != AC) {
//                        bool all_range_check = c1_val == c2_val;
//                        if (!all_range_check) {
//                            all_range_check = fill_range_table_mult(v_p_coords,
//                                    vvv_index_maps, vvv_cell_ns,v_range_table, c1, begin1, c2, begin2, max_d,
//                                    e2, t1, t2);
//                        }
                        bool all_range_check = fill_range_table_mult(v_p_coords,
                                vvv_index_maps, vvv_cell_ns,v_range_table, c1, begin1, c2, begin2, max_d,
                                e2, t1, t2);
//                        bool all_range_check = false;
//                        if (c1_val == c2_val) {
//                            assert(all_range_check);
//                        }
//                        if (all_range_check)
//                            assert(c1_val == c2_val);

                        bool update_1 = false;
                        bool update_2 = false;
                        if (all_range_check) {
                            if (vv_cell_types[t1][c1] != AC) {
                                #pragma omp atomic
                                vv_leaf_cell_nns[t1][c1] += vvv_cell_ns[t2][0][c2];
                                update_1 = true;
                            }
                            if (vv_cell_types[t2][c2] != AC) {
                                #pragma omp atomic
                                vv_leaf_cell_nns[t2][c2] += vvv_cell_ns[t1][0][c1];
                                update_2 = true;
                            }
                        } else {
                            update_cell_pair_nn_multi(vvv_index_maps, vvv_cell_ns, vv_point_nns, v_range_table,
                                    vv_is_core, c1, begin1, c2, begin2, t1, t2, update_1, update_2);
                        }
                        if (update_1) {
                            update_type(vvv_index_maps[t1][0], vvv_cell_ns[t1][0], vvv_cell_begins[t1][0],
                                    vv_leaf_cell_nns[t1], vv_point_nns[t1], vv_is_core[t1], vv_cell_types[t1],
                                    c1, m);
                        }
                        if (update_2) {
                            update_type(vvv_index_maps[t2][0], vvv_cell_ns[t2][0], vvv_cell_begins[t2][0],
                                    vv_leaf_cell_nns[t2], vv_point_nns[t2], vv_is_core[t2], vv_cell_types[t2],
                                    c2, m);
                        }
                    }
                }
            } else {
                for (uint k1 = 0; k1 < vvv_cell_ns[t1][l][c1]; ++k1) {
                    uint c1_next = vvv_index_maps[t1][l][begin1 + k1];
                    for (uint j = 0; j < vvv_cell_ns[t2][l][c2]; ++j) {
                        uint c2_next = vvv_index_maps[t2][l][begin2 + j];
                        stack5.emplace_back(l-1, c1_next, c2_next, t1, t2);
                    }
                }
            }
        }
    }

#ifdef MPI_ON
    template <class T>
    void mpi_sum_tree(std::vector<std::vector<T>> &vv_partial_cell_tree, const int n_cores,
            const int max_levels, const int mpi_size, const int n_threads, const int mpi_index,
            MPI_Datatype send_type, const bool is_verbose) {
        int elems_to_send = 0;
        int t_sizes[n_cores];
        for (int c = 0; c < n_cores; ++c) {
            t_sizes[c] = 0;
            elems_to_send += vv_partial_cell_tree[c].size();
            t_sizes[c] += vv_partial_cell_tree[c].size();
        }
        int m_sizes[mpi_size];
        int m_offsets[mpi_size];
        m_offsets[0] = 0;
        int core_cnt = 0;
        for (int i = 0; i < mpi_size; ++i) {
            m_sizes[i] = 0;
            for (int t = 0; t < n_threads; ++t) {
                m_sizes[i] += t_sizes[core_cnt++];
            }
            if (i > 0) {
                m_offsets[i] = m_offsets[i-1] + m_sizes[i-1];
            }
        }
        std::vector<T> v_payload(elems_to_send, -1);
        std::vector<T> v_sink(elems_to_send, -1);
        int index = 0;
        for (int t = 0; t < n_cores; ++t) {
            for (auto &val : vv_partial_cell_tree[t]) {
//                if (t / n_threads == mpi_index) {
                    v_payload[index] = val;
//                }
                ++index;
            }
        }
//        assert(index == elems_to_send);
        if (is_verbose) {
            std::cout << "Transmitting " << elems_to_send << " elements." << std::endl;
            print_array("mpi block sizes: ", m_sizes, mpi_size);
            print_array("mpi block offsets: ", m_offsets, mpi_size);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(&v_payload[0], &v_sink[0], elems_to_send, send_type, MPI_SUM, MPI_COMM_WORLD);
        index = 0;
        for (int t = 0; t < n_cores; ++t) {
            for (int i = 0; i < vv_partial_cell_tree[t].size(); ++i) {
//                assert(v_sink[index] != (T)-1);
                vv_partial_cell_tree[t][i] = v_sink[index++];
            }
        }
    }

    template <class T>
    void mpi_gather_cell_tree(std::vector<std::vector<std::vector<T>>> &vvv_cell_tree, const int n_cores,
            const int max_levels, const int mpi_size, const int n_threads, const int mpi_index,
            MPI_Datatype send_type, const bool is_verbose) {
        int elems_to_send = 0;
        int t_sizes[n_cores];
        for (int c = 0; c < n_cores; ++c) {
            t_sizes[c] = 0;
            for (int l = 0; l < max_levels; ++l) {
                elems_to_send += vvv_cell_tree[c][l].size();
                t_sizes[c] += vvv_cell_tree[c][l].size();
            }
        }
        int m_sizes[mpi_size];
        int m_offsets[mpi_size];
        m_offsets[0] = 0;
        int core_cnt = 0;
        for (int i = 0; i < mpi_size; ++i) {
            m_sizes[i] = 0;
            for (int t = 0; t < n_threads; ++t) {
                m_sizes[i] += t_sizes[core_cnt++];
            }
            if (i > 0) {
                m_offsets[i] = m_offsets[i-1] + m_sizes[i-1];
            }
        }
//        const int check = -99999999;
        std::vector<T> v_payload(elems_to_send);
        int index = 0;
        for (int t = 0; t < n_cores; ++t) {
            for (int l = 0; l < max_levels; ++l) {
                for (auto &val : vvv_cell_tree[t][l]) {
                    if (t / n_threads == mpi_index) {
//                        assert(val != check);
                        v_payload[index] = val;
                    }
                    ++index;
                }
            }
        }
        if (is_verbose) {
            std::cout << "Transmitting " << elems_to_send << " elements." << std::endl;
            print_array("mpi block sizes: ", m_sizes, mpi_size);
            print_array("mpi block offsets: ", m_offsets, mpi_size);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &v_payload[0], m_sizes,
                m_offsets, send_type, MPI_COMM_WORLD);
        index = 0;
        #pragma omp parallel for
        for (int t = 0; t < n_cores; ++t) {
            for (int l = 0; l < max_levels; ++l) {
                for (int i = 0; i < vvv_cell_tree[t][l].size(); ++i) {
//                    assert(v_payload[index] != (T)check);
                    vvv_cell_tree[t][l][i] = v_payload[index++];
                }
            }
        }
    }

    void mpi_cell_trees_merge(std::vector<std::vector<std::vector<uint>>> &vvv_index_maps,
            std::vector<std::vector<std::vector<ull>>> &vvv_value_maps,
            std::vector<std::vector<std::vector<uint>>> &vvv_cell_begins,
            std::vector<std::vector<std::vector<uint>>> &vvv_cell_ns,
            std::vector<std::vector<std::vector<float>>> &vvv_min_cell_dims,
            std::vector<std::vector<std::vector<float>>> &vvv_max_cell_dims,
            const int mpi_index, const int mpi_size, const uint n_threads, const int n_cores, const int max_levels,
            const uint max_d) {
        int total_levels = n_cores * max_levels;
        int core_level_elems[total_levels];
        std::fill(core_level_elems, core_level_elems + total_levels, 0);
        uint t_offset = mpi_index * n_threads;
        uint index = t_offset * max_levels;
        for (uint t = t_offset; t < (n_threads + t_offset); ++t) {
            for (uint l = 0; l < max_levels; ++l, ++index) {
//                if (vvv_index_maps[t][l].empty())
//                    std::cout << "mpi_index: " << mpi_index << " t: " << t << " level: " << l << " is empty!" << std::endl;
//                assert(!vvv_index_maps[t][l].empty());
                core_level_elems[index] += vvv_index_maps[t][l].size();
            }
        }

//        print_array("elem cnts before: ", core_level_elems, total_levels);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, core_level_elems, total_levels / mpi_size,
                MPI_INT, MPI_COMM_WORLD);
//        if (mpi_index == 0)
//            print_array("elem cnts after: ", core_level_elems, total_levels);
        index = 0;
        for (uint t = 0; t < n_cores; ++t) {
            for (uint l = 0; l < max_levels; ++l, ++index) {
                vvv_index_maps[t][l].resize(core_level_elems[index]);
                vvv_value_maps[t][l].resize(core_level_elems[index]);
                if (l > 0) {
                    vvv_cell_begins[t][l-1].resize(core_level_elems[index]);
                    vvv_cell_ns[t][l-1].resize(core_level_elems[index]);
                    vvv_min_cell_dims[t][l-1].resize(core_level_elems[index]*max_d);
                    vvv_max_cell_dims[t][l-1].resize(core_level_elems[index]*max_d);
                }
            }
            vvv_cell_begins[t][max_levels-1].resize(1);
            vvv_cell_ns[t][max_levels-1].resize(1);
            vvv_min_cell_dims[t][max_levels-1].resize(max_d);
            vvv_max_cell_dims[t][max_levels-1].resize(max_d);
        }

        mpi_gather_cell_tree(vvv_index_maps, n_cores, max_levels, mpi_size, n_threads, mpi_index, MPI_UNSIGNED, false);
        mpi_gather_cell_tree(vvv_value_maps, n_cores, max_levels, mpi_size, n_threads, mpi_index, MPI_UNSIGNED_LONG_LONG, false);
        mpi_gather_cell_tree(vvv_cell_begins, n_cores, max_levels, mpi_size, n_threads, mpi_index, MPI_UNSIGNED, false);
        mpi_gather_cell_tree(vvv_cell_ns, n_cores, max_levels, mpi_size, n_threads, mpi_index, MPI_UNSIGNED, false);
        mpi_gather_cell_tree(vvv_min_cell_dims, n_cores, max_levels, mpi_size, n_threads, mpi_index, MPI_FLOAT, false);
        mpi_gather_cell_tree(vvv_max_cell_dims, n_cores, max_levels, mpi_size, n_threads, mpi_index, MPI_FLOAT, false);
    }
#endif

    result start(const uint m, const float e, const uint n_threads, const std::string &in_file,
            const uint node_index, const uint nodes_no) noexcept {
        auto time1 = std::chrono::high_resolution_clock::now();
        omp_set_num_threads(n_threads);
        uint n, max_d;
        uint n_cores = n_threads * nodes_no;
        if (node_index == 0)
            std::cout << "Total of " << n_cores << " cores used on " << nodes_no << " nodes." << std::endl;
        std::unique_ptr<float[]> v_coords;
        uint total_samples = process_input(in_file, v_coords, n, max_d, nodes_no, node_index);
        auto time2 = std::chrono::high_resolution_clock::now();
        if (!g_quiet && node_index == 0) {
            std::cout << "Input read: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1).count()
                      << " milliseconds\n";
        }
        if (node_index == 0)
            std::cout << "Found " << n << " points in " << max_d << " dimensions" << " and read " << n <<
                " of " << total_samples << " samples." << std::endl;
        const auto e_inner = (e / 2);
        const auto e2 = e * e;

        auto v_node_sizes = std::make_unique<uint[]>(nodes_no);
        auto v_node_offsets = std::make_unique<uint[]>(nodes_no);
        deep_io::get_blocks_meta(v_node_sizes, v_node_offsets, total_samples, nodes_no);

//        std::cout << "node sizes: " << v_node_sizes[0] << " " << v_node_sizes[1] << std::endl;

        auto v_min_bounds = std::make_unique<float[]>(max_d);
        auto v_max_bounds = std::make_unique<float[]>(max_d);
        const int max_levels = determine_data_boundaries(v_coords, v_min_bounds, v_max_bounds, n,
                v_node_offsets[node_index], max_d, e_inner);
        auto v_eps_levels = std::make_unique<float[]>(max_levels);
        auto v_dims_mult = std::make_unique<ull[]>(max_levels * max_d);
        for (int l = 0; l < max_levels; l++) {
            v_eps_levels[l] = (e_inner * powf(2, l));
            calc_dims_mult(&v_dims_mult[l*max_d], max_d, v_min_bounds, v_max_bounds, v_eps_levels[l]);
        }

        std::vector<uint> v_coord_index_map(v_node_sizes[node_index]);
        std::iota(v_coord_index_map.begin(), v_coord_index_map.end(), 0);

        auto *p_v_coords = &v_coords[v_node_offsets[node_index]*max_d];
        std::vector<float> v_coords_tmp(v_node_sizes[node_index] * max_d);
        if (n_cores > 1) {
            /*
            uint v_insert_index[n_threads];
            std::fill(v_insert_index, v_insert_index+n_threads, 0);
            uint insert_size = (v_coord_index_map.size() / n_threads) / n_threads;
            std::cout << "Insert size: " << insert_size << std::endl;
            uint rest = 0;
            #pragma omp parallel
            {
                uint tid = omp_get_thread_num();
                uint size = deep_io::get_block_size(tid, v_coord_index_map.size(), n_threads);
                uint offset = deep_io::get_block_start_offset(tid, v_coord_index_map.size(), n_threads);
                #pragma omp critical
                std::cout << "t: " << tid << " size: " << size << " offset: " << offset << std::endl;
                std::sort(std::next(v_coord_index_map.begin(), offset), std::next(v_coord_index_map.begin(),
                        offset+size),[&](const auto &i1, const auto &i2) -> bool {
                    return p_v_coords[i1 * max_d] < p_v_coords[i2 * max_d];
                });
                uint work_cnt = 0;
                uint work_t = tid;
                uint mult = 0;
//                std::cout << "CHECKPOINT #1" << std::endl;
                do {
                    uint dim_offset = ((work_t*n_threads*insert_size)+(mult*insert_size))*max_d;
                    for (uint i = 0; i < insert_size; ++i) {
                        for (uint d = 0; d < max_d; ++d) {
                            assert(dim_offset < v_coords_tmp.size());
                            v_coords_tmp[dim_offset++] = p_v_coords[v_coord_index_map[work_cnt+offset] + d];
                        }
                        ++work_cnt;
                    }

//                    std::copy(std::next(v_coord_index_map.begin(), offset+work_offset), )
//                    uint off =
//                    v_coords_tmp[dim_offset++] = p_v_coords[(v_coord_index_map[i] * max_d) + ];
//                    #pragma omp barrier
//                    std::copy(std::next(v_coord_index_map.begin(), offset+(mult*insert_size)),
//                            std::next(v_coord_index_map.begin(), offset+(mult*insert_size)+insert_size),
//                            std::next(v_coords_tmp.begin(), (work_t*n_threads*insert_size)+(mult*insert_size) ));
                    if (++work_t == n_threads) {
                        work_t = 0;
                    }
                    ++mult;
                } while (work_t != tid);
//                #pragma omp barrier
//                if (size - work_offset > 0) {
                #pragma omp critical
                std::cout << "CHECK t: " << tid << " with size: " << size << " and offset: " << work_cnt << std::endl;

                #pragma omp atomic
                rest += size - work_cnt;
                #pragma omp barrier
                if (tid == 0)
                    std::cout << "rest total: " << rest << std::endl;

                // end critical
            }
            std::copy(v_coords_tmp.begin(), v_coords_tmp.end(), p_v_coords);
             */

            std::sort(v_coord_index_map.begin(), v_coord_index_map.end(), [&](const auto &i1, const auto &i2) ->
                    bool {
                return p_v_coords[i1 * max_d] < p_v_coords[i2 * max_d];
            });
            for (uint i = 0; i < v_coord_index_map.size(); ++i) {
                for (uint j = 0; j < max_d; ++j) {
                    v_coords_tmp[i * max_d + j] = p_v_coords[(v_coord_index_map[i] * max_d) + j];
                }
            }
            if (nodes_no == 1) {
                std::copy(v_coords_tmp.begin(), v_coords_tmp.end(), p_v_coords);
            }

        }
#ifdef MPI_ON
        if (nodes_no > 1) {
            /*
            std::vector<uint> v_block_sizes(nodes_no);
            std::vector<uint> v_block_offsets(nodes_no);
//        next_io::get_blocks_meta(v_block_sizes, v_block_offsets, n, blocks_no);;
            std::vector<int> v_block_sizes_in_bytes;
            v_block_sizes_in_bytes.reserve(v_block_sizes.size());
            std::vector<int> v_block_offsets_in_bytes;
            v_block_offsets_in_bytes.reserve(v_block_offsets.size());
            for (uint i = 0; i < v_block_sizes.size(); ++i) {
                v_block_sizes_in_bytes[i] = v_block_sizes[i] * max_d;
                v_block_offsets_in_bytes[i] = v_block_offsets[i] * max_d;
            }
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &v_coords[0], &v_block_sizes_in_bytes[0],
                    &v_block_offsets_in_bytes[0], MPI_FLOAT, MPI_COMM_WORLD);
                    */

            uint tmp_index = 0;
            auto v_block_block_sizes = std::make_unique<uint[]>(nodes_no * nodes_no);
            auto v_block_block_offsets = std::make_unique<uint[]>(nodes_no * nodes_no);
            deep_io::get_blocks_meta(v_block_block_sizes, v_block_block_offsets, total_samples, nodes_no * nodes_no);
//            if (node_index == 0) {
//                print_array("0 node node sizes: ", &v_block_block_sizes[0], nodes_no*nodes_no);
//                print_array("0 node node offsets: ", &v_block_block_offsets[0], nodes_no*nodes_no);
//            }
//            #pragma omp parallel for
            for (uint i = 0; i < nodes_no; i++) {
                uint node = i * nodes_no + node_index;
                p_v_coords = &v_coords[v_block_block_offsets[node] * max_d];
                std::copy(&v_coords_tmp[tmp_index], std::next(&v_coords_tmp[tmp_index],
                        v_block_block_sizes[node]*max_d), p_v_coords);
                tmp_index += v_block_block_sizes[node]*max_d;
            }
//            uint elem_cnt = 0;
//            #pragma omp parallel for reduction(+: elem_cnt)
//            for (uint i = 0; i < nodes_no*nodes_no; ++i) {
//                for (uint j = 0; j < v_block_block_sizes[i]*max_d; ++j) {
//                    ++elem_cnt;
//                }
//            }
//            assert(elem_cnt == total_samples*max_d);
            MPI_Barrier(MPI_COMM_WORLD);
//            assert(tmp_index == v_coords_tmp.size());
            auto v_block_sizes_in_bytes = std::make_unique<int[]>(nodes_no);
            auto v_block_offsets_in_bytes = std::make_unique<int[]>(nodes_no);
            for (uint i = 0; i < nodes_no; i++) {
                uint node = i * nodes_no;
                p_v_coords = &v_coords[v_block_block_offsets[node]*max_d];
                for (uint j = 0; j < nodes_no; j++) {
                    v_block_sizes_in_bytes[j] = v_block_block_sizes[node + j] * max_d;
                    v_block_offsets_in_bytes[j] = (v_block_block_offsets[node + j] - v_block_block_offsets[node]) * max_d;
                }
//                if (node_index == 0) {
//                    std::cout << "merging node: " << i << std::endl;
//                    print_array("block sizes: ", &v_block_sizes_in_bytes[0], nodes_no);
//                    print_array("block offsets: ", &v_block_offsets_in_bytes[0], nodes_no);
//                }
                MPI_Barrier(MPI_COMM_WORLD);
                MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, p_v_coords, &v_block_sizes_in_bytes[0],
                        &v_block_offsets_in_bytes[0], MPI_FLOAT, MPI_COMM_WORLD);

            }
        }
#endif
        /*
        float sum = 0;
        #pragma omp parallel for reduction(+:sum)
        for (uint i = 0; i < total_samples*max_d; ++i) {
            sum += v_coords[i];
        }
        std::cout << "SUM CHECK #1: " << sum << std::endl;
         */
        v_coords_tmp.clear();
        v_coords_tmp.shrink_to_fit();
        auto v_omp_block_sizes = std::make_unique<uint[]>(n_cores);
        auto v_omp_block_offsets = std::make_unique<uint[]>(n_cores);
        deep_io::get_blocks_meta(v_omp_block_sizes, v_omp_block_offsets, total_samples, n_cores);

        // thread_id x level x elems
        std::vector<std::vector<std::vector<uint>>> vvv_index_maps(n_cores);
        std::vector<std::vector<std::vector<ull>>> vvv_value_maps(n_cores);
        std::vector<std::vector<std::vector<uint>>> vvv_cell_begins(n_cores);
        std::vector<std::vector<std::vector<uint>>> vvv_cell_ns(n_cores);
        std::vector<std::vector<std::vector<float>>> vvv_min_cell_dims(n_cores);
        std::vector<std::vector<std::vector<float>>> vvv_max_cell_dims(n_cores);
         // Tree
        std::vector<std::vector<struct_label>> vv_labels(n_cores);
        std::vector<std::vector<bool>> vv_is_core(n_cores);
        std::vector<std::vector<uint>> vv_point_nns(n_cores);
        std::vector<std::vector<uint>> vv_leaf_cell_nns(n_cores);
        std::vector<std::vector<uint8_t>> vv_cell_types(n_cores);
        std::vector<cell_meta_3> stacks3[n_threads];
        std::vector<cell_meta_5> stacks5[n_threads];
        std::vector<std::vector<bool>> vv_range_tables(n_threads);
        std::vector<cell_meta_5> v_mult_tree_tasks;

        auto v_p_coords = std::make_unique<float*[]>(n_cores);
        for (uint t = 0; t < n_cores; ++t) {
            vvv_index_maps[t].resize(max_levels);
            vvv_value_maps[t].resize(max_levels);
            vvv_cell_begins[t].resize(max_levels);
            vvv_cell_ns[t].resize(max_levels);
            vvv_min_cell_dims[t].resize(max_levels);
            vvv_max_cell_dims[t].resize(max_levels);
            vv_labels[t].resize(v_omp_block_sizes[t]);
            vv_is_core[t].resize(v_omp_block_sizes[t], false);
            vv_point_nns[t].resize(v_omp_block_sizes[t], 0);
//            if (node_index == 0)
//                std::cout << "t: " << t << " pointer: " << v_omp_block_offsets[t]*max_d << std::endl;
            v_p_coords[t] = &v_coords[v_omp_block_offsets[t]*max_d];
        }
        auto time3 = std::chrono::high_resolution_clock::now();
        if (!g_quiet && node_index == 0) {
            std::cout << "Memory init: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(time3 - time2).count()
                      << " milliseconds\n";
        }
        uint max_points_in_leaf_cell = 0;
        uint task_size = 0;
        uint task_offset = 0;
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nid = tid + (int) (node_index * n_threads);
            uint size = v_omp_block_sizes[nid];
            for (uint l = 0; l < max_levels; ++l) {
                vvv_index_maps[nid][l].resize(size);
                vvv_value_maps[nid][l].resize(size);
//                if (node_index == 0)
//                    std::cout << "mpi_index: " << node_index << " t: " << tid << " level #" << l << " size: " << size << std::endl;
                size = index_level(v_p_coords[nid], vvv_index_maps[nid][l], vvv_value_maps[nid][l],
                        vvv_index_maps[nid], vvv_cell_begins[nid], vvv_cell_ns[nid][l], v_min_bounds,
                        &v_dims_mult[l * max_d], l, max_d, v_eps_levels[l]);
                calculate_level_cell_bounds(v_p_coords[nid], vvv_cell_begins[nid][l], vvv_cell_ns[nid][l],
                        vvv_index_maps[nid][l], vvv_min_cell_dims[nid], vvv_max_cell_dims[nid], max_d, l);
            }
            uint leaf_cell_no = vvv_cell_ns[nid][0].size();
            stacks3[tid].reserve(leaf_cell_no * (uint) std::max((int) logf(max_d), 1) / n_threads);
            stacks5[tid].reserve(leaf_cell_no * (uint) std::max((int) logf(max_d), 1) / n_threads);
            #pragma omp barrier
            #pragma omp single
            {
#ifdef MPI_ON
                if (nodes_no > 1) {
                    mpi_cell_trees_merge(vvv_index_maps, vvv_value_maps, vvv_cell_begins, vvv_cell_ns,
                            vvv_min_cell_dims, vvv_max_cell_dims, node_index, nodes_no, n_threads, n_cores,
                            max_levels, max_d);
                }
#endif
/*
                uint usum = 0;
                for (uint t = 0; t < n_cores; ++t) {
                    for (uint l = 0; l < max_levels; ++l) {
                        for (auto &val : vvv_cell_ns[t][l]) {
                            usum += val;
                        }
                    }
                }
                std::cout << "SUM CHECK #2: " << usum << std::endl;
                usum = 0;
                for (uint t = 0; t < n_cores; ++t) {
                    for (uint l = 0; l < max_levels; ++l) {
                        for (auto &val : vvv_cell_begins[t][l]) {
                            usum += val;
                        }
                    }
                }
                std::cout << "SUM CHECK #3: " << usum << std::endl;
                usum = 0;
                for (uint t = 0; t < n_cores; ++t) {
                    for (uint l = 0; l < max_levels; ++l) {
                        for (auto &val : vvv_index_maps[t][l]) {
                            usum += val;
                        }
                    }
                }
                std::cout << "SUM CHECK #4: " << usum << std::endl;
                ull ullsum = 0;
                for (uint t = 0; t < n_cores; ++t) {
                    for (uint l = 0; l < max_levels; ++l) {
                        for (auto &val : vvv_value_maps[t][l]) {
                            ullsum += val;
                        }
                    }
                }
                std::cout << "SUM CHECK #5: " << ullsum << std::endl;
                sum = 0;
                for (uint t = 0; t < n_cores; ++t) {
                    for (uint l = 0; l < max_levels; ++l) {
                        for (auto &val : vvv_min_cell_dims[t][l]) {
                            sum += val;
                        }
                    }
                }
                std::cout << "SUM CHECK #6: " << sum << std::endl;
                sum = 0;
                for (uint t = 0; t < n_cores; ++t) {
                    for (uint l = 0; l < max_levels; ++l) {
                        for (auto &val : vvv_max_cell_dims[t][l]) {
                            sum += val;
                        }
                    }
                }
                std::cout << "SUM CHECK #7: " << sum << std::endl;
                */
                uint level = n_cores > 2? max_levels-1 : max_levels-2;
                uint level_permutations = 1;
                for (uint t = 0; t < n_cores; ++t) {
                    vv_leaf_cell_nns[t].resize(vvv_cell_ns[t][0].size(), 0);
                    vv_cell_types[t].resize(vvv_cell_ns[t][0].size(), NC);
                    level_permutations *= vvv_cell_ns[t][level].size();
                }
                v_mult_tree_tasks.reserve(level_permutations);
                for (uint t1 = 0; t1 < n_cores; ++t1) {
                    for (uint i = 0; i < vvv_cell_ns[t1][level].size(); ++i) {
                        for (uint t2 = t1 + 1; t2 < n_cores; ++t2) {
                            for (uint j = 0; j < vvv_cell_ns[t2][level].size(); ++j) {
                                v_mult_tree_tasks.emplace_back(level, i, j, t1, t2);
                            }
                        }
                    }
                }
//                if (node_index == 0)
//                    std::cout << "Total number of core multi tree tasks: " << v_mult_tree_tasks.size() << std::endl;
                task_size = deep_io::get_block_size(node_index, v_mult_tree_tasks.size(), nodes_no);
                task_offset = deep_io::get_block_start_offset(node_index, v_mult_tree_tasks.size(), nodes_no);
                // end single
            }

            #pragma omp for reduction(max: max_points_in_leaf_cell)
            for (uint t = 0; t < n_cores; ++t) {
                if (vvv_cell_ns[t][0].size() > max_points_in_leaf_cell)
                    max_points_in_leaf_cell = vvv_cell_ns[t][0].size();
            }
            vv_range_tables[tid].resize(max_points_in_leaf_cell*max_points_in_leaf_cell);
            for (uint l = 0; l < max_levels; ++l) {
                process_cell_tree_level(v_p_coords[nid], vvv_cell_begins[nid], vvv_cell_ns[nid], vvv_index_maps[nid],
                        vv_leaf_cell_nns[nid], vv_cell_types[nid], vv_is_core[nid], vv_point_nns[nid], stacks3[tid],
                        vv_range_tables[tid], vvv_min_cell_dims[nid], vvv_max_cell_dims[nid], l, m, max_d, e, e2, true);
            }
            if (n_cores > 1) {
                #pragma omp barrier
                #pragma omp for schedule(dynamic)
                for (uint i = task_offset; i < task_offset + task_size; ++i) {
                    stacks5[tid].push_back(v_mult_tree_tasks[i]);
                    process_cell_tree_pairs(v_p_coords, vvv_index_maps, vvv_value_maps, vvv_cell_begins,
                            vvv_min_cell_dims, vvv_max_cell_dims, vv_leaf_cell_nns, vv_cell_types,
                            vv_range_tables[tid],
                            vv_is_core, vv_point_nns, vvv_cell_ns, stacks5[tid], max_d, m, e, e2, true);
                }
//                }
            }
        // end of parallel region
        }
#ifdef MPI_ON
        MPI_Barrier(MPI_COMM_WORLD);
        if (nodes_no > 1) {
            mpi_sum_tree(vv_point_nns, n_cores, max_levels, nodes_no, n_threads, node_index, MPI_UNSIGNED, false);
            mpi_sum_tree(vv_leaf_cell_nns, n_cores, max_levels, nodes_no, n_threads, node_index, MPI_UNSIGNED, false);
            // TODO also the type for the labels
        }
#endif
        #pragma omp parallel for
        for (uint t = 0; t < n_cores; ++t) {
            for (uint i = 0; i < vvv_cell_ns[t][0].size(); ++i) {
                uint begin = vvv_cell_begins[t][0][i];
                for (uint j = 0; j < vvv_cell_ns[t][0][i]; ++j) {
                    uint index = vvv_index_maps[t][0][begin+j];
                    if (vv_leaf_cell_nns[t][i] + vv_point_nns[t][index] >= m) {
                        vv_is_core[t][index] = true;
                    }
                }
            }
        }
//#endif
        auto time4 = std::chrono::high_resolution_clock::now();
        if (!g_quiet && node_index == 0) {
            std::cout << "Process cell trees: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(time4 - time3).count()
                      << " milliseconds\n";
        }
        auto time5 = std::chrono::high_resolution_clock::now();
        if (!g_quiet && node_index == 0) {
            std::cout << "Total Execution Time: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(time5 - time1).count()
                      << " milliseconds\n";
        }
        return calculate_output(vv_is_core, vv_labels[0], v_omp_block_sizes[0]);
    }

}