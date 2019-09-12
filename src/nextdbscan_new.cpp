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
#include <unordered_map>
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
    // TODO implement verbose
    static bool verbose = false;

    struct type_vector {
        uint8_t *data = nullptr;

        type_vector() = delete;

        inline explicit type_vector(const uint size) noexcept {
            data = new uint8_t[size];
        }

        inline ~type_vector() noexcept {
            delete[] data;
        }

        inline void fill(const uint size, const bool value) noexcept {
            std::fill(data, data + size, value);
        }

        inline uint8_t &operator[](std::size_t index) noexcept {
            return data[index];
        }

        inline const uint8_t &operator[](std::size_t index) const noexcept {
            return data[index];
        }
    };

    struct bool_vector {
        bool *data = nullptr;

        bool_vector() = delete;

        inline explicit bool_vector(const uint size) noexcept {
            data = new bool[size];
        }

        inline ~bool_vector() noexcept {
            delete[] data;
        }

        inline void fill(const uint size, const bool value) noexcept {
            std::fill(data, data + size, value);
        }

        inline bool &operator[](std::size_t index) noexcept {
            return data[index];
        }

        inline const bool &operator[](std::size_t index) const noexcept {
            return data[index];
        }
    };

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
            float *max_bounds, uint max_d) noexcept {
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

    inline void update_to_ac(std::vector<std::pair<ull, uint>> &v_index_maps, std::vector<uint> &v_cell_ns,
            std::vector<uint> &v_cell_begin, bool_vector &is_core, std::vector<uint8_t> &v_types, const uint c) {
        v_types[c] = AC;
        uint begin = v_cell_begin[c];
        for (uint j = 0; j < v_cell_ns[c]; ++j) {
            is_core[v_index_maps[begin + j].second] = true;
        }
    }

    void update_type(std::vector<std::pair<ull, uint>> &v_index_maps, std::vector<uint> &v_cell_ns,
            std::vector<uint> &v_cell_begin, std::vector<uint> &v_cell_nps, std::vector<uint> &v_point_nps,
            bool_vector &is_core, std::vector<uint8_t> &v_types, const uint c, const uint m) {
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
            uint p = v_index_maps[begin + j].second;
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

    bool fill_range_table_multi(float *v_coords, std::vector<std::pair<ull, uint>> **vv_index_map,
            std::vector<uint> **v_cell_ns_level, bool_vector &v_range_table, const uint c1,
            const uint begin1, const uint c2, const uint begin2, const uint max_d, const float e2, const uint t1, const uint t2) noexcept {
        uint size1 = v_cell_ns_level[t1][0][c1];
        uint size2 = v_cell_ns_level[t2][0][c2];
        bool all_in_range = true;
        uint index = 0;
        v_range_table.fill(size1 * size2, false);
        for (uint k1 = 0; k1 < size1; ++k1) {
            uint p1 = vv_index_map[t1][0][begin1 + k1].second;
            for (uint k2 = 0; k2 < size2; ++k2, ++index) {
                uint p2 = vv_index_map[t2][0][begin2 + k2].second;
                if (dist_leq(&v_coords[p1 * max_d], &v_coords[p2 * max_d], max_d, e2)) {
                    v_range_table[index] = true;
                } else {
                    all_in_range = false;
                }
            }
        }
        return all_in_range;
    }

    void update_cell_pair_nn_multi(std::vector<std::pair<ull, uint>> **vv_index_map, std::vector<uint> **vv_cell_ns,
            std::vector<uint> &v_point_nps, bool_vector &v_range_table, bool_vector &is_core, const uint c1,
            const uint begin1, const uint c2, const uint begin2, const uint t1, const uint t2) {
        uint size1 = vv_cell_ns[t1][0][c1];
        uint size2 = vv_cell_ns[t2][0][c2];
        uint index = 0;
        for (uint k1 = 0; k1 < size1; ++k1) {
            uint p1 = vv_index_map[t1][0][begin1 + k1].second;
            for (uint k2 = 0; k2 < size2; ++k2, ++index) {
                uint p2 = vv_index_map[t2][0][begin2 + k2].second;
                if (is_core[p1] && is_core[p2])
                    continue;
                if (v_range_table[index]) {
                    if (!is_core[p1]) {
                        #pragma omp atomic
                        ++v_point_nps[p1];
                    }
                    if (!is_core[p2]) {
                        #pragma omp atomic
                        ++v_point_nps[p2];
                    }
                }
            }
        }
    }

    void process_cell_tree_mult(struct_label **p_labels, float *v_coords, std::vector<std::pair<ull, uint>> **vv_index_maps,
            std::vector<uint> &v_point_nps, std::vector<cell_meta_5> &stack, std::vector<uint> **vv_cell_begin,
            std::vector<uint> **vv_cell_ns, std::vector<float> **vv_cell_dim_min, std::vector<float> **vv_cell_dim_max,
            omp_lock_t *mutexes, bool_vector &v_range_table, std::vector<uint> *v_cell_nps, bool_vector &is_core,
            std::vector<uint8_t> *v_types, const uint max_d, const float e, const uint m, const bool is_nn) noexcept {
        const auto e2 = e * e;
        while (!stack.empty()) {
            uint l = stack.back().l;
            uint t1 = stack.back().t1;
            uint c1 = stack.back().c1;
            uint t2 = stack.back().t2;
            uint c2 = stack.back().c2;
            stack.pop_back();
            if (vv_index_maps[t1][l][c1].first != vv_index_maps[t2][l][c2].first &&
                !is_in_reach(&vv_cell_dim_min[t1][l][c1 * max_d], &vv_cell_dim_max[t1][l][c1 * max_d],
                    &vv_cell_dim_min[t2][l][c2 * max_d], &vv_cell_dim_max[t2][l][c2 * max_d], max_d, e)) {
                continue;
            }
            uint begin1 = vv_cell_begin[t1][l][c1];
            uint begin2 = vv_cell_begin[t2][l][c2];
            if (l == 0) {
                if (is_nn) {
                    if (v_types[t1][c1] != AC || v_types[t2][c2] != AC) {
                        bool all_range_check = fill_range_table_multi(v_coords, vv_index_maps, vv_cell_ns,
                                v_range_table, c1, begin1, c2, begin2,
                                max_d, e2, t1, t2);
                        if (all_range_check) {
                            if (v_types[t1][c1] != AC) {
                                #pragma omp atomic
                                v_cell_nps[t1][c1] += vv_cell_ns[t2][0][c2];
                            }
                            if (v_types[t2][c2] != AC) {
                                #pragma omp atomic
                                v_cell_nps[t2][c2] += vv_cell_ns[t1][0][c1];
                            }
                        } else {
                            update_cell_pair_nn_multi(vv_index_maps, vv_cell_ns, v_point_nps, v_range_table,
                                    is_core, c1, begin1, c2, begin2, t1, t2);
                        }
                        update_type(vv_index_maps[t1][0], vv_cell_ns[t1][0], vv_cell_begin[t1][0], v_cell_nps[t1],
                                v_point_nps, is_core, v_types[t1], c1, m);
                        update_type(vv_index_maps[t2][0], vv_cell_ns[t2][0], vv_cell_begin[t2][0], v_cell_nps[t2],
                                v_point_nps, is_core, v_types[t2], c2, m);
                    }
                } else {
                    if (v_types[t1][c1] != NC || v_types[t2][c2] != NC) {
                        if (v_types[t1][c1] != NC && v_types[t2][c2] != NC) {
                            for (uint k1 = 0; k1 < vv_cell_ns[t1][l][c1]; ++k1) {
                                uint p1 = vv_index_maps[t1][l][begin1 + k1].second;
                                for (uint k2 = 0; k2 < vv_cell_ns[t2][l][c2]; ++k2) {
                                    uint p2 = vv_index_maps[t2][l][begin2 + k2].second;
                                    if (is_core[p1] && is_core[p2] && dist_leq(&v_coords[p1 * max_d],
                                                &v_coords[p2 * max_d], max_d, e2)) {
                                        // TODO make thread safe
                                        auto p1_label = get_label(p_labels[p1]);
                                        auto p2_label = get_label(p_labels[p2]);
                                        if (p1_label != p2_label) {
                                            set_lower_label(p1_label, p2_label);
                                        }
                                        k2 = vv_cell_ns[t2][l][c2];
                                        k1 = vv_cell_ns[t1][l][c1];
                                    }
                                }
                            }
                        } else {
                            for (uint k1 = 0; k1 < vv_cell_ns[t1][l][c1]; ++k1) {
                                uint p1 = vv_index_maps[t1][l][begin1 + k1].second;
                                auto p1_label = get_label(p_labels[p1]);
                                if (!is_core[p1] && p1_label->label != UNASSIGNED)
                                    continue;
                                for (uint k2 = 0; k2 < vv_cell_ns[t2][l][c2]; ++k2) {
                                    uint p2 = vv_index_maps[t2][l][begin2 + k2].second;
                                    auto p2_label = get_label(p_labels[p2]);
                                    if (!is_core[p2] && p2_label->label != UNASSIGNED)
                                        continue;
                                    if (is_core[p1]) {
                                        if (dist_leq(&v_coords[p1 * max_d], &v_coords[p2 * max_d], max_d, e2)) {
                                            p_labels[p2]->label_p = p1_label;
                                        }
                                    } else if (is_core[p2]) {
                                        if (dist_leq(&v_coords[p1 * max_d], &v_coords[p2 * max_d], max_d, e2)) {
                                            p_labels[p1]->label_p = p2_label;
                                            k2 = vv_cell_ns[t2][l][c2];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                for (uint k1 = 0; k1 < vv_cell_ns[t1][l][c1]; ++k1) {
                    uint c1_next = vv_index_maps[t1][l][begin1 + k1].second;
                    for (uint j = 0; j < vv_cell_ns[t2][l][c2]; ++j) {
                        uint c2_next = vv_index_maps[t2][l][begin2 + j].second;
                        stack.emplace_back(l - 1, c1_next, c2_next, t1, t2);
                    }
                }
            }
        }
    }

    bool fill_range_table(float *v_coords, std::vector<std::pair<ull, uint>> &v_index_map_level,
            std::vector<uint> &v_cell_ns_level, bool_vector &v_range_table, const uint c1,
            const uint begin1, const uint c2, const uint begin2, const uint max_d, const float e2) noexcept {
        uint size1 = v_cell_ns_level[c1];
        uint size2 = v_cell_ns_level[c2];
        bool all_in_range = true;
        uint index = 0;
        v_range_table.fill(size1 * size2, false);
        for (uint k1 = 0; k1 < size1; ++k1) {
            uint p1 = v_index_map_level[begin1 + k1].second;
            for (uint k2 = 0; k2 < size2; ++k2, ++index) {
                uint p2 = v_index_map_level[begin2 + k2].second;
                if (dist_leq(&v_coords[p1 * max_d], &v_coords[p2 * max_d], max_d, e2)) {
                    v_range_table[index] = true;
                } else {
                    all_in_range = false;
                }
            }
        }
        return all_in_range;
    }

    void update_cell_pair_nn(std::vector<std::pair<ull, uint>> &v_index_map_level, std::vector<uint> &v_cell_ns_level,
            std::vector<uint> &v_point_nps, bool_vector &v_range_table, bool_vector &is_core, const uint c1,
            const uint begin1, const uint c2, const uint begin2) {
        uint size1 = v_cell_ns_level[c1];
        uint size2 = v_cell_ns_level[c2];
        uint index = 0;
        for (uint k1 = 0; k1 < size1; ++k1) {
            uint p1 = v_index_map_level[begin1 + k1].second;
            for (uint k2 = 0; k2 < size2; ++k2, ++index) {
                uint p2 = v_index_map_level[begin2 + k2].second;
                if (is_core[p1] && is_core[p2])
                    continue;
                if (v_range_table[index]) {
                    if (!is_core[p1]) {
                        ++v_point_nps[p1];
                    }
                    if (!is_core[p2]) {
                        ++v_point_nps[p2];
                    }
                }
            }
        }
    }

    void process_pair_nn(float *v_coords, std::vector<std::pair<ull, uint>> *vv_index_maps,
            std::vector<uint> &v_point_nps, std::vector<uint> *vv_cell_begin, std::vector<uint> *vv_cell_ns,
            bool_vector &v_range_table, std::vector<uint> &v_cell_nps, bool_vector &is_core,
            std::vector<uint8_t> &v_types, const uint max_d, const float e2, const uint m, const uint l,
            const uint c1, const uint begin1, const uint c2, const uint begin2) {
        bool all_range_check = fill_range_table(v_coords, vv_index_maps[l], vv_cell_ns[l],
                v_range_table, c1, begin1, c2, begin2,
                max_d, e2);
        if (all_range_check) {
            if (v_types[c1] != AC) {
                v_cell_nps[c1] += vv_cell_ns[0][c2];
            }
            if (v_types[c2] != AC) {
                v_cell_nps[c2] += vv_cell_ns[0][c1];
            }
        } else {
            update_cell_pair_nn(vv_index_maps[l], vv_cell_ns[l], v_point_nps, v_range_table,
                    is_core, c1, begin1, c2, begin2);
        }
        update_type(vv_index_maps[0], vv_cell_ns[0], vv_cell_begin[0], v_cell_nps,
                v_point_nps, is_core, v_types, c1, m);
        update_type(vv_index_maps[0], vv_cell_ns[0], vv_cell_begin[0], v_cell_nps,
                v_point_nps, is_core, v_types, c2, m);
    }

    /*
    void process_cell_tree_single(struct_label **p_labels, float *v_coords, std::vector<std::pair<ull, uint>> *vv_index_maps,
            std::vector<uint> &v_point_nps, std::vector<cell_meta_3> &stack, std::vector<uint> *vv_cell_begin,
            std::vector<uint> *vv_cell_ns, std::vector<uint> &v_no_cells, std::vector<float> *vv_cell_dim_min,
            std::vector<float> *vv_cell_dim_max, bool_vector &v_range_table, std::vector<uint> &v_cell_nps,
            bool_vector &is_core, std::vector<uint8_t> &v_types, const uint max_levels,
            const uint max_d, const float e, const uint m, const bool is_nn) noexcept {
        const auto e2 = e * e;
        for (uint level = 1; level < max_levels; ++level) {
            for (uint i = 0; i < v_no_cells[level]; ++i) {
                uint begin = vv_cell_begin[level][i];
                for (uint c1 = 0; c1 < vv_cell_ns[level][i]; ++c1) {
                    for (uint c2 = c1 + 1; c2 < vv_cell_ns[level][i]; ++c2) {
                        stack.emplace_back(level - 1, vv_index_maps[level][begin + c1].second,
                                vv_index_maps[level][begin + c2].second);
                    }
                }
                while (!stack.empty()) {
                    uint l = stack.back().l;
                    uint c1 = stack.back().c1;
                    uint c2 = stack.back().c2;
                    stack.pop_back();
                    if (!is_in_reach(&vv_cell_dim_min[l][c1 * max_d], &vv_cell_dim_max[l][c1 * max_d],
                            &vv_cell_dim_min[l][c2 * max_d], &vv_cell_dim_max[l][c2 * max_d], max_d, e)) {
                        continue;
                    }
                    uint begin1 = vv_cell_begin[l][c1];
                    uint begin2 = vv_cell_begin[l][c2];
                    if (l == 0) {
                        if (is_nn) {
                            if (v_types[c1] != AC || v_types[c2] != AC) {
                                process_pair_nn(v_coords, vv_index_maps, v_point_nps, vv_cell_begin, vv_cell_ns,
                                        v_range_table, v_cell_nps, is_core, v_types, max_d, e2, m, l, c1, begin1, c2,
                                        begin2);
                            }
                        } else {
                            if (v_types[c1] != NC || v_types[c2] != NC) {
//                                assert(!(v_types[c1] == NC && v_types[c2] == NC));
                                if (v_types[c1] != NC && v_types[c2] != NC) {
                                    for (uint k1 = 0; k1 < vv_cell_ns[l][c1]; ++k1) {
                                        uint p1 = vv_index_maps[l][begin1 + k1].second;
                                        for (uint k2 = 0; k2 < vv_cell_ns[l][c2]; ++k2) {
                                            uint p2 = vv_index_maps[l][begin2 + k2].second;
                                            if (is_core[p1] && is_core[p2] && dist_leq(&v_coords[p1 * max_d],
                                                    &v_coords[p2 * max_d], max_d, e2)) {
                                                auto p1_label = get_label(p_labels[p1]);
                                                auto p2_label = get_label(p_labels[p2]);
                                                if (p1_label != p2_label)
                                                    set_lower_label(p1_label, p2_label);
                                                k2 = vv_cell_ns[l][c2];
                                                k1 = vv_cell_ns[l][c1];
                                            }
                                        }
                                    }
                                } else {
                                    for (uint k1 = 0; k1 < vv_cell_ns[l][c1]; ++k1) {
                                        uint p1 = vv_index_maps[l][begin1 + k1].second;
                                        auto p1_label = get_label(p_labels[p1]);
                                        if (!is_core[p1] && p1_label->label != UNASSIGNED)
                                            continue;
                                        for (uint k2 = 0; k2 < vv_cell_ns[l][c2]; ++k2) {
                                            uint p2 = vv_index_maps[l][begin2 + k2].second;
                                            auto p2_label = get_label(p_labels[p2]);
                                            if (!is_core[p2] && p2_label->label != UNASSIGNED)
                                                continue;
                                            if (is_core[p1]) {
                                                if (dist_leq(&v_coords[p1 * max_d], &v_coords[p2 * max_d], max_d, e2)) {
                                                    p_labels[p2]->label_p = p1_label;
                                                }
                                            } else if (is_core[p2]) {
                                                if (dist_leq(&v_coords[p1 * max_d], &v_coords[p2 * max_d], max_d, e2)) {
                                                    p_labels[p1]->label_p = p2_label;
                                                    k2 = vv_cell_ns[l][c2];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        for (uint k1 = 0; k1 < vv_cell_ns[l][c1]; ++k1) {
                            uint c1_next = vv_index_maps[l][begin1 + k1].second;
                            for (uint j = 0; j < vv_cell_ns[l][c2]; ++j) {
                                uint c2_next = vv_index_maps[l][begin2 + j].second;
                                stack.emplace_back(l - 1, c1_next, c2_next);
                            }
                        }
                    }
                }
            }
        }
    }
     */

    /*
    void process_cell_tree(struct_label **p_labels, float *v_coords, bool_vector &is_core,
            std::vector<std::pair<ull, uint>> **vv_index_maps, std::vector<uint> **vv_cell_begin, std::vector<uint>
                    **vv_cell_ns, std::vector<uint> *v_no_cells, std::vector<float> **vv_cell_dim_min,
                    std::vector<float> **vv_cell_dim_max, bool_vector **vv_range_tables, const uint max_levels,
                    const uint max_d, const uint n_threads, const float e, const uint n, const uint m) noexcept {
        std::vector<uint> v_point_nps(n, 0);
        std::vector<cell_meta_3> stacks3[n_threads];
        std::vector<cell_meta_5> stacks5[n_threads];
        std::vector<uint8_t> vv_cell_types[n_threads];
        std::vector<uint> vv_cell_nps[n_threads];
        omp_lock_t mutexes[n_threads];
        for (uint t = 0; t < n_threads; t++) {
            stacks3[t].reserve(v_no_cells[t][0] * (uint) std::max((int) logf(max_d), 1));
            stacks5[t].reserve(v_no_cells[t][0] * (uint) std::max((int) logf(max_d), 1));
            vv_cell_nps[t].reserve(v_no_cells[t][0]);
            vv_cell_types[t].reserve(v_no_cells[t][0]);
            std::fill(vv_cell_types[t].begin(), vv_cell_types[t].begin() + v_no_cells[t][0], NC);
        }
        uint max_points_in_cell = 0;
        #pragma omp parallel for reduction(max: max_points_in_cell)
        for (uint t = 0; t < n_threads; ++t) {
            for (uint i = 0; i < v_no_cells[t][0]; ++i) {
                uint size = vv_cell_ns[t][0][i];
                if (size > max_points_in_cell) {
                    max_points_in_cell = size;
                }
            }
        }
        // collect all the multiple thread pairs
        std::vector<cell_meta_5> v_t_pair_tasks;
        if (n_threads > 1) {
            v_t_pair_tasks.reserve(n_threads * n_threads);
            // TODO find the right balance
            uint level = max_levels - 3;
            for (uint t1 = 0; t1 < n_threads; ++t1) {
                omp_init_lock(&mutexes[t1]);
                for (uint t2 = t1 + 1; t2 < n_threads; ++t2) {
                    for (uint i = 0; i < v_no_cells[t1][level]; ++i) {
                        for (uint j = 0; j < v_no_cells[t2][level]; ++j) {
                            v_t_pair_tasks.emplace_back(level, i, j, t1, t2);
                        }
                    }
                }
            }
            std::cout << "pair tasks: " << v_t_pair_tasks.size() << std::endl;
        }
        #pragma omp parallel default(shared)
        {
            uint tid = omp_get_thread_num();
            for (uint i = 0; i < v_no_cells[tid][0]; ++i) {
                vv_cell_nps[tid][i] = vv_cell_ns[tid][0][i];
                if (vv_cell_nps[tid][i] >= m) {
                    vv_cell_types[tid][i] = AC;
                    uint begin = vv_cell_begin[tid][0][i];
                    for (uint j = 0; j < vv_cell_ns[tid][0][i]; ++j) {
                        is_core[vv_index_maps[tid][0][begin + j].second] = true;
                    }
                }
            }
            process_cell_tree_single(p_labels, v_coords, vv_index_maps[tid], v_point_nps, stacks3[tid], vv_cell_begin[tid],
                    vv_cell_ns[tid], v_no_cells[tid], vv_cell_dim_min[tid], vv_cell_dim_max[tid],
                    *vv_range_tables[tid], vv_cell_nps[tid], is_core, vv_cell_types[tid], max_levels, max_d, e, m, true);
            #pragma omp barrier
            #pragma omp for schedule(dynamic)
            for (uint i = 0; i < v_t_pair_tasks.size(); ++i) {
                stacks5[tid].push_back(v_t_pair_tasks[i]);
                process_cell_tree_mult(p_labels, v_coords, vv_index_maps, v_point_nps, stacks5[tid],
                        vv_cell_begin, vv_cell_ns, vv_cell_dim_min, vv_cell_dim_max, mutexes, *vv_range_tables[tid],
                        vv_cell_nps, is_core, vv_cell_types, max_d, e, m, true);
            }
            for (uint i = 0; i < v_no_cells[tid][0]; ++i) {
                if (vv_cell_types[tid][i] == NC) {
                    continue;
                } else {
                    uint begin = vv_cell_begin[tid][0][i];
                    uint i_core = 0;
//                    uint p;
                    if (vv_cell_types[tid][i] == SC) {
                        // find a core
                        for (uint j = 0; j < vv_cell_ns[tid][0][i]; ++j) {
                            uint p = vv_index_maps[tid][0][begin+j].second;
                            if (is_core[p]) {
                                i_core = p;
                                j = vv_cell_ns[tid][0][i];
                            }
                        }
                    } else {
                        i_core = vv_index_maps[tid][0][begin].second;
                    }
                    p_labels[i_core]->label = i_core;
                    for (uint j = 0; j < vv_cell_ns[tid][0][i]; ++j) {
                        uint p = vv_index_maps[tid][0][begin+j].second;
                        if (p == i_core)
                            continue;
                        p_labels[p]->label_p = p_labels[i_core];
                    }
                }
            }
            process_cell_tree_single(p_labels, v_coords, vv_index_maps[tid], v_point_nps, stacks3[tid], vv_cell_begin[tid],
                    vv_cell_ns[tid], v_no_cells[tid], vv_cell_dim_min[tid], vv_cell_dim_max[tid],
                    *vv_range_tables[tid], vv_cell_nps[tid], is_core, vv_cell_types[tid], max_levels, max_d, e, m, false);
            #pragma omp barrier
            #pragma omp for schedule(dynamic)
            for (uint i = 0; i < v_t_pair_tasks.size(); ++i) {
                stacks5[tid].push_back(v_t_pair_tasks[i]);
                process_cell_tree_mult(p_labels, v_coords, vv_index_maps, v_point_nps, stacks5[tid],
                        vv_cell_begin, vv_cell_ns, vv_cell_dim_min, vv_cell_dim_max, mutexes, *vv_range_tables[tid],
                        vv_cell_nps, is_core, vv_cell_types, max_d, e, m, false);
            }
        }
    }
     */

    /*
    int
    index_points(float *v_coords, std::vector<uint> **vv_index_maps, std::vector<ull> **vv_value_maps, std::vector<std::pair<ull, uint>> **vv_old,
            std::vector<uint> **vv_cell_begin, std::vector<uint> **vv_cell_ns, std::vector<uint> *v_no_cells,
            bool_vector **vv_range_tables, const uint max_d, const uint n, const uint n_threads,
            const float e_inner) noexcept {
        auto t0_1 = std::chrono::high_resolution_clock::now();
        float max_limit = INT32_MIN;
        std::vector<float> min_bounds(max_d);
        std::vector<float> max_bounds(max_d);
        std::vector<std::pair<ull, uint>> vv_buckets[n_threads];
        std::vector<uint> vv_index_buckets[n_threads];
        std::vector<ull> vv_value_buckets[n_threads];
        calc_bounds(v_coords, n, min_bounds, max_bounds, max_d);
        for (uint d = 0; d < max_d; d++) {
            if (max_bounds[d] - min_bounds[d] > max_limit)
                max_limit = max_bounds[d] - min_bounds[d];
        }
        int max_levels = static_cast<int>(ceilf(logf(max_limit / e_inner) / logf(2))) + 1;
        std::vector<float> v_eps_levels(max_levels);

        auto **dims_mult = new ull *[max_levels];
        for (int i = 0; i < max_levels; i++) {
            v_eps_levels[i] = (e_inner * powf(2, i));
            dims_mult[i] = new ull[max_d];
            calc_dims_mult(dims_mult[i], max_d, min_bounds, max_bounds, v_eps_levels[i]);
        }

        for (uint t = 0; t < n_threads; ++t) {
            vv_old[t] = new std::vector<std::pair<ull, uint>>[max_levels];
            vv_index_maps[t] = new std::vector<uint>[max_levels];
            vv_value_maps[t] = new std::vector<ull>[max_levels];
            vv_cell_begin[t] = new std::vector<uint>[max_levels];
            vv_cell_ns[t] = new std::vector<uint>[max_levels];
            v_no_cells[t].reserve(max_levels);
        }
        auto t0_2 = std::chrono::high_resolution_clock::now();
        if (!g_quiet) {
            std::cout << "Init "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(t0_2 - t0_1).count()
                      << " milliseconds\n";
        }
        uint max_points_in_cell = 0;
        auto t1 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel default(shared) reduction(max: max_points_in_cell)
        {
            const uint tid = omp_get_thread_num();
            for (int level = 0; level < max_levels; ++level) {
                std::cout << "Start level: " << level << std::endl;
                uint mem_reserve = level == 0 ? n / n_threads + 1 : v_no_cells[tid][level - 1];
                vv_old[tid][level].reserve(mem_reserve);
                vv_index_maps[tid][level].reserve(mem_reserve);
                vv_value_maps[tid][level].reserve(mem_reserve);
                vv_cell_begin[tid][level].reserve(mem_reserve);
                std::cout << "CHECKPOINT #1" << std::endl;
                if (level == 0) {
                    #pragma omp for schedule(static)
                    for (uint i = 0; i < n; ++i) {
                        ull cell_index = get_cell_index(&v_coords[i*max_d], min_bounds, dims_mult[0], max_d,
                                v_eps_levels[0]);
                        vv_index_maps[tid][0].push_back(i);
                        vv_value_maps[tid][0].push_back(cell_index);
                        vv_old[tid][0].emplace_back(cell_index, i);
                    }
                } else {
                    for (uint i = 0; i < v_no_cells[tid][level-1]; ++i) {
                        int level_mod = 1;
                        uint p_index = i;
                        while (level - level_mod >= 0) {
//                            p_index = vv_index_maps[tid][level - level_mod][vv_cell_begin[tid][level-level_mod][p_index]].second;
                            p_index = vv_value_maps[tid][level - level_mod][vv_cell_begin[tid][level-level_mod][p_index]];
                            assert(p_index == vv_old[tid][level - level_mod][vv_cell_begin[tid][level-level_mod][p_index]].second);
                            ++level_mod;
                        }
                        ull cell_index = get_cell_index(&v_coords[p_index * max_d], min_bounds, dims_mult[level],
                                max_d, v_eps_levels[level]);
                        vv_index_maps[tid][level].push_back(i);
                        vv_value_maps[tid][level].push_back(cell_index);
                        vv_old[tid][level].emplace_back(cell_index, i);
                    }
                }
                std::cout << "CHECKPOINT #2" << std::endl;
                std::sort(vv_old[tid][level].begin(), vv_old[tid][level].end());
                std::sort(vv_index_maps[tid][level].begin(), vv_index_maps[tid][level].end(),
                        [&] (const auto &i1, const auto &i2) -> bool {
                    return vv_value_maps[tid][level][i1] < vv_value_maps[tid][level][i2];
                    /*
                    if (i1 < i2) {
                        uint val = vv_value_maps[tid][level][i1];
                        vv_value_maps[tid][level][i1] = vv_value_maps[tid][level][i2];
                        vv_value_maps[tid][level][i2] = val;
//                        std::swap(vv_value_maps[tid][level][i1], vv_value_maps[tid][level][i2]);
                        return true;
                    }
                    return false;
                     */
    /*
                });

                std::cout << "CHECKPOINT #3" << std::endl;
                assert(vv_old[tid][level].size() == vv_index_maps[tid][level].size());
                for (int i = 0; i < vv_old[tid][level].size(); ++i ) {
                    assert(vv_old[tid][level][i].first == vv_index_maps[tid][level][i]);
                    assert(vv_old[tid][level][i].second == vv_value_maps[tid][level][i]);
                }
                std::cout << "CHECKPOINT #4" << std::endl;
                if (level == 0 && n_threads > 1) {
                    vv_buckets[tid].reserve(vv_index_maps[tid][level].size() + 1);
                    vv_index_buckets[tid].reserve(vv_index_maps[tid][level].size() + 1);
                    vv_value_buckets[tid].reserve(vv_index_maps[tid][level].size() + 1);
                    #pragma omp barrier
                    uint chunk = vv_index_maps[0][level].size() / n_threads;
                    for (uint t = 0; t < n_threads; ++t) {
                        auto begin = std::next(vv_old[t][level].begin(), (tid *chunk));
                        auto end = std::next(vv_old[t][level].begin(), ((tid+1) *chunk));
                        if (tid == n_threads-1)
                            end = vv_old[t][level].end();
                        vv_buckets[tid].insert(vv_buckets[tid].end(), begin, end);
                        auto begin1 = std::next(vv_index_maps[t][level].begin(), (tid *chunk));
                        auto end1 = std::next(vv_index_maps[t][level].begin(), ((tid+1) *chunk));
                        if (tid == n_threads-1)
                            end1 = vv_index_maps[t][level].end();
                        vv_index_buckets[tid].insert(vv_index_buckets[tid].end(), begin1, end1);
                        auto begin2 = std::next(vv_value_maps[t][level].begin(), (tid *chunk));
                        auto end2 = std::next(vv_value_maps[t][level].begin(), ((tid+1) *chunk));
                        if (tid == n_threads-1)
                            end2 = vv_value_maps[t][level].end();
                        vv_value_buckets[tid].insert(vv_value_buckets[tid].end(), begin2, end2);
                    }
                    std::sort(vv_buckets[tid].begin(), vv_buckets[tid].end());
                    vv_old[tid][level].clear();
                    vv_old[tid][level].shrink_to_fit();
                    vv_index_maps[tid][level].clear();
                    vv_index_maps[tid][level].shrink_to_fit();
                    vv_value_maps[tid][level].clear();
                    vv_value_maps[tid][level].shrink_to_fit();
                    #pragma omp barrier
                    vv_old[tid][level] = std::move(vv_buckets[tid]);
                    vv_index_maps[tid][level] = std::move(vv_index_buckets[tid]);
                    vv_value_maps[tid][level] = std::move(vv_value_buckets[tid]);
                    vv_buckets[tid] = std::vector<std::pair<ull, uint>>();
                    vv_index_buckets[tid] = std::vector<uint>();
                    vv_value_buckets[tid] = std::vector<ull>();
//                    median_split(vv_index_maps, v_local_medians, v_merged_medians, vv_median_indexes,
//                            vv_median_buckets, level, n_threads, tid);
                }
                std::cout << "CHECKPOINT #5" << std::endl;
                /*
                ull last_index = vv_index_maps[tid][level][0].first;
                vv_cell_begin[tid][level].push_back(0);
                for (uint i = 1; i < vv_index_maps[tid][level].size(); i++) {
                    if (vv_index_maps[tid][level][i].first != last_index) {
                        vv_cell_begin[tid][level].push_back(i);
                        last_index = vv_index_maps[tid][level][i].first;
                    }
                }
                 */
    /*

                ull last_index = vv_value_maps[tid][level][vv_index_maps[tid][level][0]];
                vv_cell_begin[tid][level].push_back(0);
                for (uint i = 1; i < vv_value_maps[tid][level].size(); i++) {
                    if (vv_value_maps[tid][level][vv_index_maps[tid][level][i]] != last_index) {
                        vv_cell_begin[tid][level].push_back(i);
                        last_index = vv_value_maps[tid][level][vv_index_maps[tid][level][i]];
                    }
                }
//                #pragma omp critical
//                std::cout << "Thread " << tid << " with " << vv_cell_begin[tid][level].size() << " cells at level " << level
//                          << std::endl;
                v_no_cells[tid][level] = vv_cell_begin[tid][level].size();
                vv_cell_ns[tid][level].reserve(v_no_cells[tid][level]);
                for (uint i = 0; i < v_no_cells[tid][level]; i++) {
                    uint begin = vv_cell_begin[tid][level][i];
                    uint end = (i == (v_no_cells[tid][level] - 1)) ? vv_index_maps[tid][level].size()
                                                                   : vv_cell_begin[tid][level][i + 1];
                    vv_cell_ns[tid][level][i] = end - begin;
                }
                if (level == 0) {
                    for (int i = 0; i < v_no_cells[tid][level]; ++i) {
                        if (vv_cell_ns[tid][level][i] > max_points_in_cell)
                            max_points_in_cell = vv_cell_ns[tid][level][i];
                    }
                }
            }
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        if (!g_quiet) {
            std::cout << "level " << " "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
                      << " milliseconds\n";
        }
        std::cout << "max points: " << max_points_in_cell << std::endl;
        for (uint t = 0; t < n_threads; ++t) {
            vv_range_tables[t] = new bool_vector(max_points_in_cell * max_points_in_cell);
        }
        return max_levels;
    }
     */

    /*
    void nextDBSCAN(struct_label **p_labels, float *v_coords, const uint m, const float e, const uint n,
            const uint max_d, bool_vector &is_core, uint n_threads) noexcept {

        std::cout << "Starting nextDBSCAN: " << n << " " << max_d << std::endl;

        auto t1 = std::chrono::high_resolution_clock::now();

        float e_inner = (e / 2);
        auto **vv_old = new std::vector<std::pair<ull, uint>> *[n_threads];
        auto **vv_index_maps = new std::vector<uint> *[n_threads];
        auto **vv_value_maps = new std::vector<ull> *[n_threads];
        auto **vv_cell_begin = new std::vector<uint> *[n_threads];
        auto **vv_cell_ns = new std::vector<uint> *[n_threads];
        auto v_no_cells = new std::vector<uint>[n_threads];
        auto **vv_cell_dims_min = new std::vector<float> *[n_threads];
        auto **vv_cell_dims_max = new std::vector<float> *[n_threads];
        auto **vv_range_tables = new bool_vector *[n_threads];
        auto t2 = std::chrono::high_resolution_clock::now();
            if (!g_quiet) {
            std::cout << "Memory and init: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
                      << " milliseconds\n";
        }

        t1 = std::chrono::high_resolution_clock::now();
//        int max_levels = index_points(v_coords, vv_index_maps, vv_value_maps, vv_old, vv_cell_begin, vv_cell_ns, v_no_cells,
//                vv_range_tables, max_d, n, n_threads, e_inner);
        int max_levels = build_grid_cell_tree(v_coords, vv_index_maps, vv_value_maps, vv_old, vv_cell_begin, vv_cell_ns,
                v_no_cells, vv_range_tables, max_d, n, n_threads, e_inner);
        t2 = std::chrono::high_resolution_clock::now();
        if (!g_quiet) {
            std::cout << "Point indexing: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
                      << " milliseconds\n";
        }

        t1 = std::chrono::high_resolution_clock::now();
//        calculate_cell_boundaries(v_coords, vv_old, vv_cell_begin, vv_cell_ns, v_no_cells, vv_cell_dims_min,
//                vv_cell_dims_max, max_levels, max_d, n_threads);
        t2 = std::chrono::high_resolution_clock::now();
        if (!g_quiet) {
            std::cout << "Calculate boundaries: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
                      << " milliseconds\n";
        }

        t1 = std::chrono::high_resolution_clock::now();
//        process_cell_tree(p_labels, v_coords, is_core, vv_old, vv_cell_begin, vv_cell_ns, v_no_cells, vv_cell_dims_min,
//                vv_cell_dims_max, vv_range_tables, max_levels, max_d, n_threads, e, n, m);
        t2 = std::chrono::high_resolution_clock::now();
        if (!g_quiet) {
            std::cout << "tree process: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
                      << " milliseconds\n";
        }
    }
*/
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
//        for (int i = 0; i < n; i++) {
//            if (is_core[i]) {
//                ++res.core_count;
//            }
//        }
        res.core_count = sum;
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

    int index_level_omp(const float *v_coords, uint *v_index_map, std::vector<uint> *v_index_lookup, ull *v_value_map,
            std::vector<uint> *vv_cell_begins,
            std::vector<uint> &v_cell_ns,
            const ull *v_dims_mult, const float eps_level,
            std::unique_ptr<float[]> &v_min_bounds,
            const uint n,
            const uint max_d, const uint n_threads, const int level) {
        uint max_points_in_cell = 0;
        std::vector<uint> v_block_sizes(n_threads);
        std::vector<uint> v_block_offsets(n_threads);
//        next_io::get_blocks_meta(v_block_sizes, v_block_offsets, n, n_threads);
        uint v_cell_no[n_threads];
        uint v_cell_offsets[n_threads];
        int total_cell_no = 0;
        std::fill(v_cell_no, v_cell_no+n_threads, 0);
        #pragma omp parallel default(shared) reduction(+:v_cell_no[:n_threads])
        {
            const uint tid = omp_get_thread_num();
            const uint size = v_block_sizes[tid];
            const uint offset = v_block_offsets[tid];
            for (uint i = offset; i < offset+size; ++i) {
                v_index_map[i] = i;
            }
            #pragma omp critical
            std::cout << "t: " << tid << " with block offset: " << offset << " and size: " << size << std::endl;
            for (uint i = offset; i < offset+size; ++i) {
                int level_mod = 1;
                uint p_index = i;
                while (level - level_mod >= 0) {
                    p_index = v_index_lookup[level-level_mod][vv_cell_begins[level-level_mod][p_index]];
                    ++level_mod;
                }
                uint coord_index = p_index*max_d;
                v_value_map[i] = get_cell_index(&v_coords[coord_index], v_min_bounds, v_dims_mult, max_d, eps_level);
            }
            std::sort(&v_index_map[offset], &v_index_map[offset+size], [&] (const auto &i1, const auto &i2) ->
            bool {
                return v_value_map[i1] < v_value_map[i2];
            });
            ull last_value = v_value_map[v_index_map[offset]];
            for (uint i = offset+1; i < offset+size; ++i) {
                if (v_value_map[v_index_map[i]] != last_value) {
                    last_value = v_value_map[v_index_map[i]];
                    ++v_cell_no[tid];
                }
            }
            // last cell
            ++v_cell_no[tid];
            #pragma omp atomic
            total_cell_no += v_cell_no[tid];
            // end parallel region
        }
#ifdef MPI_ON
//        MPI_Allreduce(&total_cell_no, &test, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
        std::cout << "Number of cells: " << total_cell_no << std::endl;
        vv_cell_begins[level].resize(total_cell_no);
        v_cell_ns.resize(total_cell_no);
        v_cell_offsets[0] = 0;
        for (uint i = 1; i < n_threads; ++i) {
            v_cell_offsets[i] = v_cell_offsets[i-1] + v_cell_no[i-1];
        }
        #pragma omp parallel default(shared) reduction(max: max_points_in_cell)
        {
            const uint tid = omp_get_thread_num();
            const uint size = v_block_sizes[tid];
            const uint offset = v_block_offsets[tid];
            assert(vv_cell_begins[level][v_cell_offsets[tid]] == 0);
            uint cell_offset = v_cell_offsets[tid];
            #pragma omp critical
            std::cout << "t: " << tid << " cell offset: " << cell_offset << " offset: " << offset << " and size: " << size << std::endl;
            vv_cell_begins[level][cell_offset++] = offset;
            ull last_value = v_value_map[v_index_map[offset]];


            for (uint i = offset+1; i < offset+size; ++i) {
                if (v_value_map[v_index_map[i]] != last_value) {
                    assert(vv_cell_begins[level][cell_offset] == 0);
                    vv_cell_begins[level][cell_offset++] = i;
                    assert(vv_cell_begins[level][cell_offset-1] < 110250);
                    last_value = v_value_map[v_index_map[i]];
                }
            }
            std::cout << "Level " << level << " cell offset: " << cell_offset << " and " << vv_cell_begins[0][1480] << std::endl;
            // last
//            ++cell_offset;
            assert(cell_offset - v_cell_offsets[tid] == v_cell_no[tid]);
            cell_offset = v_cell_offsets[tid];
            for (uint i = 0; i < v_cell_no[tid]; ++i, ++cell_offset) {
                uint begin = vv_cell_begins[level][cell_offset];
                uint end = (i == (v_cell_no[tid]-1))? size + offset: vv_cell_begins[level][cell_offset+1];
                assert(v_cell_ns[cell_offset] == 0);
                assert(end >= begin);
                v_cell_ns[cell_offset] = end-begin;
                if (level == 0) {
                    if (v_cell_ns[cell_offset] > max_points_in_cell) {
                        max_points_in_cell = v_cell_ns[cell_offset];
                    }
                }
            }
            // end parallel region
        }
        std::cout << "max points in cell: " << max_points_in_cell << std::endl;
        return total_cell_no;
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
            std::unique_ptr<float[]> &v_max_bounds, const uint n, const uint max_d, const float e_inner) {
        float max_limit = INT32_MIN;
        calc_bounds(v_coords, n, &v_min_bounds[0], &v_max_bounds[0], max_d);
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
        assert(v_index_map[0] == 0 && v_index_map[1] == 1);
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
        assert(cell_cnt == new_cells);
        v_cell_ns[cell_cnt-1] = v_value_map.size() - vv_cell_begin[level][cell_cnt-1];
        assert(cell_cnt == v_cell_ns.size());
        uint sum = 0;
        for (auto &val : v_cell_ns) {
            sum += val;
        }
        assert(sum == v_index_map.size());
        return new_cells;
    }


    void process_cell_tree_level(float *v_coords, std::vector<std::vector<uint>> &vv_cell_begins,
            std::vector<std::vector<uint>> &vv_cell_ns, std::vector<std::vector<uint>> &vv_index_maps,
            std::vector<uint> &v_leaf_cell_nns, std::vector<uint8_t > &v_cell_types, std::vector<bool> &v_is_core,
            std::vector<uint> &v_point_nns, std::vector<cell_meta_3> &stack3,
            std::vector<std::vector<float>> &vv_min_cell_dims, std::vector<std::vector<float>> &vv_max_cell_dims,
            uint level, const uint m, const uint max_d, const float e, const float e2) noexcept {
        if (level == 0) {
            for (uint i = 0; i < vv_cell_ns[0].size(); ++i) {
                uint begin = vv_cell_begins[0][i];
                uint size = vv_cell_ns[0][i];
//                std::cout << "Checkpoint #1 : " << i << " : " << begin << ", " << size << std::endl;
                v_leaf_cell_nns[i] = size;
                // TODO tmp, remove when cells are properly used
                for (uint j = 0; j < size; ++j) {
                    uint index = vv_index_maps[0][begin + j];
//                    std::cout << "index1 : " << index << std::endl;
                    v_point_nns[index] = size;
                }
                if (size >= m) {
                    v_cell_types[i] = AC;
                    for (uint j = 0; j < size; ++j) {
                        uint index = vv_index_maps[0][begin + j];
//                        std::cout << "index2 : " << index << std::endl;
                        v_is_core[index] = true;
                    }
                }
            }
            return;
        }
        for (uint i = 0; i < vv_cell_begins[level].size(); ++i) {
//            std::cout << "Checkpoint #2 : " << i << std::endl;
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
//                std::cout << "Checkpoint #3 : " << std::endl;
                if (!is_in_reach(&vv_min_cell_dims[l][c1 * max_d], &vv_max_cell_dims[l][c1 * max_d],
                        &vv_min_cell_dims[l][c2 * max_d], &vv_max_cell_dims[l][c2 * max_d], max_d, e)) {
                    continue;
                }
//                std::cout << "Checkpoint #4 : " << std::endl;
                uint begin1 = vv_cell_begins[l][c1];
                uint begin2 = vv_cell_begins[l][c2];
//                std::cout << "Checkpoint #5 : " << std::endl;
                if (l == 0) {
//                    std::cout << "Checkpoint #6 : " << std::endl;
                    for (uint k1 = 0; k1 < vv_cell_ns[0][c1]; ++k1) {
                        //                    assert((begin1+k1) < vv_index_values[0].size());
                        //                    assert(val1 == vv_index_values[0][vv_index_maps[0][begin1+k1]]);
                        for (uint k2 = 0; k2 < vv_cell_ns[0][c2]; ++k2) {
                            //                        assert((begin2+k2) < vv_index_values[0].size());
                            //                        assert(val2 == vv_index_values[0][vv_index_maps[0][begin2+k2]]);
                            uint index1 = vv_index_maps[0][begin1 + k1];
                            uint index2 = vv_index_maps[0][begin2 + k2];
                            if (v_is_core[index1] && v_is_core[index2])
                                continue;
                            if (dist_leq(&v_coords[index1 * max_d], &v_coords[index2 * max_d], max_d, e2)) {
                                ++v_point_nns[index1];
                                ++v_point_nns[index2];
                                if (v_point_nns[index1] >= m)
                                    v_is_core[index1] = true;
                                if (v_point_nns[index2] >= m)
                                    v_is_core[index2] = true;
                            }
                        }
                    }
//                    std::cout << "Checkpoint #7 : " << std::endl;
                } else {
//                    std::cout << "Checkpoint #8 : " << std::endl;
                    for (uint k1 = 0; k1 < vv_cell_ns[l][c1]; ++k1) {
                        uint c1_next = vv_index_maps[l][begin1 + k1];
                        for (uint k2 = 0; k2 < vv_cell_ns[l][c2]; ++k2) {
                            uint c2_next = vv_index_maps[l][begin2 + k2];
                            stack3.emplace_back(l - 1, c1_next, c2_next);
                        }
                    }
//                    std::cout << "Checkpoint #9 : " << std::endl;
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
            std::vector<std::vector<bool>> &vv_is_core, std::vector<std::vector<uint>> &vv_point_nns,
            std::vector<std::vector<std::vector<uint>>> &vvv_cell_ns, std::vector<cell_meta_5> &stack5,
            const int tid, const uint level, const uint max_d, const uint m, const uint n_cores, const float e,
            const float e2) noexcept {
        int tree_cnt = 0;
        int job_cnt = 0;
        for (uint t1 = 0; t1 < n_cores; ++t1) {
            for (uint i = 0; i < vvv_cell_ns[t1][level].size(); ++i) {
                for (uint t2 = t1+1; t2 < n_cores; ++t2) {
                    for (uint j = 0; j < vvv_cell_ns[t2][level].size(); ++j) {
                        // TODO use n_cores instead of n_threads
                        if ((int)((tree_cnt++)%n_cores) == tid) {
                            // TODO call another function immediately to reduce memory overhead
                            stack5.emplace_back(level, i, j, t1, t2);
//                            ++job_cnt;
                        }
                    }
                }
            }
        }
        #pragma omp critical
//        std::cout << "Thread t: " << tid << " job cnt: " << job_cnt << std::endl;
        while (!stack5.empty()) {
            uint l = stack5.back().l;
            uint t1 = stack5.back().t1;
            uint c1 = stack5.back().c1;
            uint t2 = stack5.back().t2;
            uint c2 = stack5.back().c2;
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
                for (uint k1 = 0; k1 < vvv_cell_ns[t1][0][c1]; ++k1) {
                    for (uint k2 = 0; k2 < vvv_cell_ns[t2][0][c2]; ++k2) {
                        uint index1 = vvv_index_maps[t1][0][begin1 + k1];
                        uint index2 = vvv_index_maps[t2][0][begin2 + k2];
                        if (vv_is_core[t1][index1] && vv_is_core[t2][index2])
                            continue;
                        if (dist_leq(&v_p_coords[t1][index1 * max_d], &v_p_coords[t2][index2 * max_d], max_d, e2)) {
                            #pragma omp atomic
                            ++vv_point_nns[t1][index1];
                            #pragma omp atomic
                            ++vv_point_nns[t2][index2];
                            if (vv_point_nns[t1][index1] >= m)
                                vv_is_core[t1][index1] = true;
                            if (vv_point_nns[t2][index2] >= m)
                                vv_is_core[t2][index2] = true;
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

    /*
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
     */
#ifdef MPI_ON
    void mpi_coord_merge(std::unique_ptr<float[]> &v_coords, const uint blocks_no, const uint total_samples,
            const uint max_d) {
        auto v_block_sizes = std::make_unique<uint[]>(blocks_no);
        auto v_block_offsets = std::make_unique<uint[]>(blocks_no);
        deep_io::get_blocks_meta(v_block_sizes, v_block_offsets, total_samples, blocks_no);;
        auto v_block_sizes_in_bytes = std::make_unique<int[]>(blocks_no);
        auto v_block_offsets_in_bytes = std::make_unique<int[]>(blocks_no);
        for (uint i = 0; i < blocks_no; ++i) {
            v_block_sizes_in_bytes[i] = v_block_sizes[i] * max_d;
            v_block_offsets_in_bytes[i] = v_block_offsets[i] * max_d;
        }
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &v_coords[0], &v_block_sizes_in_bytes[0],
                &v_block_offsets_in_bytes[0], MPI_FLOAT, MPI_COMM_WORLD);
    }

    void mpi_gather_int_array() {

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
        std::vector<T> v_payload(elems_to_send, -1);
        int index = 0;
        for (int t = 0; t < n_cores; ++t) {
            for (int l = 0; l < max_levels; ++l) {
                for (auto &val : vvv_cell_tree[t][l]) {
                    if (t / n_threads == mpi_index) {
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
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &v_payload[0], m_sizes,
                m_offsets, send_type, MPI_COMM_WORLD);
        index = 0;
        for (int t = 0; t < n_cores; ++t) {
            for (int l = 0; l < max_levels; ++l) {
                for (int i = 0; i < vvv_cell_tree[t][l].size(); ++i) {
                    assert(v_payload[index] != (T)-1);
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
        // TODO omp
        uint index = t_offset * max_levels;
        for (uint t = t_offset; t < (n_threads + t_offset); ++t) {
            for (uint l = 0; l < max_levels; ++l, ++index) {
//                if (vvv_index_maps[t][l].empty())
//                    std::cout << "mpi_index: " << mpi_index << " t: " << t << " level: " << l << " is empty!" << std::endl;
                assert(!vvv_index_maps[t][l].empty());
                core_level_elems[index] += vvv_index_maps[t][l].size();
            }
        }

//        print_array("elem cnts before: ", core_level_elems, total_levels);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, core_level_elems, total_levels / mpi_size,
                MPI_INT, MPI_COMM_WORLD);
//        if (mpi_index == 0)
//        print_array("elem cnts after: ", core_level_elems, total_levels);
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

        mpi_gather_cell_tree(vvv_index_maps, n_cores, max_levels, mpi_size, n_threads, mpi_index, MPI_INT, false);
        MPI_Barrier(MPI_COMM_WORLD);
        mpi_gather_cell_tree(vvv_value_maps, n_cores, max_levels, mpi_size, n_threads, mpi_index, MPI_LONG_LONG, false);
        MPI_Barrier(MPI_COMM_WORLD);
        mpi_gather_cell_tree(vvv_cell_begins, n_cores, max_levels, mpi_size, n_threads, mpi_index, MPI_INT, false);
        MPI_Barrier(MPI_COMM_WORLD);
        mpi_gather_cell_tree(vvv_cell_ns, n_cores, max_levels, mpi_size, n_threads, mpi_index, MPI_INT, false);
        MPI_Barrier(MPI_COMM_WORLD);
        mpi_gather_cell_tree(vvv_min_cell_dims, n_cores, max_levels, mpi_size, n_threads, mpi_index, MPI_FLOAT, true);
        MPI_Barrier(MPI_COMM_WORLD);
        mpi_gather_cell_tree(vvv_max_cell_dims, n_cores, max_levels, mpi_size, n_threads, mpi_index, MPI_FLOAT, true);
        MPI_Barrier(MPI_COMM_WORLD);
        std::cout << "Transmit done" << std::endl;
    }
#endif

    result start(const uint m, const float e, const uint n_threads, const std::string &in_file,
            const uint block_index, const uint blocks_no) noexcept {
        auto time1 = std::chrono::high_resolution_clock::now();
        omp_set_num_threads(n_threads);
        uint n, max_d;
        uint n_cores = n_threads * blocks_no;
        if (block_index == 0)
            std::cout << "Total of " << n_cores << " cores used." << std::endl;
        std::unique_ptr<float[]> v_coords;
        uint total_samples = process_input(in_file, v_coords, n, max_d, blocks_no, block_index);
#ifdef MPI_ON
        if (blocks_no > 1) {
            mpi_coord_merge(v_coords, blocks_no, total_samples, max_d);
        }
#endif
        auto time2 = std::chrono::high_resolution_clock::now();
        if (!g_quiet) {
            std::cout << "Input read: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1).count()
                      << " milliseconds\n";
        }
        std::cout << "Found " << n << " points in " << max_d << " dimensions" << " and read " << n <<
            " of " << total_samples << " samples." << std::endl;
        const auto e_inner = (e / 2);
        const auto e2 = e * e;

        auto v_min_bounds = std::make_unique<float[]>(max_d);
        auto v_max_bounds = std::make_unique<float[]>(max_d);
        const int max_levels = determine_data_boundaries(v_coords, v_min_bounds, v_max_bounds, total_samples, max_d, e_inner);
        auto v_eps_levels = std::make_unique<float[]>(max_levels);
        auto v_dims_mult = std::make_unique<ull[]>(max_levels * max_d);
        for (int l = 0; l < max_levels; l++) {
            v_eps_levels[l] = (e_inner * powf(2, l));
            calc_dims_mult(&v_dims_mult[l*max_d], max_d, v_min_bounds, v_max_bounds, v_eps_levels[l]);
        }
        auto v_omp_block_sizes = std::make_unique<uint[]>(n_cores);
        auto v_omp_block_offsets = std::make_unique<uint[]>(n_cores);
        deep_io::get_blocks_meta(v_omp_block_sizes, v_omp_block_offsets, total_samples, n_cores);

        // thread_id x level x elems
        // TODO make space for other node threads
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
        // TODO this can be n_threads
        std::vector<cell_meta_3> stacks3[n_cores];
        std::vector<cell_meta_5> stacks5[n_cores];
        std::vector<std::vector<uint>> vv_leaf_cell_nns(n_cores);
        std::vector<std::vector<uint8_t>> vv_cell_types(n_cores);

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
            if (block_index == 0)
                std::cout << "t: " << t << " pointer: " << v_omp_block_offsets[t]*max_d << std::endl;
            v_p_coords[t] = &v_coords[v_omp_block_offsets[t]*max_d];
        }
        auto time3 = std::chrono::high_resolution_clock::now();
        if (!g_quiet) {
            std::cout << "Memory init: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(time3 - time2).count()
                      << " milliseconds\n";
        }
        #pragma omp parallel
        {
            int tid = omp_get_thread_num() + (int) (block_index * n_threads);
            uint size = v_omp_block_sizes[tid];
            for (uint l = 0; l < max_levels; ++l) {
                vvv_index_maps[tid][l].resize(size);
                vvv_value_maps[tid][l].resize(size);
//                if (block_index == 0)
//                    std::cout << "mpi_index: " << block_index << " t: " << tid << " level #" << l << " size: " << size << std::endl;
                size = index_level(v_p_coords[tid], vvv_index_maps[tid][l], vvv_value_maps[tid][l],
                        vvv_index_maps[tid], vvv_cell_begins[tid], vvv_cell_ns[tid][l], v_min_bounds,
                        &v_dims_mult[l * max_d], l, max_d, v_eps_levels[l]);
                calculate_level_cell_bounds(v_p_coords[tid], vvv_cell_begins[tid][l], vvv_cell_ns[tid][l],
                        vvv_index_maps[tid][l], vvv_min_cell_dims[tid], vvv_max_cell_dims[tid], max_d, l);
            }
            uint leaf_cell_no = vvv_cell_ns[tid][0].size();
            stacks3[tid].reserve(leaf_cell_no * (uint) std::max((int) logf(max_d), 1) / n_threads);
            stacks5[tid].reserve(leaf_cell_no * (uint) std::max((int) logf(max_d), 1) / n_threads);
            vv_leaf_cell_nns[tid].resize(leaf_cell_no, 0);
            vv_cell_types[tid].resize(leaf_cell_no, NC);
#pragma omp barrier

#ifdef MPI_ON
            if (blocks_no > 1) {
                #pragma omp single
                {
//                    std::vector<int> core_level_elems;
//            uint index = 0;
//                    for (uint t = 0; t < n_cores; ++t) {
//                        for (uint l = 0; l < max_levels; ++l) {
//                            core_level_elems.push_back(vvv_index_maps[t][l].size());
//                        }
//                    }
//                print_array("level sizes: ", &core_level_elems[0], core_level_elems.size());
                    if (blocks_no > 1) {
                        mpi_cell_trees_merge(vvv_index_maps, vvv_value_maps, vvv_cell_begins, vvv_cell_ns,
                                vvv_min_cell_dims, vvv_max_cell_dims, block_index, blocks_no, n_threads, n_cores,
                                max_levels, max_d);
                    }
                }
            }
#endif
            vv_leaf_cell_nns[tid].resize(vvv_cell_ns[tid][0].size(), 0);
            vv_cell_types[tid].resize(vvv_cell_ns[tid][0].size(), NC);
            for (uint l = 0; l < max_levels; ++l) {
                process_cell_tree_level(v_p_coords[tid], vvv_cell_begins[tid], vvv_cell_ns[tid], vvv_index_maps[tid],
                        vv_leaf_cell_nns[tid], vv_cell_types[tid], vv_is_core[tid], vv_point_nns[tid], stacks3[tid],
                        vvv_min_cell_dims[tid], vvv_max_cell_dims[tid], l, m, max_d, e, e2);
            }
            #pragma omp barrier
            process_cell_tree_pairs(v_p_coords, vvv_index_maps, vvv_value_maps, vvv_cell_begins,
                    vvv_min_cell_dims, vvv_max_cell_dims, vv_is_core, vv_point_nns, vvv_cell_ns, stacks5[tid], tid,
                    max_levels - 2, max_d, m, n_cores, e, e2);
        // end of parallel region
        }
#ifdef MPI_ON
        if (blocks_no > 1) {

        }
#endif

        /*
        std::cout << "Check for " << total_samples << " points." << std::endl;
        std::vector<bool> check(total_samples, false);
        for (uint t = 0; t < n_cores; ++t) {
            uint level = max_levels-1;
            stacks3[t].emplace_back(level, 0, 0);
            while (!stacks3[t].empty()) {
                uint l = stacks3[t].back().l;
                uint c = stacks3[t].back().c1;
//                uint c2 = stack3.back().c2;
                stacks3[t].pop_back();
                uint begin = vvv_cell_begins[t][l][c];
                if (l == 0) {
                    for (uint i = 0; i < vvv_cell_ns[t][l][c]; ++i) {
                        uint index = v_omp_block_offsets[t] + vvv_index_maps[t][l][begin + i];
                        assert(index < total_samples);
//                        assert(!check[index]);
                        check[index] = true;
                    }
                } else {
                    for (uint i = 0; i < vvv_cell_ns[t][l][c]; ++i) {
                        assert(begin+i < vvv_index_maps[t][l].size());
                        stacks3[t].emplace_back(l-1, vvv_index_maps[t][l][begin + i], 0);
                    }
                }
            }
        }
        for (uint i = 0; i < check.size(); ++i) {
            assert(check[i]);
        }
        */

//        std::cout << "Processing pairs" << std::endl;

//            for (uint l = 0; l < max_levels; ++l) {
//                process_cell_tree_level(v_p_coords[tid], vvv_cell_begins[tid], vvv_cell_ns[tid], vvv_index_maps[tid],
//                        vv_leaf_cell_nns[tid], vv_cell_types[tid], vv_is_core[tid], vv_point_nns[tid], stacks3[tid],
//                        vvv_min_cell_dims[tid], vvv_max_cell_dims[tid], l, m, max_d, e, e2);
//            }
//            #pragma omp barrier
//            process_cell_tree_pairs(v_p_coords, vvv_index_maps, vvv_value_maps, vvv_cell_begins,
//                    vvv_min_cell_dims, vvv_max_cell_dims, vv_is_core, vv_point_nns, vvv_cell_ns, stacks5[tid], tid,
//                    max_levels-3, max_d, m, n_cores, e, e2);
//        }
        auto time4 = std::chrono::high_resolution_clock::now();
        if (!g_quiet) {
            std::cout << "Process cell trees: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(time4 - time3).count()
                      << " milliseconds\n";
        }
        auto time5 = std::chrono::high_resolution_clock::now();
        if (!g_quiet) {
            std::cout << "Total Execution Time: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(time5 - time1).count()
                      << " milliseconds\n";
        }
        return calculate_output(vv_is_core, vv_labels[0], v_omp_block_sizes[0]);
    }






    result start_old(const uint m, const float e, const uint n_threads, const std::string &in_file,
            const uint block_index, const uint blocks_no) noexcept {
        auto t1 = std::chrono::high_resolution_clock::now();
        omp_set_num_threads(n_threads);
        uint n, max_d;
        std::unique_ptr<float[]> v_coords;
        uint read_samples = process_input(in_file, v_coords, n, max_d, blocks_no, block_index);
        uint total_samples = n;
//        assert(total_samples == n);
        std::cout << "Found " << n << " points in " << max_d << " dimensions" << " and read " << read_samples << std::endl;
        std::vector<uint> v_block_sizes(blocks_no);
        std::vector<uint> v_block_offsets(blocks_no);
//        next_io::get_blocks_meta(v_block_sizes, v_block_offsets, n, blocks_no);;
        assert(v_block_sizes.size() == v_block_offsets.size());
        std::cout << "number of blocks: " << blocks_no << std::endl;
        for (uint i = 0; i < blocks_no; ++i) {
            std::cout << "block #" << i << " with sample size: " << v_block_sizes[i] << " and offset: " <<
                v_block_offsets[i] << std::endl;
        }
        // TODO MPI merge
#ifdef MPI_ON
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
#endif
        // Index points
        const float e_inner = (e / 2);
        const float e2 = e * e;
        float max_limit = INT32_MIN;
        auto v_min_bounds = std::make_unique<float[]>(max_d);
        auto v_max_bounds = std::make_unique<float[]>(max_d);
//        calc_bounds(v_coords, n, v_min_bounds, v_max_bounds, max_d);
        for (uint d = 0; d < max_d; d++) {
            if (v_max_bounds[d] - v_min_bounds[d] > max_limit)
                max_limit = v_max_bounds[d] - v_min_bounds[d];
        }
        int max_levels = static_cast<int>(ceilf(logf(max_limit / e_inner) / logf(2))) + 1;
        auto v_eps_levels = std::make_unique<float[]>(max_levels);
        auto v_dims_mult = std::make_unique<ull[]>(max_levels * max_d);
        for (int l = 0; l < max_levels; l++) {
            v_eps_levels[l] = (e_inner * powf(2, l));
            calc_dims_mult(&v_dims_mult[l*max_d], max_d, v_min_bounds, v_max_bounds, v_eps_levels[l]);
        }
        std::vector<uint> vv_cell_begins[max_levels];
        std::vector<uint> vv_cell_ns[max_levels];
        std::vector<uint> vv_index_maps[max_levels];
        std::vector<ull> vv_index_values[max_levels];
        std::vector<float> vv_min_cell_dims[max_levels];
        std::vector<float> vv_max_cell_dims[max_levels];
        uint offset = v_block_offsets[block_index];
//        offset = 0;
        uint index_offset = offset;
        uint resize_val = total_samples;
        for (uint l = 0; l < max_levels; ++l) {
            std::cout << "START OF LEVEL #" << l << " with offset: " << index_offset << " and size: " << resize_val << std::endl;
            vv_index_maps[l].resize(resize_val);
            vv_index_values[l].resize(resize_val);
            resize_val = index_level_omp(&v_coords[offset], &vv_index_maps[l][index_offset], vv_index_maps,
                    &vv_index_values[l][index_offset], vv_cell_begins, vv_cell_ns[l],
                    &v_dims_mult[l*max_d], v_eps_levels[l], v_min_bounds,
                    n, max_d, n_threads, l);
            std::cout << "Calculating bounds" << std::endl;
//            calculate_level_cell_bounds(v_coords, vv_cell_begins[l], vv_cell_ns[l], vv_index_maps[l],
//                    vv_min_cell_dims, vv_max_cell_dims, max_d, l);
            for (auto &elem : vv_min_cell_dims[l]) {
                assert(elem != UNDEFINED_VALUE);
            }
            for (auto &elem : vv_max_cell_dims[l]) {
                assert(elem != UNDEFINED_VALUE);
            }
            if (l == 0) {
                uint cnt = 0;
                for (auto &ns : vv_cell_ns[l]) {
                    cnt += ns;
                }
                assert(cnt == n);
                assert(vv_cell_begins[l].size() == vv_cell_ns[l].size());

                std::cout << "size check: " << vv_cell_begins[l].size() << std::endl;
            }
//            calculate_cell_boundaries_level(&v_points[0], vv_cell_begins, vv_cell_ns[l]);
            n = resize_val;
            // TODO MPI offset
            index_offset = 0;
            // TODO MPI gather values
        }
        // TODO MPI offset

        // TREE PARSING
        auto v_labels = std::make_unique<struct_label[]>(total_samples);
        auto v_is_core = std::make_unique<bool[]>(total_samples);
        auto v_point_nns = std::make_unique<uint[]>(total_samples);
        uint leaf_cell_no = vv_cell_begins[0].size();
        auto v_leaf_cell_nns = std::make_unique<uint[]>(leaf_cell_no);
        auto v_cell_types = std::make_unique<uint8_t[]>(leaf_cell_no);
        std::vector<cell_meta_3> stacks3[n_threads];
        std::vector<cell_meta_5> stacks5[n_threads];
        for (uint t = 0; t < n_threads; t++) {
            stacks3[t].reserve(leaf_cell_no * (uint) std::max((int) logf(max_d), 1) / n_threads);
            stacks5[t].reserve(leaf_cell_no * (uint) std::max((int) logf(max_d), 1) / n_threads);
        }
        assert(v_point_nns[4] == 0 && v_point_nns[6] == 0);
        uint cnt = 0;
        for (uint i = 0; i < vv_cell_begins[0].size(); ++i) {
            for (uint j = 0; j < vv_cell_ns[0][i]; ++j) {
                assert(!v_is_core[vv_cell_begins[0][i] + j]);
                v_is_core[vv_cell_begins[0][i] + j] = true;
                ++cnt;
            }
        }
        assert(cnt == total_samples);
        std::fill(&v_is_core[0], &v_is_core[0]+total_samples, false);

        std::cout << "cell_begins 0: " << vv_cell_begins[0].size() << std::endl;
        std::cout << "cell values: " << vv_index_values[0].size() << std::endl;
        std::cout << "cell maps: " << vv_index_maps[0].size() << std::endl;

        assert(vv_cell_begins[max_levels-1].size() == n_threads);

        #pragma omp parallel for
        for (uint i = 0; i < leaf_cell_no; ++i) {
            uint begin = vv_cell_begins[0][i];
            uint size = vv_cell_ns[0][i];
            v_leaf_cell_nns[i] = size;
            // TODO tmp
            for (uint j = 0; j < size; ++j) {
                uint index = vv_index_maps[0][begin+j];
                v_point_nns[index] = size;
            }
            if (size >= m) {
                v_cell_types[i] = AC;
                for (uint j = 0; j < size; ++j) {
                    uint index = vv_index_maps[0][begin+j];
                    v_is_core[index] = true;
                }
            }
        }

        for (uint level = 1; level < max_levels; ++level) {
            #pragma omp parallel for schedule(dynamic)
            for (uint i = 0; i < vv_cell_begins[level].size(); ++i) {
                uint tid = omp_get_thread_num();
                uint begin = vv_cell_begins[level][i];
                for (uint c1 = 0; c1 < vv_cell_ns[level][i]; ++c1) {
                    uint c1_index = vv_index_maps[level][begin + c1];
                    for (uint c2 = c1 + 1; c2 < vv_cell_ns[level][i]; ++c2) {
                        stacks3[tid].emplace_back(level-1, c1_index, vv_index_maps[level][begin+c2]);
                    }
                }
                while (!stacks3[tid].empty()) {
                    uint l = stacks3[tid].back().l;
                    uint c1 = stacks3[tid].back().c1;
                    uint c2 = stacks3[tid].back().c2;
                    stacks3[tid].pop_back();
                    if (!is_in_reach(&vv_min_cell_dims[l][c1 * max_d], &vv_max_cell_dims[l][c1 * max_d],
                            &vv_min_cell_dims[l][c2 * max_d], &vv_max_cell_dims[l][c2 * max_d], max_d, e)) {
                        continue;
                    }
                    uint begin1 = vv_cell_begins[l][c1];
                    uint begin2 = vv_cell_begins[l][c2];
                    if (l == 0) {
                        for (uint k1 = 0; k1 < vv_cell_ns[0][c1]; ++k1) {
    //                    assert((begin1+k1) < vv_index_values[0].size());
    //                    assert(val1 == vv_index_values[0][vv_index_maps[0][begin1+k1]]);
                            for (uint k2 = 0; k2 < vv_cell_ns[0][c2]; ++k2) {
    //                        assert((begin2+k2) < vv_index_values[0].size());
    //                        assert(val2 == vv_index_values[0][vv_index_maps[0][begin2+k2]]);
                                uint index1 = vv_index_maps[0][begin1 + k1];
                                uint index2 = vv_index_maps[0][begin2 + k2];
                                if (v_is_core[index1] && v_is_core[index2])
                                    continue;
                                if (dist_leq(&v_coords[index1 * max_d], &v_coords[index2 * max_d], max_d, e2)) {
                                    ++v_point_nns[index1];
                                    ++v_point_nns[index2];
                                    if (v_point_nns[index1] >= m)
                                        v_is_core[index1] = true;
                                    if (v_point_nns[index2] >= m)
                                        v_is_core[index2] = true;
                                }
                            }
                        }
                    } else {
                        for (uint k1 = 0; k1 < vv_cell_ns[l][c1]; ++k1) {
                            uint c1_next = vv_index_maps[l][begin1 + k1];
                            for (uint k2 = 0; k2 < vv_cell_ns[l][c2]; ++k2) {
                                uint c2_next = vv_index_maps[l][begin2 + k2];
                                stacks3[tid].emplace_back(l - 1, c1_next, c2_next);
                            }
                        }
                    }
                }
            }
        }

        /*
        #pragma omp parallel for
        for (uint i = 0; i < leaf_cell_no; ++i) {
            uint begin = vv_cell_begins[0][i];
            uint size = vv_cell_ns[0][i];
//            std::cout << "size: " << size << std::endl;
            for (uint j = 0; j < size; ++j) {
                uint index = vv_index_maps[0][begin+j];
                v_point_nns[index] = size;
                if (v_point_nns[index] >= m)
                    v_is_core[index] = true;
            }
        }
        std::cout << vv_min_cell_dims[0].size() << " is size" << std::endl;
        std::cout << leaf_cell_no << " is leaf size" << std::endl;
//        assert(vv_min_cell_dims[0].size() == leaf_cell_no*max_d);
//        assert(vv_max_cell_dims[0].size() == leaf_cell_no*max_d);
        uint total_check = 0;
        cnt = 0;

        #pragma omp parallel for reduction(+: cnt)
        for (uint i = 0; i < leaf_cell_no; ++i) {
            uint begin1 = vv_cell_begins[0][i];
            uint index1 = vv_index_maps[0][begin1];
//            assert((begin1) < vv_index_values[0].size());
//            ull val1 = vv_index_values[0][vv_index_maps[0][begin1]];
            for (uint j = i+1; j < leaf_cell_no; ++j) {
                uint begin2 = vv_cell_begins[0][j];
//                assert((begin2) < vv_index_values[0].size());
//                ull val2 = vv_index_values[0][vv_index_maps[0][begin2]];

                uint index2 = vv_index_maps[0][begin2];
                if (!is_in_reach(&vv_min_cell_dims[0][i*max_d], &vv_max_cell_dims[0][i*max_d],
                        &vv_min_cell_dims[0][j*max_d], &vv_max_cell_dims[0][j*max_d], max_d, e)) {
                    ++cnt;
                } else {
                    for (uint k1 = 0; k1 < vv_cell_ns[0][i]; ++k1) {
//                    assert((begin1+k1) < vv_index_values[0].size());
//                    assert(val1 == vv_index_values[0][vv_index_maps[0][begin1+k1]]);
                        for (uint k2 = 0; k2 < vv_cell_ns[0][j]; ++k2) {
//                        assert((begin2+k2) < vv_index_values[0].size());
//                        assert(val2 == vv_index_values[0][vv_index_maps[0][begin2+k2]]);
                            index1 = vv_index_maps[0][begin1 + k1];
                            index2 = vv_index_maps[0][begin2 + k2];
                            if (v_is_core[index1] && v_is_core[index2])
                                continue;
                            if (dist_leq(&v_coords[index1 * max_d], &v_coords[index2 * max_d], max_d, e2)) {
                                #pragma omp atomic
                                ++v_point_nns[index1];
                                #pragma omp atomic
                                ++v_point_nns[index2];
                                if (v_point_nns[index1] >= m)
                                    v_is_core[index1] = true;
                                if (v_point_nns[index2] >= m)
                                    v_is_core[index2] = true;
                            }
                        }
                    }
                }
            }
        }

        std::cout << "failed reach checks: " << cnt << std::endl;
        std::cout << "total pairs: " << total_check << std::endl;
        */
        auto t2 = std::chrono::high_resolution_clock::now();
        if (!g_quiet) {
            std::cout << "Execution time (excluding I/O): "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
                      << " milliseconds\n";
        }

//        return calculate_output(v_is_core, v_labels, total_samples);

    }

}