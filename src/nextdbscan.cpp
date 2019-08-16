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
#include "nextdbscan.h"

namespace nextdbscan {

    static const int UNASSIGNED = -1;
    static const int TYPE_NC = 0;
    static const int TYPE_AC = 1;
    static const int TYPE_SC = 2;

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

//    int label_counter = 0;

    struct_label *get_label(struct_label *p) noexcept {
        struct_label *p_origin = p;
        while (p->label_p != nullptr) {
//            assert(p->label_p != p);
//            ++label_counter;
            p = p->label_p;
        }
        if (p_origin->label_p != nullptr && p_origin->label_p != p) {
            p_origin->label_p = p;
        }
        return p;
    }

    void calc_bounds(const float *v_coords, uint n, std::vector<float> &min_bounds, std::vector<float> &max_bounds,
                uint max_d) noexcept {
        for (uint d = 0; d < max_d; d++) {
            min_bounds[d] = INT32_MAX;
            max_bounds[d] = INT32_MIN;
        }
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

    inline void calc_dims_mult(ull *dims_mult, const uint max_d, const std::vector<float> &min_bounds,
            const std::vector<float> &max_bounds, const float e_inner) noexcept {
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

    inline ull get_cell_index(const float *dv, const std::vector<float> &mv, const ull *dm, const uint max_d,
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
        for (uint d = 0; d < max_d; d++) {
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

    uint mark_in_range(const float *v_coords, const uint *v_c1_index, const uint size1, const uint *v_c2_index,
                      const uint size2, bool *range_table, const uint max_d, const float e2) noexcept {
        std::fill(range_table, range_table + (size1 * size2), false);
        uint cnt_range = 0;
        uint index = 0;
        for (uint i = 0; i < size1; i++) {
            for (uint j = 0; j < size2; j++, index++) {
                if (dist_leq(&v_coords[v_c1_index[i] * max_d], &v_coords[v_c2_index[j] * max_d], max_d, e2)) {
                    ++cnt_range;
                    range_table[index] = true;
                }
            }
        }
        return cnt_range;
    }

    inline uint get_type(uint **cell_indexes, std::vector<uint> &v_point_nps, std::vector<uint> &v_cell_nps, uint c1,
            uint size, uint m) {
        for (uint j = 0; j < size; ++j) {
            if (v_cell_nps[c1] + v_point_nps[cell_indexes[c1][j]] < m)
                return false;
        }
        return true;
    }

    inline bool is_AC(uint **cell_indexes, std::vector<uint> &v_point_nps, std::vector<uint> &v_cell_nps, uint c1,
            uint size, uint m) {
        for (uint j = 0; j < size; ++j) {
            if (v_cell_nps[c1] + v_point_nps[cell_indexes[c1][j]] < m)
                return false;
        }
        return true;
    }

    inline bool is_SC(uint **cell_indexes, std::vector<uint> &v_point_nps, std::vector<uint> &v_cell_nps, const uint c1,
            const uint size, const uint m) {
        for (uint j = 0; j < size; ++j) {
            if (v_cell_nps[c1] + v_point_nps[cell_indexes[c1][j]] >= m)
                return true;
        }
        return false;
    }

    void update_type_AC(struct_label **p_labels, type_vector &cell_types, uint **cell_indexes,
            const uint *v_cell_ns, bool_vector &is_core, const uint c1) noexcept {
        for (uint j = 0; j < v_cell_ns[c1]; ++j) {
            is_core[cell_indexes[c1][j]] = true;
        }
        if (cell_types[c1] == TYPE_NC) {
            uint tmp_index = cell_indexes[c1][0];
            for (uint j = 1; j < v_cell_ns[c1]; ++j) {
                p_labels[cell_indexes[c1][j]]->label_p = p_labels[tmp_index];
            }
            struct_label *c1_label = get_label(p_labels[tmp_index]);
            if (c1_label->label == UNASSIGNED) {
                c1_label->label = c1;
            }
        }
        cell_types[c1] = TYPE_AC;
    }

    void update_type(struct_label **p_labels, type_vector &cell_types, uint **cell_indexes,
            std::vector<uint> &v_point_nps, std::vector<uint> &v_cell_nps, const uint *v_cell_ns, bool_vector &is_core,
            const uint c1, const uint m) noexcept {
        bool is_ac = true;
        bool is_sc = false;
//        assert(cell_types[c1] != TYPE_AC);
        uint core_index = cell_indexes[c1][0];
        if (v_cell_nps[c1] < m) {
            for (uint j = 0; j < v_cell_ns[c1]; ++j) {
                uint p_index = cell_indexes[c1][j];
                if (is_core[p_index])
                    continue;
                if (v_cell_nps[c1] + v_point_nps[p_index] >= m) {
                    is_core[p_index] = true;
                    if (!is_sc) {
                        is_sc = true;
                        core_index = cell_indexes[c1][j];
                    }
                } else {
                    is_ac = false;
                }
            }
        }
        if (is_ac || is_sc) {
            cell_types[c1] = is_ac? TYPE_AC : TYPE_SC;
            struct_label *c1_label = get_label(p_labels[core_index]);
            if (c1_label->label == UNASSIGNED) {
                c1_label->label = c1;
                for (uint j = 0; j < v_cell_ns[c1]; ++j) {
                    uint p_index = cell_indexes[c1][j];
                    if (is_ac)
                        is_core[p_index] = true;
                    if (p_index != core_index) {
//                        assert(p_labels[p_index]->label == UNASSIGNED);
//                        assert(p_labels[p_index]->label_p == nullptr);
                        p_labels[p_index]->label_p = p_labels[core_index];
                    }
                }
            } else if (is_ac) {
                for (uint j = 0; j < v_cell_ns[c1]; ++j) {
                    uint p_index = cell_indexes[c1][j];
                    is_core[p_index] = true;
                }
            }
        }
    }

    void process_ac_ac_pair(struct_label **p_labels, const float *v_coords, const uint *v_c1_index, const uint size1,
                            const uint *v_c2_index, const uint size2, const uint max_d, const float e2) noexcept {
        struct_label *c1_label = get_label(p_labels[v_c1_index[0]]);
        struct_label *c2_label = get_label(p_labels[v_c2_index[0]]);
        if (c1_label->label == c2_label->label)
            return;
        for (uint i = 0; i < size1; ++i) {
            for (uint j = 0; j < size2; ++j) {
                if (dist_leq(&v_coords[v_c1_index[i] * max_d], &v_coords[v_c2_index[j] * max_d], max_d, e2)) {
                    set_lower_label(c1_label, c2_label);
                    return;
                }
            }
        }
    }

    void process_ac_sc_pair(uint **cell_indexes, struct_label **p_labels, const float *v_coords, const uint *v_ac_index, const uint size_ac,
                       const uint *v_sc_index, const uint size_sc, const uint sc_id, const bool_vector &is_core, const uint max_d, const float e2) noexcept {
        struct_label *ac_label = get_label(p_labels[v_ac_index[0]]);
        struct_label *sc_label = get_label(p_labels[v_sc_index[0]]);
        if (ac_label->label == sc_label->label)
            return;
        for (uint i = 0; i < size_ac; i++) {
            for (uint j = 0; j < size_sc; j++) {
                if (is_core[cell_indexes[sc_id][j]]) {
                    if (dist_leq(&v_coords[v_ac_index[i] * max_d], &v_coords[v_sc_index[j] * max_d], max_d, e2)) {
                        set_lower_label(ac_label, sc_label);
                        return;
                    }
                }
            }
        }
    }

    void process_sc_sc_pair(uint **cell_indexes, struct_label **p_labels, const float *v_coords, const uint *v_c1_index, const uint id1, const uint size1,
                       const uint *v_c2_index, const uint id2, const uint size2, const bool_vector &is_core, const uint max_d, const float e2) noexcept {
        struct_label *c1_label = get_label(p_labels[v_c1_index[0]]);
        struct_label *c2_label = get_label(p_labels[v_c2_index[0]]);
        if (c1_label->label == c2_label->label)
            return;
        for (uint i = 0; i < size1; i++) {
            if (!is_core[cell_indexes[id1][i]]) {
                continue;
            }
            for (uint j = 0; j < size2; j++) {
                if (!is_core[cell_indexes[id2][j]]) {
                    continue;
                }
                if (dist_leq(&v_coords[v_c1_index[i] * max_d], &v_coords[v_c2_index[j] * max_d], max_d, e2)) {
                    set_lower_label(c1_label, c2_label);
                    return;
                }
            }
        }
    }

    void process_pair_one_nc(struct_label **p_labels, const float *v_coords, uint **cell_indexes, const uint *v_cell_ns,
            bool *range_table, bool_vector &is_processed, const bool_vector &is_core, const uint c1_id, const uint c1_type,
            const uint c2_id, const uint c2_type, const int max_d,
            const float e2) noexcept {
        uint size1 = v_cell_ns[c1_id];
        uint size2 = v_cell_ns[c2_id];

        uint cnt_range = mark_in_range(v_coords, cell_indexes[c1_id], size1, cell_indexes[c2_id], size2, range_table,
                                       max_d, e2);

        if (cnt_range == 0) {
            return;
        }
        if (cnt_range == size1 * size2) {
            if (c1_type != TYPE_NC) {
                auto *p1 = p_labels[cell_indexes[c1_id][0]];
                for (uint i = 0; i < v_cell_ns[c2_id]; i++) {
                    p_labels[cell_indexes[c2_id][i]]->label_p = p1;
                }
                is_processed[c2_id] = true;
            } else if (c2_type != TYPE_NC) {
                auto *p2 = p_labels[cell_indexes[c2_id][0]];
                for (uint i = 0; i < v_cell_ns[c1_id]; i++) {
                    p_labels[cell_indexes[c1_id][i]]->label_p = p2;
                }
                is_processed[c1_id] = true;
            }
        } else {
            uint index = 0;
            if (c1_type != TYPE_NC) {
                for (uint i = 0; i < size1; i++) {
                    int p1_id = cell_indexes[c1_id][i];
                    if (!is_core[p1_id]) {
                        index += size2;
                        continue;
                    }
                    auto *p1 = p_labels[p1_id];
                    for (uint j = 0; j < size2; j++, index++) {
                        if (range_table[index]) {
                            p_labels[cell_indexes[c2_id][j]]->label_p = p1;
                        }
                    }
                }
                // check for completeness
                bool is_complete = true;
                for (uint j = 0; j < size2; j++) {
                    if (p_labels[cell_indexes[c2_id][j]]->label_p == nullptr)
                        is_complete = false;
                }
                is_processed[c2_id] = is_complete;
            } else {
                for (uint i = 0; i < size1; i++) {
                    for (uint j = 0; j < size2; j++, index++) {
                        if (is_core[cell_indexes[c2_id][j]] && range_table[index]) {
                            auto *p2 = p_labels[cell_indexes[c2_id][j]];
                            p_labels[cell_indexes[c1_id][i]]->label_p = p2;
                            index += size2 - j - 1;
                            j = size2;
                        }
                    }
                }
                // check for completeness
                bool is_complete = true;
                for (uint j = 0; j < size1; j++) {
                    if (p_labels[cell_indexes[c1_id][j]]->label_p == nullptr)
                        is_complete = false;
                }
                is_processed[c1_id] = is_complete;
            }
        }
    }

    void apply_to_both_cells(uint **cell_indexes, std::vector<uint> &v_point_nps, const bool *range_table,
            const uint size1, const uint size2, const uint c1_id, const uint c2_id) {
        uint index = 0;
        for (uint i = 0; i < size1; i++) {
            for (uint j = 0; j < size2; j++, index++) {
                if (range_table[index]) {
                    ++v_point_nps[cell_indexes[c1_id][i]];
                    ++v_point_nps[cell_indexes[c2_id][j]];
                }
            }
        }
    }

    void apply_to_both_cells_synchronized(uint **cell_indexes, std::vector<uint> &v_point_nps, const bool *range_table,
                             const uint size1, const uint size2, const uint c1_id, const uint c2_id) {
        uint index = 0;
        for (uint i = 0; i < size1; i++) {
            for (uint j = 0; j < size2; j++, index++) {
                if (range_table[index]) {
                    #pragma omp atomic
                    ++v_point_nps[cell_indexes[c1_id][i]];
                    #pragma omp atomic
                    ++v_point_nps[cell_indexes[c2_id][j]];
                }
            }
        }
    }

    void apply_to_first_cell(uint **cell_indexes, std::vector<uint> &v_point_nps, const bool *range_table,
            const uint size1, const uint size2, const uint c1_id) {
        uint index = 0;
        for (uint i = 0; i < size1; i++) {
            for (uint j = 0; j < size2; j++, index++) {
                if (range_table[index]) {
                    ++v_point_nps[cell_indexes[c1_id][i]];
                }
            }
        }
    }

    void apply_to_first_cell_synchronized(uint **cell_indexes, std::vector<uint> &v_point_nps, const bool *range_table,
                       const uint size1, const uint size2, const uint c1_id) {
        uint index = 0;
        for (uint i = 0; i < size1; i++) {
            for (uint j = 0; j < size2; j++, index++) {
                if (range_table[index]) {
                    #pragma omp atomic
                    ++v_point_nps[cell_indexes[c1_id][i]];
                }
            }
        }
    }

    void apply_to_second_cell(uint **cell_indexes, std::vector<uint> &v_point_nps, const bool *range_table,
                             const uint size1, const uint size2, const uint c2_id) {
        uint index = 0;
        for (uint i = 0; i < size1; i++) {
            for (uint j = 0; j < size2; j++, index++) {
                if (range_table[index]) {
                    ++v_point_nps[cell_indexes[c2_id][j]];
                }
            }
        }
    }

    void apply_to_second_cell_synched(uint **cell_indexes, std::vector<uint> &v_point_nps, const bool *range_table,
                                     const uint size1, const uint size2, const uint c2_id) {
        uint index = 0;
        for (uint i = 0; i < size1; i++) {
            for (uint j = 0; j < size2; j++, index++) {
                if (range_table[index]) {
                    #pragma omp atomic
                    ++v_point_nps[cell_indexes[c2_id][j]];
                }
            }
        }
    }

//    uint test_cnt_1 = 0;
//    uint test_cnt_2 = 0;
//    uint test_cnt_3 = 0;
//    uint test_cnt_4 = 0;
//    uint test_cnt_5 = 0;

    void set_nc_full_label(struct_label **p_labels, uint c1, struct_label *label, uint index, uint **cell_indexes,
            const uint *v_cell_ns) {

        uint tmp_index = cell_indexes[c1][index];
        for (uint i = 0; i < v_cell_ns[c1]; ++i) {
            if (i == index)
                continue;
            p_labels[cell_indexes[c1][i]]->label_p = p_labels[tmp_index];
        }
        p_labels[tmp_index]->label_p = label;
    }

    // This function assumes that !v_cell_nps[c1_id] >= m and !v_cell_nps[c2_id] >= m
    int neighbour_count_non_ac_pair_2(struct_label **p_labels, const float *v_coords, uint **cell_indexes,
            const uint *v_cell_ns, bool *range_table, std::vector<uint> &v_point_nps, std::vector<uint> &v_cell_nps,
            type_vector &cell_types, bool_vector &is_core, const uint c1_id, const uint c2_id, const uint max_d,
            const float e2, const uint m, const bool is_sync) noexcept {
        uint size1 = v_cell_ns[c1_id];
        uint size2 = v_cell_ns[c2_id];
        uint cnt_range = mark_in_range(v_coords, cell_indexes[c1_id], size1, cell_indexes[c2_id], size2, range_table,
                max_d, e2);
        int ret_val = RET_NONE;
        if (cnt_range == 0) {
            return ret_val;
        }
        if (cnt_range == size1 * size2) {
            ret_val = RET_FULL;
            if (cell_types[c1_id] != TYPE_AC) {
                v_cell_nps[c1_id] += size2;
                update_type(p_labels, cell_types, cell_indexes, v_point_nps, v_cell_nps, v_cell_ns, is_core, c1_id, m);
            }
            if (cell_types[c2_id] != TYPE_AC) {
                v_cell_nps[c2_id] += size1;
                update_type(p_labels, cell_types, cell_indexes, v_point_nps, v_cell_nps, v_cell_ns, is_core, c2_id, m);
            }
            if (cell_types[c1_id] != TYPE_NC || cell_types[c2_id] != TYPE_NC) {
                struct_label *c1_label = get_label(p_labels[cell_indexes[c1_id][0]]);
                struct_label *c2_label = get_label(p_labels[cell_indexes[c2_id][0]]);
                if (cell_types[c1_id] == TYPE_NC) {
                    if (c1_label->label == UNASSIGNED) {
                        for (uint i = 0; i < v_cell_ns[c1_id]; ++i) {
                            p_labels[cell_indexes[c1_id][i]]->label_p = c2_label;
                        }
                        ret_val = RET_NONE;
                    }
                } else if (cell_types[c2_id] == TYPE_NC) {
                    if (c2_label->label == UNASSIGNED) {
                        for (uint i = 0; i < v_cell_ns[c2_id]; ++i) {
                            p_labels[cell_indexes[c2_id][i]]->label_p = c1_label;
                        }
                        ret_val = RET_NONE;
                    }
                } else {
                    if (c1_label->label != c2_label->label) {
                        set_lower_label(c1_label, c2_label);
                    }
                    ret_val = RET_NONE;
                }
            }
        } else {
            ret_val = RET_PARTIAL;
            if (cell_types[c1_id] == TYPE_AC) {
                if (is_sync) {
                    apply_to_second_cell_synched(cell_indexes, v_point_nps, range_table, size1, size2, c2_id);
                } else {
                    apply_to_second_cell(cell_indexes, v_point_nps, range_table, size1, size2, c2_id);
                }
                update_type(p_labels, cell_types, cell_indexes, v_point_nps, v_cell_nps, v_cell_ns, is_core, c2_id, m);
            } else if (cell_types[c2_id] == TYPE_AC) {
                if (is_sync) {
                    apply_to_first_cell_synchronized(cell_indexes, v_point_nps, range_table, size1, size2, c1_id);
                } else {
                    apply_to_first_cell(cell_indexes, v_point_nps, range_table, size1, size2, c1_id);
                }
                update_type(p_labels, cell_types, cell_indexes, v_point_nps, v_cell_nps, v_cell_ns, is_core, c1_id, m);
            } else {
                if (is_sync) {
                    apply_to_both_cells_synchronized(cell_indexes, v_point_nps, range_table, size1, size2, c1_id, c2_id);
                } else {
                    apply_to_both_cells(cell_indexes, v_point_nps, range_table, size1, size2, c1_id, c2_id);
                }
                update_type(p_labels, cell_types, cell_indexes, v_point_nps, v_cell_nps, v_cell_ns, is_core, c1_id, m);
                update_type(p_labels, cell_types, cell_indexes, v_point_nps, v_cell_nps, v_cell_ns, is_core, c2_id, m);
            }
            if (cell_types[c1_id] == TYPE_AC && cell_types[c2_id] == TYPE_AC) {
                struct_label *c1_label = get_label(p_labels[cell_indexes[c1_id][0]]);
                struct_label *c2_label = get_label(p_labels[cell_indexes[c2_id][0]]);
                if (c1_label->label != c2_label->label) {
                    set_lower_label(c1_label, c2_label);
                }
                ret_val = RET_NONE;
            } else if (cell_types[c1_id] != TYPE_NC && cell_types[c2_id] != TYPE_NC) {
                struct_label *c1_label = get_label(p_labels[cell_indexes[c1_id][0]]);
                struct_label *c2_label = get_label(p_labels[cell_indexes[c2_id][0]]);
                if (c1_label->label == c2_label->label) {
                    ret_val = RET_NONE;
                } else {
                    uint index = 0;
                    for (uint i = 0; i < size1; i++) {
                        uint p1_index = cell_indexes[c1_id][i];
                        for (uint j = 0; j < size2; j++, index++) {
                            if (range_table[index]) {
                                uint p2_index = cell_indexes[c2_id][j];
                                if (is_core[p1_index] && is_core[p2_index]) {
                                    set_lower_label(c1_label, c2_label);
                                    return RET_NONE;
                                }
                            }
                        }
                    }
                }
            }
        }
        return ret_val;
    }

    void neighbour_count_non_ac_pair(const float *v_coords, uint **cell_indexes, const uint *v_cell_ns, bool *range_table,
            std::vector<uint> &v_point_nps, std::vector<uint> &v_cell_nps, const uint c1_id, const uint c2_id,
            const uint max_d, const float e2, const uint m, const bool is_sync) noexcept {
        uint size1 = v_cell_ns[c1_id];
        uint size2 = v_cell_ns[c2_id];
        uint cnt_range = mark_in_range(v_coords, cell_indexes[c1_id], size1, cell_indexes[c2_id], size2, range_table,
                max_d, e2);
        if (cnt_range == 0) {
            return;
        }
        if (cnt_range == size1 * size2) {
            if (v_cell_nps[c1_id] < m)
                v_cell_nps[c1_id] += size2;
            if (v_cell_nps[c2_id] < m)
                v_cell_nps[c2_id] += size1;
        } else {
            if (v_cell_nps[c1_id] >= m) {
                if (is_sync) {
                    apply_to_second_cell_synched(cell_indexes, v_point_nps, range_table, size1, size2, c2_id);
                } else {
                    apply_to_second_cell(cell_indexes, v_point_nps, range_table, size1, size2, c2_id);
                }
            } else if (v_cell_nps[c2_id] >= m) {
                if (is_sync) {
                    apply_to_first_cell_synchronized(cell_indexes, v_point_nps, range_table, size1, size2, c1_id);
                } else {
                    apply_to_first_cell(cell_indexes, v_point_nps, range_table, size1, size2, c1_id);
                }
            } else {
                if (is_sync) {
                    apply_to_both_cells_synchronized(cell_indexes, v_point_nps, range_table, size1, size2, c1_id, c2_id);
                } else {
                    apply_to_both_cells(cell_indexes, v_point_nps, range_table, size1, size2, c1_id, c2_id);
                }
            }
        }
    }

    inline uint traverse_and_get_cell_index(uint ***cell_indexes, const int l, const uint i) noexcept {
        int level_mod = 1;
        uint cell_index = i;
        while (l - level_mod >= 0) {
            cell_index = cell_indexes[l - level_mod][cell_index][0];
            ++level_mod;
        }
        return cell_index;
    }

    inline void allocate_resources(std::vector<float> &v_eps_levels, ull **dims_mult,
                                   const std::vector<float> &min_bounds, const std::vector<float> &max_bounds,
                                   const uint max_levels, const uint max_d, float e_inner) noexcept {
        for (uint i = 0; i < max_levels; i++) {
            v_eps_levels[i] = (e_inner * powf(2, i));
            dims_mult[i] = new ull[max_d];
            calc_dims_mult(dims_mult[i], max_d, min_bounds, max_bounds, v_eps_levels[i]);
        }
    }

    void calculate_cell_boundaries_omp(float *v_coords, uint ***cell_indexes, uint **cell_ns, float **cell_dims_min,
            float **cell_dims_max, const std::vector<uint> &v_no_of_cells, const uint max_levels,
            const uint max_d) noexcept {
        float *coord_min, *coord_max;
        for (uint l = 0; l < max_levels; l++) {
            cell_dims_min[l] = new float[v_no_of_cells[l] * max_d];
            cell_dims_max[l] = new float[v_no_of_cells[l] * max_d];
        }
        for (uint l = 0; l < max_levels; l++) {
            #pragma omp parallel for private(coord_min, coord_max)
            for (uint i = 0; i < v_no_of_cells[l]; i++) {
                if (l == 0) {
                    coord_min = &v_coords[cell_indexes[l][i][0] * max_d];
                    coord_max = &v_coords[cell_indexes[l][i][0] * max_d];
                } else {
                    coord_min = &cell_dims_min[l - 1][cell_indexes[l][i][0] * max_d];
                    coord_max = &cell_dims_max[l - 1][cell_indexes[l][i][0] * max_d];
                }
                std::copy(coord_min, coord_min + max_d, &cell_dims_min[l][i * max_d]);
                std::copy(coord_max, coord_max + max_d, &cell_dims_max[l][i * max_d]);
                for (uint j = 1; j < cell_ns[l][i]; j++) {
                    if (l == 0) {
                        coord_min = &v_coords[cell_indexes[l][i][j] * max_d];
                        coord_max = &v_coords[cell_indexes[l][i][j] * max_d];
                    } else {
                        coord_min = &cell_dims_min[l - 1][cell_indexes[l][i][j] * max_d];
                        coord_max = &cell_dims_max[l - 1][cell_indexes[l][i][j] * max_d];
                    }
                    for (uint d = 0; d < max_d; d++) {
                        if (coord_min[d] < cell_dims_min[l][i * max_d + d]) {
                            cell_dims_min[l][i * max_d + d] = coord_min[d];
                        }
                        if (coord_max[d] > cell_dims_max[l][i * max_d + d]) {
                            cell_dims_max[l][i * max_d + d] = coord_max[d];
                        }
                    }
                }
            }
        }
    }

    void print_cell_types(type_vector &cell_types, std::vector<uint> &v_no_of_cells) {
        uint types[3] {0};
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

    inline void double_capacity(uint *size, uint **s_levels, uint **s_c1_indexes, uint **s_c2_indexes, const uint t_id) {
        uint new_size = size[t_id] * 2;
        uint *new_levels = new uint[new_size];
        uint *new_c1s = new uint[new_size];
        uint *new_c2s = new uint[new_size];
        std::copy(s_levels[t_id], s_levels[t_id] + size[t_id], new_levels);
        std::copy(s_c1_indexes[t_id], s_c1_indexes[t_id] + size[t_id], new_c1s);
        std::copy(s_c2_indexes[t_id], s_c2_indexes[t_id] + size[t_id], new_c2s);
        delete[] s_levels[t_id];
        delete[] s_c1_indexes[t_id];
        delete[] s_c2_indexes[t_id];
        size[t_id] = new_size;
        s_levels[t_id] = new_levels;
        s_c1_indexes[t_id] = new_c1s;
        s_c2_indexes[t_id] = new_c2s;
    }

    void process_cell_tree_omp_2(struct_label **ps_origin, float *v_coords, uint ***cell_indexes, uint **cell_ns,
            float **cell_dims_min, float **cell_dims_max, const std::vector<uint> &v_no_of_cells, bool_vector &is_core,
            uint n_threads, uint max_levels, uint max_d, float e, float e2, uint m, const uint n) noexcept {
        uint max_points_in_cell = 0;
        std::vector<uint> v_cell_nps(v_no_of_cells[0]);
        auto **range_table = new bool *[n_threads];
        std::vector<uint> v_point_nps(n, 0);
        type_vector cell_types(v_no_of_cells[0]);
        cell_types.fill(v_no_of_cells[0], TYPE_NC);
        bool_vector is_cell_processed(v_no_of_cells[0]);
        is_cell_processed.fill(v_no_of_cells[0], false);

        #pragma omp parallel for reduction(max: max_points_in_cell)
        for (uint i = 0; i < v_no_of_cells[0]; ++i) {
            v_cell_nps[i] = cell_ns[0][i];
            if (v_cell_nps[i] > max_points_in_cell) {
                max_points_in_cell = v_cell_nps[i];
            }
            if (v_cell_nps[i] >= m) {
                update_type_AC(ps_origin, cell_types, cell_indexes[0], cell_ns[0], is_core, i);
            }
        }
        for (uint i = 0; i < n_threads; i++) {
            range_table[i] = new bool[max_points_in_cell * std::min(max_points_in_cell, m)];
        }

        auto **leaf_cell_ns = new uint *[max_levels];
        for (uint level = 0; level < max_levels; ++level) {
            leaf_cell_ns[level] = new uint[v_no_of_cells[level]];
            if (level == 0) {
                std::fill(leaf_cell_ns[0], leaf_cell_ns[0] + v_no_of_cells[0], 1);
                continue;
            }
            #pragma omp parallel for
            for (uint i = 0; i < v_no_of_cells[level]; ++i) {
                uint sum = 0;
                for (uint j = 0; j < cell_ns[level][i]; ++j) {
                    sum += leaf_cell_ns[level - 1][cell_indexes[level][i][j]];
                }
                leaf_cell_ns[level][i] = sum;
            }
        }

        uint load_limit = v_no_of_cells[0] / (n_threads * std::max((int)log(max_d), 1));
        std::vector<std::pair<int, int>> v_big_trees;
        uint t_s_capacity[n_threads];
        auto **s_levels = new uint*[n_threads];
        auto **s_c1_indexes = new uint*[n_threads];
        auto **s_c2_indexes = new uint*[n_threads];
        std::vector<std::pair<uint, uint>> v_full[n_threads];
        std::vector<std::pair<uint, uint>> v_partial[n_threads];
        for (uint t = 0; t < n_threads; t++) {
            t_s_capacity[t] = v_no_of_cells[0]*std::max((int)logf(max_d), 1);
            s_levels[t] = new uint[t_s_capacity[t]];
            s_c1_indexes[t] = new uint[t_s_capacity[t]];
            s_c2_indexes[t] = new uint[t_s_capacity[t]];
            v_full[t].reserve(v_no_of_cells[0] / n_threads);
            v_partial[t].reserve(v_no_of_cells[0] / n_threads);
        }
        for (uint level = 1; level < max_levels; level++) {
            #pragma omp parallel for schedule(dynamic)
            for (uint i = 0; i < v_no_of_cells[level]; i++) {
                uint t_id = omp_get_thread_num();
                uint s_index = 0;
                for (uint j = 0; j < cell_ns[level][i]; ++j) {
                    for (uint k = j + 1; k < cell_ns[level][i]; ++k) {
                        s_levels[t_id][s_index] = level - 1;
                        s_c1_indexes[t_id][s_index] = cell_indexes[level][i][j];
                        s_c2_indexes[t_id][s_index] = cell_indexes[level][i][k];
                        if (++s_index == t_s_capacity[t_id]) {
                            double_capacity(t_s_capacity, s_levels, s_c1_indexes, s_c2_indexes, t_id);
                        }
                    }
                }
                while (s_index > 0) {
                    --s_index;
                    uint l = s_levels[t_id][s_index];
                    uint c1 = s_c1_indexes[t_id][s_index];
                    uint c2 = s_c2_indexes[t_id][s_index];
//                    test_cnt_1++;
                    if (is_in_reach(&cell_dims_min[l][c1 * max_d], &cell_dims_max[l][c1 * max_d],
                                    &cell_dims_min[l][c2 * max_d], &cell_dims_max[l][c2 * max_d], max_d, e)) {
//                        test_cnt_2++;
                        if (l == 0) {
//                            test_cnt_3++;
                            if (cell_types[c1] == TYPE_AC && cell_types[c2] == TYPE_AC) {
//                                test_cnt_4++;
                                process_ac_ac_pair(ps_origin, v_coords, cell_indexes[l][c1], cell_ns[l][c1],
                                        cell_indexes[l][c2], cell_ns[l][c2], max_d, e2);
                            }
                            if (!(cell_types[c1] == TYPE_AC && cell_types[c2] == TYPE_AC)) {
                                int ret_val = neighbour_count_non_ac_pair_2(ps_origin, v_coords, cell_indexes[l],
                                        cell_ns[l], range_table[t_id], v_point_nps, v_cell_nps, cell_types, is_core,
                                        c1, c2, max_d, e2, m, false);
//                                if (ret_val != RET_NONE) {
//                                    test_cnt_5++;
//                                }
                                if (ret_val == RET_FULL) {
                                    v_full[t_id].emplace_back(c1, c2);
                                } else if (ret_val == RET_PARTIAL) {
                                    v_partial[t_id].emplace_back(c1, c2);
                                }
                            }
                        } else {
                            for (uint j = 0; j < cell_ns[l][c1]; ++j) {
                                for (uint k = 0; k < cell_ns[l][c2]; ++k) {
                                    s_levels[t_id][s_index] = l - 1;
                                    s_c1_indexes[t_id][s_index] = cell_indexes[l][c1][j];
                                    s_c2_indexes[t_id][s_index] = cell_indexes[l][c2][k];
                                    if (++s_index == t_s_capacity[t_id]) {
                                        double_capacity(t_s_capacity, s_levels, s_c1_indexes, s_c2_indexes, t_id);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        std::cout << "Full sizes: ";
        for (uint i = 0; i < n_threads; i++) {
            std::cout << v_full[i].size() << " ";
        }
        std::cout << std::endl;
        std::cout << "Partial sizes: ";
        for (uint i = 0; i < n_threads; i++) {
            std::cout << v_partial[i].size() << " ";
        }
        std::cout << std::endl;

        // full
        // TODO merge fulls into a single vector
        #pragma omp parallel for
        for (uint t = 0; t < n_threads; t++) {
            for (auto &elem : v_full[t]) {
                uint c1 = elem.first;
                uint c2 = elem.second;
                if (cell_types[c1] == TYPE_NC && cell_types[c2] == TYPE_NC)
                    continue;
                uint c1_index = cell_indexes[0][c1][0];
                uint c2_index = cell_indexes[0][c2][0];
                if (cell_types[c1] == TYPE_NC) {
                    struct_label *c1_label = get_label(ps_origin[c1_index]);
                    if (c1_label->label == UNASSIGNED) {
                        struct_label *c2_label = get_label(ps_origin[c2_index]);
//                        assert(c2_label->label != UNASSIGNED);
                        for (uint j = 0; j < cell_ns[0][c1]; ++j) {
                            ps_origin[cell_indexes[0][c1][j]]->label_p = c2_label;
                        }
                    }
                } else if (cell_types[c2] == TYPE_NC) {
                    struct_label *c2_label = get_label(ps_origin[c2_index]);
                    if (c2_label->label == UNASSIGNED) {
                        struct_label *c1_label = get_label(ps_origin[c1_index]);
//                        assert(c1_label->label != UNASSIGNED);
                        for (uint j = 0; j < cell_ns[0][c2]; ++j) {
                            ps_origin[cell_indexes[0][c2][j]]->label_p = c1_label;
                        }
                    }
                } else {
                    struct_label *c1_label = get_label(ps_origin[c1_index]);
                    struct_label *c2_label = get_label(ps_origin[c2_index]);
//                    assert(c1_label->label != UNASSIGNED);
//                    assert(c2_label->label != UNASSIGNED);
                    if (c1_label->label != c2_label->label) {
                        set_lower_label(c1_label, c2_label);
                    }
                }
            }
        }

        // partial
        #pragma omp parallel for
        for (uint t = 0; t < n_threads; t++) {
            for (auto &elem : v_partial[t]) {
                uint c1 = elem.first;
                uint c2 = elem.second;
                if (cell_types[c1] == TYPE_NC && cell_types[c2] == TYPE_NC)
                    continue;

                // if both ac, no need to do range mark
                if (cell_types[c1] == TYPE_AC && cell_types[c2] == TYPE_AC) {
                    process_ac_ac_pair(ps_origin, v_coords, cell_indexes[0][c1], cell_ns[0][c1],
                                       cell_indexes[0][c2], cell_ns[0][c2], max_d, e2);
                } else {
                    uint size1 = cell_ns[0][c1];
                    uint size2 = cell_ns[0][c2];
                    mark_in_range(v_coords, cell_indexes[0][c1], size1, cell_indexes[0][c2], size2,
                            range_table[t], max_d, e2);
//                    assert(cnt_range > 0);
//                    assert(cnt_range != size1 * size2);

                    uint index = 0;
                    for (uint i = 0; i < size1; i++) {
                        uint c1_index = cell_indexes[0][c1][i];
                        for (uint j = 0; j < size2; j++, index++) {
                            if (range_table[0][index]) {
                                uint c2_index = cell_indexes[0][c2][j];
                                if (is_core[c1_index] && is_core[c2_index]) {
                                    struct_label *c1_label = get_label(ps_origin[c1_index]);
                                    struct_label *c2_label = get_label(ps_origin[c2_index]);
                                    if (c1_label->label != c2_label->label) {
                                        set_lower_label(c1_label, c2_label);
                                    }
                                } else if (is_core[c1_index]) {
                                    struct_label *c2_label = get_label(ps_origin[c2_index]);
                                    if (c2_label->label == UNASSIGNED) {
                                        struct_label *c1_label = get_label(ps_origin[c1_index]);
                                        ps_origin[c2_index]->label_p = c1_label;
                                    }
                                } else if (is_core[c2_index]) {
                                    struct_label *c1_label = get_label(ps_origin[c1_index]);
                                    if (c1_label->label == UNASSIGNED) {
                                        struct_label *c2_label = get_label(ps_origin[c2_index]);
                                        ps_origin[c1_index]->label_p = c2_label;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

//        std::cout << "Total reach checks: " << test_cnt_1 << std::endl;
//        std::cout << "Succesful reach checks: " << test_cnt_2 << std::endl;
//        std::cout << "Level 0 succesful reaches: " << test_cnt_3 << std::endl;
//        std::cout << "AC/AC pair: " << test_cnt_4 << std::endl;
//        std::cout << "stored edges: " << test_cnt_5 << std::endl;
//        std::cout << "label tree traverse counter: " << label_counter << std::endl;
    }

    void process_cell_tree_omp(struct_label **ps_origin, float *v_coords, uint ***cell_indexes, uint **cell_ns,
            float **cell_dims_min, float **cell_dims_max, const std::vector<uint> &v_no_of_cells,
            bool_vector &is_core, uint n_threads, uint max_levels, uint max_d, float e, float e2, uint m,
            const uint n) noexcept {
        uint max_points_in_cell = 0;
        std::vector<uint> v_cell_nps(v_no_of_cells[0]);
        auto **range_table = new bool *[n_threads];
        std::vector<uint> v_point_nps(n, 0);
        type_vector cell_types(v_no_of_cells[0]);
        cell_types.fill(v_no_of_cells[0], TYPE_NC);
        bool_vector is_cell_processed(v_no_of_cells[0]);
        is_cell_processed.fill(v_no_of_cells[0], false);

        #pragma omp parallel for reduction(max: max_points_in_cell)
        for (uint i = 0; i < v_no_of_cells[0]; ++i) {
            v_cell_nps[i] = cell_ns[0][i];
            if (v_cell_nps[i] > max_points_in_cell) {
                max_points_in_cell = v_cell_nps[i];
            }
        }
        for (uint i = 0; i < n_threads; i++) {
            range_table[i] = new bool[max_points_in_cell * std::min(max_points_in_cell, m)];
        }

        auto **leaf_cell_ns = new uint *[max_levels];
        for (uint level = 0; level < max_levels; ++level) {
            leaf_cell_ns[level] = new uint[v_no_of_cells[level]];
            if (level == 0) {
                std::fill(leaf_cell_ns[0], leaf_cell_ns[0] + v_no_of_cells[0], 1);
                continue;
            }
            #pragma omp parallel for
            for (uint i = 0; i < v_no_of_cells[level]; ++i) {
                uint sum = 0;
                for (uint j = 0; j < cell_ns[level][i]; ++j) {
                    sum += leaf_cell_ns[level - 1][cell_indexes[level][i][j]];
                }
                leaf_cell_ns[level][i] = sum;
            }
        }

        uint load_limit = v_no_of_cells[0] / (n_threads * std::max((int)log(max_d), 1));
        std::vector<std::pair<int, int>> v_big_trees;
        uint t_s_capacity[n_threads];
        auto **s_levels = new uint*[n_threads];
        auto **s_c1_indexes = new uint*[n_threads];
        auto **s_c2_indexes = new uint*[n_threads];
        for (uint t = 0; t < n_threads; t++) {
            t_s_capacity[t] = v_no_of_cells[0]*std::max((int)logf(max_d), 1);
            s_levels[t] = new uint[t_s_capacity[t]];
            s_c1_indexes[t] = new uint[t_s_capacity[t]];
            s_c2_indexes[t] = new uint[t_s_capacity[t]];
        }

//        uint cnt = 0;
        for (uint level = 1; level < max_levels; level++) {
            #pragma omp parallel for schedule(dynamic)
            for (uint i = 0; i < v_no_of_cells[level]; i++) {
                if (n_threads > 1 && leaf_cell_ns[level][i] > load_limit) {
                    #pragma omp critical
                    v_big_trees.emplace_back(level, i);
                    continue;
                }
                uint t_id = omp_get_thread_num();
                uint s_index = 0;
                for (uint j = 0; j < cell_ns[level][i]; ++j) {
                    for (uint k = j + 1; k < cell_ns[level][i]; ++k) {
                        s_levels[t_id][s_index] = level - 1;
                        s_c1_indexes[t_id][s_index] = cell_indexes[level][i][j];
                        s_c2_indexes[t_id][s_index] = cell_indexes[level][i][k];
                        if (++s_index == t_s_capacity[t_id]) {
                            double_capacity(t_s_capacity, s_levels, s_c1_indexes, s_c2_indexes, t_id);
                        }
                    }
                }
                while (s_index > 0) {
                    --s_index;
                    uint l = s_levels[t_id][s_index];
                    uint c1 = s_c1_indexes[t_id][s_index];
                    uint c2 = s_c2_indexes[t_id][s_index];
                    if (is_in_reach(&cell_dims_min[l][c1 * max_d], &cell_dims_max[l][c1 * max_d],
                                    &cell_dims_min[l][c2 * max_d], &cell_dims_max[l][c2 * max_d], max_d, e)) {
                        if (l == 0) {
//                            if (v_cell_nps[c1] < m && v_cell_nps[c2] < m) {
//                                ++cnt;
//                            }
                            if (!(v_cell_nps[c1] >= m && v_cell_nps[c2] >= m)) {
                                neighbour_count_non_ac_pair(v_coords, cell_indexes[l], cell_ns[l],
                                                            range_table[t_id], v_point_nps, v_cell_nps, c1, c2, max_d,
                                                            e2, m, false);
                            }
                        } else {
                            for (uint j = 0; j < cell_ns[l][c1]; ++j) {
                                for (uint k = 0; k < cell_ns[l][c2]; ++k) {
                                    s_levels[t_id][s_index] = l - 1;
                                    s_c1_indexes[t_id][s_index] = cell_indexes[l][c1][j];
                                    s_c2_indexes[t_id][s_index] = cell_indexes[l][c2][k];
                                    if (++s_index == t_s_capacity[t_id]) {
                                        double_capacity(t_s_capacity, s_levels, s_c1_indexes, s_c2_indexes, t_id);
                                    }
                                }
                            }
                        }
                    }
                }
            }
//            std::cout << "cnt: " << cnt << std::endl;
            if (v_big_trees.empty())
                continue;
            #pragma omp parallel
            {
                for (auto &elem : v_big_trees) {
                    uint tree_level = elem.first;
                    uint i = elem.second;
                    uint t_id = omp_get_thread_num();
                    uint s_index = 0;
                    uint cnt = 0;
                    for (uint j = 0; j < cell_ns[tree_level][i]; ++j) {
                        for (uint k = j + 1; k < cell_ns[tree_level][i]; ++k) {
                            if (leaf_cell_ns[tree_level-1][i] + leaf_cell_ns[tree_level-1][j] > load_limit) {
                                // load balancing heuristic, go one level deeper
                                uint i2 = cell_indexes[tree_level][i][j];
                                uint ii2 = cell_indexes[tree_level][i][k];
                                for (uint j2 = 0; j2 < cell_ns[tree_level-1][i2]; ++j2) {
                                    for (uint k2 = 0; k2 < cell_ns[tree_level-1][ii2]; ++k2) {
                                        if (cnt++ % n_threads == t_id) {
                                            s_levels[t_id][s_index] = tree_level - 2;
                                            s_c1_indexes[t_id][s_index] = cell_indexes[tree_level-1][i2][j2];
                                            s_c2_indexes[t_id][s_index] = cell_indexes[tree_level-1][ii2][k2];
                                            if (++s_index == t_s_capacity[t_id]) {
                                                double_capacity(t_s_capacity, s_levels, s_c1_indexes, s_c2_indexes, t_id);
                                            }
                                        }
                                    }
                                }
                            } else if (cnt++ % n_threads == t_id) {
                                s_levels[t_id][s_index] = tree_level - 1;
                                s_c1_indexes[t_id][s_index] = cell_indexes[tree_level][i][j];
                                s_c2_indexes[t_id][s_index] = cell_indexes[tree_level][i][k];
                                if (++s_index == t_s_capacity[t_id]) {
                                    double_capacity(t_s_capacity, s_levels, s_c1_indexes, s_c2_indexes, t_id);
                                }
                            }
                        }
                    }
                    while (s_index > 0) {
                        --s_index;
                        uint l = s_levels[t_id][s_index];
                        uint c1 = s_c1_indexes[t_id][s_index];
                        uint c2 = s_c2_indexes[t_id][s_index];
                        if (is_in_reach(&cell_dims_min[l][c1 * max_d], &cell_dims_max[l][c1 * max_d],
                                        &cell_dims_min[l][c2 * max_d],
                                        &cell_dims_max[l][c2 * max_d], max_d, e)) {
                            if (l == 0) {
                                if (!(v_cell_nps[c1] >= m && v_cell_nps[c2] >= m)) {
                                    neighbour_count_non_ac_pair(v_coords, cell_indexes[l], cell_ns[l],
                                            range_table[t_id], v_point_nps, v_cell_nps, c1, c2, max_d, e2, m, true);
                                }
                            } else {
                                for (uint j = 0; j < cell_ns[l][c1]; ++j) {
                                    for (uint k = 0; k < cell_ns[l][c2]; ++k) {
                                        s_levels[t_id][s_index] = l - 1;
                                        s_c1_indexes[t_id][s_index] = cell_indexes[l][c1][j];
                                        s_c2_indexes[t_id][s_index] = cell_indexes[l][c2][k];
                                        if (++s_index >= t_s_capacity[t_id]) {
                                            double_capacity(t_s_capacity, s_levels, s_c1_indexes, s_c2_indexes, t_id);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            v_big_trees.clear();
        }
        #pragma omp parallel for
        for (uint i = 0; i < v_no_of_cells[0]; ++i) {
            if (v_cell_nps[i] >= m) {
                cell_types[i] = TYPE_AC;
                ps_origin[cell_indexes[0][i][0]]->label = i;
                for (uint j = 0; j < cell_ns[0][i]; ++j) {
                    is_core[cell_indexes[0][i][j]] = true;
                    if (j > 0) {
                        ps_origin[cell_indexes[0][i][j]]->label_p = ps_origin[cell_indexes[0][i][0]];
                    }
                }
            } else {
                uint c_index = UNASSIGNED;
                bool all_cores = true;
                for (uint j = 0; j < cell_ns[0][i]; ++j) {
                    uint p_index = cell_indexes[0][i][j];
                    if (v_cell_nps[i] + v_point_nps[p_index] >= m) {
                        is_core[p_index] = true;
                        if (cell_types[i] == TYPE_NC) {
                            cell_types[i] = TYPE_SC;
                            c_index = j;
                        }
                    } else {
                        all_cores = false;
                    }
                }
                // corner case, actually AC
                if (all_cores) {
                    cell_types[i] = TYPE_AC;
                    ps_origin[cell_indexes[0][i][0]]->label = i;
                    for (uint j = 0; j < cell_ns[0][i]; ++j) {
                        if (j > 0) {
                            ps_origin[cell_indexes[0][i][j]]->label_p = ps_origin[cell_indexes[0][i][0]];
                        }
                    }
                } else if (cell_types[i] == TYPE_SC) {
                    uint p_c_index = cell_indexes[0][i][c_index];
                    ps_origin[p_c_index]->label = i;
                    for (uint j = 0; j < cell_ns[0][i]; ++j) {
                        if (j != c_index) {
                            ps_origin[cell_indexes[0][i][j]]->label_p = ps_origin[p_c_index];
                        }
                    }
                }
            }
        }
        for (uint level = 1; level < max_levels; ++level) {
            #pragma omp parallel for schedule(dynamic)
            for (uint i = 0; i < v_no_of_cells[level]; ++i) {
                if (n_threads > 1 && leaf_cell_ns[level][i] > load_limit) {
                    #pragma omp critical
                    v_big_trees.emplace_back(level, i);
                    continue;
                }
                uint t_id = omp_get_thread_num();
                uint s_index = 0;
                for (uint j = 0; j < cell_ns[level][i]; ++j) {
                    for (uint k = j+1; k < cell_ns[level][i]; ++k) {
                        s_levels[t_id][s_index] = level-1;
                        s_c1_indexes[t_id][s_index] = cell_indexes[level][i][j];
                        s_c2_indexes[t_id][s_index] = cell_indexes[level][i][k];
                        if (++s_index == t_s_capacity[t_id]) {
                            double_capacity(t_s_capacity, s_levels, s_c1_indexes, s_c2_indexes, t_id);
                        }
                    }
                }
                while (s_index > 0) {
                    --s_index;
                    uint l = s_levels[t_id][s_index];
                    uint c1 = s_c1_indexes[t_id][s_index];
                    uint c2 = s_c2_indexes[t_id][s_index];
                    if (l == 0 && cell_types[c1] == TYPE_NC && cell_types[c2] == TYPE_NC) {
                        continue;
                    }
                    if (is_in_reach(&cell_dims_min[l][c1*max_d], &cell_dims_max[l][c1*max_d], &cell_dims_min[l][c2*max_d],
                            &cell_dims_max[l][c2*max_d], max_d, e)) {
                        if (l == 0) {
//                            ++cnt;
                            if (cell_types[c1] == TYPE_AC && cell_types[c2] == TYPE_AC) {
                                process_ac_ac_pair(ps_origin, v_coords, cell_indexes[l][c1], cell_ns[l][c1],
                                        cell_indexes[l][c2], cell_ns[l][c2], max_d, e2);
                            } else if ((cell_types[c1] == TYPE_AC && cell_types[c2] == TYPE_SC) ||
                                (cell_types[c1] == TYPE_SC && cell_types[c2] == TYPE_AC)) {
                                int ac_id = c1;
                                int sc_id = c2;
                                if (cell_types[ac_id] != TYPE_AC) {
                                    ac_id = c2;
                                    sc_id = c1;
                                }
                                process_ac_sc_pair(cell_indexes[l], ps_origin, v_coords, cell_indexes[l][ac_id],
                                        cell_ns[l][ac_id], cell_indexes[l][sc_id], cell_ns[l][sc_id], sc_id, is_core,
                                        max_d, e2);
                            } else if (cell_types[c1] == TYPE_SC && cell_types[c2] == TYPE_SC) {
                                process_sc_sc_pair(cell_indexes[l], ps_origin, v_coords, cell_indexes[l][c1], c1,
                                        cell_ns[l][c1], cell_indexes[l][c2], c2, cell_ns[l][c2], is_core, max_d, e2);
                            } else if (!(cell_types[c1] == TYPE_NC && cell_types[c2] == TYPE_NC)) {
                                if ((cell_types[c1] == TYPE_NC && is_cell_processed[c1]) ||
                                    (cell_types[c2] == TYPE_NC && is_cell_processed[c2])) {
                                    continue;
                                }
                                process_pair_one_nc(ps_origin, v_coords, cell_indexes[l], cell_ns[l], range_table[t_id],
                                        is_cell_processed, is_core, c1, cell_types[c1], c2, cell_types[c2], max_d, e2);
                            }
                        } else {
                            for (uint j = 0; j < cell_ns[l][c1]; ++j) {
                                for (uint k = 0; k < cell_ns[l][c2]; ++k) {
                                    s_levels[t_id][s_index] = l-1;
                                    s_c1_indexes[t_id][s_index] = cell_indexes[l][c1][j];
                                    s_c2_indexes[t_id][s_index] = cell_indexes[l][c2][k];
                                    if (++s_index == t_s_capacity[t_id]) {
                                        double_capacity(t_s_capacity, s_levels, s_c1_indexes, s_c2_indexes, t_id);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            v_big_trees.clear();
        }
        for (uint t = 0; t < n_threads; t++) {
            delete[] s_levels[t];
            delete[] s_c1_indexes[t];
            delete[] s_c2_indexes[t];
        }
        delete[] s_levels;
        delete[] s_c1_indexes;
        delete[] s_c2_indexes;
    }

// Future work
    void detect_border_cells(uint ***cell_indexes, uint **cell_ns, float **cell_dims_min, float **cell_dims_max,
            std::vector<uint8_t> &border_cells, const std::vector<uint> &v_no_of_cells, std::vector<uint> *s_c1_indexes,
            std::vector<uint> *s_c2_indexes, std::vector<uint> *s_levels, const uint max_levels, const uint max_d,
            const uint m, const float e, uint *t_s_capacity) noexcept {
        std::vector<uint> v_cell_nps(v_no_of_cells[0]);
        std::copy(cell_ns[0], cell_ns[0] + v_no_of_cells[0], v_cell_nps.data());
        for (uint level = 1; level < max_levels; level++) {
            #pragma omp parallel for
            for (uint i = 0; i < v_no_of_cells[level]; i++) {
                int t_id = omp_get_thread_num();
                uint s_index = 0;
                for (uint j = 0; j < cell_ns[level][i]; j++) {
                    for (uint k = j + 1; k < cell_ns[level][i]; k++) {
                        s_levels[t_id][s_index] = level - 1;
                        s_c1_indexes[t_id][s_index] = cell_indexes[level][i][j];
                        s_c2_indexes[t_id][s_index] = cell_indexes[level][i][k];
                        if (++s_index == t_s_capacity[t_id]) {
                            t_s_capacity[t_id] *= 2;
                            s_levels[t_id].resize(t_s_capacity[t_id]);
                            s_c1_indexes[t_id].resize(t_s_capacity[t_id]);
                            s_c2_indexes[t_id].resize(t_s_capacity[t_id]);
                        }
                    }
                }
                while (s_index > 0) {
                    --s_index;
                    uint l = s_levels[t_id][s_index];
                    uint c1 = s_c1_indexes[t_id][s_index];
                    uint c2 = s_c2_indexes[t_id][s_index];
                    if (l == 0 && v_cell_nps[c1] >= m && v_cell_nps[c2] >= m)
                        continue;
                    if (is_in_reach(&cell_dims_min[l][c1 * max_d], &cell_dims_max[l][c1 * max_d],
                                    &cell_dims_min[l][c2 * max_d],
                                    &cell_dims_max[l][c2 * max_d], max_d, e)) {
                        if (l == 0) {
                            if (v_cell_nps[c1] < m)
                                v_cell_nps[c1] += cell_ns[0][c2];
                            if (v_cell_nps[c2] < m)
                                v_cell_nps[c2] += cell_ns[0][c1];
                        } else {
                            for (uint j = 0; j < cell_ns[l][c1]; j++) {
                                for (uint k = 0; k < cell_ns[l][c2]; k++) {
                                    s_levels[t_id][s_index] = l - 1;
                                    s_c1_indexes[t_id][s_index] = cell_indexes[l][c1][j];
                                    s_c2_indexes[t_id][s_index] = cell_indexes[l][c2][k];
                                    if (++s_index == t_s_capacity[t_id]) {
                                        t_s_capacity[t_id] *= 2;
                                        s_levels[t_id].resize(t_s_capacity[t_id]);
                                        s_c1_indexes[t_id].resize(t_s_capacity[t_id]);
                                        s_c2_indexes[t_id].resize(t_s_capacity[t_id]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        #pragma omp parallel for schedule(static)
        for (uint i = 0; i < v_no_of_cells[0]; i++) {
            if (v_cell_nps[i] < m) {
                border_cells[i] = 1;
            }
        }
    }

    void index_cells_omp_simple(const uint no_of_cells, std::vector<std::pair<ull, uint>> *vec_index_maps,
            uint ***cell_indexes, uint **cell_ns, std::vector<uint> *vec_cell_begin, float *v_coords,
            const std::vector<float> &min_bounds, ull **dims_mult, const std::vector<float> &v_eps_levels, std::vector<uint> &v_no_of_cells,
            const uint max_d, const uint l, const uint n_threads) noexcept {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp for
            for (uint i = 0; i < no_of_cells; i++) {
                int p_index = traverse_and_get_cell_index(cell_indexes, l, i);
                ull cell_index = get_cell_index(&v_coords[p_index * max_d], min_bounds, dims_mult[l], max_d,
                        v_eps_levels[l]);
                vec_index_maps[tid].emplace_back(cell_index, i);
            }
        }
        for (uint t = 1; t < n_threads; t++) {
            vec_index_maps[0].insert(vec_index_maps[0].end(), vec_index_maps[t].begin(), vec_index_maps[t].end());
        }
        std::sort(vec_index_maps[0].begin(), vec_index_maps[0].end());
        uint cnt = 0;
        size_t last_index = vec_index_maps[0][0].first;
        vec_cell_begin[0].push_back(0);
        for (uint i = 1; i < vec_index_maps[0].size(); i++) {
            if (vec_index_maps[0][i].first != last_index) {
                vec_cell_begin[0].push_back(i);
                last_index = vec_index_maps[0][i].first;
                cnt = 1;
            } else {
                ++cnt;
            }
        }
        v_no_of_cells[l] = vec_cell_begin[0].size();
        std::cout << "no of cells at l: " << l << " is " << v_no_of_cells[l] << std::endl;
        cell_indexes[l] = new uint *[v_no_of_cells[l]];
        cell_ns[l] = new uint[v_no_of_cells[l]];
        for (uint i = 0; i < v_no_of_cells[l]; i++) {
            uint begin = vec_cell_begin[0][i];
            uint end = (i == (v_no_of_cells[l]-1)) ? no_of_cells : vec_cell_begin[0][i+1];
            cell_ns[l][i] = end - begin;
        }
        for (uint i = 0; i < v_no_of_cells[l]; i++) {
            cell_indexes[l][i] = new uint[cell_ns[l][i]];
        }
        #pragma omp parallel for
        for (uint i = 0; i < v_no_of_cells[l]; i++) {
            uint begin = vec_cell_begin[0][i];
            uint end = (i == (v_no_of_cells[l] - 1))? no_of_cells : vec_cell_begin[0][i+1];
            std::transform(&vec_index_maps[0][begin], &vec_index_maps[0][end], &cell_indexes[l][i][0],
                    [](const std::pair<ull, uint> &p) {
               return p.second;
           });
        }
    }

    #pragma omp declare reduction (merge : std::vector<std::pair<ull, uint>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

    inline std::vector<std::pair<ull, uint>>::const_iterator middle(
            std::vector<std::pair<ull, uint>>::const_iterator begin,
            std::vector<std::pair<ull, uint>>::const_iterator end) noexcept {
        return begin + ((end - begin) / 2);
    }

    void fill_medians(std::vector<std::pair<ull, uint>>::const_iterator begin,
            std::vector<std::pair<ull, uint>>::const_iterator end, ull *medians, uint index1, uint index2, uint last) noexcept {
        uint index = ((index2 - index1) / 2) + index1;
        if (index == index1 && index == last-1)
            index = last;
        auto i_median = middle(begin, end);
        medians[index] = i_median->first;
        if (index - index1 > 1) {
            fill_medians(begin, i_median, medians, index1, index, last);
        } else if (index - index1 > 0 && index1 == 0) {
            fill_medians(begin, i_median, medians, index1, index, last);
        }
        if (index2 - index > 1) {
            fill_medians(i_median, end, medians, index, index2, last);
        } else if (index2 - index > 0 && index2 == last) {
            fill_medians(i_median, end, medians, index, index2, last);
        }
    }

    void index_cells_omp_merge(const uint no_of_cells, std::vector<std::pair<ull, uint>> *vec_index_maps,
            std::vector<std::pair<ull, uint>> *vec_buckets, std::vector<uint> *vec_cell_begin, std::vector<ull> &medians,
            std::vector<ull> &median_buckets, uint ***cell_indexes, uint **cell_ns, float *v_coords,
            const std::vector<float>& min_bounds, ull **dims_mult, const std::vector<float> &v_eps_levels,
            std::vector<uint> &v_no_of_cells, ull *selected_medians, const uint max_d, const uint l, const uint n_threads) noexcept {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            uint cnt = 0;
            #pragma omp for nowait schedule(static)
            for (uint i = 0; i < no_of_cells; i++) {
                uint index = cnt + tid;
                uint p_index = traverse_and_get_cell_index(cell_indexes, l, index);
                ull cell_index = get_cell_index(&v_coords[p_index * max_d], min_bounds, dims_mult[l], max_d,
                                                v_eps_levels[l]);
                vec_index_maps[tid].emplace_back(cell_index, index);
                cnt += n_threads;
            }
            std::sort(vec_index_maps[tid].begin(), vec_index_maps[tid].end());
            fill_medians(vec_index_maps[tid].begin(), vec_index_maps[tid].end(), &medians[tid*n_threads], 0, n_threads-1,
                         n_threads-1);
        }
        for (uint i = 0; i < n_threads; i++) {
            std::copy(&medians[i*n_threads], &medians[(i+1)*n_threads], &median_buckets[i*n_threads]);
        }
        std::sort(&median_buckets[0], &median_buckets[0] + (n_threads*n_threads));
        std::fill(selected_medians, selected_medians + n_threads, 0);
        #pragma omp parallel reduction(+:selected_medians[:n_threads])
        {
            int tid = omp_get_thread_num();
            int index = tid*n_threads + (n_threads / 2);
            selected_medians[tid] = (median_buckets[index] + median_buckets[index-1]) / 2;
        }
        for (uint i = 0; i < n_threads; i++) {
            #pragma omp parallel reduction(merge: vec_buckets[i])
            {
                int tid = omp_get_thread_num();
                if (i == 0) {
                    auto iter = std::lower_bound(vec_index_maps[tid].begin(), vec_index_maps[tid].end(),
                            selected_medians[i], [](auto pair, auto val) -> bool {
                        return pair.first < val;
                    });
                    vec_buckets[i].assign(vec_index_maps[tid].begin(), iter);
                } else if (i == n_threads-1) {
                    auto iter = std::lower_bound(vec_index_maps[tid].begin(), vec_index_maps[tid].end(),
                            selected_medians[i-1], [](auto pair, auto val) -> bool {
                        return pair.first < val;
                    });
                    vec_buckets[i].assign(iter, vec_index_maps[tid].end());
                } else {
                    auto iter1 = std::lower_bound(vec_index_maps[tid].begin(), vec_index_maps[tid].end(),
                            selected_medians[i-1], [](auto pair, auto val) -> bool {
                        return pair.first < val;
                    });
                    auto iter2 = std::lower_bound(vec_index_maps[tid].begin(), vec_index_maps[tid].end(),
                            selected_medians[i], [](auto pair, auto val) -> bool {
                        return pair.first < val;
                    });
                    vec_buckets[i].assign(iter1, iter2);
                }
            }
        }
        int unique_counter = 0;
        #pragma omp parallel reduction(+:unique_counter)
        {
            int tid = omp_get_thread_num();
            std::sort(vec_buckets[tid].begin(), vec_buckets[tid].end());
            ull last = vec_buckets[tid][0].first;
            vec_cell_begin[tid].push_back(0);
            int cnt = 0;
            int index = 0;
            for (auto &pair : vec_buckets[tid]) {
                if (pair.first == last) {
                    ++cnt;
                } else {
                    vec_cell_begin[tid].push_back(index);
                    last = pair.first;
                    cnt = 1;
                }
                ++index;
            }
            unique_counter = vec_cell_begin[tid].size();
        }
        v_no_of_cells[l] = unique_counter;
        cell_indexes[l] = new uint*[v_no_of_cells[l]];
        cell_ns[l] = new uint[v_no_of_cells[l]];

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int local_index = 0;
            for (int t = 0; t < tid; t++) {
                local_index += vec_cell_begin[t].size();
            }
            for (uint i = 1; i < vec_cell_begin[tid].size(); i++, local_index++) {
                int size = vec_cell_begin[tid][i] - vec_cell_begin[tid][i-1];
                cell_ns[l][local_index] = size;
            }
            int size = vec_buckets[tid].size() - vec_cell_begin[tid][vec_cell_begin[tid].size()-1];
            cell_ns[l][local_index] = size;
        }
        for (uint i = 0; i < v_no_of_cells[l]; i++) {
            cell_indexes[l][i] = new uint[cell_ns[l][i]];
        }
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int local_index = 0;
            for (int t = 0; t < tid; t++) {
                local_index += vec_cell_begin[t].size();
            }
            for (uint i = 0; i < vec_cell_begin[tid].size(); i++) {
                int begin = vec_cell_begin[tid][i];
                int end = i == vec_cell_begin[tid].size()-1? vec_buckets[tid].size() : vec_cell_begin[tid][i+1];
                std::transform(&vec_buckets[tid][begin], &vec_buckets[tid][end], &cell_indexes[l][local_index+i][0],
                        [](auto &p) {
                   return p.second;
               });
            }
        }
    }

    /*
             auto *vv_index_maps = new std::vector<std::pair<ull, uint>>*[n_threads];
        auto *vv_cell_begin = new std::vector<uint>*[n_threads];
        auto *vv_cell_ns = new std::vector<uint>*[n_threads];
        auto v_no_cells = new std::vector<uint>[n_threads];
     */

    int index_points(float *v_coords, std::vector<std::pair<ull, uint>>** vv_index_maps, std::vector<uint>** vv_cell_begin,
                     std::vector<uint>** vv_cell_ns, std::vector<uint> *v_no_cells, const uint max_d, const uint n,
                     const uint n_threads, const float e_inner) noexcept {
        float max_limit = INT32_MIN;
        std::vector<float> min_bounds(max_d);
        std::vector<float> max_bounds(max_d);
        calc_bounds(v_coords, n, min_bounds, max_bounds, max_d);
        for (uint d = 0; d < max_d; d++) {
            if (max_bounds[d] - min_bounds[d] > max_limit)
                max_limit = max_bounds[d] - min_bounds[d];
        }
        int max_levels = static_cast<int>(ceilf(logf(max_limit / e_inner) / logf(2))) + 1;
        std::vector<float> v_eps_levels(max_levels);

        auto **dims_mult = new ull*[max_levels];
        for (int i = 0; i < max_levels; i++) {
            v_eps_levels[i] = (e_inner * powf(2, i));
            dims_mult[i] = new ull[max_d];
            calc_dims_mult(dims_mult[i], max_d, min_bounds, max_bounds, v_eps_levels[i]);
        }

        for (uint t = 0; t < n_threads; ++t) {
            vv_index_maps[t] = new std::vector<std::pair<ull, uint>>[max_levels];
            vv_cell_begin[t] = new std::vector<uint>[max_levels];
            vv_cell_ns[t] = new std::vector<uint>[max_levels];
            v_no_cells[t].reserve(max_levels);
        }

        #pragma omp parallel
        {
            const uint tid = omp_get_thread_num();
            for (int level = 0; level < max_levels; ++level) {
                uint mem_reserve = level == 0 ? n / n_threads + 1 : v_no_cells[tid][level-1];
                vv_index_maps[tid][level].reserve(mem_reserve);
                vv_cell_begin[tid][level].reserve(mem_reserve);
                if (level == 0) {
                    #pragma omp for schedule(static)
                    for (uint i = 0; i < n; ++i) {
                        ull cell_index = get_cell_index(&v_coords[i * max_d], min_bounds, dims_mult[0], max_d,
                                v_eps_levels[0]);
                        vv_index_maps[tid][0].emplace_back(cell_index, i);
                    }
                } else {
                    for (uint i = 0; i < v_no_cells[tid][level-1]; ++i) {
                        int level_mod = 1;
                        uint p_index = i;
                        while (level - level_mod >= 0) {
                            p_index = vv_index_maps[tid][level-level_mod][vv_cell_begin[tid][level-level_mod][p_index]].second;
                            ++level_mod;
                        }
                        ull cell_index = get_cell_index(&v_coords[p_index * max_d], min_bounds, dims_mult[level], max_d,
                                                        v_eps_levels[level]);
                        vv_index_maps[tid][level].emplace_back(cell_index, i);
                    }
                }
                std::sort(vv_index_maps[tid][level].begin(), vv_index_maps[tid][level].end());
                ull last_index = vv_index_maps[tid][level][0].first;
                vv_cell_begin[tid][level].push_back(0);
                for (uint i = 1; i < vv_index_maps[tid][level].size(); i++) {
                    if (vv_index_maps[tid][level][i].first != last_index) {
                        vv_cell_begin[tid][level].push_back(i);
                        last_index = vv_index_maps[tid][level][i].first;
                    }
                }
                #pragma omp critical
                std::cout << "Thread " << tid << " with " << vv_cell_begin[tid][level].size() << " cells at level " << level
                          << std::endl;
                v_no_cells[tid][level] = vv_cell_begin[tid][level].size();
                vv_cell_ns[tid][level].reserve(v_no_cells[tid][level]);
                for (uint i = 0; i < v_no_cells[tid][level]; i++) {
                    uint begin = vv_cell_begin[tid][level][i];
                    uint end = (i == (v_no_cells[tid][level]-1)) ? vv_index_maps[tid][level].size() : vv_cell_begin[tid][level][i+1];
                    vv_cell_ns[tid][level][i] = end - begin;
                }
            }
        }

        return max_levels;

        /*
        // test the grid-cell tree
        auto *test = new bool[n];
        std::fill(test, test+n, false);
        std::vector<std::pair<uint, uint>> stack;
        stack.reserve(n*2);
        stack.emplace_back(max_levels-1, 0);
        uint cnt = 0;
        while (!stack.empty()) {
            uint level = stack.back().first;
            uint index = stack.back().second;
            stack.pop_back();
            uint begin = vv_cell_begin[0][level][index];
            for (uint i = 0; i < vv_cell_ns[0][level][index]; ++i) {
                if (level > 0) {
                    stack.emplace_back(level-1, vv_index_maps[0][level][begin+i].second);
                } else {
                    uint leaf_index = vv_index_maps[0][level][begin+i].second;
                    assert(leaf_index < n);
                    assert(!test[leaf_index]);
                    test[leaf_index] = true;
                    ++cnt;
                }
            }
        }
        assert(cnt == n);
*/
        /*
        uint level = 0;
        uint cnt = 0;
        uint sum = 0;
        for (uint i = 0; i < v_no_cells[0][level]; ++i) {
            sum += vv_cell_ns[0][level][i];
            uint begin = vv_cell_begin[0][level][i];
            for (uint j = 0; j < vv_cell_ns[0][level][i]; ++j) {
                ++cnt;
                uint leaf_index = vv_index_maps[0][level][begin+j].second;
                assert(leaf_index < n);
                assert(!test[leaf_index]);
                test[leaf_index] = true;
            }
        }
        assert(sum == n);
        assert(cnt == n);
         */
    }

    void index_points_to_cells_omp_median_merge(float *v_coords, uint ***cell_indexes, uint **cell_ns,
            const std::vector<float> &min_bounds, ull **dims_mult, const std::vector<float> &v_eps_levels,
            std::vector<uint> &v_no_of_cells, int max_levels, const uint max_d, const uint n,
            const uint n_threads) noexcept {
        auto* vec_index_maps = new std::vector<std::pair<ull, uint>>[n_threads];
        auto* vec_buckets = new std::vector<std::pair<ull, uint>>[n_threads];
        auto* vec_cell_begin = new std::vector<uint>[n_threads];

        for (uint i = 0; i < n_threads; i++) {
            vec_index_maps[i].reserve(n / (n_threads + 1));
            vec_buckets[i].reserve(n / ((n_threads - 1) + 1));
            vec_cell_begin[i].reserve(n / ((n_threads - 1) + 1));
        }
        uint no_of_cells;
        std::vector<ull> medians(n_threads*n_threads);
        std::vector<ull> median_buckets(n_threads*n_threads);
        auto *selected_medians = new ull[n_threads];
        for (int l = 0; l < max_levels; l++) {
            if (l == 0) {
                no_of_cells = n;
            } else {
                no_of_cells = v_no_of_cells[l - 1];
            }
            // TODO Further investigate heuristic boundary
            if (n_threads > 2 && max_d <= 8 && no_of_cells > n_threads*100) {
                index_cells_omp_merge(no_of_cells, vec_index_maps, vec_buckets, vec_cell_begin, medians,
                        median_buckets, cell_indexes, cell_ns, v_coords, min_bounds, dims_mult, v_eps_levels,
                        v_no_of_cells, selected_medians, max_d, l, n_threads);
            } else {
                index_cells_omp_simple(no_of_cells, vec_index_maps, cell_indexes, cell_ns, vec_cell_begin, v_coords,
                        min_bounds, dims_mult, v_eps_levels, v_no_of_cells, max_d, l, n_threads);
            }
            for (uint t = 0; t < n_threads; t++) {
                vec_buckets[t].clear();
                vec_cell_begin[t].clear();
                vec_index_maps[t].clear();
            }
        }

        delete [] vec_buckets;
        delete [] vec_index_maps;
        delete [] vec_cell_begin;
        delete [] selected_medians;
    }

    void nextDBSCAN(struct_label **p_labels, float *v_coords, const uint m, const float e, const uint n,
            const uint max_d, bool_vector &is_core, uint n_threads) noexcept {



        omp_set_num_threads(n_threads);
        auto t1 = std::chrono::high_resolution_clock::now();
        float e2 = e * e;
        float e_inner = (e / 2);
        auto *vv_index_maps = new std::vector<std::pair<ull, uint>>*[n_threads];
        auto *vv_cell_begin = new std::vector<uint>*[n_threads];
        auto *vv_cell_ns = new std::vector<uint>*[n_threads];
        auto v_no_cells = new std::vector<uint>[n_threads];


//        auto ***cell_indexes = new uint**[max_levels];
//        auto **cell_ns = new uint*[max_levels];
//        auto **dims_mult = new ull*[max_levels];
//        auto **cell_dims_min = new float*[max_levels];
//        auto **cell_dims_max = new float*[max_levels];
//        std::vector<float> v_eps_levels(max_levels);
//        std::vector<uint> v_no_of_cells(max_levels, 0);

//        allocate_resources(v_eps_levels, dims_mult, min_bounds, max_bounds, max_levels, max_d, e_inner);
        auto t2 = std::chrono::high_resolution_clock::now();
        if (!g_quiet) {
            std::cout << "Memory and init: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
                      << " milliseconds\n";
        }

        t1 = std::chrono::high_resolution_clock::now();
//        index_points_to_cells_omp_median_merge(v_coords, cell_indexes, cell_ns, min_bounds, dims_mult, v_eps_levels,
//        v_no_of_cells,max_levels, max_d, n, n_threads);
        int max_levels = index_points(v_coords, vv_index_maps, vv_cell_begin, vv_cell_ns, v_no_cells, max_d, n,
                n_threads, e_inner);
        t2 = std::chrono::high_resolution_clock::now();
        if (!g_quiet) {
            std::cout << "Point indexing: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
                      << " milliseconds\n";
        }

//        for (uint l = 0; l < max_levels; l++) {
//            delete [] dims_mult[l];
//        }
//        delete [] dims_mult;

        /*
        t1 = std::chrono::high_resolution_clock::now();
        calculate_cell_boundaries_omp(v_coords, cell_indexes, cell_ns, cell_dims_min, cell_dims_max, v_no_of_cells,
                                      max_levels, max_d);
        t2 = std::chrono::high_resolution_clock::now();
        if (!g_quiet) {
            std::cout << "Cell boundaries: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
                      << " milliseconds\n";
        }

        t1 = std::chrono::high_resolution_clock::now();
        process_cell_tree_omp_2(p_labels, v_coords, cell_indexes, cell_ns, cell_dims_min, cell_dims_max, v_no_of_cells,
                is_core, n_threads, max_levels, max_d, e, e2, m, n);
        t2 = std::chrono::high_resolution_clock::now();
        if (!g_quiet) {
            std::cout << "Process cell tree: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
                      << " milliseconds\n";
        }
         */
    }

    void read_input(const std::string &in_file, float *v_points, int max_d) noexcept {
        std::ifstream is(in_file);
        std::string line, buf;
        std::stringstream ss;
        int index = 0;
        auto t1 = std::chrono::high_resolution_clock::now();
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
        auto t2 = std::chrono::high_resolution_clock::now();

        if (!g_quiet) {
            std::cout << std::endl;
            std::cout << "Read input took: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
                      << " milliseconds\n";
        }
    }

    result calculate_output(const bool_vector &is_core, struct_label** ps, int n) noexcept {
        result res{0, 0, 0, new std::vector<int>(n)};

        for (int i = 0; i < n; i++) {
            if (is_core[i]) {
                ++res.core_count;
            }
        }
        bool_vector labels(n);
        labels.fill(n, false);
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            int label = get_label(ps[i])->label;
            (*res.point_clusters)[i] = label;
            if (label != UNASSIGNED) {
                labels[label] = true;
            }
        }
        uint& clusters = res.clusters;
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

        uint& noise = res.noise;
        #pragma omp parallel for reduction(+: noise)
        for (int i = 0; i < n; i++) {
            if (get_label(ps[i])->label == UNASSIGNED) {
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

    result start(const uint m, const float e, const uint n_threads, const std::string &in_file) noexcept {
        uint n, max_d;
        count_lines_and_dimensions(in_file, n, max_d);

        std::cout << "Found " << n << " points in " << max_d << " dimensions" << std::endl;

        auto *v_points = new float[n*max_d];
        read_input(in_file, v_points, max_d);

        auto **point_labels = new struct_label *[n];
        for (uint i = 0; i < n; i++) {
            point_labels[i] = new struct_label();
        }

        bool_vector is_core(n);
        is_core.fill(n, false);

        auto t1 = std::chrono::high_resolution_clock::now();
        nextDBSCAN(point_labels, v_points, m, e, n, max_d, is_core, n_threads);
        auto t2 = std::chrono::high_resolution_clock::now();
        if (!g_quiet) {
            std::cout << "Execution time (excluding I/O): "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
                      << " milliseconds\n";
        }
        result results = calculate_output(is_core, point_labels, n);

        delete [] v_points;
        for (uint i = 0; i < n; i++) {
            delete [] point_labels[i];
        }
        delete [] point_labels;
        return results;
    }

}