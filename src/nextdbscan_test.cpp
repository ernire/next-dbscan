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
#include <vector>
#include <cstring>
#include <fstream>
#include <cstdint>
#include <omp.h>
#include <getopt.h>

#include "gtest/gtest.h"

static const int UNASSIGNED = -1;
typedef unsigned long long ull;
typedef unsigned int uint;

static bool g_quiet = false;

class struct_label {
public:
    int label;
    struct_label* label_p;
    struct_label() {
        label = UNASSIGNED;
        label_p = nullptr;
    }
};

struct scan_result {
    uint clusters;
    uint noise;
    uint core_count;
};

inline struct_label* get_label(struct_label *p) {
    while(p->label_p != nullptr)
        p = p->label_p;
    return p;
}

inline void calc_bounds(const float *v_coords, uint n, std::vector<float> &min_bounds, std::vector<float> &max_bounds, uint max_d) {
    for (uint d = 0; d < max_d; d++) {
        min_bounds[d] = INT32_MAX;
        max_bounds[d] = INT32_MIN;
    }
    for (uint i = 0; i < n; i++) {
        size_t index = i * max_d;
        for (uint d = 0; d < max_d; d++) {
            if (v_coords[index+d] > max_bounds[d]) {
                max_bounds[d] = v_coords[index+d];
            }
            if (v_coords[index+d] < min_bounds[d]) {
                min_bounds[d] = v_coords[index+d];
            }
        }
    }
}

inline void calc_dims_mult(ull *dims_mult, const uint max_d, 
        const std::vector<float> &min_bounds, const std::vector<float> &max_bounds,
        const float e_inner) {
    std::vector<uint> dims;
    dims.resize(max_d);
    dims_mult[0] = 1;
    for (uint d = 0; d < max_d; d++) {
        dims[d] = static_cast<uint>((max_bounds[d] - min_bounds[d]) / e_inner) + 1;
        if (d > 0)
            dims_mult[d] = dims_mult[d - 1] * dims[d - 1];
    }
}

inline bool dist_leq(const float *coord1, const float *coord2, int max_d, float e2) {
    float tmp = 0, tmp2;
    for (int d = 0; d < max_d; d++) {
        tmp2 = coord1[d] - coord2[d];
        tmp += tmp2*tmp2;
    }
    return tmp <= e2;
}

inline ull get_cell_index(const float *dv, const std::vector<float> &mv, const ull *dm, const uint max_d, const float size) {
    ull cell_index = 0;
    uint local_index;
    for (uint d = 0; d < max_d; d++) {
        local_index = static_cast<uint>((dv[d] - mv[d]) / size);
        cell_index += local_index * dm[d];
    }
    return cell_index;
}

inline bool is_in_reach(const float *min1, const float *max1, const float *min2, const float *max2, const uint max_d,
        const float e) noexcept {
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

void process_new_core_point(struct_label **p_labels, uint **cell_indexes, struct_label *p1_label, const bool *is_core,
        const uint c1_id, const uint size1, uint index) noexcept {
    bool has_other_cores = false;
    for (uint k = 0; k < size1; k++) {
        if (k == index)
            continue;
        if (is_core[cell_indexes[c1_id][k]]) {
            has_other_cores = true;
        }
    }
    if (!has_other_cores) {
        // first core in cell
        if (p1_label->label == UNASSIGNED) {
            p1_label->label = c1_id;
        }
        for (uint k = 0; k < size1; k++) {
            if (k == index)
                continue;
            p_labels[cell_indexes[c1_id][k]]->label_p = p1_label;
        }
    }
}

void process_point_labels_in_range(struct_label **p_labels, uint **cell_indexes, const bool *range_table,
        const uint *v_cell_ns, const bool *is_core, const uint c1_id, const uint c2_id) noexcept {
    int size1 = v_cell_ns[c1_id];
    int size2 = v_cell_ns[c2_id];
    int index = 0;
    int p1_id, p2_id;
    for (int i = 0; i < size1; i++) {
        for (int j = 0; j < size2; j++, index++) {
            if (range_table[index]) {
                p1_id = cell_indexes[c1_id][i];
                p2_id = cell_indexes[c2_id][j];

                if (is_core[p1_id] && is_core[p2_id]) {
                    auto *p1_label = get_label(p_labels[cell_indexes[c1_id][0]]);
                    auto *p2_label = get_label(p_labels[cell_indexes[c2_id][0]]);
                    if (p1_label != p2_label) {
                        set_lower_label(p1_label, p2_label);
                    }
                } else if (is_core[p1_id]) {
                    auto *p1_label = get_label(p_labels[cell_indexes[c1_id][0]]);
                    auto *p2_label = get_label(p_labels[p2_id]);
                    if (p2_label->label == UNASSIGNED) {
                        p2_label->label_p = p1_label;
                    }
                } else if (is_core[p2_id]) {
                    auto *p2_label = get_label(p_labels[cell_indexes[c2_id][0]]);
                    auto *p1_label = get_label(p_labels[p1_id]);
                    if (p1_label->label == UNASSIGNED) {
                        p1_label->label_p = p2_label;
                    }
                }
            }
        }
    }
}

void apply_marked_in_range(uint **cell_indexes, const bool *range_table, std::vector<uint> &v_point_nps, const uint *v_cell_ns,
        const bool *is_core, const std::vector<uint8_t> &is_border_cell, const uint c1_id, const uint c2_id) {
    uint size1 = v_cell_ns[c1_id];
    uint size2 = v_cell_ns[c2_id];
    uint index = 0;
    for (uint i = 0; i < size1; i++) {
        for (uint j = 0; j < size2; j++, index++) {
            if (range_table[index]) {
                uint p1_id = cell_indexes[c1_id][i];
                uint p2_id = cell_indexes[c2_id][j];
                if (is_core[p1_id]) {
                    if (!is_border_cell[c2_id]) {
                        ++v_point_nps[p2_id];
                    }
                } else if (is_core[p2_id]) {
                    if (!is_border_cell[c1_id]) {
                        ++v_point_nps[p1_id];
                    }
                } else {
                    if (!is_border_cell[c1_id]) {
                        ++v_point_nps[p1_id];
                    }
                    if (!is_border_cell[c2_id]) {
                        ++v_point_nps[p2_id];
                    }
                }
            }
        }
    }
}

int mark_in_range(const float *v_coords, const uint *v_c1_index, const uint size1, const uint *v_c2_index,
        const uint size2, bool *range_table, const uint max_d, const float e2) {
    std::fill(range_table, range_table + (size1 * size2), false);
    uint cnt_range = 0;
    uint index = 0;
    for (uint i = 0; i < size1; i++) {
        for (uint j = 0; j < size2; j++, index++) {
            if (dist_leq(&v_coords[v_c1_index[i]*max_d], &v_coords[v_c2_index[j]*max_d], max_d, e2)) {
                ++cnt_range;
                range_table[index] = true;
            }
        }
    }
    return cnt_range;
}

void process_ac_ac(struct_label **p_labels, float *v_coords, const uint *v_c1_index, const uint size1,
        const uint *v_c2_index, const uint size2, const uint max_d, const float e2) {
    struct_label *c1_label = get_label(p_labels[v_c1_index[0]]);
    struct_label *c2_label = get_label(p_labels[v_c2_index[0]]);
    if (c1_label->label == c2_label->label)
        return;
    for (uint i = 0; i < size1; i++) {
        for (uint j = 0; j < size2; j++) {
            if (dist_leq(&v_coords[v_c1_index[i]*max_d], &v_coords[v_c2_index[j]*max_d], max_d, e2)) {
                set_lower_label(c1_label, c2_label);
                return;
            }
        }
    }
}

void process_new_core_cell(struct_label **ps, uint **cell_indexes, bool *cell_has_cores, const uint *v_cell_ns,
        const std::vector<uint> &v_cell_nps, const std::vector<uint> &v_point_nps, bool *is_core, const uint c1_id, const uint m) {
    uint size = v_cell_ns[c1_id];
    for (uint i = 0; i < size; i++) {
        uint p1_id = cell_indexes[c1_id][i];
        if (!is_core[p1_id] && (v_cell_nps[c1_id] + v_point_nps[p1_id]) >= m) {
            cell_has_cores[c1_id] = true;
            is_core[p1_id] = true;
            auto *p1_label = get_label(ps[cell_indexes[c1_id][i]]);
            if (p1_label->label == UNASSIGNED) {
                p1_label->label = c1_id;
            }
            process_new_core_point(ps, cell_indexes, p1_label, is_core, c1_id, size, i);
        }
    }
}

void process_nc_labels(struct_label **p_labels, const float *v_coords, uint **cell_indexes, const uint *v_cell_ns,
        bool *range_table, const bool *cell_has_cores,const bool *is_core, const uint c1_id, const uint c2_id,
        const uint max_d, const float e2) {
    int size1 = v_cell_ns[c1_id];
    int size2 = v_cell_ns[c2_id];
    int cnt_range = mark_in_range(v_coords, cell_indexes[c1_id], size1, cell_indexes[c2_id], size2, range_table,
            max_d, e2);
    if (cnt_range == 0) {
        return;
    }
    if (cnt_range == size1 * size2) {
        if (cell_has_cores[c1_id] && cell_has_cores[c2_id]) {
            auto *p1 = get_label(p_labels[cell_indexes[c1_id][0]]);
            auto *p2 = get_label(p_labels[cell_indexes[c2_id][0]]);
            if (p1 != p2) {
                set_lower_label(p1, p2);
            }
        } else if (cell_has_cores[c1_id]) {
            auto *p = get_label(p_labels[cell_indexes[c1_id][0]]);
            for (uint i = 0; i < v_cell_ns[c2_id]; i++) {
                p_labels[cell_indexes[c2_id][i]]->label_p = p;
            }
        } else if (cell_has_cores[c2_id]) {
            auto *p = get_label(p_labels[cell_indexes[c2_id][0]]);
            for (uint i = 0; i < v_cell_ns[c1_id]; i++) {
                p_labels[cell_indexes[c1_id][i]]->label_p = p;
            }
        }
    } else if (cell_has_cores[c1_id] || cell_has_cores[c2_id]) {
        process_point_labels_in_range(p_labels, cell_indexes, range_table, v_cell_ns, is_core, c1_id, c2_id);
    }
}

void process_nc_nc(struct_label **p_labels, const float *v_coords, uint **cell_indexes, const uint *v_cell_ns,
        bool *range_table, bool *cell_has_cores, bool *is_core, const std::vector<uint8_t> &is_border_cell, 
        std::vector<uint> &v_point_nps, std::vector<uint> &v_cell_nps, const uint c1_id, const uint c2_id, const uint max_d, 
        const float e2, const uint m) {
    uint size1 = v_cell_ns[c1_id];
    uint size2 = v_cell_ns[c2_id];
    uint cnt_range = mark_in_range(v_coords, cell_indexes[c1_id], size1, cell_indexes[c2_id], size2, range_table, max_d,
            e2);
    if (cnt_range == 0) {
        return;
    }
    if (cnt_range == size1 * size2) {
        if (cell_has_cores[c1_id] && cell_has_cores[c2_id]) {
            v_cell_nps[c1_id] += size2;
            v_cell_nps[c2_id] += size1;
        } else if (cell_has_cores[c1_id]) {
            v_cell_nps[c1_id] += size2;
            if (!is_border_cell[c2_id]) {
                v_cell_nps[c2_id] += size1;
            }
        } else if (cell_has_cores[c2_id]) {
            v_cell_nps[c2_id] += size1;
            if (!is_border_cell[c1_id]) {
                v_cell_nps[c1_id] += size2;
            }
        } else {
            if (!is_border_cell[c1_id]) {
                v_cell_nps[c1_id] += size2;
            }
            if (!is_border_cell[c2_id]) {
                v_cell_nps[c2_id] += size1;
            }
        }
    } else {
        if (!is_border_cell[c1_id] || !is_border_cell[c2_id]) {
            apply_marked_in_range(cell_indexes, range_table, v_point_nps, v_cell_ns, is_core, is_border_cell, c1_id,
                    c2_id);
        }
    }
    if (!is_border_cell[c1_id]) {
        process_new_core_cell(p_labels, cell_indexes, cell_has_cores, v_cell_ns, v_cell_nps, v_point_nps, is_core, c1_id, m);
    }
    if (!is_border_cell[c2_id]) {
        process_new_core_cell(p_labels, cell_indexes, cell_has_cores, v_cell_ns, v_cell_nps, v_point_nps, is_core, c2_id, m);
    }
}

inline uint traverse_and_get_cell_index(uint ***cell_indexes, const int l, const uint i) {
    int level_mod = 1;
    uint cell_index = i;
    while (l - level_mod >= 0) {
        cell_index = cell_indexes[l-level_mod][cell_index][0];
        ++level_mod;
    }
    return cell_index;
}

inline void allocate_resources(std::vector<float> &v_eps_levels, ull **dims_mult, 
        const std::vector<float> &min_bounds, const std::vector<float> &max_bounds,
        const uint max_levels, const uint max_d, float e_inner) {
    for (uint i = 0; i < max_levels; i++) {
        v_eps_levels[i] = (e_inner * powf(2, i));
        dims_mult[i] = new ull[max_d];
        calc_dims_mult(dims_mult[i], max_d, min_bounds, max_bounds, v_eps_levels[i]);
    }
}

void calculate_cell_boundaries_omp(float *v_coords, uint ***cell_indexes, uint **cell_ns, float **cell_dims_min,
        float **cell_dims_max, const std::vector<uint> &v_no_of_cells, const uint max_levels, const uint max_d) {
    float *coord_min, *coord_max;
    for (uint l = 0; l < max_levels; l++) {
        cell_dims_min[l] = new float[v_no_of_cells[l]*max_d];
        cell_dims_max[l] = new float[v_no_of_cells[l]*max_d];
    }
    for (uint l = 0; l < max_levels; l++) {
        #pragma omp parallel for private(coord_min, coord_max)
        for (uint i = 0; i < v_no_of_cells[l]; i++) {
            if (l == 0) {
                coord_min = &v_coords[cell_indexes[l][i][0]*max_d];
                coord_max = &v_coords[cell_indexes[l][i][0]*max_d];
            } else {
                coord_min = &cell_dims_min[l-1][cell_indexes[l][i][0]*max_d];
                coord_max = &cell_dims_max[l-1][cell_indexes[l][i][0]*max_d];
            }
            std::copy(coord_min, coord_min + max_d, &cell_dims_min[l][i*max_d]);
            std::copy(coord_max, coord_max + max_d, &cell_dims_max[l][i*max_d]);
            for (uint j = 1; j < cell_ns[l][i]; j++) {
                if (l == 0) {
                    coord_min = &v_coords[cell_indexes[l][i][j]*max_d];
                    coord_max = &v_coords[cell_indexes[l][i][j]*max_d];
                } else {
                    coord_min = &cell_dims_min[l-1][cell_indexes[l][i][j]*max_d];
                    coord_max = &cell_dims_max[l-1][cell_indexes[l][i][j]*max_d];
                }
                for (uint d = 0; d < max_d; d++) {
                    if (coord_min[d] < cell_dims_min[l][i*max_d+d]) {
                        cell_dims_min[l][i*max_d+d] = coord_min[d];
                    }
                    if (coord_max[d] > cell_dims_max[l][i*max_d+d]) {
                        cell_dims_max[l][i*max_d+d] = coord_max[d];
                    }
                }
            }
        }
    }
}

void process_cell_tree_omp(struct_label **ps_origin, float *v_coords, uint ***cell_indexes, uint ** cell_ns,
        float **cell_dims_min,float **cell_dims_max, const std::vector<uint> &v_no_of_cells, bool *is_core,
        const std::vector<uint8_t> &is_border_cell, uint **s_c1_indexes, uint **s_c2_indexes, uint **s_levels, uint n_threads,
        uint max_levels, uint max_d, float e, float e2, uint m, const uint n) noexcept {
    uint max_points_in_cell = 0;
    std::vector<uint> v_cell_nps(v_no_of_cells[0]);
    auto **range_table = new bool*[n_threads];
    std::vector<uint> v_point_nps(n, 0);

    bool *cell_has_cores = new bool[v_no_of_cells[0]];
    std::fill(cell_has_cores, cell_has_cores + v_no_of_cells[0], false);

    #pragma omp parallel for reduction(max: max_points_in_cell)
    for (uint i = 0; i < v_no_of_cells[0]; i++) {
        v_cell_nps[i] = cell_ns[0][i];
        if (v_cell_nps[i] > max_points_in_cell) {
            max_points_in_cell = v_cell_nps[i];
        }
        if (v_cell_nps[i] >= m) {
            cell_has_cores[i] = true;
            ps_origin[cell_indexes[0][i][0]]->label = i;
            for (uint j = 0; j < v_cell_nps[i]; j++) {
                is_core[cell_indexes[0][i][j]] = true;
                if (j > 0) {
                    ps_origin[cell_indexes[0][i][j]]->label_p = ps_origin[cell_indexes[0][i][0]];
                }
            }
        }
    }
    for (uint i = 0; i < n_threads; i++) {
        range_table[i] = new bool[max_points_in_cell*std::min(max_points_in_cell, m)];
    }
    for (uint level = 1; level < max_levels; level++) {
        #pragma omp parallel for schedule(dynamic)
        for (uint i = 0; i < v_no_of_cells[level]; i++) {
            uint t_id = omp_get_thread_num();
            uint s_index = 0;
            for (uint j = 0; j < cell_ns[level][i]; j++) {
                for (uint k = j+1; k < cell_ns[level][i]; k++) {
                    s_levels[t_id][s_index] = level-1;
                    s_c1_indexes[t_id][s_index] = cell_indexes[level][i][j];
                    s_c2_indexes[t_id][s_index] = cell_indexes[level][i][k];
                    ++s_index;
                }
            }
            while (s_index > 0) {
                --s_index;
                uint l = s_levels[t_id][s_index];
                uint c1 = s_c1_indexes[t_id][s_index];
                uint c2 = s_c2_indexes[t_id][s_index];
                if (l == 0 && is_border_cell[c1] && is_border_cell[c2]) {
                    continue;
                }
                if (is_in_reach(&cell_dims_min[l][c1*max_d], &cell_dims_max[l][c1*max_d], &cell_dims_min[l][c2*max_d],
                                &cell_dims_max[l][c2*max_d], max_d, e)) {
                    if (l == 0) {
                        if (!(v_cell_nps[c1] >= m && v_cell_nps[c2] >= m) && !(is_border_cell[c1] && is_border_cell[c2])) {
                            process_nc_nc(ps_origin, v_coords, cell_indexes[l], cell_ns[l], range_table[t_id],
                                    cell_has_cores, is_core, is_border_cell, v_point_nps, v_cell_nps,
                                          c1, c2, max_d, e2, m);
                        }
                    } else {
                        for (uint j = 0; j < cell_ns[l][c1]; j++) {
                            for (uint k = 0; k < cell_ns[l][c2]; k++) {
                                s_levels[t_id][s_index] = l-1;
                                s_c1_indexes[t_id][s_index] = cell_indexes[l][c1][j];
                                s_c2_indexes[t_id][s_index] = cell_indexes[l][c2][k];
                                ++s_index;
                            }
                        }
                    }
                }
            }
        }
    }
    for (uint level = 1; level < max_levels; level++) {
        #pragma omp parallel for schedule(dynamic)
        for (uint i = 0; i < v_no_of_cells[level]; i++) {
            uint t_id = omp_get_thread_num();
            uint s_index = 0;
            for (uint j = 0; j < cell_ns[level][i]; j++) {
                for (uint k = j+1; k < cell_ns[level][i]; k++) {
                    s_levels[t_id][s_index] = level-1;
                    s_c1_indexes[t_id][s_index] = cell_indexes[level][i][j];
                    s_c2_indexes[t_id][s_index] = cell_indexes[level][i][k];
                    ++s_index;
                }
            }
            while (s_index > 0) {
                --s_index;
                uint l = s_levels[t_id][s_index];
                uint c1 = s_c1_indexes[t_id][s_index];
                uint c2 = s_c2_indexes[t_id][s_index];
                if (l == 0 && is_border_cell[c1] && is_border_cell[c2]) {
                    continue;
                }
                if (is_in_reach(&cell_dims_min[l][c1*max_d], &cell_dims_max[l][c1*max_d], &cell_dims_min[l][c2*max_d],
                        &cell_dims_max[l][c2*max_d], max_d, e)) {
                    if (l == 0) {
                        if (v_cell_nps[c1] >= m && v_cell_nps[c2] >= m) {
                            process_ac_ac(ps_origin, v_coords, cell_indexes[l][c1], cell_ns[l][c1], cell_indexes[l][c2],
                                          cell_ns[l][c2], max_d, e2);
                        } else if (cell_has_cores[c1] || cell_has_cores[c2]) {
                            process_nc_labels(ps_origin, v_coords, cell_indexes[l], cell_ns[l], range_table[t_id],
                                          cell_has_cores, is_core, c1, c2, max_d, e2);
                        }
                    } else {
                        for (uint j = 0; j < cell_ns[l][c1]; j++) {
                            for (uint k = 0; k < cell_ns[l][c2]; k++) {
                                s_levels[t_id][s_index] = l-1;
                                s_c1_indexes[t_id][s_index] = cell_indexes[l][c1][j];
                                s_c2_indexes[t_id][s_index] = cell_indexes[l][c2][k];
                                ++s_index;
                            }
                        }
                    }
                }
            }
        }
    }
}

void detect_border_cells(uint ***cell_indexes, uint **cell_ns, float **cell_dims_min, float **cell_dims_max,
        std::vector<uint8_t> &border_cells, const std::vector<uint> &v_no_of_cells, uint **s_c1_indexes, uint **s_c2_indexes, uint **s_levels,
        const uint max_levels, const uint max_d, const uint m, const float e) {
    auto *v_cell_nps = new uint[v_no_of_cells[0]];
    std::copy(cell_ns[0], cell_ns[0] + v_no_of_cells[0], v_cell_nps);
    for (uint level = 1; level < max_levels; level++) {
        #pragma omp parallel for
        for (uint i = 0; i < v_no_of_cells[level]; i++) {
            int t_id = omp_get_thread_num();
            int s_index = 0;
            for (uint j = 0; j < cell_ns[level][i]; j++) {
                for (uint k = j + 1; k < cell_ns[level][i]; k++) {
                    s_levels[t_id][s_index] = level - 1;
                    s_c1_indexes[t_id][s_index] = cell_indexes[level][i][j];
                    s_c2_indexes[t_id][s_index] = cell_indexes[level][i][k];
                    ++s_index;
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
                                ++s_index;
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
    delete [] v_cell_nps;
}

void index_cells_omp_simple(const uint no_of_cells, std::vector<std::pair<ull, uint>> *vec_index_maps,
        uint ***cell_indexes, uint **cell_ns, std::vector<uint> *vec_cell_begin, float *v_coords,
        const std::vector<float> &min_bounds, ull **dims_mult, const std::vector<float> &v_eps_levels, std::vector<uint> &v_no_of_cells,
        const uint max_d, const uint l, const uint n_threads) {
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
        std::vector<std::pair<ull, uint>>::const_iterator end) {
    return begin + ((end - begin) / 2);
}

void fill_medians(std::vector<std::pair<ull, uint>>::const_iterator begin,
        std::vector<std::pair<ull, uint>>::const_iterator end, ull *medians, uint index1, uint index2, uint last) {
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
        std::vector<uint> &v_no_of_cells, ull *selected_medians, const uint max_d, const uint l, const uint n_threads) {
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

void index_points_to_cells_omp_median_merge(float *v_coords, uint ***cell_indexes, uint **cell_ns,
        const std::vector<float> &min_bounds, ull **dims_mult, const std::vector<float> &v_eps_levels, std::vector<uint> &v_no_of_cells, int max_levels,
        const uint max_d, const uint n, const uint n_threads) {
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
        if (n_threads > 2 && no_of_cells > n_threads*100) {
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
        const uint max_d, bool *is_core, uint n_threads) {
    std::vector<float> min_bounds;
    min_bounds.resize(max_d);

    std::vector<float> max_bounds;
    max_bounds.resize(max_d);
    
    float max_limit = INT32_MIN;
    omp_set_num_threads(n_threads);
    auto t1 = std::chrono::high_resolution_clock::now();
    float e2 = e * e;
    float e_inner = (e / sqrt(2)) / 2;

    calc_bounds(v_coords, n, min_bounds, max_bounds, max_d);
    for (uint d = 0; d < max_d; d++) {
        if (max_bounds[d] - min_bounds[d] > max_limit)
            max_limit = max_bounds[d] - min_bounds[d];
    }
    uint max_levels = static_cast<int>(ceilf(logf(max_limit / e_inner) / logf(2))) + 1;
    auto ***cell_indexes = new uint**[max_levels];
    auto **cell_ns = new uint*[max_levels];
    auto **dims_mult = new ull*[max_levels];
    auto **cell_dims_min = new float*[max_levels];
    auto **cell_dims_max = new float*[max_levels];
    std::vector<float> v_eps_levels(max_levels);
    std::vector<uint> v_no_of_cells(max_levels, 0);
    
    // stacks
    auto** s_levels = new uint*[n_threads];
    auto** s_c1_indexes = new uint*[n_threads];
    auto** s_c2_indexes = new uint*[n_threads];

    allocate_resources(v_eps_levels, dims_mult, min_bounds, max_bounds, max_levels, max_d, e_inner);
    auto t2 = std::chrono::high_resolution_clock::now();
    if (!g_quiet) {
        std::cout << "Memory and init: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
                  << " milliseconds\n";
    }

    t1 = std::chrono::high_resolution_clock::now();
    index_points_to_cells_omp_median_merge(v_coords, cell_indexes, cell_ns, min_bounds, dims_mult, v_eps_levels,
            v_no_of_cells,max_levels, max_d, n, n_threads);
    t2 = std::chrono::high_resolution_clock::now();
    if (!g_quiet) {
        std::cout << "Point indexing: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
                  << " milliseconds\n";
    }

    for (uint l = 0; l < max_levels; l++) {
        delete [] dims_mult[l];
    }
    delete [] dims_mult;
    for (uint i = 0; i < n_threads; i++) {
        // TODO use vectors instead of hard coded heuristic
        s_levels[i] = new uint[v_no_of_cells[0]*10];
        s_c1_indexes[i] = new uint[v_no_of_cells[0]*10];
        s_c2_indexes[i] = new uint[v_no_of_cells[0]*10];
    }
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
    std::vector<uint8_t> border_cells(v_no_of_cells[0], 0);
    detect_border_cells(cell_indexes, cell_ns, cell_dims_min, cell_dims_max, border_cells, v_no_of_cells,
            s_c1_indexes, s_c2_indexes, s_levels, max_levels, max_d, m, e);
    t2 = std::chrono::high_resolution_clock::now();
    if (!g_quiet) {
        std::cout << "Border/noise cell detection: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
                  << " milliseconds\n";
    }

    t1 = std::chrono::high_resolution_clock::now();
    process_cell_tree_omp(p_labels, v_coords, cell_indexes, cell_ns, cell_dims_min, cell_dims_max, v_no_of_cells,
                            is_core, border_cells, s_c1_indexes, s_c2_indexes,
                            s_levels, n_threads, max_levels, max_d, e, e2, m, n);
    t2 = std::chrono::high_resolution_clock::now();
    if (!g_quiet) {
        std::cout << "Process cell tree: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
                  << " milliseconds\n";
    }
}

void read_input(const std::string &in_file, float *v_points, int max_d) {
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

scan_result calculate_output(const bool *is_core, struct_label** ps, int n) {
    scan_result result{0, 0, 0};
    
    for (int i = 0; i < n; i++) {
        if (is_core[i]) {
            ++result.core_count;
        }
    }

    auto* labels = new bool[n];
    std::fill(labels, labels + n, false);

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        if (get_label(ps[i])->label != UNASSIGNED) {
            labels[get_label(ps[i])->label] = true;
        }
    }

    uint& clusters = result.clusters;
    #pragma omp parallel for reduction(+: clusters)
    for (int i = 0; i < n; i++) {
        if (labels[i]) {
            ++clusters;
        }
    }

    uint& noise = result.noise;
    #pragma omp parallel for reduction(+: noise)
    for (int i = 0; i < n; i++) {
        if (get_label(ps[i])->label == UNASSIGNED) {
            ++noise;
        }
    }

    delete [] labels;

    return result;
}

int count_lines(const std::string &in_file) {
    std::ifstream is(in_file);
    std::string line;
    int cnt = 0;
    while (std::getline(is, line)) {
        ++cnt;
    }
    return cnt;
}

scan_result start_nextdbscan(const uint m, const float e, const uint max_d, const uint n_threads, const std::string &in_file) {
    uint n = count_lines(in_file);
    
    auto *v_points = new float[n*max_d];
    read_input(in_file, v_points, max_d);
    
    auto **point_labels = new struct_label *[n];
    for (uint i = 0; i < n; i++) {
        point_labels[i] = new struct_label();
    }
    
    auto *is_core = new bool[n];
    std::fill(is_core, is_core + n, false);
    nextDBSCAN(point_labels, v_points, m, e, n, max_d, is_core, n_threads);

    scan_result results = calculate_output(is_core, point_labels, n);

    delete [] v_points;
    delete [] is_core;
    for (uint i = 0; i < n; i++) {
        delete [] point_labels[i];
    }
    delete [] point_labels;

    return results;
}

TEST(ScanAccuracyAloiHsb2x2x2, m20_e001_d8_t2) {
    scan_result result = start_nextdbscan(20, 0.01, 8, 2, "../input/aloi-hsb-2x2x2.csv");
    EXPECT_EQ(result.clusters, 129);
    EXPECT_EQ(result.core_count, 55256);
    EXPECT_EQ(result.noise, 43312);
}

TEST(ScanAccuracyAloiHsb2x2x2, m10_e001_d8_t2) {
    scan_result result = start_nextdbscan(10, 0.01, 8, 2, "../input/aloi-hsb-2x2x2.csv");
    EXPECT_EQ(result.clusters, 297);
    EXPECT_EQ(result.core_count, 69606);
    EXPECT_EQ(result.noise, 30977);
}

TEST(ScanAccuracyAloiHsb2x2x2, m20_e001_d4_t2) {
    scan_result result = start_nextdbscan(20, 0.01, 4, 2, "../input/aloi-hsb-2x2x2.csv");
    EXPECT_EQ(result.clusters, 92);
    EXPECT_EQ(result.core_count, 72578);
    EXPECT_EQ(result.noise, 27284);
}

TEST(ScanAccuracyBremenSmall, m30_e20_d3_t2) {
    scan_result result = start_nextdbscan(30, 20, 3, 2, "../input/bremenSmall.csv");
    EXPECT_EQ(result.clusters, 2972);
    EXPECT_EQ(result.core_count, 1234425);
    EXPECT_EQ(result.noise, 1463461);
}

TEST(ScanAccuracyBremenSmall, m30_e10_d3_t2) {
    scan_result result = start_nextdbscan(30, 10, 3, 2, "../input/bremenSmall.csv");
    EXPECT_EQ(result.clusters, 1202);
    EXPECT_EQ(result.core_count, 263100);
    EXPECT_EQ(result.noise, 2641881);
}

TEST(ScanAccuracyBremenSmall, m10_e20_d3_t2) {
    scan_result result = start_nextdbscan(10, 20, 3, 2, "../input/bremenSmall.csv");
    EXPECT_EQ(result.clusters, 10712);
    EXPECT_EQ(result.core_count, 2286646);
    EXPECT_EQ(result.noise, 482406);
}

TEST(ScanAccuracyBremenSmall, m100_e30_d3_t2) {
    scan_result result = start_nextdbscan(100, 30, 3, 2, "../input/bremenSmall.csv");
    EXPECT_EQ(result.clusters, 480);
    EXPECT_EQ(result.core_count, 839729);
    EXPECT_EQ(result.noise, 1906449);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    g_quiet = true;
    return RUN_ALL_TESTS();
}
