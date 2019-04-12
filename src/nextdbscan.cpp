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
#include <assert.h>
#include <algorithm>
#include <vector>
#include <cstring>
#include <omp.h>
#include <fstream>
#include <getopt.h>

static const int UNASSIGNED = -1;

class struct_label {
public:
    int label;
    struct_label* label_p;
    struct_label() {
        label = UNASSIGNED;
        label_p = nullptr;
    }
};

inline struct_label* get_label(struct_label* p) {
    while(p->label_p != nullptr)
        p = p->label_p;
    return p;
}

inline void calc_bounds(const float *v_coords, int n, float* min_bounds, float* max_bounds, int max_d) {
    int i, d;

    for (d = 0; d < max_d; d++) {
        min_bounds[d] = INT32_MAX;
        max_bounds[d] = INT32_MIN;
    }
    for (i = 0; i < n; i++) {
        int index = i * max_d;
        for (d = 0; d < max_d; d++) {
            if (v_coords[index+d] > max_bounds[d]) {
                max_bounds[d] = v_coords[index+d];
            }
            if (v_coords[index+d] < min_bounds[d]) {
                min_bounds[d] = v_coords[index+d];
            }
        }
    }
}

inline void calc_dims_mult(size_t *dims_mult, int max_d, const float* min_bounds, const float* max_bounds, float e_inner) {
    int dims[max_d];
    dims_mult[0] = 1;
    for (int d = 0; d < max_d; d++) {
        dims[d] = static_cast<int>((max_bounds[d] - min_bounds[d]) / e_inner) + 1;
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

inline size_t get_cell_index(const float* dv, const float* mv, const size_t* dm, const int max_d, const float size) {
    size_t cell_index = 0;
    int local_index;
    for (int d = 0; d < max_d; d++) {
        local_index = static_cast<int>((dv[d] - mv[d]) / size);
        cell_index += local_index * dm[d];
    }
    return cell_index;
}

inline bool is_in_reach(const float* min1, const float* max1, const float* min2, const float* max2, const int max_d,
                        const float e) {
    for (int d = 0; d < max_d; d++) {
        if ((min2[d] > (max1[d] + e) || min2[d] < (min1[d] - e)) &&
            (min1[d] > (max2[d] + e) || min1[d] < (min2[d] - e)) &&
            (max2[d] > (max1[d] + e) || max2[d] < (min1[d] - e)) &&
            (max1[d] > (max2[d] + e) || max1[d] < (min2[d] - e))) {
            return false;
        }
    }
    return true;
}

inline void set_lower_label(struct_label* c1_label, struct_label* c2_label) {
    if (c1_label->label < c2_label->label) {
        c2_label->label_p = c1_label;
    } else {
        c1_label->label_p = c2_label;
    }
}

void process_new_core_point(struct_label **p_labels, int **cell_indexes, struct_label *p1_label, const bool *is_core, int c1_id, int size1,
                            int index) {
    bool has_other_cores = false;
    for (int k = 0; k < size1; k++) {
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
        for (int k = 0; k < size1; k++) {
            if (k == index)
                continue;
            p_labels[cell_indexes[c1_id][k]]->label_p = p1_label;
        }
    }
}

int process_point_labels_in_range(struct_label **p_labels, int **cell_indexes, const bool *range_table,
        const int *v_cell_ns, const bool *is_core, const int c1_id, const int c2_id) {
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

int apply_marked_in_range(int **cell_indexes, const bool *range_table, int *v_point_nps, const int *v_cell_ns,
        const bool *is_core, const bool *is_border_cell, int c1_id, int c2_id) {
    int size1 = v_cell_ns[c1_id];
    int size2 = v_cell_ns[c2_id];
    int index = 0;
    int p1_id, p2_id;
    for (int i = 0; i < size1; i++) {
        for (int j = 0; j < size2; j++, index++) {
            if (range_table[index]) {
                p1_id = cell_indexes[c1_id][i];
                p2_id = cell_indexes[c2_id][j];
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

int mark_in_range(const float *v_coords, const int *v_c1_index, const int size1, const int *v_c2_index, const int size2,
        bool *range_table, const int max_d, const float e2) {
    std::fill(range_table, range_table + (size1 * size2), false);
    int cnt_range = 0;
    int index = 0;
    for (int i = 0; i < size1; i++) {
        for (int j = 0; j < size2; j++, index++) {
            if (dist_leq(&v_coords[v_c1_index[i]*max_d], &v_coords[v_c2_index[j]*max_d], max_d, e2)) {
                ++cnt_range;
                range_table[index] = true;
            }
        }
    }
    return cnt_range;
}

void process_ac_ac(struct_label **p_labels, float *v_coords, const int *v_c1_index, const int size1,
        const int *v_c2_index, const int size2, int max_d, float e2) {
    struct_label *c1_label = get_label(p_labels[v_c1_index[0]]);
    struct_label *c2_label = get_label(p_labels[v_c2_index[0]]);
    if (c1_label->label == c2_label->label)
        return;
    for (int i = 0; i < size1; i++) {
        for (int j = 0; j < size2; j++) {
            if (dist_leq(&v_coords[v_c1_index[i]*max_d], &v_coords[v_c2_index[j]*max_d], max_d, e2)) {
                set_lower_label(c1_label, c2_label);
                return;
            }
        }
    }
}

void process_new_core_cell(struct_label **ps, int **cell_indexes, bool *cell_has_cores, const int *v_cell_ns,
                           const int *v_cell_np, const int *v_point_nps,
                           bool *is_core, int c1_id, int m) {
    int size = v_cell_ns[c1_id];
    for (int i = 0; i < size; i++) {
        int p1_id = cell_indexes[c1_id][i];
        if (!is_core[p1_id] && (v_cell_np[c1_id] + v_point_nps[p1_id]) >= m) {
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

void process_nc_labels(struct_label **p_labels, const float *v_coords, int **cell_indexes, const int *v_cell_ns,
        bool *range_table, const bool *cell_has_cores,const bool *is_core, const int c1_id, const int c2_id,
        const int max_d, const float e2) {
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
            for (int i = 0; i < v_cell_ns[c2_id]; i++) {
                p_labels[cell_indexes[c2_id][i]]->label_p = p;
            }
        } else if (cell_has_cores[c2_id]) {
            auto *p = get_label(p_labels[cell_indexes[c2_id][0]]);
            for (int i = 0; i < v_cell_ns[c1_id]; i++) {
                p_labels[cell_indexes[c1_id][i]]->label_p = p;
            }
        }
    } else if (cell_has_cores[c1_id] || cell_has_cores[c2_id]) {
        process_point_labels_in_range(p_labels, cell_indexes, range_table, v_cell_ns, is_core, c1_id, c2_id);
    }
}

void process_nc_nc(struct_label **p_labels, const float *v_coords, int **cell_indexes, const int *v_cell_ns,
        bool *range_table, bool *cell_has_cores, bool *is_core, bool *is_border_cell, int* v_point_nps, int* v_cell_np, const int c1_id,
        const int c2_id, const int max_d, const float e2, const int m) {
    int size1 = v_cell_ns[c1_id];
    int size2 = v_cell_ns[c2_id];
    int cnt_range = mark_in_range(v_coords, cell_indexes[c1_id], size1, cell_indexes[c2_id], size2, range_table, max_d,
            e2);
    if (cnt_range == 0) {
        return;
    }
    if (cnt_range == size1 * size2) {
        if (cell_has_cores[c1_id] && cell_has_cores[c2_id]) {
            v_cell_np[c1_id] += size2;
            v_cell_np[c2_id] += size1;
        } else if (cell_has_cores[c1_id]) {
            v_cell_np[c1_id] += size2;
            if (!is_border_cell[c2_id]) {
                v_cell_np[c2_id] += size1;
            }
        } else if (cell_has_cores[c2_id]) {
            v_cell_np[c2_id] += size1;
            if (!is_border_cell[c1_id]) {
                v_cell_np[c1_id] += size2;
            }
        } else {
            if (!is_border_cell[c1_id]) {
                v_cell_np[c1_id] += size2;
            }
            if (!is_border_cell[c2_id]) {
                v_cell_np[c2_id] += size1;
            }
        }
    } else {
        if (!is_border_cell[c1_id] || !is_border_cell[c2_id]) {
            apply_marked_in_range(cell_indexes, range_table, v_point_nps, v_cell_ns, is_core, is_border_cell, c1_id,
                    c2_id);
        }
    }
    if (!is_border_cell[c1_id]) {
        process_new_core_cell(p_labels, cell_indexes, cell_has_cores, v_cell_ns, v_cell_np, v_point_nps, is_core, c1_id, m);
    }
    if (!is_border_cell[c2_id]) {
        process_new_core_cell(p_labels, cell_indexes, cell_has_cores, v_cell_ns, v_cell_np, v_point_nps, is_core, c2_id, m);
    }
}

inline int traverse_and_get_cell_index(int*** cell_indexes, const int l, const int i) {
    int level_mod = 1;
    int cell_index = i;
    while (l - level_mod >= 0) {
        cell_index = cell_indexes[l-level_mod][cell_index][0];
        level_mod++;
    }
    return cell_index;
}

inline void allocate_resources(float *v_eps_levels, size_t **dims_mult, const float *min_bounds, const float *max_bounds,
        int max_levels, int max_d, float e_inner) {
    for (int i = 0; i < max_levels; i++) {
        v_eps_levels[i] = (e_inner * powf(2, i));
        dims_mult[i] = new size_t[max_d];
        calc_dims_mult(dims_mult[i], max_d, min_bounds, max_bounds, v_eps_levels[i]);
    }
}

void index_points_to_cells_omp(float *v_coords, int ***cell_indexes, int **cell_ns, const float *min_bounds,
        size_t **dims_mult, const float *v_eps_levels, int *v_no_of_cells, int max_levels, int max_d, int n,
        int n_threads) {
    std::vector<int> vec_begin_indexes;
    std::vector<std::pair<size_t, int>> vec_unique_count;
    std::vector<std::pair<size_t, int>> vec_index_maps_t[n_threads];
    for (int t = 0; t < n_threads; t++) {
        vec_index_maps_t[t].reserve((n/n_threads)+1);
    }
    vec_begin_indexes.reserve(n);
    int no_of_cells;
    for (int l = 0; l < max_levels; l++) {
        if (l == 0) {
            no_of_cells = n;
        } else {
            no_of_cells = v_no_of_cells[l - 1];
        }
        #pragma omp parallel for
        for (int i = 0; i < no_of_cells; i++) {
            int t_id = omp_get_thread_num();
            int p_index = traverse_and_get_cell_index(cell_indexes, l, i);
            size_t cell_index = get_cell_index(&v_coords[p_index * max_d], min_bounds, dims_mult[l], max_d,
                    v_eps_levels[l]);
            vec_index_maps_t[t_id].emplace_back(cell_index, i);
        }
        #pragma omp parallel for
        for (int t = 0; t < n_threads; t++) {
            std::sort(vec_index_maps_t[t].begin(), vec_index_maps_t[t].end());
        }

        std::vector<int> indexes, store;
        indexes.reserve(n_threads);
        store.reserve(n_threads);
        for (int i = 0; i < n_threads; i++) {
            indexes.push_back(i);
        }
        while (indexes.size() > 1) {
            for (int i = 0; i < indexes.size(); i += 2) {
                store.push_back(indexes[i]);
            }
            #pragma omp parallel for
            for (int i = 0; i < indexes.size(); i += 2) {
                int t1 = indexes[i];
                int t2 = indexes[i+1];
                int i1 = 0;
                int i2 = 0;
                int size1 = vec_index_maps_t[t1].size();
                int size2 = vec_index_maps_t[t2].size();
                std::vector<std::pair<size_t, int>> vec_new;
                vec_new.reserve(size1 > size2? size1 : size2);
                while (i1 < size1 || i2 < size2) {
                    if (i1 == size1) {
                        vec_new.push_back(vec_index_maps_t[t2][i2]);
                        ++i2;
                    } else if (i2 == size2) {
                        vec_new.push_back(vec_index_maps_t[t1][i1]);
                        ++i1;
                    } else {
                        if (vec_index_maps_t[t1][i1].first == vec_index_maps_t[t2][i2].first) {
                            vec_new.push_back(vec_index_maps_t[t1][i1]);
                            vec_new.push_back(vec_index_maps_t[t2][i2]);
                            ++i1;
                            ++i2;
                        } else if (vec_index_maps_t[t1][i1].first < vec_index_maps_t[t2][i2].first) {
                            vec_new.push_back(vec_index_maps_t[t1][i1]);
                            ++i1;
                        } else { //vec_index_maps_t[t1][i1].first > vec_index_maps_t[t2][i2].first
                            vec_new.push_back(vec_index_maps_t[t2][i2]);
                            ++i2;
                        }
                    }
                }
                vec_index_maps_t[t1].clear();
                std::vector<std::pair<size_t, int>>().swap( vec_index_maps_t[t1] );
                vec_index_maps_t[t2].clear();
                std::vector<std::pair<size_t, int>>().swap( vec_index_maps_t[t2] );
                vec_index_maps_t[t1] = vec_new;
            }
            indexes = store;
            store.clear();
        }
        auto vec_index_map = vec_index_maps_t[0];
        vec_begin_indexes.push_back(0);
        int cnt = 1;
        size_t last_index = vec_index_map[0].first;
        for (int i = 1; i < vec_index_map.size(); i++) {
            if (vec_index_map[i].first != last_index) {
                vec_begin_indexes.push_back(i);
                vec_unique_count.emplace_back(last_index, cnt);
                last_index = vec_index_map[i].first;
                cnt = 1;
            } else {
                ++cnt;
            }
        }
        vec_unique_count.emplace_back(last_index, cnt);
        v_no_of_cells[l] = vec_unique_count.size();
        cell_indexes[l] = new int *[v_no_of_cells[l]];
        cell_ns[l] = new int[v_no_of_cells[l]];

        #pragma omp parallel for
        for (int i = 0; i < v_no_of_cells[l]; i++) {
            int begin = vec_begin_indexes[i];
            int end = (i == (v_no_of_cells[l] - 1)) ? no_of_cells : vec_begin_indexes[i + 1];
            cell_ns[l][i] = end - begin;
        }
        for (int i = 0; i < v_no_of_cells[l]; i++) {
            cell_indexes[l][i] = new int[cell_ns[l][i]];
        }
        #pragma omp parallel for
        for (int i = 0; i < v_no_of_cells[l]; i++) {
            int begin = vec_begin_indexes[i];
            int end = (i == (v_no_of_cells[l] - 1))? no_of_cells : vec_begin_indexes[i+1];
            std::transform(&vec_index_map[begin], &vec_index_map[end], &cell_indexes[l][i][0],
                           [](const std::pair<size_t, int> &p) {
                               return p.second;
                           });
        }
        for (int t = 0; t < n_threads; t++) {
            vec_index_maps_t[t].clear();
        }
        vec_unique_count.clear();
        vec_begin_indexes.clear();
        vec_index_map.clear();
    }
    for (int t = 0; t < n_threads; t++) {
        vec_index_maps_t[t].clear();
        std::vector<std::pair<size_t, int>>().swap( vec_index_maps_t[t] );
    }
    vec_begin_indexes.clear();
    std::vector<int>().swap( vec_begin_indexes );
    vec_unique_count.clear();
    std::vector<std::pair<size_t, int>>().swap( vec_unique_count );
}

void calculate_cell_boundaries_omp(float *v_coords, int ***cell_indexes, int **cell_ns, float **cell_dims_min,
        float **cell_dims_max, const int *v_no_of_cells, const int max_levels, const int max_d) {
    float *coord_min, *coord_max;
    for (int l = 0; l < max_levels; l++) {
        cell_dims_min[l] = new float[v_no_of_cells[l]*max_d];
        cell_dims_max[l] = new float[v_no_of_cells[l]*max_d];
    }
    for (int l = 0; l < max_levels; l++) {
        #pragma omp parallel for private(coord_min, coord_max)
        for (int i = 0; i < v_no_of_cells[l]; i++) {
            if (l == 0) {
                coord_min = &v_coords[cell_indexes[l][i][0]*max_d];
                coord_max = &v_coords[cell_indexes[l][i][0]*max_d];
            } else {
                coord_min = &cell_dims_min[l-1][cell_indexes[l][i][0]*max_d];
                coord_max = &cell_dims_max[l-1][cell_indexes[l][i][0]*max_d];
            }
            std::copy(coord_min, coord_min + max_d, &cell_dims_min[l][i*max_d]);
            std::copy(coord_max, coord_max + max_d, &cell_dims_max[l][i*max_d]);
            for (int j = 1; j < cell_ns[l][i]; j++) {
                if (l == 0) {
                    coord_min = &v_coords[cell_indexes[l][i][j]*max_d];
                    coord_max = &v_coords[cell_indexes[l][i][j]*max_d];
                } else {
                    coord_min = &cell_dims_min[l-1][cell_indexes[l][i][j]*max_d];
                    coord_max = &cell_dims_max[l-1][cell_indexes[l][i][j]*max_d];
                }
                for (int d = 0; d < max_d; d++) {
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

void process_cell_tree_omp(struct_label **ps_origin, float *v_coords, int ***cell_indexes, int ** cell_ns,
        float **cell_dims_min,float **cell_dims_max, const int *v_no_of_cells, int *v_point_nps, bool *is_core,
        bool *is_border_cell, int **s_c1_indexes, int **s_c2_indexes, int **s_levels, int n_threads, int max_levels,
        int max_d, float e, float e2, int m) {
    int max_points_in_cell = 0;
    auto *v_cell_nps = new int[v_no_of_cells[0]];
    auto **range_table = new bool*[n_threads];

    bool *cell_has_cores = new bool[v_no_of_cells[0]];
    std::fill(cell_has_cores, cell_has_cores + v_no_of_cells[0], false);

    #pragma omp parallel for reduction(max: max_points_in_cell)
    for (int i = 0; i < v_no_of_cells[0]; i++) {
        v_cell_nps[i] = cell_ns[0][i];
        if (v_cell_nps[i] > max_points_in_cell) {
            max_points_in_cell = v_cell_nps[i];
        }
        if (v_cell_nps[i] >= m) {
            cell_has_cores[i] = true;
            ps_origin[cell_indexes[0][i][0]]->label = i;
            for (int j = 0; j < v_cell_nps[i]; j++) {
                is_core[cell_indexes[0][i][j]] = true;
                if (j > 0) {
                    ps_origin[cell_indexes[0][i][j]]->label_p = ps_origin[cell_indexes[0][i][0]];
                }
            }
        }
    }
    for (int i = 0; i < n_threads; i++) {
        range_table[i] = new bool[max_points_in_cell*std::min(max_points_in_cell, m)];
    }
    for (int level = 1; level < max_levels; level++) {
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < v_no_of_cells[level]; i++) {
            int t_id = omp_get_thread_num();
            int s_index = 0;
            for (int j = 0; j < cell_ns[level][i]; j++) {
                for (int k = j+1; k < cell_ns[level][i]; k++) {
                    s_levels[t_id][s_index] = level-1;
                    s_c1_indexes[t_id][s_index] = cell_indexes[level][i][j];
                    s_c2_indexes[t_id][s_index] = cell_indexes[level][i][k];
                    ++s_index;
                }
            }
            while (s_index > 0) {
                --s_index;
                int l = s_levels[t_id][s_index];
                int c1 = s_c1_indexes[t_id][s_index];
                int c2 = s_c2_indexes[t_id][s_index];
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
                        for (int j = 0; j < cell_ns[l][c1]; j++) {
                            for (int k = 0; k < cell_ns[l][c2]; k++) {
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
    for (int level = 1; level < max_levels; level++) {
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < v_no_of_cells[level]; i++) {
            int t_id = omp_get_thread_num();
            int s_index = 0;
            for (int j = 0; j < cell_ns[level][i]; j++) {
                for (int k = j+1; k < cell_ns[level][i]; k++) {
                    s_levels[t_id][s_index] = level-1;
                    s_c1_indexes[t_id][s_index] = cell_indexes[level][i][j];
                    s_c2_indexes[t_id][s_index] = cell_indexes[level][i][k];
                    ++s_index;
                }
            }
            while (s_index > 0) {
                --s_index;
                int l = s_levels[t_id][s_index];
                int c1 = s_c1_indexes[t_id][s_index];
                int c2 = s_c2_indexes[t_id][s_index];
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
                        for (int j = 0; j < cell_ns[l][c1]; j++) {
                            for (int k = 0; k < cell_ns[l][c2]; k++) {
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

void detect_border_cells(int ***cell_indexes, int **cell_ns, float **cell_dims_min, float **cell_dims_max,
        bool *border_cells, int *v_no_of_cells, int **s_c1_indexes, int **s_c2_indexes, int **s_levels,
        const int max_levels, const int max_d, const int m, const float e) {
    std::fill(border_cells, border_cells + v_no_of_cells[0], false);

    int *v_cell_nps = new int[v_no_of_cells[0]];
    std::copy(cell_ns[0], cell_ns[0] + v_no_of_cells[0], v_cell_nps);
    for (int level = 1; level < max_levels; level++) {
        #pragma omp parallel for
        for (int i = 0; i < v_no_of_cells[level]; i++) {
            int t_id = omp_get_thread_num();
            int s_index = 0;
            for (int j = 0; j < cell_ns[level][i]; j++) {
                for (int k = j + 1; k < cell_ns[level][i]; k++) {
                    s_levels[t_id][s_index] = level - 1;
                    s_c1_indexes[t_id][s_index] = cell_indexes[level][i][j];
                    s_c2_indexes[t_id][s_index] = cell_indexes[level][i][k];
                    ++s_index;
                }
            }
            while (s_index > 0) {
                --s_index;
                int l = s_levels[t_id][s_index];
                int c1 = s_c1_indexes[t_id][s_index];
                int c2 = s_c2_indexes[t_id][s_index];
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
                        for (int j = 0; j < cell_ns[l][c1]; j++) {
                            for (int k = 0; k < cell_ns[l][c2]; k++) {
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
    int detected_border_points = 0;
    #pragma omp parallel for
    for (int i = 0; i < v_no_of_cells[0]; i++) {
        if (v_cell_nps[i] < m) {
            border_cells[i] = true;
            #pragma omp atomic
            ++detected_border_points;
        }
    }
    delete [] v_cell_nps;
    std::cout << "detected border points: " << detected_border_points << std::endl;
}

void ndbscan(struct_label **p_labels, float *v_coords, const int m, const float e, const int n, const int max_d,
        bool *is_core, int n_threads) {
    float min_bounds[max_d];
    float max_bounds[max_d];
    float max_limit = INT32_MIN;
    omp_set_num_threads(n_threads);
    auto t1 = std::chrono::high_resolution_clock::now();
    float e2 = e * e;
    // TODO
    // BREMEN
//    float e_inner = (e / sqrtf(2)) - 1.1f;

    float e_inner = (e / (1.2f*sqrtf(2)));
    std::cout << "e_inner: " << e_inner << std::endl;

    calc_bounds(v_coords, n, min_bounds, max_bounds, max_d);
    for (int d = 0; d < max_d; d++) {
        if (max_bounds[d] - min_bounds[d] > max_limit)
            max_limit = max_bounds[d] - min_bounds[d];
    }
    int max_levels = static_cast<int>(ceilf(logf(max_limit / e_inner) / logf(2))) + 1;
    auto ***cell_indexes = new int**[max_levels];
    auto **cell_ns = new int*[max_levels];
    auto **dims_mult = new size_t*[max_levels];
    auto **cell_dims_min = new float*[max_levels];
    auto **cell_dims_max = new float*[max_levels];
    auto *v_eps_levels = new float[max_levels];
    auto *v_no_of_cells = new int[max_levels];
    auto *v_point_nps = new int[n];
    bool *border_cells = nullptr;
    // stack
    auto** s_levels = new int*[n_threads];
    auto** s_c1_indexes = new int*[n_threads];
    auto** s_c2_indexes = new int*[n_threads];

    std::fill(v_point_nps, v_point_nps + n, 0);
    allocate_resources(v_eps_levels, dims_mult, min_bounds, max_bounds, max_levels, max_d, e_inner);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Memory and init: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
              << " milliseconds\n";

    t1 = std::chrono::high_resolution_clock::now();
    index_points_to_cells_omp(v_coords, cell_indexes, cell_ns, min_bounds, dims_mult, v_eps_levels, v_no_of_cells,
                                max_levels, max_d, n, n_threads);

    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Point indexing: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
              << " milliseconds\n";

    for (int l = 0; l < max_levels; l++) {
        delete [] dims_mult[l];
    }
    delete [] dims_mult;
    delete [] v_eps_levels;
    for (int i = 0; i < n_threads; i++) {
        s_levels[i] = new int[v_no_of_cells[0]*100];
        s_c1_indexes[i] = new int[v_no_of_cells[0]*100];
        s_c2_indexes[i] = new int[v_no_of_cells[0]*100];
    }
    t1 = std::chrono::high_resolution_clock::now();
    calculate_cell_boundaries_omp(v_coords, cell_indexes, cell_ns, cell_dims_min, cell_dims_max, v_no_of_cells,
                                  max_levels, max_d);
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Cell boundaries: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
              << " milliseconds\n";

    t1 = std::chrono::high_resolution_clock::now();
    border_cells = new bool[v_no_of_cells[0]];
    detect_border_cells(cell_indexes, cell_ns, cell_dims_min, cell_dims_max, border_cells, v_no_of_cells,
            s_c1_indexes, s_c2_indexes, s_levels, max_levels, max_d, m, e);
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Border/noise cell detection: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
              << " milliseconds\n";

    t1 = std::chrono::high_resolution_clock::now();
    process_cell_tree_omp(p_labels, v_coords, cell_indexes, cell_ns, cell_dims_min, cell_dims_max, v_no_of_cells,
                            v_point_nps, is_core, border_cells, s_c1_indexes, s_c2_indexes,
                            s_levels, n_threads, max_levels, max_d, e, e2, m);
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Process cell tree: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
              << " milliseconds\n";
}

struct_label **read_input(char *in_file, float *v_points, int n, int max_d) {
    std::ifstream is(in_file);
    std::string line, buf;
    auto **point_labels = new struct_label *[n];
    for (int i = 0; i < n; i++) {
        point_labels[i] = new struct_label();
    }
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
    std::cout << std::endl;
    is.close();
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Read input took: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
              << " milliseconds\n";
    return point_labels;
}

void displayOutput(const bool *is_core, struct_label** ps, int n) {
    int n_cores = 0;
    for (int i = 0; i < n; i++) {
        if (is_core[i])
            ++n_cores;
    }
    std::cout << "Confirmed core count: " << n_cores << " / " << n << std::endl;

    int cnt = 0;
    auto* labels = new bool[n];
    for (int i = 0; i < n; i++) {
        labels[i] = false;
    }
    for (int i = 0; i < n; i++) {
        labels[get_label(ps[i])->label] = true;
    }
    for (int i = 0; i < n; i++) {
        if (labels[i])
            ++cnt;
    }
    std::cout << "Estimated clusters: " << cnt << std::endl;
    int p_noise = 0;
    for (int i = 0; i < n; i++) {
        if (get_label(ps[i])->label == UNASSIGNED)
            p_noise++;
    }
    std::cout << "Noise points: " << p_noise << std::endl;
}

// Erich: 2744 clusters, 1700147 points
// HPDBSCAN 2972 clusters 1463461 noise points, 1234425 core points
// 43-44 sek, 20, 10, 2972 clusters, pt_in_cls 1536539 noise 1463461
void clusterBremenSmall(int m, float e, int n_threads) {
    int d = 3;
    int n = 3000000;
    auto *v_points = new float[n*d];
    struct_label **ps = read_input(const_cast<char *>("../input/bremen_small.csv"), v_points, n, d);
//    struct_label **ps = read_input(const_cast<char *>("/sdv-work/cdeep/ernir/dbscan/input/bremen_small.csv"), n, d);

    auto t1 = std::chrono::high_resolution_clock::now();
    auto *is_core = new bool[n];
    std::fill(is_core, is_core + n, false);
    ndbscan(ps, v_points, m, e, n, d, is_core, n_threads);
    std::cout << std::endl << std::flush;
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "ndbscan "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
              //<< tot_time
              << " milliseconds\n";
    std::cout << std::endl << std::flush;
    displayOutput(is_core, ps, n);
}

void clusterBremen(int m, float e, int n_threads) {
    int d = 3;
    int n = 81398810;
    auto *v_points = new float[n*d];
//    struct_label **ps = read_input(const_cast<char *>("../input/bremen.csv"), v_points, n, d);
//    struct_label **ps = read_input(const_cast<char *>("../input/C_25GN1.txt"), v_points, n, d);
    struct_label **ps = read_input(const_cast<char *>("/sdv-work/cdeep/ernir/dbscan/input/bremen.csv"), v_points, n, d);

    auto t1 = std::chrono::high_resolution_clock::now();
    auto *is_core = new bool[n];
    std::fill(is_core, is_core + n, false);
    ndbscan(ps, v_points, m, e, n, d, is_core, n_threads);
    std::cout << std::endl << std::flush;
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "ndbscan "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
              //<< tot_time
              << " milliseconds\n";
    std::cout << std::endl << std::flush;
    displayOutput(is_core, ps, n);
}

// 3613730 points in clusters, 5061 clusters, 90.621 noise
//Number of clusters: 5059, Total points 3704351 pt_in_cls 3613723 noise 90628
// Erich 5061 clustered points 3613730
// HPDBSCAN 5060 clusters, 3613734 cluster points, 90617 noise, 3587026 cores
// Me 90617
void clusterTwitter(int m, float e, int n_threads) {
    int d = 2;
    int n = 3704351;
    auto *v_points = new float[n*d];
    struct_label **ps = read_input(const_cast<char *>("../input/twitter_small.csv"), v_points, n, d);

    auto t1 = std::chrono::high_resolution_clock::now();
    auto *is_core = new bool[n];
    std::fill(is_core, is_core + n, false);
    ndbscan(ps, v_points, m, e, n, d, is_core, n_threads);
    std::cout << std::endl << std::flush;
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "ndbscan "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
              //<< tot_time
              << " milliseconds\n";
    std::cout << std::endl << std::flush;
    displayOutput(is_core, ps, n);
}

void clusterAloi4x4x4(int m, float e, int n_threads) {
    int d = 65;
    int n = 110250;
//    struct_label **ps = read_input(const_cast<char *>("../input/aloi-hsb-2x2x2.csv"), n, d);
    auto *v_points = new float[n*d];
    struct_label **ps = read_input(const_cast<char *>("../input/aloi-hsb-4x4x4.csv"), v_points, n, d);

    auto t1 = std::chrono::high_resolution_clock::now();
    auto *is_core = new bool[n];
    std::fill(is_core, is_core + n, false);
    ndbscan(ps, v_points, m, e, n, d, is_core, n_threads);
    std::cout << std::endl << std::flush;
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "ndbscan "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
              //<< tot_time
              << " milliseconds\n";
    std::cout << std::endl << std::flush;
    displayOutput(is_core, ps, n);
}

void clusterAloi7x7x7(int m, float e, int n_threads) {
    int d = 344;
    int n = 110250;
//    struct_label **ps = read_input(const_cast<char *>("../input/aloi-hsb-2x2x2.csv"), n, d);
    auto *v_points = new float[n*d];
    struct_label **ps = read_input(const_cast<char *>("../input/aloi-hsb-7x7x7.csv"), v_points, n, d);

    auto t1 = std::chrono::high_resolution_clock::now();
    auto *is_core = new bool[n];
    std::fill(is_core, is_core + n, false);
    ndbscan(ps, v_points, m, e, n, d, is_core, n_threads);
    std::cout << std::endl << std::flush;
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "ndbscan "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
              //<< tot_time
              << " milliseconds\n";
    std::cout << std::endl << std::flush;
    displayOutput(is_core, ps, n);
}

void start_nextdbscan() {

}

void clusterAloi(int m, float e, int n_threads) {
    int max_d = 8;
    int n = 110250;
    auto *v_points = new float[n*max_d];
//    struct_label **ps = read_input(const_cast<char *>("../input/aloi_sample.csv"), v_points, n, max_d);
    struct_label **ps = read_input(const_cast<char *>("../input/aloi-hsb-2x2x2.csv"), v_points, n, max_d);

    auto t1 = std::chrono::high_resolution_clock::now();
    auto *is_core = new bool[n];
    std::fill(is_core, is_core + n, false);
    ndbscan(ps, v_points, m, e, n, max_d, is_core, n_threads);
    std::cout << std::endl << std::flush;
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "NextDBSCAN runtime took: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
              << " milliseconds\n";
    std::cout << std::endl << std::flush;
    displayOutput(is_core, ps, n);
}


int main(int argc, char* const* argv) {
    char option;
    int m = -1;
    float e = -1;
    int n_threads = -1;
    int errors = 0;

    while ((option = getopt(argc, argv, "hm:e:o:t:")) != -1) {
        switch (option) {
            case 'm': {
                ssize_t minPoints = std::stoll(optarg);
                if (minPoints <= 0L) {
//                    parameters.minPoints = 1L;
                    std::cerr << "minPoints needs to be a positive integer number, but was " << optarg << std::endl;
                    ++errors;
                }
                else
                {
                    m = (size_t) minPoints;
                }
                break;
            }
            case 'e':
            {
                float epsilon = std::stof(optarg);
                if (epsilon <= 0.0f)
                {
//                    parameters.epsilon = 1.0f;
                    std::cerr << "epsilon needs to be a positive floating struct_label number, but was " << optarg << std::endl;
                    ++errors;
                }
                else
                {
                    e = epsilon;
                }
                break;
            }
            case 't':
            {
                ssize_t threads = std::stoll(optarg);
                if (threads <= 0L)
                {
//                    parameters.threads = 1L;
                    std::cerr << "thread count needs to be a positive integer number, but was " << optarg << std::endl;
                    ++errors;
                }
                else
                {
                    n_threads = (size_t) threads;
                }
                break;
            }
        }
    }
    if (errors)
    {
        std::cerr << "INPUT ERROR" << std::endl;
//        usage(argv[0]);
        std::exit(EXIT_FAILURE);
    }
    if (n_threads > 1 && n_threads % 2 == 1) {
        std::cerr << "The number of threads must be 1, or an even number and a multiple of 2." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::cout << "Starting NextDBSCAN with m: " << m << ", e: " << e << ", t: " << n_threads << std::endl;
//    clusterAloi(20, 0.01, 8);
//    clusterAloi7x7x7(20, 0.02, 8);
//    clusterAloi4x4x4(20, 0.01, 1);
    clusterBremen(m, e, n_threads);
//    clusterBremen(35, 25, 8);
//    clusterBremenSmall(30, 20, 8);
}