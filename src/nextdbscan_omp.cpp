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
#include <numeric>
#include <omp.h>
#include <algorithm>
#include <memory>
#include "nextdbscan_omp.h"
#include "deep_io.h"

inline ull get_cell_index(const float *dv, const s_vec<float> &mv, const ull *dm, const uint max_d,
        const float size) noexcept {
    ull cell_index = 0;
    for (uint d = 0; d < max_d; d++) {
        cell_index += (ull)((dv[d] - mv[d]) / size) * dm[d];
    }
    return cell_index;
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

void determine_index_values(const float *v_coords,
        s_vec<float> &v_min_bounds,
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

uint
index_level_and_get_cells(float *v_coords, s_vec<float> &v_min_bounds, d_vec<uint> &vv_index_map,
        d_vec<uint> &vv_cell_begin, s_vec<uint> &v_cell_ns, std::vector<ull> &v_value_map,
        std::vector<std::vector<uint>> &v_bucket, std::vector<ull> &v_bucket_separator,
        std::vector<ull> &v_bucket_separator_tmp, t_uint_iterator &v_iterator, uint size, int l, uint max_d,
        uint node_offset, float level_eps, ull *dims_mult, uint n_threads) noexcept {
//    return 0;
//}
//uint index_level_and_get_cells(s_vec<float> &v_coords,
//        s_vec<float> &v_min_bounds,
//        d_vec<uint> &vv_index_map,
//        d_vec<uint> &vv_cell_begin,
//        s_vec<uint> &v_cell_ns,
//        std::vector<ull> &v_value_map,
//        std::vector<std::vector<uint>> &v_bucket,
//        std::vector<ull> &v_bucket_separator,
//        std::vector<ull> &v_bucket_separator_tmp,
//        t_uint_iterator &v_iterator,
//        const uint size, const int l, const uint max_d, const uint node_offset, const float level_eps,
//        const ull *dims_mult, const uint n_threads) noexcept {
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

void calculate_level_cell_bounds(float *v_coords, s_vec<uint> &v_cell_begins,
        s_vec<uint> &v_cell_ns, s_vec<uint> &v_index_maps,
        std::vector<std::vector<float>> &vv_min_cell_dims,
        std::vector<std::vector<float>> &vv_max_cell_dims,
        const uint max_d, const uint l) noexcept {
    vv_min_cell_dims[l].resize(v_cell_begins.size() * max_d);
    vv_max_cell_dims[l].resize(vv_min_cell_dims[l].size());
    float *coord_min = nullptr, *coord_max = nullptr;

    #pragma omp parallel for private(coord_min, coord_max)
    for (uint i = 0; i < v_cell_begins.size(); i++) {
        uint begin = v_cell_begins[i];
        uint coord_offset = v_index_maps[begin] * max_d;
        if (l == 0) {
            coord_min = &v_coords[coord_offset];
            coord_max = &v_coords[coord_offset];
        } else {
            coord_min = &vv_min_cell_dims[l - 1][coord_offset];
            coord_max = &vv_max_cell_dims[l - 1][coord_offset];
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

void nc_tree::index_points(s_vec<float> &v_eps_levels, s_vec<ull> &v_dims_mult) noexcept {
    uint size = n_coords;
    for (int l = 0; l < n_level; ++l) {
        std::vector<ull> v_value_map;
        std::vector<std::vector<uint>> v_bucket(n_threads);
        std::vector<ull> v_bucket_separator;
        v_bucket_separator.reserve(n_threads);
        std::vector<ull> v_bucket_separator_tmp;
        v_bucket_separator_tmp.reserve(n_threads * n_threads);
        t_uint_iterator v_iterator(n_threads);
        size = index_level_and_get_cells(v_coords, v_min_bounds, vv_index_map, vv_cell_begin,
                vv_cell_ns[l], v_value_map, v_bucket, v_bucket_separator, v_bucket_separator_tmp,
                v_iterator, size, l, n_dim, 0, v_eps_levels[l],
                &v_dims_mult[l * n_dim], n_threads);
        calculate_level_cell_bounds(v_coords, vv_cell_begin[l], vv_cell_ns[l],
                vv_index_map[l], vv_min_cell_dim, vv_max_cell_dim, n_dim, l);
    }
}

uint fill_range_table(const float *v_coords, s_vec<uint> &v_index_map_level,
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

void update_points(s_vec<uint> &v_index_map_level, s_vec<uint> &v_cell_nps,
        s_vec<uint> &v_point_nps, uint *v_range_cnt, const uint size, const uint begin,
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

bool update_cell_pair_nn(s_vec<uint> &v_index_map_level, const uint size1, const uint size2,
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
    return (is_update1 || is_update2);
}

uint8_t process_pair_proximity(const float *v_coords,
        s_vec<uint> &v_index_maps,
        s_vec<uint> &v_point_nps,
        s_vec<uint> &v_cell_ns,
        std::vector<bool> &v_range_table,
        std::vector<uint> &v_range_cnt,
        s_vec<uint> &v_cell_nps,
        const uint max_d, const float e2, const uint m,
        const uint c1, const uint begin1, const uint c2, const uint begin2) noexcept {
    uint8_t are_connected = NOT_CONNECTED;
    uint size1 = v_cell_ns[c1];
    uint size2 = v_cell_ns[c2];
    uint hits = fill_range_table(v_coords, v_index_maps, size1, size2,
            v_range_table, begin1, begin2, max_d, e2);
    if (hits == size1*size2) {
        if (v_cell_nps[c1] < m) {
            #pragma omp atomic
            v_cell_nps[c1] += v_cell_ns[c2];
        }
        if (v_cell_nps[c2] < m) {
            #pragma omp atomic
            v_cell_nps[c2] += v_cell_ns[c1];
        }
        are_connected = FULLY_CONNECTED;
    } else if (hits > 0) {
        if (update_cell_pair_nn(v_index_maps, size1, size2, v_cell_nps, v_point_nps, v_range_table,
                v_range_cnt, c1, begin1, c2, begin2, v_cell_nps[c1] < m,
                v_cell_nps[c2] < m)) {
            are_connected = PARTIALLY_CONNECTED;
        }
    }
    return are_connected;
}

inline void update_to_ac(s_vec<uint> &v_index_maps, s_vec<uint> &v_cell_ns,
        s_vec<uint> &v_cell_begin, s_vec<uint8_t> &is_core, s_vec<uint8_t> &v_types,
        const uint c) noexcept {
    v_types[c] = ALL_CORES;
    uint begin = v_cell_begin[c];
    for (uint j = 0; j < v_cell_ns[c]; ++j) {
        is_core[v_index_maps[begin + j]] = 1;
    }
}

void update_type(s_vec<uint> &v_index_maps, s_vec<uint> &v_cell_ns,
        s_vec<uint> &v_cell_begin, s_vec<uint> &v_cell_nps, s_vec<uint> &v_point_nps,
        s_vec<uint8_t> &is_core, s_vec<uint8_t> &v_types, const uint c, const uint m) noexcept {
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
        v_types[c] = ALL_CORES;
    } else if (some_cores) {
        v_types[c] = SOME_CORES;
    }
}

void nc_tree::infer_types_and_max_clusters() noexcept {
//    std::cout << "CHECKPOINT #1: " << v_leaf_cell_np.size() << " , " << v_leaf_cell_type.size() << std::endl;
    uint max_clusters = 0;
    v_is_core.resize(n_coords, UNKNOWN);
    v_point_labels.resize(n_coords, UNASSIGNED);
//    std::cout << "CHECKPOINT #2" << std::endl;
    #pragma omp parallel for reduction(+: max_clusters)
    for (uint i = 0; i < vv_cell_ns[0].size(); ++i) {
//        std::cout << "LOOP #" << i << std::endl;
        update_type(vv_index_map[0], vv_cell_ns[0], vv_cell_begin[0],
                v_leaf_cell_np, v_point_np, v_is_core, v_leaf_cell_type, i, m);
//        std::cout << "CHECKPOINT #3" << std::endl;
        if (v_leaf_cell_type[i] != UNKNOWN) {
            ++max_clusters;
            uint begin = vv_cell_begin[0][i];
            int core_p = UNASSIGNED;
            for (uint j = 0; j < vv_cell_ns[0][i]; ++j) {
                uint p = vv_index_map[0][begin + j];
                if (core_p != UNASSIGNED) {
                    v_point_labels[p] = core_p;
                } else if (v_is_core[p]) {
                    core_p = p;
                    v_point_labels[core_p] = core_p;
                    for (uint k = 0; k < j; ++k) {
                        p = vv_index_map[0][begin + k];
                        v_point_labels[p] = core_p;
                    }
                }
            }
        }
//        std::cout << "CHECKPOINT #4" << std::endl;
        if (v_leaf_cell_type[i] == UNKNOWN) {
            v_leaf_cell_type[i] = NO_CORES;
        }
    }
}

void nc_tree::process_proximity_queries() noexcept {
    v_leaf_cell_np = vv_cell_ns[0];
    v_leaf_cell_type.resize(v_leaf_cell_np.size(), UNKNOWN);
    v_point_np.resize(n_coords, 0);
    v_edge_conn.resize(v_edges.size()/2, UNKNOWN);

//    uint sum = 0;
//    for (uint i = 0; i < vv_cell_ns[0].size(); ++i) {
//        if (vv_cell_ns[0][i] >= m) {
//            sum += vv_cell_ns[0][i];
//        }
//    }
//    std::cout << "sum: " << sum << std::endl;
    #pragma omp parallel
    {
        #pragma omp for
        for (uint i = 0; i < v_edges.size(); i += 2) {
            uint c1 = v_edges[i];
            uint c2 = v_edges[i+1];
            if (v_leaf_cell_np[c1] < m) {
                #pragma omp atomic
                v_leaf_cell_np[c1] += vv_cell_ns[0][c2];
            }
            if (v_leaf_cell_np[c2] < m) {
                #pragma omp atomic
                v_leaf_cell_np[c2] += vv_cell_ns[0][c1];
            }
        }
        #pragma omp barrier
        #pragma omp for reduction(max: max_points_in_cell)
        for (uint i = 0; i < v_leaf_cell_np.size(); ++i) {
            if (v_leaf_cell_np[i] < m) {
                v_leaf_cell_type[i] = NO_CORES;
            } else if (vv_cell_ns[0][i] >= m) {
                v_leaf_cell_type[i] = ALL_CORES;
            }
            if (vv_cell_ns[0][i] > max_points_in_cell) {
                max_points_in_cell = vv_cell_ns[0][i];
            }
        }
    }
//    std::cout << "All Cores cell cnt: " << cnt_leaf_cells_of_type(ALL_CORES) << std::endl;
//    std::cout << "No Cores cell cnt: " << cnt_leaf_cells_of_type(NO_CORES) << std::endl;
    v_leaf_cell_np = vv_cell_ns[0];
    std::vector<std::vector<bool>> vv_range_table(n_threads);
    std::vector<std::vector<uint>> vv_range_counts(n_threads);
    #pragma omp parallel
    {
        uint tid = omp_get_thread_num();
        vv_range_table[tid].resize(max_points_in_cell * max_points_in_cell);
        vv_range_counts[tid].resize(max_points_in_cell * 2);
        // TODO guided ?
        #pragma omp for schedule(dynamic, 8)
        for (uint i = 0; i < v_edges.size(); i += 2) {
            uint c1 = v_edges[i];
            uint c2 = v_edges[i+1];
            if (v_leaf_cell_np[c1] >= m && v_leaf_cell_np[c2] >= m) {
                continue;
            }
            if (v_leaf_cell_type[c1] == NO_CORES && v_leaf_cell_type[c2] == NO_CORES) {
                v_edge_conn[i/2] = NOT_CONNECTED;
                continue;
            }
            uint begin1 = vv_cell_begin[0][c1];
            uint begin2 = vv_cell_begin[0][c2];
            uint8_t are_connected = process_pair_proximity(v_coords, vv_index_map[0], v_point_np,
                    vv_cell_ns[0], vv_range_table[tid], vv_range_counts[tid], v_leaf_cell_np,
                    n_dim, e2, m, c1, begin1, c2, begin2);
            v_edge_conn[i/2] = are_connected;
        }
    }
}
