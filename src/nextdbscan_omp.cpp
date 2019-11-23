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

void determine_index_values(s_vec<float> &v_coords,
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

uint index_level_and_get_cells(s_vec<float> &v_coords,
        s_vec<float> &v_min_bounds,
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

void omp_index_points(s_vec<float> &v_coords,
        s_vec<float> &v_eps_levels,
        s_vec<ull> &v_dims_mult,
        s_vec<float> &v_min_bounds,
        d_vec<uint> &vv_index_map,
        d_vec<uint> &vv_cell_begin,
        d_vec<uint> &vv_cell_ns,
        d_vec<float> &vv_min_cell_dim,
        d_vec<float> &vv_max_cell_dim,
        const uint max_d, const uint n_threads,
        const uint max_levels, uint size) noexcept {
    for (int l = 0; l < max_levels; ++l) {
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
        calculate_level_cell_bounds(&v_coords[0], vv_cell_begin[l], vv_cell_ns[l],
                vv_index_map[l], vv_min_cell_dim, vv_max_cell_dim, max_d, l);
    }
}