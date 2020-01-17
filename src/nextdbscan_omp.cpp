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
#include <chrono>
#include <cassert>
#include "nextdbscan_omp.h"
#include "deep_io.h"
#include "next_util.h"

struct cell_meta {
    uint l, c;

    cell_meta(uint l, uint c) : l(l), c(c) {}
};

struct cell_meta_pair_level {
    uint l, c1, c2;

    cell_meta_pair_level(uint l, uint c1, uint c2) : l(l), c1(c1), c2(c2) {}
};

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
    #pragma omp simd
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

void calc_cell_indexes(const float *v_coords, d_vec<uint> &vv_index_map,
        d_vec<uint> &vv_cell_begin,
        s_vec<float> &v_min_bounds,
        s_vec<uint> &v_index_dims, const int l,
        const uint n_dim, const float eps) {
    #pragma omp parallel for
    for (uint i = 0; i < vv_index_map[l].size(); ++i) {
        uint p_index = i;
        int level_mod = 1;
        while (l - level_mod >= 0) {
            p_index = vv_index_map[l - level_mod][vv_cell_begin[l - level_mod][p_index]];
            ++level_mod;
        }
        uint coord_index = (p_index) * n_dim;
//        assert(i < v_index_map.size());
//        uint point_index = vv_index_map[l][i]*n_dim;
        for (uint d = 0; d < n_dim; ++d) {
            v_index_dims[i*n_dim+d] = (v_coords[coord_index+d] - v_min_bounds[d]) / eps;
        }
    }
}

void select_partition_dims(const float *v_coords,
        d_vec<uint> &vv_index_map,
        d_vec<uint> &vv_cell_begin,
        s_vec<uint> &v_primes,
        s_vec<uint> &v_prime_dims,
        s_vec<float> &v_min_bounds,
        s_vec<float> &v_max_bounds, s_vec<uint> &v_index_dims,
        const uint n_dim, const uint n_points, const uint n_partitions, const float eps_p) {
    s_vec<uint> v_dim_cells(n_dim);
    s_vec<float> v_perfect_cell_score(n_dim);
    for (uint d = 0; d < n_dim; ++d) {
        v_dim_cells[d] = ((v_max_bounds[d] - v_min_bounds[d]) / eps_p) + 1;
        v_perfect_cell_score[d] = (float)n_points / v_dim_cells[d];
    }
    next_util::print_array("dim cells: ", &v_dim_cells[0], v_dim_cells.size());
    next_util::print_array("perfect dim cell score: ", &v_perfect_cell_score[0], v_perfect_cell_score.size());

    calc_cell_indexes(v_coords, vv_index_map, vv_cell_begin, v_min_bounds, v_index_dims, 0, n_dim, eps_p);
    s_vec<int> v_cell_cnt;
    s_vec<double> v_dim_scores(n_dim, 0);
    for (uint d = 0; d < n_dim; ++d) {
        if (v_dim_cells[d] < n_partitions) {
            v_dim_scores[d] = INT32_MAX;
            continue;
        }
        v_cell_cnt.clear();
        v_cell_cnt.resize(v_dim_cells[d], 0);
        for (uint i = 0; i < n_points; ++i) {
            uint point_index = i * n_dim;
            assert(v_index_dims[point_index+d] < v_cell_cnt.size());
            ++v_cell_cnt[v_index_dims[point_index+d]];
        }
        double score = 0;
        double tmp = 0;
        for (uint i = 0; i < v_cell_cnt.size(); ++i) {
            tmp = v_cell_cnt[i] / v_perfect_cell_score[d];
            score += tmp * tmp;
        }
        v_dim_scores[d] = score / v_dim_cells[d];
    }
    next_util::print_array("dim scores: ", &v_dim_scores[0], v_dim_scores.size());


    next_util::get_small_prime_factors(v_primes, n_partitions);
    next_util::print_array("prime factors: ", &v_primes[0], v_primes.size());

    // streamline primes factors if necessary
    while (v_primes.size() > n_dim) {
        v_primes[v_primes.size()-2] *= v_primes[v_primes.size()-1];
        v_primes.resize(v_primes.size()-1);
        std::sort(v_primes.begin(), v_primes.end(), std::greater<>());
    }
    next_util::print_array("prime factors post streamline: ", &v_primes[0], v_primes.size());

    std::iota(v_prime_dims.begin(), v_prime_dims.end(), 0);
    std::sort(v_prime_dims.begin(), v_prime_dims.end(), [&v_dim_scores] (const uint &i1, const uint &i2) -> bool {
        return v_dim_scores[i1] < v_dim_scores[i2];
    });

    v_prime_dims.resize(v_primes.size());
    next_util::print_array("selected dims: ", &v_prime_dims[0], v_primes.size());
}

uint partition_dataset(s_vec<float> &v_min_bounds, s_vec<float> &v_max_bounds,
        s_vec<uint> &v_primes, s_vec<uint> &v_prime_dims, s_vec<uint> &v_index_dims, const uint n_dim,
        const uint n_points, const uint n_partitions, const float eps_p) {

    uint used_partitions = 1;
    for (int p = 0; p < v_primes.size()-1; ++p) {
        used_partitions *= v_primes[p];
    }
    std::vector<uint> v_point_parts(n_partitions);
    std::vector<std::vector<uint>> vv_cell_cnts(used_partitions);
    std::vector<uint> v_partition_point_cnt(used_partitions, 0);

    for (uint i = 0; i < used_partitions; ++i) {
        vv_cell_cnts[i].resize(n_points, 0);
    }
    std::vector<uint> v_splits;
    used_partitions = 1;
    uint last_d = INT32_MAX;
    for (int p = 0; p < v_primes.size(); ++p) {
//        std::cout << "STARTING prime: " << v_primes[p] << std::endl;
        uint n_parts = v_primes[p];
        uint d = v_prime_dims[p];
        uint n_dim_cells = ((v_max_bounds[d] - v_min_bounds[d]) / eps_p) + 1;
        if (n_parts > n_dim_cells) {
            return INT32_MAX;
        }
//        std::cout << "d: " << d << " n_dim_cells: " << n_dim_cells << std::endl;
        assert(n_parts <= n_dim_cells);

        // Count cell points in current dimension
        #pragma omp parallel for
        for (uint i = 0; i < n_points; ++i) {
            uint p_index = i*n_dim;
            assert(p_index < v_index_dims.size());
            assert(v_index_dims[p_index+d] < n_dim_cells);
            uint part_index = 0;
            if (p > 0) {
                part_index = v_index_dims[p_index+last_d];
                assert(part_index < used_partitions);
                #pragma omp atomic
                ++v_partition_point_cnt[part_index];
            } else {
                v_partition_point_cnt[0] = n_points;
            }
            #pragma omp atomic
            ++vv_cell_cnts[part_index][v_index_dims[p_index+d]];
        }
        next_util::print_array("partition points: ", &v_partition_point_cnt[0], used_partitions);
        v_splits.resize(used_partitions*(n_parts-1), 0);
//        std::cout << "split array size: " << v_splits.size() << std::endl;
        // Make splits
        #pragma omp parallel for
        for (uint i = 0; i < used_partitions; ++i) {
            uint split_index = i*(n_parts-1);
            long long perfect_score = v_partition_point_cnt[i] / n_parts;
//            std::cout << "Perfect score: " << perfect_score << " for i: " << i << std::endl;
            long long score = 0;
            long long last_score = INT64_MAX;
            long long sum = 0;
            for (uint c = 0; c < n_dim_cells; ++c) {
                sum += vv_cell_cnts[i][c];
//                std::cout << "sum: " << sum << " where c: " << c << std::endl;
                score = perfect_score - sum;
                if (labs(last_score) < labs(score)) {
                    --c;
//                    std::cout << "set split index " << split_index << " to " << c << std::endl;
                    v_splits[split_index++] = c;
                    assert(split_index-(i*(n_parts-1)) < n_parts);
                    perfect_score += last_score;
                    if (split_index == (i+1)*(n_parts-1)) {
                        c = n_dim_cells;
                    }
                    last_score = INT64_MAX;
                    sum = 0;
                } else {
                    last_score = score;
                }
            }
        }
//        next_util::print_array("splits: ", &v_splits[0], v_splits.size());
        #pragma omp parallel for
        for (uint i = 0; i < n_points; ++i) {
            uint p_index = i*n_dim;
            uint part_index = p == 0? 0 : v_index_dims[p_index+last_d];
            uint split_index = part_index*n_parts;
            uint val = v_index_dims[p_index+d];
            uint partition_id = INT32_MAX;
            assert(p_index < v_index_dims.size());
            assert(v_index_dims[p_index+d] < n_dim_cells);
            for (uint s = 0; s < n_parts-1; ++s) {
                if (val <= v_splits[(part_index*(n_parts-1))+s]) {
                    partition_id = split_index+s;
                    s = n_parts;
                }
            }
            v_index_dims[p_index+d] = partition_id == INT32_MAX? split_index+(n_parts-1) : partition_id;
        }

        // cleanup
        last_d = d;
        std::fill(v_partition_point_cnt.begin(), v_partition_point_cnt.end(), 0);
        for (uint i = 0; i < used_partitions; ++i) {
            std::fill(vv_cell_cnts[i].begin(), vv_cell_cnts[i].end(), 0);
        }
        used_partitions *= n_parts;
    }
    return used_partitions;
}

void partition_coords(const float *v_coords, d_vec<uint> &vv_index_map,
        d_vec<uint> &vv_cell_begin, s_vec<float> &v_min_bounds,
        s_vec<float> &v_max_bounds, std::vector<std::vector<uint>> &vv_part_coord_index,
        const uint n_dim, const uint n_points, const uint n_partitions, const float eps_p) {

    s_vec<uint> v_primes;
    s_vec<uint> v_prime_dims(n_dim);
    s_vec<uint> v_index_dims(n_points*n_dim);
    select_partition_dims(v_coords, vv_index_map, vv_cell_begin, v_primes, v_prime_dims, v_min_bounds,
            v_max_bounds,v_index_dims, n_dim, n_points, n_partitions, eps_p);

    uint used_partitions = partition_dataset(v_min_bounds, v_max_bounds, v_primes, v_prime_dims,
            v_index_dims, n_dim, n_points, n_partitions, eps_p);

    if (used_partitions == INT32_MAX) {
        // Unable to partition, epsilon too big
        vv_part_coord_index[0].resize(n_points);
        std::iota(vv_part_coord_index[0].begin(), vv_part_coord_index[0].end(), 0);
    } else {
        std::vector<uint> v_partition_point_cnt(used_partitions, 0);
        uint last_d = v_prime_dims[v_prime_dims.size() - 1];
        #pragma omp parallel for
        for (uint i = 0; i < n_points; ++i) {
            uint p_index = i * n_dim;
            assert(p_index < v_index_dims.size());
            #pragma omp atomic
            ++v_partition_point_cnt[v_index_dims[p_index + last_d]];
        }
        next_util::print_array("final partition sizes: ", &v_partition_point_cnt[0], used_partitions);
        assert(vv_part_coord_index.size() == v_partition_point_cnt.size());
        for (int i = 0; i < v_partition_point_cnt.size(); ++i) {
            vv_part_coord_index[i].resize(v_partition_point_cnt[i]);
        }
        #pragma omp parallel
        {
            uint tid = omp_get_thread_num();
            uint index = 0;
            for (uint i = 0; i < n_points; ++i) {
                uint p_index = i * n_dim;
                if (v_index_dims[p_index + last_d] == tid) {
                    vv_part_coord_index[tid][index++] = i;
                }
            }
        }
    }
}

uint index_level(const float *v_coords, s_vec<float> &v_min_bounds, s_vec<uint> &v_dim_cells,
        d_vec<uint> &vv_index_map, d_vec<uint> &vv_cell_begin, s_vec<uint> &v_cell_ns,
        const float level_eps,
        const int l, const uint size, const uint n_dim
        /*
        d_vec<uint> &vv_index_map,
        d_vec<uint> &vv_cell_begin, s_vec<uint> &v_cell_ns, std::vector<ull> &v_value_map,
        std::vector<std::vector<uint>> &v_bucket, std::vector<ull> &v_bucket_separator,
        std::vector<ull> &v_bucket_separator_tmp, t_uint_iterator &v_iterator, uint size, int l, uint n_dim,
        uint node_offset, float level_eps, ull *dims_mult, uint n_threads
         */
        ) noexcept {

    // Find epsilon which supports int

    v_dim_cells.resize(size * n_dim);

    #pragma omp parallel for
    for (uint i = 0; i < size; ++i) {
        uint p_index = i;
        int level_mod = 1;
        while (l - level_mod >= 0) {
            p_index = vv_index_map[l - level_mod][vv_cell_begin[l - level_mod][p_index]];
            ++level_mod;
        }
        uint coord_index = (p_index) * n_dim;
        for (uint d = 0; d < n_dim; ++d) {
            v_dim_cells[(i*n_dim) + d] = floorf((v_coords[coord_index + d] - v_min_bounds[d]) / level_eps)+1;
//            assert(v_dim_cells[(i*n_dim) + d] <= v_max_dim_values[d]);
        }
    }
    vv_index_map[l].resize(size);
    std::iota(vv_index_map[l].begin(), vv_index_map[l].end(), 0);
    std::sort(vv_index_map[l].begin(), vv_index_map[l].end(),
            [&size, &n_dim, &v_dim_cells](const uint &i1, const uint &i2) -> bool {
                const uint ci1 = i1 * n_dim;
                const uint ci2 = i2 * n_dim;
                for (uint d = 0; d < n_dim; ++d) {
                    if (v_dim_cells[ci1+d] < v_dim_cells[ci2+d]) {
                        return true;
                    }
                    if (v_dim_cells[ci1+d] > v_dim_cells[ci2+d]) {
                        return false;
                    }
                }
                return false;
            });

    uint n_cells = 1;
    uint ci1 = vv_index_map[l][0] * n_dim;
    for (uint i = 1; i < vv_index_map[l].size(); ++i) {
        uint ci2 = vv_index_map[l][i] * n_dim;
        for (uint d = 0; d < n_dim; ++d) {
            if (v_dim_cells[ci1+d] != v_dim_cells[ci2+d]) {
                ++n_cells;
                ci1 = ci2;
                d = n_dim;
            }
        }
    }
    vv_cell_begin[l].resize(n_cells);
    v_cell_ns.resize(n_cells, 0);
    uint index = 0;
    uint n_cnt = 1;
    vv_cell_begin[l][0] = 0;
    ci1 = vv_index_map[l][0] * n_dim;
    for (uint i = 1; i < vv_index_map[l].size(); ++i) {
        uint ci2 = vv_index_map[l][i] * n_dim;
        bool is_equal = true;
        for (uint d = 0; d < n_dim && is_equal; ++d) {
            if (v_dim_cells[ci1+d] != v_dim_cells[ci2+d]) {
                v_cell_ns[index] = n_cnt;
                ++index;
                vv_cell_begin[l][index] = i;
                n_cnt = 1;
                ci1 = ci2;
                is_equal = false;
            }
        }
        if (is_equal) {
            ++n_cnt;
        }
    }
    v_cell_ns[index] = n_cnt;
    return v_cell_ns.size();
}

uint index_level_and_get_cells(float *v_coords, s_vec<float> &v_min_bounds, d_vec<uint> &vv_index_map,
        d_vec<uint> &vv_cell_begin, s_vec<uint> &v_cell_ns, std::vector<ull> &v_value_map,
        std::vector<std::vector<uint>> &v_bucket, std::vector<ull> &v_bucket_separator,
        std::vector<ull> &v_bucket_separator_tmp, t_uint_iterator &v_iterator, uint size, int l, uint max_d,
        uint node_offset, float level_eps, ull *dims_mult, uint n_threads) noexcept {
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

uint sort_and_count_cells(std::vector<uint> &v_coord_index, std::vector<uint> &v_index_dims,
        std::vector<uint> &v_cell_begin, std::vector<uint> &v_cell_ns, const uint n_dim) {
//    uint tid = omp_get_thread_num();
    uint n_t_cells = 0;
    if (!v_coord_index.empty()) {
        std::sort(v_coord_index.begin(), v_coord_index.end(),
                [&n_dim, &v_index_dims](const uint &i1, const uint &i2) -> bool {
                    const uint ci1 = i1 * n_dim;
                    const uint ci2 = i2 * n_dim;
                    for (uint d = 0; d < n_dim; ++d) {
                        if (v_index_dims[ci1 + d] < v_index_dims[ci2 + d]) {
                            return true;
                        }
                        if (v_index_dims[ci1 + d] > v_index_dims[ci2 + d]) {
                            return false;
                        }
                    }
                    return false;
                });
        n_t_cells = 1;
        uint ci1 = v_coord_index[0] * n_dim;
        for (uint i = 1; i < v_coord_index.size(); ++i) {
            uint ci2 = v_coord_index[i] * n_dim;
            for (uint d = 0; d < n_dim; ++d) {
                if (v_index_dims[ci1 + d] != v_index_dims[ci2 + d]) {
                    ++n_t_cells;
                    ci1 = ci2;
                    d = n_dim;
                }
            }
        }
        uint index = 0;
        uint cnt = 1;
        v_cell_begin.resize(n_t_cells);
        v_cell_ns.resize(n_t_cells);
        v_cell_begin[index] = 0;
        ci1 = v_coord_index[0] * n_dim;
        for (uint i = 1; i < v_coord_index.size(); ++i) {
            uint ci2 = v_coord_index[i] * n_dim;
            bool is_new_cell = false;
            for (uint d = 0; d < n_dim; ++d) {
                if (v_index_dims[ci1 + d] != v_index_dims[ci2 + d]) {
                    is_new_cell = true;
                    ci1 = ci2;
                    d = n_dim;
                }
            }
            if (is_new_cell) {
                v_cell_ns[index] = cnt;
                cnt = 1;
                v_cell_begin[++index] = i;
            } else {
                ++cnt;
            }
        }
//        assert(index < v_cell_ns.size());
        v_cell_ns[index] = cnt;
    }
    return n_t_cells;
}

uint index_level_parallel(const float *v_coords, s_vec<float> &v_min_bounds, s_vec<float> &v_max_bounds,
        d_vec<uint> &vv_part_coord_index,
        d_vec<uint> &vv_part_cell_begin,
        d_vec<uint> &vv_part_cell_ns,
        std::vector<uint> &v_t_n_cells,
        std::vector<uint> &v_t_offsets,
        std::vector<uint> &v_index_dims,
        d_vec<uint> &vv_index_map, d_vec<uint> &vv_cell_begin, d_vec<uint> &vv_cell_ns,
        const int l, const uint size, const uint n_dim, const uint n_threads, const float e_inner, const uint n_parallel_level) {
    vv_index_map[l].resize(size);
    if (l == 0) {
        std::iota(vv_index_map[0].begin(), vv_index_map[0].end(), 0);
        auto start_timestamp = std::chrono::high_resolution_clock::now();
        partition_coords(v_coords, vv_index_map, vv_cell_begin, v_min_bounds, v_max_bounds,
                vv_part_coord_index,n_dim, size, n_threads, e_inner*powf(2, n_parallel_level));
        auto end_timestamp = std::chrono::high_resolution_clock::now();
        std::cout << "Partition Data: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end_timestamp - start_timestamp).count()
                  << " milliseconds\n";
        v_t_offsets[0] = 0;
        for (uint t = 1; t < n_threads; ++t) {
            v_t_offsets[t] = v_t_offsets[t-1] + vv_part_coord_index[t-1].size();
        }
    } else {
        #pragma omp parallel
        {
            uint tid = omp_get_thread_num();
            vv_part_coord_index[tid].resize(v_t_n_cells[tid]);
            std::iota(vv_part_coord_index[tid].begin(), vv_part_coord_index[tid].end(), v_t_offsets[tid]);
        }

    }
    v_index_dims.resize(size*n_dim);
    calc_cell_indexes(v_coords, vv_index_map, vv_cell_begin, v_min_bounds, v_index_dims, l,
            n_dim,e_inner*powf(2, (float)l));
    #pragma omp parallel
    {
        uint tid = omp_get_thread_num();
        v_t_n_cells[tid] = sort_and_count_cells(vv_part_coord_index[tid], v_index_dims,
                vv_part_cell_begin[tid], vv_part_cell_ns[tid], n_dim);
        if (tid > 0) {
            for (uint i = 0; i < vv_part_cell_begin[tid].size(); ++i) {
                vv_part_cell_begin[tid][i] += v_t_offsets[tid];
            }
        }
        std::copy(vv_part_coord_index[tid].begin(), vv_part_coord_index[tid].end(),
                std::next(vv_index_map[l].begin(), v_t_offsets[tid]));
        #pragma omp barrier
        #pragma omp single
        {
            uint n_cells = next_util::sum_array(&v_t_n_cells[0], v_t_n_cells.size());
            vv_cell_ns[l].resize(n_cells, 0);
            vv_cell_begin[l].resize(n_cells);
            v_t_offsets[0] = 0;
            for (uint t = 1; t < n_threads; ++t) {
                v_t_offsets[t] = v_t_offsets[t-1] + v_t_n_cells[t-1];
            }
        }
        std::copy(vv_part_cell_begin[tid].begin(), vv_part_cell_begin[tid].end(),
                std::next(vv_cell_begin[l].begin(), v_t_offsets[tid]));
        std::copy(vv_part_cell_ns[tid].begin(), vv_part_cell_ns[tid].end(),
                std::next(vv_cell_ns[l].begin(), v_t_offsets[tid]));
    }
//    calculate_level_cell_bounds(v_coords, vv_cell_begin[l], vv_cell_ns[l],
//            vv_index_map[l], vv_min_cell_dim, vv_max_cell_dim, n_dim, l);
    return vv_cell_begin[l].size();
}

void nc_tree::index_points(s_vec<float> &v_eps_levels, s_vec<ull> &v_dims_mult) noexcept {
    uint size = n_coords;
    std::vector<uint> v_index_dims;
    std::vector<uint> v_t_n_cells(n_threads);
    std::vector<uint> v_t_offsets(n_threads);
    int n_parallel_level = n_level / 3;
    std::cout << "max level: " << n_level << ", max parallel level: " << n_parallel_level << std::endl;
    std::vector<std::vector<uint>> vv_part_coord_index(n_threads);
    std::vector<std::vector<uint>> vv_part_cell_begin(n_threads);
    std::vector<std::vector<uint>> vv_part_cell_ns(n_threads);

    for (int l = 0; l < n_level; ++l) {
        if (l <= n_parallel_level) {
            size = index_level_parallel(v_coords, v_min_bounds, v_max_bounds, vv_part_coord_index,
                    vv_part_cell_begin, vv_part_cell_ns, v_t_n_cells, v_t_offsets,
                    v_index_dims, vv_index_map, vv_cell_begin, vv_cell_ns, l, size,
                    n_dim, n_threads, e_inner, n_parallel_level);
        } else {
            size = index_level(v_coords, v_min_bounds, v_index_dims, vv_index_map, vv_cell_begin,
                    vv_cell_ns[l], v_eps_levels[l], l, size, n_dim);
        }
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

void nc_tree::infer_types() noexcept {
    uint max_clusters = 0;
    v_is_core.resize(n_coords, UNKNOWN);
    v_point_labels.resize(n_coords, UNASSIGNED);
    #pragma omp parallel for reduction(+: max_clusters)
    for (uint i = 0; i < vv_cell_ns[0].size(); ++i) {
        update_type(vv_index_map[0], vv_cell_ns[0], vv_cell_begin[0],
                v_leaf_cell_np, v_point_np, v_is_core, v_leaf_cell_type, i, m);
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
        if (v_leaf_cell_type[i] == UNKNOWN) {
            v_leaf_cell_type[i] = NO_CORES;
        }
    }
}

void process_pair_stack(s_vec<uint> &v_edges,
        d_vec<uint> &vv_index_map,
        d_vec<uint> &vv_cell_begin,
        d_vec<uint> &vv_cell_ns,
        d_vec<float> &vv_min_cell_dim,
        d_vec<float> &vv_max_cell_dim,
        std::vector<cell_meta_pair_level> &v_stack,
        const uint n_dim, const float e) noexcept {
    while (!v_stack.empty()) {
        uint l = v_stack.back().l;
        uint c1 = v_stack.back().c1;
        uint c2 = v_stack.back().c2;
        v_stack.pop_back();
        uint begin1 = vv_cell_begin[l][c1];
        uint begin2 = vv_cell_begin[l][c2];
        if (l == 0) {
            // CUDA doesn't support emplace_back
            v_edges.push_back(c1);
            v_edges.push_back(c2);
        } else {
            for (uint k1 = 0; k1 < vv_cell_ns[l][c1]; ++k1) {
                uint c1_next = vv_index_map[l][begin1 + k1];
                for (uint k2 = 0; k2 < vv_cell_ns[l][c2]; ++k2) {
                    uint c2_next = vv_index_map[l][begin2 + k2];
                    if (is_in_reach(&vv_min_cell_dim[l - 1][c1_next * n_dim],
                            &vv_max_cell_dim[l - 1][c1_next * n_dim],
                            &vv_min_cell_dim[l - 1][c2_next * n_dim],
                            &vv_max_cell_dim[l - 1][c2_next * n_dim], n_dim, e)) {
                        v_stack.emplace_back(l - 1, c1_next, c2_next);
                    }
                }
            }
        }
    }
}

void nc_tree::collect_proximity_queries() noexcept {
    std::vector<cell_meta> v_tasks;
    std::vector<std::vector<cell_meta_pair_level>> vv_stack(n_threads);
    d_vec<uint> vv_edges(n_threads);
    uint task_size = 0;
    #pragma omp parallel for reduction(+:task_size)
    for (uint l = 1; l < n_level; ++l) {
        task_size += vv_cell_begin[l].size();
    }
    v_tasks.reserve(task_size);
    for (uint l = 1; l < n_level; ++l) {
        for (uint i = 0; i < vv_cell_begin[l].size(); ++i) {
            v_tasks.emplace_back(l,i);
        }
    }

//    uint test;
//    for (uint l = 0; l < n_level; ++l) {
//        test = 0;
//        for (uint i = 0; i < vv_cell_ns[l].size(); ++i) {
//            test += (vv_cell_ns[l][i] * (vv_cell_ns[l][i] - 1)) / 2;
//        }
//        std::cout << "l: " << l << " " << test << std::endl;
//    }

//    auto start_timestamp_1 = std::chrono::high_resolution_clock::now();
    uint total_edges = 0;
    #pragma omp parallel reduction(+: total_edges)
    {
        uint tid = omp_get_thread_num();
        vv_edges[tid].reserve(v_tasks.size() / n_threads);
    #pragma omp for schedule(dynamic)
        for (uint i = 0; i < v_tasks.size(); ++i) {
            uint l = v_tasks[i].l;
            uint c = v_tasks[i].c;
            uint begin = vv_cell_begin[l][c];
            for (uint c1 = 0; c1 < vv_cell_ns[l][c]; ++c1) {
                uint c1_index = vv_index_map[l][begin + c1];
                for (uint c2 = c1 + 1; c2 < vv_cell_ns[l][c]; ++c2) {
                    uint c2_index = vv_index_map[l][begin + c2];
                    if (is_in_reach(&vv_min_cell_dim[l - 1][c1_index * n_dim],
                            &vv_max_cell_dim[l - 1][c1_index * n_dim],
                            &vv_min_cell_dim[l - 1][c2_index * n_dim],
                            &vv_max_cell_dim[l - 1][c2_index * n_dim], n_dim, e)) {
                        vv_stack[tid].emplace_back(l - 1, c1_index, c2_index);
                        process_pair_stack(vv_edges[tid], vv_index_map, vv_cell_begin, vv_cell_ns,
                                vv_min_cell_dim, vv_max_cell_dim, vv_stack[tid], n_dim, e);
                    }
                }
            }
        }
        total_edges += vv_edges[tid].size();
    } // end parallel region
    for (uint t = 0; t < n_threads; ++t) {
        vv_stack[t].clear();
        vv_stack[t].shrink_to_fit();
    }
    for (uint l = 1; l < n_level; ++l) {
        vv_index_map[l].clear();
        vv_index_map[l].shrink_to_fit();
        vv_cell_ns[l].clear();
        vv_cell_ns[l].shrink_to_fit();
        vv_cell_begin[l].clear();
        vv_cell_begin[l].shrink_to_fit();
        vv_min_cell_dim[l].clear();
        vv_min_cell_dim[l].shrink_to_fit();
        vv_max_cell_dim[l].clear();
        vv_max_cell_dim[l].shrink_to_fit();
    }
//    auto end_timestamp_1 = std::chrono::high_resolution_clock::now();
//    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end_timestamp_1 - start_timestamp_1).count()
//              << " milliseconds\n";
//    auto start_timestamp_2 = std::chrono::high_resolution_clock::now();
    v_edges.reserve(total_edges);
    for (uint t = 0; t < vv_edges.size(); ++t) {
        v_edges.insert(v_edges.end(), std::make_move_iterator(vv_edges[t].begin()),
                std::make_move_iterator(vv_edges[t].end()));
    }
//    auto end_timestamp_2 = std::chrono::high_resolution_clock::now();
//    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end_timestamp_2 - start_timestamp_2).count()
//              << " milliseconds\n";
}

void nc_tree::process_proximity_queries() noexcept {
    v_leaf_cell_np = vv_cell_ns[0];
    v_leaf_cell_type.resize(v_leaf_cell_np.size(), UNKNOWN);
    v_point_np.resize(n_coords, 0);
    v_edge_conn.resize(v_edges.size()/2, UNKNOWN);

    #pragma omp parallel for
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
    uint max_points_in_cell = 0;
    #pragma omp parallel for reduction(max: max_points_in_cell)
    for (uint i = 0; i < vv_cell_ns[0].size(); ++i) {
        if (v_leaf_cell_np[i] < m) {
            v_leaf_cell_type[i] = NO_CORES;
        } else if (vv_cell_ns[0][i] >= m) {
            v_leaf_cell_type[i] = ALL_CORES;
        }
        if (vv_cell_ns[0][i] > max_points_in_cell) {
            max_points_in_cell = vv_cell_ns[0][i];
        }
    }
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


/*
void _atomic_op(T* address, T value, O op) {
    T previous = __sync_fetch_and_add(address, 0);

    while (op(value, previous)) {
        if  (__sync_bool_compare_and_swap(address, previous, value)) {
            break;
        } else {
            previous = __sync_fetch_and_add(address, 0);
        }
    }
}
 */
//void atomic_min(uint *p_val, uint val) {
//    uint prev = __sync_fetch_and_add(p_val, 0);
//}

bool are_core_connected(const float *v_coords, s_vec<uint> &v_index_map, s_vec<uint> &v_cell_begin,
        s_vec<uint> &v_cell_ns, s_vec<uint8_t> &v_is_core,
        const uint c1, const uint c2, const uint n_dim, const float e2) {
    uint begin1 = v_cell_begin[c1];
    uint begin2 = v_cell_begin[c2];
    for (uint k1 = 0; k1 < v_cell_ns[c1]; ++k1) {
        uint p1 = v_index_map[begin1 + k1];
        if (!v_is_core[p1]) {
            continue;
        }
        for (uint k2 = 0; k2 < v_cell_ns[c2]; ++k2) {
            uint p2 = v_index_map[begin2 + k2];
            if (!v_is_core[p2]) {
                continue;
            }
            if (dist_leq(&v_coords[p1 * n_dim],
                    &v_coords[p2 * n_dim], n_dim, e2)) {
                return true;
            }
        }
    }
    return false;
}

void nc_tree::determine_cell_labels() noexcept {
    std::vector<int> v_local_min_labels(vv_cell_begin[0].size(), ROOT_CLUSTER);

    #pragma omp parallel for
    for (uint i = 0; i < v_local_min_labels.size(); ++i) {
        if (v_leaf_cell_type[i] == NO_CORES)
            v_local_min_labels[i] = UNASSIGNED;
    }
    #pragma omp parallel for schedule(guided)
    for (uint i = 0; i < v_edges.size(); i += 2) {
        int c1 = v_edges[i];
        int c2 = v_edges[i+1];
        int c_lower, c_higher;
        auto conn = v_edge_conn[i/2];
        if (conn == NOT_CONNECTED) {
            continue;
        }
        if (v_leaf_cell_type[c1] != NO_CORES && v_leaf_cell_type[c2] != NO_CORES) {
            // both either AC or SC
            c_lower = (c1 < c2)? c1 : c2;
            c_higher = (c1 < c2)? c2 : c1;
            if (v_local_min_labels[c_higher] <= c_lower) {
                continue;
            }
            if (conn == UNKNOWN || conn == PARTIALLY_CONNECTED) {
                // Find out if smaller than current
                if (are_core_connected(v_coords, vv_index_map[0], vv_cell_begin[0], vv_cell_ns[0],
                        v_is_core, c1, c2, n_dim, e2)) {
                    v_edge_conn[i/2] = CORE_CONNECTED;
                    // TODO atomic min
                    v_local_min_labels[c_higher] = c_lower;
                } else {
                    v_edge_conn[i/2] = NOT_CORE_CONNECTED;
                }
            } else if (conn == FULLY_CONNECTED) {
                // We know they are connected
                // TODO atomic min
                v_local_min_labels[c_higher] = c_lower;
            }
        } else if (conn == FULLY_CONNECTED && (v_leaf_cell_type[c1] != NO_CORES || v_leaf_cell_type[c2] != NO_CORES)
                   && (v_leaf_cell_type[c1] == NO_CORES || v_leaf_cell_type[c2] == NO_CORES)) {
            c_lower = (v_leaf_cell_type[c1] != NO_CORES)? c1 : c2;
            c_higher = (v_leaf_cell_type[c1] != NO_CORES)? c2 : c1;
            if (v_local_min_labels[c_higher] != UNASSIGNED)
                continue;
            // Permitted race condition (DBSCAN border point labels are not deterministic)
            v_local_min_labels[c_higher] = c_lower;
        }
    }

    // flatten
    #pragma omp parallel for schedule(guided)
    for (uint i = 0; i < v_local_min_labels.size(); ++i) {
        if (v_local_min_labels[i] == ROOT_CLUSTER || v_local_min_labels[i] == UNASSIGNED)
            continue;
        uint label = v_local_min_labels[i];
        bool update = false;
        while (v_local_min_labels[label] != ROOT_CLUSTER) {
            label = v_local_min_labels[label];
            update = true;
        }
        if (update)
            v_local_min_labels[i] = label;
    }
    #pragma omp parallel for schedule(guided)
    for (uint i = 0; i < v_edges.size(); i += 2) {
        uint c1 = v_edges[i];
        uint c2 = v_edges[i+1];
        uint label1, label2;
        auto conn = v_edge_conn[i/2];
        if (conn == NOT_CONNECTED) {
            continue;
        }
        if (v_leaf_cell_type[c1] != NO_CORES && v_leaf_cell_type[c2] != NO_CORES) {
            if (conn == UNKNOWN || conn == PARTIALLY_CONNECTED) {
                if (are_core_connected(v_coords, vv_index_map[0], vv_cell_begin[0], vv_cell_ns[0],
                        v_is_core, c1, c2, n_dim, e2)) {
                    v_edge_conn[i/2] = CORE_CONNECTED;
                } else {
                    v_edge_conn[i/2] = NOT_CORE_CONNECTED;
                }
                conn = v_edge_conn[i/2];
            }
            if (conn == NOT_CORE_CONNECTED) {
                continue;
            }
            label1 = c1;
            while (v_local_min_labels[label1] != ROOT_CLUSTER) {
                label1 = v_local_min_labels[label1];
            }
            label2 = c2;
            while (v_local_min_labels[label2] != ROOT_CLUSTER) {
                label2 = v_local_min_labels[label2];
            }
            if (label1 != label2) {
                // TODO atomic min
                if (label1 < label2) {
                    v_local_min_labels[label2] = label1;
                } else {
                    v_local_min_labels[label1] = label2;
                }
            }
        } else if ((v_leaf_cell_type[c1] != NO_CORES || v_leaf_cell_type[c2] != NO_CORES)
                   && (v_leaf_cell_type[c1] == NO_CORES || v_leaf_cell_type[c2] == NO_CORES)) {
            int c_lower = (v_leaf_cell_type[c1] != NO_CORES)? c1 : c2;
            int c_higher = (v_leaf_cell_type[c1] != NO_CORES)? c2 : c1;
            if (v_local_min_labels[c_higher] != UNASSIGNED)
                continue;
            uint begin1 = vv_cell_begin[0][c_lower];
            uint begin2 = vv_cell_begin[0][c_higher];
            for (uint k1 = 0; k1 < vv_cell_ns[0][c_lower]; ++k1) {
                uint p1 = vv_index_map[0][begin1 + k1];
                if (!v_is_core[p1])
                    continue;
                for (uint k2 = 0; k2 < vv_cell_ns[0][c_higher]; ++k2) {
                    uint p2 = vv_index_map[0][begin2 + k2];
                    if (v_point_labels[p2] != UNASSIGNED)
                        continue;
                    if (dist_leq(&v_coords[p1 * n_dim], &v_coords[p2 * n_dim], n_dim, e2)) {
                        v_point_labels[p2] = p1;
                    }
                }
            }
        }
    }
    #pragma omp parallel for
    for (uint i = 0; i < v_local_min_labels.size(); ++i) {
        if (v_local_min_labels[i] == ROOT_CLUSTER || v_local_min_labels[i] == UNASSIGNED)
            continue;
        if (v_leaf_cell_type[i] != NO_CORES) {
            uint begin = vv_cell_begin[0][i];
            int p = vv_index_map[0][begin];
            if (p != v_point_labels[p]) {
                p = v_point_labels[p];
            }
            v_point_labels[p] = v_point_labels[vv_index_map[0][vv_cell_begin[0][v_local_min_labels[i]]]];
        } else {
            uint begin = vv_cell_begin[0][i];
            uint label = v_point_labels[vv_index_map[0][vv_cell_begin[0][v_local_min_labels[i]]]];
            for (uint j = 0; j < vv_cell_ns[0][i]; ++j) {
                v_point_labels[vv_index_map[0][begin+j]] = label;
            }
        }
    }
}
