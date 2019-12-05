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

#include <cstdint>
#include <iostream>
#include <cassert>
#include <chrono>
#include <numeric>
#include <omp.h>
#include "nc_tree.h"
#ifdef CUDA_ON
#include "nextdbscan_cuda.h"
#endif
#ifndef CUDA_ON
#include "nextdbscan_omp.h"
#endif
#include "next_util.h"

struct cell_meta {
    uint l, c;

    cell_meta(uint l, uint c) : l(l), c(c) {}
};

/*
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
 */


void nc_tree::calc_bounds(float *min_bounds, float *max_bounds) noexcept {
    for (uint d = 0; d < n_dim; d++) {
        min_bounds[d] = INT32_MAX;
        max_bounds[d] = INT32_MIN;
    }
    #pragma omp parallel for reduction(max:max_bounds[:n_dim]) reduction(min:min_bounds[:n_dim])
    for (uint i = 0; i < n_coords; i++) {
        size_t index = i * n_dim;
        for (uint d = 0; d < n_dim; d++) {
            if (v_coords[index + d] > max_bounds[d]) {
                max_bounds[d] = v_coords[index + d];
            }
            if (v_coords[index + d] < min_bounds[d]) {
                min_bounds[d] = v_coords[index + d];
            }
        }
    }
}

void nc_tree::calc_dims_mult(ull *dims_mult, const uint max_d, s_vec<float> &min_bounds,
        s_vec<float> &max_bounds, const float e_inner) noexcept {
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

uint nc_tree::determine_data_boundaries() noexcept {
    float max_limit = INT32_MIN;
    calc_bounds(&v_min_bounds[0], &v_max_bounds[0]);
#ifdef MPI_ON
        auto v_global_min_bounds = std::make_unique<float[]>(max_d);
        auto v_global_max_bounds = std::make_unique<float[]>(max_d);
        MPI_Allreduce(&v_min_bounds[0], &v_global_min_bounds[0], max_d, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&v_max_bounds[0], &v_global_max_bounds[0], max_d, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
        std::copy(&v_global_min_bounds[0], &v_global_min_bounds[max_d], &v_min_bounds[0]);
        std::copy(&v_global_max_bounds[0], &v_global_max_bounds[max_d], &v_max_bounds[0]);
#endif
    #pragma omp parallel for reduction(max: max_limit)
    for (uint d = 0; d < n_dim; d++) {
        if (v_max_bounds[d] - v_min_bounds[d] > max_limit)
            max_limit = v_max_bounds[d] - v_min_bounds[d];
    }
    return static_cast<unsigned int>(ceilf(logf(max_limit / e_inner) / logf(2))) + 1;
}

/*
 * ull cell_index = 0;
    for (uint d = 0; d < max_d; d++) {
        cell_index += (ull)((dv[d] - mv[d]) / size) * dm[d];
    }
    return cell_index;
 */

void nc_tree::build_tree(uint n_threads) noexcept {
    s_vec<float> v_eps_levels(n_level);
    s_vec<ull> v_dims_mult(n_level * n_dim);
    #pragma omp parallel for
    for (uint l = 0; l < n_level; l++) {
        v_eps_levels[l] = (e_inner * pow(2, l));
        calc_dims_mult(&v_dims_mult[l * n_dim], n_dim, v_min_bounds, v_max_bounds, v_eps_levels[l]);
    }
#ifdef CUDA_ON
    nextdbscan_cuda::index_points(v_coords, v_eps_levels, v_dims_mult, v_min_bounds, vv_index_map, vv_cell_begin,
                vv_cell_ns, vv_min_cell_dim, vv_max_cell_dim, max_d, n_threads, max_levels, size);
#endif
#ifndef CUDA_ON
    nextdbscan_omp::index_points(v_coords, v_eps_levels, v_dims_mult, v_min_bounds, vv_index_map,
            vv_cell_begin,vv_cell_ns, vv_min_cell_dim, vv_max_cell_dim, n_dim, n_threads, n_level,
            n_coords);
#endif
}

/*
void nc_tree::build_tree() noexcept {
    uint l = 0;
    auto cube_dim = (float)(e_inner * pow(2, l));

    s_vec <uint> v_dim_index(n_coords);
    s_vec<uint> v_p_index(n_coords);
    s_vec<uint> v_dim_cnt(n_coords);
    s_vec<uint> v_dim_offset(n_coords);
//    s_vec<uint> v_dim_tmp_cnt(n_coords);
//    s_vec<uint> v_dim_cnt_trimmed;
    std::iota(v_p_index.begin(), v_p_index.end(), 0);
    std::cout << "Iota check: " << v_p_index[0] << " : " << v_p_index[1] << " : " << v_p_index[2] << std::endl;
    s_vec<int> v_p_index_tmp(n_coords);
    s_vec<std::pair<uint, uint>> v_cell_begin;
    s_vec<std::pair<uint, uint>> v_cell_begin_tmp;
    v_cell_begin.emplace_back(0, n_coords);
    // TODO

    for (uint d = 0; d < n_dim; ++d) {
        uint max_dim_cells = ((v_max_bounds[d] - v_min_bounds[d]) / cube_dim) + 1;
        std::cout << "d: " << d << " max cells: " << max_dim_cells << std::endl;
        // First index all the n_coords values
        for (uint i = 0; i < n_coords; ++i) {
            assert(v_p_index[i] < n_coords);
            v_dim_index[i] = ((v_coords[v_p_index[i] * n_dim + d] - v_min_bounds[d]) / cube_dim);
        }
        std::fill(v_p_index_tmp.begin(), v_p_index_tmp.end(), -1);
        for (uint c = 0; c < v_cell_begin.size(); ++c) {
            uint begin = v_cell_begin[c].first;
            uint end = v_cell_begin[c].second;
            if (end-begin < 2) {
                v_p_index_tmp[begin] = v_p_index[begin];
                v_cell_begin_tmp.emplace_back(begin, begin+1);
                continue;
            }
            assert(end > begin);
//            v_dim_cnt.resize(max_dim_cells);
//            v_dim_offset.resize(max_dim_cells);
//            v_dim_tmp_cnt.resize(max_dim_cells);
            std::fill(v_dim_cnt.begin(), std::next(v_dim_cnt.begin(), max_dim_cells), 0);
            std::fill(v_dim_offset.begin(), std::next(v_dim_offset.begin(), max_dim_cells), 0);
//            std::fill(v_dim_tmp_cnt.begin(), std::next(v_dim_tmp_cnt.begin(), max_dim_cells), 0);
            // Count for parts from last dim
            for (uint i = begin; i < end; ++i) {
                assert(v_dim_index[i] < max_dim_cells);
                ++v_dim_cnt[v_dim_index[i]];
            }
//            uint sum = 0;
//            for (uint i = begin; i < end; ++i) {
//                sum += v_dim_cnt[i];
//            }
            uint offset = begin;
            for (uint i = 0; i < max_dim_cells; ++i) {
                if (v_dim_cnt[i] > 0) {
                    v_dim_offset[i] = offset;
                    offset += v_dim_cnt[i];
                    v_cell_begin_tmp.emplace_back(v_dim_offset[i], offset);
                }
            }
            // sort
            for (uint i = begin; i < end; ++i) {
                assert(v_dim_cnt[v_dim_index[i]] > 0);
                v_p_index_tmp[v_dim_offset[v_dim_index[i]]++] = v_p_index[i];
//                v_p_index_tmp[v_dim_offset[v_dim_index[i]]+v_dim_tmp_cnt[v_dim_index[i]]] = v_p_index[i];
//                assert(v_dim_tmp_cnt[v_dim_index[i]] < v_dim_cnt[v_dim_index[i]]);
//                ++v_dim_tmp_cnt[v_dim_index[i]];;
            }
        }
//         check
//        for (uint i = 0; i < v_p_index_tmp.size(); ++i) {
//            assert(v_p_index_tmp[i] != -1);
//        }
        v_cell_begin.clear();
        v_cell_begin.assign(v_cell_begin_tmp.begin(), v_cell_begin_tmp.end());
        v_cell_begin_tmp.clear();
//        v_p_index.clear();
//        v_p_index.assign(v_p_index_tmp.begin(), v_p_index_tmp.end());
        std::copy(v_p_index_tmp.begin(), v_p_index_tmp.end(), v_p_index.begin());
//        assert((uint)v_p_index_tmp[10] == v_p_index[10]);
        std::cout << "d: " << d << " cell no: " << v_cell_begin.size() << std::endl;
    }
    std::cout << "vec begin: ";
    for (uint i = 20; i < 40; ++i) {
        std::cout << v_cell_begin[i].first << " ";
    }
    std::cout << std::endl;
    print_array("vec index: ", &v_p_index[20], 20);

//    for (uint d = 1; d < 3; ++d) {
//        auto start_timestamp = std::chrono::high_resolution_clock::now();
//        std::fill(v_dim_cnt.begin(), v_dim_cnt.end(), 0);
//        v_dim_cnt.resize(v_dim_unique_size[d-1] * v_dim_unique_size[d]);
//        auto end_timestamp = std::chrono::high_resolution_clock::now();
//        std::cout << "Memory init: "
//                << std::chrono::duration_cast<std::chrono::milliseconds>(end_timestamp - start_timestamp).count()
//                << " milliseconds\n";
//        std::cout << "dim_cnt size: " << v_dim_cnt.size() << std::endl;
//        start_timestamp = std::chrono::high_resolution_clock::now();
//        uint offset = v_dim_unique_size[d-1];
//        for (uint i = 0; i < n_coords; ++i) {
//            ++v_dim_cnt[(v_dim_index[d][i] * offset) + v_dim_index[d-1][i]];
//        }
//        uint index_cnt = 0;
//        for (uint i = 0; i < v_dim_cnt.size(); ++i) {
//            if (v_dim_cnt[i] > 0) {
//                v_dim_cnt_trimmed.push_back(v_dim_cnt[i]);
//                v_dim_cnt[i] = index_cnt++;
//            }
//        }
//        for (uint i = 0; i < n_coords; ++i) {
//            v_dim_index[d][i] = v_dim_cnt[(v_dim_index[d][i] * offset) + v_dim_index[d - 1][i]];
//        }
//        v_dim_unique_size[d] = v_dim_cnt_trimmed.size();
//        end_timestamp = std::chrono::high_resolution_clock::now();
//        std::cout << "counting work: "
//                << std::chrono::duration_cast<std::chrono::milliseconds>(end_timestamp - start_timestamp).count()
//                << " milliseconds\n";
//        std::cout << "Updated used cells for l: " << l << " and dim: " << d << " are " << v_dim_unique_size[d]
//                  << std::endl;
//    }

}
*/

void nc_tree::init() noexcept {
    v_min_bounds.resize(n_dim);
    v_max_bounds.resize(n_dim);
    n_level = determine_data_boundaries();
    vv_index_map.resize(n_level);
    vv_cell_begin.resize(n_level);
    vv_cell_ns.resize(n_level);
    vv_min_cell_dim.resize(n_level);
    vv_max_cell_dim.resize(n_level);
}

void nc_tree::cache_bounding_box_queries() {
    cache_tasks();
    is_cached = true;
}

void nc_tree::cache_tasks() noexcept {
    /*
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
     */
}
