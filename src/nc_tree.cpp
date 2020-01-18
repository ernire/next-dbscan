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
#include <omp.h>
#include "nc_tree.h"
#ifdef CUDA_ON
#include "nextdbscan_cuda.h"
#endif
#ifndef CUDA_ON
#include "nextdbscan_omp.h"
#endif
#include "next_util.h"


// TODO remove this when not needed anymore
//inline bool dist_leq(const float *coord1, const float *coord2, const int max_d, const float e2) noexcept {
//    float tmp = 0;
//    #pragma unroll
//    for (int d = 0; d < max_d; d++) {
//        float tmp2 = coord1[d] - coord2[d];
//        tmp += tmp2 * tmp2;
//    }
//    return tmp <= e2;
//}



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
    /*
    std::vector<uint> dims(max_d);
    dims_mult[0] = 1;
    for (uint d = 0; d < max_d; d++) {
        dims[d] = ((max_bounds[d] - min_bounds[d]) / e_innere) + 1;
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
     */
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

void nc_tree::build_tree() noexcept {
    s_vec<float> v_eps_levels(n_level);
    s_vec<ull> v_dims_mult(n_level * n_dim);
    #pragma omp parallel for
    for (uint l = 0; l < n_level; l++) {
        v_eps_levels[l] = (e_inner * pow(2, l));
        calc_dims_mult(&v_dims_mult[l * n_dim], n_dim, v_min_bounds, v_max_bounds, v_eps_levels[l]);
    }
    index_points(v_eps_levels, v_dims_mult);
}

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



