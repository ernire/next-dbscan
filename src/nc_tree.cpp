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
#include "nc_tree.h"

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

void nc_tree::init() noexcept {
    v_min_bounds.resize(n_dim);
    v_max_bounds.resize(n_dim);
    n_level = determine_data_boundaries();
}
