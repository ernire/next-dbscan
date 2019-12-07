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
#ifndef NEXT_DBSCAN_NEXTDBSCAN_CUDA_H
#define NEXT_DBSCAN_NEXTDBSCAN_CUDA_H

#include <thrust/device_vector.h>
#include "nextdbscan.h"

// V100
static const int CUDA_BLOCKS = 128;
static const int CUDA_THREADS = 1024;

template <class T>
using s_vec = thrust::host_vector<T>;
template <class T>
using d_vec = thrust::host_vector<thrust::host_vector<T>>;
using t_uint_iterator = thrust::host_vector<thrust::host_vector<thrust::host_vector<uint>::iterator>>;
#include "nc_tree.h"

/*
class nextdbscan_cuda {
private:

    static void calculate_level_cell_bounds(
            thrust::device_vector<float> &v_coords,
            thrust::device_vector<uint> &v_device_cell_begin,
            thrust::device_vector<uint> &v_device_index_map,
            thrust::device_vector<uint> &v_device_cell_ns,
            thrust::device_vector<float> &v_min_cell_dim,
            thrust::device_vector<float> &v_last_min_cell_dim,
            thrust::device_vector<float> &v_max_cell_dim,
            thrust::device_vector<float> &v_last_max_cell_dim,
            const uint l, const uint max_d) noexcept;

public:

    static void index_points(s_vec<float> &v_coords,
            s_vec<float> &v_eps_levels,
            s_vec<ull> &v_dims_mult,
            s_vec<float> &v_min_bounds,
            d_vec<uint> &vv_index_map,
            d_vec<uint> &vv_cell_begin,
            d_vec<uint> &vv_cell_ns,
            d_vec<float> &vv_min_cell_dim,
            d_vec<float> &vv_max_cell_dim,
            uint max_d, uint n_threads,
            uint max_levels, uint size) noexcept;

    static uint index_level_and_get_cells(thrust::device_vector<float> &v_coords,
            thrust::device_vector<uint> &v_device_index_map,
            thrust::device_vector<uint> &v_device_cell_ns,
            thrust::device_vector<uint> &v_device_cell_begin,
            thrust::device_vector<float> &v_min_bounds,
            thrust::device_vector<ull> &v_device_dims_mult,
            thrust::device_vector<float> &v_level_eps,
            thrust::device_vector<ull> &v_value_map,
            thrust::device_vector<uint> &v_coord_indexes,
            thrust::device_vector<uint> &v_unique_cnt,
            thrust::device_vector<uint> &v_indexes,
            thrust::device_vector<ull> &v_dims_mult,
            thrust::device_vector<uint> &v_tmp,
            const uint size, const uint l, const uint max_d) noexcept;
};
 */

void index_points(float *v_coords,
        s_vec<float> &v_eps_levels,
        s_vec<ull> &v_dims_mult,
        s_vec<float> &v_min_bounds,
        d_vec<uint> &vv_index_map,
        d_vec<uint> &vv_cell_begin,
        d_vec<uint> &vv_cell_ns,
        d_vec<float> &vv_min_cell_dim,
        d_vec<float> &vv_max_cell_dim,
        uint max_d, uint n_threads,
        uint max_levels, uint size) noexcept;

#endif //NEXT_DBSCAN_NEXTDBSCAN_CUDA_H
