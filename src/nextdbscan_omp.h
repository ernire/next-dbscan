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
#ifndef NEXT_DBSCAN_NEXTDBSCAN_OMP_H
#define NEXT_DBSCAN_NEXTDBSCAN_OMP_H

#include <vector>
#include "nextdbscan.h"

template <class T>
using s_vec = std::vector<T>;
template <class T>
using d_vec = std::vector<std::vector<T>>;
using t_uint_iterator = std::vector<std::vector<std::vector<uint>::iterator>>;

class nextdbscan_omp {

private:
    static void calculate_level_cell_bounds(float *v_coords, s_vec<uint> &v_cell_begins,
            s_vec<uint> &v_cell_ns, s_vec<uint> &v_index_maps,
            std::vector<std::vector<float>> &vv_min_cell_dims,
            std::vector<std::vector<float>> &vv_max_cell_dims, uint max_d, uint l) noexcept;

    static uint index_level_and_get_cells(float *v_coords,
            s_vec<float> &v_min_bounds, d_vec<uint> &vv_index_map,
            d_vec<uint> &vv_cell_begin, s_vec<uint> &v_cell_ns,
            std::vector<ull> &v_value_map, std::vector<std::vector<uint>> &v_bucket,
            std::vector<ull> &v_bucket_separator, std::vector<ull> &v_bucket_separator_tmp,
            t_uint_iterator &v_iterator, uint size, int l, uint max_d, uint node_offset,
            float level_eps, ull *dims_mult, uint n_threads) noexcept;

public:
    static void index_points(float *v_coords,
            s_vec<float> &v_eps_levels,
            s_vec<ull> &v_dims_mult,
            s_vec<float> &v_min_bounds,
            d_vec<uint> &vv_index_map,
            d_vec<uint> &vv_cell_begin,
            d_vec<uint> &vv_cell_ns,
            d_vec<float> &vv_min_cell_dim,
            d_vec<float> &vv_max_cell_dim,
            uint max_d, uint n_threads,
            uint max_levels, uint n) noexcept;
};


#endif //NEXT_DBSCAN_NEXTDBSCAN_OMP_H
