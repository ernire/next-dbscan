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

#ifndef NEXT_DBSCAN_NC_TREE_H
#define NEXT_DBSCAN_NC_TREE_H

#include <vector>
#include <cmath>

typedef unsigned int uint;
typedef unsigned long long ull;

template <class T>
using s_vec = std::vector<T>;
template <class T>
using d_vec = std::vector<std::vector<T>>;
using t_uint_iterator = std::vector<std::vector<std::vector<uint>::iterator>>;

static const int UNDEFINED = -1;

class nc_tree {
private:
    bool is_cached = false;
    float *v_coords;
    const uint n_dim;
    const uint n_coords;
    const uint m;
    const float e;
    const float e2;
    const float e_inner;
    s_vec<float> v_min_bounds;
    s_vec<float> v_max_bounds;
    d_vec<uint> vv_index_map;
    d_vec<uint> vv_cell_begin;
    d_vec<uint> vv_cell_ns;
    d_vec<float> vv_min_cell_dim;
    d_vec<float> vv_max_cell_dim;

    void cache_tasks() noexcept;

    void calc_bounds(float *min_bounds, float *max_bounds) noexcept;

    void calc_dims_mult(ull *dims_mult, uint max_d, s_vec<float> &min_bounds,
            s_vec<float> &max_bounds, float e_inner) noexcept;

    uint determine_data_boundaries() noexcept;

public:
    uint n_level = 0;

    explicit nc_tree(float* v_coords, uint n_dim, uint n_coords, float e, uint m)
        : v_coords(v_coords), n_dim(n_dim), n_coords(n_coords), m(m), e(e), e2(e*e), e_inner((e / sqrtf(3))) {}

    void init() noexcept;

    void build_tree(uint n_threads) noexcept;

    void cache_bounding_box_queries();

    int get_no_of_cells(uint tree_level) noexcept {
        if (tree_level > n_level)
            return UNDEFINED;
        return vv_cell_ns[tree_level].size();
    }

};


#endif //NEXT_DBSCAN_NC_TREE_H
