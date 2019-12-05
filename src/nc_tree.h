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

// TODO CUDA
template <class T>
using s_vec = std::vector<T>;
template <class T>
using d_vec = std::vector<std::vector<T>>;
using t_uint_iterator = std::vector<std::vector<std::vector<uint>::iterator>>;

static const int UNDEFINED = -1;
static const uint8_t UNKNOWN = 0;
static const uint8_t NO_CORES = 0x1;
static const uint8_t SOME_CORES = 0x2;
static const uint8_t ALL_CORES = 0x3;

static const uint8_t NOT_CONNECTED = 0x1;
static const uint8_t FULLY_CONNECTED = 0x2;
static const uint8_t PARTIALLY_CONNECTED = 0x2;
/*
        std::vector<std::vector<cell_meta_3>> vv_stacks3(n_threads);
        std::vector<std::vector<bool>> vv_range_table(n_threads);
        std::vector<std::vector<uint>> vv_range_counts(n_threads);
        std::vector<uint> v_leaf_cell_np(vv_cell_ns[0].size(), 0);
        std::vector<uint> v_point_np(n, 0);
        std::vector<uint8_t> v_cell_type(vv_cell_ns[0].size(), NC);
        std::vector<uint8_t> v_is_core(n, 0);
 */

struct cell_meta_pair {
    uint c1, c2;

    cell_meta_pair(uint c1, uint c2) : c1(c1), c2(c2) {}
};

class nc_tree {
private:
    float *v_coords;
    const float e2;
    const float e_inner;
    s_vec<float> v_min_bounds;
    s_vec<float> v_max_bounds;
    d_vec<uint> vv_index_map;
    d_vec<uint> vv_cell_begin;
    d_vec<uint> vv_cell_ns;
    d_vec<float> vv_min_cell_dim;
    d_vec<float> vv_max_cell_dim;
    s_vec<cell_meta_pair> v_edges;
    uint max_points_in_cell = 0;
    std::vector<uint> v_leaf_cell_np;
    std::vector<uint> v_point_np;
    std::vector<uint8_t> v_leaf_cell_type;

    void calc_bounds(float *min_bounds, float *max_bounds) noexcept;

    void calc_dims_mult(ull *dims_mult, uint max_d, s_vec<float> &min_bounds,
            s_vec<float> &max_bounds, float e_inner) noexcept;

    uint determine_data_boundaries() noexcept;

public:
    uint n_level = 0;
    const uint n_dim;
    const uint n_coords;
    const uint m;
    const float e;
    const uint n_threads;

    explicit nc_tree(float* v_coords, uint n_dim, uint n_coords, float e, uint m, uint n_threads)
        : v_coords(v_coords), n_dim(n_dim), n_coords(n_coords), m(m), e(e), n_threads(n_threads), e2(e*e),
        e_inner((e / sqrtf(3))) {}

    void init() noexcept;

    void build_tree() noexcept;

    void collect_proximity_queries() noexcept;

    void process_proximity_queries() noexcept;

    int get_no_of_cells(uint tree_level) noexcept {
        if (tree_level > n_level)
            return UNDEFINED;
        return vv_cell_ns[tree_level].size();
    }

    uint get_no_of_edges() noexcept {
        return v_edges.size();
    }

};


#endif //NEXT_DBSCAN_NC_TREE_H
