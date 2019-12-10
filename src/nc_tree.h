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
#include "nextdbscan.h"
#ifdef CUDA_ON
#include "nextdbscan_cuda.h"
#endif
#ifndef CUDA_ON
#include "nextdbscan_omp.h"
#endif

typedef unsigned int uint;
typedef unsigned long long ull;

static const int UNDEFINED = -1;
static const uint8_t UNKNOWN = 0;
static const uint8_t NO_CORES = 0x1;
static const uint8_t SOME_CORES = 0x2;
static const uint8_t ALL_CORES = 0x3;

static const uint8_t NOT_CONNECTED = 0x1;
static const uint8_t FULLY_CONNECTED = 0x2;
static const uint8_t PARTIALLY_CONNECTED = 0x3;
static const uint8_t NOT_CORE_CONNECTED = 0x4;
static const uint8_t CORE_CONNECTED = 0x5;

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
    s_vec<uint> v_edges;
    s_vec<uint8_t> v_edge_conn;
    uint max_points_in_cell = 0;
    s_vec<uint> v_leaf_cell_np;
    s_vec<uint> v_point_np;
    s_vec<uint8_t> v_leaf_cell_type;
    s_vec<uint8_t> v_is_core;
    s_vec<int> v_point_labels;
#ifdef CUDA_ON
    thrust::device_vector<float> v_gpu_coords;
    thrust::device_vector<uint> v_gpu_edges;
    thrust::device_vector<int> v_gpu_point_labels;
    thrust::device_vector<uint8_t> v_gpu_is_core;
    thrust::device_vector<uint8_t> v_gpu_leaf_cell_type;
#endif

    void calc_bounds(float *min_bounds, float *max_bounds) noexcept;

    void calc_dims_mult(ull *dims_mult, uint max_d, s_vec<float> &min_bounds,
            s_vec<float> &max_bounds, float e_inner) noexcept;

    uint determine_data_boundaries() noexcept;

    void index_points(s_vec<float> &v_eps_levels, s_vec<ull> &v_dims_mult) noexcept;

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

    void build_tree() noexcept;

    void collect_proximity_queries() noexcept;

    void init() noexcept;

    void infer_types_and_max_clusters() noexcept;

    void process_proximity_queries() noexcept;

    void determine_cell_labels() noexcept;

    int get_no_of_cells(uint tree_level) noexcept {
        if (tree_level > n_level)
            return UNDEFINED;
        return vv_cell_ns[tree_level].size();
    }

    int cnt_leaf_cells_of_type(const uint type) {
        if (n_level == 0 || v_leaf_cell_type.empty())
            return UNDEFINED;
        uint cnt = 0;
        #pragma omp parallel for reduction(+:cnt)
        for (uint i = 0; i < v_leaf_cell_type.size(); ++i) {
            if (v_leaf_cell_type[i] == type)
                ++cnt;
        }
        return cnt;
    }

    uint get_no_of_edges() noexcept {
        return v_edges.size() / 2;
    }

    uint get_no_of_cores() noexcept {
        uint sum = 0;
        #pragma omp parallel for reduction(+:sum)
        for (uint i = 0; i < v_is_core.size(); ++i) {
            if (v_is_core[i])
                ++sum;
        }
        return sum;
    }

    uint get_no_of_clusters() noexcept {
        uint sum = 0;
        #pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < v_point_labels.size(); ++i) {
            if (v_point_labels[i] == i)
                ++sum;
        }
        return sum;
    }

    uint get_no_of_noise() noexcept {
        uint sum = 0;
        #pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < v_point_labels.size(); ++i) {
            // TODO replace with const value from nextdbscan.h
            if (v_point_labels[i] == -1)
                ++sum;
        }
        return sum;
    }

};


#endif //NEXT_DBSCAN_NC_TREE_H
