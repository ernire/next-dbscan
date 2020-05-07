//
// Created by Ernir Erlingsson on 6.5.2020.
//

#ifndef NEXT_DBSCAN_NC_TREE_NEW_H
#define NEXT_DBSCAN_NC_TREE_NEW_H


#include <math.h>
#include <cstdint>
#ifdef CUDA_ON
#include "nextdbscan_cuda.h"
#endif
#ifndef CUDA_ON
#include "nextdbscan_omp.h"
#endif

class nc_tree_new {
private:
    float const e_lowest;
    d_vec<float> vv_min_cell_dim;
    d_vec<float> vv_max_cell_dim;

public:
    float const e;
    float const e2;
    unsigned long const n_level;
    unsigned long const m;
    unsigned long const n_coords;
    unsigned long const n_dim;
    d_vec<long> vv_index_map;
    d_vec<long> vv_cell_begin;
    d_vec<long> vv_cell_ns;
    explicit nc_tree_new(
            float const e,
            float const e_lowest,
            unsigned long n_dim,
            unsigned long n_level,
            unsigned long n_coords,
            unsigned long const m)
            : m(m), e(e), e2(e*e), n_coords(n_coords), n_dim(n_dim), n_level(n_level), e_lowest(e_lowest) {
        vv_index_map.resize(n_level);
        vv_cell_begin.resize(n_level);
        vv_cell_ns.resize(n_level);
        vv_min_cell_dim.resize(n_level);
        vv_max_cell_dim.resize(n_level);
    }

//    void init() noexcept;

    void build_tree(s_vec<float> &v_coords, s_vec<float> &v_min_bounds) noexcept;

    void collect_edges(s_vec<long> &v_edges) noexcept;

    inline long get_no_of_cells(long tree_level) noexcept {
        if (tree_level > n_level)
            return -1;
        return vv_cell_ns[tree_level].size();
    }

    inline long get_total_no_of_cells() noexcept {
        long sum = 0;
        for (long l = 0; l < n_level; ++l) {
            sum += vv_cell_ns[l].size();
        }
        return sum;
    }

};


#endif //NEXT_DBSCAN_NC_TREE_NEW_H
