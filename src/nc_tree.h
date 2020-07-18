//
// Created by Ernir Erlingsson on 6.5.2020.
//

#ifndef NEXT_DBSCAN_NC_TREE_H
#define NEXT_DBSCAN_NC_TREE_H


#include <math.h>
#include <cstdint>
#ifdef CUDA_ON
#include "nextdbscan_cu.cuh"
#endif
#ifndef CUDA_ON
#include "nextdbscan_omp.h"
#endif

class nc_tree {
private:
    struct cell_meta_pair_level {
        long l, c1, c2;

        cell_meta_pair_level(long l, long c1, long c2) : l(l), c1(c1), c2(c2) {}
    };
    d_vec<float> vv_min_cell_dim;
    d_vec<float> vv_max_cell_dim;
    s_vec<float> &v_coords;
    s_vec<float> &v_min_bounds;
    s_vec<float> &v_max_bounds;
    s_vec<unsigned long> v_part_offset;
    s_vec<unsigned long> v_part_size;

    unsigned long set_partition_level(s_vec<unsigned long> &v_ordered_dim,
            unsigned long &level,
            long const min_sample_size) {
        long max_cells = 1;
        float e_lvl = 0;
        while (level > 0 && max_cells < min_sample_size) {
            --level;
            e_lvl = (e_lowest * powf(2, level));
            max_cells = 1;
            for (auto const &d : v_ordered_dim) {
                max_cells *= static_cast<unsigned long>(((v_max_bounds[d] - v_min_bounds[d]) / e_lvl) + 1);
                if  (max_cells > min_sample_size) {
                    return static_cast<unsigned long>(max_cells);
                }
            }
        }
    }

    unsigned long select_partition_dimensions(long const min_sample_size, s_vec<unsigned long> &v_ordered_dim,
            s_vec<unsigned long> &v_cell_size_mul, float const e_lvl) noexcept;

    void process_stack(std::vector<cell_meta_pair_level> &v_stack, s_vec<long> &v_edges) noexcept;

    void process_tree_node(std::vector<cell_meta_pair_level> &v_stack, s_vec<long> &v_edges, long const l,
            long const c) noexcept;


public:
    unsigned long const m;
    float const e;
    float const e2;
    unsigned long const n_coords;
    unsigned long const n_dim;
    unsigned long const n_level;
    unsigned long n_level_parallel;
    float const e_lowest;
    d_vec<long> vv_index_map;
    d_vec<long> vv_cell_begin;
    d_vec<long> vv_cell_ns;
    explicit nc_tree(s_vec<float> &v_coords,
            s_vec<float> &v_min_bounds,
            s_vec<float> &v_max_bounds,
            unsigned long const m,
            float const e,
            unsigned long n_coords,
            unsigned long n_dim,
            unsigned long n_level,
            float const e_lowest)
            : v_coords(v_coords), v_min_bounds(v_min_bounds), v_max_bounds(v_max_bounds), m(m), e(e), e2(e*e),
            n_coords(n_coords), n_dim(n_dim), n_level(n_level), n_level_parallel(n_level), e_lowest(e_lowest) {
        vv_index_map.resize(n_level);
        vv_cell_begin.resize(n_level);
        vv_cell_ns.resize(n_level);
        vv_min_cell_dim.resize(n_level);
        vv_max_cell_dim.resize(n_level);
    }

    void partition_data(long const min_partitions, unsigned long const n_threads) noexcept;

    void partition_and_distribute(long const min_partitions, unsigned long const n_nodes) noexcept;

    void build_tree_parallel(unsigned long const n_threads) noexcept;

    void build_tree() noexcept;

    void collect_edges(s_vec<long> &v_edges) noexcept;

    void collect_edges_parallel_old(s_vec<long> &v_edges, unsigned long const n_threads) noexcept;

    inline unsigned long get_no_of_cells(long tree_level) noexcept {
        return vv_cell_ns[tree_level].size();
    }

    inline unsigned long get_total_no_of_cells() noexcept {
        unsigned long sum = 0;
        for (unsigned long l = 0; l < n_level; ++l) {
            sum += vv_cell_ns[l].size();
        }
        return sum;
    }

    void print_tree_meta_data() {
        std::cout << "NC-tree levels: " << n_level << std::endl;
        for (unsigned long l = 0; l < n_level; ++l) {
            std::cout << "Level: " << l << " has " << get_no_of_cells(l) << " cells" << std::endl;
        }
    }

    void collect_edges_parallel(s_vec<long> &v_edges, unsigned long const n_threads) noexcept ;
};


#endif //NEXT_DBSCAN_NC_TREE_H
