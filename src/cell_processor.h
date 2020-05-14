//
// Created by Ernir Erlingsson on 7.5.2020.
//

#ifndef NEXT_DBSCAN_CELL_PROCESSOR_H
#define NEXT_DBSCAN_CELL_PROCESSOR_H

#include <cstdint>
#include "nc_tree_new.h"

/*
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
*/

class cell_processor {
private:
    s_vec<char> v_edge_conn;
    s_vec<long> v_leaf_cell_np;
    s_vec<long> v_point_np;
    s_vec<char> v_leaf_cell_type;
    s_vec<char> v_is_core;
    s_vec<long> v_point_labels;
    int const n_threads;
public:
    explicit cell_processor(int const n_threads) : n_threads(n_threads) {}

    void infer_types(nc_tree_new &nc) noexcept;

    void process_edges(s_vec<float> &v_coords,
            s_vec<long> v_edges,
            nc_tree_new &nc) noexcept;

    void process_edges(s_vec<float> &v_coords,
            s_vec<long> v_edges,
            nc_tree_new &nc1,
            nc_tree_new &nc2) noexcept;

    void determine_cell_labels(s_vec<float> &v_coords,
            s_vec<long> v_edges,
            nc_tree_new &nc) noexcept;

    void get_result_meta(long &n_cores, long &n_noise, long &clusters) noexcept;

    void partition_data(std::vector<float> &v_coords,
            s_vec<float> &v_min_bounds,
            s_vec<float> &v_max_bounds,
            long const n_partitions,
            unsigned long const n_coords,
            unsigned long const n_dim,
            long const n_level,
            float const e_lowest,
            s_vec<unsigned long> &v_part_coord,
            s_vec<unsigned long> &v_part_offset,
            s_vec<unsigned long> &v_part_size) noexcept;
};


#endif //NEXT_DBSCAN_CELL_PROCESSOR_H
