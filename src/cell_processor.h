//
// Created by Ernir Erlingsson on 7.5.2020.
//

#ifndef NEXT_DBSCAN_CELL_PROCESSOR_H
#define NEXT_DBSCAN_CELL_PROCESSOR_H

#include <cstdint>
#include "nc_tree.h"

static const int UNDEFINED = -1;
static const int UNASSIGNED = -1;
static const uint8_t UNKNOWN = 0;
static const uint8_t NO_CORES = 0x1;
static const uint8_t SOME_CORES = 0x2;
static const uint8_t ALL_CORES = 0x3;

static const uint8_t NOT_CONNECTED = 0x1;
static const uint8_t FULLY_CONNECTED = 0x2;
static const uint8_t PARTIALLY_CONNECTED = 0x3;
static const uint8_t NOT_CORE_CONNECTED = 0x4;
static const uint8_t CORE_CONNECTED = 0x5;

class cell_processor {
private:
    s_vec<char> v_edge_conn;
    s_vec<long> v_leaf_cell_np;
    s_vec<long> v_point_np;
    s_vec<char> v_leaf_cell_type;
    s_vec<char> v_is_core;
    s_vec<long> v_point_labels;
    long const n_threads;
public:
    explicit cell_processor(long const n_threads) : n_threads(n_threads) {}

    void infer_types(nc_tree &nc) noexcept;

    void process_edges(s_vec<float> &v_coords,
            s_vec<long> v_edges,
            nc_tree &nc) noexcept;

    void process_edges(s_vec<float> &v_coords,
            s_vec<long> v_edges,
            nc_tree &nc1,
            nc_tree &nc2) noexcept;

    void determine_cell_labels(s_vec<float> &v_coords,
            s_vec<long> v_edges,
            nc_tree &nc) noexcept;

    void get_result_meta(long &n_cores, long &n_noise, long &clusters) noexcept;

};


#endif //NEXT_DBSCAN_CELL_PROCESSOR_H