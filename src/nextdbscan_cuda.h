//
// Created by Ernir Erlingsson on 22.11.2019.
//

#ifndef NEXT_DBSCAN_NEXTDBSCAN_CUDA_H
#define NEXT_DBSCAN_NEXTDBSCAN_CUDA_H

typedef unsigned long long ull;
typedef unsigned int uint;

#include <thrust/device_vector.h>

static const int CUDA_BLOCKS = 128;
static const int CUDA_THREADS = 1024;

template <class T>
using s_vec = thrust::host_vector<T>;
template <class T>
using d_vec = thrust::host_vector<thrust::host_vector<T>>;
using t_uint_iterator = thrust::host_vector<thrust::host_vector<thrust::host_vector<uint>::iterator>>;

uint cu_index_level_and_get_cells(thrust::device_vector<float> &v_coords,
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

void cu_calculate_level_cell_bounds(
        thrust::device_vector<float> &v_coords,
        thrust::device_vector<uint> &v_device_cell_begin,
        thrust::device_vector<uint> &v_device_index_map,
        thrust::device_vector<uint> &v_device_cell_ns,
        thrust::device_vector<float> &v_min_cell_dim,
        thrust::device_vector<float> &v_last_min_cell_dim,
        thrust::device_vector<float> &v_max_cell_dim,
        thrust::device_vector<float> &v_last_max_cell_dim,
        const uint l, const uint max_d) noexcept;

#endif //NEXT_DBSCAN_NEXTDBSCAN_CUDA_H
