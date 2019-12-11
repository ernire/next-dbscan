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
#include "nextdbscan_cuda.h"

__global__ void index_kernel(const float* v_coord, uint* v_index, ull* v_map, const float* v_min,
        const ull* v_mult, const uint size, const uint max_d, const float eps) {

    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint coord_index;
    ull cell_index;

    while (globalIdx < size) {
        coord_index = v_index[globalIdx] * max_d;
        cell_index = 0;
        #pragma unroll
        for (uint d = 0; d < max_d; d++) {
            cell_index += (ull)((v_coord[coord_index+d] - v_min[d]) / eps) * v_mult[d];
        }
        v_map[globalIdx] = cell_index;
        globalIdx += blockDim.x * gridDim.x;
        __syncthreads();
    }
}

__global__ void count_unique_groups(const ull* v_input, uint* v_output, const uint size) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint8_t is_processing;
    int cnt, index;
    ull val;

    while (globalIdx < size) {
        is_processing = 0;
        if (globalIdx == 0) {
            is_processing = 1;
        } else {
            if (v_input[globalIdx] != v_input[globalIdx-1]) {
                is_processing = 1;
            }
        }
        if (is_processing > 0) {
            cnt = 0;
            index = globalIdx;
            val = v_input[index];
            while(index < size && val == v_input[index]) {
                ++cnt;
                ++index;
            }
            v_output[globalIdx] = cnt;
        }
        globalIdx += blockDim.x * gridDim.x;
    }
}

__global__ void fill_point_np(uint *v_point_np, const uint *v_index, const uint *v_cell_begin,
        const uint *v_cell_ns, const uint size) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint index, begin, p;
    while (globalIdx < size) {
        index = globalIdx;
        begin = v_cell_begin[index];
        for (uint i = 0; i < v_cell_ns[index]; ++i) {
            p = v_index[begin + i];
            v_point_np[p] = v_cell_ns[index];
        }
        globalIdx += blockDim.x * gridDim.x;
    }
}

__global__ void fill_is_core(uint *v_point_np, uint8_t *v_is_core, const uint m, const uint size) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    while (globalIdx < size) {
        if (v_point_np[globalIdx] >= m)
            v_is_core[globalIdx] = 1;
        globalIdx += blockDim.x * gridDim.x;
    }
}

__global__ void init_cell_label(int *v_point_labels, uint8_t *v_cell_type, uint8_t *v_is_core,
        const uint *v_index, const uint *v_cell_begin, const uint *v_cell_ns, const uint size) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint index, begin, p, core_p;

    while (globalIdx < size) {
        index = globalIdx;
        assert(v_cell_type[index] != UNKNOWN);
        if (v_cell_type[index] != NO_CORES) {
            begin = v_cell_begin[index];
            for (uint i = 0; i < v_cell_ns[index]; ++i) {
                p = v_index[begin + i];
                if (v_is_core[p]) {
                    core_p = p;
                    v_point_labels[p] = p;
                    i = v_cell_ns[index];
                }
            }
            for (uint i = 0; i < v_cell_ns[index]; ++i) {
                p = v_index[begin + i];
                if (p == core_p)
                    continue;
                v_point_labels[p] = core_p;
            }
        }
        globalIdx += blockDim.x * gridDim.x;
    }
}

__global__ void update_cell_type(uint8_t *v_cell_type, uint8_t *v_is_core, const uint *v_index,
        const uint *v_cell_begin, const uint *v_cell_ns, const uint size) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint index, begin, p;
    uint8_t type;

    while (globalIdx < size) {
        index = globalIdx;
        begin = v_cell_begin[index];
        type = UNKNOWN;
        for (uint i = 0; i < v_cell_ns[index]; ++i) {
            p = v_index[begin + i];
            if (v_is_core[p]) {
                if (type == UNKNOWN) {
                    type = ALL_CORES;
                } else if (type == NO_CORES) {
                    type = SOME_CORES;
                }
            } else {
                if (type == UNKNOWN) {
                    type = NO_CORES;
                } else if (type == ALL_CORES) {
                    type = SOME_CORES;
                }
            }
        }
        v_cell_type[index] = type;
        globalIdx += blockDim.x * gridDim.x;
    }
}

__device__ bool dist_leq_e2(const float *v_coords, const uint p1, const uint p2,
        const uint n_dim, const float e2) {
    float tmp = 0, tmp2;
    #pragma unroll
    for (int d = 0; d < n_dim; d++) {
        tmp2 = v_coords[(p1*n_dim)+d] - v_coords[(p2*n_dim)+d];
        tmp += tmp2 * tmp2;
    }
    return tmp <= e2;
}

__device__ bool are_core_connected(const float *v_coords, const uint *v_index_map, const uint *v_cell_begin,
        const uint *v_cell_ns, const uint8_t *v_is_core,
        const uint c1, const uint c2, const uint n_dim, const float e2) {
    uint begin1 = v_cell_begin[c1];
    uint begin2 = v_cell_begin[c2];
    for (uint k1 = 0; k1 < v_cell_ns[c1]; ++k1) {
        uint p1 = v_index_map[begin1 + k1];
        if (!v_is_core[p1]) {
            continue;
        }
        for (uint k2 = 0; k2 < v_cell_ns[c2]; ++k2) {
            uint p2 = v_index_map[begin2 + k2];
            if (!v_is_core[p2]) {
                continue;
            }
            if (dist_leq_e2(v_coords, p1, p2, n_dim, e2)) {
                return true;
            }
        }
    }
    return false;
}

__global__ void process_edge_queries(const float *v_coords, const uint *v_edges, const uint *v_index,
        const uint *v_cell_begin, const uint *v_cell_ns, uint *v_point_np, const uint size, const uint m,
        const uint n_dim, const float e2) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint edge_index;
    uint c1, c2, size1, size2, begin1, begin2, p1, p2;

    while (globalIdx < size) {
        edge_index = globalIdx * 2;
        c1 = v_edges[edge_index];
        c2 = v_edges[edge_index+1];
        size1 = v_cell_ns[c1];
        size2 = v_cell_ns[c2];
        begin1 = v_cell_begin[c1];
        begin2 = v_cell_begin[c2];

        for (uint i = 0; i < size1; ++i) {
            p1 = v_index[begin1 + i];
            for (uint j = 0; j < size2; ++j) {
                p2 = v_index[begin2 + j];
                if (v_point_np[p1] < m || v_point_np[p2] < m) {
                    if (dist_leq_e2(v_coords, p1, p2, n_dim, e2)) {
                        if (v_point_np[p1] < m) {
                            atomicAdd(&v_point_np[p1], 1);
                        }
                        if (v_point_np[p2] < m) {
                            atomicAdd(&v_point_np[p2], 1);
                        }
                    }
                }
            }
        }
        globalIdx += blockDim.x * gridDim.x;
    }
}

__global__ void flatten_label_graph(int *v_min_label, const uint size) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint index, label;
    bool update;

    while (globalIdx < size) {
        index = globalIdx;
        if (v_min_label[index] != ROOT_CLUSTER && v_min_label[index] != UNASSIGNED) {
            label = v_min_label[index];
            update = false;
            while (v_min_label[label] != ROOT_CLUSTER) {
                label = v_min_label[label];
                update = true;
            }
            if (update)
                v_min_label[index] = label;
        }
        globalIdx += blockDim.x * gridDim.x;
    }
}

__global__ void update_point_labels(int *v_point_labels, const int *v_min_label, const uint *v_index,
        const uint *v_cell_begin, const uint *v_cell_ns, const uint8_t *v_cell_type, const uint size) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint index, begin, label, j;
    int p;

    while (globalIdx < size) {
        index = globalIdx;
        if (v_min_label[index] != ROOT_CLUSTER && v_min_label[index] != UNASSIGNED) {
            index = globalIdx;
            begin = v_cell_begin[index];
            if (v_cell_type[index] != NO_CORES) {
                p = v_index[begin];
                if (p != v_point_labels[p]) {
                    p = v_point_labels[p];
                }
                v_point_labels[p] = v_point_labels[v_index[v_cell_begin[v_min_label[index]]]];
            } else {
                label = v_point_labels[v_index[v_cell_begin[v_min_label[index]]]];
                for (j = 0; j < v_cell_ns[index]; ++j) {
                    v_point_labels[v_index[begin+j]] = label;
                }
            }
        }
        globalIdx += blockDim.x * gridDim.x;
    }
}

__global__ void update_cell_labels(const float *v_coords, const uint *v_edges, int *v_min_label,
        const uint8_t *v_cell_type, const uint8_t *v_is_core, const uint *v_index,
        const uint *v_cell_begin, const uint *v_cell_ns, int *v_point_labels,
        const uint size, const uint n_dim, const float e2) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint edge_index;
    uint c1, c2;
    uint label1, label2;
    uint begin1, begin2;
    int c_lower, c_higher;

    while (globalIdx < size) {
        edge_index = globalIdx * 2;
        c1 = v_edges[edge_index];
        c2 = v_edges[edge_index+1];
        if (v_cell_type[c1] != NO_CORES || v_cell_type[c2] != NO_CORES) {
            if (v_cell_type[c1] != NO_CORES && v_cell_type[c2] != NO_CORES) {
                if (are_core_connected(v_coords, v_index, v_cell_begin, v_cell_ns, v_is_core,
                        c1, c2, n_dim, e2)) {
                    label1 = c1;
                    while (v_min_label[label1] != ROOT_CLUSTER) {
                        label1 = v_min_label[label1];
                    }
                    label2 = c2;
                    while (v_min_label[label2] != ROOT_CLUSTER) {
                        label2 = v_min_label[label2];
                    }
                    if (label1 != label2) {
                        // TODO atomic min
                        if (label1 < label2) {
//                            atomicMin(&v_min_label[label2], label1);
                            v_min_label[label2] = label1;
                        } else {
//                            atomicMin(&v_min_label[label1], label2);
                            v_min_label[label1] = label2;
                        }
                    }
                }
            } else if ((v_cell_type[c1] != NO_CORES || v_cell_type[c2] != NO_CORES)
                      && (v_cell_type[c1] == NO_CORES || v_cell_type[c2] == NO_CORES)) {
                c_lower = (v_cell_type[c1] != NO_CORES)? c1 : c2;
                c_higher = (v_cell_type[c1] != NO_CORES)? c2 : c1;
                if (v_min_label[c_higher] == UNASSIGNED) {
                    begin1 = v_cell_begin[c_lower];
                    begin2 = v_cell_begin[c_higher];
                    for (uint k1 = 0; k1 < v_cell_ns[c_lower]; ++k1) {
                        uint p1 = v_index[begin1 + k1];
                        if (v_is_core[p1]) {
                            for (uint k2 = 0; k2 < v_cell_ns[c_higher]; ++k2) {
                                uint p2 = v_index[begin2 + k2];
                                if (v_point_labels[p2] == UNASSIGNED) {
                                    if (dist_leq_e2(v_coords, p1, p2, n_dim, e2)) {
                                        v_point_labels[p2] = p1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        globalIdx += blockDim.x * gridDim.x;
    }
}


__global__ void discover_min_local_labels(const float *v_coords, const uint *v_edges, int *v_min_label,
        const uint8_t *v_cell_type, const uint8_t *v_is_core, const uint *v_index,
        const uint *v_cell_begin, const uint *v_cell_ns, const uint size,
        const uint n_dim, const float e2) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint edge_index;
    uint c1, c2;
    int c_lower, c_higher;

    while (globalIdx < size) {
        edge_index = globalIdx * 2;
        c1 = v_edges[edge_index];
        c2 = v_edges[edge_index+1];

        if (v_cell_type[c1] != NO_CORES || v_cell_type[c2] != NO_CORES) {
            if (v_cell_type[c1] != NO_CORES && v_cell_type[c2] != NO_CORES) {
                c_lower = (c1 < c2)? c1 : c2;
                c_higher = (c1 < c2)? c2 : c1;
                if (v_min_label[c_higher] > c_lower) {
                    if (are_core_connected(v_coords, v_index, v_cell_begin, v_cell_ns, v_is_core,
                            c1, c2, n_dim, e2)) {
                        atomicMin(&v_min_label[c_higher], c_lower);
                    }
                }
            }
        }
        globalIdx += blockDim.x * gridDim.x;
    }
}

__global__ void sum_edge_queries(const uint *v_edges, uint *v_edge_size, const uint *v_cell_ns,
        const uint8_t *v_cell_type, const uint size) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint edge_index;
    uint c1, c2;

    while (globalIdx < size) {
        edge_index = globalIdx * 2;
        c1 = v_edges[edge_index];
        c2 = v_edges[edge_index+1];
        if ((v_cell_type[c1] != NO_CORES && v_cell_type[c1] != ALL_CORES)
            || (v_cell_type[c2] != NO_CORES && v_cell_type[c2] != ALL_CORES)) {
            v_edge_size[globalIdx] = v_cell_ns[c1] * v_cell_ns[c2];
        }
        globalIdx += blockDim.x * gridDim.x;
    }
}

__global__ void sum_range_queries(const uint* v_edges, const uint *v_cell_ns, uint *v_cell_np,
        const uint m, const uint size) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint edge_index;
    uint c1, c2;

    while (globalIdx < size) {
        edge_index = globalIdx * 2;
        c1 = v_edges[edge_index];
        c2 = v_edges[edge_index+1];
        if (v_cell_np[c1] < m) {
            atomicAdd(&v_cell_np[c1], v_cell_ns[c2]);
        }
        if (v_cell_np[c2] < m) {
            atomicAdd(&v_cell_np[c2], v_cell_ns[c1]);
        }
        globalIdx += blockDim.x * gridDim.x;
    }
}

__global__ void determine_min_max(const uint* v_index_map, const uint* v_begin, const uint* v_ns,
        const float* v_min_input, const float* v_max_input, float* v_min_output,
        float* v_max_output, const uint size, const uint max_d, const uint l) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint input_index, output_index;
    while (globalIdx < size*max_d) {
        uint i = globalIdx / max_d;
        uint d = globalIdx % max_d;
        uint begin = v_begin[i];
        input_index = v_index_map[begin] * max_d;
        output_index = i*max_d+d;
        v_min_output[output_index] = v_min_input[input_index + d];
        v_max_output[output_index] = v_max_input[input_index + d];

        for (uint j = 1; j < v_ns[i]; j++) {
            input_index = v_index_map[begin+j] * max_d;
            if (v_min_input[input_index + d] < v_min_output[i * max_d + d]) {
                v_min_output[output_index] = v_min_input[input_index + d];
            }
            if (v_max_input[input_index + d] > v_max_output[output_index]) {
                v_max_output[output_index] = v_max_input[input_index + d];
            }
        }
        globalIdx += blockDim.x * gridDim.x;
    }
}

void print_cuda_memory_usage() {
    size_t free_byte;
    size_t total_byte;
    auto cuda_status = cudaMemGetInfo( &free_byte, &total_byte );

    if ( cudaSuccess != cuda_status ) {
        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
        exit(1);
    }
    double free_db = (double)free_byte ;
    double total_db = (double)total_byte ;
    double used_db = total_db - free_db ;
    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
            used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}

uint index_level_and_get_cells(thrust::device_vector<float> &v_coords,
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
        const uint size, const uint l, const uint max_d) noexcept {
    if (l == 0) {
        v_value_map.resize(size);
        print_cuda_memory_usage();
        v_unique_cnt.resize(size);
        print_cuda_memory_usage();
        v_coord_indexes.resize(size);
        thrust::sequence(v_coord_indexes.begin(), v_coord_indexes.end());
        print_cuda_memory_usage();
        v_indexes.resize(size);
        thrust::sequence(v_indexes.begin(), v_indexes.end());
        print_cuda_memory_usage();
    }
    v_device_index_map.resize(size);
    thrust::sequence(v_device_index_map.begin(), v_device_index_map.end());
    thrust::fill(v_unique_cnt.begin(), v_unique_cnt.end(), 0);
    index_kernel<<<CUDA_BLOCKS ,CUDA_THREADS>>>(
            thrust::raw_pointer_cast(&v_coords[0]),
                    thrust::raw_pointer_cast(&v_coord_indexes[0]),
                    thrust::raw_pointer_cast(&v_value_map[0]),
                    thrust::raw_pointer_cast(&v_min_bounds[0]),
                    thrust::raw_pointer_cast(&v_dims_mult[l * max_d]),
                    size,
                    max_d,
                    v_level_eps[l]
    );
//    print_cuda_memory_usage();
    thrust::sort_by_key(v_value_map.begin(), v_value_map.begin() + size, v_device_index_map.begin());
    v_tmp.resize(size);
    auto ptr_indexes = thrust::raw_pointer_cast(&v_coord_indexes[0]);
    thrust::transform(thrust::device,
            v_device_index_map.begin(),
            v_device_index_map.end(),
            v_tmp.begin(),
            [=] __device__ (const auto val) { return ptr_indexes[val]; });
    count_unique_groups<<<CUDA_BLOCKS,CUDA_THREADS>>>(
            thrust::raw_pointer_cast(&v_value_map[0]),
                    thrust::raw_pointer_cast(&v_unique_cnt[0]),
                    size);
    int result = thrust::count_if(v_unique_cnt.begin(), v_unique_cnt.begin() + size,
    [] __device__ (auto val) { return val > 0; });
    v_device_cell_ns.resize(result);
    v_device_cell_begin.resize(v_device_cell_ns.size());
    v_coord_indexes.resize(v_device_cell_ns.size());
    thrust::copy_if(v_unique_cnt.begin(), v_unique_cnt.begin() + size, v_device_cell_ns.begin(),
            [] __device__ (auto val) { return val > 0; });
    auto ptr = thrust::raw_pointer_cast(&v_unique_cnt[0]);
    thrust::copy_if(v_indexes.begin(), v_indexes.begin() + size, v_device_cell_begin.begin(),
            [=] __device__ (auto val) { return ptr[val] > 0; });
    thrust::copy_if(thrust::device, v_tmp.begin(), v_tmp.end(), v_indexes.begin(),
            v_coord_indexes.begin(),
            [=] __device__ (auto val) { return ptr[val] > 0; });
    return result;
}

void calculate_level_cell_bounds(thrust::device_vector<float> &v_coords,
        thrust::device_vector<uint> &v_device_cell_begin,
        thrust::device_vector<uint> &v_device_index_map,
        thrust::device_vector<uint> &v_device_cell_ns,
        thrust::device_vector<float> &v_min_cell_dim,
        thrust::device_vector<float> &v_last_min_cell_dim,
        thrust::device_vector<float> &v_max_cell_dim,
        thrust::device_vector<float> &v_last_max_cell_dim,
        const uint l, const uint max_d) noexcept {
    v_min_cell_dim.resize(v_device_cell_begin.size() * max_d, 0);
    v_max_cell_dim.resize(v_min_cell_dim.size(), 0);
    float* v_min_input_ptr;
    float* v_max_input_ptr;
    if (l == 0) {
        v_min_input_ptr = thrust::raw_pointer_cast(&v_coords[0]);
        v_max_input_ptr = thrust::raw_pointer_cast(&v_coords[0]);
    } else {
        v_min_input_ptr = thrust::raw_pointer_cast(&v_last_min_cell_dim[0]);
        v_max_input_ptr = thrust::raw_pointer_cast(&v_last_max_cell_dim[0]);
    }
    determine_min_max<<<CUDA_BLOCKS, CUDA_THREADS>>>(
            thrust::raw_pointer_cast(&v_device_index_map[0]),
                    thrust::raw_pointer_cast(&v_device_cell_begin[0]),
                    thrust::raw_pointer_cast(&v_device_cell_ns[0]),
                    v_min_input_ptr,
                    v_max_input_ptr,
                    thrust::raw_pointer_cast(&v_min_cell_dim[0]),
                    thrust::raw_pointer_cast(&v_max_cell_dim[0]),
                    v_device_cell_begin.size(),
                    max_d,
                    l);
    v_last_min_cell_dim.resize(v_min_cell_dim.size());
    thrust::copy(v_min_cell_dim.begin(), v_min_cell_dim.end(), v_last_min_cell_dim.begin());
    v_last_max_cell_dim.resize(v_max_cell_dim.size());
    thrust::copy(v_max_cell_dim.begin(), v_max_cell_dim.end(), v_last_max_cell_dim.begin());
}

void nc_tree::index_points(s_vec<float> &v_eps_levels, s_vec<ull> &v_dims_mult) noexcept {
    uint size = n_coords;
    v_gpu_coords.resize(size*n_dim);
    thrust::copy(v_coords, v_coords+(size*n_dim), v_gpu_coords.begin());
    thrust::device_vector<float> v_device_min_bounds(v_min_bounds);
    thrust::device_vector<float> v_device_eps_levels(v_eps_levels);
    thrust::device_vector <ull> v_device_dims_mult(v_dims_mult);
    thrust::device_vector <ull> v_device_value_map;
    thrust::device_vector <uint> v_device_index_map;
    thrust::device_vector <uint> v_device_cell_ns;
    thrust::device_vector <uint> v_device_cell_begin;
    thrust::device_vector <uint> v_coord_indexes;
    thrust::device_vector <uint> v_unique_cnt;
    thrust::device_vector <uint> v_indexes;
    thrust::device_vector<float> v_min_cell_dim;
    thrust::device_vector<float> v_last_min_cell_dim;
    thrust::device_vector<float> v_max_cell_dim;
    thrust::device_vector<float> v_last_max_cell_dim;
    thrust::device_vector <uint> v_tmp;

    for (int l = 0; l < n_level; ++l) {
//        print_cuda_memory_usage();
        size = index_level_and_get_cells(v_gpu_coords, v_device_index_map,
                v_device_cell_ns, v_device_cell_begin, v_device_min_bounds,
                v_device_dims_mult, v_device_eps_levels, v_device_value_map,
                v_coord_indexes, v_unique_cnt, v_indexes, v_device_dims_mult, v_tmp,
                size, l, n_dim);

        cudaDeviceSynchronize();

        calculate_level_cell_bounds(v_gpu_coords, v_device_cell_begin, v_device_index_map,
                v_device_cell_ns, v_min_cell_dim, v_last_min_cell_dim, v_max_cell_dim,
                v_last_max_cell_dim, l, n_dim);

        cudaDeviceSynchronize();
        vv_index_map[l] = v_device_index_map;
        vv_cell_ns[l] = v_device_cell_ns;
        vv_cell_begin[l] = v_device_cell_begin;
        vv_min_cell_dim[l] = v_min_cell_dim;
        vv_max_cell_dim[l] = v_max_cell_dim;
    }
}

void nc_tree::determine_cell_labels() noexcept {
    thrust::device_vector<int> v_gpu_min_labels(v_gpu_begin.size());
    print_cuda_memory_usage();

    thrust::transform(v_gpu_leaf_cell_type.begin(), v_gpu_leaf_cell_type.end(), v_gpu_min_labels.begin(),
        [] __device__ (const int8_t val) {
            if (val == NO_CORES) return UNASSIGNED; return ROOT_CLUSTER;
        });

    discover_min_local_labels<<<CUDA_BLOCKS, CUDA_THREADS>>>(
            thrust::raw_pointer_cast(&v_gpu_coords[0]),
            thrust::raw_pointer_cast(&v_gpu_edges[0]),
            thrust::raw_pointer_cast(&v_gpu_min_labels[0]),
            thrust::raw_pointer_cast(&v_gpu_leaf_cell_type[0]),
            thrust::raw_pointer_cast(&v_gpu_is_core[0]),
            thrust::raw_pointer_cast(&v_gpu_index[0]),
            thrust::raw_pointer_cast(&v_gpu_begin[0]),
            thrust::raw_pointer_cast(&v_gpu_cell_ns[0]),
            v_gpu_edges.size()/2, n_dim, e2);

    flatten_label_graph<<<CUDA_BLOCKS, CUDA_THREADS>>>(
            thrust::raw_pointer_cast(&v_gpu_min_labels[0]),
            v_gpu_begin.size());

    update_cell_labels<<<CUDA_BLOCKS, CUDA_THREADS>>>(
            thrust::raw_pointer_cast(&v_gpu_coords[0]),
            thrust::raw_pointer_cast(&v_gpu_edges[0]),
            thrust::raw_pointer_cast(&v_gpu_min_labels[0]),
            thrust::raw_pointer_cast(&v_gpu_leaf_cell_type[0]),
            thrust::raw_pointer_cast(&v_gpu_is_core[0]),
            thrust::raw_pointer_cast(&v_gpu_index[0]),
            thrust::raw_pointer_cast(&v_gpu_begin[0]),
            thrust::raw_pointer_cast(&v_gpu_cell_ns[0]),
            thrust::raw_pointer_cast(&v_gpu_point_labels[0]),
            v_gpu_edges.size()/2, n_dim, e2);

    update_point_labels<<<CUDA_BLOCKS, CUDA_THREADS>>>(
            thrust::raw_pointer_cast(&v_gpu_point_labels[0]),
            thrust::raw_pointer_cast(&v_gpu_min_labels[0]),
            thrust::raw_pointer_cast(&v_gpu_index[0]),
            thrust::raw_pointer_cast(&v_gpu_begin[0]),
            thrust::raw_pointer_cast(&v_gpu_cell_ns[0]),
            thrust::raw_pointer_cast(&v_gpu_leaf_cell_type[0]),
            v_gpu_begin.size());

    cudaDeviceSynchronize();
    print_cuda_memory_usage();
    v_point_labels = v_gpu_point_labels;
}

void nc_tree::infer_types() noexcept {
    v_is_core = v_gpu_is_core;
}


void nc_tree::process_proximity_queries() noexcept {
    v_gpu_index = vv_index_map[0];
    v_gpu_begin = vv_cell_begin[0];
    v_gpu_cell_ns = vv_cell_ns[0];
    thrust::device_vector<uint> v_gpu_leaf_cell_np = v_gpu_cell_ns;
    v_gpu_edges = v_edges;
    v_gpu_leaf_cell_type.resize(v_gpu_begin.size(), UNKNOWN);
    thrust::device_vector<uint> v_gpu_point_np(n_coords, 0);
    v_gpu_is_core.resize(v_gpu_point_np.size(), 0);
    v_gpu_point_labels.resize(n_coords, UNASSIGNED);

//    std::cout << "processing size: " << v_gpu_begin.size() << std::endl;
//    std::cout << "Coords size: " << n_coords << std::endl;
//    std::cout << "m: " << m << std::endl;

    print_cuda_memory_usage();

//    uint max_points_in_cell = thrust::reduce(v_gpu_cell_ns.begin(), v_gpu_cell_ns.end(), 0, thrust::maximum<int>());
//    std::cout << "Max points in cell: " << max_points_in_cell << std::endl;

    fill_point_np<<<CUDA_BLOCKS, CUDA_THREADS>>>(
            thrust::raw_pointer_cast(&v_gpu_point_np[0]),
            thrust::raw_pointer_cast(&v_gpu_index[0]),
            thrust::raw_pointer_cast(&v_gpu_begin[0]),
            thrust::raw_pointer_cast(&v_gpu_cell_ns[0]),
            v_gpu_begin.size());

    process_edge_queries<<<CUDA_BLOCKS, CUDA_THREADS>>>(
            thrust::raw_pointer_cast(&v_gpu_coords[0]),
            thrust::raw_pointer_cast(&v_gpu_edges[0]),
            thrust::raw_pointer_cast(&v_gpu_index[0]),
            thrust::raw_pointer_cast(&v_gpu_begin[0]),
            thrust::raw_pointer_cast(&v_gpu_cell_ns[0]),
            thrust::raw_pointer_cast(&v_gpu_point_np[0]),
            v_gpu_edges.size()/2, m, n_dim, e2);

    fill_is_core<<<CUDA_BLOCKS, CUDA_THREADS>>>(
            thrust::raw_pointer_cast(&v_gpu_point_np[0]),
            thrust::raw_pointer_cast(&v_gpu_is_core[0]),
            m, v_gpu_point_np.size());


    update_cell_type<<<CUDA_BLOCKS, CUDA_THREADS>>>(
            thrust::raw_pointer_cast(&v_gpu_leaf_cell_type[0]),
            thrust::raw_pointer_cast(&v_gpu_is_core[0]),
            thrust::raw_pointer_cast(&v_gpu_index[0]),
            thrust::raw_pointer_cast(&v_gpu_begin[0]),
            thrust::raw_pointer_cast(&v_gpu_cell_ns[0]),
            v_gpu_begin.size());

    init_cell_label<<<CUDA_BLOCKS, CUDA_THREADS>>>(
            thrust::raw_pointer_cast(&v_gpu_point_labels[0]),
            thrust::raw_pointer_cast(&v_gpu_leaf_cell_type[0]),
            thrust::raw_pointer_cast(&v_gpu_is_core[0]),
            thrust::raw_pointer_cast(&v_gpu_index[0]),
            thrust::raw_pointer_cast(&v_gpu_begin[0]),
            thrust::raw_pointer_cast(&v_gpu_cell_ns[0]),
            v_gpu_begin.size());

    cudaDeviceSynchronize();
    print_cuda_memory_usage();
}