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

__global__ void determine_min_max_old(const float* v_coords, const uint* v_index_map, const uint* v_begin,
        const uint* v_ns, float* v_min_input, float* v_max_input, float* v_min_output,
        float* v_max_output, const uint size, const uint max_d, const uint l) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint index;
    while (globalIdx < size) {
        uint i = globalIdx;
        uint begin = v_begin[i];
        index = v_index_map[begin] * max_d;

        if (l == 0) {
            for (uint d = 0; d < max_d; ++d) {
                v_min_output[i*max_d+d] = v_coords[index + d];
                v_max_output[i*max_d+d] = v_coords[index + d];
            }
        } else {
            for (uint d = 0; d < max_d; ++d) {
                v_min_output[i*max_d+d] = v_min_input[index + d];
                v_max_output[i*max_d+d] = v_max_input[index + d];
            }
        }
        for (uint j = 1; j < v_ns[i]; j++) {
            index = v_index_map[begin+j] * max_d;
            if (l == 0) {
                for (uint d = 0; d < max_d; ++d) {
                    if (v_coords[index + d] < v_min_output[i * max_d + d]) {
                        v_min_output[i * max_d + d] = v_coords[index + d];
                    }
                    if (v_coords[index + d] > v_max_output[i * max_d + d]) {
                        v_max_output[i * max_d + d] = v_coords[index + d];
                    }
                }
            } else {
                for (uint d = 0; d < max_d; ++d) {
                    if (v_min_input[index + d] < v_min_output[i * max_d + d]) {
                        v_min_output[i * max_d + d] = v_min_input[index + d];
                    }
                    if (v_max_input[index + d] > v_max_output[i * max_d + d]) {
                        v_max_output[i * max_d + d] = v_max_input[index + d];
                    }
                }
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
/*
         size_t free_byte ;

        size_t total_byte ;

        cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;

        if ( cudaSuccess != cuda_status ){

            printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );

            exit(1);

        }



        double free_db = (double)free_byte ;

        double total_db = (double)total_byte ;

        double used_db = total_db - free_db ;

        printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",

            used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
 */

// CUDA does not allow kernel parents to be private/protected members of a class
uint nextdbscan_cuda::index_level_and_get_cells(thrust::device_vector<float> &v_coords,
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
    // Start with 6 GB
    int result = 0;
    // 4 GB
    v_value_map.resize(size);
    if (l == 0) {
        // 2 GB
        v_coord_indexes.resize(size);
        thrust::sequence(v_coord_indexes.begin(), v_coord_indexes.end());
    }
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
    std::cout << "CHECKPOINT 1" << std::endl;
    print_cuda_memory_usage();
    if (l == 0) {
        thrust::sort_by_key(v_value_map.begin(), v_value_map.begin() + size, v_coord_indexes.begin());
    } else {
        v_device_index_map.resize(size);
        thrust::sequence(v_device_index_map.begin(), v_device_index_map.end());
        thrust::sort_by_key(v_value_map.begin(), v_value_map.begin() + size, v_device_index_map.begin());
    }
    std::cout << "CHECKPOINT 1.5" << std::endl;
    // 2 GB
    v_unique_cnt.resize(size);
    std::cout << "CHECKPOINT 2" << std::endl;
    print_cuda_memory_usage();
    thrust::fill(v_unique_cnt.begin(), v_unique_cnt.end(), 0);
    count_unique_groups<<<CUDA_BLOCKS,CUDA_THREADS>>>(
            thrust::raw_pointer_cast(&v_value_map[0]),
            thrust::raw_pointer_cast(&v_unique_cnt[0]),
            size);
    v_value_map.clear();
    v_value_map.shrink_to_fit();
    result = thrust::count_if(v_unique_cnt.begin(), v_unique_cnt.begin() + size,
        [] __device__ (auto val) { return val > 0; });
    v_device_cell_ns.resize(result);
    thrust::copy_if(v_unique_cnt.begin(), v_unique_cnt.begin() + size, v_device_cell_ns.begin(),
            [] __device__ (auto val) { return val > 0; });
    v_device_cell_begin.resize(v_device_cell_ns.size());



    return result;
    /*
    if (l == 0) {
        std::cout << "setting size: " << size << std::endl;
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
    print_cuda_memory_usage();
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
     */
}
//uint nextdbscan_cuda::index_level_and_get_cells(thrust::device_vector<float> &v_coords,
//        thrust::device_vector<uint> &v_device_index_map,
//        thrust::device_vector<uint> &v_device_cell_ns,
//        thrust::device_vector<uint> &v_device_cell_begin,
//        thrust::device_vector<float> &v_min_bounds,
//        thrust::device_vector<ull> &v_device_dims_mult,
//        thrust::device_vector<float> &v_level_eps,
//        thrust::device_vector<ull> &v_value_map,
//        thrust::device_vector<uint> &v_coord_indexes,
//        thrust::device_vector<uint> &v_unique_cnt,
//        thrust::device_vector<uint> &v_indexes,
//        thrust::device_vector<ull> &v_dims_mult,
//        thrust::device_vector<uint> &v_tmp,
//        const uint size, const uint l, const uint max_d) noexcept {
//    if (l == 0) {
//        std::cout << "setting size: " << size << std::endl;
//        v_value_map.resize(size);
//        print_cuda_memory_usage();
//        v_unique_cnt.resize(size);
//        print_cuda_memory_usage();
//        v_coord_indexes.resize(size);
//        thrust::sequence(v_coord_indexes.begin(), v_coord_indexes.end());
//        print_cuda_memory_usage();
//        v_indexes.resize(size);
//        thrust::sequence(v_indexes.begin(), v_indexes.end());
//        print_cuda_memory_usage();
//    }
//    v_device_index_map.resize(size);
//    thrust::sequence(v_device_index_map.begin(), v_device_index_map.end());
//    thrust::fill(v_unique_cnt.begin(), v_unique_cnt.end(), 0);
//    index_kernel<<<CUDA_BLOCKS ,CUDA_THREADS>>>(
//            thrust::raw_pointer_cast(&v_coords[0]),
//                    thrust::raw_pointer_cast(&v_coord_indexes[0]),
//                    thrust::raw_pointer_cast(&v_value_map[0]),
//                    thrust::raw_pointer_cast(&v_min_bounds[0]),
//                    thrust::raw_pointer_cast(&v_dims_mult[l * max_d]),
//                    size,
//                    max_d,
//                    v_level_eps[l]
//    );
//    print_cuda_memory_usage();
//    thrust::sort_by_key(v_value_map.begin(), v_value_map.begin() + size, v_device_index_map.begin());
//    v_tmp.resize(size);
//    auto ptr_indexes = thrust::raw_pointer_cast(&v_coord_indexes[0]);
//    thrust::transform(thrust::device,
//            v_device_index_map.begin(),
//            v_device_index_map.end(),
//            v_tmp.begin(),
//            [=] __device__ (const auto val) { return ptr_indexes[val]; });
//    count_unique_groups<<<CUDA_BLOCKS,CUDA_THREADS>>>(
//            thrust::raw_pointer_cast(&v_value_map[0]),
//                    thrust::raw_pointer_cast(&v_unique_cnt[0]),
//                    size);
//    int result = thrust::count_if(v_unique_cnt.begin(), v_unique_cnt.begin() + size,
//    [] __device__ (auto val) { return val > 0; });
//    v_device_cell_ns.resize(result);
//    v_device_cell_begin.resize(v_device_cell_ns.size());
//    v_coord_indexes.resize(v_device_cell_ns.size());
//    thrust::copy_if(v_unique_cnt.begin(), v_unique_cnt.begin() + size, v_device_cell_ns.begin(),
//            [] __device__ (auto val) { return val > 0; });
//    auto ptr = thrust::raw_pointer_cast(&v_unique_cnt[0]);
//    thrust::copy_if(v_indexes.begin(), v_indexes.begin() + size, v_device_cell_begin.begin(),
//            [=] __device__ (auto val) { return ptr[val] > 0; });
//    thrust::copy_if(thrust::device, v_tmp.begin(), v_tmp.end(), v_indexes.begin(),
//            v_coord_indexes.begin(),
//            [=] __device__ (auto val) { return ptr[val] > 0; });
//    return result;
//}

void nextdbscan_cuda::calculate_level_cell_bounds(
        thrust::device_vector<float> &v_coords,
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

void nextdbscan_cuda::index_points(s_vec<float> &v_coords,
        s_vec<float> &v_eps_levels,
        s_vec<ull> &v_dims_mult,
        s_vec<float> &v_min_bounds,
        d_vec<uint> &vv_index_map,
        d_vec<uint> &vv_cell_begin,
        d_vec<uint> &vv_cell_ns,
        d_vec<float> &vv_min_cell_dim,
        d_vec<float> &vv_max_cell_dim,
        const uint max_d, const uint n_threads,
        const uint max_levels, uint size) noexcept {
    thrust::device_vector<float> v_device_coords(v_coords);
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
    for (int l = 0; l < max_levels; ++l) {
        std::cout << "l: " << l << std::endl;
        print_cuda_memory_usage();
        size = index_level_and_get_cells(v_device_coords, v_device_index_map,
                v_device_cell_ns, v_device_cell_begin, v_device_min_bounds,
                v_device_dims_mult, v_device_eps_levels, v_device_value_map,
                v_coord_indexes, v_unique_cnt, v_indexes, v_device_dims_mult, v_tmp,
                size, l, max_d);
        std::cout << "l: " << l << " size: " << size << std::endl;
        calculate_level_cell_bounds(v_device_coords, v_device_cell_begin, v_device_index_map,
                v_device_cell_ns, v_min_cell_dim, v_last_min_cell_dim, v_max_cell_dim,
                v_last_max_cell_dim, l, max_d);
        vv_index_map[l] = v_device_index_map;
        vv_cell_ns[l] = v_device_cell_ns;
        vv_cell_begin[l] = v_device_cell_begin;
        vv_min_cell_dim[l] = v_min_cell_dim;
        vv_max_cell_dim[l] = v_max_cell_dim;
    }
}