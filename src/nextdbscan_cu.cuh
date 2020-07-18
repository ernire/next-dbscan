//
// Created by Ernir Erlingsson on 5.6.2020.
//

#ifndef NEXT_DBSCAN_NEXTDBSCAN_CU_CUH
#define NEXT_DBSCAN_NEXTDBSCAN_CU_CUH

//#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "nextdbscan.h"

// V100
static const int CUDA_BLOCKS = 128;
static const int CUDA_THREADS = 1024;

template <typename T>
using s_vec = thrust::host_vector<T>;
template <typename T>
using d_vec = thrust::host_vector<thrust::host_vector<T>>;

template<class T, class O>
void _atomic_op(T* address, T value, O op) {
    /*
    T previous = __sync_fetch_and_add(address, 0);

    while (op(value, previous)) {
        if  (__sync_bool_compare_and_swap(address, previous, value)) {
            break;
        } else {
            previous = __sync_fetch_and_add(address, 0);
        }
    }
     */
}

class nextdbscan_cu {

};


#endif //NEXT_DBSCAN_NEXTDBSCAN_CU_CUH
