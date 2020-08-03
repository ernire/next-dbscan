//
// Created by Ernir Erlingsson on 6.5.2020.
//

#ifndef NEXT_DBSCAN_NEXT_DATA_OMP_H
#define NEXT_DBSCAN_NEXT_DATA_OMP_H


#include <mpi.h>
#include "next_util.h"
//#define MPI_ON

class next_data {
public:
    inline static bool is_in_reach(const float *min1, const float *max1, const float *min2, const float *max2,
            long const max_d, float const e) noexcept {
        for (auto d = 0; d < max_d; ++d) {
            if ((min2[d] > (max1[d] + e) || max2[d] < (min1[d] - e))) {
                return false;
            }
//            if ((min2[d] > (max1[d] + e) || min2[d] < (min1[d] - e)) &&
//                (min1[d] > (max2[d] + e) || min1[d] < (min2[d] - e)) &&
//                (max2[d] > (max1[d] + e) || max2[d] < (min1[d] - e)) &&
//                (max1[d] > (max2[d] + e) || max1[d] < (min2[d] - e))) {
//                return false;
//            }
        }
        return true;
    }

    static void calc_bounds(float *min_bounds, float *max_bounds, float *v_coords, long const n_dim,
            unsigned long const n_coords) noexcept {
        for (auto d = 0; d < n_dim; d++) {
            min_bounds[d] = INT32_MAX;
            max_bounds[d] = INT32_MIN;
        }
        #pragma omp parallel for reduction(max:max_bounds[:n_dim]) reduction(min:min_bounds[:n_dim])
        for (unsigned long i = 0; i < n_coords; i++) {
            auto index = i * n_dim;
            for (long d = 0; d < n_dim; d++) {
                if (v_coords[index + d] > max_bounds[d]) {
                    max_bounds[d] = v_coords[index + d];
                }
                if (v_coords[index + d] < min_bounds[d]) {
                    min_bounds[d] = v_coords[index + d];
                }
            }
        }
    }

    static unsigned long determine_data_boundaries(float *min_bounds, float *max_bounds, float *v_coords,
            long const n_dim, unsigned long const n_coords, float const lowest_e) noexcept {
        float max_limit = INT32_MIN;
        calc_bounds(min_bounds, max_bounds, v_coords, n_dim, n_coords);
        #ifdef MPI_ON
        std::vector<float> v_global_min_bounds(static_cast<unsigned long long>(n_dim));
        std::vector<float> v_global_max_bounds(static_cast<unsigned long long>(n_dim));
        MPI_Allreduce(&min_bounds[0], &v_global_min_bounds[0], n_dim, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&max_bounds[0], &v_global_max_bounds[0], n_dim, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
        std::copy(v_global_min_bounds.begin(), v_global_min_bounds.end(), &min_bounds[0]);
        std::copy(v_global_max_bounds.begin(), v_global_max_bounds.end(), &max_bounds[0]);
        #endif
        #pragma omp parallel for reduction(max: max_limit)
        for (auto d = 0; d < n_dim; d++) {
            if (max_bounds[d] - min_bounds[d] > max_limit)
                max_limit = max_bounds[d] - min_bounds[d];
        }
        return static_cast<unsigned long>(ceilf(logf(max_limit / lowest_e) / logf(2))) + 1;
    }

    static void reorder_coords(s_vec<float>::iterator source_begin,
            s_vec<long>::iterator index_begin,
            s_vec<long>::iterator index_end,
            s_vec<float>::iterator sink_begin,
            long const n_dim) {
        while (index_begin != index_end) {
            std::copy(source_begin+(*index_begin)*n_dim, (source_begin+((*index_begin)*n_dim))+n_dim, sink_begin);
            sink_begin += n_dim;
            ++index_begin;
        }
    }
};


#endif //NEXT_DBSCAN_NEXT_DATA_OMP_H
