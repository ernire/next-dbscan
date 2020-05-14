//
// Created by Ernir Erlingsson on 7.5.2020.
//

#include <iostream>
#include <omp.h>
#include <numeric>
#include <cassert>
#include "cell_processor.h"
#include "next_util.h"

inline bool dist_leq(const float *coord1, const float *coord2, long const max_d, float const e2) noexcept {
    float tmp = 0;
//    #pragma omp simd
    for (auto d = 0; d < max_d; d++) {
        tmp += (coord1[d] - coord2[d]) * (coord1[d] - coord2[d]);
    }
    return tmp <= e2;
}

bool are_core_connected(s_vec<float> &v_coords, s_vec<long> &v_cell_begin,
        s_vec<long> &v_cell_ns, s_vec<uint8_t> &v_is_core,
        long const c1, long const c2, long const n_dim, const float e2) noexcept {
    for (auto k1 = 0; k1 < v_cell_ns[c1]; ++k1) {
        if (!v_is_core[v_cell_begin[c1]+k1]) {
            continue;
        }
        for (auto k2 = 0; k2 < v_cell_ns[c2]; ++k2) {
            if (!v_is_core[v_cell_begin[c2]+k2]) {
                continue;
            }
            if (dist_leq(&v_coords[(v_cell_begin[c1]+k1)*n_dim], &v_coords[(v_cell_begin[c2]+k2)*n_dim], n_dim, e2)) {
                return true;
            }
        }
    }
    return false;
}

bool flatten_labels(std::vector<long> &v_local_min_labels) noexcept {
    bool is_update = false;
    #pragma omp parallel for schedule(guided)
    for (auto i = 0; i < v_local_min_labels.size(); ++i) {
        if (v_local_min_labels[i] == i || v_local_min_labels[i] == UNASSIGNED)
            continue;
        auto label = v_local_min_labels[i];
        bool update = false;
        while (v_local_min_labels[label] != label) {
            label = v_local_min_labels[label];
            update = true;
        }
        if (update) {
            v_local_min_labels[i] = label;
            is_update = true;
        }
    }
    return is_update;
}

void cell_processor::determine_cell_labels(s_vec<float> &v_coords, s_vec<long> v_edges, nc_tree_new &nc) noexcept {
    std::vector<long> v_local_min_labels(nc.vv_cell_begin[0].size());
    std::iota(v_local_min_labels.begin(), v_local_min_labels.end(), 0);

    #pragma omp parallel for
    for (uint i = 0; i < v_local_min_labels.size(); ++i) {
        if (v_leaf_cell_type[i] == NO_CORES)
            v_local_min_labels[i] = UNASSIGNED;
    }
    #pragma omp parallel for schedule(dynamic)
    for (uint i = 0; i < v_edges.size(); i += 2) {
        auto conn = v_edge_conn[i/2];
        if (conn == NOT_CONNECTED) {
            continue;
        }
        auto c1 = v_edges[i];
        auto c2 = v_edges[i+1];
        long c_lower, c_higher;
        if (v_leaf_cell_type[c1] != NO_CORES && v_leaf_cell_type[c2] != NO_CORES) {
            // both either AC or SC
            c_lower = (c1 < c2)? c1 : c2;
            c_higher = (c1 < c2)? c2 : c1;
            if (v_local_min_labels[c_higher] <= v_local_min_labels[c_lower]) {
                continue;
            }
            if (conn == UNKNOWN || conn == PARTIALLY_CONNECTED) {
                if (are_core_connected(v_coords, nc.vv_cell_begin[0], nc.vv_cell_ns[0],
                        v_is_core, c1, c2, nc.n_dim, nc.e2)) {
                    v_edge_conn[i/2] = FULLY_CONNECTED;
                    _atomic_op(&v_local_min_labels[c_higher], v_local_min_labels[c_lower], std::less<>());
                } else {
                    v_edge_conn[i/2] = NOT_CONNECTED;
                }
            } else if (conn == FULLY_CONNECTED) {
                // We know they are connected
                _atomic_op(&v_local_min_labels[c_higher], v_local_min_labels[c_lower], std::less<>());
            }
        } else if ((v_leaf_cell_type[c1] != NO_CORES || v_leaf_cell_type[c2] != NO_CORES)
                  && (v_leaf_cell_type[c1] == NO_CORES || v_leaf_cell_type[c2] == NO_CORES)) {
            // One AC/SC and one NC
            // No need to reprocess
            v_edge_conn[i/2] = NOT_CONNECTED;
            if (conn == FULLY_CONNECTED) {
                c_lower = (v_leaf_cell_type[c1] != NO_CORES)? c1 : c2;
                c_higher = (v_leaf_cell_type[c1] != NO_CORES)? c2 : c1;
                if (v_local_min_labels[c_higher] != UNASSIGNED)
                    continue;
                // Permitted race condition (DBSCAN border point labels are not deterministic)
                v_local_min_labels[c_higher] = c_lower;
            } else {
                // else handle border cell partials
                c_lower = (v_leaf_cell_type[c1] != NO_CORES)? c1 : c2;
                c_higher = (v_leaf_cell_type[c1] != NO_CORES)? c2 : c1;
                if (v_local_min_labels[c_higher] != UNASSIGNED)
                    continue;
                for (long k1 = 0; k1 < nc.vv_cell_ns[0][c_lower]; ++k1) {
                    auto p1 = nc.vv_cell_begin[0][c_lower] + k1;
                    if (!v_is_core[p1])
                        continue;
                    for (long k2 = 0; k2 < nc.vv_cell_ns[0][c_higher]; ++k2) {
                        auto p2 = nc.vv_cell_begin[0][c_higher] + k2;
                        if (v_point_labels[p2] != UNASSIGNED)
                            continue;
                        if (dist_leq(&v_coords[p1 * nc.n_dim], &v_coords[p2 * nc.n_dim], nc.n_dim, nc.e2)) {
                            v_point_labels[p2] = p1;
                        }
                    }
                }
            }
        } else {
            v_edge_conn[i/2] = NOT_CONNECTED;
        }
    }

    flatten_labels(v_local_min_labels);
    bool loop = true;
    while (loop) {
//        uint cnt = 0;
        #pragma omp parallel for schedule(dynamic)
        for (uint i = 0; i < v_edges.size(); i += 2) {
            auto conn = v_edge_conn[i / 2];
            if (conn == NOT_CONNECTED) {
                continue;
            }
            auto c1 = v_edges[i];
            auto c2 = v_edges[i + 1];
            long c_lower, c_higher;
//            assert(v_leaf_cell_type[c1] != NO_CORES && v_leaf_cell_type[c2] != NO_CORES);
//            if (v_leaf_cell_type[c1] != NO_CORES && v_leaf_cell_type[c2] != NO_CORES) {
            if (v_local_min_labels[c1] != v_local_min_labels[c2]) {
                if (conn == UNKNOWN || conn == PARTIALLY_CONNECTED) {
                    if (are_core_connected(v_coords, nc.vv_cell_begin[0],
                            nc.vv_cell_ns[0],v_is_core, c1, c2, nc.n_dim, nc.e2)) {
                        v_edge_conn[i / 2] = FULLY_CONNECTED;
                    } else {
                        v_edge_conn[i / 2] = NOT_CONNECTED;
                    }
                }
                if (conn == FULLY_CONNECTED) {
                    c_lower = (v_local_min_labels[c1] < v_local_min_labels[c2]) ?
                              v_local_min_labels[c1] : v_local_min_labels[c2];
                    c_higher = (v_local_min_labels[c1] < v_local_min_labels[c2]) ?
                               v_local_min_labels[c2] : v_local_min_labels[c1];
                    _atomic_op(&v_local_min_labels[c_higher], c_lower, std::less<>());
                }
            }
//            }
        }
//        std::cout << "loop relabel cnt: " << cnt << std::endl;
        if (!flatten_labels(v_local_min_labels)) {
            loop = false;
        }
    }


    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < v_local_min_labels.size(); ++i) {
        if (v_local_min_labels[i] == i || v_local_min_labels[i] == UNASSIGNED)
            continue;
        if (v_leaf_cell_type[i] != NO_CORES) {
            auto p = nc.vv_cell_begin[0][i];
            if (p != v_point_labels[p]) {
                p = v_point_labels[p];
            }
            v_point_labels[p] = v_point_labels[nc.vv_cell_begin[0][v_local_min_labels[i]]];
        } else {
            auto label = v_point_labels[nc.vv_cell_begin[0][v_local_min_labels[i]]];
            for (uint j = 0; j < nc.vv_cell_ns[0][i]; ++j) {
                v_point_labels[nc.vv_cell_begin[0][i]+j] = label;
            }
        }
    }
}

//inline void update_to_ac(s_vec<long> &v_cell_ns, s_vec<long> &v_cell_begin,
//        s_vec<uint8_t> &is_core, s_vec<uint8_t> &v_types,
//        long const c) noexcept {
//    v_types[c] = ALL_CORES;
//    for (uint j = 0; j < v_cell_ns[c]; ++j) {
//        is_core[v_cell_begin[c] + j] = 1;
//    }
//}

void update_type(s_vec<long> &v_cell_ns,
        s_vec<long> &v_cell_begin, s_vec<long> &v_cell_nps, s_vec<long> &v_point_nps,
        s_vec<uint8_t> &is_core, s_vec<uint8_t> &v_types, const long c, const long m) noexcept {
    if (v_cell_nps[c] >= m) {
        v_types[c] = ALL_CORES;
        for (uint j = 0; j < v_cell_ns[c]; ++j) {
            is_core[v_cell_begin[c] + j] = 1;
        }
//        update_to_ac(v_cell_ns, v_cell_begin, is_core, v_types, c);
    }
    bool all_cores = true;
    bool some_cores = false;
    for (long j = 0; j < v_cell_ns[c]; ++j) {
        long p = v_cell_begin[c] + j;
        if (is_core[p])
            continue;
        if (v_cell_nps[c] + v_point_nps[p] >= m) {
            is_core[p] = 1;
            some_cores = true;
        } else {
            all_cores = false;
        }
    }
    if (all_cores) {
        v_types[c] = ALL_CORES;
    } else if (some_cores) {
        v_types[c] = SOME_CORES;
    }
}

void cell_processor::infer_types(nc_tree_new &nc) noexcept {
    v_is_core.resize(nc.n_coords, UNKNOWN);
    v_point_labels.resize(nc.n_coords, UNASSIGNED);
    #pragma omp parallel for schedule(guided)
    for (long i = 0; i < nc.vv_cell_ns[0].size(); ++i) {
        update_type(nc.vv_cell_ns[0], nc.vv_cell_begin[0],
                v_leaf_cell_np, v_point_np, v_is_core, v_leaf_cell_type, i, nc.m);
        if (v_leaf_cell_type[i] != UNKNOWN) {
            auto begin = nc.vv_cell_begin[0][i];
            long core_p = UNASSIGNED;
            for (uint j = 0; j < nc.vv_cell_ns[0][i]; ++j) {
                if (core_p != UNASSIGNED) {
                    v_point_labels[begin+j] = core_p;
                } else if (v_is_core[begin+j]) {
                    core_p = begin+j;
                    v_point_labels[core_p] = core_p;
                    for (uint k = 0; k < j; ++k) {
                        v_point_labels[begin+k] = core_p;
                    }
                }
            }
        }
        if (v_leaf_cell_type[i] == UNKNOWN) {
            v_leaf_cell_type[i] = NO_CORES;
        }
    }
}

uint8_t process_pair_proximity(s_vec<float> &v_coords,
        s_vec<long> &v_point_nps,
        s_vec<long> &v_cell_ns,
        std::vector<long> &v_range_cnt,
        s_vec<long> &v_cell_nps,
        const long max_d, const float e2, const long m,
        const long c1, const long begin1, const long c2, const long begin2) noexcept {
    uint8_t are_connected = NOT_CONNECTED;
    auto size1 = v_cell_ns[c1];
    auto size2 = v_cell_ns[c2];
    std::fill(v_range_cnt.begin(), std::next(v_range_cnt.begin(), size1+size2), 0);
    for (auto k1 = 0; k1 < size1; ++k1) {
        for (long k2 = 0; k2 < size2; ++k2) {
            if (dist_leq(&v_coords[(begin1 + k1) * max_d], &v_coords[(begin2+k2) * max_d], max_d, e2)) {
                ++v_range_cnt[k1];
                ++v_range_cnt[size1+k2];
            }
        }
    }
    auto hits = 0;
    for (auto i = 0; i < size1+size2; ++i) {
        hits += v_range_cnt[i];
    }
    if (hits == size1*size2*2) {
        if (v_cell_nps[c1] < m) {
            #pragma omp atomic
            v_cell_nps[c1] += v_cell_ns[c2];
        }
        if (v_cell_nps[c2] < m) {
            #pragma omp atomic
            v_cell_nps[c2] += v_cell_ns[c1];
        }
        are_connected = FULLY_CONNECTED;
    } else if (hits > 0) {
        if (v_cell_nps[c1] < m) {
            long min = INT32_MAX;
            for (uint k1 = 0; k1 < size1; ++k1) {
                if (v_range_cnt[k1] < min)
                    min = v_range_cnt[k1];
            }
            if (min > 0) {
                #pragma omp atomic
                v_cell_nps[c1] += min;
            }
            for (uint k1 = 0; k1 < size1; ++k1) {
                if (v_range_cnt[k1] - min > 0) {
                    #pragma omp atomic
                    v_point_nps[begin1 + k1] += v_range_cnt[k1] - min;
                }
            }
        }
        if (v_cell_nps[c2] < m) {
            long min = INT32_MAX;
            for (long k2 = size1; k2 < size1+size2; ++k2) {
                if (v_range_cnt[k2] < min)
                    min = v_range_cnt[k2];
            }
            if (min > 0) {
                #pragma omp atomic
                v_cell_nps[c2] += min;
            }
            for (long k2 = 0; k2 < size2; ++k2) {
                if (v_range_cnt[k2+size1] - min > 0) {
                    #pragma omp atomic
                    v_point_nps[begin2 + k2] += v_range_cnt[k2 + size1] - min;
                }
            }
        }
        are_connected = PARTIALLY_CONNECTED;
    }
    return are_connected;
}

void cell_processor::process_edges(s_vec<float> &v_coords, s_vec<long> v_edges, nc_tree_new &nc) noexcept {
    v_leaf_cell_np = nc.vv_cell_ns[0];
    v_leaf_cell_type.resize(v_leaf_cell_np.size(), UNKNOWN);
    v_point_np.resize(nc.n_coords, 0);
    v_edge_conn.resize(v_edges.size()/2, UNKNOWN);

    #pragma omp parallel for
    for (long i = 0; i < v_edges.size(); i += 2) {
        long c1 = v_edges[i];
        long c2 = v_edges[i+1];
        if (v_leaf_cell_np[c1] < nc.m) {
            #pragma omp atomic
            v_leaf_cell_np[c1] += nc.vv_cell_ns[0][c2];
        }
        if (v_leaf_cell_np[c2] < nc.m) {
            #pragma omp atomic
            v_leaf_cell_np[c2] += nc.vv_cell_ns[0][c1];
        }
    }

    long max_points_in_cell = 0;
    #pragma omp parallel for reduction(max: max_points_in_cell)
    for (long i = 0; i < nc.vv_cell_ns[0].size(); ++i) {
        if (v_leaf_cell_np[i] < nc.m) {
            v_leaf_cell_type[i] = NO_CORES;
        } else if (nc.vv_cell_ns[0][i] >= nc.m) {
            v_leaf_cell_type[i] = ALL_CORES;
        }
        if (nc.vv_cell_ns[0][i] > max_points_in_cell) {
            max_points_in_cell = nc.vv_cell_ns[0][i];
        }
    }
    // reset
    v_leaf_cell_np = nc.vv_cell_ns[0];
    std::vector<std::vector<long>> vv_range_counts((unsigned long) n_threads);
//    std::cout << "max points in cell: " << max_points_in_cell << std::endl;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        // TODO
        vv_range_counts[tid].resize(static_cast<unsigned long>(max_points_in_cell * 2));
        #pragma omp for schedule(dynamic, 2)
        for (uint i = 0; i < v_edges.size(); i += 2) {
            auto c1 = v_edges[i];
            auto c2 = v_edges[i+1];
            if (v_leaf_cell_np[c1] >= nc.m && v_leaf_cell_np[c2] >= nc.m) {
                continue;
            }
            if (v_leaf_cell_type[c1] == NO_CORES && v_leaf_cell_type[c2] == NO_CORES) {
                v_edge_conn[i/2] = NOT_CONNECTED;
                continue;
            }
            auto begin1 = nc.vv_cell_begin[0][c1];
            auto begin2 = nc.vv_cell_begin[0][c2];
            uint8_t are_connected = process_pair_proximity(v_coords, v_point_np,
                    nc.vv_cell_ns[0], vv_range_counts[tid], v_leaf_cell_np,
                    nc.n_dim, nc.e2, nc.m, c1, begin1, c2, begin2);
            v_edge_conn[i/2] = are_connected;
        }
    }

}

void cell_processor::get_result_meta(long &n_cores, long &n_noise, long &clusters) noexcept {
    long sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < v_is_core.size(); ++i) {
        if (v_is_core[i])
            ++sum;
    }
    n_cores = sum;

    sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < v_point_labels.size(); ++i) {
        if (v_point_labels[i] == UNDEFINED)
            ++sum;
    }
    n_noise = sum;

    sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < v_point_labels.size(); ++i) {
        if (v_point_labels[i] == i)
            ++sum;
    }
    clusters = sum;
}

void cell_processor::process_edges(s_vec<float> &v_coords, s_vec<long> v_edges, nc_tree_new &nc1,
        nc_tree_new &nc2) noexcept {

}

static unsigned long set_partition_level(std::vector<unsigned long> &v_ordered_dim,
        std::vector<float> &v_min_bounds,
        std::vector<float> &v_max_bounds,
        long &level,
        long const min_sample_size,
        float const e_lowest) {
    unsigned long max_cells = 1;
    float e_lvl = 0;
    while (level > 0 && max_cells < min_sample_size) {
        --level;
        e_lvl = (e_lowest * powf(2, level));
        max_cells = 1;
        for (auto const &d : v_ordered_dim) {
            max_cells *= ((v_max_bounds[d] - v_min_bounds[d]) / e_lvl) + 1;
            if  (max_cells > min_sample_size) {
                return max_cells;
            }
        }
    }
}

void cell_processor::partition_data(std::vector<float> &v_coords,
        s_vec<float> &v_min_bounds,
        s_vec<float> &v_max_bounds,
        long const n_partitions,
        unsigned long const n_coords,
        unsigned long const n_dim,
        long const n_level,
        float const e_lowest,
        s_vec<unsigned long> &v_part_coord,
        s_vec<unsigned long> &v_part_offset,
        s_vec<unsigned long> &v_part_size) noexcept {
    std::cout << "n_coords: " << n_coords << std::endl;
    long const min_sample_size = static_cast<long>(ceil(n_partitions * n_dim * n_dim * log10(n_coords)));
    std::cout << "sample size: " << min_sample_size << std::endl;

    auto level = n_level - 1;
//        long max_cells = 1;
//        double e_lvl = 0;
    s_vec<unsigned long> v_dim_cell_size(n_dim);
    std::iota(v_dim_cell_size.begin(), v_dim_cell_size.end(), 0);
    auto v_ordered_dim = v_dim_cell_size;

    unsigned long max_cells = set_partition_level(v_ordered_dim, v_min_bounds, v_max_bounds, level,
            min_sample_size, e_lowest);
    std::cout << "sufficient level: " << level << " of " << n_level << std::endl;
    std::cout << "sufficient level max cells: " << max_cells << std::endl;
//    n_level = level;
    float const e_lvl = (e_lowest * powf(2, level));
    for (auto &d : v_dim_cell_size) {
        d = static_cast<unsigned long>(((v_max_bounds[d] - v_min_bounds[d]) / e_lvl) + 1);
    }
    next_util::print_vector("dim sizes: ", v_dim_cell_size);
    std::vector<long> v_dim_cell_index(n_coords*n_dim, 0);
    std::vector<long> v_dim_offset(n_dim);
    std::iota(v_dim_offset.begin(), v_dim_offset.end(), 0);
    for (auto &d : v_dim_offset) {
        d *= n_coords;
    }
    for (size_t i = 0; i < n_coords; ++i) {
        for (auto const &d : v_ordered_dim) {
            assert(v_dim_cell_index[v_dim_offset[d]+i] == 0);
            v_dim_cell_index[v_dim_offset[d]+i] = static_cast<long>((v_coords[(i*n_dim)+d] - v_min_bounds[d]) / e_lvl);
        }
    }
    s_vec<long> v_dim_cell_cnt;
    s_vec<long> v_dim_cell_cnt_nz;
    s_vec<float> v_dim_entropy(n_dim, 0);
    // cell dimension distributions
    for (auto const &d : v_ordered_dim) {
        v_dim_cell_cnt.resize(v_dim_cell_size[d]);
        std::fill(v_dim_cell_cnt.begin(), v_dim_cell_cnt.end(), 0);
        auto offset = v_dim_offset[d];
        #pragma omp parallel for
        for (size_t i = 0; i < n_coords; ++i) {
            ++v_dim_cell_cnt[v_dim_cell_index[offset+i]];
        }
        // TODO MPI merge
//        next_util::print_vector("dim value: " , v_dim_cell_cnt);
        v_dim_cell_cnt_nz.resize(v_dim_cell_cnt.size());
        auto const it = std::copy_if (v_dim_cell_cnt.begin(), v_dim_cell_cnt.end(), v_dim_cell_cnt_nz.begin(),
                [](auto const &val){return val > 0;} );
        v_dim_cell_cnt_nz.resize(static_cast<unsigned long>(std::distance(v_dim_cell_cnt_nz.begin(), it)));
        //        std::cout << "dim: " << d << " unique count: " << v_dim_cell_cnt_nz.size() << std::endl;
        float entropy = 0;
        auto dim_sum = next_util::sum_array_omp(&v_dim_cell_cnt_nz[0], v_dim_cell_cnt_nz.size());

        #pragma omp parallel for reduction(+:entropy)
        for (size_t i = 0; i < v_dim_cell_cnt_nz.size(); ++i) {
            auto p = (double)v_dim_cell_cnt_nz[i] / dim_sum;
            entropy -= p*log2(p);
        }
        //        std::cout << "dim: " << d << " entropy: " << entropy << std::endl;
        v_dim_entropy[d] = entropy;
    }
    std::sort(v_ordered_dim.begin(), v_ordered_dim.end(), [&v_dim_entropy] (auto const &d1, auto const &d2) -> bool {
        return v_dim_entropy[d1] > v_dim_entropy[d2];
    });

    next_util::print_vector("dim order: ", v_ordered_dim);

    // Find the number of dimensions needed
    max_cells = 1;
    unsigned long dim_size = 0;
    for (auto const &d : v_ordered_dim) {
        max_cells *= v_dim_cell_size[d];
        ++dim_size;
        if (max_cells > min_sample_size) {
            break;
        }
    }
    v_ordered_dim.resize(dim_size);
    std::vector<long> v_cell_cnt(max_cells, 0);
    std::cout << "processed dim size: " << dim_size << std::endl;
    std::vector<unsigned long> v_cell_index(n_coords);
    s_vec<unsigned long> v_cell_size_mul(v_ordered_dim.size());
    v_cell_size_mul[0] = 1;
    for (size_t d = 1; d < v_cell_size_mul.size(); ++d) {
//        std::cout << v_dim_cell_size[v_ordered_dim[d-1]] << " ";
        v_cell_size_mul[d] = v_cell_size_mul[d-1] * v_dim_cell_size[v_ordered_dim[d-1]];
    }
//    std::cout << std::endl;

    // Finally the indexing
    for (size_t i = 0; i < n_coords; ++i) {
        unsigned long cell_index = 0;
        auto d_cnt = 0;
        for (auto const &d : v_ordered_dim) {
            assert(d >= 0 && d < n_dim);
            assert(v_cell_size_mul[d_cnt] > 0);
            cell_index += ((v_coords[(i*n_dim)+d] - v_min_bounds[d]) / e_lvl) * v_cell_size_mul[d_cnt];
            ++d_cnt;
        }
        assert(cell_index < v_cell_cnt.size());
        v_cell_index[i] = cell_index;
        ++v_cell_cnt[cell_index];
    }
    // TODO MPI merge

    std::vector<long> v_ordered_cell_cnt(v_cell_cnt.size());
    // TODO iota omp
    std::iota(v_ordered_cell_cnt.begin(), v_ordered_cell_cnt.end(), 0);
    std::sort(v_ordered_cell_cnt.begin(), v_ordered_cell_cnt.end(), [&] (auto const &i1, auto const &i2) -> bool {
        return v_cell_cnt[i1] > v_cell_cnt[i2];
    });
    std::cout << "total cells: " << v_cell_cnt.size() << std::endl;
    auto new_end = std::lower_bound(v_ordered_cell_cnt.begin(), v_ordered_cell_cnt.end(), 0,
            [&] (auto const &i1, auto const &val) -> bool { return v_cell_cnt[i1] > val; });
    v_ordered_cell_cnt.resize(static_cast<unsigned long>(std::distance(v_ordered_cell_cnt.begin(), new_end)));
    std::cout << "total non empty cells: " << v_ordered_cell_cnt.size() << std::endl;

    // TODO remove this
    auto sum = next_util::sum_array(&v_cell_cnt[0], v_cell_cnt.size());
    assert(sum == n_coords);

    // Divide the data
    next_util::print_value_vector("final tallies: ", v_cell_cnt, v_ordered_cell_cnt);


//    std::vector<uint32_t> v_partition_size(n_partitions, 0);


    /*
    v_part_size.resize(n_partitions);
    v_part_offset.resize(n_partitions);
    std::vector<size_t> v_ordered_cell_marker(v_ordered_cell_cnt.size());
    assert(v_part_size.size() <= v_ordered_cell_cnt.size());
    for (size_t i = 0; i < v_part_size.size(); ++i) {
        v_part_size[i] = v_cell_cnt[v_ordered_cell_cnt[i]];
        v_ordered_cell_marker[i] = i;
    }
//    std::fill(v_cell_marker.begin(), std::next(v_cell_marker.begin(), n_partitions), 1);
    for (size_t i = v_part_size.size(); i < v_ordered_cell_cnt.size(); ++i) {
        size_t min_index = UINT32_MAX;
        uint32_t min_value = UINT32_MAX;
        for (size_t j = 0; j < v_part_size.size(); ++j) {
            if (v_part_size[j] < min_value) {
                min_value = v_part_size[j];
                min_index = j;
            }
        }
        v_part_size[min_index] += v_cell_cnt[v_ordered_cell_cnt[i]];
        v_ordered_cell_marker[i] = min_index;
    }

    sum = next_util::sum_array(&v_cell_cnt[0], static_cast<uint32_t>(v_cell_cnt.size()));
    assert(sum == n_coords);

    next_util::print_vector("final data partion sizes: ", v_part_size);
    next_util::fill_offsets(v_part_offset, v_part_size);
    s_vec<unsigned long> v_tmp_offset = v_part_offset;
//    next_util::print_vector("offsets: ", v_part_offset);
    std::cout << "PRE FINAL" << std::endl;
    v_part_coord.resize(n_coords);
    for (uint32_t i = 0; i < n_coords; ++i) {
//        std::cout << "i: " << i << std::endl;
        bool check = false;
        auto cell_index = v_cell_index[i];
        for (size_t j = 0; j < v_ordered_cell_cnt.size(); ++j) {
            if (v_ordered_cell_cnt[j] == cell_index) {
                check = true;
                cell_index = j;
                break;
            }
        }
        assert(check);
//        std::cout << "cell_index: " << cell_index << " v_cell_marker size: " << v_ordered_cell_marker.size() << std::endl;
        assert(cell_index < v_ordered_cell_marker.size());
        auto cell_marker = v_ordered_cell_marker[cell_index];
//        std::cout << "cell_marker: " << cell_marker << std::endl;
        assert(cell_marker < v_tmp_offset.size());
        auto offset = v_tmp_offset[cell_marker];
//        std::cout << "offset: " << offset << std::endl;
        assert(offset < v_part_coord.size());
        v_part_coord[v_tmp_offset[cell_marker]++] = i;
//        v_tmp_offset[v_cell_marker[v_cell_index[i]]]++;


    }
    std::cout << "CHECKPOINT FINAL" << std::endl;
    for (size_t i = 0; i < v_tmp_offset.size(); ++i) {
        std::cout << v_tmp_offset[i] << " : " << v_part_offset[i] << " : " << v_part_size[i] << std::endl;
        assert(v_tmp_offset[i] == v_part_offset[i] + v_part_size[i]);
    }

     */
}
