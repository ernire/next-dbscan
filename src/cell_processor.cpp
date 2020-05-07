//
// Created by Ernir Erlingsson on 7.5.2020.
//

#include <iostream>
#include <omp.h>
#include <numeric>
#include "cell_processor.h"

inline bool dist_leq(const float *coord1, const float *coord2, long const max_d, float const e2) noexcept {
    float tmp = 0;
    #pragma omp simd
    for (int d = 0; d < max_d; d++) {
        float tmp2 = coord1[d] - coord2[d];
        tmp += tmp2 * tmp2;
    }
    return tmp <= e2;
}

bool are_core_connected(s_vec<float> &v_coords,/*long> &v_index_map,*/ s_vec<long> &v_cell_begin,
        s_vec<long> &v_cell_ns, s_vec<uint8_t> &v_is_core,
        long const c1, long const c2, long const n_dim, const float e2) noexcept {
    auto begin1 = v_cell_begin[c1];
    auto begin2 = v_cell_begin[c2];
    for (long k1 = 0; k1 < v_cell_ns[c1]; ++k1) {
//        auto p1 = v_index_map[begin1 + k1];
        auto p1 = begin1 + k1;
        if (!v_is_core[p1]) {
            continue;
        }
        for (long k2 = 0; k2 < v_cell_ns[c2]; ++k2) {
            auto p2 = begin2 + k2;
//            auto p2 = v_index_map[begin2 + k2];
            if (!v_is_core[p2]) {
                continue;
            }
            if (dist_leq(&v_coords[p1 * n_dim], &v_coords[p2 * n_dim], n_dim, e2)) {
                return true;
            }
        }
    }
    return false;
}

bool flatten_labels(std::vector<long> &v_local_min_labels) noexcept {
    bool is_update = false;
    #pragma omp parallel for schedule(guided)
    for (long i = 0; i < v_local_min_labels.size(); ++i) {
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
        auto c1 = v_edges[i];
        auto c2 = v_edges[i+1];
        long c_lower, c_higher;
        auto conn = v_edge_conn[i/2];
        if (conn == NOT_CONNECTED) {
            continue;
        }
        if (v_leaf_cell_type[c1] != NO_CORES && v_leaf_cell_type[c2] != NO_CORES) {
            // both either AC or SC
            c_lower = (c1 < c2)? c1 : c2;
            c_higher = (c1 < c2)? c2 : c1;
            if (v_local_min_labels[c_higher] <= v_local_min_labels[c_lower]) {
                continue;
            }
            if (conn == UNKNOWN || conn == PARTIALLY_CONNECTED) {
                if (are_core_connected(v_coords, /*nc.vv_index_map[0],*/ nc.vv_cell_begin[0], nc.vv_cell_ns[0],
                        v_is_core, c1, c2, nc.n_dim, nc.e2)) {
                    v_edge_conn[i/2] = CORE_CONNECTED;
                    _atomic_op(&v_local_min_labels[c_higher], v_local_min_labels[c_lower], std::less<>());
                } else {
                    v_edge_conn[i/2] = NOT_CORE_CONNECTED;
                }
            } else if (conn == FULLY_CONNECTED) {
                // We know they are connected
                _atomic_op(&v_local_min_labels[c_higher], v_local_min_labels[c_lower], std::less<>());
            }
        } else if (conn == FULLY_CONNECTED && (v_leaf_cell_type[c1] != NO_CORES || v_leaf_cell_type[c2] != NO_CORES)
                   && (v_leaf_cell_type[c1] == NO_CORES || v_leaf_cell_type[c2] == NO_CORES)) {
            c_lower = (v_leaf_cell_type[c1] != NO_CORES)? c1 : c2;
            c_higher = (v_leaf_cell_type[c1] != NO_CORES)? c2 : c1;
            if (v_local_min_labels[c_higher] != UNASSIGNED)
                continue;
            // Permitted race condition (DBSCAN border point labels are not deterministic)
            v_local_min_labels[c_higher] = c_lower;
        }  else if ((v_leaf_cell_type[c1] != NO_CORES || v_leaf_cell_type[c2] != NO_CORES)
                    && (v_leaf_cell_type[c1] == NO_CORES || v_leaf_cell_type[c2] == NO_CORES)) {
            // else handle border cell partials
            c_lower = (v_leaf_cell_type[c1] != NO_CORES)? c1 : c2;
            c_higher = (v_leaf_cell_type[c1] != NO_CORES)? c2 : c1;
            if (v_local_min_labels[c_higher] != UNASSIGNED)
                continue;
            auto begin1 = nc.vv_cell_begin[0][c_lower];
            auto begin2 = nc.vv_cell_begin[0][c_higher];
            for (long k1 = 0; k1 < nc.vv_cell_ns[0][c_lower]; ++k1) {
//                auto p1 = nc.vv_index_map[0][begin1 + k1];
                auto p1 = begin1 + k1;
                if (!v_is_core[p1])
                    continue;
                for (long k2 = 0; k2 < nc.vv_cell_ns[0][c_higher]; ++k2) {
//                    auto p2 = nc.vv_index_map[0][begin2 + k2];
                    auto p2 = begin2 + k2;
                    if (v_point_labels[p2] != UNASSIGNED)
                        continue;
                    if (dist_leq(&v_coords[p1 * nc.n_dim], &v_coords[p2 * nc.n_dim], nc.n_dim, nc.e2)) {
                        v_point_labels[p2] = p1;
                    }
                }
            }
        }
    }
    flatten_labels(v_local_min_labels);
    bool loop = true;
    while (loop) {
//        uint cnt = 0;
        #pragma omp parallel for schedule(guided)
        for (uint i = 0; i < v_edges.size(); i += 2) {
            auto c1 = v_edges[i];
            auto c2 = v_edges[i + 1];
            long c_lower, c_higher;
            auto conn = v_edge_conn[i / 2];
            if (conn == NOT_CONNECTED || conn == NOT_CORE_CONNECTED) {
                continue;
            }
            if (v_leaf_cell_type[c1] != NO_CORES && v_leaf_cell_type[c2] != NO_CORES) {
//                assert(v_leaf_cell_type[c1] != UNKNOWN && v_leaf_cell_type[c2] != UNKNOWN);
                if (v_local_min_labels[c1] != v_local_min_labels[c2]) {
                    if (conn == UNKNOWN || conn == PARTIALLY_CONNECTED) {
                        if (are_core_connected(v_coords, /*nc.vv_index_map[0],*/ nc.vv_cell_begin[0],
                                nc.vv_cell_ns[0],v_is_core, c1, c2, nc.n_dim, nc.e2)) {
                            v_edge_conn[i / 2] = CORE_CONNECTED;
                        } else {
                            v_edge_conn[i / 2] = NOT_CORE_CONNECTED;
                        }
                    }
                    if (conn == CORE_CONNECTED || conn == FULLY_CONNECTED) {
                        c_lower = (v_local_min_labels[c1] < v_local_min_labels[c2]) ?
                                  v_local_min_labels[c1] : v_local_min_labels[c2];
                        c_higher = (v_local_min_labels[c1] < v_local_min_labels[c2]) ?
                                   v_local_min_labels[c2] : v_local_min_labels[c1];
//                        ++cnt;
                        _atomic_op(&v_local_min_labels[c_higher], c_lower, std::less<>());
                    }
                }
            }
        }
//        std::cout << "loop relabel cnt: " << cnt << std::endl;
        if (!flatten_labels(v_local_min_labels)) {
            loop = false;
        }
    }
    #pragma omp parallel for
    for (int i = 0; i < v_local_min_labels.size(); ++i) {
        if (v_local_min_labels[i] == i || v_local_min_labels[i] == UNASSIGNED)
            continue;
        if (v_leaf_cell_type[i] != NO_CORES) {
            auto begin = nc.vv_cell_begin[0][i];
//            auto p = nc.vv_index_map[0][begin];
            auto p = begin;
            if (p != v_point_labels[p]) {
                p = v_point_labels[p];
            }
//            v_point_labels[p] = v_point_labels[nc.vv_index_map[0][nc.vv_cell_begin[0][v_local_min_labels[i]]]];
            v_point_labels[p] = v_point_labels[nc.vv_cell_begin[0][v_local_min_labels[i]]];
        } else {
            auto begin = nc.vv_cell_begin[0][i];
//            auto label = v_point_labels[nc.vv_index_map[0][nc.vv_cell_begin[0][v_local_min_labels[i]]]];
            auto label = v_point_labels[nc.vv_cell_begin[0][v_local_min_labels[i]]];
            for (uint j = 0; j < nc.vv_cell_ns[0][i]; ++j) {
//                v_point_labels[nc.vv_index_map[0][begin+j]] = label;
                v_point_labels[begin+j] = label;
            }
        }
    }
}

inline void update_to_ac(/*s_vec<long> &v_index_maps, */s_vec<long> &v_cell_ns,
        s_vec<long> &v_cell_begin, s_vec<uint8_t> &is_core, s_vec<uint8_t> &v_types,
        const long c) noexcept {
    v_types[c] = ALL_CORES;
    long begin = v_cell_begin[c];
    for (uint j = 0; j < v_cell_ns[c]; ++j) {
        is_core[begin + j] = 1;
//        is_core[v_index_maps[begin + j]] = 1;
    }
}

void update_type(/*s_vec<long> &v_index_maps, */s_vec<long> &v_cell_ns,
        s_vec<long> &v_cell_begin, s_vec<long> &v_cell_nps, s_vec<long> &v_point_nps,
        s_vec<uint8_t> &is_core, s_vec<uint8_t> &v_types, const long c, const long m) noexcept {
    if (v_cell_nps[c] >= m) {
        update_to_ac(/*v_index_maps, */v_cell_ns, v_cell_begin, is_core, v_types, c);
    }
    bool all_cores = true;
    bool some_cores = false;
    long begin = v_cell_begin[c];
    for (long j = 0; j < v_cell_ns[c]; ++j) {
//        long p = v_index_maps[begin + j];
        long p = begin + j;
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
//    uint max_clusters = 0;
    v_is_core.resize(nc.n_coords, UNKNOWN);
    v_point_labels.resize(nc.n_coords, UNASSIGNED);
    #pragma omp parallel for //reduction(+: max_clusters)
    for (long i = 0; i < nc.vv_cell_ns[0].size(); ++i) {
        update_type(/*nc.vv_index_map[0],*/ nc.vv_cell_ns[0], nc.vv_cell_begin[0],
                v_leaf_cell_np, v_point_np, v_is_core, v_leaf_cell_type, i, nc.m);
        if (v_leaf_cell_type[i] != UNKNOWN) {
//            ++max_clusters;
            auto begin = nc.vv_cell_begin[0][i];
            long core_p = UNASSIGNED;
            for (uint j = 0; j < nc.vv_cell_ns[0][i]; ++j) {
//                auto p = nc.vv_index_map[0][begin + j];
                auto p = begin + j;
                if (core_p != UNASSIGNED) {
                    v_point_labels[p] = core_p;
                } else if (v_is_core[p]) {
                    core_p = p;
                    v_point_labels[core_p] = core_p;
                    for (uint k = 0; k < j; ++k) {
//                        p = nc.vv_index_map[0][begin + k];
                        p = begin + k;
                        v_point_labels[p] = core_p;
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
//        s_vec<long> &v_index_maps,
        s_vec<long> &v_point_nps,
        s_vec<long> &v_cell_ns,
        std::vector<long> &v_range_cnt,
        s_vec<long> &v_cell_nps,
        const long max_d, const float e2, const long m,
        const long c1, const long begin1, const long c2, const long begin2) noexcept {
    uint8_t are_connected = NOT_CONNECTED;
    long size1 = v_cell_ns[c1];
    long size2 = v_cell_ns[c2];
    std::fill(v_range_cnt.begin(), std::next(v_range_cnt.begin(), size1+size2), 0);
    for (long k1 = 0; k1 < size1; ++k1) {
//        long p1 = v_index_maps[begin1 + k1];
        long p1 = begin1 + k1;
        for (long k2 = 0; k2 < size2; ++k2) {
//            long p2 = v_index_maps[begin2 + k2];
            long p2 = begin2 + k2;
            if (dist_leq(&v_coords[p1 * max_d], &v_coords[p2 * max_d], max_d, e2)) {
                ++v_range_cnt[k1];
                ++v_range_cnt[size1+k2];
            }
        }
    }
    ull hits = 0;
    ull last_hits = 0;
    for (uint i = 0; i < size1+size2; ++i) {
        last_hits = hits;
        hits += v_range_cnt[i];
        if (hits < last_hits) {
            std::cerr << "OVERFLOW ERROR" << std::endl;
            exit(-3);
        }
    }
    ull limit = (ull)size1*(ull)size2*2L;
    if (hits > limit) {
        std::cerr << "64 bit Overflow detected: " << size1 << " : " << size2 << std::endl;
    }
    //    assert(hits <= limit);
    if (hits == limit) {
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
                long p1 = begin1 + k1;
//                long p1 = v_index_maps[begin1 + k1];
    #pragma omp atomic
                v_point_nps[p1] += v_range_cnt[k1] - min;
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
//                long p2 = v_index_maps[begin2 + k2];
                long p2 = begin2 + k2;
                #pragma omp atomic
                v_point_nps[p2] += v_range_cnt[k2+size1] - min;
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
    std::cout << "max points in cell: " << max_points_in_cell << std::endl;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        vv_range_counts[tid].resize(static_cast<unsigned long>(max_points_in_cell * 2));
        // TODO guided ?
        #pragma omp for schedule(dynamic)
        for (uint i = 0; i < v_edges.size(); i += 2) {
//            std::cout << "i: " << i << std::endl;
//            std::cout << "CHECKPOINT" << std::endl;
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
            uint8_t are_connected = process_pair_proximity(v_coords, /*nc.vv_index_map[0],*/ v_point_np,
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
