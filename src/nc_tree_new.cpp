//
// Created by Ernir Erlingsson on 6.5.2020.
//

#include <numeric>
#include <iostream>
#include "nc_tree_new.h"
#include "next_data_omp.h"

/*
struct cell_meta {
    uint l, c;

    cell_meta(uint l, uint c) : l(l), c(c) {}
};
 */
struct cell_meta_pair_level {
    long l, c1, c2;

    cell_meta_pair_level(long l, long c1, long c2) : l(l), c1(c1), c2(c2) {}
};

void calculate_level_cell_bounds(s_vec<float> &v_coords, s_vec<long> &v_cell_begins,
        s_vec<long> &v_cell_ns, s_vec<long> &v_index_maps,
        std::vector<std::vector<float>> &vv_min_cell_dims,
        std::vector<std::vector<float>> &vv_max_cell_dims,
        const long max_d, const long l) noexcept {
    vv_min_cell_dims[l].resize(v_cell_begins.size() * max_d);
    vv_max_cell_dims[l].resize(vv_min_cell_dims[l].size());
    float *coord_min = nullptr, *coord_max = nullptr;

//#pragma omp parallel for private(coord_min, coord_max)
    for (long i = 0; i < v_cell_begins.size(); i++) {
        long begin = v_cell_begins[i];
        long coord_offset = v_index_maps[begin] * max_d;
        if (l == 0) {
            coord_min = &v_coords[coord_offset];
            coord_max = &v_coords[coord_offset];
        } else {
            coord_min = &vv_min_cell_dims[l - 1][coord_offset];
            coord_max = &vv_max_cell_dims[l - 1][coord_offset];
        }
        std::copy(coord_min, coord_min + max_d, &vv_min_cell_dims[l][i * max_d]);
        std::copy(coord_max, coord_max + max_d, &vv_max_cell_dims[l][i * max_d]);

        for (uint j = 1; j < v_cell_ns[i]; j++) {
            long coord_offset_inner = 0;
            if (l == 0) {
                coord_offset_inner = v_index_maps[begin + j] * max_d;
                coord_min = &v_coords[coord_offset_inner];
                coord_max = &v_coords[coord_offset_inner];
            } else {
                coord_min = &vv_min_cell_dims[l - 1][v_index_maps[begin + j] * max_d];
                coord_max = &vv_max_cell_dims[l - 1][v_index_maps[begin + j] * max_d];
            }
            for (uint d = 0; d < max_d; d++) {
                if (coord_min[d] < vv_min_cell_dims[l][i * max_d + d]) {
                    vv_min_cell_dims[l][i * max_d + d] = coord_min[d];
                }
                if (coord_max[d] > vv_max_cell_dims[l][i * max_d + d]) {
                    vv_max_cell_dims[l][i * max_d + d] = coord_max[d];
                }
            }
        }
    }
}

uint index_level(s_vec<float> &v_coords, s_vec<float> &v_min_bounds, s_vec<long> &v_dim_cells,
        d_vec<long> &vv_index_map, d_vec<long> &vv_cell_begin, s_vec<long> &v_cell_ns,
        const float level_eps,
        const int l, const uint size, const uint n_dim) noexcept {
    v_dim_cells.resize(size * n_dim);
    #pragma omp parallel for
    for (uint i = 0; i < size; ++i) {
        uint p_index = i;
        int level_mod = 1;
        while (l - level_mod >= 0) {
            p_index = vv_index_map[l - level_mod][vv_cell_begin[l - level_mod][p_index]];
            ++level_mod;
        }
        uint coord_index = (p_index) * n_dim;
        for (uint d = 0; d < n_dim; ++d) {
            v_dim_cells[(i*n_dim) + d] = floorf((v_coords[coord_index + d] - v_min_bounds[d]) / level_eps)+1;
//            assert(v_dim_cells[(i*n_dim) + d] <= v_max_dim_values[d]);
        }
    }
    vv_index_map[l].resize(size);
    std::iota(vv_index_map[l].begin(), vv_index_map[l].end(), 0);
    std::sort(vv_index_map[l].begin(), vv_index_map[l].end(),
            [&n_dim, &v_dim_cells](const uint &i1, const uint &i2) -> bool {
                const uint ci1 = i1 * n_dim;
                const uint ci2 = i2 * n_dim;
                for (uint d = 0; d < n_dim; ++d) {
                    if (v_dim_cells[ci1+d] < v_dim_cells[ci2+d]) {
                        return true;
                    }
                    if (v_dim_cells[ci1+d] > v_dim_cells[ci2+d]) {
                        return false;
                    }
                }
                return false;
            });

    uint n_cells = 1;
    uint ci1 = vv_index_map[l][0] * n_dim;
    for (uint i = 1; i < vv_index_map[l].size(); ++i) {
        uint ci2 = vv_index_map[l][i] * n_dim;
        for (uint d = 0; d < n_dim; ++d) {
            if (v_dim_cells[ci1+d] != v_dim_cells[ci2+d]) {
                ++n_cells;
                ci1 = ci2;
                d = n_dim;
            }
        }
    }
    vv_cell_begin[l].resize(n_cells);
    v_cell_ns.resize(n_cells, 0);
    uint index = 0;
    uint n_cnt = 1;
    vv_cell_begin[l][0] = 0;
    ci1 = vv_index_map[l][0] * n_dim;
    for (uint i = 1; i < vv_index_map[l].size(); ++i) {
        uint ci2 = vv_index_map[l][i] * n_dim;
        bool is_equal = true;
        for (uint d = 0; d < n_dim && is_equal; ++d) {
            if (v_dim_cells[ci1+d] != v_dim_cells[ci2+d]) {
                v_cell_ns[index] = n_cnt;
                ++index;
                vv_cell_begin[l][index] = i;
                n_cnt = 1;
                ci1 = ci2;
                is_equal = false;
            }
        }
        if (is_equal) {
            ++n_cnt;
        }
    }
    v_cell_ns[index] = n_cnt;
    return v_cell_ns.size();
}

void nc_tree_new::build_tree(s_vec<float> &v_coords, s_vec<float> &v_min_bounds) noexcept {
    s_vec<float> v_eps_levels(n_level);
//#pragma omp parallel for
    for (uint l = 0; l < n_level; l++) {
        // TODO maybe keep double ?
        v_eps_levels[l] = static_cast<float>(e_lowest * pow(2, l));
    }
    std::vector<long> v_index_dims;
    auto size = n_coords;
    for (int l = 0; l < n_level; ++l) {
        size = index_level(v_coords, v_min_bounds, v_index_dims, vv_index_map, vv_cell_begin,
                vv_cell_ns[l], v_eps_levels[l], l, size, n_dim);
        calculate_level_cell_bounds(v_coords, vv_cell_begin[l], vv_cell_ns[l],
                vv_index_map[l], vv_min_cell_dim, vv_max_cell_dim, n_dim, l);
        /*
        if (l <= n_parallel_level) {
            size = index_level_parallel(v_coords, v_min_bounds, v_max_bounds, vv_part_coord_index,
                    vv_part_cell_begin, vv_part_cell_ns, v_t_n_cells, v_t_offsets,
                    v_index_dims, vv_index_map, vv_cell_begin, vv_cell_ns, l, size,
                    n_dim, n_threads, e_inner, n_parallel_level);
        } else {
            size = index_level(v_coords, v_min_bounds, v_index_dims, vv_index_map, vv_cell_begin,
                    vv_cell_ns[l], v_eps_levels[l], l, size, n_dim);
        }
        calculate_level_cell_bounds(v_coords, vv_cell_begin[l], vv_cell_ns[l],
                vv_index_map[l], vv_min_cell_dim, vv_max_cell_dim, n_dim, l);
                */
    }
}

void nc_tree_new::collect_edges(s_vec<long> &v_edges) noexcept {
    std::vector<cell_meta_pair_level> v_stack;
    // TODO make this dynamic
    v_stack.reserve(1024);
    v_edges.reserve(static_cast<unsigned long>(get_total_no_of_cells()));
    for (long l = 1; l < n_level; ++l) {
//        std::cout << "l: " << l << " cells: " << get_no_of_cells(l) << std::endl;
        for (long c = 0; c < get_no_of_cells(l); ++c) {
            auto begin = vv_cell_begin[l][c];
            for (uint c1 = 0; c1 < vv_cell_ns[l][c]; ++c1) {
                auto c1_index = vv_index_map[l][begin + c1];
                for (uint c2 = c1 + 1; c2 < vv_cell_ns[l][c]; ++c2) {
                    auto c2_index = vv_index_map[l][begin + c2];
                    if (next_data::is_in_reach(&vv_min_cell_dim[l - 1][c1_index * n_dim],
                            &vv_max_cell_dim[l - 1][c1_index * n_dim],
                            &vv_min_cell_dim[l - 1][c2_index * n_dim],
                            &vv_max_cell_dim[l - 1][c2_index * n_dim], n_dim, e)) {
                        v_stack.emplace_back(l - 1, c1_index, c2_index);
                        while (!v_stack.empty()) {
                            auto li = v_stack.back().l;
                            auto ci1 = v_stack.back().c1;
                            auto ci2 = v_stack.back().c2;
                            v_stack.pop_back();
                            auto begin1 = vv_cell_begin[li][ci1];
                            auto begin2 = vv_cell_begin[li][ci2];
                            if (li == 0) {
                                // CUDA doesn't support emplace_back
//                                v_edges.emplace_back(ci1, ci2);
                                v_edges.push_back(ci1);
                                v_edges.push_back(ci2);
                            } else {
                                for (uint k1 = 0; k1 < vv_cell_ns[li][ci1]; ++k1) {
                                    auto c1_next = vv_index_map[li][begin1 + k1];
                                    for (uint k2 = 0; k2 < vv_cell_ns[li][ci2]; ++k2) {
                                        auto c2_next = vv_index_map[li][begin2 + k2];
                                        if (next_data::is_in_reach(&vv_min_cell_dim[li - 1][c1_next * n_dim],
                                                &vv_max_cell_dim[li - 1][c1_next * n_dim],
                                                &vv_min_cell_dim[li - 1][c2_next * n_dim],
                                                &vv_max_cell_dim[li - 1][c2_next * n_dim], n_dim, e)) {
                                            v_stack.emplace_back(li - 1, c1_next, c2_next);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
/*
    std::vector<cell_meta> v_tasks;
    std::vector<std::vector<cell_meta_pair_level>> vv_stack(n_threads);
    d_vec<uint> vv_edges(n_threads);
    uint task_size = 0;
    #pragma omp parallel for reduction(+:task_size)
    for (uint l = 1; l < n_level; ++l) {
        task_size += vv_cell_begin[l].size();
    }
    v_tasks.reserve(task_size);
    for (uint l = 1; l < n_level; ++l) {
        for (uint i = 0; i < vv_cell_begin[l].size(); ++i) {
            v_tasks.emplace_back(l,i);
        }
    }
    uint total_edges = 0;
    #pragma omp parallel reduction(+: total_edges)
    {
        uint tid = omp_get_thread_num();
        vv_edges[tid].reserve(v_tasks.size() / n_threads);
        #pragma omp for schedule(dynamic)
        for (uint i = 0; i < v_tasks.size(); ++i) {
            uint l = v_tasks[i].l;
            uint c = v_tasks[i].c;
            uint begin = vv_cell_begin[l][c];
            for (uint c1 = 0; c1 < vv_cell_ns[l][c]; ++c1) {
                uint c1_index = vv_index_map[l][begin + c1];
                for (uint c2 = c1 + 1; c2 < vv_cell_ns[l][c]; ++c2) {
                    uint c2_index = vv_index_map[l][begin + c2];
                    if (is_in_reach(&vv_min_cell_dim[l - 1][c1_index * n_dim],
                            &vv_max_cell_dim[l - 1][c1_index * n_dim],
                            &vv_min_cell_dim[l - 1][c2_index * n_dim],
                            &vv_max_cell_dim[l - 1][c2_index * n_dim], n_dim, e)) {
                        vv_stack[tid].emplace_back(l - 1, c1_index, c2_index);
                        process_pair_stack(vv_edges[tid], vv_index_map, vv_cell_begin, vv_cell_ns,
                                vv_min_cell_dim, vv_max_cell_dim, vv_stack[tid], n_dim, e);
                    }
                }
            }
        }
        total_edges += vv_edges[tid].size();
    } // end parallel region
    for (uint t = 0; t < n_threads; ++t) {
        vv_stack[t].clear();
        vv_stack[t].shrink_to_fit();
    }
    for (uint l = 1; l < n_level; ++l) {
        vv_index_map[l].clear();
        vv_index_map[l].shrink_to_fit();
        vv_cell_ns[l].clear();
        vv_cell_ns[l].shrink_to_fit();
        vv_cell_begin[l].clear();
        vv_cell_begin[l].shrink_to_fit();
        vv_min_cell_dim[l].clear();
        vv_min_cell_dim[l].shrink_to_fit();
        vv_max_cell_dim[l].clear();
        vv_max_cell_dim[l].shrink_to_fit();
    };
    v_edges.reserve(total_edges);
    for (uint t = 0; t < vv_edges.size(); ++t) {
        v_edges.insert(v_edges.end(), std::make_move_iterator(vv_edges[t].begin()),
                std::make_move_iterator(vv_edges[t].end()));
    }
 */
}

