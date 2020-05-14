//
// Created by Ernir Erlingsson on 6.5.2020.
//

#include <numeric>
#include <iostream>
#include <cassert>
#include <omp.h>
#include <unordered_set>
#include "nc_tree_new.h"
#include "next_data_omp.h"
#include "next_util.h"
#include "deep_io.h"


void calculate_level_cell_bounds(s_vec<float> &v_coords, s_vec<long> &v_cell_begins,
        s_vec<long> &v_cell_ns, s_vec<long> &v_index_maps,
        std::vector<std::vector<float>> &vv_min_cell_dims,
        std::vector<std::vector<float>> &vv_max_cell_dims,
        const long max_d, const long l) noexcept {
    vv_min_cell_dims[l].resize(v_cell_begins.size() * max_d);
    vv_max_cell_dims[l].resize(vv_min_cell_dims[l].size());
    float *coord_min = nullptr, *coord_max = nullptr;

    #pragma omp parallel for private(coord_min, coord_max)
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

void new_index_level(s_vec<float> &v_coords,
        s_vec<long> &v_dim_index,
        s_vec<long> &v_index_map,
        s_vec<float> &v_min_bounds,
        s_vec<long> &v_cell_begin,
        std::vector<long> &v_begin_shortcut,
        std::vector<long> &v_tmp,
        s_vec<long> &v_cell_ns,
        long const n_dim,
        float const e_lvl) {

    for (auto i = 0; i < v_begin_shortcut.size(); ++i) {
        for (uint d = 0; d < n_dim; ++d) {
            v_dim_index[(i*n_dim) + d] = static_cast<long>(
                    floorf((v_coords[(v_begin_shortcut[i]*n_dim)+d] - v_min_bounds[d]) / e_lvl) + 1);
        }
    }
    v_index_map.resize(v_begin_shortcut.size());
    std::iota(v_index_map.begin(), v_index_map.end(), 0);
    std::sort(v_index_map.begin(), v_index_map.end(),
            [&] (auto const &i1, auto const &i2) -> bool {
                auto const ci1 = i1 * n_dim;
                auto const ci2 = i2 * n_dim;
                for (uint d = 0; d < n_dim; ++d) {
                    if (v_dim_index[ci1+d] < v_dim_index[ci2+d]) {
                        return true;
                    }
                    if (v_dim_index[ci1+d] > v_dim_index[ci2+d]) {
                        return false;
                    }
                }
                return false;
            });
    v_cell_begin.reserve(v_begin_shortcut.size());
    v_cell_begin.push_back(0);
    auto ci1 = v_index_map[0] * n_dim;
    for (auto i = 1; i < v_index_map.size(); ++i) {
        auto ci2 = v_index_map[i] * n_dim;
        for (auto d = 0; d < n_dim; ++d) {
            if (v_dim_index[ci1+d] != v_dim_index[ci2+d]) {
                ci1 = ci2;
                v_cell_begin.push_back(i);
                break;
            }
        }
    }
    v_cell_begin.shrink_to_fit();
    v_cell_ns.resize(v_cell_begin.size());
    for (auto i = 1; i < v_cell_begin.size(); ++i) {
        v_cell_ns[i-1] = v_cell_begin[i] - v_cell_begin[i-1];
    }
    v_cell_ns[v_cell_ns.size()-1] = v_begin_shortcut.size() - v_cell_begin[v_cell_begin.size()-1];
    v_tmp.clear();
    v_tmp.assign(v_begin_shortcut.begin(), v_begin_shortcut.end());
    for (auto i = 0; i < v_cell_begin.size(); ++i) {
        v_begin_shortcut[i] = v_tmp[v_index_map[v_cell_begin[i]]];
    }
    v_begin_shortcut.resize(v_cell_begin.size());
}

void nc_tree_new::build_tree() noexcept {
    std::vector<float> v_eps_levels(n_level);
    std::vector<long> v_dim_index(n_coords * n_dim);
    std::vector<long> v_begin_shortcut(n_coords);
    std::vector<long> v_tmp;
    std::iota(v_begin_shortcut.begin(), v_begin_shortcut.end(), 0);
    for (long l = 0; l < n_level; l++) {
        // TODO maybe keep double for precision?
        v_eps_levels[l] = static_cast<float>(e_lowest * pow(2, l));
        new_index_level(v_coords, v_dim_index, vv_index_map[l], v_min_bounds, vv_cell_begin[l], v_begin_shortcut,
                v_tmp, vv_cell_ns[l], n_dim, v_eps_levels[l]);
        calculate_level_cell_bounds(v_coords, vv_cell_begin[l], vv_cell_ns[l],
                vv_index_map[l], vv_min_cell_dim, vv_max_cell_dim, n_dim, l);
    }
    std::vector<float> v_coords_copy = v_coords;
    next_data::reorder_coords(v_coords_copy.begin(), vv_index_map[0].begin(), vv_index_map[0].end(),
            v_coords.begin(), n_dim);
    vv_index_map[0].clear();
    vv_index_map[0].shrink_to_fit();
}

void nc_tree_new::build_tree_parallel(unsigned long const n_threads) noexcept {
    std::cout << "Build tree parallel with " << n_level_parallel << " levels." << std::endl;

    std::vector<float> v_eps_levels(n_level);
    // TODO remove -1 x 2
    std::vector<long> v_dim_index(n_coords * n_dim, -1);
    std::vector<long> v_coord_index(n_coords);
    std::vector<long> v_tmp(n_coords);
    std::iota(v_coord_index.begin(), v_coord_index.end(), 0);
    auto v_part_level_size = v_part_size;
    auto v_part_level_offset = v_part_offset;

    for (auto l = 0; l < n_level_parallel; l++) {
        v_eps_levels[l] = static_cast<float>(e_lowest * pow(2, l));
    }
    std::vector<unsigned long> v_marker_cnt(n_threads);
    std::vector<unsigned long> v_marker_offset(n_threads);

    std::vector<int8_t > v_begin_marker(n_coords);
    for (auto l = 0; l < n_level_parallel; l++) {
        vv_index_map[l].resize(l == 0? n_coords : vv_cell_begin[l-1].size(), -1);
        std::iota(vv_index_map[l].begin(), vv_index_map[l].end(), 0);
//        std::fill(v_dim_index.begin(), v_dim_index.end(), -1);

        #pragma omp parallel for
        for (auto i = 0; i < vv_index_map[l].size(); ++i) {
            for (auto d = 0; d < n_dim; ++d) {
                v_dim_index[i*n_dim+d] = static_cast<long>(
                        floorf((v_coords[(v_coord_index[i]*n_dim)+d] - v_min_bounds[d]) / v_eps_levels[l]));
            }
        }
//        for (auto par = 0; par < v_part_level_offset.size()-1; ++par) {
//            assert(v_part_level_offset[par] + v_part_level_size[par] == v_part_level_offset[par+1]);
//        }

        #pragma omp parallel for schedule(dynamic)
        for (auto par = 0; par < v_part_level_offset.size(); ++par) {
            auto index_begin = std::next(vv_index_map[l].begin(), v_part_level_offset[par]);
            auto index_end = std::next(index_begin, v_part_level_size[par]);
            std::sort(index_begin, index_end, [&] (auto const &i1, auto const &i2) -> bool {
                auto const ci1 = i1 * n_dim;
                auto const ci2 = i2 * n_dim;
                for (uint d = 0; d < n_dim; ++d) {
//                    assert(v_dim_index[ci1+d] >= 0);
//                    assert(v_dim_index[ci2+d] >= 0);
                    if (v_dim_index[ci1+d] < v_dim_index[ci2+d]) {
                        return true;
                    }
                    if (v_dim_index[ci1+d] > v_dim_index[ci2+d]) {
                        return false;
                    }
                }
                return false;
            });
        }
        // TODO parallel
//        std::fill(v_begin_marker.begin(), v_begin_marker.end(), -1);
        std::fill(v_begin_marker.begin(), std::next(v_begin_marker.begin(), vv_index_map[l].size()), 0);

        unsigned long total_cells = 0;
        // parallel
        #pragma omp parallel
        {
            auto tid = omp_get_thread_num();
            unsigned long t_cells = 0;
            auto ts = deep_io::get_block_size(tid, vv_index_map[l].size(), n_threads);
            auto ti = deep_io::get_block_start_offset(tid, vv_index_map[l].size(), n_threads);
            for (auto i = ti; i < ti+ts; ++i) {
//                assert(v_begin_marker[i] != -1);
                if (i == 0) {
                    v_begin_marker[i] = 1;
                    ++t_cells;
                    continue;
                }
                auto c1 = vv_index_map[l][i-1] * n_dim;
                auto c2 = vv_index_map[l][i] * n_dim;
                for (auto d = 0; d < n_dim; ++d) {
                    if (v_dim_index[c1 + d] != v_dim_index[c2 + d]) {
                        v_begin_marker[i] = 1;
                        ++t_cells;
                        break;
                    }
                }
            }
            v_marker_cnt[tid] = t_cells;
            #pragma omp barrier
            #pragma omp for schedule(dynamic)
            for (auto par = 0; par < v_part_level_offset.size(); ++par) {
                t_cells = 0;
                for (auto i = v_part_level_offset[par]; i < v_part_level_offset[par] + v_part_level_size[par]; ++i) {
                    if (v_begin_marker[i] == 1)
                        ++t_cells;
                }
                v_part_level_size[par] = t_cells;
            }
            #pragma omp single
            {
                v_part_level_offset[0] = 0;
                for (auto t = 1; t < v_part_level_offset.size(); ++t) {
                    v_part_level_offset[t] = v_part_level_offset[t-1] + v_part_level_size[t-1];
                }
                v_marker_offset[0] = 0;
                total_cells += v_marker_cnt[0];
                for (auto t = 1; t < v_marker_offset.size(); ++t) {
                    v_marker_offset[t] = v_marker_offset[t-1] + v_marker_cnt[t-1];
                    total_cells += v_marker_cnt[t];
                }
                vv_cell_begin[l].resize(total_cells+1, -1);
                vv_cell_begin[l][vv_cell_begin[l].size()-1] = vv_index_map[l].size();
                vv_cell_ns[l].resize(total_cells, 0);
            }
            auto offset = v_marker_offset[tid];
            for (auto i = ti; i < ti+ts; ++i) {
                if (v_begin_marker[i] == 1) {
                    vv_cell_begin[l][offset++] = i;
                }
            }
//            assert(offset == v_marker_cnt[tid] + v_marker_offset[tid]);
            #pragma omp barrier
            #pragma omp for
            for (auto i = 0; i < vv_cell_begin[l].size()-1; ++i) {
                vv_cell_ns[l][i] = vv_cell_begin[l][i+1] - vv_cell_begin[l][i];
            }

        } // end of parallel region
        vv_cell_begin[l].resize(vv_cell_begin[l].size()-1);
        v_tmp.clear();
        v_tmp.assign(v_coord_index.begin(), v_coord_index.end());
        v_coord_index.resize(vv_cell_begin[l].size());

        for (auto i = 0; i < vv_cell_begin[l].size(); ++i) {
            v_coord_index[i] = v_tmp[vv_index_map[l][vv_cell_begin[l][i]]];
        }
        std::cout << "Level " << l << " total cells: " << total_cells << std::endl;
        /*
        for (auto i = 0; i < vv_cell_begin[l].size(); ++i) {
            assert(vv_cell_begin[l][i] != -1);
        }
        long lastval = -1;
        for (auto &val : vv_cell_begin[l]) {
            assert(val != -1);
            if (lastval == -1) {
                lastval = val;
            } else {
                assert(lastval < val);
                lastval = val;
            }
        }
        auto sum = 0;
        for (auto i = 0; i < vv_cell_ns[l].size(); ++i) {
            assert(vv_cell_ns[l][i] != 0);
            sum += vv_cell_ns[l][i];
        }
        assert(sum == vv_index_map[l].size());
        */
        calculate_level_cell_bounds(v_coords, vv_cell_begin[l], vv_cell_ns[l],
                vv_index_map[l], vv_min_cell_dim, vv_max_cell_dim, n_dim, l);
    } // for level
    // TODO parallelize
    std::vector<float> v_coords_copy = v_coords;
    next_data::reorder_coords(v_coords_copy.begin(), vv_index_map[0].begin(), vv_index_map[0].end(),
            v_coords.begin(), n_dim);
    vv_index_map[0].clear();
    vv_index_map[0].shrink_to_fit();
}

void nc_tree_new::partition_data(long const min_partitions) noexcept {
    std::cout << "n_coords: " << n_coords << std::endl;
    long const min_sample_size = static_cast<long>(ceil(min_partitions * /*n_dim **/ n_dim * log10(n_coords)));
    std::cout << "sample size: " << min_sample_size << std::endl;

    auto level = n_level - 1;
    s_vec<unsigned long> v_dim_cell_size(n_dim);
    std::iota(v_dim_cell_size.begin(), v_dim_cell_size.end(), 0);
    auto v_ordered_dim = v_dim_cell_size;

    unsigned long max_cells = set_partition_level(v_ordered_dim, level, min_sample_size);
    std::cout << "sufficient level: " << level << " of " << n_level << std::endl;
    std::cout << "sufficient level max cells: " << max_cells << std::endl;

    float const e_lvl = (e_lowest * powf(2, level));
    for (auto &d : v_dim_cell_size) {
        d = static_cast<unsigned long>(ceilf((v_max_bounds[d] - v_min_bounds[d]) / e_lvl));
    }
    next_util::print_vector("dim sizes: ", v_dim_cell_size);
    std::vector<long> v_dim_cell_index(n_coords * n_dim, 0);
    std::vector<long> v_dim_offset(n_dim);
    std::iota(v_dim_offset.begin(), v_dim_offset.end(), 0);
    for (auto &d : v_dim_offset) {
        d *= n_coords;
    }
    for (auto i = 0; i < n_coords; ++i) {
        for (auto const &d : v_ordered_dim) {
            assert(v_dim_cell_index[v_dim_offset[d] + i] == 0);
            v_dim_cell_index[v_dim_offset[d] + i] =
                    static_cast<long>((v_coords[(i * n_dim) + d] - v_min_bounds[d]) / e_lvl);
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
            assert(v_dim_cell_index[offset + i] >= 0 && v_dim_cell_index[offset + i] < v_dim_cell_cnt.size());
            ++v_dim_cell_cnt[v_dim_cell_index[offset + i]];
        }
        // TODO MPI merge
//        next_util::print_vector("dim value: " , v_dim_cell_cnt);
        v_dim_cell_cnt_nz.resize(v_dim_cell_cnt.size());
        auto const it = std::copy_if(v_dim_cell_cnt.begin(), v_dim_cell_cnt.end(), v_dim_cell_cnt_nz.begin(),
                [](auto const &val) { return val > 0; });
        v_dim_cell_cnt_nz.resize(static_cast<unsigned long>(std::distance(v_dim_cell_cnt_nz.begin(), it)));
        //        std::cout << "dim: " << d << " unique count: " << v_dim_cell_cnt_nz.size() << std::endl;
        float entropy = 0;
        auto dim_sum = next_util::sum_array_omp(&v_dim_cell_cnt_nz[0], v_dim_cell_cnt_nz.size());

#pragma omp parallel for reduction(+:entropy)
        for (size_t i = 0; i < v_dim_cell_cnt_nz.size(); ++i) {
            auto p = (double) v_dim_cell_cnt_nz[i] / dim_sum;
            entropy -= p * log2(p);
        }
        //        std::cout << "dim: " << d << " entropy: " << entropy << std::endl;
        v_dim_entropy[d] = entropy;
    }
    std::sort(v_ordered_dim.begin(), v_ordered_dim.end(), [&v_dim_entropy](auto const &d1, auto const &d2) -> bool {
        return v_dim_entropy[d1] > v_dim_entropy[d2];
    });


    next_util::print_vector("dim order: ", v_ordered_dim);

    // TODO TMP
//    v_ordered_dim.resize(n_dim);
//    std::iota(v_ordered_dim.begin(), v_ordered_dim.end(), 0);

    // Find the number of dimensions needed
    max_cells = 1;
    unsigned long dim_size = 0;
    for (auto const &d : v_ordered_dim) {
        max_cells *= v_dim_cell_size[d];
        ++dim_size;
//        if (max_cells > min_sample_size) {
//            break;
//        }
    }
    v_ordered_dim.resize(dim_size);

    s_vec<unsigned long> v_cell_size_mul(v_ordered_dim.size());
    v_cell_size_mul[0] = 1;
    for (auto d = 1; d < v_cell_size_mul.size(); ++d) {
//        std::cout << v_dim_cell_size[v_ordered_dim[d-1]] << " ";
        v_cell_size_mul[d] = (v_cell_size_mul[d - 1] * v_dim_cell_size[v_ordered_dim[d - 1]]);
    }

    unsigned long max = 0;
    for (auto const &d : v_ordered_dim) {
        max += v_cell_size_mul[d] * v_dim_cell_size[d];
    }
    std::cout << "max: " << max << std::endl;
    std::vector<unsigned long> v_cell_cnt(max, 0);
//    std::vector<unsigned long> v_cell_cnt(next_util::sum_array(&v_cell_size_mul[0], v_cell_size_mul.size()), 0);
    std::cout << "processed dim size: " << v_ordered_dim.size() << std::endl;
//    std::cout << "max cells: " << next_util::sum_array(&v_cell_size_mul[0], v_cell_size_mul.size()) << std::endl;
    std::vector<unsigned long> v_cell_index(n_coords);

//    std::cout << std::endl;

    next_util::print_vector("mul", v_cell_size_mul);
//    unsigned long mul_max = 1;
//    for (auto d = 0; d < v_cell_size_mul.size(); ++d) {
//        mul_max *= v_cell_size_mul[d];
//    }
//    std::cout << "mul max: " << mul_max << " : " << v_cell_cnt.size() << std::endl;
//    assert(mul_max < v_cell_cnt.size());

    next_util::print_vector("indexing dim order: ", v_ordered_dim);

    // Finally the indexing
    for (auto i = 0; i < n_coords; ++i) {
        unsigned long long cell_index = 0;
        auto d_cnt = 0;
        for (auto const &d : v_ordered_dim) {
            assert(d >= 0 && d < n_dim);
            assert(v_cell_size_mul[d_cnt] > 0);
            assert(d_cnt < v_cell_size_mul.size());
            auto tmp = (((v_coords[(i * n_dim) + d] - v_min_bounds[d]) / e_lvl));
            assert(tmp < v_dim_cell_size[d]);
            cell_index += static_cast<unsigned long>((v_coords[(i * n_dim) + d] - v_min_bounds[d]) / e_lvl) *
                          v_cell_size_mul[d_cnt];
//            cell_index *= ;
//            cell_index += v_dim_cell_index[v_dim_offset[d] + i] * v_cell_size_mul[d_cnt];
            ++d_cnt;
        }
        if (cell_index >= v_cell_cnt.size()) {
            std::cerr << "cell_index: " << cell_index << " cnt: " << v_cell_cnt.size() << std::endl;
        }
        assert(cell_index < v_cell_cnt.size());
        v_cell_index[i] = cell_index;
        ++v_cell_cnt[cell_index];
    }
    // TODO MPI merge

    for (auto i = 0; i < v_cell_cnt.size(); ++i) {
        if (v_cell_cnt[i] > 0) {
            auto cnt = 0;
            unsigned long arr[n_dim];
            unsigned long long cell_index = 0;
            for (auto j = 0; j < v_cell_index.size(); ++j) {
                if (v_cell_index[j] == i) {
                    unsigned long long cell_index_2 = 0;
                    unsigned long arr2[n_dim];
                    if (cnt == 0) {
                        for (auto d = 0; d < n_dim; ++d) {
                            arr[d] = static_cast<unsigned long>((v_coords[(j * n_dim) + d] - v_min_bounds[d]) / e_lvl);
                            cell_index += (((v_coords[(j * n_dim) + d] - v_min_bounds[d]) / e_lvl) *
                                           v_cell_size_mul[d]);
                        }
                    } else {
                        bool check = true;
                        for (auto d = 0; d < n_dim; ++d) {
                            unsigned long tmp = static_cast<unsigned long>((
                                    (v_coords[(j * n_dim) + d] - v_min_bounds[d]) / e_lvl));
                            arr2[d] = tmp;
                            cell_index_2 += (((v_coords[(j * n_dim) + d] - v_min_bounds[d]) / e_lvl) *
                                             v_cell_size_mul[d]);
                            if (arr[d] != tmp)
                                check = false;
//                            assert(arr[d] == tmp);
                        }
                        if (!check) {
                            std::cout << "Fail : " << i << " : " << cell_index << " : " << cell_index_2 << std::endl;
//                            std::cout << "Fail: " << i << std::endl;
                            next_util::print_array("array1: ", &arr[0], n_dim);
                            next_util::print_array("array2: ", &arr2[0], n_dim);
                            std::cout << std::flush << std::endl;
//                            std::cerr << "Faile : " << next_util:
                        }
                    }
                    ++cnt;
                }
            }
            assert(cnt == v_cell_cnt[i]);
        }
    }

    std::vector<unsigned long> v_ordered_cell_cnt(v_cell_cnt.size());
    // TODO iota omp
    std::iota(v_ordered_cell_cnt.begin(), v_ordered_cell_cnt.end(), 0);
    std::sort(v_ordered_cell_cnt.begin(), v_ordered_cell_cnt.end(), [&](auto const &i1, auto const &i2) -> bool {
        return v_cell_cnt[i1] > v_cell_cnt[i2];
    });
    std::cout << "total cells: " << v_cell_cnt.size() << std::endl;
    auto new_end = std::lower_bound(v_ordered_cell_cnt.begin(), v_ordered_cell_cnt.end(), 0,
            [&](auto const &i1, auto const &val) -> bool { return v_cell_cnt[i1] > val; });
    v_ordered_cell_cnt.resize(static_cast<unsigned long>(std::distance(v_ordered_cell_cnt.begin(), new_end)));
    std::cout << "total non empty cells: " << v_ordered_cell_cnt.size() << std::endl;

    // TODO remove this
    auto sum = next_util::sum_array(&v_cell_cnt[0], v_cell_cnt.size());
    assert(sum == n_coords);
    sum = 0;
    for (auto i = 0; i < v_ordered_cell_cnt.size(); ++i) {
        sum += v_cell_cnt[v_ordered_cell_cnt[i]];
    }
    assert(sum == n_coords);

    // Divide the data
//    next_util::print_value_vector("final tallies: ", v_cell_cnt, v_ordered_cell_cnt);

//    std::vector<unsigned long> v_part_coord(n_coords);
    v_part_offset.resize(v_ordered_cell_cnt.size());
    v_part_size.resize(v_ordered_cell_cnt.size());

    v_part_offset[0] = 0;
    v_part_size[0] = v_cell_cnt[v_ordered_cell_cnt[0]];
    assert(v_cell_cnt[v_ordered_cell_cnt[0]] > 0);
    for (auto i = 1; i < v_ordered_cell_cnt.size(); ++i) {
        v_part_size[i] = v_cell_cnt[v_ordered_cell_cnt[i]];
        assert(v_cell_cnt[v_ordered_cell_cnt[i]] > 0);
        v_part_offset[i] = v_part_offset[i - 1] + v_part_size[i - 1];
    }
    std::vector<long> v_coord_offset(v_cell_cnt.size(), -1);
    for (auto i = 0; i < v_ordered_cell_cnt.size(); ++i) {
        v_coord_offset[v_ordered_cell_cnt[i]] = v_part_offset[i] * n_dim;
    }
    auto v_coords_copy = v_coords;
    std::fill(v_coords.begin(), v_coords.end(), MAXFLOAT);
//    #pragma omp parallel for
    for (auto i = 0; i < n_coords; ++i) {
        assert(v_coord_offset[v_cell_index[i]] >= 0);
        std::copy(std::next(v_coords_copy.begin(), i * n_dim), std::next(v_coords_copy.begin(), (i * n_dim) + n_dim),
                std::next(v_coords.begin(), v_coord_offset[v_cell_index[i]]));
        v_coord_offset[v_cell_index[i]] += n_dim;
    }
    for (auto const &val : v_coords) {
        assert(val != MAXFLOAT);
    }

    n_level_parallel = level;
}

void nc_tree_new::collect_edges(s_vec<long> &v_edges) noexcept {
    std::vector<cell_meta_pair_level> v_stack;
    v_stack.reserve(get_no_of_cells(0));
    v_edges.reserve(get_total_no_of_cells());
    for (long l = 1; l < n_level; ++l) {
        for (long c = 0; c < get_no_of_cells(l); ++c) {
            process_tree_node(v_stack, v_edges, l, c);
        }
    }
}

void nc_tree_new::collect_edges_parallel(s_vec<long> &v_edges, unsigned long const n_threads) noexcept {
    std::vector<cell_meta_pair_level> v_stack;

    std::vector<std::vector<cell_meta_pair_level>> vv_stack(n_threads);
    std::vector<std::vector<long>> vv_edges(n_threads);

    std::vector<long> v_cells_per_level(n_level_parallel);
    for (auto l = 0; l < v_cells_per_level.size(); ++l) {
        v_cells_per_level[l] = get_no_of_cells(l);
    }
    for (auto t = 0; t < n_threads; ++t) {
        vv_stack[t].reserve(v_cells_per_level[0] / n_threads);
        vv_stack[t].reserve(get_total_no_of_cells() / n_threads);
    }
    unsigned long n_edges = 0;
    std::vector<long> v_t_edge_size(n_threads, 0);
    std::vector<long> v_t_edge_offset(n_threads, 0);

//    std::cout << "level parallel size: " << v_cells_per_level[n_level_parallel] << std::endl;

    #pragma omp parallel
    {
        auto tid = omp_get_thread_num();
        for (long l = 1; l < n_level_parallel; ++l) {
            #pragma omp for schedule(dynamic) nowait
            for (long c = 0; c < v_cells_per_level[l]; ++c) {
                process_tree_node(vv_stack[tid], vv_edges[tid], l, c);
            }
        }
        auto l = n_level_parallel-1;
        for (long c1 = 0; c1 < v_cells_per_level[l]-1; ++c1) {
            #pragma omp for schedule(dynamic) nowait
            for (long c2 = c1+1; c2 < v_cells_per_level[l]; ++c2) {
                if (next_data::is_in_reach(&vv_min_cell_dim[n_level_parallel - 1][c1 * n_dim],
                        &vv_max_cell_dim[n_level_parallel - 1][c1 * n_dim],
                        &vv_min_cell_dim[n_level_parallel - 1][c2 * n_dim],
                        &vv_max_cell_dim[n_level_parallel - 1][c2 * n_dim], n_dim, e)) {
                    vv_stack[tid].emplace_back(l, c1, c2);
                    process_stack(vv_stack[tid], vv_edges[tid]);
                }
            }
        }
        #pragma omp atomic
        n_edges += vv_edges[tid].size();
        v_t_edge_size[tid] = vv_edges[tid].size();
        vv_stack[tid].shrink_to_fit();
        #pragma omp barrier

        #pragma omp single
        {
            v_t_edge_offset[0] = 0;
            for (auto t = 1; t < n_threads; ++t) {
                v_t_edge_offset[t] = v_t_edge_offset[t-1] + v_t_edge_size[t-1];
            }
            v_edges.resize(n_edges);
        }
        std::move(vv_edges[tid].begin(), vv_edges[tid].end(), std::next(v_edges.begin(), v_t_edge_offset[tid]));
    }


//    for (auto t = 0; t < n_threads; ++t) {
//        std::move(vv_edges[t].begin(), vv_edges[t].end(), std::next(v_edges.begin(), v_t_edge_offset[t]));
//    }
}

void nc_tree_new::process_tree_node(std::vector<cell_meta_pair_level> &v_stack, s_vec<long> &v_edges,
    long const l, long const c) noexcept {
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
                process_stack(v_stack, v_edges);
            }
        }
    }
}

void nc_tree_new::process_stack(std::vector<cell_meta_pair_level> &v_stack, s_vec<long> &v_edges) noexcept {
    while (!v_stack.empty()) {
        auto li = v_stack.back().l;
        auto ci1 = v_stack.back().c1;
        auto ci2 = v_stack.back().c2;
        v_stack.pop_back();
        auto begin1 = vv_cell_begin[li][ci1];
        auto begin2 = vv_cell_begin[li][ci2];
        if (li == 0) {
            // Note: CUDA doesn't support emplace_back
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

