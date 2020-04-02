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

#include <cstdint>
#include <iostream>
#include <cassert>
#include <chrono>
#include <omp.h>
#include <stack>
#include <unordered_set>
#include "nc_tree.h"
#ifdef CUDA_ON
#include "nextdbscan_cuda.h"
#endif
#ifndef CUDA_ON
#include "nextdbscan_omp.h"
#endif
#include "next_util.h"


// TODO remove this when not needed anymore
//inline bool dist_leq(const float *coord1, const float *coord2, const int max_d, const float e2) noexcept {
//    float tmp = 0;
//    #pragma unroll
//    for (int d = 0; d < max_d; d++) {
//        float tmp2 = coord1[d] - coord2[d];
//        tmp += tmp2 * tmp2;
//    }
//    return tmp <= e2;
//}



void nc_tree::calc_bounds(float *min_bounds, float *max_bounds) noexcept {
    for (uint d = 0; d < n_dim; d++) {
        min_bounds[d] = INT32_MAX;
        max_bounds[d] = INT32_MIN;
    }
    #pragma omp parallel for reduction(max:max_bounds[:n_dim]) reduction(min:min_bounds[:n_dim])
    for (uint i = 0; i < n_coords; i++) {
        size_t index = i * n_dim;
        for (uint d = 0; d < n_dim; d++) {
            if (v_coords[index + d] > max_bounds[d]) {
                max_bounds[d] = v_coords[index + d];
            }
            if (v_coords[index + d] < min_bounds[d]) {
                min_bounds[d] = v_coords[index + d];
            }
        }
    }
}

uint nc_tree::determine_data_boundaries() noexcept {
    float max_limit = INT32_MIN;
    calc_bounds(&v_min_bounds[0], &v_max_bounds[0]);
#ifdef MPI_ON
        auto v_global_min_bounds = std::make_unique<float[]>(max_d);
        auto v_global_max_bounds = std::make_unique<float[]>(max_d);
        MPI_Allreduce(&v_min_bounds[0], &v_global_min_bounds[0], max_d, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&v_max_bounds[0], &v_global_max_bounds[0], max_d, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
        std::copy(&v_global_min_bounds[0], &v_global_min_bounds[max_d], &v_min_bounds[0]);
        std::copy(&v_global_max_bounds[0], &v_global_max_bounds[max_d], &v_max_bounds[0]);
#endif
    #pragma omp parallel for reduction(max: max_limit)
    for (uint d = 0; d < n_dim; d++) {
        if (v_max_bounds[d] - v_min_bounds[d] > max_limit)
            max_limit = v_max_bounds[d] - v_min_bounds[d];
    }
    return static_cast<unsigned int>(ceilf(logf(max_limit / e_inner) / logf(2))) + 1;
}

void nc_tree::build_tree() noexcept {
    s_vec<float> v_eps_levels(n_level);
    #pragma omp parallel for
    for (uint l = 0; l < n_level; l++) {
        v_eps_levels[l] = (e_inner * pow(2, l));
    }
    index_points(v_eps_levels);
}

void nc_tree::init() noexcept {
    v_min_bounds.resize(n_dim);
    v_max_bounds.resize(n_dim);
    n_level = determine_data_boundaries();
    vv_index_map.resize(n_level);
    vv_cell_begin.resize(n_level);
    vv_cell_ns.resize(n_level);
    vv_min_cell_dim.resize(n_level);
    vv_max_cell_dim.resize(n_level);
}

void nc_tree::collect_all_permutations(const int n_partitions) {
    std::cout << std::endl;
    s_vec<uint32_t> v_primes;
    next_util::get_small_prime_factors(v_primes, n_partitions);
    next_util::print_array("primes: ", &v_primes[0], v_primes.size());
    s_vec<size_t> v_base;
    s_vec<size_t> v_perm;
    const auto n_comb_depth = std::min(v_primes.size(), n_dim);
    next_util::get_permutation_base(v_base, v_primes.size(), n_comb_depth);
    // TODO parallelize ?
    for (auto i = 0; i < v_base.size(); i += v_primes.size()) {
        next_util::collect_permutations(v_perm, std::next(v_base.begin(), i),
                std::next(v_base.begin(), i + v_primes.size()));
    }
    std::cout << "perms size: " << v_perm.size() / v_primes.size() << std::endl;

    s_vec<uint32_t> v_mult(v_primes.size());
    for (auto i = 0; i < v_perm.size(); i += v_primes.size()) {
        std::fill(v_mult.begin(), v_mult.end(), 1);
        for (auto j = 0; j < v_primes.size(); ++j) {
            v_mult[v_perm[i+j]] *= v_primes[j];
        }
        // TODO copy
        for (auto j = 0; j < v_primes.size(); ++j) {
            v_perm[i+j] = v_mult[j];
        }
    }

    std::unordered_set<std::string> unique_set;
    s_vec<size_t> v_unique_perm;
    v_unique_perm.reserve(v_perm.size()/2);


    for (auto i = 0; i < v_perm.size(); i += v_primes.size()) {
        std::string s;
        for (auto j = 0; j < v_primes.size(); ++j) {
            s.append(std::to_string(v_perm[i+j]*10));
        }
        if (unique_set.find(s) == unique_set.end()) {
            unique_set.insert(s);
            // TODO copy
            for (auto j = 0; j < v_primes.size(); ++j) {
                v_unique_perm.push_back(v_perm[i+j]);
            }
        }
//        std::cout << "s: " << s << std::endl;

    }
    std::cout << "Unique Perm size: " << v_unique_perm.size() / v_primes.size() << std::endl;

    s_vec<size_t> v_combination_index(v_unique_perm.size() / v_primes.size());
    std::iota(v_combination_index.begin(), v_combination_index.end(), 0);
    const auto comb_size = v_primes.size();
    next_util::print_vector("perm index: ", v_combination_index);
    std::sort(v_combination_index.begin(), v_combination_index.end(), [&] (const auto &i1, const auto &i2) -> bool {
        auto index1 = i1 * comb_size;
        auto index2 = i2 * comb_size;
        for (size_t j = 0; j < comb_size; ++j) {
            if (v_unique_perm[index1+j] > v_unique_perm[index2+j])
                return true;
            else if (v_unique_perm[index1+j] < v_unique_perm[index2+j])
                return false;
        }
        return false;
    });
    next_util::print_vector("combination index: ", v_combination_index);

    for (size_t i = 0; i < v_combination_index.size(); ++i) {
        auto index = v_combination_index[i] * comb_size;
        for (size_t j = 0; j < comb_size; ++j) {
            std::cout << v_unique_perm[index+j] << " ";
        }
        std::cout << std::endl;
    }
}

void nc_tree::partition_data(const int n_partitions) noexcept {

    collect_all_permutations(n_partitions);
//    s_vec<uint32_t> v_primes;
//    next_util::get_small_prime_factors(v_primes, n_partitions);
//    next_util::print_array("primes: ", &v_primes[0], v_primes.size());
//    s_vec<size_t> v_base;
//    s_vec<size_t> v_perm;
//    s_vec<uint32_t> v_perms(v_primes.size(), 0);

//    const auto n_comb_depth = std::min(v_primes.size(), n_dim);
//    next_util::get_permutation_base(v_base, v_primes.size(), n_comb_depth);
//    // TODO parallelize ?
//    for (auto i = 0; i < v_base.size(); i += v_primes.size()) {
//        next_util::collect_permutations(v_perm, std::next(v_base.begin(), i),
//                std::next(v_base.begin(), i + v_primes.size()));
//    }
//    std::cout << "perms size: " << v_perm.size() / v_primes.size() << std::endl;

//    s_vec<uint32_t> v_mult(v_primes.size());
//    for (auto i = 0; i < v_perm.size(); i += v_primes.size()) {
//        std::fill(v_mult.begin(), v_mult.end(), 1);
//        for (auto j = 0; j < v_primes.size(); ++j) {
//            v_mult[v_perm[i+j]] *= v_primes[j];
//        }
//        // TODO copy
//        for (auto j = 0; j < v_primes.size(); ++j) {
//            v_perm[i+j] = v_mult[j];
//        }
//    }
//
//    std::unordered_set<std::string> unique_set;
//    s_vec<size_t> v_unique_perm;
//    v_unique_perm.reserve(v_perm.size()/2);
//
//
//    for (auto i = 0; i < v_perm.size(); i += v_primes.size()) {
//        std::string s;
//        for (auto j = 0; j < v_primes.size(); ++j) {
//            s.append(std::to_string(v_perm[i+j]*10));
//        }
//        if (unique_set.find(s) == unique_set.end()) {
//            unique_set.insert(s);
//            // TODO copy
//            for (auto j = 0; j < v_primes.size(); ++j) {
//                v_unique_perm.push_back(v_perm[i+j]);
//            }
//        }
////        std::cout << "s: " << s << std::endl;
//
//    }
//
//    std::cout << "Unique Perm size: " << v_unique_perm.size() / v_primes.size() << std::endl;

//    s_vec<size_t> v_combination_index(v_unique_perm.size() / v_primes.size());
//    std::iota(v_combination_index.begin(), v_combination_index.end(), 0);
//    const auto comb_size = v_primes.size();
//    next_util::print_vector("perm index: ", v_combination_index);
//    std::sort(v_combination_index.begin(), v_combination_index.end(), [&] (const auto &i1, const auto &i2) -> bool {
//        auto index1 = i1 * comb_size;
//        auto index2 = i2 * comb_size;
//        for (size_t j = 0; j < comb_size; ++j) {
//            if (v_unique_perm[index1+j] > v_unique_perm[index2+j])
//                return true;
//            else if (v_unique_perm[index1+j] < v_unique_perm[index2+j])
//                return false;
//        }
//        return false;
//    });
//    next_util::print_vector("combination index: ", v_combination_index);
//
//    for (size_t i = 0; i < v_combination_index.size(); ++i) {
//        auto index = v_combination_index[i] * comb_size;
//        for (size_t j = 0; j < comb_size; ++j) {
//            std::cout << v_unique_perm[index+j] << " ";
//        }
//        std::cout << std::endl;
//    }
}


/*
void nc_tree::partition_data(const int n_partitions) noexcept {
    std::cout << std::endl;
    std::cout << "nc tree height: " << n_level << std::endl;
    auto e_part = e_inner;//*powf(2, (float)n_level/3);
    s_vec<uint32_t> v_primes;
    next_util::get_small_prime_factors(v_primes, n_partitions);
    next_util::print_array("primes: ", &v_primes[0], v_primes.size());

    s_vec<uint32_t> v_perms(v_primes.size(), 0);
    s_vec<uint32_t> v_label_cnt(v_primes.size()-1, 0);
    s_vec<uint32_t> v_mults;
    v_mults.reserve(v_primes.size() * v_primes.size());

    s_vec<size_t> v_base;
    s_vec<size_t> v_perm;
    const auto n_comb_depth = std::min(v_primes.size(), n_dim);
    next_util::get_permutation_base(v_base, v_primes.size(), n_comb_depth);
    for (auto i = 0; i < v_base.size(); i += v_primes.size()) {
        next_util::collect_permutations(v_perm, std::next(v_base.begin(), i),
                std::next(v_base.begin(), i + v_primes.size()));
    }
    std::cout << "perms size: " << v_perm.size() / v_primes.size() << std::endl;
    s_vec<uint32_t> v_mult(v_primes.size());
    for (auto i = 0; i < v_perm.size(); i += v_primes.size()) {
        std::fill(v_mult.begin(), v_mult.end(), 1);
        for (auto j = 0; j < v_primes.size(); ++j) {
            v_mult[v_perm[i+j]] *= v_primes[j];
        }
        // TODO copy
        for (auto j = 0; j < v_primes.size(); ++j) {
            v_perm[i+j] = v_mult[j];
        }
    }
    std::unordered_set<std::string> unique_set;
    s_vec<size_t> v_unique_perm;
    v_unique_perm.reserve(v_perm.size()/2);
//    v_perm.clear();
    for (auto i = 0; i < v_perm.size(); i += v_primes.size()) {
        std::string s;
        for (auto j = 0; j < v_primes.size(); ++j) {
            s.append(std::to_string(v_perm[i+j]*10));
        }
        if (unique_set.find(s) == unique_set.end()) {
            unique_set.insert(s);
            // TODO copy
            for (auto j = 0; j < v_primes.size(); ++j) {
                v_unique_perm.push_back(v_perm[i+j]);
            }
        }
//        std::cout << "s: " << s << std::endl;

    }
    std::cout << "Unique Perm size: " << v_unique_perm.size() / v_primes.size() << std::endl;
    s_vec<size_t> v_combination_index(v_unique_perm.size() / v_primes.size());
    std::iota(v_combination_index.begin(), v_combination_index.end(), 0);
    const auto comb_size = v_primes.size();
    next_util::print_vector("perm index: ", v_combination_index);
    std::sort(v_combination_index.begin(), v_combination_index.end(), [&] (const auto &i1, const auto &i2) -> bool {
        auto index1 = i1 * comb_size;
        auto index2 = i2 * comb_size;
        for (size_t j = 0; j < comb_size; ++j) {
            if (v_unique_perm[index1+j] > v_unique_perm[index2+j])
                return true;
            else if (v_unique_perm[index1+j] < v_unique_perm[index2+j])
                return false;
        }
        return false;
    });
    next_util::print_vector("combination index: ", v_combination_index);

    for (size_t i = 0; i < v_combination_index.size(); ++i) {
        auto index = v_combination_index[i] * comb_size;
        for (size_t j = 0; j < comb_size; ++j) {
            std::cout << v_unique_perm[index+j] << " ";
        }
        std::cout << std::endl;
    }

    s_vec<size_t> v_depth_comb_size(n_comb_depth, 0);
    v_depth_comb_size[0] = 1;
    #pragma omp parallel for collapse(2)
    for (size_t depth = 1; depth < n_comb_depth; ++depth) {
        for (size_t i = 0; i < v_combination_index.size(); ++i) {
            auto index = v_combination_index[i] * comb_size;
            if (v_unique_perm[index + depth] > 1) {
                auto cnt = 1;
                for (size_t d = depth; d > 0; --d) {
                    cnt *= v_unique_perm[index + d - 1];
                }
                #pragma omp atomic
                v_depth_comb_size[depth] += cnt;
            }
        }
    }
//    const uint n_sample_size = n_partitions*ceil(log10(n_coords));
    const uint n_sample_size = n_partitions*n_partitions*ceil(log10(n_coords));

//    const uint n_sample_size = n_partitions*log10(n_coords);
//    next_util::print_vector("combination sizes: ", v_depth_comb_size);
//    d_vec<size_t> vv_comb_cnts(n_comb_depth);
//    for (size_t i = 0; i < vv_comb_cnts.size(); ++i) {
//        vv_comb_cnts[i].resize(v_depth_comb_size[i]*n_sample_size, 0);
//    }



    // SAMPLING

    s_vec<size_t> v_rand;
    next_util::random_vector_no_repeat(v_rand, n_sample_size, n_coords);
    std::cout << "Random size: " << v_rand.size() << std::endl;

    s_vec<float_t> v_rand_coords(v_rand.size() * n_dim, -1);
    // TODO parallel
    for (size_t d = 0, n = 0; d < n_dim; ++d) {
        for (size_t i = 0; i < v_rand.size(); ++i, ++n) {
            auto index = (v_rand[i] * n_dim) + d;
            auto snap_index = std::roundf((v_coords[index]) / e_inner);
            v_rand_coords[n] = snap_index * e_inner;
//            v_rand_coords[n] = v_coords[index];
        }
        assert(n <= v_rand_coords.size());
    }

    // TODO remove
    for (auto val : v_rand_coords) {
        assert(val != -1);
    }
    for (size_t d = 0; d < n_dim; ++d) {
        std::sort(std::next(v_rand_coords.begin(), d*v_rand.size()), std::next(v_rand_coords.begin(), (d+1)*v_rand.size()));
    }
    for (size_t i = 0; i < v_rand_coords.size(); ++i) {
        if (i % (v_rand_coords.size() / n_dim) != 0) {
            assert(v_rand_coords[i] >= v_rand_coords[i-1]);
        }
    }

    const auto dim_rand_size = v_rand_coords.size() / n_dim;
    s_vec<size_t> v_coord_bucket(n_coords*n_dim);
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < n_coords; ++i) {
        for (size_t d = 0; d < n_dim; ++d) {
            auto coord_index = (i*n_dim)+d;
            auto begin = std::next(v_rand_coords.begin(), d * dim_rand_size);
            auto end = std::next(v_rand_coords.begin(), (d+1) * dim_rand_size);
            auto low = std::lower_bound (begin, end, v_coords[coord_index]);
            v_coord_bucket[coord_index] = (low - begin);
        }
    }

    // TOGETHER
    const auto n_bucket_size = n_sample_size+1;
    s_vec<size_t> v_bucket_cnt;
    for (size_t i = 0; i < 1; ++i) {
        const auto no_of_buckets = v_depth_comb_size[i]*(n_dim-i);
        v_bucket_cnt.clear();
        v_bucket_cnt.resize(n_bucket_size * no_of_buckets, 0);
        std::cout << "v_bucket_cnt size: " << v_bucket_cnt.size() << std::endl;
        for (size_t j = 0; j < n_coords; ++j) {
            for (size_t d = 0; d < n_dim; ++d) {
                auto coord_index = (j*n_dim)+d;
                // TODO eliminate dims
                ++v_bucket_cnt[n_bucket_size*d + v_coord_bucket[coord_index]];
            }
        }
        size_t cnt = 0;
        for (size_t j = 0; j < v_bucket_cnt.size(); ++j) {
            if (j % n_bucket_size == 0) {
                if (j > 0) {
                    assert(cnt == n_coords);
                }
                cnt = 0;
            }
            cnt += v_bucket_cnt[j];
        }
        // split
    }

    const double_t optimal_score = (double_t)n_coords / n_partitions;
    for (size_t i = 0; i < 2; ++i) {
        auto index = v_combination_index[i] * n_comb_depth;
        s_vec<size_t> v_sample_cnt(n_bucket_size*n_dim, 0);
        s_vec<size_t> v_sample_split;
        s_vec<double_t> v_sample_score(n_dim, 0);
        for (size_t j = 0; j < 1; ++j) {
//            std::cout << v_unique_perm[index + j] << " ";
            size_t n_split = v_unique_perm[index+j];
            if (n_split == 1) {
                continue;
            } else {
//                v_sample_split.resize(n_split*n_dim, 0);
            }
            // todo higher dims
            for (size_t k = 0; k < n_coords; ++k) {
                for (size_t d = 0; d < n_dim; ++d) {
                    auto coord_index = (k * n_dim) + d;
                    // TODO eliminate dims
                    assert(n_bucket_size*d + v_coord_bucket[coord_index] < v_sample_cnt.size());
                    ++v_sample_cnt[n_bucket_size*d + v_coord_bucket[coord_index]];
                }
            }
            // Validation
            size_t cnt = 0;
            for (size_t k = 0; k < v_sample_cnt.size(); ++k) {
                if (k % n_bucket_size == 0) {
                    if (k > 0) {
//                        std::cout << std::endl;
                        assert(cnt == n_coords);
                    }
                    cnt = 0;
                }
                cnt += v_sample_cnt[k];
//                std::cout << v_sample_cnt[k] << " ";
            }
//            std::cout << std::endl;
            const double_t local_optimal_score = (double_t)n_coords / n_split;
            std::cout << "local optimal score: " << local_optimal_score << std::endl;
            // score
//            std::cout <<
            for (size_t d = 0; d < n_dim; ++d) {
                std::cout << "Dimension: " << d << " split " << n_split << std::endl;
                double_t curr_cnt = 0;
                auto bucket_index = (d * n_bucket_size);
                for (size_t k = 0; k < n_bucket_size && v_sample_split.size() < n_split-1; ++k) {
                    // TODO skip last bucket
                    if (curr_cnt + v_sample_cnt[bucket_index + k] > local_optimal_score) {
//                        assert(curr_cnt <= local_optimal_score);
                        auto score_prev = local_optimal_score - curr_cnt;
                        auto score_curr = (curr_cnt + v_sample_cnt[bucket_index + k]) - local_optimal_score;
                        if (score_prev < score_curr) {
                            v_sample_split.push_back(k);
                            curr_cnt = score_prev + v_sample_cnt[bucket_index + k];
                        } else {
                            v_sample_split.push_back(k+1);
                            curr_cnt = -score_curr;
                        }
//                        curr_cnt = curr_cnt - local_optimal_score;
                    } else {
                        curr_cnt += v_sample_cnt[bucket_index + k];
                    }
                }

                next_util::print_vector("splits: ", v_sample_split);

                if (v_sample_split.size() < n_split-1) {
                    std::cout << "FAILED" << std::endl;
                    continue;
                }

                uint64_t sum = 0;
                for (size_t k = 0, s = 0; k < n_bucket_size; ++k) {
                    if (k == v_sample_split[s]) {
                        std::cout << sum << " : ";
                        sum = 0;
                        ++s;
                    }
                    sum += v_sample_cnt[bucket_index+k];
                }
                std::cout << sum << std::endl;

                sum = 0;
                double_t score = 0;
                for (size_t k = 0, s = 0; k < n_bucket_size; ++k) {
                    if (k == v_sample_split[s]) {
                        auto tmp = ((double_t)sum - local_optimal_score) / local_optimal_score;
                        score += (tmp*tmp);
                        std::cout << (tmp*tmp) << " : ";
                        sum = 0;
                        ++s;
                    }
                    sum += v_sample_cnt[bucket_index+k];
                }
                auto tmp = ((double_t)sum - local_optimal_score) / local_optimal_score;
                std::cout << (tmp*tmp) << " , score: " << score << std::endl;

                v_sample_split.clear();
//                std::cout << std::endl;
            }
        }
//        std::cout << std::endl;
    }

    for (size_t d = 0; d < n_comb_depth; ++d) {
        // count the number of splits for the current dim
        for (size_t i = 0; i < v_combination_index.size(); ++i) {
            auto index = v_combination_index[i] * n_comb_depth + d;
//            std::cout << v_unique_perm[index] << " ";
        }
//        std::cout << std::endl;
        // count the number of combinations for the acurrent dim

        // count the buckets

        // split the buckets

    }
}
 */



