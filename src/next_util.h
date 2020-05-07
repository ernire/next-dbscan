//
// Created by Ernir Erlingsson on 23.11.2019.
//

#ifndef NEXT_DBSCAN_NEXT_UTIL_H
#define NEXT_DBSCAN_NEXT_UTIL_H

#include <string>
#include <iostream>
#include <algorithm>
#include <stack>
#include <random>
#include "nc_tree_new.h"

template<typename T>
using random_distribution = std::conditional_t<std::is_integral<T>::value,
        std::uniform_int_distribution<T>,
        std::conditional_t<std::is_floating_point<T>::value,
                std::uniform_real_distribution<T>,
                void>
>;

class next_util {
public:

    template<typename T>
    static void fill_offsets(s_vec<T> &v_offset, s_vec<T> &v_size) noexcept {
        v_offset[0] = 0;
        for (size_t i = 1; i < v_offset.size(); ++i) {
            v_offset[i] = v_offset[i-1] + v_size[i-1];
        }
    }

    template<typename T, typename K>
    static void print_value_vector(const std::string &name, s_vec<T> &v_val_vec, s_vec<K> &v_index_vec) noexcept {
        std::cout << name << ": ";
        for (int i = 0; i < v_index_vec.size(); ++i) {
            std::cout << v_val_vec[v_index_vec[i]] << " ";
        }
        std::cout << std::endl;
    }

    template<class T>
    static void print_vector(const std::string &name, s_vec<T> &v_vec) noexcept {
        std::cout << name << ": ";
        for (int i = 0; i < v_vec.size(); ++i) {
            std::cout << v_vec[i] << " ";
        }
        std::cout << std::endl;
    }

    template<class T>
    static void print_array(const std::string &name, T *arr, uint32_t size) noexcept {
        std::cout << name << ": ";
        for (int i = 0; i < size; ++i) {
            std::cout << arr[i] << " ";
        }
        std::cout << std::endl;
    }

    template<class T>
    static T sum_array(T *arr, uint32_t size) noexcept {
        T sum = 0;
        for (uint32_t i = 0; i < size; ++i) {
            sum += arr[i];
        }
        return sum;
    }

    template<class T>
    static T sum_array_omp(T *arr, uint32_t size) noexcept {
        T sum = 0;
        #pragma omp parallel for reduction(+:sum)
        for (uint32_t i = 0; i < size; ++i) {
            sum += arr[i];
        }
        return sum;
    }

    template <class T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    static void random_vector_no_repeat(s_vec<T> &vec, const size_t vec_size, const size_t pool_size) noexcept {
//        std::default_random_engine generator(std::random_device{}());
        // TODO not constant seed value
        std::default_random_engine generator(12345);
        random_distribution<T> rnd_dist(0, pool_size);
        auto rnd_gen = std::bind(rnd_dist, generator);
        auto start = vec.size();
        vec.resize(vec_size);
        for (size_t i = start; i < vec.size(); ++i) {
            T tmp = rnd_gen();
            bool insert = true;
            for (auto val : vec) {
                if (val == tmp) {
                    insert = false;
                    break;
                }
            }
            if (insert)
                vec[i] = rnd_gen();
            else
                --i;
        }
    }

    // 2, 3, 5, 7 only
    static bool small_prime_factor(std::vector<uint32_t> &v_prime_cnt, uint32_t number) {
        v_prime_cnt.resize(4, 0);
        int primes[] {2,3,5,7};
        for (int i = 0; i < 4; ++i) {
            while (number % primes[i] == 0) {
                number /= primes[i];
                ++v_prime_cnt[i];
            }
        }
        return number <= 1;
    }

    static void get_split_offsets(std::vector<uint32_t> &v_offsets, uint32_t number, uint32_t split_size) {
        v_offsets.resize(split_size, 0);
        uint32_t block = number / split_size;
        uint32_t rest = number % split_size;
        v_offsets[0] = 0;
        for (uint32_t i = 1; i < split_size; ++i) {
            v_offsets[i] = v_offsets[i-1] + block;
            if (rest > 0) {
                ++v_offsets[i];
                --rest;
            }
        }
    }

    static void collect_permutations(s_vec<size_t> &collector, s_vec<size_t>::iterator begin, s_vec<size_t>::iterator end) {
        std::sort(begin, end);
        do {
            std::copy(begin, end, back_inserter(collector));
        } while(std::next_permutation(begin, end));
    }

    static void get_permutation_base(s_vec<size_t> &v_all_perms, size_t places, size_t max) {
        std::stack<size_t> stack;
        s_vec<size_t> v_perms(places, 0);

        for (size_t d = 0; d < v_perms.size()-1; ++d) {
            std::fill(v_perms.begin(), v_perms.end(), 0);
            for (size_t i = 0; i <= std::min(d, max-1); ++i) {
                v_perms[i] = i;
            }

            for (auto val : v_perms) {
                v_all_perms.push_back(val);
            }
            stack.push(d+1);
            while (!stack.empty()) {
                size_t index = stack.top();
                if (v_perms[index] < std::min(d, max-1)) {
                    ++v_perms[index];
                } else {
                    v_perms[index] = 0;
                    stack.pop();
                    continue;
                }
                for (auto val : v_perms) {
                    v_all_perms.push_back(val);
                }
                if (index < v_perms.size()-1) {
                    stack.push(index+1);
                }
            }
        }
        for (size_t i = 0; i < std::min(places, max); ++i) {
            v_all_perms.push_back(i);
        }
    }

    static void get_small_prime_factors(std::vector<uint32_t> &v_primes, uint32_t number) {
        std::vector<uint32_t> v_prime_cnt;
        int primes[] {2,3,5,7,11,13,17};
        if (small_prime_factor(v_prime_cnt, number)) {
            for (int i = 0; i < 4; ++i) {
                while (v_prime_cnt[i] > 0) {
                    v_primes.push_back(primes[i]);
                    v_prime_cnt[i]--;
                }
            }
//            std::reverse(v_primes.begin(), v_primes.end());
        }
    }

    static void print_tree_meta_data(nc_tree_new &nc_tree) {
        std::cout << "NC-tree levels: " << nc_tree.n_level << std::endl;
        for (long l = 0; l < nc_tree.n_level; ++l) {
            std::cout << "Level: " << l << " has " << nc_tree.get_no_of_cells(l) << " cells" << std::endl;
        }
    }

};
#endif //NEXT_DBSCAN_NEXT_UTIL_H
