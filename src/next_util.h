//
// Created by Ernir Erlingsson on 23.11.2019.
//

#ifndef NEXT_DBSCAN_NEXT_UTIL_H
#define NEXT_DBSCAN_NEXT_UTIL_H

#include <string>
#include <iostream>
#include <algorithm>

class next_util {
public:
    template<class T>
    static void print_array(const std::string &name, T *arr, uint n_dim) noexcept {
        std::cout << name << ": ";
        for (int i = 0; i < n_dim; ++i) {
            std::cout << arr[i] << " ";
        }
        std::cout << std::endl;
    }

    template<class T>
    static T sum_array(T *arr, uint size) noexcept {
        T sum = 0;
        for (uint i = 0; i < size; ++i) {
            sum += arr[i];
        }
        return sum;
    }

    // 2, 3, 5, 7 only
    static bool small_prime_factor(std::vector<uint> &v_prime_cnt, uint number) {
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

    static void get_small_prime_factors(std::vector<uint> &v_primes, uint number) {
        std::vector<uint> v_prime_cnt;
        int primes[] {2,3,5,7};
        if (small_prime_factor(v_prime_cnt, number)) {
            for (int i = 0; i < 4; ++i) {
                while (v_prime_cnt[i] > 0) {
                    v_primes.push_back(primes[i]);
                    v_prime_cnt[i]--;
                }
            }
            std::reverse(v_primes.begin(), v_primes.end());
        }
    }

    static void print_tree_meta_data(nc_tree &nc_tree) {
        std::cout << "NC-tree levels: " << nc_tree.n_level << std::endl;
        for (uint l = 0; l < nc_tree.n_level; ++l) {
            std::cout << "Level: " << l << " has " << nc_tree.get_no_of_cells(l) << " cells" << std::endl;
        }
    }
};
#endif //NEXT_DBSCAN_NEXT_UTIL_H
