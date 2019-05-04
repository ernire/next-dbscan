//
// Created by Ernir Erlingsson on 17.4.2019.
//

#include <vector>
#include <stdlib.h>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <random>
#include "assert.h"
#include "omp.h"

double randZeroToOne() {
    return rand() / (RAND_MAX + 1.);
}

int randZeroTo500k() {
    return rand() % 5000000;
}

void merge_pairs_omp(std::vector<std::pair<size_t, int>> v1, std::vector<std::pair<size_t, int>> v2,
        std::vector<std::pair<size_t, int>> v_new1, std::vector<std::pair<size_t, int>> v_new2) {
    int size1 = v1.size();
    int size2 = v2.size();
    #pragma omp parallel num_threads(2)
    {
        int i1 = 0;
        int i2 = 0;
        int t_id = omp_get_thread_num();
        int begin_t = t_id == 0? 0 : size1 / 2;
        int end_t = t_id == 0? size1 / 2 : size1;
        /*
         * size_t begin_val_t = vec_index_maps_t[t1][begin_t].first;
                    int end_t = (t_id == num_inner_threads - 1) ? size1 : size1 / num_inner_threads * (t_id + 1);
                    while (begin_t > 0 && begin_val_t == vec_index_maps_t[t1][begin_t - 1].first) {
                        --begin_t;
                    }
                    size_t end_val_t = vec_index_maps_t[t1][end_t].first;
                    if (end_t != size1) {
                        while (end_t > 0 && end_val_t == vec_index_maps_t[t1][end_t - 1].first) {
                            --end_t;
                        }
                    }
                    if (begin_t == end_t)
                        continue;
         */
    }
}

void merge_pairs_2(std::vector<std::pair<size_t, int>> v1, std::vector<std::pair<size_t, int>> v2,
        std::vector<std::pair<size_t, int>> v_new) {
    int size1 = v1.size();
    int size2 = v2.size();
    int i1 = 0;
    int i2 = 0;
    while (i1 < size1 && i2 < size2) {
        if (i1 == size1) {
            v_new.insert(v_new.end(), std::next(v2.begin(), i2), v2.end());
            i2 = size2;
        } else if (i2 == size2) {
            v_new.insert(v_new.end(), std::next(v1.begin(), i1), v1.end());
            i1 = size1;
        } else {
            if (v1[i1].first == v2[i2].first) {
                v_new.emplace_back(v1[i1].first, v1[i1].second + v2[i2].second);
                ++i1;
                ++i2;
            } else if (v1[i1].first < v2[i2].first) {
                v_new.emplace_back(v1[i1].first, v1[i1].second);
                ++i1;
            } else { //vec_index_maps_t[t1][i1].first > vec_index_maps_t[t2][i2].first
                v_new.emplace_back(v2[i2].first, v2[i2].second);
                ++i2;
            }
        }
    }
//    while (i1 < size1 && i2 < size2) {
        /*
        if (i1 == size1) {
            vec_new.push_back(vec_index_maps_t[t2][i2]);
            ++i2;
        } else if (i2 == size2) {
            vec_new.push_back(vec_index_maps_t[t1][i1]);
            ++i1;
        } else {
            if (vec_index_maps_t[t1][i1].first == vec_index_maps_t[t2][i2].first) {
                vec_new.push_back(vec_index_maps_t[t1][i1]);
                vec_new.push_back(vec_index_maps_t[t2][i2]);
                ++i1;
                ++i2;
            } else if (vec_index_maps_t[t1][i1].first < vec_index_maps_t[t2][i2].first) {
                vec_new.push_back(vec_index_maps_t[t1][i1]);
                ++i1;
            } else { //vec_index_maps_t[t1][i1].first > vec_index_maps_t[t2][i2].first
                vec_new.push_back(vec_index_maps_t[t2][i2]);
                ++i2;
            }
        }
         */
//    }/
}

void merge_pairs_1(std::vector<std::pair<size_t, int>> v1, std::vector<std::pair<size_t, int>> v2) {
    int size1 = v1.size();
    int size2 = v2.size();
    int i1 = 0;
    int i2 = 0;
    while (i1 < size1 && i2 < size2) {
        if (i1 == size1) {
            v1.insert(v1.end(), std::next(v2.begin(), i2), v2.end());
            i2 = size2;
        } else if (i2 == size2) {
            i1 = size1;
        } else {
            if (v1[i1].first == v2[i2].first) {
                v1[i1].second += v2[i2].second;
                ++i1;
                ++i2;
            } else if (v1[i1].first < v2[i2].first) {
                ++i1;
            } else { //vec_index_maps_t[t1][i1].first > vec_index_maps_t[t2][i2].first
                v1.insert(std::next(v1.begin(), i1), v2[i2]);
                ++i1;
                ++i2;
            }
        }
    }
}

void confirm_vector_count(std::vector<std::pair<size_t, int>> v, int count) {
    int cnt = 0;
    for (auto& p : v) {
        cnt += p.second;
    }
    if (cnt != count) {
        std::cout << "Failed check: " << cnt << " : " << count << std::endl;
    }
    assert(cnt == count);
}

int main() {
    int n = 10000000;
    int n_threads = 8;
    omp_set_num_threads(n_threads);
//    std::vector<std::pair<size_t, int>> vec_unique_count[n_threads];
//    std::vector<std::pair<size_t, int>> vec_index_maps_t[n_threads];
//    std::vector<std::pair<size_t, int>> vec_new[n_threads];
//    for (int t = 0; t < n_threads; t++) {
//        vec_unique_count[t].reserve(n/n_threads);
//        vec_index_maps_t[t].reserve(n/n_threads);
//        vec_new[t].reserve(n/n_threads);
//    }

    std::random_device rd;     //Get a random seed from the OS entropy device, or whatever
    std::mt19937_64 eng(rd()); //Use the 64-bit Mersenne Twister 19937 generator
    //and seed it with entropy.

    //Define the distribution, by default it goes from 0 to MAX(unsigned long long)
    //or what have you.
    std::uniform_int_distribution<unsigned long long> distr;

    auto *vec_index_maps = new std::pair<size_t, size_t>[n];
    auto *rands = new unsigned long long[n];
    for (int i = 0; i < n; i++) {
//        rands[i] = randZeroTo500k();
        rands[i] = distr(eng);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
//        int t_id = omp_get_thread_num();
//        vec_index_maps.emplace_back(rands[i], i);
//        #pragma omp critical
//        std::cout << "i: " << i << std::endl;
        vec_index_maps[i] = std::make_pair(rands[i], i);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Initial processing: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
              << " milliseconds\n";

    auto *is_processed = new bool[n] {false};
    int total_count = 0;
    auto *cnts = new int[64] {0};
    unsigned long long vals[64];
    for (unsigned int i = 0; i < 64; i++) {
        vals[i] = 1u << i;
    }
    t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++) {
//        std::cout << "i: " << i << std::endl;
        for (unsigned int j = 0; j < 64; j++) {
            if ((vals[63-j] & vec_index_maps[i].first) > 0) {
                ++cnts[j];
                j = 64;
//                break;
            }
        }
    }
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Method 1: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
              << " milliseconds\n";
//    for (unsigned int j = 63; j >= 0; j--) {
//        total_count += cnts[j];
//    }
    /*
    t1 = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < 64; i++) {
        int cnt = 0;
        for (int j = 0; j < n; j++) {
            if (is_processed[j])
                continue;
            if ((vals[63-i] & vec_index_maps[j].first) > 0) {
                is_processed[j] = true;
                ++cnt;
            }
        }
        total_count += cnt;
//        if (cnt > 0)
//            std::cout << "Level: " << i << " CNT: " << cnt << std::endl;
    }
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Method 2: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
              << " milliseconds\n";
    */
//    assert(total_count == n);
    /*
    int sum = 0;
    for (int t = 0; t < n_threads; t++) {
        sum += vec_index_maps_t[t].size();
    }
    assert(sum == n);
    t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int t = 0; t < n_threads; t++) {
        std::sort(vec_index_maps_t[t].begin(), vec_index_maps_t[t].end());
    }
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Sorting: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
              << " milliseconds\n";
    t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int t = 0; t < n_threads; t++) {
        size_t last_index = vec_index_maps_t[t][0].first;
        int cnt = 1;
        for (int i = 1; i < vec_index_maps_t[t].size(); i++) {
            if (vec_index_maps_t[t][i].first != last_index) {
                vec_unique_count[t].emplace_back(last_index, cnt);
                last_index = vec_index_maps_t[t][i].first;
                cnt = 1;
            } else {
                ++cnt;
            }
        }
        vec_unique_count[t].emplace_back(last_index, cnt);
    }
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Counting: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
              << " milliseconds\n";

    sum = 0;
    for (int t = 0; t < n_threads; t++) {
        std::cout << "unique t: " << t << " is " << vec_unique_count[t].size() << std::endl;
        for (auto& p : vec_unique_count[t]) {
            sum += p.second;
        }
    }
    if (sum != n) {
        std::cout << "ERROR sum not n: " << sum << " " << n << std::endl;
    }
    assert(sum == n);
    std::vector<int> indexes, store;
    indexes.reserve(n_threads);
    store.reserve(n_threads);
    for (int i = 0; i < n_threads; i++) {
        indexes.push_back(i);
    }
    omp_set_nested(1);
    auto t11 = std::chrono::high_resolution_clock::now();
    while (indexes.size() > 1) {
        for (int i = 0; i < indexes.size(); i += 2) {
            store.push_back(indexes[i]);
        }
        auto time1 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for
        for (int t = 0; t < indexes.size(); t += 2) {
            int t_id1 = indexes[t];
            int t_id2 = indexes[t + 1];
            #pragma omp critical
            std::cout << "Merging " << t_id1 << " and " << t_id2 << std::endl;
            merge_pairs_omp(vec_unique_count[t_id1], vec_unique_count[t_id2], vec_new[t_id1], vec_new[t_id2]);
            vec_unique_count[t_id2].clear();
//        std::cout << "0 size: " << vec_unique_count[0].size() << std::endl;
        }
        auto time2 = std::chrono::high_resolution_clock::now();
        std::cout << "Merge: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1).count()
                  << " milliseconds\n";
        indexes = store;
        store.clear();
    }
    auto t22 = std::chrono::high_resolution_clock::now();
    std::cout << "Merge Total: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t22 - t11).count()
              << " milliseconds\n";

    // sanity check
//    confirm_vector_count(vec_unique_count[0], n);
//
     */
    /*
     *         size_t last_index = vec_index_map[0].first;
        for (int i = 1; i < vec_index_map.size(); i++) {
            if (vec_index_map[i].first != last_index) {
                vec_begin_indexes.push_back(i);
                vec_unique_count.emplace_back(last_index, cnt);
                last_index = vec_index_map[i].first;
                cnt = 1;
            } else {
                ++cnt;
            }
        }
        vec_unique_count.emplace_back(last_index, cnt);
        v_no_of_cells[l] = vec_unique_count.size();
     */
}