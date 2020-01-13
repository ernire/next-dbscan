//
// Created by Ernir Erlingsson on 23.11.2019.
//

#ifndef NEXT_DBSCAN_NEXT_UTIL_H
#define NEXT_DBSCAN_NEXT_UTIL_H

#include <string>
#include <iostream>

class next_util {
public:
    template<class T>
    static void print_array(const std::string &name, T *arr, uint max_d) noexcept {
        std::cout << name << ": ";
        for (int i = 0; i < max_d; ++i) {
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

    static void print_tree_meta_data(nc_tree &nc_tree) {
        std::cout << "NC-tree levels: " << nc_tree.n_level << std::endl;
        for (uint l = 0; l < nc_tree.n_level; ++l) {
            std::cout << "Level: " << l << " has " << nc_tree.get_no_of_cells(l) << " cells" << std::endl;
        }
    }
};
#endif //NEXT_DBSCAN_NEXT_UTIL_H
