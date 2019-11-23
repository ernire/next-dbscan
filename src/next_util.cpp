//
// Created by Ernir Erlingsson on 23.11.2019.
//

#include <string>
#include <iostream>
#include "next_util.h"

template<class T>
void next_util::print_array(const std::string &name, T *arr, const uint max_d) noexcept {
    std::cout << name << ": ";
    for (int i = 0; i < max_d; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}