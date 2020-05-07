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
#ifndef NEXT_DBSCAN_NEXTDBSCAN_OMP_H
#define NEXT_DBSCAN_NEXTDBSCAN_OMP_H

#include <vector>
template <class T>
using s_vec = std::vector<T>;
template <class T>
using d_vec = std::vector<std::vector<T>>;
typedef unsigned int uint;
using t_uint_iterator = std::vector<std::vector<std::vector<uint>::iterator>>;
#include "nc_tree.h"
#include <functional>

template<class T, class O>
void _atomic_op(T* address, T value, O op) {
    T previous = __sync_fetch_and_add(address, 0);

    while (op(value, previous)) {
        if  (__sync_bool_compare_and_swap(address, previous, value)) {
            break;
        } else {
            previous = __sync_fetch_and_add(address, 0);
        }
    }
}

#endif //NEXT_DBSCAN_NEXTDBSCAN_OMP_H
