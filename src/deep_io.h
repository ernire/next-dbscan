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

#ifndef NEXT_DBSCAN_DEEP_IO_H
#define NEXT_DBSCAN_DEEP_IO_H

#include <istream>
#include <memory>
#include <fstream>
#include <iostream>
#include <cstring>
#include <vector>
#include <sstream>
#include <iterator>
#ifdef CUDA_ON
#include <thrust/host_vector.h>
#endif

#ifdef CUDA_ON
//template <class T>
//using s_vec = thrust::host_vector<T>;
//template <class T>
//using d_vec = thrust::host_vector<thrust::host_vector<T>>;
//using t_uint_iterator = thrust::host_vector<thrust::host_vector<thrust::host_vector<uint>::iterator>>;
#endif
#ifndef CUDA_ON
template <class T>
using s_vec = std::vector<T>;
template <class T>
using d_vec = std::vector<std::vector<T>>;
using t_uint_iterator = std::vector<std::vector<std::vector<uint>::iterator>>;
#endif

const int UNDEFINED_VALUE = -1;
typedef unsigned int uint;

class deep_io {
private:
    const char *file;
    const uint block_no, block_index, max_samples_per_batch;
    int feature_offset = UNDEFINED_VALUE;
    bool is_initialized = false;

    void load_meta_data(std::istream &is, s_vec<float> &v_samples) noexcept;

public:
    uint unread_samples;
    uint sample_no, feature_no, sample_read_no;
    uint block_sample_offset;

    deep_io(char *file, uint number_of_blocks, uint block_index) : deep_io(file, number_of_blocks, block_index,
            INT32_MAX) {}

    deep_io(const char *file, uint number_of_blocks, uint block_index, uint max_samples_per_batch)
            : file(file), block_no(number_of_blocks), block_index(block_index),
              max_samples_per_batch(max_samples_per_batch) {
        sample_no = 0;
        feature_no = 0;
        sample_read_no = 0;
        unread_samples = 0;
        block_sample_offset = 0;
    }

    ~deep_io() = default;

    int load_next_samples(s_vec<float> &v_samples) noexcept;

    static void count_lines_and_dimensions(const std::string &in_file, uint &lines, uint &dimensions) noexcept;

    static void get_blocks_meta(std::unique_ptr<uint[]> &v_sizes, std::unique_ptr<uint[]> &v_offsets,
            uint number_of_samples, uint number_of_blocks) noexcept;

    static uint get_block_size(uint block_index, uint number_of_samples, uint number_of_blocks) noexcept;

    static uint get_block_start_offset(uint block_index, uint number_of_samples, uint number_of_blocks) noexcept;
};

std::streampos get_file_size(const char *filePath);

#endif //NEXT_DBSCAN_DEEP_IO_H
