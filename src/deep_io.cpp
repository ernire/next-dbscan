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
#include "deep_io.h"

void deep_io::count_lines_and_dimensions(const std::string &in_file, uint &lines, uint &dimensions) noexcept {
    std::ifstream is(in_file);
    std::string line, buf;
    int cnt = 0;
    dimensions = 0;
    while (std::getline(is, line)) {
        if (dimensions == 0) {
            std::istringstream iss(line);
            std::vector<std::string> results(std::istream_iterator<std::string>{iss},
                    std::istream_iterator<std::string>());
            dimensions = results.size();
        }
        ++cnt;
    }
    lines = cnt;
    is.close();
}

uint deep_io::get_block_size(const uint block_index, const uint number_of_samples,
        const uint number_of_blocks) noexcept {
    uint block = (number_of_samples / number_of_blocks);
    uint reserve = number_of_samples % number_of_blocks;
    // Some processes will need one more sample if the data size does not fit completely with the number of processes
    if (reserve > 0 && block_index < reserve) {
        return block + 1;
    }
    return block;
}

uint deep_io::get_block_start_offset(const uint part_index, const uint number_of_samples,
        const uint number_of_blocks) noexcept {
    int offset = 0;
    for (int i = 0; i < part_index; i++) {
        offset += get_block_size(i, number_of_samples, number_of_blocks);
    }
    return offset;
}

void deep_io::load_meta_data(std::istream &is, s_vec<float> &v_samples) noexcept {
    is.read((char *) &sample_no, sizeof(int));
    is.read((char *) &feature_no, sizeof(int));
    unread_samples = get_block_size(block_index, sample_no, block_no);
    block_sample_offset = get_block_start_offset(block_index, sample_no, block_no);
    feature_offset = 2 * sizeof(int) + (block_sample_offset * feature_no * sizeof(float));
    v_samples.resize(unread_samples * feature_no);
}

int deep_io::load_next_samples(s_vec<float> &v_samples) noexcept {
    std::ifstream ifs(file, std::ios::in | std::ifstream::binary);
    if (!is_initialized) {
        load_meta_data(ifs, v_samples);
    }
    uint buffer_samples = std::min(max_samples_per_batch, unread_samples);
//    std::cout << "unread samples: " << unread_samples << " : buffer : " << buffer_samples << std::endl;
    uint bytes_read = 0;
    ifs.seekg(feature_offset, std::istream::beg);
//    std::cout << "feature offset: " << feature_offset << " about to read " << buffer_samples << " samples" << std::endl;
    if (!ifs.read((char *) &v_samples[0/*block_sample_offset * feature_no*/], buffer_samples * feature_no * sizeof(float))) {
        if (ifs.bad()) {
            std::cerr << "Error: " << strerror(errno);
        } else {
            std::cout << "error: only " << ifs.gcount() << " feature bytes could be read" << std::endl;
        }
        return UNDEFINED_VALUE;
    }
    bytes_read += ifs.gcount();
    unread_samples -= buffer_samples;
    sample_read_no = buffer_samples;
    ifs.close();
    return bytes_read;
}

void deep_io::get_blocks_meta(std::unique_ptr<uint[]> &v_sizes, std::unique_ptr<uint[]> &v_offsets,
        uint number_of_samples, uint number_of_blocks) noexcept {
    uint total_size = 0;
    for (uint i = 0; i < number_of_blocks; ++i) {
        uint size = get_block_size(i, number_of_samples, number_of_blocks);
        v_offsets[i] = total_size;
        v_sizes[i] = size;
        total_size += size;
    }
}
