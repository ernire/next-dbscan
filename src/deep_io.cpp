//
// Created by Ernir Erlingsson on 2.9.2019.
//
#include <fstream>
#include <iostream>
#include <cstring>
#include <vector>
#include "deep_io.h"

uint deep_io::get_block_size(const uint block_index, const uint number_of_samples, const uint number_of_blocks) {
    uint block = (number_of_samples / number_of_blocks);
    uint reserve = number_of_samples % number_of_blocks;
    // Some processes will need one more sample if the data size does not fit completely with the number of processes
    if (reserve > 0 && block_index < reserve) {
        return block + 1;
    }
    return block;
}

uint deep_io::get_block_start_offset(const uint part_index, const uint number_of_samples, const uint number_of_blocks) {
    int offset = 0;
    for (int i = 0; i < part_index; i++) {
        offset += get_block_size(i, number_of_samples, number_of_blocks);
    }
    return offset;
}

void deep_io::load_meta_data(std::istream &is, std::vector<float> &v_samples) {
    is.read((char *) &sample_no, sizeof(int));
    is.read((char *) &feature_no, sizeof(int));
    unread_samples = get_block_size(block_index, sample_no, block_no);
    block_sample_offset = get_block_start_offset(block_index, sample_no, block_no);
    feature_offset = 2 * sizeof(int) + (block_sample_offset * feature_no * sizeof(float));
    v_samples.resize(unread_samples * feature_no);

//    v_samples = std::make_unique<float[]>(sample_no * feature_no);
//    std::fill(&v_samples[0], &v_samples[0] + sample_no * feature_no, UNDEFINED_VALUE);
}

int deep_io::load_next_samples(std::vector<float> &v_samples) {
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

void deep_io::get_blocks_meta(std::unique_ptr<uint[]> &v_sizes, std::unique_ptr<uint[]> &v_offsets, uint number_of_samples,
        uint number_of_blocks) {
    uint total_size = 0;
    for (uint i = 0; i < number_of_blocks; ++i) {
        uint size = get_block_size(i, number_of_samples, number_of_blocks);
        v_offsets[i] = total_size;
        v_sizes[i] = size;
        total_size += size;
    }
}
