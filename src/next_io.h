//
// Created by Ernir Erlingsson on 2.9.2019.
//

#ifndef NEXT_DBSCAN_NEXT_IO_H
#define NEXT_DBSCAN_NEXT_IO_H

#include <istream>

const int UNDEFINED_VALUE = -1;
typedef unsigned int uint;

class next_io {
private:
    const char *file;
    const uint block_no, block_index, max_samples_per_batch;
    int feature_offset = UNDEFINED_VALUE;
    bool is_initialized = false;

    void load_meta_data(std::istream &is);

    static uint get_part_size(uint part_index, uint number_of_samples, uint number_of_parts);

    static uint get_block_start_offset(uint part_index, uint number_of_samples, uint number_of_blocks);

public:
    float *features;
    uint unread_samples;
    uint sample_no, feature_no, sample_read_no;
    uint block_sample_offset;

    next_io(char *file, int number_of_blocks, int block_index) : next_io(file, number_of_blocks, block_index,
            INT32_MAX) {}

    next_io(const char *file, uint number_of_blocks, uint block_index, uint max_samples_per_batch)
            : file(file), block_no(number_of_blocks), block_index(block_index),
              max_samples_per_batch(max_samples_per_batch) {
        features = nullptr;
        sample_no = 0;
        feature_no = 0;
        sample_read_no = 0;
        unread_samples = 0;
        block_sample_offset = 0;
    }

    ~next_io() = default;

    int load_next_samples();

    static void get_parts_meta(std::vector<int> &v_sizes, std::vector<int> &v_offsets, uint number_of_samples,
            uint number_of_parts, uint dimensions);
};

std::streampos get_file_size(const char *filePath);

#endif //NEXT_DBSCAN_NEXT_IO_H
