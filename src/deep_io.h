//
// Created by Ernir Erlingsson on 2.9.2019.
//

#ifndef NEXT_DBSCAN_DEEP_IO_H
#define NEXT_DBSCAN_DEEP_IO_H

#include <istream>
#include <memory>

const int UNDEFINED_VALUE = -1;
typedef unsigned int uint;

class deep_io {
private:
    const char *file;
    const uint block_no, block_index, max_samples_per_batch;
    int feature_offset = UNDEFINED_VALUE;
    bool is_initialized = false;

    void load_meta_data(std::istream &is, std::vector<float> &v_samples);

public:
//    float *features;
    uint unread_samples;
    uint sample_no, feature_no, sample_read_no;
    uint block_sample_offset;

    deep_io(char *file, uint number_of_blocks, uint block_index) : deep_io(file, number_of_blocks, block_index,
            INT32_MAX) {}

    deep_io(const char *file, uint number_of_blocks, uint block_index, uint max_samples_per_batch)
            : file(file), block_no(number_of_blocks), block_index(block_index),
              max_samples_per_batch(max_samples_per_batch) {
//        features = nullptr;
        sample_no = 0;
        feature_no = 0;
        sample_read_no = 0;
        unread_samples = 0;
        block_sample_offset = 0;
    }

    ~deep_io() = default;

    int load_next_samples(std::vector<float> &v_samples);

    static void get_blocks_meta(std::unique_ptr<uint[]> &v_sizes, std::unique_ptr<uint[]> &v_offsets,
            uint number_of_samples, uint number_of_blocks);

    static uint get_block_size(uint block_index, uint number_of_samples, uint number_of_blocks);

    static uint get_block_start_offset(uint block_index, uint number_of_samples, uint number_of_blocks);
};

std::streampos get_file_size(const char *filePath);

#endif //NEXT_DBSCAN_DEEP_IO_H
