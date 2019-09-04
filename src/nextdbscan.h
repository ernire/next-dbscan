#ifndef NEXTDBSCAN
#define NEXTDBSCAN

#include <string>
#include <vector>

namespace nextdbscan {

    struct result {
        unsigned int clusters;
        unsigned int noise;
        unsigned int core_count;
        std::vector<int> *point_clusters;
    };

    result start(
            unsigned int m,
            float e,
            unsigned int n_threads,
            const std::string &in_file) noexcept;

    result start_mpi(
            unsigned int m,
            float e,
            unsigned int n_threads,
            const std::string &in_file,
            int mpi_rank,
            int mpi_size) noexcept;

};

#endif