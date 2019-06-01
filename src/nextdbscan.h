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
        const unsigned int m, 
        const float e,
        const unsigned int n_threads, 
        const std::string &in_file) noexcept;

};

#endif