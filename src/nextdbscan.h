#ifndef NEXTDBSCAN
#define NEXTDBSCAN

#include <string>

namespace nextdbscan {

    struct result {
        unsigned int clusters;
        unsigned int noise;
        unsigned int core_count;
    };

    result start(
        const unsigned int m, 
        const float e, 
        const unsigned int max_d, 
        const unsigned int n_threads, 
        const std::string &in_file) noexcept;

};

#endif