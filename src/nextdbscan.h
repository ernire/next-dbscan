#ifndef NEXTDBSCAN
#define NEXTDBSCAN

#include <string>
#include <vector>

namespace nextdbscan {

    struct result {
        unsigned int clusters;
        unsigned int noise;
        unsigned int core_count;
        unsigned int n;
        // TODO avoid memory leak
        int *point_clusters;
    };

    result start(
            unsigned int m,
            float e,
            unsigned int n_threads,
            const std::string &in_file,
            unsigned int node_index,
            unsigned int n_nodes) noexcept;

};

#endif