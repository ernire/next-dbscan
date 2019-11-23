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
#include "nextdbscan_mpi.h"

/*
#ifdef MPI_ON
void process_nodes_nearest_neighbour(std::unique_ptr<float[]> &v_coords,
        std::unique_ptr<uint[]> &v_node_offset,
        std::vector<std::vector<std::vector<uint>>> &vvv_index_map,
        std::vector<std::vector<std::vector<uint>>> &vvv_cell_begin,
        std::vector<std::vector<std::vector<uint>>> &vvv_cell_ns,
        std::vector<std::vector<cell_meta_3>> &vv_stacks3,
        std::vector<std::vector<bool>> &vv_range_table,
        // TODO implement the range counts
        std::vector<std::vector<uint>> &vv_range_counts,
        std::vector<std::vector<uint>> &vv_leaf_cell_nn,
        std::vector<std::vector<uint>> &vv_point_nn,
        std::vector<std::vector<uint8_t>> &vv_cell_type,
        std::vector<std::vector<uint8_t>> &vv_is_core,
        std::vector<std::vector<std::vector<float>>> &vvv_min_cell_dim,
        std::vector<std::vector<std::vector<float>>> &vvv_max_cell_dim,
        const uint n_nodes, const uint max_d, const uint m, const float e, const uint max_levels,
        const uint node_index) noexcept {
    const float e2 = e * e;
    std::vector<uint> v_payload;
    std::vector<uint> v_sink_cells;
    std::vector<uint> v_sink_points;
    mpi_sum_vectors(vv_point_nn, v_payload, v_sink_points, v_sink_points, n_nodes,
            MPI_UNSIGNED, false);
    mpi_sum_vectors(vv_leaf_cell_nn, v_payload, v_sink_cells, v_sink_cells, n_nodes,
            MPI_UNSIGNED, false);
    int mod = n_nodes == 2 ? 5 : 4;
    int level = std::max((int)max_levels - mod, 0);
    std::vector<cell_meta_5> v_pair_task;
    v_pair_task.reserve(vvv_cell_ns[0][level].size() * n_nodes);
    int cnt = 0;
    for (uint n1 = 0; n1 < n_nodes; ++n1) {
        for (uint i = 0; i < vvv_cell_ns[n1][level].size(); ++i) {
            for (uint n2 = n1 + 1; n2 < n_nodes; ++n2) {
                for (uint j = 0; j < vvv_cell_ns[n2][level].size(); ++j) {
                    if (++cnt % n_nodes == node_index) {
                        v_pair_task.emplace_back(level, i, j, n1, n2);
                    }
                }
            }
        }
    }
    if (node_index == 0)
        std::cout << "pair tasks: " << v_pair_task.size() << std::endl;
    #pragma omp parallel for schedule(dynamic)
    for (uint i = 0; i < v_pair_task.size(); ++i) {
        uint tid = omp_get_thread_num();
        uint n1 = v_pair_task[i].n1;
        uint n2 = v_pair_task[i].n2;
        vv_stacks3[tid].emplace_back(v_pair_task[i].l, v_pair_task[i].c1, v_pair_task[i].c2);
        process_cell_pair(&v_coords[v_node_offset[n1] * max_d], &v_coords[v_node_offset[n2] * max_d],
                vvv_index_map[n1], vvv_index_map[n2],
                vvv_cell_ns[n1], vvv_cell_ns[n2], vvv_cell_begin[n1], vvv_cell_begin[n2],
                vvv_min_cell_dim[n1], vvv_max_cell_dim[n1], vvv_min_cell_dim[n2], vvv_max_cell_dim[n2],
                vv_leaf_cell_nn[n1], vv_leaf_cell_nn[n2], vv_point_nn[n1], vv_point_nn[n2],
                vv_stacks3[tid], vv_range_table[tid], max_d, m, e, e2, true);
    }
    std::vector<uint> v_sink;
    mpi_sum_vectors(vv_point_nn, v_payload, v_sink, v_sink_points, n_nodes,
            MPI_UNSIGNED, true);
    mpi_sum_vectors(vv_leaf_cell_nn, v_payload, v_sink, v_sink_cells, n_nodes,
            MPI_UNSIGNED, true);
}

#endif
 */


/*
#ifdef MPI_ON
// Share coordinates
auto time_mpi1 = std::chrono::high_resolution_clock::now();
int sizes[n_nodes];
int offsets[n_nodes];
for (int n = 0; n < n_nodes; ++n) {
    sizes[n] = v_node_sizes[n] * max_d;
    offsets[n] = v_node_offsets[n] * max_d;
}
MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &v_coords[0], sizes,
        offsets, MPI_FLOAT, MPI_COMM_WORLD);

auto time_mpi2 = std::chrono::high_resolution_clock::now();
if (!g_quiet && node_index == 0) {
    std::cout << "MPI Point Merge: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(time_mpi2 - time_mpi1).count()
              << " milliseconds\n";
}
if (n_nodes > 1) {
    uint node_offset = v_node_offsets[node_index];
    uint node_size = v_node_sizes[node_index];
    coord_partition_and_merge(v_coords, v_min_bounds, &v_dims_mult[0], v_eps_levels[0],
            node_size, node_offset, max_d, n_threads, node_index, n_nodes, total_samples);
}
auto time_mpi3 = std::chrono::high_resolution_clock::now();
if (!g_quiet && node_index == 0) {
    std::cout << "MPI Point Sort: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(time_mpi3 - time_mpi2).count()
              << " milliseconds\n";
}
#endif
 */

#ifdef MPI_ON
void coord_partition_and_merge(std::unique_ptr<float[]> &v_coords,
            std::unique_ptr<float[]> &v_min_bounds,
            const ull* dims_mult, const float eps_level,
            const uint node_size, const uint node_offset, const uint max_d, const uint n_threads,
            const uint node_index, const uint n_nodes, const uint total_samples) {
        /*
        int sizes[n_nodes];
        int offsets[n_nodes];
        std::vector<uint> v_index_map(node_size);
        std::iota(v_index_map.begin(), v_index_map.end(), 0);
        std::vector<ull> v_value_map(node_size);
        uint index = 0;
        for (uint i = node_offset; i < node_size + node_offset; ++i, ++index) {
            v_value_map[index] = get_cell_index(&v_coords[i * max_d], v_min_bounds,
                    dims_mult, max_d, eps_level);
        }
        std::sort(v_index_map.begin(), v_index_map.end(), [&](const auto &i1, const auto &i2) -> bool {
            return v_value_map[i1] < v_value_map[i2];
        });
        std::vector<ull> v_bucket_seperator;
        v_bucket_seperator.reserve(n_threads);
        std::vector<ull> v_bucket_seperator_tmp;
        v_bucket_seperator_tmp.reserve(n_threads * n_threads);
        std::vector<std::vector<uint>::iterator> v_iterator;
        mpi_sort_merge(v_index_map, v_value_map, v_bucket_seperator, v_bucket_seperator_tmp, v_iterator,
                n_nodes, node_index);
        int block_sizes[n_nodes * n_nodes];
//         TODO tmp
//        for (uint i = 0; i < n_nodes * n_nodes; ++i) {
//            block_sizes[i] = 0;
//        }
        int block_offsets[n_nodes * n_nodes];
        index = node_index * n_nodes;
        offsets[0] = 0;
        for (uint n = 0; n < n_nodes; ++n) {
            block_sizes[index + n] = v_iterator[n + 1] - v_iterator[n];
            sizes[n] = n_nodes;
            if (n > 0)
                offsets[n] = offsets[n - 1] + sizes[n - 1];
        }
//        print_array("pre block sizes: ", &block_sizes[0], n_nodes * n_nodes);

        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &block_sizes[0], sizes,
                offsets, MPI_FLOAT, MPI_COMM_WORLD);
//        if (node_index == 0) {
//            print_array("post block sizes: ", &block_sizes[0], n_nodes * n_nodes);
//        }

        int last_size = 0;
        int last_val = 0;
        int val = 0;
        for (int n1 = 0; n1 < n_nodes; ++n1) {
            for (int n2 = 0; n2 < n_nodes; ++n2) {
                index = n2 * n_nodes + n1;
                val = last_val + last_size;
                block_offsets[index] = val;
                last_val = val;
                last_size = block_sizes[index];
            }
        }
//        if (node_index == 0) {
//            print_array("post block offsets: ", &block_offsets[0], n_nodes * n_nodes);
//        }
        std::vector<float> v_coord_copy(total_samples * max_d);

        index = node_index * n_nodes;
        for (uint n = 0; n < n_nodes; ++n) {
            uint begin_coord = block_offsets[index + n] * max_d;
            uint begin_block = (v_iterator[n] - v_iterator[0]);
//            std::cout << "node: " << node_index << " begin_block: " << begin_block << std::endl;
            for (uint i = 0; i < block_sizes[index + n]; ++i) {
                assert(begin_coord + i < v_coord_copy.size());
                for (uint j = 0; j < max_d; ++j) {
                    assert(begin_coord + (i * max_d) + j < v_coord_copy.size());
                    v_coord_copy[begin_coord + (i * max_d) + j] = v_coords[
                            (node_offset + v_index_map[begin_block + i]) * max_d + j];
                }

            }
        }
        last_size = 0;
        last_val = 0;
        val = 0;
        for (uint n1 = 0; n1 < n_nodes; ++n1) {
            index = 0;
            for (uint n2 = 0; n2 < n_nodes; ++n2) {
                sizes[index] = block_sizes[n2 * n_nodes + n1];
                val = last_val + last_size;
                offsets[index] = val;
                last_val = val;
                last_size = sizes[index];
                sizes[index] *= max_d;
                offsets[index] *= max_d;
                ++index;
            }
//            if (node_index == 1) {
//                print_array("transmit sizes: ", &sizes[0], n_nodes);
//                print_array("transmit offsets: ", &offsets[0], n_nodes);
//            }
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &v_coord_copy[0],
                    sizes, offsets, MPI_FLOAT, MPI_COMM_WORLD);
        }
//        for (auto &elem : v_coord_copy) {
//            assert(elem != INT32_MAX);
//        }
        std::copy(v_coord_copy.begin(), v_coord_copy.end(), &v_coords[0]);
        */
    }
#endif

#ifdef MPI_ON
/*
void mpi_sort_merge(std::vector<uint> &v_index_map,
            std::vector<ull> &v_value_map,
            std::vector<ull> &v_bucket_seperator,
            std::vector<ull> &v_bucket_seperator_tmp,
            std::vector<std::vector<uint>::iterator> &v_iterator,
            const uint n_nodes, const uint node_index) noexcept {

        v_bucket_seperator.clear();
//            v_bucket_seperator.resize(n_nodes);
        v_bucket_seperator_tmp.clear();
        v_bucket_seperator_tmp.resize(n_nodes * (n_nodes - 1), 0);
        std::vector<int> v_block_sizes(n_nodes, 0);
        std::vector<int> v_block_offsets(n_nodes, 0);
        int block_offset = 0;
        for (uint n = 0; n < n_nodes; ++n) {
            if (n < n_nodes - 1) {
                uint index = (node_index * (n_nodes - 1)) + n;
                uint map_index = (n + 1) * (v_index_map.size() / n_nodes);
                v_bucket_seperator_tmp[index] = v_value_map[v_index_map[map_index]];
            }
            v_block_sizes[n] = n_nodes - 1;
            v_block_offsets[n] = block_offset;
            block_offset += v_block_sizes[n];
        }
//            std::cout << "v_block_sizes_in_bytes[0]: " << v_block_sizes_in_bytes[0] << std::endl;
//        print_array("block sizes: ", &v_block_sizes[0], v_block_sizes.size());
//        print_array("block offsets: ", &v_block_offsets[0], v_block_offsets.size());
//        print_array("Pre: ", &v_bucket_seperator_tmp[0], v_bucket_seperator_tmp.size());
//        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &v_bucket_seperator_tmp[0],
                &v_block_sizes[0], &v_block_offsets[0], MPI_UNSIGNED_LONG_LONG,
                MPI_COMM_WORLD);
//        MPI_Barrier(MPI_COMM_WORLD);
//        print_array("Post: ", &v_bucket_seperator_tmp[0], v_bucket_seperator_tmp.size());
//        MPI_Barrier(MPI_COMM_WORLD);
        std::sort(v_bucket_seperator_tmp.begin(), v_bucket_seperator_tmp.end());
        for (uint n = 0; n < n_nodes - 1; ++n) {
            uint index = (n * n_nodes) + (n_nodes / 2);
            v_bucket_seperator.push_back((v_bucket_seperator_tmp[index] + v_bucket_seperator_tmp[index - 1]) / 2);
        }
//        MPI_Barrier(MPI_COMM_WORLD);
//        print_array("Selected: ", &v_bucket_seperator[0], v_bucket_seperator.size());
//        MPI_Barrier(MPI_COMM_WORLD);
//        std::vector<std::vector<uint>::iterator> v_iterator;
//        std::vector<std::vector<uint>> v_node_bucket(n_nodes);
        v_iterator.push_back(v_index_map.begin());
        // TODO parallelize
        for (auto &separator : v_bucket_seperator) {
            auto iter = std::lower_bound(
                    v_index_map.begin(),
                    v_index_map.end(),
                    separator,
                    [&v_value_map](const auto &i1, const auto &val) -> bool {
                        return v_value_map[i1] < val;
                    });
            v_iterator.push_back(iter);
        }
        v_iterator.push_back(v_index_map.end());
    }
*/
#endif


#ifdef MPI_ON
template<class T>
    void mpi_sum_vectors(std::vector<std::vector<T>> &vv_vector, std::vector<T> &v_payload,
            std::vector<T> &v_sink, std::vector<T> &v_additive, const int n_nodes,
            MPI_Datatype send_type, const bool is_additive) noexcept {
        int send_cnt = 0;
        int size[n_nodes];
        int offset[n_nodes];
        offset[0] = 0;
        for (int n = 0; n < n_nodes; ++n) {
            size[n] = vv_vector[n].size();
            send_cnt += size[n];
            if (n > 0) {
                offset[n] = offset[n - 1] + size[n - 1];
            }
        }
        if (v_payload.empty()) {
            v_payload.resize(send_cnt);
        }
        if (v_sink.empty()) {
            v_sink.resize(send_cnt);
        }
        int index = 0;
        for (int n = 0; n < n_nodes; ++n) {
            #pragma omp parallel for
            for (uint i = 0; i < vv_vector[n].size(); ++i) {
                v_payload[index+i] = is_additive ? vv_vector[n][i] - v_additive[index+i] : vv_vector[n][i];
            }
            index += vv_vector[n].size();
        }
        MPI_Allreduce(&v_payload[0], &v_sink[0], send_cnt, send_type, MPI_SUM, MPI_COMM_WORLD);
        index = 0;
        for (int n = 0; n < n_nodes; ++n) {
            #pragma omp parallel for
            for (uint i = 0; i < vv_vector[n].size(); ++i) {
                if (is_additive) {
                    vv_vector[n][i] = v_additive[i+index] + v_sink[i+index];
                } else {
                    vv_vector[n][i] = v_sink[i+index];
                }
            }
            index += vv_vector[n].size();
        }
    }

    /*
#ifdef MPI_ON
    template<class T>
    void mpi_gather_cell_tree(std::vector<std::vector<std::vector<T>>> &vvv_cell_tree,
            const int max_levels, const int n_nodes, const int node_index, std::vector<T> &v_buffer,
            MPI_Datatype send_type) noexcept {
        int size[n_nodes];
        int offset[n_nodes];
        for (int n = 0; n < n_nodes; ++n) {
            size[n] = 0;
            offset[n] = 0;
        }
        for (int n = 0; n < n_nodes; ++n) {
            for (int l = 0; l < max_levels; ++l) {
                size[n] += vvv_cell_tree[n][l].size();
            }
        }
        offset[0] = 0;
        for (int n = 1; n < n_nodes; ++n) {
            offset[n] = offset[n - 1] + size[n - 1];
        }
        int total_size = 0;
        for (int n = 0; n < n_nodes; ++n) {
            total_size += size[n];
        }
        v_buffer.resize(total_size, INT32_MAX);
//        print_array("Transmit size: ", size, n_nodes);
//        print_array("Transmit offset: ", offset, n_nodes);
        int index = 0;
        // TODO make smarter
        for (int n = 0; n < n_nodes; ++n) {
            for (int l = 0; l < max_levels; ++l) {
                for (auto &val : vvv_cell_tree[n][l]) {
                    assert(index < v_buffer.size());
                    if (n == node_index) {
                        v_buffer[index] = val;
                    }
                    ++index;
                }
            }
        }
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &v_buffer[0], size,
                offset, send_type, MPI_COMM_WORLD);
        index = 0;
        for (int n = 0; n < n_nodes; ++n) {
            for (int l = 0; l < max_levels; ++l) {
                // TODO skip node index
                for (int i = 0; i < vvv_cell_tree[n][l].size(); ++i) {
                    assert(index < v_buffer.size());
                    assert(v_buffer[index] != (T) INT32_MAX);
                    vvv_cell_tree[n][l][i] = v_buffer[index];
                    ++index;
                }
            }
        }
        assert(index == v_buffer.size());
    }

#endif

    void mpi_merge_cell_trees(std::vector<std::vector<std::vector<uint>>> &vvv_index_maps,
            std::vector<std::vector<std::vector<uint>>> &vvv_cell_begins,
            std::vector<std::vector<std::vector<uint>>> &vvv_cell_ns,
            std::vector<std::vector<std::vector<float>>> &vvv_min_cell_dims,
            std::vector<std::vector<std::vector<float>>> &vvv_max_cell_dims,
            const int node_index, const int n_nodes, const int max_levels,
            const uint max_d) noexcept {
        // count the number of elements and share it
        int total_levels = n_nodes * max_levels;
        auto n_node_level_elem = std::make_unique<int[]>(total_levels);
        std::fill(&n_node_level_elem[0], &n_node_level_elem[0] + total_levels, 0);
        uint index = node_index * max_levels;
        for (uint l = 0; l < max_levels; ++l, ++index) {
            n_node_level_elem[index] += vvv_index_maps[node_index][l].size();
        }
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &n_node_level_elem[0], max_levels,
                MPI_INT, MPI_COMM_WORLD);

        index = 0;
        for (uint n = 0; n < n_nodes; ++n) {
            for (uint l = 0; l < max_levels; ++l, ++index) {
                vvv_index_maps[n][l].resize(n_node_level_elem[index]);
                if (l > 0) {
                    vvv_cell_begins[n][l - 1].resize(n_node_level_elem[index]);
                    vvv_cell_ns[n][l - 1].resize(n_node_level_elem[index]);
                    vvv_min_cell_dims[n][l - 1].resize(n_node_level_elem[index] * max_d);
                    vvv_max_cell_dims[n][l - 1].resize(n_node_level_elem[index] * max_d);
                }
            }
            vvv_cell_begins[n][max_levels - 1].resize(1);
            vvv_cell_ns[n][max_levels - 1].resize(1);
            vvv_min_cell_dims[n][max_levels - 1].resize(max_d);
            vvv_max_cell_dims[n][max_levels - 1].resize(max_d);
        }

        std::vector<uint> v_uint_buffer;
        std::vector<float> v_float_buffer;
        mpi_gather_cell_tree(vvv_index_maps, max_levels, n_nodes, node_index, v_uint_buffer,
                MPI_UNSIGNED);
        mpi_gather_cell_tree(vvv_cell_begins, max_levels, n_nodes, node_index, v_uint_buffer,
                MPI_UNSIGNED);
        mpi_gather_cell_tree(vvv_cell_ns, max_levels, n_nodes, node_index, v_uint_buffer,
                MPI_UNSIGNED);
        mpi_gather_cell_tree(vvv_min_cell_dims, max_levels, n_nodes, node_index, v_float_buffer,
                MPI_FLOAT);
        mpi_gather_cell_tree(vvv_max_cell_dims, max_levels, n_nodes, node_index, v_float_buffer,
                MPI_FLOAT);
    }

#endif
*/

#ifdef MPI_ON
if (n_nodes > 1) {
            /*
            measure_duration("Node Trees Proximity: ", node_index == 0, [&]() -> void {
                std::vector<float> v_min_send_buf;
                std::vector<float> v_max_send_buf;
                auto v_gather_buf = std::make_unique<int[]>(n_nodes * n_nodes);
                // TODO reserve
                auto v_send_offsets = std::make_unique<int[]>(n_nodes);
                auto v_send_cnts = std::make_unique<int[]>(n_nodes);
//                auto v_gather_cnts = std::make_unique<int[]>(n_nodes);
//                auto v_gather_offsets = std::make_unique<int[]>(n_nodes);
//                for (uint i = 0; i < n_nodes; ++i) {
//                    v_gather_cnts[i] =
//                }
                // While stack not empty code pattern
//                vv_stacks3[0].push_back()
                uint gather_index = node_index * n_nodes;
                int offset = 0;
                for (int n = 0; n < n_nodes; ++n) {
                    uint cnt = 0;
                    int l = max_level-1;
                    for (uint i = 0; i < vv_cell_begin[l].size(); ++i) {
                        for (uint c = 0; c < vv_cell_ns[l][i]; ++c) {
                            uint begin = vv_cell_begin[l][c];
                            uint c_index = vv_index_map[l][begin + c];
                            // Insert instead of iteration
                            for (uint j = 0; j < max_d; ++j) {
                                v_min_send_buf.push_back(vv_min_cell_dim[l-1][c_index * max_d]);
                                v_max_send_buf.push_back(vv_max_cell_dim[l-1][c_index * max_d]);
                                ++cnt;
                            }
                        }
                    }
                    v_send_cnts[n] = cnt;
                    v_send_offsets[n] = offset;
                    v_gather_buf[gather_index + n] = cnt;
                }
                MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &v_gather_buf[0],
                        n_nodes, MPI_INT, MPI_COMM_WORLD);

//                MPI_Alltoallv(&v_min_send_buf[0], &v_send_cnts[0], &v_send_offsets[0],
//                        MPI_FLOAT,nullptr, nullptr,
//                        nullptr, MPI_FLOAT, MPI_COMM_WORLD);
            });
             */
        }
#endif

/*
#ifdef MPI_ON
if (n_nodes > 1) {
    measure_duration("MPI Grid-cell Tree Merge: ", node_index == 0, [&]() -> void {
        mpi_merge_cell_trees(vvv_index_map, vvv_cell_begin, vvv_cell_ns, vvv_min_cell_dim,
                vvv_max_cell_dim, node_index, n_nodes, max_levels, max_d);
    });
}
#endif
 */

/*
#ifdef MPI_ON
measure_duration("Update Shared Labels: ", node_index == 0, [&]() -> void {
    for (uint n = 0; n < n_nodes; ++n) {
        if (n == node_index)
            continue;
        #pragma omp parallel for
        for (uint i = 0; i < vvv_cell_ns[n][0].size(); ++i) {
            update_type(vvv_index_map[n][0], vvv_cell_ns[n][0], vvv_cell_begin[n][0],
                    vv_leaf_cell_nn[n], vv_point_nn[n], vv_is_core[n], vv_cell_type[n], i, m);
        }
    }
});
#endif
 */