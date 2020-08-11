//
// Created by Ernir Erlingsson on 4.8.2020.
//

#ifndef NEXT_DBSCAN_NEXT_MPI_H
#define NEXT_DBSCAN_NEXT_MPI_H


#include <type_traits>
#include <cassert>
#include <mpi.h>
#include "next_util.h"

class nextMPI {
private:

#ifdef MPI_ON
    template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    int inferType(std::vector<T> &v) noexcept {
        if (std::is_floating_point<T>::value) {
            return MPI_FLOAT;
        } else {
            return MPI_LONG;
        }
    }
#endif

    template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void allReduce(std::vector<T> &v_sendbuf, std::vector<T> &v_recvbuf, int const size_per_node,
            int const mpi_op) noexcept {
#ifdef MPI_ON
        MPI_Allreduce(v_sendbuf == v_recvbuf ? MPI_IN_PLACE : &v_sendbuf[0],
                &v_recvbuf[0],
                size_per_node,
                inferType(v_recvbuf),
                mpi_op,
                comm);
#endif
    }


public:
    int const rank, comm, n_nodes;

    explicit nextMPI(int const mpi_rank, int const mpi_comm, int const n_nodes) :
            rank(mpi_rank), comm(mpi_comm), n_nodes(n_nodes) {
#if defined(DEBUG_ON) && defined(MPI_ON)
        assert(n_nodes > 0);
        // TODO support dynamic comm size
        assert(mpi_comm == MPI_COMM_WORLD);
        int size;
        MPI_Comm_size(mpi_comm, &size);
        assert(n_nodes == size);
#endif;
    }

    template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void allGather(std::vector<T> &v_sendbuf, std::vector<T> &v_recvbuf, int const size_per_node) noexcept {
#ifdef MPI_ON
//        std::cout << "v_sendbuf == v_recvbuf? " << (v_sendbuf == v_recvbuf) << std::endl;
        MPI_Allgather(v_sendbuf == v_recvbuf ? MPI_IN_PLACE : &v_sendbuf[0],
                n_nodes,
                inferType(v_sendbuf),
                &v_recvbuf[0],
                size_per_node,
                inferType(v_recvbuf),
                comm);
#endif
    }

    template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void allReduceMin(std::vector<T> &v_sendbuf, std::vector<T> &v_recvbuf, int const size_per_node) noexcept {
#ifdef MPI_ON
        allReduce(v_sendbuf, v_recvbuf, size_per_node, MPI_MIN);
#endif
    }

    template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void allReduceMax(std::vector<T> &v_sendbuf, std::vector<T> &v_recvbuf, int const size_per_node) noexcept {
#ifdef MPI_ON
        allReduce(v_sendbuf, v_recvbuf, size_per_node, MPI_MAX);
#endif
    }

    template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void allReduceSum(std::vector<T> &v_sendbuf, std::vector<T> &v_recvbuf, int const size_per_node) noexcept {
#ifdef MPI_ON
        allReduce(v_sendbuf, v_recvbuf, size_per_node, MPI_SUM);
#endif
    }

    // fills v_receive size automatically if it's empty, otherwise it trusts its accuracy

    template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void alltoallv(std::vector<T> &v_sendbuf, std::vector<T> &v_recvbuf, std::vector<int> &v_transmit_size,
            std::vector<int> &v_receive_size) noexcept {
#ifdef MPI_ON
        std::vector<int> v_transmit_offset(static_cast<unsigned long>(n_nodes));
        std::exclusive_scan(v_transmit_size.begin(), std::next(v_transmit_size.begin(), n_nodes),
                v_transmit_offset.begin(), 0);

        for (int i = 0; i < v_transmit_size.size(); i++) {
            if (v_transmit_size[i] + v_transmit_offset[i] > v_sendbuf.size()) {
#ifdef DEBUG_ON
                std::cout << "MPI buffer overflow detected - adjusting" << std::endl;
#endif
                v_transmit_offset[i] = static_cast<int>(v_sendbuf.size() - v_transmit_size[i]);
            }
        }
        if (v_receive_size.empty()) {
            std::vector<int> v_size_agg(static_cast<unsigned long>(n_nodes * n_nodes));
            allGather(v_transmit_size, v_size_agg, n_nodes);
            v_receive_size.clear();
            v_receive_size.resize(static_cast<unsigned long>(n_nodes), 0);
//            unsigned long total_size = 0;
            for (int i = rank, j = 0; i < v_size_agg.size(); i += n_nodes, ++j) {
                v_receive_size[j] = v_size_agg[i];
//                total_size += v_size_agg[i];
            }
        }
        std::vector<int> v_recv_offset(static_cast<unsigned long>(n_nodes));
        std::exclusive_scan(v_receive_size.begin(), v_receive_size.end(), v_recv_offset.begin(), 0);

        v_recvbuf.resize(std::reduce(v_receive_size.begin(), v_receive_size.end()));
        /*
        if (rank == 0) {
//            next_util::print_vector("v_sendbuf: ", v_sendbuf.size());
            for (int i = 0; i < v_transmit_size.size(); i++) {
                if (v_transmit_size[i] + v_transmit_offset[i] > v_sendbuf.size()) {
#ifdef DEBUG_ON
                    std::cout << "MPI buffer overflow detected - adjusting" << std::endl;
#endif
                    v_transmit_offset[i] = static_cast<int>(v_sendbuf.size() - v_transmit_size[i]);
                }
            }
            std::cout << "sendbuf size: " << v_sendbuf.size() << std::endl;
            next_util::print_vector("v_transmit_size: ", v_transmit_size);
            next_util::print_vector("v_transmit_offset: ", v_transmit_offset);
//            for (auto &val : v_transmit_offset) {
//                if (val >= )
//            }
            next_util::print_vector("v_receive_size: ", v_receive_size);
        }
         */
        MPI_Alltoallv(&v_sendbuf[0], &v_transmit_size[0], &v_transmit_offset[0], inferType(v_sendbuf),
                &v_recvbuf[0], &v_receive_size[0], &v_recv_offset[0], inferType(v_recvbuf), comm);
#endif
    }

};


#endif //NEXT_DBSCAN_NEXT_MPI_H
