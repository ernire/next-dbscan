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

#include <hdf5.h>
#include <istream>
#include <memory>
#include <fstream>
#include <iostream>
#include <cstring>
#include <vector>
#include <sstream>
#include <iterator>

void count_lines_and_dimensions(const std::string &in_file, uint &lines, uint &dimensions) noexcept {
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

void read_input_csv(const std::string &in_file, std::vector<float> &v_points, int max_d) noexcept {
    std::ifstream is(in_file);
    std::string line, buf;
    std::stringstream ss;
    int index = 0;
    while (std::getline(is, line)) {
        ss.str(std::string());
        ss.clear();
        ss << line;
        for (int j = 0; j < max_d; j++) {
            ss >> buf;
            v_points[index++] = atof(buf.c_str());
        }
    }
    is.close();
}

int main(int argc, char* const* argv) {

    if (argc != 3) {
        std::cout << "Wrong number of parameters, should be 2 (input file, output file)" << std::endl;
        exit(-1);
    }

    std::string input_file = argv[1];
    std::string output_file = argv[2];
    uint n, n_dim;
    count_lines_and_dimensions(input_file, n, n_dim);

    hid_t file = H5Fopen(output_file.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    hid_t dset = H5Dopen1(file, "Clusters");

    /*
    // Create data space
    hsize_t chunkSize =(this->m_totalSize / this->m_mpiSize) + 1;
    hsize_t count[2] = {this->m_size, 1};
    hsize_t start[2] = {this->m_mpiRank * chunkSize , 0};
    hid_t memSpace = H5Screate_simple(1, count, NULL);
    hid_t fileSpace = H5Dget_space(dset);

    // Select area to write
    H5Sselect_hyperslab(fileSpace,H5S_SELECT_SET, start, NULL, count, NULL);

    // Write
    H5Dwrite(dset, H5T_NATIVE_LONG, memSpace, fileSpace, H5P_DEFAULT, this->m_clusters);

          */
    // Close
    H5Dclose(dset);
    H5Fclose(file);
}