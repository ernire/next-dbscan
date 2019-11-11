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

#include <sstream>
#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>

int count_lines(const std::string &in_file) {
    std::ifstream is(in_file);
    std::string line;
    int cnt = 0;
    while (std::getline(is, line)) {
        ++cnt;
    }
    return cnt;
}

void write_output(const std::string &out_file, const float *v_points, const int max_d, const int n) {
    std::ofstream os(out_file);
    for (int i = 0; i < n; i ++) {
        for (int j = 0; j < max_d; j++) {
            os << v_points[i*max_d + j] << " ";
        }
        os << '\n';
    }
    os.flush();
    os.close();
}

void read_input(const std::string &in_file, float *v_points, int max_d) noexcept {
    std::ifstream is(in_file);
    std::string line, buf;
    std::stringstream ss;
    int index = 0;
    auto t1 = std::chrono::high_resolution_clock::now();
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
    auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << std::endl;
    std::cout << "Read input took: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
              << " milliseconds\n";
}

int main(int argc, char* const* argv) {
    std::string in_file = argv[1];
    std::string out_file = argv[2];
    std::cout << "Input file: " << in_file << std::endl;
    uint max_d = 7;

    uint n = count_lines(in_file);
    std::cout << "Number of samples: " << n << std::endl;
    auto *v_points = new float[n*max_d];
    read_input(in_file, v_points, max_d);

    float mins[max_d];
    float maxs[max_d];
    float minmaxs[max_d];
    for (uint j = 0; j < max_d; j++) {
        mins[j] = v_points[j];
        maxs[j] = v_points[j];
    }
    std::cout << std::endl;
    for (uint i = 1; i < n; i ++) {
        for (uint j = 0; j < max_d; j++) {
            float val = v_points[i*max_d + j];
            if (val < mins[j]) {
                mins[j] = val;
            }
            if (val > maxs[j]) {
                maxs[j] = val;
            }
        }
    }
    for (uint j = 0; j < max_d; j++) {
        std::cout << "Min: " << mins[j] << ", Max: " << maxs[j] << std::endl;
        minmaxs[j] = maxs[j] - mins[j];
    }
    for (uint i = 0; i < n; i ++) {
        for (uint j = 0; j < max_d; j++) {
            if (minmaxs[j] == 0) {
                v_points[i*max_d + j] = 0;
            } else {
                float val = v_points[i*max_d + j];
                v_points[i*max_d + j] = ((val - mins[j]) / minmaxs[j]) * 100000;
            }
        }
    }
    std::cout << "Writing output file: " << out_file << std::endl;
    write_output(out_file, v_points, max_d, n);
}