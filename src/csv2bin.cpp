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

#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <functional>
#include <iterator>

void count_lines_and_dimensions(const std::string &in_file, int &lines, int &dimensions) noexcept {
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

bool convert_light_to_bin(const std::string &in_file, char *out_file, const std::function<void(int, int)> &meta_callback) {
    std::vector<float *> features;
    std::vector<int> classes;
    int max_features = 0;
    int total_samples = 0;
    count_lines_and_dimensions(in_file, total_samples, max_features);
    meta_callback(total_samples, max_features);
    std::ifstream is(in_file, std::ios::in | std::ifstream::binary);
    std::ofstream os(out_file, std::ios::out | std::ofstream::binary);
    if (!is.is_open() || !os.is_open()) {
        return false;
    }
    os.write(reinterpret_cast<const char *>(&total_samples), sizeof(int));
    os.write(reinterpret_cast<const char *>(&max_features), sizeof(int));
    is.clear();
    is.seekg(0, std::istream::beg);
    std::string line;
    float line_features[max_features];
    int line_cnt = 0;
    int one_tenth = total_samples / 10;
    int percent = 0;
    while (std::getline(is, line)) {
//        std::fill(sample_features, sample_features+3, 0);
        std::istringstream iss(line);
        // class
        for (int i = 0; i < max_features; ++i) {
            iss >> line_features[i];
        }
        for (int i = 0; i < max_features; ++i) {
            os.write(reinterpret_cast<const char *>(&line_features[i]), sizeof(float));
        }
        if (++line_cnt % one_tenth == 0) {
            percent += 10;
            std::cout << "Finished: " << percent << "%" << std::endl;
        }
    }
    os.flush();
    os.close();
    is.close();
    return true;
}

std::streampos get_file_size(const char *filePath) {
    std::streampos fsize = 0;
    std::ifstream file(filePath, std::ios::in | std::ios::binary);

    fsize = file.tellg();
    file.seekg(0, std::ios::end);
    fsize = file.tellg() - fsize;
    file.close();

    return fsize;
}

void light_to_bin(char *in_file, char *out_file) {
    std::cout << "Converting " << in_file << " with size: " << get_file_size(in_file) << std::endl;
    if (!convert_light_to_bin(in_file, out_file, [](int total_samples, int no_of_features) -> void {
        std::cout << "Number of samples: " << total_samples << std::endl;
        std::cout << "Number of dimensions: " << no_of_features << std::endl;
    })) {
        std::cout << "Unable to open input or output files: " << in_file << ", " << out_file << std::endl;
    }
}

int main(int argc, char* const* argv) {
    if (argc == 3) {
        light_to_bin(argv[1], argv[2]);
    } else {
        std::cout << "Wrong number of parameters, should be 2 (input file, output file)" << std::endl;
    }
    exit(0);
}