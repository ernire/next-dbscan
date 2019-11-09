//
// Created by Ernir Erlingsson (ernire@gmail.com, ernire.org) on 20.2.2019.
//
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

#include "nextdbscan.h"
#include "gtest/gtest.h"

TEST(ScanAccuracyAloiHsb2x2x2, m20_e001_d8_t2) {
    nextdbscan::result result = nextdbscan::start(20, 0.01, 2, "../input/aloi-hsb-2x2x2_trimmed.bin", 0, 1);
    EXPECT_EQ(result.clusters, 129);
    EXPECT_EQ(result.core_count, 55256);
    EXPECT_EQ(result.noise, 43312);
}

TEST(ScanAccuracyAloiHsb2x2x2, m10_e001_d8_t2) {
    nextdbscan::result result = nextdbscan::start(10, 0.01, 2, "../input/aloi-hsb-2x2x2_trimmed.csv", 0, 1);
    EXPECT_EQ(result.clusters, 297);
    EXPECT_EQ(result.core_count, 69606);
    EXPECT_EQ(result.noise, 30977);
}


TEST(ScanAccuracyBremenSmall, m30_e20_d3_t2) {
    nextdbscan::result result = nextdbscan::start(30, 20, 2, "../input/bremen_small.bin", 0, 1);
    EXPECT_EQ(result.clusters, 2972);
    EXPECT_EQ(result.core_count, 1234425);
    EXPECT_EQ(result.noise, 1463461);
}

TEST(ScanAccuracyBremenSmall, m30_e10_d3_t2) {
    nextdbscan::result result = nextdbscan::start(30, 10, 2, "../input/bremen_small.csv", 0, 1);
    EXPECT_EQ(result.clusters, 1202);
    EXPECT_EQ(result.core_count, 263100);
    EXPECT_EQ(result.noise, 2641881);
}

TEST(ScanAccuracyBremenSmall, m10_e20_d3_t2) {
    nextdbscan::result result = nextdbscan::start(10, 20, 2, "../input/bremen_small.bin", 0, 1);
    EXPECT_EQ(result.clusters, 10712);
    EXPECT_EQ(result.core_count, 2286646);
    EXPECT_EQ(result.noise, 482406);
}

TEST(ScanAccuracyBremenSmall, m100_e30_d3_t2) {
    nextdbscan::result result = nextdbscan::start(100, 30, 2, "../input/bremen_small.bin", 0, 1);
    EXPECT_EQ(result.clusters, 480);
    EXPECT_EQ(result.core_count, 839729);
    EXPECT_EQ(result.noise, 1906449);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
