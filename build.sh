#!/bin/bash
g++ -mavx2 -std=c++17 test.cpp -o test -lgtest -lgtest_main -pthread
g++ -std=c++17 -O3 -march=native -mavx -mavx2 -mfma benchmark.cpp -lbenchmark -lpthread -o benchmark
./benchmark --benchmark_format=console --benchmark_color=true
./test