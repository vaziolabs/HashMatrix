g++ -O3 benchmark.cpp -lbenchmark -lpthread -fopenmp -o benchmark
./benchmark --benchmark_format=console --benchmark_color=true