#include <benchmark/benchmark.h>
#include "hashmatrix.h"
#include <vector>
#include <unordered_map>
#include <random>

struct TestData {
    std::vector<std::tuple<int, int, double>> sparseData;
    std::vector<std::tuple<int, int, double>> queryData;
    std::vector<std::tuple<int, int, double>> updateData;
    std::vector<std::pair<int, int>> deleteData;
    
    static TestData generate(int size, double sparsity) {
        TestData data;
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        
        // Generate initial sparse data
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (dist(rng) < sparsity) {
                    data.sparseData.emplace_back(i, j, dist(rng));
                }
            }
        }
        
        // Generate query locations (mix of existing and non-existing)
        for (int i = 0; i < size * sparsity * 2; i++) {
            data.queryData.emplace_back(
                rand() % size,
                rand() % size,
                dist(rng)
            );
        }
        
        // Generate update data
        for (int i = 0; i < size * sparsity; i++) {
            data.updateData.emplace_back(
                rand() % size,
                rand() % size,
                dist(rng)
            );
        }
        
        // Generate delete locations
        for (int i = 0; i < size * sparsity / 2; i++) {
            data.deleteData.emplace_back(
                rand() % size,
                rand() % size
            );
        }
        
        return data;
    }
};

// Benchmark  matrix operations
static void Matrix_Insert(benchmark::State& state) {
    const int size = state.range(0);
    const double sparsity = 0.01;
    auto testData = TestData::generate(size, sparsity);
    
    for (auto _ : state) {
        std::vector<std::vector<double>> matrix(size, std::vector<double>(size, 0.0));
        for (const auto& [i, j, val] : testData.sparseData) {
            matrix[i][j] = val;
        }
        benchmark::DoNotOptimize(matrix);
    }
}

static void Matrix_Access(benchmark::State& state) {
    const int size = state.range(0);
    const double sparsity = 0.01;
    auto testData = TestData::generate(size, sparsity);
    
    std::vector<std::vector<double>> matrix(size, std::vector<double>(size, 0.0));
    for (const auto& [i, j, val] : testData.sparseData) {
        matrix[i][j] = val;
    }
    
    for (auto _ : state) {
        double sum = 0.0;
        for (const auto& [i, j, _] : testData.queryData) {
            sum += matrix[i][j];
        }
        benchmark::DoNotOptimize(sum);
    }
}

static void Matrix_Update(benchmark::State& state) {
    const int size = state.range(0);
    const double sparsity = 0.01;
    auto testData = TestData::generate(size, sparsity);
    
    std::vector<std::vector<double>> matrix(size, std::vector<double>(size, 0.0));
    
    for (auto _ : state) {
        for (const auto& [i, j, val] : testData.updateData) {
            matrix[i][j] = val;
        }
        benchmark::DoNotOptimize(matrix);
    }
}

static void Matrix_Search(benchmark::State& state) {
    const int size = state.range(0);
    const double sparsity = 0.01;
    auto testData = TestData::generate(size, sparsity);
    
    // Initialize matrix
    std::vector<std::vector<double>> matrix(size, std::vector<double>(size, 0.0));
    for (const auto& [i, j, val] : testData.sparseData) {
        matrix[i][j] = val;
    }
    
    for (auto _ : state) {
        double sum = 0.0;
        for (const auto& [i, j, _] : testData.queryData) {
            if (matrix[i][j] != 0.0) {
                sum += matrix[i][j];
            }
        }
        benchmark::DoNotOptimize(sum);
    }
}

// Benchmark  hashmap operations
static void HashMap_Insert(benchmark::State& state) {
    const int size = state.range(0);
    const double sparsity = 0.01;
    auto testData = TestData::generate(size, sparsity);
    
    for (auto _ : state) {
        std::unordered_map<std::pair<int, int>, double, PairHash> matrix;
        for (const auto& [i, j, val] : testData.sparseData) {
            matrix[{i, j}] = val;
        }
        benchmark::DoNotOptimize(matrix);
    }
}

static void HashMap_Access(benchmark::State& state) {
    const int size = state.range(0);
    const double sparsity = 0.01;
    auto testData = TestData::generate(size, sparsity);
    
    std::unordered_map<std::pair<int, int>, double, PairHash> matrix;
    for (const auto& [i, j, val] : testData.sparseData) {
        matrix[{i, j}] = val;
    }
    
    for (auto _ : state) {
        double sum = 0.0;
        for (const auto& [i, j, _] : testData.queryData) {
            auto it = matrix.find({i, j});
            if (it != matrix.end()) {
                sum += it->second;
            }
        }
        benchmark::DoNotOptimize(sum);
    }
}

static void HashMap_Update(benchmark::State& state) {
    const int size = state.range(0);
    const double sparsity = 0.01;
    auto testData = TestData::generate(size, sparsity);
    
    std::unordered_map<std::pair<int, int>, double, PairHash> matrix;
    
    for (auto _ : state) {
        for (const auto& [i, j, val] : testData.updateData) {
            matrix[{i, j}] = val;
        }
        benchmark::DoNotOptimize(matrix);
    }
}

static void HashMap_Delete(benchmark::State& state) {
    const int size = state.range(0);
    const double sparsity = 0.01;
    auto testData = TestData::generate(size, sparsity);
    
    for (auto _ : state) {
        std::unordered_map<std::pair<int, int>, double, PairHash> matrix;
        // Insert data
        for (const auto& [i, j, val] : testData.sparseData) {
            matrix[{i, j}] = val;
        }
        // Delete operations
        for (const auto& [i, j] : testData.deleteData) {
            matrix.erase({i, j});
        }
        benchmark::DoNotOptimize(matrix);
    }
}

static void HashMap_Add(benchmark::State& state) {
    const int size = state.range(0);
    const double sparsity = 0.01;
    
    std::unordered_map<std::pair<int, int>, double, PairHash> matrix1, matrix2;
    auto testData = TestData::generate(size, sparsity);
    
    // Initialize matrices
    for (const auto& [i, j, val] : testData.sparseData) {
        matrix1[{i, j}] = val;
        matrix2[{i, j}] = val * 0.5;
    }
    
    for (auto _ : state) {
        std::unordered_map<std::pair<int, int>, double, PairHash> result;
        // Add corresponding elements
        for (const auto& [key, val] : matrix1) {
            auto it = matrix2.find(key);
            if (it != matrix2.end()) {
                result[key] = val + it->second;
            } else {
                result[key] = val;
            }
        }
        // Add remaining elements from matrix2
        for (const auto& [key, val] : matrix2) {
            if (matrix1.find(key) == matrix1.end()) {
                result[key] = val;
            }
        }
        benchmark::DoNotOptimize(result);
    }
}

static void HashMap_Multiply(benchmark::State& state) {
    const int size = state.range(0);
    const double sparsity = 0.01;
    
    std::unordered_map<std::pair<int, int>, double, PairHash> matrix1, matrix2;
    auto testData = TestData::generate(size, sparsity);
    
    // Initialize matrices
    for (const auto& [i, j, val] : testData.sparseData) {
        matrix1[{i, j}] = val;
        matrix2[{i, j}] = val * 0.5;
    }
    
    for (auto _ : state) {
        std::unordered_map<std::pair<int, int>, double, PairHash> result;
        // Perform sparse matrix multiplication
        for (const auto& [pos1, val1] : matrix1) {
            int i = pos1.first;
            int k = pos1.second;
            for (const auto& [pos2, val2] : matrix2) {
                if (k == pos2.first) {  // Only multiply if indices match
                    int j = pos2.second;
                    auto resultKey = std::make_pair(i, j);
                    result[resultKey] += val1 * val2;
                }
            }
        }
        benchmark::DoNotOptimize(result);
    }
}

static void HashMap_Search(benchmark::State& state) {
    const int size = state.range(0);
    const double sparsity = 0.01;
    auto testData = TestData::generate(size, sparsity);
    
    // Initialize hashmap
    std::unordered_map<std::pair<int, int>, double, PairHash> matrix;
    for (const auto& [i, j, val] : testData.sparseData) {
        matrix[{i, j}] = val;
    }
    
    for (auto _ : state) {
        double sum = 0.0;
        for (const auto& [i, j, _] : testData.queryData) {
            auto it = matrix.find({i, j});
            if (it != matrix.end()) {
                sum += it->second;
            }
        }
        benchmark::DoNotOptimize(sum);
    }
}

// Benchmark HashMatrix operations
static void HashMatrix_Insert(benchmark::State& state) {
    const int size = state.range(0);
    const double sparsity = 0.01;
    auto testData = TestData::generate(size, sparsity);
    
    for (auto _ : state) {
        HashMatrix<double> matrix;
        for (const auto& [i, j, val] : testData.sparseData) {
            matrix.insert(i, j, val);
        }
        benchmark::DoNotOptimize(matrix);
    }
}

static void HashMatrix_Access(benchmark::State& state) {
    const int size = state.range(0);
    const double sparsity = 0.01;
    auto testData = TestData::generate(size, sparsity);
    
    HashMatrix<double> matrix;
    for (const auto& [i, j, val] : testData.sparseData) {
        matrix.insert(i, j, val);
    }
    
    for (auto _ : state) {
        double sum = 0.0;
        for (const auto& [i, j, _] : testData.queryData) {
            sum += matrix.get(i, j);
        }
        benchmark::DoNotOptimize(sum);
    }
}

static void HashMatrix_Update(benchmark::State& state) {
    const int size = state.range(0);
    const double sparsity = 0.01;
    auto testData = TestData::generate(size, sparsity);
    
    HashMatrix<double> matrix;
    
    for (auto _ : state) {
        for (const auto& [i, j, val] : testData.updateData) {
            matrix.insert(i, j, val);
        }
        benchmark::DoNotOptimize(matrix);
    }
}

// Matrix Addition Benchmarks
static void Matrix_Add(benchmark::State& state) {
    const int size = state.range(0);
    const double sparsity = 0.01;
    
    // Create two matrices
    std::vector<std::vector<double>> matrix1(size, std::vector<double>(size, 0.0));
    std::vector<std::vector<double>> matrix2(size, std::vector<double>(size, 0.0));
    auto testData = TestData::generate(size, sparsity);
    
    // Initialize matrices
    for (const auto& [i, j, val] : testData.sparseData) {
        matrix1[i][j] = val;
        matrix2[i][j] = val * 0.5;
    }
    
    for (auto _ : state) {
        std::vector<std::vector<double>> result(size, std::vector<double>(size, 0.0));
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                result[i][j] = matrix1[i][j] + matrix2[i][j];
            }
        }
        benchmark::DoNotOptimize(result);
    }
}

static void Matrix_Multiply(benchmark::State& state) {
    const int size = state.range(0);
    const double sparsity = 0.01;
    
    std::vector<std::vector<double>> matrix1(size, std::vector<double>(size, 0.0));
    std::vector<std::vector<double>> matrix2(size, std::vector<double>(size, 0.0));
    auto testData = TestData::generate(size, sparsity);
    
    for (const auto& [i, j, val] : testData.sparseData) {
        matrix1[i][j] = val;
        matrix2[i][j] = val * 0.5;
    }
    
    for (auto _ : state) {
        std::vector<std::vector<double>> result(size, std::vector<double>(size, 0.0));
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                for (int k = 0; k < size; k++) {
                    result[i][j] += matrix1[i][k] * matrix2[k][j];
                }
            }
        }
        benchmark::DoNotOptimize(result);
    }
}

static void HashMatrix_Add(benchmark::State& state) {
    const int size = state.range(0);
    const double sparsity = 0.01;
    
    HashMatrix<double> matrix1, matrix2;
    auto testData = TestData::generate(size, sparsity);
    
    for (const auto& [i, j, val] : testData.sparseData) {
        matrix1.insert(i, j, val);
        matrix2.insert(i, j, val * 0.5);
    }
    
    for (auto _ : state) {
        auto result = matrix1.add(matrix2);
        benchmark::DoNotOptimize(result);
    }
}

static void HashMatrix_Multiply(benchmark::State& state) {
    const int size = state.range(0);
    const double sparsity = 0.01;
    
    HashMatrix<double> matrix1, matrix2;
    auto testData = TestData::generate(size, sparsity);
    
    for (const auto& [i, j, val] : testData.sparseData) {
        matrix1.insert(i, j, val);
        matrix2.insert(i, j, val * 0.5);
    }
    
    for (auto _ : state) {
        auto result = matrix1.multiply(matrix2, size);
        benchmark::DoNotOptimize(result);
    }
}

static void HashMatrix_Search(benchmark::State& state) {
    const int size = state.range(0);
    const double sparsity = 0.01;
    auto testData = TestData::generate(size, sparsity);
    
    // Initialize HashMatrix
    HashMatrix<double> matrix;
    for (const auto& [i, j, val] : testData.sparseData) {
        matrix.insert(i, j, val);
    }
    
    for (auto _ : state) {
        double sum = 0.0;
        for (const auto& [i, j, _] : testData.queryData) {
            double val = matrix.get(i, j);
            if (val != 0.0) {
                sum += val;
            }
        }
        benchmark::DoNotOptimize(sum);
    }
}

// Register benchmarks with different sizes and metrics
BENCHMARK(Matrix_Insert)
    ->RangeMultiplier(2)->Range(64, 1024)->Unit(benchmark::kMicrosecond);
BENCHMARK(Matrix_Access)
    ->RangeMultiplier(2)->Range(64, 1024)->Unit(benchmark::kMicrosecond);
BENCHMARK(Matrix_Update)
    ->RangeMultiplier(2)->Range(64, 1024)->Unit(benchmark::kMicrosecond);
BENCHMARK(Matrix_Add)
    ->RangeMultiplier(2)->Range(64, 1024)->Unit(benchmark::kMicrosecond);
BENCHMARK(Matrix_Multiply)
    ->RangeMultiplier(2)->Range(64, 1024)->Unit(benchmark::kMicrosecond);
BENCHMARK(Matrix_Search)
    ->RangeMultiplier(2)->Range(64, 1024)->Unit(benchmark::kMicrosecond);

BENCHMARK(HashMap_Insert)
    ->RangeMultiplier(2)->Range(64, 1024)->Unit(benchmark::kMicrosecond);
BENCHMARK(HashMap_Access)
    ->RangeMultiplier(2)->Range(64, 1024)->Unit(benchmark::kMicrosecond);
BENCHMARK(HashMap_Update)
    ->RangeMultiplier(2)->Range(64, 1024)->Unit(benchmark::kMicrosecond);
BENCHMARK(HashMap_Delete)
    ->RangeMultiplier(2)->Range(64, 1024)->Unit(benchmark::kMicrosecond);
BENCHMARK(HashMap_Add)
    ->RangeMultiplier(2)->Range(64, 1024)->Unit(benchmark::kMicrosecond);
BENCHMARK(HashMap_Multiply)
    ->RangeMultiplier(2)->Range(64, 1024)->Unit(benchmark::kMicrosecond);
BENCHMARK(HashMap_Search)
    ->RangeMultiplier(2)->Range(64, 1024)->Unit(benchmark::kMicrosecond);

BENCHMARK(HashMatrix_Insert)
    ->RangeMultiplier(2)->Range(64, 1024)->Unit(benchmark::kMicrosecond);
BENCHMARK(HashMatrix_Access)
    ->RangeMultiplier(2)->Range(64, 1024)->Unit(benchmark::kMicrosecond);
BENCHMARK(HashMatrix_Update)
    ->RangeMultiplier(2)->Range(64, 1024)->Unit(benchmark::kMicrosecond);

BENCHMARK(HashMatrix_Add)
    ->RangeMultiplier(2)->Range(64, 1024)->Unit(benchmark::kMicrosecond);
BENCHMARK(HashMatrix_Multiply)
    ->RangeMultiplier(2)->Range(64, 1024)->Unit(benchmark::kMicrosecond);
BENCHMARK(HashMatrix_Search)
    ->RangeMultiplier(2)->Range(64, 1024)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();