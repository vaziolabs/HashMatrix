#include <gtest/gtest.h>
#include "hashmatrix.h"
#include "optimized_hashmatrix.h"
#include <random>
#include <chrono>

class HashMatrixTest : public ::testing::Test {
protected:
    static constexpr size_t SMALL_SIZE = 64;
    static constexpr size_t MEDIUM_SIZE = 256;
    static constexpr size_t LARGE_SIZE = 1024;
    
    // Helper function to create a matrix with known values
    template<typename MatrixType>
    void fillMatrix(MatrixType& matrix, const std::vector<std::tuple<size_t, size_t, double>>& values) {
        for (const auto& [row, col, val] : values) {
            matrix.insert(row, col, val);
        }
    }

    // Helper function to verify matrix values
    template<typename MatrixType>
    void verifyMatrix(const MatrixType& matrix, const std::vector<std::tuple<size_t, size_t, double>>& expected) {
        for (const auto& [row, col, val] : expected) {
            EXPECT_NEAR(matrix.get(row, col), val, 1e-10) 
                << "Mismatch at position (" << row << ", " << col << ")";
        }
    }

    // Helper to generate random sparse matrix data
    std::vector<std::tuple<size_t, size_t, double>> generateSparseData(
        size_t rows, size_t cols, double density) {
        std::vector<std::tuple<size_t, size_t, double>> data;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(1.0, 100.0);
        
        size_t elements = static_cast<size_t>(rows * cols * density);
        std::set<std::pair<size_t, size_t>> positions;
        
        while (positions.size() < elements) {
            size_t row = gen() % rows;
            size_t col = gen() % cols;
            if (positions.insert({row, col}).second) {
                data.emplace_back(row, col, dis(gen));
            }
        }
        
        return data;
    }
};

// Basic Operation Tests
TEST_F(HashMatrixTest, BasicOperations) {
    hash_matrix<double> matrix(SMALL_SIZE, SMALL_SIZE);
    optimized_hashmatrix<double> opt_matrix(SMALL_SIZE, SMALL_SIZE);
    
    std::vector<std::tuple<size_t, size_t, double>> testData = {
        {0, 0, 1.0}, {0, 1, 2.0}, {1, 0, 3.0}, {1, 1, 4.0}
    };
    
    // Test insert and get
    for (const auto& [row, col, val] : testData) {
        matrix.insert(row, col, val);
        opt_matrix.insert(row, col, val);
        
        EXPECT_DOUBLE_EQ(matrix.get(row, col), val);
        EXPECT_DOUBLE_EQ(opt_matrix.get(row, col), val);
    }
    
    // Test remove
    matrix.remove(0, 0);
    opt_matrix.remove(0, 0);
    EXPECT_DOUBLE_EQ(matrix.get(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(opt_matrix.get(0, 0), 0.0);
}

// Batch Insert Test
TEST_F(HashMatrixTest, BatchInsert) {
    hash_matrix<double> matrix(MEDIUM_SIZE, MEDIUM_SIZE);
    optimized_hashmatrix<double> opt_matrix(MEDIUM_SIZE, MEDIUM_SIZE);
    
    auto testData = generateSparseData(MEDIUM_SIZE, MEDIUM_SIZE, 0.1);
    
    matrix.batchInsert(testData);
    opt_matrix.batchInsert(testData);
    
    // Verify all values were inserted correctly
    for (const auto& [row, col, val] : testData) {
        EXPECT_DOUBLE_EQ(matrix.get(row, col), val);
        EXPECT_DOUBLE_EQ(opt_matrix.get(row, col), val);
    }
}

// Matrix Addition Test
TEST_F(HashMatrixTest, Addition) {
    hash_matrix<double> matrix1(SMALL_SIZE, SMALL_SIZE);
    hash_matrix<double> matrix2(SMALL_SIZE, SMALL_SIZE);
    optimized_hashmatrix<double> opt_matrix1(SMALL_SIZE, SMALL_SIZE);
    optimized_hashmatrix<double> opt_matrix2(SMALL_SIZE, SMALL_SIZE);
    
    std::vector<std::tuple<size_t, size_t, double>> data1 = {
        {0, 0, 1.0}, {0, 1, 2.0}, {1, 0, 3.0}, {1, 1, 4.0}
    };
    std::vector<std::tuple<size_t, size_t, double>> data2 = {
        {0, 0, 2.0}, {0, 1, 3.0}, {1, 0, 4.0}, {1, 1, 5.0}
    };
    
    fillMatrix(matrix1, data1);
    fillMatrix(matrix2, data2);
    fillMatrix(opt_matrix1, data1);
    fillMatrix(opt_matrix2, data2);
    
    auto result = matrix1.add(matrix2);
    auto opt_result = opt_matrix1.add(opt_matrix2);
    
    std::vector<std::tuple<size_t, size_t, double>> expected = {
        {0, 0, 3.0}, {0, 1, 5.0}, {1, 0, 7.0}, {1, 1, 9.0}
    };
    
    verifyMatrix(result, expected);
    verifyMatrix(opt_result, expected);
}

// Matrix Multiplication Test
TEST_F(HashMatrixTest, Multiplication) {
    hash_matrix<double> matrix1(SMALL_SIZE, SMALL_SIZE);
    hash_matrix<double> matrix2(SMALL_SIZE, SMALL_SIZE);
    optimized_hashmatrix<double> opt_matrix1(SMALL_SIZE, SMALL_SIZE);
    optimized_hashmatrix<double> opt_matrix2(SMALL_SIZE, SMALL_SIZE);
    
    std::vector<std::tuple<size_t, size_t, double>> data1 = {
        {0, 0, 1.0}, {0, 1, 2.0}, {1, 0, 3.0}, {1, 1, 4.0}
    };
    std::vector<std::tuple<size_t, size_t, double>> data2 = {
        {0, 0, 2.0}, {0, 1, 3.0}, {1, 0, 4.0}, {1, 1, 5.0}
    };
    
    fillMatrix(matrix1, data1);
    fillMatrix(matrix2, data2);
    fillMatrix(opt_matrix1, data1);
    fillMatrix(opt_matrix2, data2);
    
    auto result = matrix1.multiply(matrix2);
    auto opt_result = opt_matrix1.multiply(opt_matrix2);
    
    std::vector<std::tuple<size_t, size_t, double>> expected = {
        {0, 0, 10.0}, {0, 1, 13.0}, {1, 0, 22.0}, {1, 1, 29.0}
    };
    
    verifyMatrix(result, expected);
    verifyMatrix(opt_result, expected);
}

// Large Matrix Operations Test
TEST_F(HashMatrixTest, LargeMatrixOperations) {
    hash_matrix<double> matrix(LARGE_SIZE, LARGE_SIZE);
    optimized_hashmatrix<double> opt_matrix(LARGE_SIZE, LARGE_SIZE);
    
    auto testData = generateSparseData(LARGE_SIZE, LARGE_SIZE, 0.01);
    
    // Test batch insert
    matrix.batchInsert(testData);
    opt_matrix.batchInsert(testData);
    
    // Verify random elements
    for (size_t i = 0; i < 100; i++) {
        size_t row = rand() % LARGE_SIZE;
        size_t col = rand() % LARGE_SIZE;
        EXPECT_DOUBLE_EQ(matrix.get(row, col), opt_matrix.get(row, col));
    }
}

// Edge Cases Test
TEST_F(HashMatrixTest, EdgeCases) {
    hash_matrix<double> matrix(SMALL_SIZE, SMALL_SIZE);
    optimized_hashmatrix<double> opt_matrix(SMALL_SIZE, SMALL_SIZE);
    
    // Test zero values
    matrix.insert(0, 0, 0.0);
    opt_matrix.insert(0, 0, 0.0);
    EXPECT_DOUBLE_EQ(matrix.get(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(opt_matrix.get(0, 0), 0.0);
    
    // Test overwriting values
    matrix.insert(0, 0, 1.0);
    matrix.insert(0, 0, 2.0);
    opt_matrix.insert(0, 0, 1.0);
    opt_matrix.insert(0, 0, 2.0);
    EXPECT_DOUBLE_EQ(matrix.get(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(opt_matrix.get(0, 0), 2.0);
    
    // Test out of bounds
    EXPECT_THROW(matrix.insert(SMALL_SIZE, 0, 1.0), std::out_of_range);
    EXPECT_THROW(opt_matrix.insert(SMALL_SIZE, 0, 1.0), std::out_of_range);
    EXPECT_THROW(matrix.get(SMALL_SIZE, 0), std::out_of_range);
    EXPECT_THROW(opt_matrix.get(SMALL_SIZE, 0), std::out_of_range);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
