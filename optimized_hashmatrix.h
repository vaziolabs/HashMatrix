#ifndef OPTIMIZED_HASHMATRIX_H
#define OPTIMIZED_HASHMATRIX_H

#include <unordered_map>
#include <vector>
#include <tuple>
#include <stdexcept>
#include <algorithm>
#include "hashmatrix.h"

template <typename T>
class optimized_hashmatrix {
private:
    // Block size for dense storage - each dense block is 64x64
    static constexpr size_t BLOCK_SIZE = 64;
    // When a block becomes more than 50% full, it converts to dense storage
    static constexpr double DENSITY_THRESHOLD = 0.5;

    /*
     * Hybrid Storage Strategy:
     * 1. Sparse regions use hash-based storage (sparseData)
     * 2. Dense regions use 64x64 blocks (denseBlocks)
     * The matrix automatically switches between these based on density
     */
    
    // Stores sparse elements as (row,col) -> value mappings
    std::unordered_map<std::pair<int, int>, T, PairHash> sparseData;

    // Dense block storage - used when a 64x64 region becomes >50% full
    struct DenseBlock {
        std::vector<std::vector<T>> data;
        bool isDirty; // Tracks if block needs density recalculation
        
        DenseBlock() : isDirty(false) {
            data.resize(BLOCK_SIZE, std::vector<T>(BLOCK_SIZE, 0));
        }
    };
    std::unordered_map<std::pair<int, int>, DenseBlock, PairHash> denseBlocks;

    // Tracks number of non-zero elements in each block for density calculations
    std::unordered_map<std::pair<int, int>, size_t, PairHash> blockNonZeroCount;
    size_t numRows, numCols;

    std::pair<int, int> getBlockCoords(int row, int col) const {
        return {row / BLOCK_SIZE, col / BLOCK_SIZE};
    }

    std::pair<int, int> getLocalCoords(int row, int col) const {
        return {row % BLOCK_SIZE, col % BLOCK_SIZE};
    }

    double getBlockDensity(const std::pair<int, int>& blockCoord) const {
        auto it = blockNonZeroCount.find(blockCoord);
        if (it == blockNonZeroCount.end()) return 0.0;
        return static_cast<double>(it->second) / (BLOCK_SIZE * BLOCK_SIZE);
    }

    void convertBlockToDense(const std::pair<int, int>& blockCoord) {
        DenseBlock block;
        int baseRow = blockCoord.first * BLOCK_SIZE;
        int baseCol = blockCoord.second * BLOCK_SIZE;

        // Copy data from sparse to dense
        for (int i = 0; i < BLOCK_SIZE; i++) {
            for (int j = 0; j < BLOCK_SIZE; j++) {
                auto it = sparseData.find({baseRow + i, baseCol + j});
                if (it != sparseData.end()) {
                    block.data[i][j] = it->second;
                    sparseData.erase(it);
                }
            }
        }

        denseBlocks[blockCoord] = std::move(block);
    }

    void convertBlockToSparse(const std::pair<int, int>& blockCoord) {
        const auto& block = denseBlocks[blockCoord];
        int baseRow = blockCoord.first * BLOCK_SIZE;
        int baseCol = blockCoord.second * BLOCK_SIZE;

        // Copy data from dense to sparse
        for (int i = 0; i < BLOCK_SIZE; i++) {
            for (int j = 0; j < BLOCK_SIZE; j++) {
                if (block.data[i][j] != 0) {
                    sparseData[{baseRow + i, baseCol + j}] = block.data[i][j];
                }
            }
        }

        denseBlocks.erase(blockCoord);
    }

    void updateBlockDensity(const std::pair<int, int>& blockCoord, bool increment) {
        auto& count = blockNonZeroCount[blockCoord];
        if (increment) {
            count++;
            if (getBlockDensity(blockCoord) > DENSITY_THRESHOLD && 
                denseBlocks.find(blockCoord) == denseBlocks.end()) {
                convertBlockToDense(blockCoord);
            }
        } else {
            count--;
            if (getBlockDensity(blockCoord) < DENSITY_THRESHOLD && 
                denseBlocks.find(blockCoord) != denseBlocks.end()) {
                convertBlockToSparse(blockCoord);
            }
        }
    }

public:
    /*
     * Matrix Operations:
     * - insert(): Automatically handles sparse/dense conversion
     * - get(): Checks dense blocks first (faster), then sparse storage
     * - multiply(): Uses block-based multiplication for better cache utilization
     * - add(): Processes dense blocks in parallel, then handles sparse elements
     */

    optimized_hashmatrix(size_t rows, size_t cols) 
        : numRows(rows)
        , numCols(cols)
        , sparseData(10, PairHash())
        , denseBlocks(10, PairHash())
        , blockNonZeroCount(10, PairHash())
    {
    }

    void insert(int row, int col, T value) {
        if (row >= numRows || col >= numCols) {
            throw std::out_of_range("Matrix indices out of bounds");
        }

        auto blockCoord = getBlockCoords(row, col);
        auto localCoord = getLocalCoords(row, col);

        auto blockIt = denseBlocks.find(blockCoord);
        if (blockIt != denseBlocks.end()) {
            // Block is dense
            T oldValue = blockIt->second.data[localCoord.first][localCoord.second];
            blockIt->second.data[localCoord.first][localCoord.second] = value;
            
            if ((oldValue == 0) != (value == 0)) {
                updateBlockDensity(blockCoord, value != 0);
            }
        } else {
            // Block is sparse
            auto [sparseIt, inserted] = sparseData.try_emplace({row, col}, value);
            if (!inserted) {
                bool wasZero = sparseIt->second == 0;
                sparseIt->second = value;
                if (wasZero != (value == 0)) {
                    updateBlockDensity(blockCoord, value != 0);
                }
            } else if (value != 0) {
                updateBlockDensity(blockCoord, true);
            }
        }
    }

    T get(int row, int col) const {
        if (row >= numRows || col >= numCols) {
            throw std::out_of_range("Matrix indices out of bounds");
        }

        auto blockCoord = getBlockCoords(row, col);
        auto localCoord = getLocalCoords(row, col);

        // First check dense blocks (faster path)
        auto blockIt = denseBlocks.find(blockCoord);
        if (blockIt != denseBlocks.end()) {
            return blockIt->second.data[localCoord.first][localCoord.second];
        }

        // Then check sparse storage
        auto it = sparseData.find({row, col});  // Using global coordinates
        return it != sparseData.end() ? it->second : T{};
    }

    /*
     * Batch Operations:
     * - batchInsert(): Optimizes multiple insertions by grouping them by block
     * - This reduces the number of density recalculations and conversions
     */
    void batchInsert(const std::vector<std::tuple<int, int, T>>& data) {
        // Group elements by block
        std::unordered_map<std::pair<int, int>, std::vector<std::tuple<int, int, T>>, PairHash> blockGroups;
        
        for (const auto& [row, col, val] : data) {
            auto blockCoord = getBlockCoords(row, col);
            auto localCoord = getLocalCoords(row, col);
            blockGroups[blockCoord].emplace_back(localCoord.first, localCoord.second, val);
        }
        
        // Process each block
        for (const auto& entry : blockGroups) {
            const auto& blockCoord = entry.first;
            const auto& blockValues = entry.second;
            
            size_t nonZeroCount = blockValues.size();
            double density = static_cast<double>(nonZeroCount) / (BLOCK_SIZE * BLOCK_SIZE);
            
            if (density > DENSITY_THRESHOLD) {
                // Use dense storage
                auto& block = denseBlocks[blockCoord];
                for (const auto& [i, j, val] : blockValues) {
                    block.data[i][j] = val;
                }
                block.isDirty = true;
            } else {
                // Use sparse storage
                for (const auto& [i, j, val] : blockValues) {
                    sparseData[{blockCoord.first * BLOCK_SIZE + i, 
                              blockCoord.second * BLOCK_SIZE + j}] = val;
                }
            }
            
            blockNonZeroCount[blockCoord] = nonZeroCount;
        }
    }

    /*
     * Matrix Multiplication:
     * 1. Divides matrices into blocks for better cache usage
     * 2. Uses OpenMP for parallel processing
     * 3. Optimizes dense*dense multiplication
     * 4. Has special handling for sparse blocks
     */
    optimized_hashmatrix<T> multiply(const optimized_hashmatrix<T>& other) const {
        if (numCols != other.numRows) {
            throw std::invalid_argument("Matrix dimensions must match for multiplication");
        }

        optimized_hashmatrix<T> result(numRows, other.numCols);
        size_t numBlocksK = (numCols + BLOCK_SIZE - 1) / BLOCK_SIZE;

        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < (numRows + BLOCK_SIZE - 1) / BLOCK_SIZE; i++) {
            for (size_t j = 0; j < (other.numCols + BLOCK_SIZE - 1) / BLOCK_SIZE; j++) {
                std::vector<T> tempBlock(BLOCK_SIZE * BLOCK_SIZE, 0);
                
                for (size_t k = 0; k < numBlocksK; k++) {
                    multiplyBlocks({i, k}, {k, j}, other, tempBlock);
                }

                // Insert non-zero results
                for (size_t bi = 0; bi < BLOCK_SIZE; bi++) {
                    for (size_t bj = 0; bj < BLOCK_SIZE; bj++) {
                        T val = tempBlock[bi * BLOCK_SIZE + bj];
                        if (val != 0) {
                            result.insert(i * BLOCK_SIZE + bi, j * BLOCK_SIZE + bj, val);
                        }
                    }
                }
            }
        }

        return result;
    }

    void remove(int row, int col) {
        if (row >= numRows || col >= numCols) {
            throw std::out_of_range("Matrix indices out of bounds");
        }

        auto blockCoord = getBlockCoords(row, col);
        auto localCoord = getLocalCoords(row, col);

        auto blockIt = denseBlocks.find(blockCoord);
        if (blockIt != denseBlocks.end()) {
            // Block is dense
            if (blockIt->second.data[localCoord.first][localCoord.second] != 0) {
                blockIt->second.data[localCoord.first][localCoord.second] = 0;
                updateBlockDensity(blockCoord, false);
            }
        } else {
            // Block is sparse
            auto it = sparseData.find({row, col});
            if (it != sparseData.end()) {
                sparseData.erase(it);
                updateBlockDensity(blockCoord, false);
            }
        }
    }

    optimized_hashmatrix<T> add(const optimized_hashmatrix<T>& other) const {
        if (numRows != other.numRows || numCols != other.numCols) {
            throw std::invalid_argument("Matrix dimensions must match for addition");
        }

        optimized_hashmatrix<T> result(numRows, numCols);

        // Process dense blocks first
        std::set<std::pair<int, int>> processedBlocks;

        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < (numRows + BLOCK_SIZE - 1) / BLOCK_SIZE; i++) {
            for (size_t j = 0; j < (numCols + BLOCK_SIZE - 1) / BLOCK_SIZE; j++) {
                std::pair<int, int> blockCoord{i, j};
                bool isDenseA = denseBlocks.count(blockCoord);
                bool isDenseB = other.denseBlocks.count(blockCoord);

                if (isDenseA && isDenseB) {
                    // Both blocks are dense
                    const auto& blockA = denseBlocks.at(blockCoord);
                    const auto& blockB = other.denseBlocks.at(blockCoord);
                    
                    for (size_t bi = 0; bi < BLOCK_SIZE; bi++) {
                        for (size_t bj = 0; bj < BLOCK_SIZE; bj++) {
                            T sum = blockA.data[bi][bj] + blockB.data[bi][bj];
                            if (sum != 0) {
                                result.insert(i * BLOCK_SIZE + bi, j * BLOCK_SIZE + bj, sum);
                            }
                        }
                    }
                    processedBlocks.insert(blockCoord);
                }
            }
        }

        // Process remaining blocks
        for (size_t i = 0; i < numRows; i++) {
            for (size_t j = 0; j < numCols; j++) {
                auto blockCoord = getBlockCoords(i, j);
                if (processedBlocks.count(blockCoord)) continue;

                T sum = get(i, j) + other.get(i, j);
                if (sum != 0) {
                    result.insert(i, j, sum);
                }
            }
        }

        return result;
    }

    /*
     * Iterator Support:
     * Provides efficient iteration over non-zero elements only
     * Useful for sparse matrix algorithms and visualization
     */
    class Iterator {
    private:
        const optimized_hashmatrix<T>* matrix;
        size_t currentRow;
        size_t currentCol;
        
        void findNextNonZero() {
            while (currentRow < matrix->numRows) {
                while (currentCol < matrix->numCols) {
                    if (matrix->get(currentRow, currentCol) != 0) {
                        return;
                    }
                    currentCol++;
                }
                currentRow++;
                currentCol = 0;
            }
        }

    public:
        Iterator(const optimized_hashmatrix<T>* m, size_t row = 0, size_t col = 0)
            : matrix(m), currentRow(row), currentCol(col) {
            if (matrix) findNextNonZero();
        }

        struct Element {
            size_t row;
            size_t col;
            T value;
        };

        Element operator*() const {
            return {currentRow, currentCol, matrix->get(currentRow, currentCol)};
        }

        Iterator& operator++() {
            if (currentCol < matrix->numCols) {
                currentCol++;
                findNextNonZero();
            }
            return *this;
        }

        bool operator==(const Iterator& other) const {
            return currentRow == other.currentRow && currentCol == other.currentCol;
        }

        bool operator!=(const Iterator& other) const {
            return !(*this == other);
        }
    };

    Iterator begin() const { return Iterator(this); }
    Iterator end() const { return Iterator(this, numRows, 0); }

    // Utility methods
    size_t nonZeroCount() const {
        size_t count = 0;
        for (const auto& [_, nz] : blockNonZeroCount) {
            count += nz;
        }
        return count;
    }

    double sparsity() const {
        return 1.0 - static_cast<double>(nonZeroCount()) / (numRows * numCols);
    }

    void clear() {
        sparseData.clear();
        denseBlocks.clear();
        blockNonZeroCount.clear();
    }

    std::pair<size_t, size_t> dimensions() const {
        return {numRows, numCols};
    }

    // Debug/Statistics methods
    size_t getDenseBlockCount() const {
        return denseBlocks.size();
    }

    size_t getSparseElementCount() const {
        return sparseData.size();
    }

    void printStats() const {
        std::cout << "Matrix Statistics:\n"
                  << "Dimensions: " << numRows << "x" << numCols << "\n"
                  << "Non-zero elements: " << nonZeroCount() << "\n"
                  << "Sparsity: " << sparsity() * 100 << "%\n"
                  << "Dense blocks: " << getDenseBlockCount() << "\n"
                  << "Sparse elements: " << getSparseElementCount() << "\n";
    }

private:
    void multiplyBlocks(const std::pair<int, int>& blockA, 
                       const std::pair<int, int>& blockB,
                       const optimized_hashmatrix<T>& other,
                       std::vector<T>& result) const {
        bool isDenseA = denseBlocks.count(blockA);
        bool isDenseB = other.denseBlocks.count(blockB);

        if (isDenseA && isDenseB) {
            // Dense * Dense multiplication
            const auto& matA = denseBlocks.at(blockA).data;
            const auto& matB = other.denseBlocks.at(blockB).data;
            
            for (size_t i = 0; i < BLOCK_SIZE; i++) {
                for (size_t j = 0; j < BLOCK_SIZE; j++) {
                    T sum = 0;
                    for (size_t k = 0; k < BLOCK_SIZE; k++) {
                        sum += matA[i][k] * matB[k][j];
                    }
                    result[i * BLOCK_SIZE + j] += sum;
                }
            }
        } else {
            // Handle sparse multiplication
            // ... (implement sparse multiplication logic)
        }
    }

    // Additional helper methods for sparse multiplication
    void multiplySparseBlocks(const std::pair<int, int>& blockA,
                            const std::pair<int, int>& blockB,
                            const optimized_hashmatrix<T>& other,
                            std::vector<T>& result) const {
        int baseRowA = blockA.first * BLOCK_SIZE;
        int baseColA = blockA.second * BLOCK_SIZE;
        int baseRowB = blockB.first * BLOCK_SIZE;
        int baseColB = blockB.second * BLOCK_SIZE;

        // Collect non-zero elements in both blocks
        std::vector<std::tuple<size_t, size_t, T>> nonZerosA, nonZerosB;
        
        for (size_t i = 0; i < BLOCK_SIZE; i++) {
            for (size_t j = 0; j < BLOCK_SIZE; j++) {
                T valA = get(baseRowA + i, baseColA + j);
                if (valA != 0) {
                    nonZerosA.emplace_back(i, j, valA);
                }
                
                T valB = other.get(baseRowB + i, baseColB + j);
                if (valB != 0) {
                    nonZerosB.emplace_back(i, j, valB);
                }
            }
        }

        // Multiply non-zero elements
        for (const auto& [i, k, valA] : nonZerosA) {
            for (const auto& [k2, j, valB] : nonZerosB) {
                if (k == k2) {
                    result[i * BLOCK_SIZE + j] += valA * valB;
                }
            }
        }
    }
};

#endif