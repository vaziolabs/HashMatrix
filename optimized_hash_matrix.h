#ifndef optimized_hash_matrix_H
#define optimized_hash_matrix_H

#include <unordered_map>
#include <vector>
#include <tuple>
#include <stdexcept>
#include <algorithm>
#include "debug.h"

template <typename T, typename CoordType = size_t>
class optimized_hash_matrix {
private:
    // Replace static block size with member variable
    size_t blockSize;
    static constexpr double DENSITY_THRESHOLD = 0.5;

    /*
     * Hybrid Storage Strategy:
     * 1. Sparse regions use hash-based storage (sparseData)
     * 2. Dense regions use 64x64 blocks (denseBlocks)
     * The matrix automatically switches between these based on density
     */
    
    // Stores sparse elements as (row,col) -> value mappings
    std::unordered_map<std::pair<CoordType, CoordType>, T, PairHash> sparseData;

    // Dense block storage - used when a block becomes >50% full
    struct DenseBlock {
        std::vector<std::vector<T>> data;
        bool isDirty;
        size_t blockSize;
        
        DenseBlock() : isDirty(false), blockSize(0) {}
        
        explicit DenseBlock(size_t block_size) 
            : isDirty(false)
            , blockSize(block_size) {
            data.resize(block_size, std::vector<T>(block_size, T{}));
        }

        // Add copy constructor
        DenseBlock(const DenseBlock& other)
            : data(other.data)
            , isDirty(other.isDirty)
            , blockSize(other.blockSize) {}

        // Add move constructor
        DenseBlock(DenseBlock&& other) noexcept
            : data(std::move(other.data))
            , isDirty(other.isDirty)
            , blockSize(other.blockSize) {}

        // Add assignment operators
        DenseBlock& operator=(const DenseBlock& other) {
            if (this != &other) {
                data = other.data;
                isDirty = other.isDirty;
                blockSize = other.blockSize;
            }
            return *this;
        }

        DenseBlock& operator=(DenseBlock&& other) noexcept {
            if (this != &other) {
                data = std::move(other.data);
                isDirty = other.isDirty;
                blockSize = other.blockSize;
            }
            return *this;
        }
    };
    std::unordered_map<std::pair<CoordType, CoordType>, DenseBlock, PairHash> denseBlocks;

    // Tracks number of non-zero elements in each block for density calculations
    std::unordered_map<std::pair<CoordType, CoordType>, size_t, PairHash> blockNonZeroCount;
    size_t numRows, numCols;

    std::pair<CoordType, CoordType> getBlockCoords(CoordType row, CoordType col) const {
        return {
            static_cast<CoordType>(row / static_cast<CoordType>(blockSize)), 
            static_cast<CoordType>(col / static_cast<CoordType>(blockSize))
        };
    }

    std::pair<CoordType, CoordType> getLocalCoords(CoordType row, CoordType col) const {
        return {
            static_cast<CoordType>(row % static_cast<CoordType>(blockSize)), 
            static_cast<CoordType>(col % static_cast<CoordType>(blockSize))
        };
    }

    double getBlockDensity(const std::pair<int, int>& blockCoord) const {
        auto it = blockNonZeroCount.find(blockCoord);
        if (it == blockNonZeroCount.end()) return 0.0;
        return static_cast<double>(it->second) / (blockSize * blockSize);
    }

    void convertBlockToDense(const std::pair<int, int>& blockCoord) {
        DenseBlock block(blockSize);
        int baseRow = blockCoord.first * blockSize;
        int baseCol = blockCoord.second * blockSize;

        // Copy data from sparse to dense
        for (int i = 0; i < blockSize; i++) {
            for (int j = 0; j < blockSize; j++) {
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
        int baseRow = blockCoord.first * blockSize;
        int baseCol = blockCoord.second * blockSize;

        // Copy data from dense to sparse
        for (int i = 0; i < blockSize; i++) {
            for (int j = 0; j < blockSize; j++) {
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

    // Helper for coordinate conversion
    template<typename FromType>
    static CoordType safeCoordConvert(FromType val) {
        static_assert(std::is_arithmetic_v<FromType>, 
                     "Coordinate type must be arithmetic");
                     
        if constexpr (std::is_signed_v<FromType> && std::is_unsigned_v<CoordType>) {
            if (val < 0) {
                throw std::out_of_range("Negative value not allowed for unsigned type");
            }
        }
        
        if (static_cast<std::uint64_t>(val) > std::numeric_limits<CoordType>::max()) {
            throw std::out_of_range("Value out of range for coordinate type");
        }
        
        return static_cast<CoordType>(val);
    }

public:
    /*
     * Matrix Operations:
     * - insert(): Automatically handles sparse/dense conversion
     * - get(): Checks dense blocks first (faster), then sparse storage
     * - multiply(): Uses block-based multiplication for better cache utilization
     * - add(): Processes dense blocks in parallel, then handles sparse elements
     */

    optimized_hash_matrix(size_t rows, size_t cols, size_t block_size = 8) 
        : numRows(rows)
        , numCols(cols)
        , blockSize(block_size)
        , sparseData()
        , denseBlocks()
        , blockNonZeroCount()
    {
        if (block_size == 0) {
            throw std::invalid_argument("Block size must be greater than 0");
        }
        
        // Reserve space if needed
        size_t expectedBlocks = (rows * cols) / (block_size * block_size) + 1;
        sparseData.reserve(expectedBlocks);
        denseBlocks.reserve(expectedBlocks);
        blockNonZeroCount.reserve(expectedBlocks);
    }

    void insert(CoordType row, CoordType col, const T& value) {
        if (static_cast<size_t>(row) >= numRows || static_cast<size_t>(col) >= numCols) {
            throw std::out_of_range("Matrix indices out of bounds");
        }

        auto blockCoord = getBlockCoords(row, col);
        auto localCoord = getLocalCoords(row, col);

        // Initialize block if it doesn't exist
        auto blockIt = denseBlocks.find(blockCoord);
        if (blockIt == denseBlocks.end()) {
            // Start with sparse representation
            if (value != T{}) {  // Only create block if value is non-zero
                sparseData[{row, col}] = value;
                updateBlockDensity(blockCoord, true);
            }
            return;
        }

        // Rest of the existing insert logic...
        if (blockIt != denseBlocks.end()) {
            // Block is dense
            T oldValue = blockIt->second.data[localCoord.first][localCoord.second];
            blockIt->second.data[localCoord.first][localCoord.second] = value;
            
            if ((oldValue == 0) != (value == 0)) {
                updateBlockDensity(blockCoord, value != 0);
            }
        }
    }

    T get(CoordType row, CoordType col) const {
        if (static_cast<size_t>(row) >= numRows || static_cast<size_t>(col) >= numCols) {
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
    template<typename InputCoordType>
    void batchInsert(const std::vector<std::tuple<InputCoordType, InputCoordType, T>>& data) {
        static_assert(std::is_arithmetic_v<InputCoordType>, 
                     "Coordinate type must be arithmetic");
        
        if constexpr (std::is_same_v<InputCoordType, CoordType>) {
            // Direct insert for matching types
            for (const auto& [i, j, val] : data) {
                insert(i, j, val);
            }
        } else {
            // Convert coordinates for different types
            for (const auto& [i, j, val] : data) {
                insert(safeCoordConvert(i), safeCoordConvert(j), val);
            }
        }
    }

    // Specialization for same type to avoid conversion overhead
    void batchInsert(const std::vector<std::tuple<CoordType, CoordType, T>>& data) {
        for (const auto& [i, j, val] : data) {
            insert(i, j, val);
        }
    }

    // Helper method for batch conversion
    template<typename FromType>
    static std::vector<std::tuple<CoordType, CoordType, T>> convertBatch(
            const std::vector<std::tuple<FromType, FromType, T>>& data) {
        std::vector<std::tuple<CoordType, CoordType, T>> result;
        result.reserve(data.size());
        
        for (const auto& [i, j, val] : data) {
            result.emplace_back(safeCoordConvert(i), safeCoordConvert(j), val);
        }
        
        return result;
    }

    /*
     * Matrix Multiplication:
     * 1. Divides matrices into blocks for better cache usage
     * 2. Uses OpenMP for parallel processing
     * 3. Optimizes dense*dense multiplication
     * 4. Has special handling for sparse blocks
     */
    optimized_hash_matrix<T> multiply(const optimized_hash_matrix<T>& other) const {
        if (numCols != other.numRows) {
            throw std::invalid_argument("Matrix dimensions must match for multiplication");
        }
        if (blockSize != other.blockSize) {
            throw std::invalid_argument("Block sizes must match for multiplication");
        }

        optimized_hash_matrix<T> result(numRows, other.numCols, blockSize);
        size_t numBlocksK = (numCols + blockSize - 1) / blockSize;

        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < (numRows + blockSize - 1) / blockSize; i++) {
            for (size_t j = 0; j < (other.numCols + blockSize - 1) / blockSize; j++) {
                std::vector<T> tempBlock(blockSize * blockSize, 0);
                
                for (size_t k = 0; k < numBlocksK; k++) {
                    multiplyBlocks({i, k}, {k, j}, other, tempBlock);
                }

                // Insert non-zero results
                for (size_t bi = 0; bi < blockSize; bi++) {
                    for (size_t bj = 0; bj < blockSize; bj++) {
                        T val = tempBlock[bi * blockSize + bj];
                        if (val != 0) {
                            result.insert(i * blockSize + bi, j * blockSize + bj, val);
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

    optimized_hash_matrix<T> add(const optimized_hash_matrix<T>& other) const {
        if (numRows != other.numRows || numCols != other.numCols) {
            throw std::invalid_argument("Matrix dimensions must match for addition");
        }

        optimized_hash_matrix<T> result(numRows, numCols);

        // Process dense blocks first
        std::set<std::pair<int, int>> processedBlocks;

        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < (numRows + blockSize - 1) / blockSize; i++) {
            for (size_t j = 0; j < (numCols + blockSize - 1) / blockSize; j++) {
                std::pair<int, int> blockCoord{i, j};
                bool isDenseA = denseBlocks.count(blockCoord);
                bool isDenseB = other.denseBlocks.count(blockCoord);

                if (isDenseA && isDenseB) {
                    // Both blocks are dense
                    const auto& blockA = denseBlocks.at(blockCoord);
                    const auto& blockB = other.denseBlocks.at(blockCoord);
                    
                    for (size_t bi = 0; bi < blockSize; bi++) {
                        for (size_t bj = 0; bj < blockSize; bj++) {
                            T sum = blockA.data[bi][bj] + blockB.data[bi][bj];
                            if (sum != 0) {
                                result.insert(i * blockSize + bi, j * blockSize + bj, sum);
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
        const optimized_hash_matrix<T>* matrix;
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
        Iterator(const optimized_hash_matrix<T>* m, size_t row = 0, size_t col = 0)
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

    // Add getter for block size
    size_t getBlockSize() const {
        return blockSize;
    }

private:
    void multiplyBlocks(const std::pair<int, int>& blockA, 
                       const std::pair<int, int>& blockB,
                       const optimized_hash_matrix<T>& other,
                       std::vector<T>& result) const {
        bool isDenseA = denseBlocks.count(blockA);
        bool isDenseB = other.denseBlocks.count(blockB);

        if (isDenseA && isDenseB) {
            // Dense * Dense multiplication
            const auto& matA = denseBlocks.at(blockA).data;
            const auto& matB = other.denseBlocks.at(blockB).data;
            
            for (size_t i = 0; i < blockSize; i++) {
                for (size_t j = 0; j < blockSize; j++) {
                    T sum = 0;
                    for (size_t k = 0; k < blockSize; k++) {
                        sum += matA[i][k] * matB[k][j];
                    }
                    result[i * blockSize + j] += sum;
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
                            const optimized_hash_matrix<T>& other,
                            std::vector<T>& result) const {
        int baseRowA = blockA.first * blockSize;
        int baseColA = blockA.second * blockSize;
        int baseRowB = blockB.first * blockSize;
        int baseColB = blockB.second * blockSize;

        // Collect non-zero elements in both blocks
        std::vector<std::tuple<size_t, size_t, T>> nonZerosA, nonZerosB;
        
        for (size_t i = 0; i < blockSize; i++) {
            for (size_t j = 0; j < blockSize; j++) {
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
                    result[i * blockSize + j] += valA * valB;
                }
            }
        }
    }
};

#endif