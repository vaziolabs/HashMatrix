#ifndef HASHMATRIX_H
#define HASHMATRIX_H

#include <iostream>
#include <unordered_map>
#include <vector>
#include <array>
#include <tuple>
#include <memory>
#include <immintrin.h>  // For SIMD operations
#include <omp.h>
#include <cpuid.h>

// Custom hash for matrix coordinates
struct PairHash {
    template<typename T1, typename T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

template <typename T>
class hash_matrix {
private:
    static constexpr size_t BLOCK_SIZE = 64;
    static constexpr double ULTRA_SPARSE_THRESHOLD = 0.1;
    static constexpr double SPARSE_THRESHOLD = 0.5;
    static constexpr double DENSE_THRESHOLD = 0.9;

    // Block type classification for optimized operations
    enum class BlockType {
        ULTRA_SPARSE,  // < 10% density
        SPARSE,        // < 50% density
        DENSE,         // >= 50% density
        FULL          // > 90% density
    };

    // Metadata for each block
    struct BlockMetadata {
        BlockType type;
        size_t nonZeroCount;
        size_t dataIndex;  // Index into appropriate storage
        double density;

        BlockMetadata() : type(BlockType::ULTRA_SPARSE), nonZeroCount(0), dataIndex(0), density(0.0) {}
    };

    // Efficient memory pool for matrix elements
    class MemoryPool {
    private:
        static constexpr size_t POOL_BLOCK_SIZE = 1024 * 1024; // 1MB blocks
        static constexpr size_t CACHE_LINE_SIZE = 64;
        
        struct alignas(CACHE_LINE_SIZE) MemoryBlock {
            char data[POOL_BLOCK_SIZE];
            size_t used;
            MemoryBlock() : used(0) {}
        };
        
        std::vector<std::unique_ptr<MemoryBlock>> blocks;
        std::vector<std::pair<void*, size_t>> freeList; // ptr, size pairs

    public:
        MemoryPool() = default;
        
        // Delete copy constructor and assignment
        MemoryPool(const MemoryPool&) = delete;
        MemoryPool& operator=(const MemoryPool&) = delete;
        
        // Add move constructor and assignment
        MemoryPool(MemoryPool&&) = default;
        MemoryPool& operator=(MemoryPool&&) = default;

        template<typename U>
        U* allocate(size_t count) {
            size_t size = count * sizeof(U);
            size_t alignment = alignof(U);
            
            // Check free list first
            for (auto it = freeList.begin(); it != freeList.end(); ++it) {
                if (it->second >= size) {
                    void* ptr = it->first;
                    if (it->second == size) {
                        freeList.erase(it);
                    } else {
                        it->first = static_cast<char*>(it->first) + size;
                        it->second -= size;
                    }
                    return static_cast<U*>(ptr);
                }
            }

            // Allocate new block if needed
            for (auto& block : blocks) {
                size_t aligned_offset = (block->used + alignment - 1) & ~(alignment - 1);
                if (aligned_offset + size <= POOL_BLOCK_SIZE) {
                    block->used = aligned_offset + size;
                    return reinterpret_cast<U*>(block->data + aligned_offset);
                }
            }

            // Create new block
            auto newBlock = std::make_unique<MemoryBlock>();
            size_t aligned_offset = (alignment - 1) & ~(alignment - 1);
            newBlock->used = aligned_offset + size;
            U* result = reinterpret_cast<U*>(newBlock->data + aligned_offset);
            blocks.push_back(std::move(newBlock));
            return result;
        }

        void deallocate(void* ptr, size_t size) {
            freeList.emplace_back(ptr, size);
            // Optional: Coalesce adjacent free blocks
            if (freeList.size() > 1) {
                std::sort(freeList.begin(), freeList.end());
                for (size_t i = 1; i < freeList.size(); ) {
                    auto& prev = freeList[i-1];
                    auto& curr = freeList[i];
                    if (static_cast<char*>(prev.first) + prev.second == curr.first) {
                        prev.second += curr.second;
                        freeList.erase(freeList.begin() + i);
                    } else {
                        ++i;
                    }
                }
            }
        }

        ~MemoryPool() {
            // Clear all allocated blocks
            blocks.clear();
            freeList.clear();
        }

        void clear() {
            // Reset all blocks
            for (auto& block : blocks) {
                block->used = 0;
            }
            freeList.clear();
        }

        // Add memory statistics
        size_t getAllocatedSize() const {
            return blocks.size() * POOL_BLOCK_SIZE;
        }

        size_t getUsedSize() const {
            size_t used = 0;
            for (const auto& block : blocks) {
                used += block->used;
            }
            return used;
        }

        template<typename U>
        U& get(size_t index) {
            if (index >= blocks.size()) {
                throw std::out_of_range("Invalid block index");
            }
            return *reinterpret_cast<U*>(blocks[index]->data);
        }

        // Specify array type explicitly for dense blocks
        std::array<T, BLOCK_SIZE * BLOCK_SIZE>& get(size_t index) {
            if (index >= blocks.size()) {
                throw std::out_of_range("Invalid block index");
            }
            return *reinterpret_cast<std::array<T, BLOCK_SIZE * BLOCK_SIZE>*>(blocks[index]->data);
        }

        const std::array<T, BLOCK_SIZE * BLOCK_SIZE>& get(size_t index) const {
            if (index >= blocks.size()) {
                throw std::out_of_range("Invalid block index");
            }
            return *reinterpret_cast<const std::array<T, BLOCK_SIZE * BLOCK_SIZE>*>(blocks[index]->data);
        }
    };

    // Main storage components
    MemoryPool densePool;
    std::vector<std::unordered_map<std::pair<int, int>, T, PairHash>> sparseStorage;
    std::unordered_map<std::pair<int, int>, BlockMetadata, PairHash> blockIndex;
    size_t numRows, numCols;

    // Fast entry lookup cache
    struct MatrixEntry {
        T value;
        bool isDense;
        size_t blockIndex;
    };
    std::unordered_map<std::pair<int, int>, MatrixEntry, PairHash> fastLookup;

    // Helper methods
    std::pair<int, int> getBlockCoords(size_t row, size_t col) const {
        return {row / BLOCK_SIZE, col / BLOCK_SIZE};
    }

    size_t getLocalOffset(size_t row, size_t col) const {
        return (row % BLOCK_SIZE) * BLOCK_SIZE + (col % BLOCK_SIZE);
    }

    void updateDensity(BlockMetadata& metadata) {
        metadata.density = static_cast<double>(metadata.nonZeroCount) / (BLOCK_SIZE * BLOCK_SIZE);
        
        // Update block type based on density
        if (metadata.density < ULTRA_SPARSE_THRESHOLD) {
            metadata.type = BlockType::ULTRA_SPARSE;
        } else if (metadata.density < SPARSE_THRESHOLD) {
            metadata.type = BlockType::SPARSE;
        } else if (metadata.density < DENSE_THRESHOLD) {
            metadata.type = BlockType::DENSE;
        } else {
            metadata.type = BlockType::FULL;
        }
    }

    // Add these new helper methods for matrix operations
    enum class MultiplyStrategy {
        ULTRA_SPARSE_SPARSE,
        SPARSE_SPARSE,
        DENSE_DENSE,
        HYBRID
    };

    MultiplyStrategy getOptimalMultiplyStrategy(BlockType a, BlockType b) const {
        if (a == BlockType::ULTRA_SPARSE || b == BlockType::ULTRA_SPARSE) {
            return MultiplyStrategy::ULTRA_SPARSE_SPARSE;
        }
        if (a >= BlockType::DENSE && b >= BlockType::DENSE) {
            return MultiplyStrategy::DENSE_DENSE;
        }
        if (a <= BlockType::SPARSE && b <= BlockType::SPARSE) {
            return MultiplyStrategy::SPARSE_SPARSE;
        }
        return MultiplyStrategy::HYBRID;
    }

    // CPU Feature Detection
    struct CPUFeatures {
        static bool hasAVX() {
            unsigned int eax, ebx, ecx, edx;
            if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
                return (ecx & (1 << 28)) != 0;  // AVX bit
            }
            return false;
        }

        static bool hasFMA() {
            unsigned int eax, ebx, ecx, edx;
            if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
                return (ecx & (1 << 12)) != 0;  // FMA bit
            }
            return false;
        }
    };

    // SIMD multiplication helper
    void multiplyDenseSIMD(size_t blockIdxA, size_t blockIdxB, 
                          std::array<T, BLOCK_SIZE * BLOCK_SIZE>& result) const {
        const auto& blockA = densePool.get(blockIdxA);
        const auto& blockB = densePool.get(blockIdxB);

        if (CPUFeatures::hasAVX() && CPUFeatures::hasFMA()) {
            // AVX path
            #pragma omp parallel for collapse(2)
            for (size_t i = 0; i < BLOCK_SIZE; i++) {
                for (size_t j = 0; j < BLOCK_SIZE; j += 4) {
                    __m256d sum = _mm256_setzero_pd();
                    
                    for (size_t k = 0; k < BLOCK_SIZE; k++) {
                        __m256d a = _mm256_set1_pd(blockA[i * BLOCK_SIZE + k]);
                        __m256d b = _mm256_load_pd(&blockB[k * BLOCK_SIZE + j]);
                        sum = _mm256_fmadd_pd(a, b, sum);
                    }
                    
                    _mm256_store_pd(&result[i * BLOCK_SIZE + j], sum);
                }
            }
        } else {
            // Fallback path without SIMD
            #pragma omp parallel for collapse(2)
            for (size_t i = 0; i < BLOCK_SIZE; i++) {
                for (size_t j = 0; j < BLOCK_SIZE; j++) {
                    T sum = 0;
                    for (size_t k = 0; k < BLOCK_SIZE; k++) {
                        sum += blockA[i * BLOCK_SIZE + k] * blockB[k * BLOCK_SIZE + j];
                    }
                    result[i * BLOCK_SIZE + j] = sum;
                }
            }
        }
    }

    // Optimized sparse block multiplication
    void multiplySparse(const std::pair<int, int>& coordA, 
                       const std::pair<int, int>& coordB,
                       hash_matrix<T>& result) const {
        const auto& sparseA = sparseStorage[blockIndex.at(coordA).dataIndex];
        const auto& sparseB = sparseStorage[blockIndex.at(coordB).dataIndex];

        // Use sorted vectors for better cache performance
        std::vector<std::tuple<size_t, size_t, T>> sortedA, sortedB;
        for (const auto& [pos, val] : sparseA) {
            sortedA.emplace_back(pos.first, pos.second, val);
        }
        for (const auto& [pos, val] : sparseB) {
            sortedB.emplace_back(pos.first, pos.second, val);
        }
        std::sort(sortedA.begin(), sortedA.end());
        std::sort(sortedB.begin(), sortedB.end());

        // Multiply sorted sparse blocks
        for (const auto& [rowA, colA, valA] : sortedA) {
            for (const auto& [rowB, colB, valB] : sortedB) {
                if (colA == rowB) {
                    result.insert(rowA, colB, valA * valB);
                }
            }
        }
    }

    // Add cache alignment and prefetching
    static constexpr size_t CACHE_LINE_SIZE = 64;
    
    // Aligned storage for better cache performance
    struct alignas(CACHE_LINE_SIZE) DenseBlock {
        std::array<T, BLOCK_SIZE * BLOCK_SIZE> data;
    };

    // Prefetch helper
    void prefetchBlock(const std::pair<int, int>& blockCoord) {
        auto it = blockIndex.find(blockCoord);
        if (it != blockIndex.end()) {
            __builtin_prefetch(&densePool.get(it->second.dataIndex));
        }
    }

    void multiplyHybrid(const std::pair<int, int>& coordA, 
                        const std::pair<int, int>& coordB,
                        hash_matrix<T>& result) const {
        const auto& metadataA = blockIndex.at(coordA);
        const auto& metadataB = blockIndex.at(coordB);
        
        if (metadataA.type >= BlockType::DENSE) {
            const auto& denseBlock = densePool.get(metadataA.dataIndex);
            const auto& sparseBlock = sparseStorage[metadataB.dataIndex];
            
            if (CPUFeatures::hasAVX()) {
                // AVX path
                std::vector<std::pair<size_t, T>> sortedSparse;
                for (const auto& [pos, val] : sparseBlock) {
                    sortedSparse.emplace_back(pos.second, val);
                }
                std::sort(sortedSparse.begin(), sortedSparse.end());
                
                // Use SIMD for dense rows
                alignas(32) T temp[BLOCK_SIZE];
                for (size_t i = 0; i < BLOCK_SIZE; i++) {
                    __m256d sum = _mm256_setzero_pd();
                    size_t sparseIdx = 0;
                    
                    while (sparseIdx < sortedSparse.size()) {
                        size_t k = sortedSparse[sparseIdx].first;
                        T val = sortedSparse[sparseIdx].second;
                        
                        __m256d sparse = _mm256_set1_pd(val);
                        __m256d dense = _mm256_load_pd(&denseBlock[i * BLOCK_SIZE + k]);
                        sum = _mm256_fmadd_pd(sparse, dense, sum);
                        
                        sparseIdx++;
                    }
                    
                    _mm256_store_pd(temp, sum);
                    T finalSum = temp[0] + temp[1] + temp[2] + temp[3];
                    if (finalSum != 0) {
                        result.insert(coordA.first * BLOCK_SIZE + i, 
                                    coordB.second * BLOCK_SIZE, 
                                    finalSum);
                    }
                }
            } else {
                // Fallback path
                for (const auto& [pos, val] : sparseBlock) {
                    for (size_t i = 0; i < BLOCK_SIZE; i++) {
                        T product = denseBlock[i * BLOCK_SIZE + pos.first] * val;
                        if (product != 0) {
                            result.insert(coordA.first * BLOCK_SIZE + i,
                                       coordB.second * BLOCK_SIZE + pos.second,
                                       product);
                        }
                    }
                }
            }
        } else {
            // Sparse-Dense multiplication
            const auto& sparseBlock = sparseStorage[metadataA.dataIndex];
            const auto& denseBlock = densePool.get(metadataB.dataIndex);
            
            // Convert sparse block to sorted format for better cache performance
            std::vector<std::tuple<size_t, size_t, T>> sortedSparse;
            for (const auto& [pos, val] : sparseBlock) {
                sortedSparse.emplace_back(pos.first, pos.second, val);
            }
            std::sort(sortedSparse.begin(), sortedSparse.end());
            
            // Use SIMD for dense columns
            alignas(32) T temp[BLOCK_SIZE];
            for (const auto& [rowA, colA, valA] : sortedSparse) {
                __m256d sparse_val = _mm256_set1_pd(valA);
                
                for (size_t j = 0; j < BLOCK_SIZE; j += 4) {
                    __m256d dense = _mm256_load_pd(&denseBlock[colA * BLOCK_SIZE + j]);
                    __m256d prod = _mm256_mul_pd(sparse_val, dense);
                    _mm256_store_pd(temp, prod);
                    
                    for (size_t k = 0; k < 4; k++) {
                        if (temp[k] != 0) {
                            result.insert(coordA.first * BLOCK_SIZE + rowA,
                                        coordB.second * BLOCK_SIZE + j + k,
                                        temp[k]);
                        }
                    }
                }
            }
        }
    }

    void coalesceBlocks() {
        std::vector<std::pair<int, int>> candidates;
        
        // Find adjacent blocks that could be merged
        for (const auto& [coord, metadata] : blockIndex) {
            if (metadata.type == BlockType::SPARSE) {
                auto nextCoord = std::make_pair(coord.first, coord.second + 1);
                auto nextBlock = blockIndex.find(nextCoord);
                
                if (nextBlock != blockIndex.end() && 
                    nextBlock->second.type == BlockType::SPARSE) {
                    double combinedDensity = 
                        (metadata.nonZeroCount + nextBlock->second.nonZeroCount) / 
                        (2.0 * BLOCK_SIZE * BLOCK_SIZE);
                    
                    if (combinedDensity >= SPARSE_THRESHOLD) {
                        candidates.push_back(coord);
                    }
                }
            }
        }

        // Merge candidate blocks
        for (const auto& coord : candidates) {
            auto nextCoord = std::make_pair(coord.first, coord.second + 1);
            mergeBlocks(coord, nextCoord);
        }
    }

    void mergeBlocks(const std::pair<int, int>& coord1, 
                     const std::pair<int, int>& coord2) {
        auto& block1 = blockIndex[coord1];
        auto& block2 = blockIndex[coord2];
        
        // Allocate new dense block
        size_t newBlockIdx = densePool.allocate();
        auto& newBlock = densePool.get(newBlockIdx);
        
        // Copy data from sparse blocks to dense block
        for (const auto& [pos, val] : sparseStorage[block1.dataIndex]) {
            size_t localRow = pos.first % BLOCK_SIZE;
            size_t localCol = pos.second % BLOCK_SIZE;
            newBlock[localRow * BLOCK_SIZE + localCol] = val;
        }
        
        for (const auto& [pos, val] : sparseStorage[block2.dataIndex]) {
            size_t localRow = pos.first % BLOCK_SIZE;
            size_t localCol = pos.second % BLOCK_SIZE;
            newBlock[localRow * BLOCK_SIZE + localCol] = val;
        }
        
        // Update metadata
        block1.type = BlockType::DENSE;
        block1.dataIndex = newBlockIdx;
        blockIndex.erase(coord2);
    }

private:
    void batchInsertImpl(const std::vector<std::tuple<size_t, size_t, T>>& values) {
        // Original implementation
        std::unordered_map<std::pair<int, int>, 
                          std::vector<std::tuple<size_t, size_t, T>>, 
                          PairHash> blockGroups;
        // ... rest of implementation ...
    }

public:
    void batchInsert(const std::vector<std::tuple<int, int, T>>& values) {
        std::vector<std::tuple<size_t, size_t, T>> converted;
        converted.reserve(values.size());
        
        for (const auto& [row, col, val] : values) {
            converted.emplace_back(static_cast<size_t>(row), 
                                 static_cast<size_t>(col), 
                                 val);
        }
        
        batchInsertImpl(converted);
    }

    void remove(size_t row, size_t col) {
        insert(row, col, T{});
    }

    hash_matrix(size_t rows, size_t cols) 
        : numRows(rows)
        , numCols(cols) {
        size_t numBlocks = ((rows + BLOCK_SIZE - 1) / BLOCK_SIZE) * 
                          ((cols + BLOCK_SIZE - 1) / BLOCK_SIZE);
        sparseStorage.resize(numBlocks);
    }

    // Efficient iterator implementation
    class Iterator {
    public:
        // Type definitions must come before member variables
        using value_type = std::tuple<size_t, size_t, T>;
        using reference = const value_type&;
        using pointer = const value_type*;
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;

    private:
        const hash_matrix* matrix;
        size_t currentBlockRow;
        size_t currentBlockCol;
        size_t localRow;
        size_t localCol;
        size_t currentDenseBlock;
        std::vector<std::pair<int, int>> denseBlocks;
        value_type current_value;
        std::pair<size_t, size_t> currentPos;

    public:
        // Fix constructor
        Iterator(const hash_matrix* m) 
            : matrix(m), currentBlockRow(0), currentBlockCol(0),
              localRow(0), localCol(0), currentDenseBlock(0) {
            // Pre-cache dense blocks
            for (const auto& [coord, metadata] : m->blockIndex) {
                if (metadata.type >= BlockType::DENSE) {
                    denseBlocks.push_back(coord);
                }
            }
            findNextNonZero();
        }

        void findNextNonZero() {
            // First check dense blocks (SIMD-friendly)
            while (currentDenseBlock < denseBlocks.size()) {
                const auto& blockCoord = denseBlocks[currentDenseBlock];
                const auto& metadata = matrix->blockIndex.at(blockCoord);
                const auto& block = matrix->densePool.get(metadata.dataIndex);
                
                // Prefetch next block
                if (currentDenseBlock + 1 < denseBlocks.size()) {
                    matrix->prefetchBlock(denseBlocks[currentDenseBlock + 1]);
                }

                // SIMD scan for non-zero elements
                for (size_t i = localRow; i < BLOCK_SIZE; i++) {
                    __m256d zero = _mm256_setzero_pd();
                    for (size_t j = localCol; j < BLOCK_SIZE; j += 4) {
                        __m256d val = _mm256_load_pd(&block[i * BLOCK_SIZE + j]);
                        __m256d mask = _mm256_cmp_pd(val, zero, _CMP_NEQ_OQ);
                        unsigned int bits = _mm256_movemask_pd(mask);
                        if (bits) {
                            localRow = i;
                            localCol = j + __builtin_ctz(bits);
                            return;
                        }
                    }
                }
                currentDenseBlock++;
                localRow = localCol = 0;
            }

            // Then check sparse blocks
            for (const auto& [coord, metadata] : matrix->blockIndex) {
                if (metadata.type >= BlockType::DENSE) continue;
                
                const auto& sparseBlock = matrix->sparseStorage[metadata.dataIndex];
                for (const auto& [pos, val] : sparseBlock) {
                    if (pos.first > currentPos.first || 
                        (pos.first == currentPos.first && pos.second > currentPos.second)) {
                        currentPos = pos;
                        return;
                    }
                }
            }
        }

        bool operator==(const Iterator& other) const {
            return matrix == other.matrix && 
                   currentBlockRow == other.currentBlockRow &&
                   currentBlockCol == other.currentBlockCol &&
                   localRow == other.localRow &&
                   localCol == other.localCol;
        }

        Iterator& operator++() {
            localCol++;
            if (localCol >= BLOCK_SIZE) {
                localCol = 0;
                localRow++;
                if (localRow >= BLOCK_SIZE) {
                    localRow = 0;
                    findNextNonZero();
                }
            }
            return *this;
        }

        reference operator*() const {
            size_t globalRow = currentBlockRow * BLOCK_SIZE + localRow;
            size_t globalCol = currentBlockCol * BLOCK_SIZE + localCol;
            return current_value = std::make_tuple(globalRow, globalCol, 
                matrix->get(globalRow, globalCol));
        }

        pointer operator->() const {
            return &(operator*());
        }

        // Add constructor overload for end iterator
        Iterator(const hash_matrix* m, size_t maxRow, size_t maxCol) 
            : matrix(m), currentBlockRow(maxRow / BLOCK_SIZE), 
              currentBlockCol(maxCol / BLOCK_SIZE),
              localRow(maxRow % BLOCK_SIZE), localCol(maxCol % BLOCK_SIZE),
              currentDenseBlock(std::numeric_limits<size_t>::max()) {}
    };

    Iterator begin() const { return Iterator(this); }
    Iterator end() const { return Iterator(this, numRows * numCols, 0); }

    // Core operations with SIMD support
    void insert(size_t row, size_t col, T value) {
        auto blockCoord = getBlockCoords(row, col);
        auto& metadata = blockIndex[blockCoord];
        
        if (metadata.type == BlockType::DENSE || metadata.type == BlockType::FULL) {
            size_t offset = getLocalOffset(row, col);
            auto& block = densePool.get(metadata.dataIndex);
            bool wasZero = block[offset] == 0;
            block[offset] = value;
            
            if (wasZero && value != 0) metadata.nonZeroCount++;
            else if (!wasZero && value == 0) metadata.nonZeroCount--;
        } else {
            auto& sparseBlock = sparseStorage[metadata.dataIndex];
            if (value != 0) {
                auto [it, inserted] = sparseBlock.insert({{row, col}, value});
                if (inserted) metadata.nonZeroCount++;
            } else {
                if (sparseBlock.erase({row, col}) > 0) metadata.nonZeroCount--;
            }
        }

        updateDensity(metadata);
        fastLookup[{row, col}] = {value, metadata.type >= BlockType::DENSE, metadata.dataIndex};
    }

    T get(size_t row, size_t col) const {
        // Try fast lookup first
        auto it = fastLookup.find({row, col});
        if (it != fastLookup.end()) {
            return it->second.value;
        }

        // Fall back to regular lookup
        auto blockCoord = getBlockCoords(row, col);
        auto metaIt = blockIndex.find(blockCoord);
        if (metaIt == blockIndex.end()) return T{};

        const auto& metadata = metaIt->second;
        if (metadata.type >= BlockType::DENSE) {
            size_t offset = getLocalOffset(row, col);
            return densePool.get(metadata.dataIndex)[offset];
        } else {
            const auto& sparseBlock = sparseStorage[metadata.dataIndex];
            auto it = sparseBlock.find({row, col});
            return it != sparseBlock.end() ? it->second : T{};
        }
    }

    // Optimized matrix multiplication
    hash_matrix<T> multiply(const hash_matrix<T>& other) const {
        if (numCols != other.numRows) {
            throw std::invalid_argument("Matrix dimensions must match for multiplication");
        }

        hash_matrix<T> result(numRows, other.numCols);
        
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < (numRows + BLOCK_SIZE - 1) / BLOCK_SIZE; i++) {
            for (size_t j = 0; j < (other.numCols + BLOCK_SIZE - 1) / BLOCK_SIZE; j++) {
                std::array<T, BLOCK_SIZE * BLOCK_SIZE> tempResult{};
                
                for (size_t k = 0; k < (numCols + BLOCK_SIZE - 1) / BLOCK_SIZE; k++) {
                    std::pair<int, int> blockA{i, k}, blockB{k, j};
                    
                    auto strategyA = blockIndex.find(blockA);
                    auto strategyB = other.blockIndex.find(blockB);
                    
                    if (strategyA == blockIndex.end() || strategyB == other.blockIndex.end()) {
                        continue;  // Skip empty blocks
                    }

                    switch (getOptimalMultiplyStrategy(strategyA->second.type, 
                                                     strategyB->second.type)) {
                        case MultiplyStrategy::DENSE_DENSE:
                            multiplyDenseSIMD(strategyA->second.dataIndex,
                                            strategyB->second.dataIndex,
                                            tempResult);
                            break;
                        case MultiplyStrategy::SPARSE_SPARSE:
                        case MultiplyStrategy::ULTRA_SPARSE_SPARSE:
                            multiplySparse(blockA, blockB, result);
                            break;
                        case MultiplyStrategy::HYBRID:
                            multiplyHybrid(blockA, blockB, result);
                            break;
                    }
                }

                // Insert non-zero results
                for (size_t bi = 0; bi < BLOCK_SIZE; bi++) {
                    for (size_t bj = 0; bj < BLOCK_SIZE; bj++) {
                        T val = tempResult[bi * BLOCK_SIZE + bj];
                        if (val != 0) {
                            result.insert(i * BLOCK_SIZE + bi, j * BLOCK_SIZE + bj, val);
                        }
                    }
                }
            }
        }

        return result;
    }

    // Optimized matrix addition
    hash_matrix<T> add(const hash_matrix<T>& other) const {
        if (numRows != other.numRows || numCols != other.numCols) {
            throw std::invalid_argument("Matrix dimensions must match for addition");
        }

        hash_matrix<T> result(numRows, numCols);
        std::vector<std::pair<int, int>> processedBlocks;

        // Process dense blocks first using SIMD
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < (numRows + BLOCK_SIZE - 1) / BLOCK_SIZE; i++) {
            for (size_t j = 0; j < (numCols + BLOCK_SIZE - 1) / BLOCK_SIZE; j++) {
                std::pair<int, int> blockCoord{i, j};
                auto blockA = blockIndex.find(blockCoord);
                auto blockB = other.blockIndex.find(blockCoord);

                if (blockA != blockIndex.end() && blockB != other.blockIndex.end() &&
                    blockA->second.type >= BlockType::DENSE && 
                    blockB->second.type >= BlockType::DENSE) {
                    
                    const auto& dataA = densePool.get(blockA->second.dataIndex);
                    const auto& dataB = other.densePool.get(blockB->second.dataIndex);
                    
                    #pragma omp simd
                    for (size_t k = 0; k < BLOCK_SIZE * BLOCK_SIZE; k++) {
                        T sum = dataA[k] + dataB[k];
                        if (sum != 0) {
                            size_t localRow = k / BLOCK_SIZE;
                            size_t localCol = k % BLOCK_SIZE;
                            result.insert(i * BLOCK_SIZE + localRow, 
                                        j * BLOCK_SIZE + localCol, 
                                        sum);
                        }
                    }
                    
                    #pragma omp critical
                    processedBlocks.push_back(blockCoord);
                }
            }
        }

        // Process remaining sparse blocks
        for (const auto& [coord, metadata] : blockIndex) {
            if (std::find(processedBlocks.begin(), processedBlocks.end(), coord) != 
                processedBlocks.end()) {
                continue;
            }

            // Handle sparse addition
            for (size_t i = 0; i < BLOCK_SIZE; i++) {
                for (size_t j = 0; j < BLOCK_SIZE; j++) {
                    size_t globalRow = coord.first * BLOCK_SIZE + i;
                    size_t globalCol = coord.second * BLOCK_SIZE + j;
                    
                    if (globalRow >= numRows || globalCol >= numCols) continue;
                    
                    T sum = get(globalRow, globalCol) + other.get(globalRow, globalCol);
                    if (sum != 0) {
                        result.insert(globalRow, globalCol, sum);
                    }
                }
            }
        }

        return result;
    }

    // Add batch operations for better performance
    void batchInsert(const std::vector<std::tuple<size_t, size_t, T>>& values) {
        // Group insertions by block
        std::unordered_map<std::pair<int, int>, 
                          std::vector<std::tuple<size_t, size_t, T>>, 
                          PairHash> blockGroups;

        for (const auto& [row, col, val] : values) {
            auto blockCoord = getBlockCoords(row, col);
            blockGroups[blockCoord].emplace_back(row % BLOCK_SIZE, 
                                               col % BLOCK_SIZE, 
                                               val);
        }

        // Process each block
        #pragma omp parallel for
        for (const auto& [blockCoord, blockValues] : blockGroups) {
            size_t nonZeroCount = std::count_if(blockValues.begin(), 
                                              blockValues.end(),
                                              [](const auto& t) { 
                                                  return std::get<2>(t) != 0; 
                                              });

            double density = static_cast<double>(nonZeroCount) / (BLOCK_SIZE * BLOCK_SIZE);
            auto& metadata = blockIndex[blockCoord];
            
            if (density >= SPARSE_THRESHOLD) {
                // Use dense storage
                size_t blockIdx = metadata.dataIndex;
                if (metadata.type < BlockType::DENSE) {
                    blockIdx = densePool.allocate();
                    metadata.dataIndex = blockIdx;
                }
                
                auto& block = densePool.get(blockIdx);
                for (const auto& [i, j, val] : blockValues) {
                    block[i * BLOCK_SIZE + j] = val;
                }
                
                metadata.type = BlockType::DENSE;
            } else {
                // Use sparse storage
                for (const auto& [i, j, val] : blockValues) {
                    if (val != 0) {
                        sparseStorage[metadata.dataIndex][{blockCoord.first * BLOCK_SIZE + i, 
                                                         blockCoord.second * BLOCK_SIZE + j}] = val;
                    }
                }
            }
            
            metadata.nonZeroCount = nonZeroCount;
            updateDensity(metadata);
        }
    }
};

#endif // HASHMATRIX_H