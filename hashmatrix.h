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
#include <algorithm>  // for std::sort, std::count_if
#include <utility>    // for std::pair

/**
 * @brief Custom hash function for matrix coordinates
 * 
 * Implements efficient hashing for pair of coordinates using XOR operation
 * to generate well-distributed hash values for sparse matrix storage.
 */
struct PairHash {
    template<typename T1, typename T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

/**
 * @brief Hybrid sparse-dense matrix implementation with block-based storage
 * 
 * Implements a matrix that automatically switches between sparse and dense
 * storage formats at the block level based on element density. Uses SIMD
 * operations and OpenMP for performance optimization.
 * 
 * @tparam T The data type of matrix elements (typically numeric types)
 */
template <typename T>
class hash_matrix {
private:
    /** @brief Size of matrix blocks for storage optimization */
    static constexpr size_t BLOCK_SIZE = 64;
    
    /** @brief Density threshold for ultra-sparse storage */
    static constexpr double ULTRA_SPARSE_THRESHOLD = 0.1;
    
    /** @brief Density threshold for sparse storage */
    static constexpr double SPARSE_THRESHOLD = 0.5;
    
    /** @brief Density threshold for dense storage */
    static constexpr double DENSE_THRESHOLD = 0.9;

    /**
     * @brief Enumeration of block storage types based on density
     * 
     * Used to optimize storage and operations based on element density within blocks
     */
    enum class BlockType {
        ULTRA_SPARSE,  // < 10% density
        SPARSE,        // < 50% density
        DENSE,         // >= 50% density
        FULL          // > 90% density
    };

    /**
     * @brief Metadata for matrix blocks
     * 
     * Stores information about block storage type, element count, and storage location
     */
    struct BlockMetadata {
        BlockType type;           ///< Storage format type
        size_t nonZeroCount;     ///< Number of non-zero elements
        size_t dataIndex;        ///< Index into appropriate storage
        double density;          ///< Ratio of non-zero elements

        BlockMetadata() : type(BlockType::ULTRA_SPARSE), nonZeroCount(0), dataIndex(0), density(0.0) {}
    };

    /**
     * @brief Custom memory pool for efficient matrix element storage
     * 
     * Implements a block-based memory allocator with alignment and caching
     * optimizations for matrix operations
     */
    class MemoryPool {
    private:
        /** @brief Size of memory blocks in bytes */
        static constexpr size_t POOL_BLOCK_SIZE = 1024 * 1024; // 1MB blocks
        
        /** @brief Cache line size for alignment */
        static constexpr size_t CACHE_LINE_SIZE = 64;
        
        /**
         * @brief Memory block structure with cache line alignment
         */
        struct alignas(CACHE_LINE_SIZE) MemoryBlock {
            char data[POOL_BLOCK_SIZE];  ///< Raw storage
            size_t used;                 ///< Bytes allocated
            MemoryBlock() : used(0) {}
        };
        
        std::vector<std::unique_ptr<MemoryBlock>> blocks;
        std::vector<std::pair<void*, size_t>> freeList; // ptr, size pairs

    public:
        /**
         * @brief Default constructor for MemoryPool
         */
        MemoryPool() = default;
        
        // Delete copy operations to prevent accidental sharing of memory resources
        MemoryPool(const MemoryPool&) = delete;
        MemoryPool& operator=(const MemoryPool&) = delete;
        
        // Enable move semantics for efficient resource transfer
        MemoryPool(MemoryPool&&) = default;
        MemoryPool& operator=(MemoryPool&&) = default;

        /**
         * @brief Allocates memory for a specified number of elements
         * 
         * Implements a sophisticated allocation strategy that:
         * 1. Checks the free list for available space
         * 2. Reuses existing blocks when possible
         * 3. Creates new blocks when necessary
         * 4. Maintains proper alignment requirements
         * 
         * @tparam U Type of elements to allocate
         * @param count Number of elements to allocate
         * @return Pointer to allocated memory
         */
        template<typename U>
        U* allocate(size_t count) {
            size_t size = count * sizeof(U);
            size_t alignment = alignof(U);
            
            // Try to find suitable space in free list
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

            // Search existing blocks for available space
            for (auto& block : blocks) {
                size_t aligned_offset = (block->used + alignment - 1) & ~(alignment - 1);
                if (aligned_offset + size <= POOL_BLOCK_SIZE) {
                    block->used = aligned_offset + size;
                    return reinterpret_cast<U*>(block->data + aligned_offset);
                }
            }

            // Allocate new block when necessary
            auto newBlock = std::make_unique<MemoryBlock>();
            size_t aligned_offset = (alignment - 1) & ~(alignment - 1);
            newBlock->used = aligned_offset + size;
            U* result = reinterpret_cast<U*>(newBlock->data + aligned_offset);
            blocks.push_back(std::move(newBlock));
            return result;
        }

        /**
         * @brief Deallocates previously allocated memory
         * 
         * Adds the memory back to the free list and performs coalescing
         * of adjacent free blocks when possible.
         * 
         * @param ptr Pointer to memory to deallocate
         * @param size Size of memory block in bytes
         */
        void deallocate(void* ptr, size_t size) {
            freeList.emplace_back(ptr, size);
            
            // Coalesce adjacent free blocks for better memory utilization
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

        /**
         * @brief Destructor that ensures proper cleanup of memory resources
         */
        ~MemoryPool() {
            blocks.clear();
            freeList.clear();
        }

        /**
         * @brief Resets the memory pool to its initial state
         * 
         * Maintains allocated blocks but marks them as unused
         */
        void clear() {
            for (auto& block : blocks) {
                block->used = 0;
            }
            freeList.clear();
        }

        /**
         * @brief Returns total allocated size in bytes
         */
        size_t getAllocatedSize() const {
            return blocks.size() * POOL_BLOCK_SIZE;
        }

        /**
         * @brief Returns total used size in bytes
         */
        size_t getUsedSize() const {
            size_t used = 0;
            for (const auto& block : blocks) {
                used += block->used;
            }
            return used;
        }

        /**
         * @brief Retrieves a reference to a stored element
         * 
         * @tparam U Type of element to retrieve
         * @param index Block index to access
         * @return Reference to the stored element
         * @throws std::out_of_range if index is invalid
         */
        template<typename U>
        U& get(size_t index) {
            if (index >= blocks.size()) {
                throw std::out_of_range("Invalid block index");
            }
            return *reinterpret_cast<U*>(blocks[index]->data);
        }

        /**
         * @brief Retrieves a reference to a dense block array
         * 
         * Specialized version for accessing dense matrix blocks
         * 
         * @param index Block index to access
         * @return Reference to the dense block array
         * @throws std::out_of_range if index is invalid
         */
        std::array<T, BLOCK_SIZE * BLOCK_SIZE>& get(size_t index) {
            if (index >= blocks.size()) {
                throw std::out_of_range("Invalid block index");
            }
            return *reinterpret_cast<std::array<T, BLOCK_SIZE * BLOCK_SIZE>*>(blocks[index]->data);
        }

        /**
         * @brief Retrieves a const reference to a dense block array
         * 
         * Const version for read-only access to dense matrix blocks
         * 
         * @param index Block index to access
         * @return Const reference to the dense block array
         * @throws std::out_of_range if index is invalid
         */
        const std::array<T, BLOCK_SIZE * BLOCK_SIZE>& get(size_t index) const {
            if (index >= blocks.size()) {
                throw std::out_of_range("Invalid block index");
            }
            return *reinterpret_cast<const std::array<T, BLOCK_SIZE * BLOCK_SIZE>*>(blocks[index]->data);
        }
    };

    /**
     * @brief Main storage components for the matrix
     */
    MemoryPool densePool;                    ///< Pool for dense block storage
    std::vector<std::unordered_map<std::pair<int, int>, T, PairHash>> sparseStorage;  ///< Storage for sparse blocks
    std::unordered_map<std::pair<int, int>, BlockMetadata, PairHash> blockIndex;      ///< Metadata for all blocks
    size_t numRows;                          ///< Total number of matrix rows
    size_t numCols;                          ///< Total number of matrix columns

    /**
     * @brief Cache structure for fast element lookup
     */
    struct MatrixEntry {
        T value;              ///< Stored value
        bool isDense;         ///< Whether the value is in dense storage
        size_t blockIndex;    ///< Index of the containing block
    };
    std::unordered_map<std::pair<int, int>, MatrixEntry, PairHash> fastLookup;  ///< Cache for recent accesses

    /**
     * @brief Helper method to convert global coordinates to block coordinates
     * 
     * @param row Global row index
     * @param col Global column index
     * @return Pair of block coordinates
     */
    std::pair<int, int> getBlockCoords(size_t row, size_t col) const {
        return {row / BLOCK_SIZE, col / BLOCK_SIZE};
    }

    /**
     * @brief Helper method to convert global coordinates to local block offset
     * 
     * @param row Global row index
     * @param col Global column index
     * @return Local offset within the block
     */
    size_t getLocalOffset(size_t row, size_t col) const {
        return (row % BLOCK_SIZE) * BLOCK_SIZE + (col % BLOCK_SIZE);
    }

    /**
     * @brief Updates block metadata based on current element density
     * 
     * Recalculates density and updates storage type classification
     * 
     * @param metadata Reference to block metadata to update
     */
    void updateDensity(BlockMetadata& metadata) {
        metadata.density = static_cast<double>(metadata.nonZeroCount) / (BLOCK_SIZE * BLOCK_SIZE);
        
        // Update block type based on density thresholds
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

    /**
     * @brief Enumeration of matrix multiplication strategies
     * 
     * Defines different approaches for block multiplication based on storage types
     */
    enum class MultiplyStrategy {
        ULTRA_SPARSE_SPARSE,  ///< Optimized for ultra-sparse * sparse blocks
        SPARSE_SPARSE,        ///< Optimized for sparse * sparse blocks
        DENSE_DENSE,          ///< SIMD-optimized for dense * dense blocks
        HYBRID               ///< Mixed-mode multiplication strategy
    };

    /**
     * @brief Determines the optimal multiplication strategy for two blocks
     * 
     * @param a Type of first block
     * @param b Type of second block
     * @return Optimal multiplication strategy
     */
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

    /**
     * @brief CPU feature detection utilities
     * 
     * Provides methods to check for available CPU features like AVX and FMA
     */
    struct CPUFeatures {
        /**
         * @brief Checks if AVX instructions are available
         * 
         * @return true if AVX is supported, false otherwise
         */
        static bool hasAVX() {
            unsigned int eax, ebx, ecx, edx;
            if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
                return (ecx & (1 << 28)) != 0;  // AVX bit
            }
            return false;
        }

        /**
         * @brief Checks if FMA instructions are available
         * 
         * @return true if FMA is supported, false otherwise
         */
        static bool hasFMA() {
            unsigned int eax, ebx, ecx, edx;
            if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
                return (ecx & (1 << 12)) != 0;  // FMA bit
            }
            return false;
        }
    };

    /**
     * @brief Performs SIMD-optimized multiplication of dense blocks
     * 
     * Uses AVX and FMA instructions when available, with fallback to scalar
     * multiplication. Includes OpenMP parallelization for larger blocks.
     * 
     * @param blockIdxA Index of first block in dense storage
     * @param blockIdxB Index of second block in dense storage
     * @param result Array to store multiplication result
     */
    void multiplyDenseSIMD(size_t blockIdxA, size_t blockIdxB, 
                          std::array<T, BLOCK_SIZE * BLOCK_SIZE>& result) const {
        const auto& blockA = densePool.get(blockIdxA);
        const auto& blockB = densePool.get(blockIdxB);

        if (CPUFeatures::hasAVX() && CPUFeatures::hasFMA()) {
            // AVX path with FMA optimization
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

    /**
     * @brief Optimized sparse block multiplication implementation
     * 
     * Performs efficient multiplication of sparse blocks by:
     * 1. Converting to sorted vectors for better cache performance
     * 2. Using coordinate-based multiplication
     * 3. Only computing non-zero element combinations
     * 
     * @param coordA Coordinates of first block
     * @param coordB Coordinates of second block
     * @param result Matrix to store multiplication result
     */
    void multiplySparse(const std::pair<int, int>& coordA, 
                       const std::pair<int, int>& coordB,
                       hash_matrix<T>& result) const {
        const auto& sparseA = sparseStorage[blockIndex.at(coordA).dataIndex];
        const auto& sparseB = sparseStorage[blockIndex.at(coordB).dataIndex];

        // Convert to sorted vectors for better cache performance
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

    /**
     * @brief Cache alignment and prefetching optimizations
     */
    static constexpr size_t CACHE_LINE_SIZE = 64;
    
    /**
     * @brief Aligned storage structure for dense blocks
     */
    struct alignas(CACHE_LINE_SIZE) DenseBlock {
        std::array<T, BLOCK_SIZE * BLOCK_SIZE> data;
    };

    /**
     * @brief Prefetch helper for optimizing memory access patterns
     * 
     * @param blockCoord Coordinates of block to prefetch
     */
    void prefetchBlock(const std::pair<int, int>& blockCoord) {
        auto it = blockIndex.find(blockCoord);
        if (it != blockIndex.end()) {
            __builtin_prefetch(&densePool.get(it->second.dataIndex));
        }
    }

    /**
     * @brief Hybrid multiplication implementation for mixed sparse-dense blocks
     * 
     * Implements an optimized multiplication strategy for cases where one block
     * is dense and the other is sparse. Uses SIMD operations when available
     * and includes cache optimization techniques.
     * 
     * @param coordA Coordinates of first block
     * @param coordB Coordinates of second block
     * @param result Matrix to store multiplication result
     */
    void multiplyHybrid(const std::pair<int, int>& coordA, 
                        const std::pair<int, int>& coordB,
                        hash_matrix<T>& result) const {
        const auto& metadataA = blockIndex.at(coordA);
        const auto& metadataB = blockIndex.at(coordB);
        
        if (metadataA.type >= BlockType::DENSE) {
            // Dense-Sparse multiplication path
            const auto& denseBlock = densePool.get(metadataA.dataIndex);
            const auto& sparseBlock = sparseStorage[metadataB.dataIndex];
            
            if (CPUFeatures::hasAVX()) {
                // Optimized AVX path
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
                // Scalar fallback path
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
            // Sparse-Dense multiplication path
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

    /**
     * @brief Merges adjacent blocks to optimize storage and computation
     * 
     * Analyzes block density patterns and merges adjacent blocks when beneficial.
     * This helps reduce fragmentation and improve computation efficiency.
     */
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

    /**
     * @brief Merges two adjacent blocks into a single block
     * 
     * Combines the data from two blocks into a single block, optimizing
     * storage format based on the combined density.
     * 
     * @param coord1 Coordinates of first block
     * @param coord2 Coordinates of second block
     */
    void mergeBlocks(const std::pair<int, int>& coord1, 
                     const std::pair<int, int>& coord2) {
        auto& block1 = blockIndex[coord1];
        auto& block2 = blockIndex[coord2];
        
        // Allocate new block with template keyword
        size_t newBlockIdx = densePool.template allocate<std::array<T, BLOCK_SIZE * BLOCK_SIZE>>(1);
        auto& newBlock = densePool.get(newBlockIdx);
        
        // Copy data from original blocks
        const auto& block1Data = densePool.get(blockIndex[coord1].dataIndex);
        const auto& block2Data = densePool.get(blockIndex[coord2].dataIndex);
        
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
        
        // Update density information
        updateDensity(block1);
    }

    /**
     * @brief Implements batch insertion of multiple values into the matrix
     * 
     * This private method handles the efficient insertion of multiple values by:
     * 1. Grouping insertions by their target blocks to minimize storage format changes
     * 2. Determining optimal storage format (dense/sparse) based on resulting density
     * 3. Performing bulk updates to the chosen storage format
     * 4. Maintaining metadata and cache consistency
     * 
     * @param values Vector of tuples containing (row, column, value) to be inserted
     */
    void batchInsertImpl(const std::vector<std::tuple<size_t, size_t, T>>& values) {
        // Create a mapping of block coordinates to their pending insertions
        // This groups operations by block to minimize storage transitions
        std::unordered_map<std::pair<int, int>, 
                          std::vector<std::tuple<size_t, size_t, T>>, 
                          PairHash> blockGroups;

        // Group all insertions by their target block coordinates
        // Convert global coordinates to block-local coordinates during grouping
        for (const auto& [row, col, val] : values) {
            auto blockCoord = getBlockCoords(row, col);
            blockGroups[blockCoord].emplace_back(row % BLOCK_SIZE, 
                                               col % BLOCK_SIZE, 
                                               val);
        }

        // Process each block's insertions
        for (const auto& [blockCoord, blockValues] : blockGroups) {
            // Count non-zero values to determine optimal storage format
            size_t nonZeroCount = std::count_if(blockValues.begin(), 
                                              blockValues.end(),
                                              [](const auto& t) { 
                                                  return std::get<2>(t) != 0; 
                                              });

            // Calculate density to choose between sparse and dense storage
            double density = static_cast<double>(nonZeroCount) / (BLOCK_SIZE * BLOCK_SIZE);
            auto& metadata = blockIndex[blockCoord];
            
            if (density >= SPARSE_THRESHOLD) {
                // Dense storage is more efficient for this block
                size_t blockIdx;
                
                // Allocate new dense block or reuse existing one
                if (metadata.type < BlockType::DENSE) {
                    blockIdx = densePool.template allocate<std::array<T, BLOCK_SIZE * BLOCK_SIZE>>(1);
                    metadata.dataIndex = blockIdx;
                } else {
                    blockIdx = metadata.dataIndex;
                }
                
                // Initialize dense block and populate with new values
                auto& block = densePool.get(blockIdx);
                std::fill(block.begin(), block.end(), T{});  // Clear existing values
                
                for (const auto& [i, j, val] : blockValues) {
                    block[i * BLOCK_SIZE + j] = val;
                }
                
                metadata.type = BlockType::DENSE;
            } else {
                // Sparse storage is more efficient for this block
                
                // Convert from dense if necessary
                if (metadata.type >= BlockType::DENSE) {
                    sparseStorage[metadata.dataIndex].clear();
                    metadata.type = BlockType::SPARSE;
                }
                
                // Insert non-zero values into sparse storage
                for (const auto& [i, j, val] : blockValues) {
                    if (val != 0) {
                        sparseStorage[metadata.dataIndex][{i, j}] = val;
                    }
                }
            }
            
            // Update block metadata
            metadata.nonZeroCount = nonZeroCount;
            updateDensity(metadata);

            // Update fast lookup cache for all affected positions
            // This ensures O(1) access time for recently inserted values
            for (const auto& [i, j, val] : blockValues) {
                size_t globalRow = blockCoord.first * BLOCK_SIZE + i;
                size_t globalCol = blockCoord.second * BLOCK_SIZE + j;
                fastLookup[{globalRow, globalCol}] = {
                    val, 
                    metadata.type >= BlockType::DENSE, 
                    metadata.dataIndex
                };
            }
        }
    }

public:
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
                    blockIdx = densePool.template allocate<std::array<T, BLOCK_SIZE * BLOCK_SIZE>>(1);
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

    void remove(size_t row, size_t col) {
        insert(row, col, T{});
    }

    /**
     * @brief Constructs a new hash_matrix with specified dimensions
     * 
     * Initializes the matrix storage structures and allocates initial memory.
     * 
     * @param rows Number of rows in the matrix
     * @param cols Number of columns in the matrix
     */
    hash_matrix(size_t rows, size_t cols) 
        : numRows(rows)
        , numCols(cols) {
        size_t numBlocks = ((rows + BLOCK_SIZE - 1) / BLOCK_SIZE) * 
                          ((cols + BLOCK_SIZE - 1) / BLOCK_SIZE);
        sparseStorage.resize(numBlocks);
    }

    /**
     * @brief Iterator class for efficient matrix traversal
     * 
     * Implements a forward iterator that efficiently traverses non-zero elements
     * in the matrix, utilizing SIMD operations for dense blocks and optimized
     * access patterns for sparse blocks.
     */
    class Iterator {
    public:
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
        /**
         * @brief Constructs an iterator for the given matrix
         * 
         * @param m Pointer to the matrix being iterated
         */
        Iterator(const hash_matrix* m) 
            : matrix(m), currentBlockRow(0), currentBlockCol(0),
              localRow(0), localCol(0), currentDenseBlock(0) {
            // Pre-cache dense blocks for efficient traversal
            for (const auto& [coord, metadata] : m->blockIndex) {
                if (metadata.type >= BlockType::DENSE) {
                    denseBlocks.push_back(coord);
                }
            }
            findNextNonZero();
        }

        /**
         * @brief Finds the next non-zero element in the matrix
         * 
         * Uses SIMD operations for dense blocks and optimized traversal
         * for sparse blocks to efficiently locate non-zero elements.
         */
        void findNextNonZero() {
            // First check dense blocks (SIMD-friendly)
            while (currentDenseBlock < denseBlocks.size()) {
                const auto& blockCoord = denseBlocks[currentDenseBlock];
                const auto& metadata = matrix->blockIndex.at(blockCoord);
                const auto& block = matrix->densePool.get(metadata.dataIndex);
                
                // Prefetch next block for better cache utilization
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

        /**
         * @brief Equality comparison operator
         */
        bool operator==(const Iterator& other) const {
            return matrix == other.matrix && 
                   currentBlockRow == other.currentBlockRow &&
                   currentBlockCol == other.currentBlockCol &&
                   localRow == other.localRow &&
                   localCol == other.localCol;
        }

        /**
         * @brief Increment operator
         * 
         * Advances the iterator to the next non-zero element
         */
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

        /**
         * @brief Dereference operator
         * 
         * @return Tuple containing (row, column, value) of current element
         */
        reference operator*() const {
            size_t globalRow = currentBlockRow * BLOCK_SIZE + localRow;
            size_t globalCol = currentBlockCol * BLOCK_SIZE + localCol;
            return current_value = std::make_tuple(globalRow, globalCol, 
                matrix->get(globalRow, globalCol));
        }

        pointer operator->() const {
            return &(operator*());
        }

        /**
         * @brief Constructs an end iterator
         * 
         * @param m Pointer to the matrix
         * @param maxRow Maximum row index
         * @param maxCol Maximum column index
         */
        Iterator(const hash_matrix* m, size_t maxRow, size_t maxCol) 
            : matrix(m), currentBlockRow(maxRow / BLOCK_SIZE), 
              currentBlockCol(maxCol / BLOCK_SIZE),
              localRow(maxRow % BLOCK_SIZE), localCol(maxCol % BLOCK_SIZE),
              currentDenseBlock(std::numeric_limits<size_t>::max()) {}
    };

    /**
     * @brief Returns an iterator to the beginning of the matrix
     */
    Iterator begin() const { return Iterator(this); }

    /**
     * @brief Returns an iterator to the end of the matrix
     */
    Iterator end() const { return Iterator(this, numRows * numCols, 0); }

    /**
     * @brief Inserts or updates a value in the matrix
     * 
     * Automatically manages storage format transitions and updates metadata.
     * Uses cache-optimized access patterns and maintains the fast lookup table.
     * 
     * @param row Row index of the element
     * @param col Column index of the element
     * @param value Value to insert
     */
    void insert(size_t row, size_t col, T value) {
        auto blockCoord = getBlockCoords(row, col);
        auto& metadata = blockIndex[blockCoord];
        
        if (metadata.type == BlockType::DENSE || metadata.type == BlockType::FULL) {
            // Handle dense storage format
            size_t offset = getLocalOffset(row, col);
            auto& block = densePool.get(metadata.dataIndex);
            bool wasZero = block[offset] == 0;
            block[offset] = value;
            
            // Update non-zero count
            if (wasZero && value != 0) metadata.nonZeroCount++;
            else if (!wasZero && value == 0) metadata.nonZeroCount--;
        } else {
            // Handle sparse storage format
            auto& sparseBlock = sparseStorage[metadata.dataIndex];
            if (value != 0) {
                auto [it, inserted] = sparseBlock.insert({{row, col}, value});
                if (inserted) metadata.nonZeroCount++;
            } else {
                if (sparseBlock.erase({row, col}) > 0) metadata.nonZeroCount--;
            }
        }

        // Update metadata and cache
        updateDensity(metadata);
        fastLookup[{row, col}] = {value, metadata.type >= BlockType::DENSE, metadata.dataIndex};
    }

    /**
     * @brief Retrieves a value from the matrix
     * 
     * Uses a multi-level caching strategy:
     * 1. Checks fast lookup cache first
     * 2. Falls back to block storage lookup if needed
     * 3. Returns zero for non-existent elements
     * 
     * @param row Row index of the element
     * @param col Column index of the element
     * @return Value at the specified position
     */
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

    /**
     * @brief Performs matrix multiplication
     * 
     * Implements a highly optimized block-based matrix multiplication that:
     * 1. Uses different strategies based on block types
     * 2. Employs SIMD operations for dense blocks
     * 3. Optimizes sparse block operations
     * 4. Parallelizes computation using OpenMP
     * 
     * @param other Matrix to multiply with
     * @return Result of multiplication
     * @throws std::invalid_argument if dimensions don't match
     */
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

                    // Choose optimal multiplication strategy
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

    /**
     * @brief Optimized matrix addition
     * 
     * Implements block-based matrix addition with:
     * 1. SIMD optimization for dense blocks
     * 2. Efficient sparse block handling
     * 3. OpenMP parallelization
     * 4. Cache-friendly processing order
     * 
     * @param other Matrix to add
     * @return Result of addition
     * @throws std::invalid_argument if dimensions don't match
     */
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
};

#endif // HASHMATRIX_H