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
#include <functional>
#include <cstdint>    // Add for uintmax_t
#include <limits>     // Add for numeric_limits
#include <type_traits>  // Add for type traits
#include <cstring>  // for std::memcpy

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
template <typename T, typename CoordType = size_t>
class hash_matrix {
private:
    size_t numRows;
    size_t numCols;
    size_t blockSize;  // Dynamic block size
    
    // Define SparseBlock type at the beginning
    using SparseBlock = std::unordered_map<std::pair<CoordType, CoordType>, T, PairHash>;

    enum class BlockType {
        EMPTY,
        ULTRA_SPARSE,
        SPARSE,
        DENSE,
        FULL
    };

    // Move BlockMetadata definition before the functions that use it
    struct BlockMetadata {
        size_t dataIndex;      ///< Index into memory pool
        BlockType type;        ///< Storage format type
        size_t nonZeroCount;   ///< Number of non-zero elements
        double density;        ///< Ratio of non-zero elements to total elements
        void* dataPtr;         ///< Direct pointer to block data

        BlockMetadata() : dataIndex(0), type(BlockType::SPARSE), 
                         nonZeroCount(0), density(0.0), dataPtr(nullptr) {}
    };

    std::unordered_map<std::pair<CoordType, CoordType>, BlockMetadata, PairHash> blockIndex;

    std::pair<CoordType, CoordType> getLocalCoords(size_t row, size_t col) const {
        return {static_cast<CoordType>(row % blockSize), 
                static_cast<CoordType>(col % blockSize)};
    }

    // Move coordinate conversion to private section
    struct CoordConverter {
        template<typename InputType>
        static CoordType convert(InputType val) {
            static_assert(std::is_arithmetic_v<InputType>, 
                         "Coordinate type must be arithmetic");
                         
            if constexpr (std::is_signed_v<InputType> && std::is_unsigned_v<CoordType>) {
                if (val < 0) {
                    throw std::out_of_range("Negative value not allowed for unsigned type");
                }
            }
            
            if (static_cast<std::uint64_t>(val) > std::numeric_limits<CoordType>::max()) {
                throw std::out_of_range("Value out of range for coordinate type");
            }
            
            return static_cast<CoordType>(val);
        }
    };

    // Memory management class
    class MemoryPool {
    private:
        struct Block {
            void* ptr;
            size_t size;
            std::function<void(void*)> deleter;
            
            Block(void* p, size_t s, std::function<void(void*)> d)
                : ptr(p), size(s), deleter(std::move(d)) {}
            ~Block() {
                if (ptr && deleter) deleter(ptr);
            }
            
            // Delete copy operations
            Block(const Block&) = delete;
            Block& operator=(const Block&) = delete;
            
            // Allow move operations
            Block(Block&& other) noexcept
                : ptr(other.ptr), size(other.size), deleter(std::move(other.deleter)) {
                other.ptr = nullptr;
            }
            Block& operator=(Block&& other) noexcept {
                if (this != &other) {
                    if (ptr && deleter) deleter(ptr);
                    ptr = other.ptr;
                    size = other.size;
                    deleter = std::move(other.deleter);
                    other.ptr = nullptr;
                }
                return *this;
            }
        };
        
        std::vector<Block> blocks;

    public:
        // Add default constructor
        MemoryPool() = default;

        // Copy constructor
        MemoryPool(const MemoryPool& other) {
            // Deep copy of blocks
            for (const auto& block : other.blocks) {
                if (block.ptr) {
                    void* newPtr = ::operator new(block.size);
                    std::memcpy(newPtr, block.ptr, block.size);
                    blocks.emplace_back(newPtr, block.size, block.deleter);
                }
            }
        }

        // Move constructor
        MemoryPool(MemoryPool&& other) noexcept = default;

        // Add copy assignment operator
        MemoryPool& operator=(const MemoryPool& other) {
            if (this != &other) {
                MemoryPool temp(other);
                std::swap(blocks, temp.blocks);
            }
            return *this;
        }

        // Add move assignment operator
        MemoryPool& operator=(MemoryPool&& other) noexcept = default;

        ~MemoryPool() = default;

        template<typename U>
        U* allocate(size_t count) {
            U* ptr = new U[count];
            blocks.emplace_back(ptr, count * sizeof(U),
                [](void* p) { delete[] static_cast<U*>(p); });
            return ptr;
        }

        void* allocateRaw(size_t bytes, std::function<void(void*)> deleter) {
            void* ptr = ::operator new(bytes);
            blocks.emplace_back(ptr, bytes, std::move(deleter));
            return ptr;
        }
    };

    // Now these functions can use BlockMetadata
    void validateBlockAccess(const BlockMetadata& metadata) const {
        if (!metadata.dataPtr) {
            throw std::runtime_error("Attempting to access null block pointer");
        }
    }

    T* getBlockData(BlockMetadata& metadata) {
        validateBlockAccess(metadata);
        return static_cast<T*>(metadata.dataPtr);
    }

    const T* getBlockData(const BlockMetadata& metadata) const {
        validateBlockAccess(metadata);
        return static_cast<const T*>(metadata.dataPtr);
    }

    void convertBlockToDense(const std::pair<CoordType, CoordType>& blockCoord) {
        auto& block = blockIndex[blockCoord];
        if (block.type != BlockType::SPARSE) return;

        auto* sparseBlock = static_cast<SparseBlock*>(block.dataPtr);
        T* denseBlock = new T[blockSize * blockSize]();  // Initialize with zeros

        // Copy data from sparse to dense
        for (const auto& entry : *sparseBlock) {
            size_t offset = entry.first.first * blockSize + entry.first.second;
            denseBlock[offset] = entry.second;
        }

        // Clean up old sparse block and update metadata
        delete sparseBlock;
        block.dataPtr = denseBlock;
        block.type = BlockType::DENSE;
    }

public:  // Make public interface clear
    void insert(CoordType row, CoordType col, const T& value) {
        checkBounds(row, col);
        if (value == T{}) return;

        auto blockCoord = getBlockCoords(row, col);
        auto localCoord = getLocalCoords(row, col);
        
        auto& block = blockIndex[blockCoord];
        if (block.type == BlockType::EMPTY) {
            // Initialize new block
            block.type = BlockType::SPARSE;
            block.dataPtr = new SparseBlock();
        }

        if (block.type == BlockType::SPARSE) {
            auto* sparseBlock = static_cast<SparseBlock*>(block.dataPtr);
            std::pair<CoordType, CoordType> key(localCoord.first, localCoord.second);
            sparseBlock->insert({key, value});
            
            // Convert to dense if threshold reached
            if (sparseBlock->size() > (blockSize * blockSize) / 2) {
                convertBlockToDense(blockCoord);
            }
        } else {
            auto* denseData = static_cast<T*>(block.dataPtr);
            denseData[localCoord.first * blockSize + localCoord.second] = value;
        }
    }

    T get(CoordType row, CoordType col) const {
        checkBounds(row, col);
        
        const auto& metadata = getBlockMetadata(row, col);
        if (metadata.type == BlockType::DENSE) {
            auto* block = static_cast<T*>(metadata.dataPtr);
            return block[getLocalOffset(row, col)];
        }
        auto* sparseBlock = static_cast<const SparseBlock*>(metadata.dataPtr);
        if (!sparseBlock) return T{};
        
        auto it = sparseBlock->find({row % blockSize, col % blockSize});
        return it != sparseBlock->end() ? it->second : T{};
    }

private:
    /** @brief Density threshold for ultra-sparse storage */
    static constexpr double ULTRA_SPARSE_THRESHOLD = 0.1;
    
    /** @brief Density threshold for sparse storage */
    static constexpr double SPARSE_THRESHOLD = 0.5;
    
    /** @brief Density threshold for dense storage */
    static constexpr double DENSE_THRESHOLD = 0.9;

    MemoryPool memoryPool;
    std::vector<std::unordered_map<std::pair<CoordType, CoordType>, T, PairHash>> sparseStorage;  ///< Storage for sparse blocks

    /**
     * @brief Cache structure for fast element lookup
     */
    struct MatrixEntry {
        T value;
        bool isDense;
        void* blockPtr;
        
        MatrixEntry(T v = T{}, bool dense = false, void* ptr = nullptr)
            : value(v), isDense(dense), blockPtr(ptr) {}
    };
    std::unordered_map<std::pair<CoordType, CoordType>, MatrixEntry, PairHash> fastLookup;  ///< Cache for recent accesses

    /**
     * @brief Helper method to convert global coordinates to block coordinates
     * 
     * @param row Global row index
     * @param col Global column index
     * @return Pair of block coordinates
     */
    std::pair<CoordType, CoordType> getBlockCoords(size_t row, size_t col) const {
        return {static_cast<CoordType>(row / blockSize), 
                static_cast<CoordType>(col / blockSize)};
    }

    /**
     * @brief Helper method to convert global coordinates to local block offset
     * 
     * @param row Global row index
     * @param col Global column index
     * @return Local offset within the block
     */
    size_t getLocalOffset(size_t row, size_t col) const {
        return (row % blockSize) * blockSize + (col % blockSize);
    }

    /**
     * @brief Updates block metadata based on current element density
     * 
     * Recalculates density and updates storage type classification
     * 
     * @param metadata Reference to block metadata to update
     */
    void updateDensity(BlockMetadata& metadata) {
        metadata.density = static_cast<double>(metadata.nonZeroCount) / (blockSize * blockSize);
        
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
    void multiplyDenseSIMD(const void* blockPtrA, const void* blockPtrB, T* result) const {
        const auto* blockA = static_cast<const T*>(blockPtrA);
        const auto* blockB = static_cast<const T*>(blockPtrB);
        
        #ifdef __AVX2__
        // AVX2 optimized path
        for (size_t i = 0; i < blockSize; i++) {
            for (size_t k = 0; k < blockSize; k++) {
                __m256d a = _mm256_set1_pd(blockA[i * blockSize + k]);
                for (size_t j = 0; j < blockSize; j += 4) {
                    __m256d b = _mm256_loadu_pd(&blockB[k * blockSize + j]);
                    __m256d c = _mm256_loadu_pd(&result[i * blockSize + j]);
                    __m256d prod = _mm256_mul_pd(a, b);
                    c = _mm256_add_pd(c, prod);
                    _mm256_storeu_pd(&result[i * blockSize + j], c);
                }
            }
        }
        #else
        // Fallback scalar path
        for (size_t i = 0; i < blockSize; i++) {
            for (size_t k = 0; k < blockSize; k++) {
                const T aik = blockA[i * blockSize + k];
                for (size_t j = 0; j < blockSize; j++) {
                    result[i * blockSize + j] += aik * blockB[k * blockSize + j];
                }
            }
        }
        #endif
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
    void multiplySparse(const std::pair<CoordType, CoordType>& coordA,
                       const std::pair<CoordType, CoordType>& coordB,
                       hash_matrix<T>& result) const {
        auto blockA = blockIndex.find(coordA);
        auto blockB = blockIndex.find(coordB);

        if (blockA == blockIndex.end() || blockB == blockIndex.end()) {
            return;  // One of the blocks is empty, no contribution to result
        }

        // Get the sparse blocks
        const auto* sparseBlockA = static_cast<const SparseBlock*>(blockA->second.dataPtr);
        const auto* sparseBlockB = static_cast<const SparseBlock*>(blockB->second.dataPtr);

        // For each non-zero element in block A
        for (const auto& [posA, valA] : *sparseBlockA) {
            // For each non-zero element in block B
            for (const auto& [posB, valB] : *sparseBlockB) {
                if (posA.second == posB.first) {  // If inner dimensions match
                    // Calculate global positions
                    size_t globalRow = coordA.first * blockSize + posA.first;
                    size_t globalCol = coordB.second * blockSize + posB.second;
                    
                    // Add the product to the result
                    // Note: Using += instead of = to accumulate products
                    result.insert(globalRow, globalCol, 
                                result.get(globalRow, globalCol) + valA * valB);
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
        std::vector<T> data;
        
        explicit DenseBlock(size_t size) : data(size) {}
    };

    /**
     * @brief Prefetch helper for optimizing memory access patterns
     * 
     * @param blockCoord Coordinates of block to prefetch
     */
    void prefetchBlock(const std::pair<CoordType, CoordType>& blockCoord) {
        auto it = blockIndex.find(blockCoord);
        if (it != blockIndex.end()) {
            __builtin_prefetch(&memoryPool.get(it->second.dataIndex));
        }
    }

    /**
     * @brief Merges adjacent blocks to optimize storage and computation
     * 
     * Analyzes block density patterns and merges adjacent blocks when beneficial.
     * This helps reduce fragmentation and improve computation efficiency.
     */
    void coalesceBlocks() {
        std::vector<std::pair<CoordType, CoordType>> candidates;
        
        // Find adjacent blocks that could be merged
        for (const auto& [coord, metadata] : blockIndex) {
            if (metadata.type == BlockType::SPARSE) {
                auto nextCoord = std::make_pair(coord.first, coord.second + 1);
                auto nextBlock = blockIndex.find(nextCoord);
                
                if (nextBlock != blockIndex.end() && 
                    nextBlock->second.type == BlockType::SPARSE) {
                    double combinedDensity = 
                        (metadata.nonZeroCount + nextBlock->second.nonZeroCount) / 
                        (2.0 * blockSize * blockSize);
                    
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
    void mergeBlocks(const std::pair<CoordType, CoordType>& coord1, 
                     const std::pair<CoordType, CoordType>& coord2) {
        auto& block1 = blockIndex[coord1];
        auto& block2 = blockIndex[coord2];
        
        // Allocate new block
        T* newBlock = new T[blockSize * blockSize]();
        
        // Copy data from original blocks
        const auto& block1Data = memoryPool.get(blockIndex[coord1].dataIndex);
        const auto& block2Data = memoryPool.get(blockIndex[coord2].dataIndex);
        
        // Copy data from sparse blocks to dense block
        for (const auto& [pos, val] : sparseStorage[block1.dataIndex]) {
            size_t localRow = pos.first % blockSize;
            size_t localCol = pos.second % blockSize;
            newBlock[localRow * blockSize + localCol] = val;
        }
        
        for (const auto& [pos, val] : sparseStorage[block2.dataIndex]) {
            size_t localRow = pos.first % blockSize;
            size_t localCol = pos.second % blockSize;
            newBlock[localRow * blockSize + localCol] = val;
        }
        
        // Update metadata
        block1.type = BlockType::DENSE;
        block1.dataIndex = memoryPool.getLastIndex();
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
        std::unordered_map<std::pair<CoordType, CoordType>, 
                          std::vector<std::tuple<size_t, size_t, T>>, 
                          PairHash> blockGroups;

        // Group all insertions by their target block coordinates
        // Convert global coordinates to block-local coordinates during grouping
        for (const auto& [row, col, val] : values) {
            auto blockCoord = getBlockCoords(row, col);
            blockGroups[blockCoord].emplace_back(row % blockSize, 
                                               col % blockSize, 
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
            double density = static_cast<double>(nonZeroCount) / (blockSize * blockSize);
            auto& metadata = blockIndex[blockCoord];
            
            // Properly allocate and track memory
            if (!metadata.dataPtr) {
                if (density >= SPARSE_THRESHOLD) {
                    metadata.dataPtr = memoryPool.template allocate<T>(blockSize * blockSize);
                    metadata.type = BlockType::DENSE;
                } else {
                    metadata.dataPtr = memoryPool.template allocate<SparseBlock>(1);
                    metadata.type = BlockType::SPARSE;
                    new (metadata.dataPtr) SparseBlock();
                }
                metadata.dataIndex = memoryPool.getLastIndex();
            }
            
            // Update block metadata
            metadata.nonZeroCount = nonZeroCount;
            updateDensity(metadata);

            // Update fast lookup cache for all affected positions
            // This ensures O(1) access time for recently inserted values
            for (const auto& [i, j, val] : blockValues) {
                size_t globalRow = blockCoord.first * blockSize + i;
                size_t globalCol = blockCoord.second * blockSize + j;
                fastLookup[{globalRow, globalCol}] = MatrixEntry(
                    val, 
                    metadata.type >= BlockType::DENSE, 
                    metadata.dataIndex
                );
            }
        }
    }

    BlockMetadata& getBlockMetadata(CoordType row, CoordType col) {
        std::pair<CoordType, CoordType> blockCoord = {
            row / blockSize, 
            col / blockSize
        };
        
        auto& metadata = blockIndex[blockCoord];
        if (!metadata.dataPtr) {
            if (metadata.type == BlockType::DENSE) {
                metadata.dataPtr = memoryPool.template allocate<T>(blockSize * blockSize);
                std::fill_n(static_cast<T*>(metadata.dataPtr), blockSize * blockSize, T{});
            } else {
                metadata.dataPtr = memoryPool.template allocate<SparseBlock>(1);
                new (metadata.dataPtr) SparseBlock();
            }
        }
        return metadata;
    }

    const BlockMetadata& getBlockMetadata(CoordType row, CoordType col) const {
        std::pair<CoordType, CoordType> blockCoord = {
            row / blockSize, 
            col / blockSize
        };
        
        auto it = blockIndex.find(blockCoord);
        if (it == blockIndex.end()) {
            static const BlockMetadata emptyBlock;
            return emptyBlock;
        }
        return it->second;
    }

    // Add type conversion helper for coordinate types
    template<typename FromType>
    static std::vector<std::tuple<CoordType, CoordType, T>> convertData(
            const std::vector<std::tuple<FromType, FromType, T>>& data) {
        std::vector<std::tuple<CoordType, CoordType, T>> result;
        result.reserve(data.size());
        for (const auto& [i, j, val] : data) {
            result.emplace_back(static_cast<CoordType>(i), 
                              static_cast<CoordType>(j), 
                              val);
        }
        return result;
    }

    void checkBounds(CoordType row, CoordType col) const {
        if (row >= numRows || col >= numCols) {
            throw std::out_of_range("Matrix index out of bounds");
        }
    }

    void multiplyBlock(const std::pair<CoordType, CoordType>& coordA,
                     const std::pair<CoordType, CoordType>& coordB,
                     hash_matrix<T>& result) const {
        const auto blockA = blockIndex.find(coordA);
        const auto blockB = blockIndex.find(coordB);
        
        if (blockA == blockIndex.end() || blockB == blockIndex.end()) {
            return;
        }

        const auto& metadataA = blockA->second;
        const auto& metadataB = blockB->second;

        // Choose multiplication strategy based on block types
        if (metadataA.type == BlockType::DENSE && metadataB.type == BlockType::DENSE) {
            multiplyDenseDense(metadataA, metadataB, coordA, coordB, result);
        } else if (metadataA.type == BlockType::SPARSE && metadataB.type == BlockType::SPARSE) {
            multiplySparse(coordA, coordB, result);
        } else {
            multiplyHybrid(coordA, coordB, result);
        }
    }

    void multiplyDenseDense(const BlockMetadata& blockA,
                           const BlockMetadata& blockB,
                           const std::pair<CoordType, CoordType>& coordA,
                           const std::pair<CoordType, CoordType>& coordB,
                           hash_matrix<T>& result) const {
        const T* dataA = static_cast<const T*>(blockA.dataPtr);
        const T* dataB = static_cast<const T*>(blockB.dataPtr);
        std::vector<T> accumulator(blockSize * blockSize, T{});

        for (size_t i = 0; i < blockSize; i++) {
            for (size_t k = 0; k < blockSize; k++) {
                const T aik = dataA[i * blockSize + k];
                for (size_t j = 0; j < blockSize; j++) {
                    accumulator[i * blockSize + j] += aik * dataB[k * blockSize + j];
                }
            }
        }

        // Add accumulated results to result matrix
        for (size_t i = 0; i < blockSize; i++) {
            for (size_t j = 0; j < blockSize; j++) {
                size_t globalRow = coordA.first * blockSize + i;
                size_t globalCol = coordB.second * blockSize + j;
                if (accumulator[i * blockSize + j] != T{}) {
                    result.insert(globalRow, globalCol, 
                                result.get(globalRow, globalCol) + 
                                accumulator[i * blockSize + j]);
                }
            }
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
    void multiplyHybrid(const std::pair<CoordType, CoordType>& coordA,
                        const std::pair<CoordType, CoordType>& coordB,
                        hash_matrix<T>& result) const {
        auto blockA = blockIndex.find(coordA);
        auto blockB = blockIndex.find(coordB);
        
        if (blockA == blockIndex.end() || blockB == blockIndex.end()) {
            return;
        }

        const auto& metadataA = blockA->second;
        const auto& metadataB = blockB->second;

        if (metadataA.type >= BlockType::DENSE) {
            // Dense-Sparse multiplication
            const T* denseBlock = static_cast<const T*>(metadataA.dataPtr);
            const auto* sparseBlock = static_cast<const SparseBlock*>(metadataB.dataPtr);

            for (const auto& [posB, valB] : *sparseBlock) {
                for (size_t i = 0; i < blockSize; i++) {
                    T valA = denseBlock[i * blockSize + posB.first];
                    if (valA != T{}) {
                        size_t globalRow = coordA.first * blockSize + i;
                        size_t globalCol = coordB.second * blockSize + posB.second;
                        
                        if (globalRow < result.numRows && globalCol < result.numCols) {
                            T product = valA * valB;
                            T current = result.get(globalRow, globalCol);
                            result.insert(globalRow, globalCol, current + product);
                        }
                    }
                }
            }
        } else {
            // Sparse-Dense multiplication
            const auto* sparseBlock = static_cast<const SparseBlock*>(metadataA.dataPtr);
            const T* denseBlock = static_cast<const T*>(metadataB.dataPtr);

            for (const auto& [posA, valA] : *sparseBlock) {
                for (size_t j = 0; j < blockSize; j++) {
                    T valB = denseBlock[posA.second * blockSize + j];
                    if (valB != T{}) {
                        size_t globalRow = coordA.first * blockSize + posA.first;
                        size_t globalCol = coordB.second * blockSize + j;
                        
                        if (globalRow < result.numRows && globalCol < result.numCols) {
                            T product = valA * valB;
                            T current = result.get(globalRow, globalCol);
                            result.insert(globalRow, globalCol, current + product);
                        }
                    }
                }
            }
        }
    }

public:
    /**
     * @brief Multiplies this matrix with another matrix
     * 
     * @param other The matrix to multiply with
     * @return Result of multiplication
     * @throws std::invalid_argument if dimensions don't match
     */
    hash_matrix<T> multiply(const hash_matrix<T>& other) const {
        if (numCols != other.numRows) {
            throw std::invalid_argument("Matrix dimensions must match for multiplication");
        }

        hash_matrix<T> result(numRows, other.numCols);
        
        std::cout << "\n=== Starting Matrix Multiplication ===\n";
        std::cout << "Matrix 1: " << numRows << "x" << numCols << "\n";
        std::cout << "Matrix 2: " << other.numRows << "x" << other.numCols << "\n";

        // Log input matrices
        std::cout << "\nMatrix 1 values:\n";
        for (size_t i = 0; i < numRows; i++) {
            for (size_t j = 0; j < numCols; j++) {
                std::cout << get(i, j) << " ";
            }
            std::cout << "\n";
        }

        std::cout << "\nMatrix 2 values:\n";
        for (size_t i = 0; i < other.numRows; i++) {
            for (size_t j = 0; j < other.numCols; j++) {
                std::cout << other.get(i, j) << " ";
            }
            std::cout << "\n";
        }

        for (size_t i = 0; i < (numRows + blockSize - 1) / blockSize; i++) {
            for (size_t j = 0; j < (other.numCols + blockSize - 1) / blockSize; j++) {
                for (size_t k = 0; k < (numCols + blockSize - 1) / blockSize; k++) {
                    std::pair<CoordType, CoordType> coordA{static_cast<CoordType>(i), 
                                                          static_cast<CoordType>(k)};
                    std::pair<CoordType, CoordType> coordB{static_cast<CoordType>(k), 
                                                          static_cast<CoordType>(j)};

                    std::cout << "\nProcessing block multiplication:";
                    std::cout << "\nBlock A: (" << coordA.first << "," << coordA.second << ")";
                    std::cout << "\nBlock B: (" << coordB.first << "," << coordB.second << ")\n";

                    auto blockA = blockIndex.find(coordA);
                    auto blockB = other.blockIndex.find(coordB);

                    if (blockA != blockIndex.end() && blockB != other.blockIndex.end()) {
                        if (blockA->second.type >= BlockType::DENSE && 
                            blockB->second.type >= BlockType::DENSE) {
                            std::cout << "Using dense-dense multiplication\n";
                            multiplyDenseDense(blockA->second, blockB->second, coordA, coordB, result);
                        } else if (blockA->second.type < BlockType::DENSE && 
                                 blockB->second.type < BlockType::DENSE) {
                            std::cout << "Using sparse-sparse multiplication\n";
                            multiplySparse(coordA, coordB, result);
                        } else {
                            std::cout << "Using hybrid multiplication\n";
                            multiplyHybrid(coordA, coordB, result);
                        }
                    }
                }
            }
        }

        std::cout << "\nFinal result matrix:\n";
        for (size_t i = 0; i < result.numRows; i++) {
            for (size_t j = 0; j < result.numCols; j++) {
                std::cout << result.get(i, j) << " ";
            }
            std::cout << "\n";
        }

        return result;
    }

    /**
     * @brief Multiplication operator
     * 
     * @param other The matrix to multiply with
     * @return Result of multiplication
     */
    hash_matrix<T> operator*(const hash_matrix<T>& other) const {
        return multiply(other);
    }

    // Use the converter in public methods
    template<typename InputCoordType>
    void batchInsert(const std::vector<std::tuple<InputCoordType, InputCoordType, T>>& data) {
        for (const auto& [i, j, val] : data) {
            insert(CoordConverter::convert(i), CoordConverter::convert(j), val);
        }
    }

    // Specialization for matching types
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
            result.emplace_back(
                CoordConverter::convert(i),
                CoordConverter::convert(j),
                val
            );
        }
        
        return result;
    }

    void remove(CoordType row, CoordType col) {
        checkBounds(row, col);
        
        auto blockCoord = getBlockCoords(row, col);
        auto localCoord = getLocalCoords(row, col);
        
        auto it = blockIndex.find(blockCoord);
        if (it == blockIndex.end()) return;
        
        auto& block = it->second;
        if (block.type == BlockType::SPARSE) {
            auto* sparseBlock = static_cast<SparseBlock*>(block.dataPtr);
            sparseBlock->erase({localCoord.first, localCoord.second});
            if (sparseBlock->empty()) {
                delete sparseBlock;
                blockIndex.erase(it);
            }
        } else if (block.type == BlockType::DENSE) {
            auto* denseData = static_cast<T*>(block.dataPtr);
            denseData[localCoord.first * blockSize + localCoord.second] = T{};
        }
    }

    /**
     * @brief Constructs a new hash_matrix with specified dimensions
     * 
     * Initializes the matrix storage structures and allocates initial memory.
     * 
     * @param rows Number of rows in the matrix
     * @param cols Number of columns in the matrix
     */
    hash_matrix(size_t rows, size_t cols, size_t blockSize = 8) 
        : numRows(rows), numCols(cols), blockSize(blockSize) {
        if (blockSize == 0) {
            throw std::invalid_argument("Block size must be greater than 0");
        }
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
        std::vector<std::pair<CoordType, CoordType>> denseBlocks;
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
                const auto& block = matrix->memoryPool.get(metadata.dataIndex);
                
                // Prefetch next block for better cache utilization
                if (currentDenseBlock + 1 < denseBlocks.size()) {
                    matrix->prefetchBlock(denseBlocks[currentDenseBlock + 1]);
                }

                // SIMD scan for non-zero elements
                for (size_t i = localRow; i < blockSize; i++) {
                    __m256d zero = _mm256_setzero_pd();
                    for (size_t j = localCol; j < blockSize; j += 4) {
                        __m256d val = _mm256_load_pd(&block[i * blockSize + j]);
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
            if (localCol >= blockSize) {
                localCol = 0;
                localRow++;
                if (localRow >= blockSize) {
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
            size_t globalRow = currentBlockRow * blockSize + localRow;
            size_t globalCol = currentBlockCol * blockSize + localCol;
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
            : matrix(m), currentBlockRow(maxRow / blockSize), 
              currentBlockCol(maxCol / blockSize),
              localRow(maxRow % blockSize), localCol(maxCol % blockSize),
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
        std::vector<std::pair<CoordType, CoordType>> processedBlocks;

        // Process dense blocks first using SIMD
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < (numRows + blockSize - 1) / blockSize; i++) {
            for (size_t j = 0; j < (numCols + blockSize - 1) / blockSize; j++) {
                std::pair<CoordType, CoordType> blockCoord{i, j};
                auto blockA = blockIndex.find(blockCoord);
                auto blockB = other.blockIndex.find(blockCoord);

                if (blockA != blockIndex.end() && blockB != other.blockIndex.end() &&
                    blockA->second.type >= BlockType::DENSE && 
                    blockB->second.type >= BlockType::DENSE) {
                    
                    const auto* dataA = static_cast<const T*>(blockA->second.dataPtr);
                    const auto* dataB = static_cast<const T*>(blockB->second.dataPtr);
                    
                    #pragma omp simd
                    for (size_t k = 0; k < blockSize * blockSize; k++) {
                        T sum = dataA[k] + dataB[k];
                        if (sum != 0) {
                            size_t localRow = k / blockSize;
                            size_t localCol = k % blockSize;
                            result.insert(i * blockSize + localRow, 
                                        j * blockSize + localCol, 
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
            for (size_t i = 0; i < blockSize; i++) {
                for (size_t j = 0; j < blockSize; j++) {
                    size_t globalRow = coord.first * blockSize + i;
                    size_t globalCol = coord.second * blockSize + j;
                    
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

    // Add proper copy constructor
    hash_matrix(const hash_matrix& other)
        : numRows(other.numRows)
        , numCols(other.numCols)
        , blockSize(other.blockSize)
        , memoryPool(other.memoryPool)
        , blockIndex(other.blockIndex)
        , fastLookup(other.fastLookup) {}

    // Add move constructor
    hash_matrix(hash_matrix&& other) noexcept = default;

    // Add proper assignment operators
    hash_matrix& operator=(const hash_matrix& other) {
        if (this != &other) {
            numRows = other.numRows;
            numCols = other.numCols;
            blockSize = other.blockSize;
            memoryPool = other.memoryPool;
            blockIndex = other.blockIndex;
            fastLookup = other.fastLookup;
        }
        return *this;
    }

    hash_matrix& operator=(hash_matrix&& other) noexcept = default;

    // Add method to get current block size
    size_t getBlockSize() const {
        return blockSize;
    }
};

#endif // HASHMATRIX_H