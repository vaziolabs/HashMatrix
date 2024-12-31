#ifndef HASHMATRIX_H
#define HASHMATRIX_H

// Created by Richard I. Christopher, Vazio Labs, 2024.
#include <iostream>
#include <unordered_map>
#include <vector>
#include <functional>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <array>
#include <omp.h>

// Optimized hash function for pair using bit manipulation
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
    static constexpr size_t DENSITY_THRESHOLD = 0.5; // 50% density threshold
    static constexpr size_t BLOCK_SIZE = 64; // Size of dense sub-matrices

    struct Block {
        std::vector<std::vector<T>> dense;  // Dense storage
        std::unordered_map<std::pair<int, int>, T, PairHash> sparse; // Sparse storage
        bool isDense;
        
        Block() : isDense(false) {}
    };

    // Grid of blocks
    std::vector<std::vector<Block>> blocks;
    size_t numRows, numCols;

    const Block& getBlock(int row, int col) const {
        int blockRow = row / BLOCK_SIZE;
        int blockCol = col / BLOCK_SIZE;
        return blocks[blockRow][blockCol];
    }

    Block& getBlock(int row, int col) {
        int blockRow = row / BLOCK_SIZE;
        int blockCol = col / BLOCK_SIZE;
        return blocks[blockRow][blockCol];
    }

    void convertBlockToDense(Block& block) {
        if (block.isDense) return;
        
        block.dense = std::vector<std::vector<T>>(
            BLOCK_SIZE, std::vector<T>(BLOCK_SIZE, 0));
            
        for (const auto& [pos, val] : block.sparse) {
            int localRow = pos.first % BLOCK_SIZE;
            int localCol = pos.second % BLOCK_SIZE;
            block.dense[localRow][localCol] = val;
        }
        
        block.sparse.clear();
        block.isDense = true;
    }

    void convertBlockToSparse(Block& block) {
        if (!block.isDense) return;
        
        for (int i = 0; i < BLOCK_SIZE; i++) {
            for (int j = 0; j < BLOCK_SIZE; j++) {
                if (block.dense[i][j] != 0) {
                    block.sparse[{i, j}] = block.dense[i][j];
                }
            }
        }
        
        block.dense.clear();
        block.isDense = false;
    }

public:
    hash_matrix(size_t rows, size_t cols) : numRows(rows), numCols(cols) {
        size_t numBlockRows = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        size_t numBlockCols = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
        blocks.resize(numBlockRows, std::vector<Block>(numBlockCols));
    }

    void insert(int row, int col, T value) {
        Block& block = getBlock(row, col);
        int localRow = row % BLOCK_SIZE;
        int localCol = col % BLOCK_SIZE;

        if (block.isDense) {
            block.dense[localRow][localCol] = value;
        } else {
            if (value == 0) {
                block.sparse.erase({localRow, localCol});
            } else {
                block.sparse[{localRow, localCol}] = value;
                
                // Check density and convert if needed
                if (block.sparse.size() > BLOCK_SIZE * BLOCK_SIZE * DENSITY_THRESHOLD) {
                    convertBlockToDense(block);
                }
            }
        }
    }

    T get(int row, int col) const {
        const Block& block = getBlock(row, col);
        int localRow = row % BLOCK_SIZE;
        int localCol = col % BLOCK_SIZE;

        if (block.isDense) {
            return block.dense[localRow][localCol];
        } else {
            auto it = block.sparse.find({localRow, localCol});
            return it != block.sparse.end() ? it->second : 0;
        }
    }

    void remove(int row, int col) {
        Block& block = getBlock(row, col);
        int localRow = row % BLOCK_SIZE;
        int localCol = col % BLOCK_SIZE;

        if (block.isDense) {
            block.dense[localRow][localCol] = 0;
            
            // Check if we should convert back to sparse
            size_t nonZeroCount = 0;
            for (const auto& row : block.dense) {
                for (const auto& val : row) {
                    if (val != 0) nonZeroCount++;
                }
            }
            if (nonZeroCount < BLOCK_SIZE * BLOCK_SIZE * DENSITY_THRESHOLD) {
                convertBlockToSparse(block);
            }
        } else {
            block.sparse.erase({localRow, localCol});
        }
    }

    void batchInsert(const std::vector<std::tuple<int, int, T>>& values) {
        // Group insertions by block
        std::unordered_map<std::pair<int, int>, std::vector<std::tuple<int, int, T>>, PairHash> blockGroups;
        
        for (const auto& [row, col, val] : values) {
            int blockRow = row / BLOCK_SIZE;
            int blockCol = col / BLOCK_SIZE;
            blockGroups[{blockRow, blockCol}].push_back({row, col, val});
        }

        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < blocks.size(); i++) {
            for (size_t j = 0; j < blocks[0].size(); j++) {
                auto it = blockGroups.find({i, j});
                if (it != blockGroups.end()) {
                    for (const auto& [row, col, val] : it->second) {
                        insert(row, col, val);
                    }
                }
            }
        }
    }

    hash_matrix<T> add(const hash_matrix<T>& other) const {
        if (numRows != other.numRows || numCols != other.numCols) {
            throw std::invalid_argument("Matrix dimensions must match for addition");
        }

        hash_matrix<T> result(numRows, numCols);

        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < blocks.size(); i++) {
            for (size_t j = 0; j < blocks[0].size(); j++) {
                const Block& block1 = blocks[i][j];
                const Block& block2 = other.blocks[i][j];

                if (block1.isDense && block2.isDense) {
                    // Both dense
                    for (int r = 0; r < BLOCK_SIZE; r++) {
                        for (int c = 0; c < BLOCK_SIZE; c++) {
                            T sum = block1.dense[r][c] + block2.dense[r][c];
                            if (sum != 0) {
                                result.insert(i * BLOCK_SIZE + r, j * BLOCK_SIZE + c, sum);
                            }
                        }
                    }
                } else {
                    // At least one is sparse
                    std::set<std::pair<int, int>> positions;
                    
                    // Collect positions from both blocks
                    if (block1.isDense) {
                        for (int r = 0; r < BLOCK_SIZE; r++) {
                            for (int c = 0; c < BLOCK_SIZE; c++) {
                                if (block1.dense[r][c] != 0) {
                                    positions.insert({r, c});
                                }
                            }
                        }
                    } else {
                        for (const auto& [pos, _] : block1.sparse) {
                            positions.insert(pos);
                        }
                    }

                    if (block2.isDense) {
                        for (int r = 0; r < BLOCK_SIZE; r++) {
                            for (int c = 0; c < BLOCK_SIZE; c++) {
                                if (block2.dense[r][c] != 0) {
                                    positions.insert({r, c});
                                }
                            }
                        }
                    } else {
                        for (const auto& [pos, _] : block2.sparse) {
                            positions.insert(pos);
                        }
                    }

                    // Process all positions
                    for (const auto& [r, c] : positions) {
                        T val1 = getValue(block1, r, c);
                        T val2 = getValue(block2, r, c);
                        T sum = val1 + val2;
                        if (sum != 0) {
                            result.insert(i * BLOCK_SIZE + r, j * BLOCK_SIZE + c, sum);
                        }
                    }
                }
            }
        }

        return result;
    }

    hash_matrix<T> multiply(const hash_matrix<T>& other) const {
        if (numCols != other.numRows) {
            throw std::invalid_argument("Matrix dimensions must match for multiplication");
        }

        hash_matrix<T> result(numRows, other.numCols);
        size_t numBlocksK = (numCols + BLOCK_SIZE - 1) / BLOCK_SIZE;

        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < blocks.size(); i++) {
            for (size_t j = 0; j < other.blocks[0].size(); j++) {
                for (size_t k = 0; k < numBlocksK; k++) {
                    multiplyBlocks(blocks[i][k], other.blocks[k][j], result, i, j, k);
                }
            }
        }

        return result;
    }

private:
    // Helper method for multiply
    void multiplyBlocks(const Block& blockA, const Block& blockB, hash_matrix<T>& result,
                       size_t blockRow, size_t blockCol, size_t blockK) const {
        if (blockA.isDense && blockB.isDense) {
            // Dense multiplication
            for (int i = 0; i < BLOCK_SIZE; i++) {
                for (int j = 0; j < BLOCK_SIZE; j++) {
                    T sum = 0;
                    for (int k = 0; k < BLOCK_SIZE; k++) {
                        sum += blockA.dense[i][k] * blockB.dense[k][j];
                    }
                    if (sum != 0) {
                        result.insert(blockRow * BLOCK_SIZE + i, blockCol * BLOCK_SIZE + j, sum);
                    }
                }
            }
        } else {
            // Sparse multiplication
            for (int i = 0; i < BLOCK_SIZE; i++) {
                for (int j = 0; j < BLOCK_SIZE; j++) {
                    T sum = 0;
                    for (int k = 0; k < BLOCK_SIZE; k++) {
                        sum += getValue(blockA, i, k) * getValue(blockB, k, j);
                    }
                    if (sum != 0) {
                        result.insert(blockRow * BLOCK_SIZE + i, blockCol * BLOCK_SIZE + j, sum);
                    }
                }
            }
        }
    }

    T getValue(const Block& block, int localRow, int localCol) const {
        if (block.isDense) {
            return block.dense[localRow][localCol];
        } else {
            auto it = block.sparse.find({localRow, localCol});
            return it != block.sparse.end() ? it->second : T();
        }
    }
};

#endif // HASHMATRIX_H