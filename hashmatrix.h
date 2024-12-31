// Created by Richard I. Christopher, Vazio Labs, 2024.
#include <iostream>
#include <unordered_map>
#include <vector>
#include <functional>
#include <thread>
#include <mutex>
#include <omp.h>

// Hash function for pair
struct PairHash {
    template <typename T1, typename T2>
    std::size_t operator()(const std::pair<T1, T2>& pair) const {
        return std::hash<T1>()(pair.first) ^ (std::hash<T2>()(pair.second) << 1);
    }
};

template <typename T>
class HashMatrix {
private:
    // Using a hashmap for sparse matrix storage
    std::unordered_map<std::pair<int, int>, T, PairHash> elements;
    mutable std::mutex mutex_; // For thread-safe operations

    // Internal utility to insert or update elements
    void safeInsert(int row, int col, T value) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (value == static_cast<T>(0)) {
            elements.erase({row, col}); // Remove zero entries
        } else {
            elements[{row, col}] = value;
        }
    }

public:
    HashMatrix() = default;

    // Insert or update a value
    void insert(int row, int col, T value) {
        safeInsert(row, col, value);
    }

    // Get the value at (row, col)
    T get(int row, int col) const {
        auto it = elements.find({row, col});
        return (it != elements.end()) ? it->second : static_cast<T>(0);
    }

    // Add two matrices
    HashMatrix<T> add(const HashMatrix<T>& other) const {
        HashMatrix<T> result;
        for (const auto& [key, value] : elements) {
            result.insert(key.first, key.second, value + other.get(key.first, key.second));
        }
        for (const auto& [key, value] : other.elements) {
            if (elements.find(key) == elements.end()) { // Avoid double processing
                result.insert(key.first, key.second, value);
            }
        }
        return result;
    }

    // Multiply by scalar
    HashMatrix<T> scalarMultiply(T scalar) const {
        HashMatrix<T> result;
        for (const auto& [key, value] : elements) {
            result.insert(key.first, key.second, value * scalar);
        }
        return result;
    }

    // Matrix multiplication with parallel processing and sparsity awareness
    HashMatrix<T> multiply(const HashMatrix<T>& other, int resultCols) const {
        HashMatrix<T> result;

        #pragma omp parallel for
        for (auto it = elements.begin(); it != elements.end(); ++it) {
            const auto& [keyA, valueA] = *it;
            int rowA = keyA.first, colA = keyA.second;

            for (int colB = 0; colB < resultCols; ++colB) {
                T valueB = other.get(colA, colB);
                if (valueB != static_cast<T>(0)) {
                    T currentValue = result.get(rowA, colB);
                    result.insert(rowA, colB, currentValue + valueA * valueB);
                }
            }
        }
        return result;
    }

    // Print the non-zero elements for debugging
    void print() const {
        for (const auto& [key, value] : elements) {
            std::cout << "Value at (" << key.first << ", " << key.second << "): " << value << "\n";
        }
    }

    // Memory usage stats
    size_t memoryUsage() const {
        return elements.size() * (sizeof(std::pair<int, int>) + sizeof(T));
    }

    // Matrix transpose
    HashMatrix<T> transpose() const {
        HashMatrix<T> result;
        for (const auto& [key, value] : elements) {
            result.insert(key.second, key.first, value);
        }
        return result;
    }
};
