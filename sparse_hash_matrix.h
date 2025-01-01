#ifndef SPARSE_HASH_MATRIX_H
#define SPARSE_HASH_MATRIX_H

#include "debug.h"
#include <vector>
#include <cstddef>
#include <type_traits>
#include <stdexcept>

template<typename T = double>
class sparse_hash_matrix {
private:
    size_t rows_;
    size_t cols_;
    size_t block_size_;
    
    struct MatrixElement {
        size_t row;
        size_t col;
        T value;
        MatrixElement* next;
        
        MatrixElement(size_t r, size_t c, T val) 
            : row(r), col(c), value(val), next(nullptr) {}
    };
    
    std::vector<MatrixElement*> hash_table_;
    size_t table_size_;

    size_t hash(size_t row, size_t col) const {
        return (row * cols_ + col) % table_size_;
    }

public:
    // Constructor and destructor
    sparse_hash_matrix(size_t rows, size_t cols, size_t block_size = 16);
    ~sparse_hash_matrix();
    
    // Copy constructor and assignment
    sparse_hash_matrix(const sparse_hash_matrix& other);
    sparse_hash_matrix& operator=(const sparse_hash_matrix& other);
    
    // Move constructor and assignment
    sparse_hash_matrix(sparse_hash_matrix&& other) noexcept;
    sparse_hash_matrix& operator=(sparse_hash_matrix&& other) noexcept;

    // Basic operations
    void insert(size_t row, size_t col, T value);
    T get(size_t row, size_t col) const;
    void remove(size_t row, size_t col);

    // Arithmetic operators
    sparse_hash_matrix operator+(const sparse_hash_matrix& other) const;
    sparse_hash_matrix operator-(const sparse_hash_matrix& other) const;
    sparse_hash_matrix operator*(const sparse_hash_matrix& other) const;
    sparse_hash_matrix operator*(const T scalar) const;
    
    // Compound assignment operators
    sparse_hash_matrix& operator+=(const sparse_hash_matrix& other);
    sparse_hash_matrix& operator-=(const sparse_hash_matrix& other);
    sparse_hash_matrix& operator*=(const T scalar);

    // Matrix operations
    sparse_hash_matrix transpose() const;
    
    // Element access operators
    T& operator()(size_t row, size_t col);
    const T& operator()(size_t row, size_t col) const;
    
    // Iterator support
    class iterator;
    class const_iterator;
    iterator begin();
    iterator end();
    const_iterator cbegin() const;
    const_iterator cend() const;

    // Utility functions
    size_t num_rows() const { return rows_; }
    size_t num_cols() const { return cols_; }
    bool is_empty() const;
    void clear();
    size_t non_zero_elements() const;
};

// Non-member operators
template<typename T>
sparse_hash_matrix<T> operator*(const T scalar, const sparse_hash_matrix<T>& matrix);

// Iterator implementation
template<typename T>
class sparse_hash_matrix<T>::iterator {
private:
    sparse_hash_matrix<T>* matrix_;
    size_t current_bucket_;
    MatrixElement* current_element_;
    
    void find_next_element() {
        while (current_bucket_ < matrix_->table_size_ && 
               !matrix_->hash_table_[current_bucket_]) {
            ++current_bucket_;
        }
        if (current_bucket_ < matrix_->table_size_) {
            current_element_ = matrix_->hash_table_[current_bucket_];
        } else {
            current_element_ = nullptr;
        }
    }

public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using reference = T&;

    iterator(sparse_hash_matrix<T>* matrix, size_t bucket = 0)
        : matrix_(matrix), current_bucket_(bucket), current_element_(nullptr) {
        find_next_element();
    }

    iterator& operator++() {
        if (current_element_) {
            current_element_ = current_element_->next;
            if (!current_element_) {
                ++current_bucket_;
                find_next_element();
            }
        }
        return *this;
    }

    bool operator==(const iterator& other) const {
        return current_element_ == other.current_element_;
    }

    bool operator!=(const iterator& other) const {
        return !(*this == other);
    }

    MatrixElement& operator*() {
        return *current_element_;
    }
};

template<typename T>
typename sparse_hash_matrix<T>::iterator sparse_hash_matrix<T>::begin() {
    return iterator(this);
}

template<typename T>
typename sparse_hash_matrix<T>::iterator sparse_hash_matrix<T>::end() {
    return iterator(this, table_size_);
}

template<typename T>
class sparse_hash_matrix<T>::const_iterator {
    // Const iterator implementation details
};

template<typename T>
sparse_hash_matrix<T>::sparse_hash_matrix(size_t rows, size_t cols, size_t block_size)
    : rows_(rows)
    , cols_(cols)
    , block_size_(block_size)
    , table_size_(rows * cols) // Or use a prime number close to rows * cols for better distribution
{
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    hash_table_.resize(table_size_, nullptr);
}

template<typename T>
sparse_hash_matrix<T>::~sparse_hash_matrix() {
    clear();
}

template<typename T>
void sparse_hash_matrix<T>::clear() {
    for (size_t i = 0; i < table_size_; ++i) {
        MatrixElement* current = hash_table_[i];
        while (current) {
            MatrixElement* next = current->next;
            delete current;
            current = next;
        }
        hash_table_[i] = nullptr;
    }
}

template<typename T>
void sparse_hash_matrix<T>::insert(size_t row, size_t col, T value) {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Index out of bounds");
    }

    size_t index = hash(row, col);
    
    // If value is zero, just remove any existing element
    if (value == T{}) {
        remove(row, col);
        return;
    }

    // Check if element already exists
    MatrixElement* current = hash_table_[index];
    MatrixElement* prev = nullptr;

    while (current) {
        if (current->row == row && current->col == col) {
            current->value = value;
            return;
        }
        prev = current;
        current = current->next;
    }

    // Create new element
    MatrixElement* newElement = new MatrixElement(row, col, value);
    if (prev) {
        prev->next = newElement;
    } else {
        hash_table_[index] = newElement;
    }
}

template<typename T>
T sparse_hash_matrix<T>::get(size_t row, size_t col) const {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Index out of bounds");
    }

    size_t index = hash(row, col);
    MatrixElement* current = hash_table_[index];

    while (current) {
        if (current->row == row && current->col == col) {
            return current->value;
        }
        current = current->next;
    }

    return T{};
}

template<typename T>
sparse_hash_matrix<T> sparse_hash_matrix<T>::operator+(const sparse_hash_matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }

    sparse_hash_matrix result(rows_, cols_, block_size_);
    
    // Add elements from this matrix
    for (size_t i = 0; i < table_size_; ++i) {
        MatrixElement* current = hash_table_[i];
        while (current) {
            result.insert(current->row, current->col, current->value);
            current = current->next;
        }
    }

    // Add elements from other matrix
    for (size_t i = 0; i < other.table_size_; ++i) {
        MatrixElement* current = other.hash_table_[i];
        while (current) {
            T sum = result.get(current->row, current->col) + current->value;
            result.insert(current->row, current->col, sum);
            current = current->next;
        }
    }

    return result;
}

template<typename T>
sparse_hash_matrix<T> sparse_hash_matrix<T>::operator*(const T scalar) const {
    sparse_hash_matrix result(rows_, cols_, block_size_);
    
    for (size_t i = 0; i < table_size_; ++i) {
        MatrixElement* current = hash_table_[i];
        while (current) {
            result.insert(current->row, current->col, current->value * scalar);
            current = current->next;
        }
    }
    
    return result;
}

template<typename T>
bool sparse_hash_matrix<T>::is_empty() const {
    for (size_t i = 0; i < table_size_; ++i) {
        if (hash_table_[i] != nullptr) {
            return false;
        }
    }
    return true;
}

template<typename T>
size_t sparse_hash_matrix<T>::non_zero_elements() const {
    size_t count = 0;
    for (size_t i = 0; i < table_size_; ++i) {
        MatrixElement* current = hash_table_[i];
        while (current) {
            count++;
            current = current->next;
        }
    }
    return count;
}

template<typename T>
sparse_hash_matrix<T> sparse_hash_matrix<T>::transpose() const {
    sparse_hash_matrix result(cols_, rows_, block_size_);
    
    for (size_t i = 0; i < table_size_; ++i) {
        MatrixElement* current = hash_table_[i];
        while (current) {
            result.insert(current->col, current->row, current->value);
            current = current->next;
        }
    }
    
    return result;
}

template<typename T>
void sparse_hash_matrix<T>::remove(size_t row, size_t col) {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Index out of bounds");
    }

    size_t index = hash(row, col);
    MatrixElement* current = hash_table_[index];
    MatrixElement* prev = nullptr;

    while (current) {
        if (current->row == row && current->col == col) {
            if (prev) {
                prev->next = current->next;
            } else {
                hash_table_[index] = current->next;
            }
            delete current;
            return;
        }
        prev = current;
        current = current->next;
    }
}

template<typename T>
sparse_hash_matrix<T> sparse_hash_matrix<T>::operator*(const sparse_hash_matrix& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Matrix dimensions must match for multiplication");
    }

    sparse_hash_matrix result(rows_, other.cols_, block_size_);

    // For each non-zero element in this matrix
    for (size_t i = 0; i < table_size_; ++i) {
        MatrixElement* current = hash_table_[i];
        while (current) {
            // For each non-zero element in the corresponding row of other matrix
            for (size_t j = 0; j < other.table_size_; ++j) {
                MatrixElement* other_current = other.hash_table_[j];
                while (other_current) {
                    if (current->col == other_current->row) {
                        T product = current->value * other_current->value;
                        T existing = result.get(current->row, other_current->col);
                        result.insert(current->row, other_current->col, existing + product);
                    }
                    other_current = other_current->next;
                }
            }
            current = current->next;
        }
    }

    return result;
}

// Copy constructor implementation
template<typename T>
sparse_hash_matrix<T>::sparse_hash_matrix(const sparse_hash_matrix& other)
    : rows_(other.rows_), cols_(other.cols_), block_size_(other.block_size_),
      table_size_(other.table_size_) {
    hash_table_.resize(table_size_, nullptr);
    
    // Copy each element
    for (size_t i = 0; i < table_size_; ++i) {
        MatrixElement* current = other.hash_table_[i];
        MatrixElement** target = &hash_table_[i];
        
        while (current) {
            *target = new MatrixElement(current->row, current->col, current->value);
            target = &((*target)->next);
            current = current->next;
        }
    }
}

// Move constructor implementation
template<typename T>
sparse_hash_matrix<T>::sparse_hash_matrix(sparse_hash_matrix&& other) noexcept
    : rows_(other.rows_), cols_(other.cols_), block_size_(other.block_size_),
      table_size_(other.table_size_), hash_table_(std::move(other.hash_table_)) {
    other.rows_ = 0;
    other.cols_ = 0;
    other.table_size_ = 0;
}

// Copy assignment operator
template<typename T>
sparse_hash_matrix<T>& sparse_hash_matrix<T>::operator=(const sparse_hash_matrix& other) {
    if (this != &other) {
        clear();
        rows_ = other.rows_;
        cols_ = other.cols_;
        block_size_ = other.block_size_;
        table_size_ = other.table_size_;
        hash_table_.resize(table_size_, nullptr);
        
        // Copy each element
        for (size_t i = 0; i < table_size_; ++i) {
            MatrixElement* current = other.hash_table_[i];
            MatrixElement** target = &hash_table_[i];
            
            while (current) {
                *target = new MatrixElement(current->row, current->col, current->value);
                target = &((*target)->next);
                current = current->next;
            }
        }
    }
    return *this;
}

// Move assignment operator
template<typename T>
sparse_hash_matrix<T>& sparse_hash_matrix<T>::operator=(sparse_hash_matrix&& other) noexcept {
    if (this != &other) {
        clear();
        rows_ = other.rows_;
        cols_ = other.cols_;
        block_size_ = other.block_size_;
        table_size_ = other.table_size_;
        hash_table_ = std::move(other.hash_table_);
        
        other.rows_ = 0;
        other.cols_ = 0;
        other.table_size_ = 0;
    }
    return *this;
}

#endif