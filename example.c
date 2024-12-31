#include "hashmatrix.h"

int main() {
    // Initialize matrices
    HashMatrix<double> matrixA, matrixB;

    // Populate matrixA
    matrixA.insert(0, 0, 1.0);
    matrixA.insert(0, 1, 2.0);
    matrixA.insert(1, 0, 3.0);

    // Populate matrixB
    matrixB.insert(0, 0, 4.0);
    matrixB.insert(1, 0, 5.0);

    // Add matrices
    auto sum = matrixA.add(matrixB);
    std::cout << "Matrix A + Matrix B:\n";
    sum.print();

    // Scalar multiplication
    auto scaled = matrixA.scalarMultiply(2.0);
    std::cout << "Matrix A * 2:\n";
    scaled.print();

    // Multiply matrices
    auto product = matrixA.multiply(matrixB, 1); // Assume 1 column in B for simplicity
    std::cout << "Matrix A * Matrix B:\n";
    product.print();

    // Transpose
    auto transposed = matrixA.transpose();
    std::cout << "Transposed Matrix A:\n";
    transposed.print();

    return 0;
}
