#include "../include/Matrix.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <stdexcept>
#include <omp.h>

// Constructor: Empty matrix
Matrix::Matrix() : rows_(0), cols_(0) {}

// Constructor: Matrix with specified dimensions (initialized to 0)
Matrix::Matrix(size_t rows, size_t cols) 
    : rows_(rows), cols_(cols), data_(rows * cols, 0.0) {}

// Constructor: Matrix with specified dimensions and initial value
Matrix::Matrix(size_t rows, size_t cols, double value)
    : rows_(rows), cols_(cols), data_(rows * cols, value) {}

// Constructor: From 2D vector
Matrix::Matrix(const std::vector<std::vector<double>>& data) {
    if (data.empty()) {
        rows_ = 0;
        cols_ = 0;
        return;
    }
    
    rows_ = data.size();
    cols_ = data[0].size();
    data_.resize(rows_ * cols_);
    
    for (size_t i = 0; i < rows_; ++i) {
        if (data[i].size() != cols_) {
            throw std::invalid_argument("All rows must have the same number of columns");
        }
        for (size_t j = 0; j < cols_; ++j) {
            data_[i * cols_ + j] = data[i][j];
        }
    }
}

// Copy constructor
Matrix::Matrix(const Matrix& other)
    : rows_(other.rows_), cols_(other.cols_), data_(other.data_) {}

// Assignment operator
Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        rows_ = other.rows_;
        cols_ = other.cols_;
        data_ = other.data_;
    }
    return *this;
}

// Element access (non-const)
double& Matrix::operator()(size_t row, size_t col) {
    checkBounds(row, col);
    return data_[row * cols_ + col];
}

// Element access (const)
const double& Matrix::operator()(size_t row, size_t col) const {
    checkBounds(row, col);
    return data_[row * cols_ + col];
}

/**
 * Matrix Multiplication (Dot Product)
 * 
 * Computes C = A * B where:
 * - A is (m x n)
 * - B is (n x p)
 * - C is (m x p)
 * 
 * Uses OpenMP for parallel computation across rows.
 * Each thread processes a subset of rows independently.
 */
Matrix Matrix::dot_product(const Matrix& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument(
            "Matrix dimensions incompatible for multiplication: " +
            std::to_string(rows_) + "x" + std::to_string(cols_) + " * " +
            std::to_string(other.rows_) + "x" + std::to_string(other.cols_)
        );
    }
    
    Matrix result(rows_, other.cols_);
    
    // Parallelize outer loop (rows) using OpenMP
    // Each thread computes a subset of result rows
    #pragma omp parallel for
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < other.cols_; ++j) {
            double sum = 0.0;
            // Inner loop: dot product of row i of A and column j of B
            for (size_t k = 0; k < cols_; ++k) {
                sum += data_[i * cols_ + k] * other.data_[k * other.cols_ + j];
            }
            result.data_[i * other.cols_ + j] = sum;
        }
    }
    
    return result;
}

/**
 * Transpose Matrix
 * 
 * Creates a new matrix where rows become columns and vice versa.
 * A^T[i][j] = A[j][i]
 */
Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);
    
    #pragma omp parallel for
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result.data_[j * rows_ + i] = data_[i * cols_ + j];
        }
    }
    
    return result;
}

/**
 * Scalar Multiplication
 * 
 * Multiplies every element by a scalar value.
 * Uses OpenMP to parallelize the operation.
 */
Matrix Matrix::scalar_multiply(double scalar) const {
    Matrix result(rows_, cols_);
    
    #pragma omp parallel for
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * scalar;
    }
    
    return result;
}

/**
 * Matrix Subtraction
 * 
 * Performs element-wise subtraction: C = A - B
 * Both matrices must have the same dimensions.
 */
Matrix Matrix::subtract(const Matrix& other) const {
    checkDimensions(other, "subtraction");
    
    Matrix result(rows_, cols_);
    
    #pragma omp parallel for
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
    
    return result;
}

/**
 * Matrix Addition
 * 
 * Performs element-wise addition: C = A + B
 * Both matrices must have the same dimensions.
 */
Matrix Matrix::add(const Matrix& other) const {
    checkDimensions(other, "addition");
    
    Matrix result(rows_, cols_);
    
    #pragma omp parallel for
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    
    return result;
}

/**
 * Element-wise Multiplication (Hadamard Product)
 * 
 * Performs element-wise multiplication: C[i][j] = A[i][j] * B[i][j]
 * Both matrices must have the same dimensions.
 */
Matrix Matrix::multiply_elementwise(const Matrix& other) const {
    checkDimensions(other, "element-wise multiplication");
    
    Matrix result(rows_, cols_);
    
    #pragma omp parallel for
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }
    
    return result;
}

// Fill matrix with a constant value
void Matrix::fill(double value) {
    #pragma omp parallel for
    for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] = value;
    }
}

// Randomize matrix values
void Matrix::randomize(double min, double max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(min, max);
    
    for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] = dis(gen);
    }
}

// Print matrix
void Matrix::print(const std::string& name) const {
    if (!name.empty()) {
        std::cout << name << " (" << rows_ << "x" << cols_ << "):\n";
    }
    
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(4) 
                      << data_[i * cols_ + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

// Flatten matrix to 1D vector
std::vector<double> Matrix::flatten() const {
    return data_;
}

// Static: Create zero matrix
Matrix Matrix::zeros(size_t rows, size_t cols) {
    return Matrix(rows, cols, 0.0);
}

// Static: Create ones matrix
Matrix Matrix::ones(size_t rows, size_t cols) {
    return Matrix(rows, cols, 1.0);
}

// Static: Create matrix from vector
Matrix Matrix::fromVector(const std::vector<double>& vec, bool asColumn) {
    if (asColumn) {
        Matrix result(vec.size(), 1);
        for (size_t i = 0; i < vec.size(); ++i) {
            result.data_[i] = vec[i];
        }
        return result;
    } else {
        Matrix result(1, vec.size());
        for (size_t i = 0; i < vec.size(); ++i) {
            result.data_[i] = vec[i];
        }
        return result;
    }
}

// Helper: Check array bounds
void Matrix::checkBounds(size_t row, size_t col) const {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range(
            "Matrix index out of bounds: (" + std::to_string(row) + ", " +
            std::to_string(col) + ") for matrix of size " +
            std::to_string(rows_) + "x" + std::to_string(cols_)
        );
    }
}

// Helper: Check dimensions match
void Matrix::checkDimensions(const Matrix& other, const std::string& op) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument(
            "Matrix dimensions must match for " + op + ": " +
            std::to_string(rows_) + "x" + std::to_string(cols_) + " vs " +
            std::to_string(other.rows_) + "x" + std::to_string(other.cols_)
        );
    }
}

