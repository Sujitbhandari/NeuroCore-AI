#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <string>

/**
 * Matrix Class
 * 
 * A high-performance matrix implementation using std::vector<double> for memory safety.
 * Designed for deep learning operations with OpenMP parallelization support.
 * 
 * Memory Layout: Row-major order (data[row * cols + col])
 */
class Matrix {
public:
    // Constructors
    Matrix();
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, double value);
    Matrix(const std::vector<std::vector<double>>& data);
    
    // Copy constructor and assignment
    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);
    
    // Destructor
    ~Matrix() = default;
    
    // Accessors
    size_t getRows() const { return rows_; }
    size_t getCols() const { return cols_; }
    double& operator()(size_t row, size_t col);
    const double& operator()(size_t row, size_t col) const;
    
    // Matrix operations
    Matrix dot_product(const Matrix& other) const;  // Matrix multiplication
    Matrix transpose() const;                        // Transpose matrix
    Matrix scalar_multiply(double scalar) const;     // Element-wise scalar multiplication
    Matrix subtract(const Matrix& other) const;      // Element-wise subtraction
    Matrix add(const Matrix& other) const;           // Element-wise addition
    Matrix multiply_elementwise(const Matrix& other) const;  // Hadamard product
    
    // Utility functions
    void fill(double value);
    void randomize(double min = -1.0, double max = 1.0);
    void print(const std::string& name = "") const;
    std::vector<double> flatten() const;
    
    // Static factory methods
    static Matrix zeros(size_t rows, size_t cols);
    static Matrix ones(size_t rows, size_t cols);
    static Matrix fromVector(const std::vector<double>& vec, bool asColumn = true);

private:
    size_t rows_;
    size_t cols_;
    std::vector<double> data_;
    
    // Helper to check bounds
    void checkBounds(size_t row, size_t col) const;
    void checkDimensions(const Matrix& other, const std::string& op) const;
};

#endif // MATRIX_H

