#include <iostream>
#include <stdexcept>
#include "matrix_utils.cpp" 
// #include "matmul_implementations.cpp" 

template <int rows, int columns, int inners>
void matmulImplNaive(const float* left, const float* right, float* result) {
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < columns; col++) {
            for (int inner = 0; inner < inners; inner++) {
                result[row * columns + col] +=
                    left[row * inners + inner] * right[inner * columns + col];
            }
        }
    }
}


int main() {
    // Create matrices and result matrix
    const int rows = 1048;
    const int columns = 1048;
    const int inner = 1048;

    float *left = new float[rows * inner];
    float *right = new float[inner * columns];
    float *result = new float[rows * columns];
    float *check_result = new float[rows * columns];

    load_matrix(left, "matrix_a.bin", rows, columns);
    load_matrix(right, "matrix_b.bin", rows, columns);

    // Matmul 
    matmulImplNaive<rows, columns, inner>(left, right, result);

    // Check 
    load_matrix(check_result, "matrix_c.bin", rows, columns);
    bool arrays_match = compare_matricies(check_result, result, rows*columns);
    std::cout << (arrays_match ? "Result is correct!" : " ---- Arrays do not match. ---- ") << std::endl;

    // Free allocated memory
    delete[] left;
    delete[] right;
    delete[] result;
    delete[] check_result;
    return 0;
}