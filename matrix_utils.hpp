
#ifndef MATRIX_UTILS_HPP
#define MATRIX_UTILS_HPP

#include <fstream>
#include <stdexcept>
#include <cmath>


void load_matrix(float *matrix, const char* filename, int rows, int cols) {
    std::ifstream file(filename, std::ios::binary);
    file.read(reinterpret_cast<char*>(matrix), sizeof(float) * rows * cols);

    // Check if the file read was successful
    if (!file) {
        delete[] matrix;  // Clean up memory in case of an error
        throw std::runtime_error("Error reading matrix file.");
    }
}


bool compare_matrices(const float* array_a, const float* array_b, int size, float tolerance=1e-5) {
    for (int i = 0; i < size; ++i) {if (std::abs(array_a[i] - array_b[i]) > tolerance) {return false;}}
    return true;
}

#endif