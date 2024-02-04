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

// Explicit instantiation for the required sizes
template void matmulImplNaive<1048, 1048, 1048>(const float*, const float*, float*);