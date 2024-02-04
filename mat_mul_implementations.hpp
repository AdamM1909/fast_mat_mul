#ifndef MATRIX_MULTIPLY_HPP
#define MATRIX_MULTIPLY_HPP

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

template <int rows, int columns, int inners>
inline void matmulImpCacheAware(const float *left, const float *right,
                                float *result) {
  for (int row = 0; row < rows; row++) {
    for (int inner = 0; inner < inners; inner++) {
      for (int col = 0; col < columns; col++) {
        result[row * columns + col] +=
            left[row * columns + inner] * right[inner * columns + col];
            } 
        } 
    } 
}

#endif // MATRIX_MULTIPLY_HPP