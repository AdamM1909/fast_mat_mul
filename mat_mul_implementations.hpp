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
inline void matmulImpCacheAware(const float *left, const float *right, float *result) {
  for (int row = 0; row < rows; row++) {
    for (int inner = 0; inner < inners; inner++) {
      for (int col = 0; col < columns; col++) {
        result[row * columns + col] +=
            left[row * columns + inner] * right[inner * columns + col];
            } 
        } 
    } 
}

template <int rows, int columns, int inners>
inline void matmulImplTiling(const float *left, const float *right, float *result,const int tileSize) {
  for (int innerTile = 0; innerTile < inners; innerTile += tileSize) {
    
    
    for (int row = 0; row < rows; row++) {
      int innerTileEnd = std::min(inners, innerTile + tileSize);

      for (int inner = innerTile; inner < innerTileEnd; inner++) {
        for (int column = 0; column < columns; column++) {
          result[row * columns + column] +=
              left[row * inners + inner] * right[inner * columns + column];}
      }
    }
  }
}
#endif // MATRIX_MULTIPLY_HPP