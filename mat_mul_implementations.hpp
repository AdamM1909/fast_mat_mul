#ifndef MATRIX_MULTIPLY_HPP
#define MATRIX_MULTIPLY_HPP

#include <omp.h>

template <int rows, int columns, int inners>
void mat_mul_niave(const float* left, const float* right, float* result) {
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < columns; col++) {
            for (int inner = 0; inner < inners; inner++) {
                
                result[row * columns + col] += left[row * inners + inner] * right[inner * columns + col];
            }
        }
    }
}

template <int rows, int columns, int inners>
inline void mat_mul_cache_aware(const float *left, const float *right, float *result) {
  for (int row = 0; row < rows; row++) {
    for (int inner = 0; inner < inners; inner++) {
      for (int col = 0; col < columns; col++) {
        
        result[row * columns + col] += left[row * columns + inner] * right[inner * columns + col];
            } 
        } 
    } 
}

template <int rows, int columns, int inners>
inline void mat_mul_tiled_1d(const float *left, const float *right, float *result, const int tileSize) {
  for (int innerTile = 0; innerTile < inners; innerTile += tileSize) {
    
    
    for (int row = 0; row < rows; row++) {
      int innerTileEnd = std::min(inners, innerTile + tileSize);

      for (int inner = innerTile; inner < innerTileEnd; inner++) {
        for (int column = 0; column < columns; column++) {
          
          result[row * columns + column] += left[row * inners + inner] * right[inner * columns + column];}
      }
    }
  }
}

template <int rows, int columns, int inners, int inner_tile_size>
inline void mat_mul_tiled_3d_multithread(const float *left,
                                                const float *right,
                                                float *result) {

  #pragma omp parallel for shared(result, left, right) default(none) collapse(2) num_threads(8)
  for (int rowTile = 0; rowTile < rows; rowTile += 128) {
    for (int columnTile = 0; columnTile < columns; columnTile += 128) {
      for (int innerTile = 0; innerTile < inners; innerTile += inner_tile_size) {
        
        // Matrix loops.
        for (int row = rowTile; row < rowTile + 128; row++) {
          int innerTileEnd = std::min(inners, innerTile + inner_tile_size);
          for (int inner = innerTile; inner < innerTileEnd; inner++) {
            for (int col = columnTile; col < columnTile + 128; col++) {
              
              result[row * columns + col] += left[row * inners + inner] * right[inner * columns + col];
            } 
          }
        }
      }
    }
  } 
}
#endif // MATRIX_MULTIPLY_HPP