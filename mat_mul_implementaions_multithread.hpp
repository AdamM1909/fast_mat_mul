#ifndef MATRIX_MULTIPLY_MULTI_THREAD_HPP
#define MATRIX_MULTIPLY_MULTI_THREAD_HPP

#include <omp.h>

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

#endif // MATRIX_MULTIPLY_MULTI_THREAD_HPP