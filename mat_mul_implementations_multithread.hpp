#ifndef MATRIX_MULTIPLY_MULTI_THREAD_HPP
#define MATRIX_MULTIPLY_MULTI_THREAD_HPP

#include <omp.h>

template <int rows, int columns, int inners, int inner_tile_size, int outer_tile_size>
inline void mat_mul_tiled_3d_multithread(const float *left,
                                         const float *right,
                                         float *result) {

  #pragma omp parallel for shared(result, left, right) default(none) collapse(2) num_threads(8)
  //   Split the matrix into mini matricies (outer_tile_size X outer_tile_size) for each thread to handle.
  for (int rowTile = 0; rowTile < rows; rowTile += outer_tile_size) {
    for (int columnTile = 0; columnTile < columns; columnTile += outer_tile_size) {
      
      // Now apply the inner tiling as before.                   
      for (int innerTile = 0; innerTile < inners; innerTile += inner_tile_size) {
        for (int row = rowTile; row < rowTile + outer_tile_size; row++) {
          int innerTileEnd = std::min(inners, innerTile + inner_tile_size);
          for (int inner = innerTile; inner < innerTileEnd; inner++) {
            for (int col = columnTile; col < columnTile + outer_tile_size; col++) {
              result[row * columns + col] += left[row * inners + inner] * right[inner * columns + col];
            } 
          }
        }
      }
    }
  } 
}

#endif // MATRIX_MULTIPLY_MULTI_THREAD_HPP