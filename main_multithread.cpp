
#include <iostream>
#include <stdexcept>
#include "matrix_utils.hpp" 
#include "mat_mul_implementations_multithread.hpp"
#include "benchmark/benchmark.h"

// g++ main.cpp -std=c++11 -lbenchmark -lpthread -o main -march=native -ffast-math -O3 -Xpreprocessor -fopenmp -I/opt/homebrew/Cellar/libomp/17.0.6/include -L/opt/homebrew/Cellar/libomp/17.0.6/lib -lomp

static void BM_matmulImpl(benchmark::State& state) {
    // Create matrices and result matrix
    const int rows = 1024;
    const int columns = 1024;
    const int inner = 1024;

    float *left = new float[rows * inner];
    float *right = new float[inner * columns];
    float *result = new float[rows * columns];
    float *check_result = new float[rows * columns];

    load_matrix(left, "matrix_a.bin", rows, columns);
    load_matrix(right, "matrix_b.bin", rows, columns);


    for (auto _ : state) {
        state.PauseTiming();
        std::fill_n(result, rows * columns, 0);
        state.ResumeTiming();
        const int inner_tile_size = state.range(0);
        // Cases must be templates for multthreading compilation. 
        switch(inner_tile_size){
            case 8: mat_mul_tiled_3d_multithread<rows, columns, inner, 8, 128>(left, right, result); break;
            case 16: mat_mul_tiled_3d_multithread<rows, columns, inner, 16, 128>(left, right, result); break;
            case 32: mat_mul_tiled_3d_multithread<rows, columns, inner, 32, 128>(left, right, result); break;
            case 64: mat_mul_tiled_3d_multithread<rows, columns, inner, 64, 128>(left, right, result); break;
            case 128: mat_mul_tiled_3d_multithread<rows, columns, inner, 128, 128>(left, right, result); break;
            default: break;
        }
    }
    // Check 
    load_matrix(check_result, "matrix_c.bin", rows, columns);
    bool arrays_match = compare_matrices(check_result, result, rows*columns);
    std::cout << (arrays_match ? "********** Result is correct. **********" : " ********** Results do not match. ********** ") << std::endl;

     // Free allocated memory
    delete[] left;
    delete[] right;
    delete[] result;
    delete[] check_result;
}

BENCHMARK(BM_matmulImpl)
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(2)->Range(8, 128)
    ->UseRealTime();

BENCHMARK_MAIN();