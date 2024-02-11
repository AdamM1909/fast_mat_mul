#include <iostream>
#include <stdexcept>
#include "matrix_utils.hpp" 
#include "mat_mul_implementations.hpp"
#include "benchmark/benchmark.h"
// g++ main.cpp -std=c++11 -lbenchmark -lpthread -o main -march=native -ffast-math -O3

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
        const int tile_size = state.range(0);
        mat_mul_tiled_1d<rows, columns, inner>(left, right, result, tile_size);
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
    ->Arg(19)->Arg(20)->Arg(21)->Arg(22)->Arg(23)->Arg(24);

BENCHMARK_MAIN();
