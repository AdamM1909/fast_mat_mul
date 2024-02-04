#include <iostream>
#include <stdexcept>
#include "matrix_utils.hpp" 
#include "mat_mul_implementations.hpp"
#include "benchmark/benchmark.h"

// static void BM_matmulImplNaive(benchmark::State& state) {
//     // Create matrices and result matrix
//     const int rows = 1048;
//     const int columns = 1048;
//     const int inner = 1048;

//     float *left = new float[rows * inner];
//     float *right = new float[inner * columns];
//     float *result = new float[rows * columns];
//     float *check_result = new float[rows * columns];

//     load_matrix(left, "matrix_a.bin", rows, columns);
//     load_matrix(right, "matrix_b.bin", rows, columns);

//     for (auto _ : state) {matmulImplNaive<rows, columns, inner>(left, right, result);}

//     // Check 
//     load_matrix(check_result, "matrix_c.bin", rows, columns);
//     bool arrays_match = compare_matricies(check_result, result, rows*columns);
//     std::cout << (arrays_match ? "********** Result is correct! **********" : " ********** Results do not match. ********** ") << std::endl;

//      // Free allocated memory
//     delete[] left;
//     delete[] right;
//     delete[] result;
//     delete[] check_result;
// }

// BENCHMARK(BM_matmulImplNaive)
//     ->Unit(benchmark::kMillisecond);

// BENCHMARK_MAIN();




// Define a benchmark function
template <int Rows, int Columns, int Inner, typename MatmulFunc>
static void BM_matmulImpl(benchmark::State& state, MatmulFunc matmulFunc) {
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

    for (auto _ : state) {
        matmulFunc(left, right, result);
    }

    // Check 
    load_matrix(check_result, "matrix_c.bin", rows, columns);
    bool arrays_match = compare_matrices(check_result, result, rows * columns);
    std::cout << (arrays_match ? "********** Result is correct! **********" : " ********** Results do not match. ********** ") << std::endl;

    // Free allocated memory
    delete[] left;
    delete[] right;
    delete[] result;
    delete[] check_result;
}

// Instantiate benchmarks for different versions of matmulImpl
BENCHMARK_CAPTURE(BM_matmulImpl, matmulImplNaive<1048, 1048, 1048>);
BENCHMARK_CAPTURE(BM_matmulImpl, matmulImpCacheAware<1048, 1048, 1048>);

// Run the benchmarks
BENCHMARK_MAIN();