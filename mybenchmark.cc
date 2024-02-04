#include <benchmark/benchmark.h>
#include <random>

// Function to initialize a matrix with random values
void initializeRandomMatrix(float *matrix, int rows, int columns) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    for (int i = 0; i < rows * columns; i++) {
        matrix[i] = dis(gen);
    }
}

template <int rows, int columns, int inners>
inline void matmulImplNaive(const float *left, const float *right,
                            float *result) {
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < columns; col++) {
      for (int inner = 0; inner < inners; inner++) {
        result[row * columns + col] +=
            left[row * columns + inner] * right[inner * columns + col];
} } } }

template <int rows, int columns, int inners>
inline void matmulImplNaiveRegisterAcc(const float *left, const float *right,
                                       float *result) {
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < columns; col++) {
      float acc = 0.0;
      for (int inner = 0; inner < inners; inner++) {
        acc += left[row * columns + inner] * right[inner * columns + col];
      }
      result[row * columns + col] = acc;
} } }

// Benchmark for matmulImplNaive function
static void BM_MatrixMultiplicationNaive(benchmark::State &state) {
    const int rows = 1048;
    const int columns = 1048;
    const int inners = 1048;

    // Create matrices and result matrix
    float *left = new float[rows * inners];
    float *right = new float[inners * columns];
    float *result = new float[rows * columns];

    // Initialize matrices with random values
    initializeRandomMatrix(left, rows, inners);
    initializeRandomMatrix(right, inners, columns);

    // Warm-up: Run the multiplication once without measuring time
    matmulImplNaive<rows, columns, inners>(left, right, result);

    // Reset the result matrix
    std::fill(result, result + rows * columns, 0.0f);

    for (auto _ : state) {
        // Call the matrix multiplication function without timing the initialization
        matmulImplNaive<rows, columns, inners>(left, right, result);
    }

    // Free allocated memory
    delete[] left;
    delete[] right;
    delete[] result;
}

// Benchmark for matmulImplNaiveRegisterAcc function
static void BM_MatrixMultiplicationNaiveRegisterAcc(benchmark::State &state) {
    const int rows = 1048;
    const int columns = 1048;
    const int inners = 1048;

    // Create matrices and result matrix
    float *left = new float[rows * inners];
    float *right = new float[inners * columns];
    float *result = new float[rows * columns];

    // Initialize matrices with random values
    initializeRandomMatrix(left, rows, inners);
    initializeRandomMatrix(right, inners, columns);

    // Warm-up: Run the multiplication once without measuring time
    matmulImplNaiveRegisterAcc<rows, columns, inners>(left, right, result);

    // Reset the result matrix
    std::fill(result, result + rows * columns, 0.0f);

    for (auto _ : state) {
        // Call the matrix multiplication function without timing the initialization
        matmulImplNaiveRegisterAcc<rows, columns, inners>(left, right, result);
    }

    // Free allocated memory
    delete[] left;
    delete[] right;
    delete[] result;
}

// Register the benchmarks
BENCHMARK(BM_MatrixMultiplicationNaive);
BENCHMARK(BM_MatrixMultiplicationNaiveRegisterAcc);

// Run the benchmarks
BENCHMARK_MAIN();
