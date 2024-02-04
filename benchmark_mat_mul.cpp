#include <iostream>
#include <chrono>
#include <random>
#include <Accelerate/Accelerate.h>
// /opt/homebrew/Cellar/gcc/13.2.0/bin/gcc-13 -o vectorizing_loop_add vectorizing_loop_add.c

// Function to generate a random matrix
void generateRandomMatrix(double** matrix, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix[i][j] = dis(gen);
        }
    }
}

// Function to measure matrix multiplication time
double measureMatMulTime(int matrixSize) {
    // Create two random matrices of size matrixSize x matrixSize
    double** matrixA = new double*[matrixSize];
    double** matrixB = new double*[matrixSize];
    for (int i = 0; i < matrixSize; ++i) {
        matrixA[i] = new double[matrixSize];
        matrixB[i] = new double[matrixSize];
    }

    // Generate random matrices
    generateRandomMatrix(matrixA, matrixSize);
    generateRandomMatrix(matrixB, matrixSize);

    // Record start time
    auto startTime = std::chrono::high_resolution_clock::now();

    // Perform matrix multiplication
    double** result = new double*[matrixSize];
    for (int i = 0; i < matrixSize; ++i) {
        result[i] = new double[matrixSize];
        for (int j = 0; j < matrixSize; ++j) {
            result[i][j] = 0.0;
            for (int k = 0; k < matrixSize; ++k) {
                result[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }

    // Record end time
    auto endTime = std::chrono::high_resolution_clock::now();

    // Calculate elapsed time
    double elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() / 1e6;

    // Deallocate memory
    for (int i = 0; i < matrixSize; ++i) {
        delete[] matrixA[i];
        delete[] matrixB[i];
        delete[] result[i];
    }
    delete[] matrixA;
    delete[] matrixB;
    delete[] result;

    return elapsedTime;
}

int main() {
    int matrixSize = 1048;
    int numTests = 1;
    double totalExecutionTime = 0.0;

    for (int i = 0; i < numTests; ++i) {
        double executionTime = measureMatMulTime(matrixSize);
        totalExecutionTime += executionTime;
    }

    double averageExecutionTime = totalExecutionTime / numTests;
    double gflops = 2 * std::pow(matrixSize, 3) / averageExecutionTime / 1e9;

    std::cout<<"Matrix size: "<<matrixSize<<"x"<<matrixSize<<std::endl;
    std::cout<<"Average execution time over "<<numTests<<" tests: " 
              <<averageExecutionTime<<" seconds" 
              <<"GFLOPs: "<<gflops<<std::endl;

    return 0;
}
