#include <iostream>
#include <omp.h>

// clang++ -o testing_multithreading testing_multithreading.cpp -std=c++11 -Xpreprocessor -fopenmp -I/opt/homebrew/Cellar/libomp/17.0.6/include -L/opt/homebrew/Cellar/libomp/17.0.6/lib -lomp

int main() {
    // Number of threads
    const int num_threads = 10;

    // Start the parallel region with 10 threads
    #pragma omp parallel num_threads(num_threads)
    {
        // Get the thread number
        int thread_number = omp_get_thread_num();

        // Print the thread number
        std::cout << "Thread " << thread_number << " is running" << std::endl;
    }

    return 0;
}

