/*
 * CUDA Particle Swarm Optimization
 * Helena Wu
 * CS 179 Final Project
 */


#ifndef PSO_DEVICE_CUH
#define PSO_DEVICE_CUH

#include <cstdio>
#include <curand.h>
#include "cuda_header.cuh"
// #include "benchmark_functions.h"
using namespace std;

/*
 * NOTE: You can use this macro to easily check cuda error codes
 * and get more information.
 *
 * Modified from:
 * http://stackoverflow.com/questions/14038589/
 *   what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
 */
#define gpu_errchk(ans) { gpu_assert((ans), __FILE__, __LINE__); }
inline void gpu_assert(cudaError_t code, const char *file, int line,
                       bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "gpu_assert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        exit(code);
    }
}


/* Benchmark optimization problem known in literature as Rosenbrock's function.
 * The minimum of the function is at (1, 1) with value 0. We search in the
 * domain [-5.12, 5.12]^2 as in other benchmark tests.
 */
CUDA_CALLABLE float rosenbrock(float* solution);


/* Benchmark optimization problem known in literature as the Rastrigin function.
 * The minimum of the function is at (0, 0, 0, 0) with value 0. We search in the
 * domain [-5.12, 5.12]^dim as in other benchmark tests.
 */
CUDA_CALLABLE float rastrigin(float* solution, int dim);


/* Returns the value of the objective function, which we are trying to minimize.
 * User defines which objective function to use for benchmark tests.
 */
CUDA_CALLABLE float cost(int objective, float* solution, int dim);


/* Returns 1 if solution 1 < solution 2, and returns 0 otherwise.
 */
CUDA_CALLABLE float is_min_cost(int objective, float* solution1, float* solution2, int dim);


// This function will be called from the host code to invoke the kernel
// function. Any memory address/pointer locations passed to this function
// must be host addresses. This function will be in charge of allocating
// GPU memory, invoking the kernel, and cleaning up afterwards. The result
// will be stored in out_data. The function returns the amount of time that
// it took for the function to complete (prior to returning) in milliseconds.
float cuda_call_pso_kernel(const unsigned int blocks,
                           const unsigned int threads_per_block,
                           float *in_solutions, float *in_velocities,
                           float *out_data, const unsigned int num_particles,
                           const unsigned int dim, const int benchmark,
                           const float c1, const float c2, const float w);

#endif
