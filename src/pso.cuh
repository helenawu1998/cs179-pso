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

#define PI 3.14159265358979

/* Benchmark optimization problem known in literature as Rosenbrock's function.
 * The minimum of the function is at (1, 1) with value 0. We search in the
 * domain [-5.12, 5.12]^2 as in other benchmark tests.
 */
float rosenbrock(float* solution) {
    // Penalize solutions that are out of the domain
    if (solution[0] < -5.12 || solution[0] > 5.12 || solution[1] < -5.12 || solution[1] > 5.12 ){
        return 1000;
    }
    return (100 * pow(pow(solution[0], 2) - solution[1], 2) +
        pow(1 - solution[0], 2));
}

/* Benchmark optimization problem known in literature as the Rastrigin function.
 * The minimum of the function is at (0, 0, 0, 0) with value 0. We search in the
 * domain [-5.12, 5.12]^dim as in other benchmark tests.
 */
float rastrigin(float* solution, int dim) {
    float ans = 10 * dim;
    // Penalize solutions that are out of the domain. Otherwise compute objective.
    for (int i = 0; i < dim; i++) {
        if (solution[i] < -5.12 || solution[i] > 5.12)
            return 1000;
        ans += pow(solution[i], 2) - 10 * cos(2 * PI * solution[i]);
    }
    return ans;
}
// // This function will conduct the convolution for a particular thread index
/* Returns the value of the objective function, which we are trying to minimize.
 * User defines which objective function to use for benchmark tests.
 */
CUDA_CALLABLE float cost(int objective, float* solution, int dim) {
    if (objective == 0)
        return rosenbrock(solution);
    else {
        return rastrigin(solution, dim);
    }
}

/* Returns 1 if solution 1 < solution 2, and returns 0 otherwise.
 */
CUDA_CALLABLE float is_min_cost(int objective, float* solution1, float* solution2, int dim) {
    if (cost(objective, solution1, dim) < cost(objective, solution2, dim)) {
        return 1;
    }
    return 0;
}
// // given all the other inputs. (See how this function is called in blur.cu
// // to get an understanding of what it should do.
// CUDA_CALLABLE void cuda_pso_kernel(float *gpu_solutions, float *gpu_velocities,
//                       float *gpu_g_best, const int num_particles, const int dim,
//                       const float c1, const float c2, const float w);

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
