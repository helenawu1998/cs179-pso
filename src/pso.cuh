/*
 * CUDA Particle Swarm Optimization
 * Helena Wu
 * CS 179 Final Project
 */


#ifndef PSO_DEVICE_CUH
#define PSO_DEVICE_CUH

#include <cstdio>
#include "cuda_header.cuh"

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

// // This function will conduct the convolution for a particular thread index
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
                           float *in_solutions, float *in_velocities
                           float *out_data, const unsigned int num_particles
                           const unsigned int dim, const int benchmark,
                           const float c1, const float c2, const float w);

#endif
