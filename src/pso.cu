/*
 * CUDA Particle Swarm Optimization
 * Helena Wu
 * CS 179 Final Project
 */

#include "pso.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iostream>
#include <time.h>

#include <cuda_runtime.h>
#include "benchmark_functions.h"
#include "cuda_header.cuh"

/*
Atomic-min function. Update address if it is the min.

Source:
http://stackoverflow.com/questions/17399119/
cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
*/
// __device__ static float atomicMinCost(float* address, float val)
// {
//     int* address_as_i = (int*) address;
//     int old = *address_as_i, assumed;
//     do {
//         assumed = old;
//         old = ::atomicCAS(address_as_i, assumed,
//             __float_as_int(::fmaxf(val, __int_as_float(assumed))));
//     } while (assumed != old);
//     return __int_as_float(old);
// }


// CUDA_CALLABLE
// void cuda_pso_kernel_convolution(uint thread_index, const float* gpu_raw_data,
//                                   const float* gpu_blur_v, float* gpu_out_data,
//                                   const unsigned int n_frames,
//                                   const unsigned int blur_v_size) {
//     for (int j = 0; j < blur_v_size && j <= thread_index; j++) {
//         gpu_out_data[thread_index] += gpu_raw_data[thread_index - j] *
//             gpu_blur_v[j];
//     }
// }

// This function will run the PSO algorithm until the convergence condition is
// reached. Each thread represents one particle of the swarm and will do all
// computations for its position and velocity updates.
__global__
void cuda_pso_kernel(float *gpu_solutions, float *gpu_velocities,
                      float *gpu_p_best, float *gpu_g_best,
                      const int num_particles, const int dim,
                      const int benchmark, const float c1,
                      const float c2, const float w) {

    extern __shared__ float sdata[];
    //       Compute the current thread index.
    uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    uint tid = threadIdx.x;
    //      While loop handles all indices
    while (thread_index < num_particles) {
        //  First initialize the personal bests with input solutions
        gpu_p_best[thread_index] = gpu_solutions[thread_index];
        //  Update the thread index
        thread_index += blockDim.x * gridDim.x;
    }

    float r1;
    float r2;
    float temp;

    thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    //  Repeat until stopping criteria is satisfied
    while (abs(cost(benchmark, gpu_g_best, dim)) > 0.0001) {

        r1 = ((float) rand()) / RAND_MAX;
        r2 = ((float) rand()) / RAND_MAX;
        // Latency hiding with arithmetic operations and memory accesses
        for (int dim_idx = 0; dim_idx < dim; dim_idx++) {
            uint i = thread_index * dim + dim_idx;

            // Update particle's velocity
            temp = 0;
            temp += w * gpu_velocities[i];
            temp += c1 * r1 * (gpu_p_best[i] - gpu_solutions[i]);
            temp += c2 * r2 * (gpu_g_best[i] - gpu_solutions[i]);
            gpu_velocities[i] = temp;

            // Update particle's position
            gpu_solutions[i] += temp;
            // Store solutions for this block in shared memory
            sdata[tid * dim + dim_idx] = gpu_solutions[i];
        }
        // Compute cost of updated solution
        float solution_cost = cost(benchmark, &gpu_solutions[thread_index], dim);
        // Update personal best if better
        if (solution_cost < cost(benchmark, &gpu_p_best[thread_index], dim)) {
            for (int i = 0; i < dim; i++) {
                gpu_p_best[thread_index + i] = gpu_solutions[thread_index + i];
            }
        }

        // Synchronize so all threads in block have completed the iteration and
        // stored it in shared memory
        __syncthreads();

        // Each thread does reduction in shared mem
        // Reference: "Optimizing Parallel Reduction in CUDA" by Mark Harris
        for (uint s = blockDim.x/2; s > 0; s >>= 1){
            if (tid < s){
                if (is_min_cost(benchmark, &sdata[tid*dim + s*dim], &sdata[tid*dim], dim)) {
                    // Copy solution elements to smaller index for reduction
                    for (int i = 0; i < dim; i++) {
                        sdata[tid*dim +i] = sdata[tid*dim + s*dim + i];
                    }
                }
            }
            // Synchronize threads between steps
            __syncthreads();
        }

        // Write result to gpu_g_best if minimum
        if (tid == 0 && is_min_cost(benchmark, &sdata[0], gpu_g_best, dim)) {
            for (int i = 0; i < dim; i++) {
                gpu_g_best[i] = sdata[i];
            }
        }
    }
}

// This function will be in charge of allocating GPU memory, invoking the kernel
// for running the PSO algorithm, and cleaning up afterwards. The result
// will be stored in out_data. The function returns the amount of time that
// it took for the function to complete (prior to returning) in milliseconds.
float cuda_call_pso_kernel(const unsigned int blocks,
                           const unsigned int threads_per_block,
                           float *in_solutions, float *in_velocities,
                           float *out_data, const unsigned int num_particles,
                           const unsigned int dim, const int benchmark,
                           const float c1, const float c2, const float w) {
    //       Use the CUDA machinery for recording time
    cudaEvent_t start_gpu, stop_gpu;
    float time_milli = -1;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    cudaEventRecord(start_gpu);

    //       Allocate GPU memory for the raw input data (randomly generated
    //       solutions, velocities in the initial population).
    //       The data is of type float and has dim * num_particles elements.
    //       We copy the input data into the allocated GPU memory.
    float* gpu_solutions;
    gpu_errchk(cudaMalloc((void **) &gpu_solutions,
        dim * num_particles * sizeof(float)));
    gpu_errchk(cudaMemcpy(gpu_solutions, in_solutions,
        dim * num_particles * sizeof(float), cudaMemcpyHostToDevice));

    float* gpu_velocities;
    gpu_errchk(cudaMalloc((void **) &gpu_velocities,
        dim * num_particles * sizeof(float)));
    gpu_errchk(cudaMemcpy(gpu_velocities, in_velocities,
        dim * num_particles * sizeof(float), cudaMemcpyHostToDevice));

    //       Allocate GPU memory to store the personal best solution for each
    //       particle in the swarm.
    float* gpu_p_best;
    gpu_errchk(cudaMalloc((void **) &gpu_p_best,
        dim * num_particles * sizeof(float)));

            //       Allocate GPU memory to store the global best solution, with dim
    //       number of elements of type float.
    float* gpu_g_best;
    gpu_errchk(cudaMalloc((void **) &gpu_g_best, dim * sizeof(float)));
    //      Initialize the gpu_g_best with first solution in population, as
    //      currently stored in out_data.
    gpu_errchk(cudaMemcpy(gpu_g_best, out_data,
        dim * sizeof(float), cudaMemcpyHostToDevice));

    //      Call the kernel function.
    cuda_pso_kernel<<<blocks, threads_per_block, threads_per_block * dim * sizeof(float)>>>
        (gpu_solutions, gpu_velocities, gpu_p_best, gpu_g_best, num_particles, dim,
        benchmark, c1, c2, w);

    //      Check for errors on kernel call
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err)
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
    else
        fprintf(stderr, "No kernel error detected\n");

    //       Now that kernel calls have finished, copy the output signal
    //       back from the GPU to host memory.
    gpu_errchk(cudaMemcpy(out_data, gpu_g_best, dim * sizeof(float),
        cudaMemcpyDeviceToHost));

    //       Now that we have finished our computations on the GPU, free the
    //       GPU resources.
    gpu_errchk(cudaFree(gpu_solutions));
    gpu_errchk(cudaFree(gpu_velocities));
    gpu_errchk(cudaFree(gpu_g_best));

    // Stop the recording timer and return the computation time
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&time_milli, start_gpu, stop_gpu);
    return time_milli;
}
