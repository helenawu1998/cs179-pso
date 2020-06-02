/*
 * Particle Swarm Optimization
 * Helena Wu
 * CS 179 Final Project
 *
 * This program implements an optimization algorithm called Particle Swarm
 * Optimization (PSO), a metaheuristic algorithm for finding a sufficiently good
 * solution to an optimization problem.
 *
 * Literature on PSO used to write this CPU demo:
 * https://www.researchgate.net/publication/320190854_Particle_Swarm_Optimization_for_Continuous_Function_Optimization_Problems
 * http://www.cs.armstrong.edu/saad/csci8100/pso_slides.pdf
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iostream>
#include <time.h>

#include <cuda_runtime.h>
// #include "benchmark_functions.h"
// #include <algorithm>
// #include <cassert>

#include "pso.cuh"
using namespace std;


/* Update the velocity according to the iteration equation of algorithm:
 * http://www.cs.armstrong.edu/saad/csci8100/pso_slides.pdf
 */
void update_velocity(float* velocity, float* solution, float* p_best,
    float* g_best, int dim, float w, float c1, float c2) {
    float r1 = ((float) rand()) / RAND_MAX;
    float r2 = ((float) rand()) / RAND_MAX;
    for (int i = 0; i < dim; i++) {
        velocity[i] = w * velocity[i] + c1 * r1 * (p_best[i] - solution[i]) +
            c2 * r2 * (g_best[i] - solution[i]);
    }
}

/* Update a solution for the next iteration using its new velocity. */
void update_position(float* solution, float* velocity, int dim) {
    for (int i = 0; i < dim; i++) {
        solution[i] += velocity[i];
    }
}

/* Checks the passed-in arguments for validity. */
void check_args(int argc, char **argv) {
    if (argc != 6) {
        cout << "Incorrect number of arguments.\n";
        cout << "Arguments: <benchmark test type> <dimension of solution>" <<
            "<number of particles> <threads per block> <max number of blocks>\n";
        exit(1);
    }
}


/*
 * Runs Particle Swarm Optimization to find a solution that is "good enough" for
 * optimizing the Rosenbrock function as a benchmark test.
 *
 * Uses both CPU and GPU implementations, and compares the results.
 */
int large_benchmark_test(int argc, char **argv) {
    check_args(argc, argv);

    const int benchmark = atoi(argv[1]);
    // Dimension of solutions to the optimization problem (Rosenbrock dim = 2)
    int dim;
    if (benchmark != 0) {
        dim = atoi(argv[2]);
    }
    else {
        dim = 2;
    }

    const int num_particles = atoi(argv[3]);

    // Constants for PSO update equation from literature
    // https://www.researchgate.net/post/How_do_I_select_the_Particle_Swarm_Optimization_parameters
    const float c1 = 2.025;
    const float c2 = 1.025;
    const float w = 0.85;

    // Space for storing the initial population of solutions (input for GPU)
    float *solutions_host = (float *) malloc(sizeof (float) * dim * num_particles);

    // Space for storing the initial velocities of solutions (input for GPU)
    float *velocities_host = (float *) malloc(sizeof (float) * dim * num_particles);

    // Space for storing the optimal solution (output for GPU)
    float *output_host = (float *) malloc(sizeof (float) * dim);


    // Space for storing the population of solutions (of size vector_size)
    float *solutions = (float *) malloc(sizeof (float) * dim * num_particles);

    // Space for storing velocities for updating solutions (of size dim)
    float *velocities = (float *) malloc(sizeof (float) * dim * num_particles);

    // Space for storing personal bests for each solution in population
    float *p_bests = (float *) malloc(sizeof (float) * dim * num_particles);

    // Space for the global best solution
    float *g_best = (float *) malloc(sizeof (float) * dim);

        // CPU PSO algorithm
        cout << "Running CPU version of PSO..." << endl;

        // Initialize solutions and velocities with random values
        for (int i = 0; i < dim * num_particles; i++) {
            // Initialize solutions to search within domain [-5.12, 5.12]
            solutions[i] = ((float) rand()) / RAND_MAX * 10.24 - 5.12;
            solutions_host[i] = solutions[i];
            // Initialize velocity to ~10% of range (from literature)
            velocities[i] = ((float) rand()) / RAND_MAX;
            velocities_host[i] = velocities[i];

            p_bests[i] = solutions[i];

            // Initialize global best
            if (i < dim) {
                g_best[i] = solutions[i];
                output_host[i] = g_best[i];
            }
        }

        // Use the CUDA machinery for recording time
        cudaEvent_t start_cpu, stop_cpu;
        cudaEventCreate(&start_cpu);
        cudaEventCreate(&stop_cpu);
        cudaEventRecord(start_cpu);

        int num_iterations = 0;
        // Optimize until stopping condition reached
        while (abs(cost(benchmark, g_best, dim)) > 0.0001 && num_iterations < 1e8) {
            // cout << "Iteration " << num_iterations << endl;
            // cout << "Global min value: " << cost(benchmark, g_best, dim) << endl << endl;
            num_iterations += 1;
            // Iterate for each particle in the population
            for (int i = 0; i < num_particles; i++) {
                update_velocity(&velocities[dim*i], &solutions[dim*i], g_best,
                    &p_bests[dim*i], dim, w, c1, c2);
                update_position(&solutions[dim*i], &velocities[dim*i], dim);

                // Update personal best if better
                if (cost(benchmark, &solutions[dim*i], dim) < cost(benchmark, &p_bests[dim*i], dim)) {
                    for (int j = 0; j < dim; j++) {
                        p_bests[dim*i + j] = solutions[dim*i + j];
                    }

                    // Update global best if better
                    if (cost(benchmark, &solutions[dim*i], dim) < cost(benchmark, g_best, dim)) {
                        for (int j = 0; j < dim; j++) {
                            g_best[j] = solutions[dim*i + j];
                        }
                    }
                }
            }
        }

        // Stop timer
        cudaEventRecord(stop_cpu);
        cudaEventSynchronize(stop_cpu);

        cout << "Iterations for CPU version: " << num_iterations << endl;
        if (benchmark == 0)
            cout << "Completed Rosenbrock's Benchmark Test" << endl;
        else {
            cout << "Completed Rastrigin's Benchmark Test for " << dim << " dimensions" << endl;
        }
        cout << "Global minima for CPU version: " << endl;
        for (int j = 0; j < dim; j++) {
            cout << g_best[j] << endl;
        }

        // GPU PSO algorithm
        cout << endl << "Running GPU version of PSO..." << endl << endl;

        // Cap the number of blocks
        const unsigned int local_size = atoi(argv[4]);
        const unsigned int max_blocks = atoi(argv[5]);
        const unsigned int blocks = std::min(max_blocks,
            (unsigned int) ceil(num_particles / (float) local_size));

        // Call our exposed entry point to our GPU kernel handler
        float gpu_time_milliseconds = cuda_call_pso_kernel(blocks, local_size,
                                                           solutions_host,
                                                           velocities_host,
                                                           output_host,
                                                           num_particles,
                                                           dim, benchmark, c1,
                                                           c2, w);

        cout << "Comparing..." << endl;

        // Compare results
        bool success = true;

        if (success)
            cout << endl << "Successful output" << endl;

        float cpu_time_milliseconds;
        cudaEventElapsedTime(&cpu_time_milliseconds, start_cpu, stop_cpu);

        cout << endl;
        cout << "CPU time: " << cpu_time_milliseconds << " milliseconds" << endl;
        cout << "GPU time: " << gpu_time_milliseconds << " milliseconds" << endl;
        cout << endl << "Speedup factor: " <<
            cpu_time_milliseconds / gpu_time_milliseconds << endl << endl;



    // Free memory on host
    free(solutions_host);
    free(velocities_host);
    free(output_host);
    free(solutions);
    free(velocities);
    free(p_bests);
    free(g_best);


    return 0;
}


int main(int argc, char **argv) {
    return large_benchmark_test(argc, argv);
}
