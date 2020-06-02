# CS 179 Final Project -- Particle Swarm Optimization (PSO)
Helena Wu 

## Motivation
Inspired by social organisms in nature, PSO is a metaheuristic algorithm that uses a population of potential solutions (“particle swarm”) to explore the search-space for for an optimization problem. PSO could benefit from GPU acceleration because many of its computations could be computed in parallel, which may allow faster exploration of large search spaces.

## PSO Algorithm Overview
Particle swarm optimization is a population-based approach for solving optimization problems. The objective function f(X) of an optimization problem determines the cost of a potential solution, which is encoded as a vector X=[x1   x2   x3 …. xn] with dimension `dim`. In PSO, each potential solution (aka particle) in the swarm has its own position vector (Xit) and velocity vector (Vit)  for each iteration t. In each iteration, first update the individual and global best solutions, then update all position and velocity vectors as follows: 

vi(t +1) = wvi(t)+ c1r1[xi(best)(t) − xi(t)]+ c2r2[g(t) − xi(t)]

xi(t +1) = xi(t) + vi(t +1)

Here, w, c1, c2 are constant coefficients, i is the particle index, c1, c2 are constant coefficients, r1, r2 are random values regenerated for every update, xi(best)(t) is individual best solution, and g(t) is the global best. We can stop the PSO algorithm once we reach the stopping condition to ensure that we find a sufficiently good solution that minimizes cost.

For a bounded search space, we can penalize solutions that are outside of the domain with a heavy cost. From these equations, we see that PSO balances exploration in global search with the exploitation of local search to find a good solution. 

The PSO algorithm is a useful metaheuristic algorithm because there are few parameters to tune, and the idea is relatively simple. However, solving complex optimization problems often involves high-dimensional search spaces, requiring a large number of particles to explore the domain with high computational costs. Parallelizing PSO would allow us take advantage of the GPU architecture to greatly accelerate the algorithm. 

## GPU Optimizations and Implementation
I designed and implemented a parallelized PSO algorithm to demonstrate how we can take advantage of GPU architecture. Each thread represents a particle in the swarm (population) and is responsible for computing its position and velocity updates (following the same equation as above). By performing the computations in parallel for a large swarm of particles, this GPU-accelerated version is expected to reach an optimal solution faster than the serialized PSO for the CPU version. 

While the velocity and position updates are fairly straightforward for each particle, the function evaluations are computation-heavy. Thus, we can take advantage of latency hiding to keep the GPU productive during the "waiting time" of memory accesses by performing arithmetic instructions and non-dependent reads in the meantime. 

One challenge was that the velocity update equations require knowing the "global best solution" for each iteration. This requires cooperation across the particles (threads), and inevitably slows down the algorithm. I decided to use shared memory (of size num_threads * dim) and parallel reduction to find the min-cost solution in each block. After synchronizing all the threads in the block, I followed Mark Harris's suggestions with sequential addressing to optimize the parallel reduction for "Optimizing Parallel Reduction in CUDA" so that the threads are efficiently searching for the global best in parallel. Lastly, the thread with threadID 0 will update the global best with shared memory[0] solution if it is the global minimum; while this does cause thread divergence, it is still a better solution than a serial approach for searching for the global best.

## Code Structure 
`benchmark.cpp` contains the benchmark problems for the objective cost function (Rosenbrock's, Rastrigin's) and is_min_cost which compares two solutions to determine the lower-cost solution. Header file at `benchmark.h`.

`pso.cpp` contains the CPU version of the PSO algorithm and calls the GPU version. Arguments are `<benchmark test type> <dimension of solution> <number of particles> <threads per block> <max number of blocks>`, where benchmark test type refers to 0 or 1 for Rosenbrock or Rastrigin respectively. Header file at `pso.h`.

`pso.cu`contains the GPU version of the PSO algorithm, including the set up, kernel invocation, parallel reduction, and objective functions. Header file at `pso.cuh`.

`Makefile` compiles the files needed to run pso. Use the command ```make pso``` to get started.

`demo_2020.sh` is the bash script that compiles using the Makefile and then runs the CPU and GPU versions of PSO on six different test cases for two benchmark functions with various dimensions and particle swarm sizes.

# Testing with Demo Script
The demo bash script runs the Makefile, and tests the PSO algorithm on several benchmark functions, including Rosenbrock's Banana Function (2 dimensions) and Rastrigin's Function for various dimensions. To run the demo script:
```
 chmod +x demo_2020.sh
 ./demo_2020.sh
```

## Results
The GPU run uses the same initial population of solutions and velocities as in the CPU version for fair comparison of the results. We run the demo script for several test cases (varying particle swarm sizes) for two benchmark functions, Rosenbrock's Banana Function (only 2 dimensions) and Rastrigin's Function (higher dimension n). 

The GPU-accelerated version was also able to find an optimal solution for minimizing the objective function for each test case. However, for the lower dimension problems (Rosenbrock's), the CPU PSO implementation was actually faster than the GPU-accelerated version. This can be attributed to the overhead that the GPU implementation requires in allocating the threads and data transfers. For higher dimension search spaces, we see that the GPU-accelerated implementation of PSO finds an optimal solution much faster, with up to 7 times speedup factor.

Although my results show that utilizing the GPU does accelerate the PSO algorithm for large search spaces and complex problems, I believe I could have tried other techniques to get more significant performance improvements given more time. For example, with more time I would try using constant memory for faster accesses to constants, or implementing a thread task pool as mentioned here: https://www.researchgate.net/publication/232886511_Accelerating_parallel_particle_swarm_optimization_via_GPU

### Output
Below is the output after running the demo bash script (demo_2020.sh):

Running CPU version of PSO...
Iterations for CPU version: 145
Completed Rosenbrock's Benchmark Test
Global minima for CPU version: 
0.99153
0.983155

Running GPU version of PSO...

No kernel error detected
Completed Rosenbrock's Benchmark Test
Comparing...

Success: GPU PSO found sufficient global minima.

CPU time: 0.483136 milliseconds
GPU time: 32.9109 milliseconds

Speedup factor: 0.0146801

Running CPU version of PSO...
Iterations for CPU version: 82
Completed Rosenbrock's Benchmark Test
Global minima for CPU version: 
1.00325
1.00704

Running GPU version of PSO...

No kernel error detected
Completed Rosenbrock's Benchmark Test
Comparing...

Success: GPU PSO found sufficient global minima.

CPU time: 0.53536 milliseconds
GPU time: 30.9729 milliseconds

Speedup factor: 0.0172848

Running CPU version of PSO...
Iterations for CPU version: 145
Completed Rastrigin's Benchmark Test for 2 dimensions
Global minima for CPU version: 
8.51937e-05
0.000508833

Running GPU version of PSO...

No kernel error detected
Completed Rastrigin's Benchmark Test for 2 dimensions
Comparing...

Success: GPU PSO found sufficient global minima.

CPU time: 0.880896 milliseconds
GPU time: 7.85626 milliseconds

Speedup factor: 0.112127

Running CPU version of PSO...
Iterations for CPU version: 95
Completed Rastrigin's Benchmark Test for 2 dimensions
Global minima for CPU version: 
-0.000508053
-0.000347927

Running GPU version of PSO...

No kernel error detected
Completed Rastrigin's Benchmark Test for 2 dimensions
Comparing...

Success: GPU PSO found sufficient global minima.

CPU time: 1.1632 milliseconds
GPU time: 21.9694 milliseconds

Speedup factor: 0.0529464

Running CPU version of PSO...
Iterations for CPU version: 366
Completed Rastrigin's Benchmark Test for 3 dimensions
Global minima for CPU version: 
0.00025848
0.00025479
-0.000566229

Running GPU version of PSO...

No kernel error detected
Completed Rastrigin's Benchmark Test for 3 dimensions
Comparing...

Success: GPU PSO found sufficient global minima.

CPU time: 57.4466 milliseconds
GPU time: 8.31293 milliseconds

Speedup factor: 6.91052

Running CPU version of PSO...
Iterations for CPU version: 99
Completed Rastrigin's Benchmark Test for 4 dimensions
Global minima for CPU version: 
-0.000478148
0.000267603
0.000272203
-0.000172861

Running GPU version of PSO...

No kernel error detected
Completed Rastrigin's Benchmark Test for 4 dimensions
Comparing...

Success: GPU PSO found sufficient global minima.

CPU time: 30.8486 milliseconds
GPU time: 8.76029 milliseconds

Speedup factor: 3.52141

