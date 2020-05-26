# CS 179: GPU Computing Final Project -- Particle Swarm Optimization (PSO)
Helena Wu 

## Motivation
Inspired by social organisms in nature, PSO is a metaheuristic algorithm that uses a population of potential solutions (“particle swarm”) to explore the search-space for for an optimization problem. PSO could benefit from GPU acceleration because many of its computations could be computed in parallel, which may allow faster exploration of large search spaces.

## PSO Algorithm Overview
Particle swarm optimization is a population-based approach for solving optimization problems. The objective function f(X) of an optimization problem determines the “fitness” of a potential solution, which is encoded as a vector X=[x1   x2   x3 …. xn]. In PSO, each potential solution (aka particle) in the particle swarm has its own position vector (Xit) and velocity vector (Vit)  for each iteration t. In each iteration, first update the individual and global best solutions, then update all position and velocity vectors as follows: 

vi(t +1) = wvi(t)+ c1r1[xi(best)(t) − xi(t)]+ c2r2[g(t) − xi(t)]

xi(t +1) = xi(t) + vi(t +1)

Here, w, c1, c2 are constant coefficients, i is the particle index, c1, c2 are constant coefficients, r1, r2 are random values regenerated for every update, xi(best)(t) is individual best solution, and g(t) is the global best. For a bounded search space, we can penalize solutions that are outside of the domain with a heavy cost. From these equations, we see that PSO balances exploration in global search with the exploitation of local search to find a good solution.  

The PSO algorithm is a useful metaheuristic algorithm because there are few parameters to tune, and the idea is relatively simple. However, solving complex optimization problems often involves high-dimensional search spaces, requiring a large number of particles to explore the domain with high computational costs. Parallelizing PSO would allow us take advantage of the GPU architecture to greatly accelerate the algorithm. 

# Testing
pso.cpp contains the PSO algorithm for CPU. pso.cu is a work in progress for the GPU-accelerated version of PSO.

The demo bash script runs the Makefile, and tests the PSO algorithm on several benchmark functions, including Rosenbrock's Banana Function (2 dimensions) and Rastrigin's Function for various dimensions. To run the CPU demo script:
```
 chmod +x cpu_demo_2020.sh
 ./cpu_demo_2020.sh
```

