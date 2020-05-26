#!/bin/bash
# Runs the CPU demo script for Particle Swarm Optimization algorithm on
# different benchmark problems.
make clean
make pso

# Test Rosenbrock's Benchmark (benchmark test 0) (2 dim), 20 particles
./pso 0 2 20 512 512

# Test Rosenbrock's Benchmark (benchmark test 0) (2 dim), 40 particles
./pso 0 2 40 512 512

# Test Rastrigin's Benchmark (benchmark test 1) (2 dim), 20 particles
./pso 1 2 20 512 512

# Test Rastrigin's Benchmark (benchmark test 1) (2 dim), 40 particles
./pso 1 2 40 512 512

# Test Rastrigin's Benchmark (benchmark test 1) (4 dim), 40 particles
./pso 1 4 40 512 512
