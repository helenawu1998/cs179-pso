#ifndef benchmark_functions
#define benchmark_functions

/* Benchmark optimization problem known in literature as Rosenbrock's function.
 * The minimum of the function is at (1, 1) with value 0. We search in the
 * domain [-5.12, 5.12]^2 as in other benchmark tests.
 */
float rosenbrock(float* solution);

/* Benchmark optimization problem known in literature as the Rastrigin function.
 * The minimum of the function is at (0, 0, 0, 0) with value 0. We search in the
 * domain [-5.12, 5.12]^dim as in other benchmark tests.
 */
float rastrigin(float* solution, int dim);

/* Returns the value of the objective function, which we are trying to minimize.
 * User defines which objective function to use for benchmark tests.
 */
float cost(int objective, float* solution, int dim);

/* Returns 1 if solution 1 < solution 2, and returns 0 otherwise.
 */
float is_min_cost(int objective, float* solution1, float* solution2, int dim);

#endif
