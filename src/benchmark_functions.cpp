#include <cstdlib>
#include <cmath>
#include "benchmark_functions.h"

using namespace std;
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

/* Returns the value of the objective function, which we are trying to minimize.
 * User defines which objective function to use for benchmark tests.
 */
float cost(int objective, float* solution, int dim) {
    if (objective == 0)
        return rosenbrock(solution);
    else {
        return rastrigin(solution, dim);
    }
}
