
/*Raul P. Pelaez 2022. Lanczos Algotihm,
  Computes the matrix-vector product sqrt(M)·v using a recursive algorithm.
  For that, it requires a functor in which the () operator takes an output real*
array and an input real* (both device memory) as: inline void operator()(real*
in_v, real * out_Mv); This function must fill "out" with the result of
performing the M·v dot product- > out = M·a_v. If M has size NxN and the cost of
the dot product is O(M). The total cost of the algorithm is O(m·M). Where m <<
N. If M·v performs a dense M-V product, the cost of the algorithm would be
O(m·N^2). References: [1] Krylov subspace methods for computing hydrodynamic
interactions in Brownian dynamics simulations J. Chem. Phys. 137, 064106 (2012);
doi: 10.1063/1.4742347 Some notes:

  From what I have seen, this algorithm converges to an error of ~1e-3 in a few
steps (<5) and from that point a lot of iterations are needed to lower the
error. It usually achieves machine precision in under 50 iterations.

  If the matrix does not have a sqrt (not positive definite, not symmetric...)
it will usually be reflected as a nan in the current error estimation. An
exception will be thrown in this case.
*/

#pragma once

#include "utils/defines.h"
#include "utils/device_blas.h"
#include "utils/device_container.h"
#include <functional>
namespace lanczos {
using Dot = std::function<void(real *, real *)>;
using Callback = std::function<void(int, real)>;
struct Solver {
  Solver();

  int run(Dot &dot, real *Bv, const real *v, real tolerance, int N,
          Callback callback);

  void setIterationHardLimit(int newLimit) {
    this->iterationHardLimit = newLimit;
  }

private:
  real computeError(real *Bz, int N);
  void registerRequiredStepsForConverge(int steps_needed);

  Blas blas;
  device_container<real> oldBz;
  int check_convergence_steps;
  int iterationHardLimit = 200;
};
} // namespace lanczos
#include "LanczosAlgorithm.cu"
