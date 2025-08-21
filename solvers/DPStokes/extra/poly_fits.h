/* Ryker Fish 2024. Helpers to compute values of c^{-1} from [1].

The ES kernel has a function c(beta) that provides a relationship between the
hydrodynamic radius and the kernel parameters alpha and beta. There is no closed
form for c, but we can approximate c and c^{-1} with a polynomial fit.

We have rH = h*w*c(beta), where rH is the hydrodynamic radius, h is the grid
spacing, and w is the width of the kernel. Thus, we can write c^{-1}(rH/(h*w)) =
beta.

References:
[1] Computing hydrodynamic interactions in confined doubly periodic geometries
in linear time. A. Hashemi et al. J. Chem. Phys. 158, 154101 (2023)
https://doi.org/10.1063/5.0141371
 */

#include <vector>

namespace dpstokes_polys {

/* Evaluates a polynomial at x with coefficients in descending order, so the
highest order coefficients are at the start of polyCoeffs. e.g. for a polynomial
of order n, this computes polyCoeffs[n+1] + polyCoeffs[n]*x +
polyCoeffs[n-2]*x^2 + ... + polyCoeffs[0]*x^n
*/
double polyEval(std::vector<double> polyCoeffs, double x) {

  int order = polyCoeffs.size() - 1;
  double accumulator = polyCoeffs[order];
  double current_x = x;
  for (int i = 1; i <= order; i++) {
    accumulator += polyCoeffs[order - i] * current_x;
    current_x *= x;
  }

  return accumulator;
}

// Coefficients for the polynomial fit of c^{-1} from [1]
// the magic numbers come from the original DPStokes repo
// https://github.com/stochasticHydroTools/DoublyPeriodicStokes/blob/main/source/cpu/python/GridAndKernelConfig.py
std::vector<double> cbeta_monopole_inv = {
    4131643418.193291,  -10471683395.26777, 11833009228.6429,
    -7851132955.882548, 3388121732.651829,  -994285251.2185925,
    201183449.7086889,  -27776767.88241613, 2515647.646492857,
    -136305.2970161326, 3445.959503226691};

std::vector<double> cbeta_dipole_inv = {
    39129605.95992675,  -141095001.9380851, 226461386.9399433,
    -213225975.7007014, 130589579.4920038,  -54470021.75408537,
    15723274.21737542,  -3119398.842468358, 411336.473729197,
    -33239.91568304499, 1311.988715157221};
} // namespace dpstokes_polys