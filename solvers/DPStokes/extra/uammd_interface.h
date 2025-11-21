/*Raul P. Pelaez 2021.
  An interface code between uammd_wrapper.cu and uammd_python.cpp.
*/
#include <memory>
#include <string>
namespace uammd_dpstokes {
// This is in order not to use any UAMMD related includes here.
// Instead of using uammd::real I have to re define real here.
#ifndef DOUBLE_PRECISION
using real = float;
using real3 = float3;
#else
using real = double;
using real3 = double3;
#endif

// This function returns either 'single' or 'double' according to the UAMMD's
// compiled precision.
std::string getPrecision();

struct PyParameters {
  // The number of cells in each direction
  // If -1, they will be autocomputed from the tolerance if possible (DP cannot
  // do it, FCM can)
  int nx = -1;
  int ny = -1;
  int nz = -1;
  real viscosity;
  real Lx;
  real Ly;
  real zmin, zmax;
  // Tolerance will be ignored in DP mode, TP will use only tolerance and nxy/nz
  real tolerance = 1e-5;
  real delta = 1e-3; // RFD step size
  real w, w_d;
  real hydrodynamicRadius = -1;
  real3 beta = {-1.0, -1.0, -1.0};
  real3 beta_d = {-1.0, -1.0, -1.0};
  real alpha = -1;
  real alpha_d = -1;
  // Can be either none, bottom, slit or periodic
  std::string mode;
  bool allowChangingBoxSize = false;
};

class DPStokesUAMMD;
class DPStokesGlue {
  std::shared_ptr<DPStokesUAMMD> dpstokes;

public:
  int numberParticles;

  // Initialize the modules with a certain set of parameters
  // Reinitializes if the module was already initialized
  void initialize(PyParameters pypar);

  // Clears all memory allocated by the module.
  // This leaves the module in an unusable state until initialize is called
  // again.
  void clear();
  // Set positions to compute mobility matrix
  void setPositions(const real *h_pos, int numberParticles);

  const real *getStoredPositions();
  // Compute the dot product of the mobility matrix with the forces and/or
  // torques acting on the previously provided positions
  void Mdot(const real *h_forces, const real *h_torques, real *h_MF, real *h_MT,
            int numberParticles, bool includeAngular);

private:
  void throwIfInvalid();
};

} // namespace uammd_dpstokes
