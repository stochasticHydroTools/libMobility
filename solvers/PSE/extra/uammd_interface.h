/*Raul P. Pelaez 2021-2022.
  An interface code between uammd_wrapper.cu and uammd_python.cpp.
 */
#ifndef UAMMD_INTERFACE_H
#define UAMMD_INTERFACE_H
#include <memory>
#include <string>
namespace uammd_pse {
// This is in order not to use any UAMMD related includes here.
// Instead of using uammd::real I have to re define real here.
#ifndef DOUBLE_PRECISION
using real = float;
#else
using real = double;
#endif

// This function returns either 'single' or 'double' according to the UAMMD's
// compiled precision.
namespace uammd_wrapper {
std::string getPrecision();
}

struct PyParameters {
  real viscosity;
  real hydrodynamicRadius;
  real Lx, Ly, Lz;
  real tolerance;
  real psi;
  real shearStrain;
};

class UAMMD_PSE;
class UAMMD_PSE_Glue {
  std::shared_ptr<UAMMD_PSE> pse;

public:
  UAMMD_PSE_Glue(PyParameters pypar, int numberParticles);

  void MdotNearField(const real *h_pos, const real *h_F, real *h_MF);

  void MdotFarField(const real *h_pos, const real *h_F, real *h_MF);

  void computeHydrodynamicDisplacements(const real *h_pos, const real *h_F,
                                        real *h_MF, real temperature,
                                        real prefactor);

  void setShearStrain(real newStrain);

  void clean();
};
} // namespace uammd_pse
#endif
