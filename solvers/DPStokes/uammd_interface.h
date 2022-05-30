/*Raul P. Pelaez 2022.
  An interface code between uammd_wrapper.cu and uammd_python.cpp.
 */
#ifndef UAMMD_INTERFACE_H
#define UAMMD_INTERFACE_H
#include<string>
#include<memory>
namespace uammd_dpstokes{
  //This is in order not to use any UAMMD related includes here.
  //Instead of using uammd::real I have to re define real here.
#ifndef DOUBLE_PRECISION
  using real = float;
#else
  using real = double;
#endif

  //This function returns either 'single' or 'double' according to the UAMMD's compiled precision.
  namespace uammd_wrapper{
    std::string getPrecision();
  }

  struct DPStokesParameters{
    real viscosity;
    real hydrodynamicRadius;
    real Lx, Ly, Lz;
    real tolerance;
  };

  class UAMMD_DPStokes;

  class UAMMD_DPStokes_Glue{
    std::shared_ptr<UAMMD_DPStokes> dpstokes;
  public:

    UAMMD_DPStokes_Glue(DPStokesParameters pypar, int numberParticles);

    void Mdot(const real* forces, const real *torques, real* result);

    void clean();
  };
}
#endif
