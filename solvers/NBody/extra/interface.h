#ifndef NBODYRPY_INTERFACE_H
#define NBODYRPY_INTERFACE_H

#include <MobilityInterface/defines.h>

namespace nbody_rpy{
  using  real  = libmobility::real;

  enum class algorithm{fast, naive, block, advise};
  void callBatchedNBodyOpenBoundaryRPY(const real* h_pos, const real* h_forces, const real* h_torques,
				       real* h_MF, real* h_MT, int Nbatches, int NperBatch,
				       real selfMobility, real hydrodynamicRadius, algorithm alg);

  void callBatchedNBodyBottomWallRPY(const real* h_pos, const real* h_forces, const real* h_torques,
             real* h_MF, real* h_MT, int Nbatches, int NperBatch,
				     real selfMobility, real hydrodynamicRadius, algorithm alg);


}
#endif
