#ifndef NBODYRPY_INTERFACE_H
#define NBODYRPY_INTERFACE_H

#include "memory/container.h"
#include "MobilityInterface/defines.h"
namespace nbody_rpy {
enum class kernel_type { open_rpy, bottom_wall };
using namespace libmobility;

enum class algorithm { fast, naive, block, advise };
void callBatchedNBody(device_span<const real> pos,
                      device_span<const real> forces,
                      device_span<const real> torques, device_span<real> MF,
                      device_span<real> MT, int Nbatches, int NperBatch,
                      real transMobility, real rotMobility,
                      real transRotMobility, real hydrodynamicRadius, bool needsTorque,
                      algorithm alg, kernel_type kernel);

} // namespace nbody_rpy
#endif
