#ifndef NBODYRPY_INTERFACE_H
#define NBODYRPY_INTERFACE_H

#include <MobilityInterface/container.h>
#include <MobilityInterface/defines.h>
namespace nbody_rpy {
enum class kernel_type { open_rpy, bottom_wall };
using namespace libmobility;

enum class algorithm { fast, naive, block, advise };
void callBatchedNBody(device_span<const real> h_pos,
                      device_span<const real> h_forces,
                      device_span<const real> h_torques, device_span<real> h_MF,
                      device_span<real> h_MT, int Nbatches, int NperBatch,
                      real transMobility, real rotMobility,
                      real transRotMobility, real hydrodynamicRadius,
                      algorithm alg, kernel_type kernel);

} // namespace nbody_rpy
#endif
