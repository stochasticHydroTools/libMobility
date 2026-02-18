#ifndef MOBILITYINTERFACEDEFINES_H
#define MOBILITYINTERFACEDEFINES_H
#include "cuda_runtime.h"
#define LIBMOBILITYVERSION "3.0"
#ifndef DOUBLE_PRECISION
#define SINGLE_PRECISION
#endif
namespace libmobility {
#if defined SINGLE_PRECISION
using real = float;
#else
using real = double;
#endif

} // namespace libmobility
#endif
