#ifndef MOBILITYINTERFACEDEFINES_H
#define MOBILITYINTERFACEDEFINES_H
#define LIBMOBILITYVERSION "2.0"
#ifndef DOUBLE_PRECISION
#define SINGLE_PRECISION
#endif
namespace libmobility{
#if defined SINGLE_PRECISION
  using  real  = float;
#else
  using  real  = double;
#endif
}
#endif
