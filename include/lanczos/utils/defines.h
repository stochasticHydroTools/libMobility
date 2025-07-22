#ifndef LANCZOS_DEFINES_H
#define LANCZOS_DEFINES_H
#ifndef DOUBLE_PRECISION
#define SINGLE_PRECISION
#endif

namespace lanczos{
#ifndef DOUBLE_PRECISION
  using real = float;
#else
  using real = double;
#endif
}
#endif
