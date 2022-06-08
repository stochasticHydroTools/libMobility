#include"mobility.h"
#include <MobilityInterface/pythonify.h>


MOBILITY_PYTHONIFY_WITH_EXTRA_CODE(PSE,
				   solver.def("setParametersPSE",
					      &PSE::setParametersPSE,
					      "psi"_a, "Splitting parameter", "Lx"_a, "Ly"_a, "Lz"_a, "shearStrain"_a);,
  "This module computes the RPY mobility in triply periodic boundaries using Ewald splitting in the GPU.");
