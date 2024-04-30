#include "mobility.h"
#include <MobilityInterface/pythonify.h>

static const char *docstringSetParameters = R"pbdoc(
		Set the parameters for the PSE solver.

		Parameters
		----------
		psi : float
				The Splitting parameter.
		Lx : float
				The box size in the x direction.
		Ly : float
				The box size in the y direction.
		Lz : float
				The box size in the z direction.
		shearStrain : float
				The shear strain.
		)pbdoc";

static const char *docstring = R"pbdoc(
This module computes the RPY mobility in triply periodic boundaries using Ewald splitting with the Positively Split Ewald method.)pbdoc";

MOBILITY_PYTHONIFY_WITH_EXTRA_CODE(
    PSE,
    solver.def(
        "setParametersPSE",
        [](PSE &self, real psi, real Lx, real Ly, real Lz, real shearStrain) {
          self.setParametersPSE({psi, Lx, Ly, Lz, shearStrain});
        },
        docstringSetParameters,
	"psi"_a, "Lx"_a,
        "Ly"_a, "Lz"_a,
        "shearStrain"_a);
    , docstring);
