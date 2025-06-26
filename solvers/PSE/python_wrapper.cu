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
This module computes the RPY mobility in triply periodic boundaries using Ewald splitting with the Positively Split Ewald method [1].


This module will only accept periodic boundary conditions in the three directions.

**References**

[1] Andrew M. Fiore, Florencio Balboa Usabiaga, Aleksandar Donev, James W. Swan; Rapid sampling of stochastic displacements in Brownian dynamics simulations. J. Chem. Phys. 28 March 2017; 146 (12): 124116. https://doi.org/10.1063/1.4978242

)pbdoc";

MOBILITY_PYTHONIFY_WITH_EXTRA_CODE(
    PSE,
    solver.def(
        "setParameters",
        [](PSE &self, real psi, real Lx, real Ly, real Lz, real shearStrain) {
          self.setParametersPSE({psi, Lx, Ly, Lz, shearStrain});
        },
        docstringSetParameters, "psi"_a, "Lx"_a, "Ly"_a, "Lz"_a,
        "shearStrain"_a);
    , docstring);
