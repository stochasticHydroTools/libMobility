/*Raul P. Pelaez 2021-2022. Python wrapper for the SelfMobility module
 */
#include "mobility.h"
#include <MobilityInterface/pythonify.h>

static const char *setparameters_docstring = R"pbdoc(
Fixes the parameters of the SelfMobility module. For this module, this is just an example on how to set parameters and it is ignored.

Parameters
----------
parameter : float
    An example parameter that is not used in this module.
)pbdoc";

static const char *docstring = R"pbdoc(
This module ignores hydrodynamic interactions, i.e. the mobility matrix is :math:`\frac{1}{6\pi\eta a}\mathbb{I}`.
This module will only accept open boundary conditions in the three directions.
)pbdoc";

MOBILITY_PYTHONIFY_WITH_EXTRA_CODE(SelfMobility,
                                   solver.def(
                                       "setParameters",
                                       [](SelfMobility &self, real parameter) {
                                         self.setParametersSelfMobility(
                                             parameter);
                                       },
                                       "parameter"_a, setparameters_docstring);
                                   , docstring)

// A module that does not use setParametersSolver can pythonify like this
// instead MOBILITY_PYTHONIFY(SelfMobility, "This module ignores hydrodynamic
// interactions, AKA the mobility matrix is simply (1/(6*pi*eta*a))*I");
