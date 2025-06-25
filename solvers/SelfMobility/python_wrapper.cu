/*Raul P. Pelaez 2021-2022. Python wrapper for the SelfMobility module
 */
#include "mobility.h"
#include <MobilityInterface/pythonify.h>

MOBILITY_PYTHONIFY_WITH_EXTRA_CODE(
    SelfMobility, solver.def(
                      "setParameters",
                      [](SelfMobility &self, real parameter) {
                        self.setParametersSelfMobility(parameter);
                      },
                      "parameter"_a, "Some example parameter");
    , "This module ignores hydrodynamic interactions, AKA the mobility matrix "
      "is simply (1/(6*pi*eta*a))*I");

// A module that does not use setParametersSolver can pythonify like this
// instead MOBILITY_PYTHONIFY(SelfMobility, "This module ignores hydrodynamic
// interactions, AKA the mobility matrix is simply (1/(6*pi*eta*a))*I");
