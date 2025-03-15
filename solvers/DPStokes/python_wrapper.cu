/*Raul P. Pelaez 2022. Python wrapper for the DPStokes module
 */
#include "mobility.h"
#include <MobilityInterface/pythonify.h>
using DPStokesParameters = uammd_dpstokes::PyParameters;
static const char *docstring = R"pbdoc(
In the Doubly periodic Stokes geometry (DPStokes), an incompressible fluid exists in a domain which is periodic in the plane and open (or walled) in the third direction.

When the periodicity is set to :code:`single_wall` a wall in the bottom of the domain is added.

Parameters
----------
Lx : float
		The box size in the x direction.
Ly : float
		The box size in the y direction.
zmin : float
		The minimum value of the z coordinate.
zmax : float
		The maximum value of the z coordinate.
allowChangingBoxSize : bool
    Whether the periodic extents Lx & Ly can be modified during parameter selection. Default: false.
)pbdoc";

MOBILITY_PYTHONIFY_WITH_EXTRA_CODE(DPStokes,
                                   solver.def(
                                       "setParameters",
                                       [](DPStokes &self, real Lx,
                                          real Ly, real zmin, real zmax, bool allowChangingBoxSize)
                                       {
                                           DPStokesParameters params;
                                           params.Lx = Lx;
                                           params.Ly = Ly;
                                           params.zmin = zmin;
                                           params.zmax = zmax;
                                           params.allowChangingBoxSize = allowChangingBoxSize;
                                           self.setParametersDPStokes(params);
                                       },
                                       "Lx"_a, "Ly"_a, "zmin"_a,
                                       "zmax"_a, "allowChangingBoxSize"_a = false);
                                   , docstring);
