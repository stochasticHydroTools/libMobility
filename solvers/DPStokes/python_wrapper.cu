/*Raul P. Pelaez 2022. Python wrapper for the DPStokes module
 */
#include "mobility.h"
#include <MobilityInterface/pythonify.h>
using DPStokesParameters = uammd_dpstokes::PyParameters;
static const char *setparameters_docstring = R"pbdoc(

When the periodicity is set to :code:`single_wall` a wall in the bottom of the domain is added.
When the periodicity is set to :code:`two_walls` a wall in the bottom and top of the domain is added.

Even in open mode (Z periodicity set to `open`) the values of :code:`zmin` and :code:`zmax` are still required. The algorithm needs to define a grid in the z direction, and these values define the extents of that grid. The code will fail if a position outside of these extents is used.

Parameters
----------
Lx : float
		The box size in the x direction.
Ly : float
		The box size in the y direction.
zmin : float
		The minimum value of the z coordinate. This is the position of the bottom wall if the Z periodicity is `single_wall` or `two_walls`.
zmax : float
		The maximum value of the z coordinate. This is the position of the top wall if the Z periodicity is `two_walls`.
allowChangingBoxSize : bool
    Whether the periodic extents Lx & Ly can be modified during parameter selection. Default: false.
delta : float
    The finite difference step size for random finite differences. Specified in units of hydrodynamicRadius. Default is 1e-3.
)pbdoc";

static const char *docstring = R"pbdoc(
In the Doubly periodic Stokes geometry (DPStokes), an incompressible fluid exists in a domain which is periodic in the plane and open (or walled) in the third direction. The algorithm is described in [1].

The periodicity must be set to `periodic` in the X and Y directions. The Z periodicity can be set to `open`, `single_wall`, or `two_walls`. The `open` option allows for an open boundary condition in the Z direction, while `single_wall` and `two_walls` add walls at the bottom and/or top of the simulation box.

**References**

[1] Aref Hashemi, Raúl P. Peláez, Sachin Natesh, Brennan Sprinkle, Ondrej Maxian, Zecheng Gan, Aleksandar Donev; Computing hydrodynamic interactions in confined doubly periodic geometries in linear time. J. Chem. Phys. 21 April 2023; 158 (15): 154101. https://doi.org/10.1063/5.0141371
)pbdoc";

MOBILITY_PYTHONIFY_WITH_EXTRA_CODE(
    DPStokes,
    solver.def(
        "setParameters",
        [](DPStokes &self, real Lx, real Ly, real zmin, real zmax,
           bool allowChangingBoxSize, real delta) {
          DPStokesParameters params;
          params.Lx = Lx;
          params.Ly = Ly;
          params.zmin = zmin;
          params.zmax = zmax;
          params.allowChangingBoxSize = allowChangingBoxSize;
          params.delta = delta;
          self.setParametersDPStokes(params);
        },
        "Lx"_a, "Ly"_a, "zmin"_a, "zmax"_a, "allowChangingBoxSize"_a = false,
        "delta"_a = 1e-3, setparameters_docstring);
    , docstring);
