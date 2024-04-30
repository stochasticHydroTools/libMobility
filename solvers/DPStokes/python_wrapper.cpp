/*Raul P. Pelaez 2022. Python wrapper for the DPStokes module
 */
#include "mobility.h"
#include <MobilityInterface/pythonify.h>
using DPStokesParameters = uammd_dpstokes::PyParameters;
static const char *docstring = R"pbdoc(
In the Doubly periodic Stokes geometry (DPStokes), an incompressible fluid exists in a domain which is periodic in the plane and open (or walled) in the third direction.

Parameters
----------
dt : float
		The time step.
Lx : float
		The box size in the x direction.
Ly : float
		The box size in the y direction.
zmin : float
		The minimum value of the z coordinate.
zmax : float
		The maximum value of the z coordinate.
)pbdoc";

MOBILITY_PYTHONIFY_WITH_EXTRA_CODE(DPStokes,
                                   solver.def(
                                       "setParameters",
                                       [](DPStokes &self, real dt, real Lx,
                                          real Ly, real zmin, real zmax) {
                                         DPStokesParameters params;
                                         params.dt = dt;
                                         params.Lx = Lx;
                                         params.Ly = Ly;
                                         params.zmin = zmin;
                                         params.zmax = zmax;
                                         params.w = 6;
                                         //	  params.w_d = w_d;
                                         params.beta = 1.714*params.w;
                                         // params.beta_d = beta_d;
                                         params.alpha = params.w/2.0;
                                         // params.alpha_d = alpha_d;
                                         self.setParametersDPStokes(params);
                                       },
                                       "dt"_a, "Lx"_a, "Ly"_a, "zmin"_a,
                                       "zmax"_a);
                                   , docstring);
