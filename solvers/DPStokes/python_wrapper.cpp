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
viscosity : float
		The viscosity of the fluid.
Lx : float
		The box size in the x direction.
Ly : float
		The box size in the y direction.
zmin : float
		The minimum value of the z coordinate.
zmax : float
		The maximum value of the z coordinate.
w : float
		The support of the particle force spreading kernel.
w_d : float
		The support of the dipole force spreading kernel.
hydrodynamicRadius : float
		The hydrodynamic radius of the particle.
beta : float
		The width of the Exponential of the Semicircle (ES) particle force spreading kernel.
beta_d : float
		The width of the Exponential of the Semicircle (ES) particle dipole spreading kernel.
alpha : float
		Distance normalization factor for the particle force spreading kernel.
alpha_d : float
		Distance normalization factor for the dipole force spreading kernel.
mode : str
		The wall mode. Options are 'slit', 'bottom' and 'nowall'.
)pbdoc";

MOBILITY_PYTHONIFY_WITH_EXTRA_CODE(
    DPStokes,
    solver.def(
        "setParameters",
        [](DPStokes &self, real dt, real viscosity, real Lx, real Ly, real zmin,
           real zmax, real w, real w_d, real hydrodynamicRadius, real beta,
           real beta_d, real alpha, real alpha_d, std::string mode) {
	  DPStokesParameters params;
	  params.dt = dt;
	  params.viscosity = viscosity;
	  params.Lx = Lx;
	  params.Ly = Ly;
	  params.zmin = zmin;
	  params.zmax = zmax;
	  params.w = w;
	  params.w_d = w_d;
	  params.hydrodynamicRadius = hydrodynamicRadius;
	  params.beta = beta;
	  params.beta_d = beta_d;
	  params.alpha = alpha;
	  params.alpha_d = alpha_d;
	  params.mode = mode;
          self.setParametersDPStokes(params);
        },
	"dt"_a, "viscosity"_a, "Lx"_a, "Ly"_a, "zmin"_a, "zmax"_a, "w"_a,
	"w_d"_a, "hydrodynamicRadius"_a, "beta"_a, "beta_d"_a, "alpha"_a,
	"alpha_d"_a, "mode"_a);,
    docstring);
