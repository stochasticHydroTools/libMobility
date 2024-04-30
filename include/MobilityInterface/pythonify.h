/*Raul P. Pelaez 2021.
The MOBILITY_PYTHONIFY(className, description) macro creates a pybind11 module from a class (called className) that inherits from libmobility::Mobility. "description" is a string that will be printed when calling help(className) from python (accompanied by the default documentation of the mobility interface.
 */
#include <stdexcept>
#ifndef MOBILITY_PYTHONIFY_H
#include"MobilityInterface.h"
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
namespace py = pybind11;
using namespace pybind11::literals;
using pyarray = py::array;

#define MOBILITYSTR(s) xMOBILITYSTR(s)
#define xMOBILITYSTR(s) #s

inline auto string2Periodicity(std::string per){
  using libmobility::periodicity_mode;
  if(per == "open") return periodicity_mode::open;
  else if(per == "unspecified") return periodicity_mode::unspecified;
  else if(per == "single_wall") return periodicity_mode::single_wall;
  else if(per == "two_walls") return periodicity_mode::two_walls;
  else if(per == "periodic") return periodicity_mode::periodic;
  else throw std::runtime_error("[libMobility] Invalid periodicity");
}

inline auto createConfiguration(std::string perx, std::string pery, std::string perz){
  libmobility::Configuration conf;
  conf.periodicityX = string2Periodicity(perx);
  conf.periodicityY = string2Periodicity(pery);
  conf.periodicityZ = string2Periodicity(perz);
  return conf;
}

template<typename T>
void check_dtype(pyarray &arr){
  if(not py::isinstance<py::array_t<T>>(arr)){
    throw std::runtime_error("Input array must have the correct data type.");
  }
  if(not py::isinstance<py::array_t<T, py::array::c_style | py::array::forcecast>>(arr)){
    throw std::runtime_error("The input array is not contiguous and cannot be used as a buffer.");
  }

}

libmobility::real* cast_to_real(pyarray &arr){
  check_dtype<libmobility::real>(arr);
  return static_cast<libmobility::real*>(arr.mutable_data());
}

const libmobility::real* cast_to_const_real(pyarray &arr){
  check_dtype<const libmobility::real>(arr);
  return static_cast<const libmobility::real*>(arr.data());
}
const char *constructor_docstring = R"pbdoc(
Initialize the module with a given set of periodicity conditions.

Each periodicity condition can be one of the following:
	- open: No periodicity in the corresponding direction.
	- unspecified: The periodicity is not specified.
	- single_wall: The system is bounded by a single wall in the corresponding direction.
	- two_walls: The system is bounded by two walls in the corresponding direction.
	- periodic: The system is periodic in the corresponding direction.

Parameters
----------
periodicityX : str
		Periodicity condition in the x direction.
periodicityY : str
		Periodicity condition in the y direction.
periodicityZ : str
		Periodicity condition in the z direction.
)pbdoc";

const char *initialize_docstring = R"pbdoc(
Initialize the module with a given set of parameters.

Parameters
----------
temperature : float
		Temperature of the system in energy units (i.e. kT).
viscosity : float
		Viscosity of the fluid.
hydrodynamicRadius : float
		Hydrodynamic radius of the particles.
numberParticles : int
		Number of particles in the system.
)pbdoc";

const char *mdot_docstring = R"pbdoc(
Computes the product of the Mobility matrix with a group of forces, :math:`\boldsymbol{\mathcal{M}}\boldsymbol{F}`.

It is required that :py:mod:`setPositions` has been called before calling this function.
Both inputs must have precision given by the precision attribute of the module.
Both inputs must have size 3*N, where N is the number of particles.
The arrays are ordered as :code:`[f0x, f0y, f0z, f1x, f1y, f1z, ...]`.

Parameters
----------
forces : array_like,
		Forces acting on the particles.
result : array_like,
		Where the result will be stored.
)pbdoc";

const char *sqrtMdotW_docstring = R"pbdoc(
Computes the stochastic contribution, :math:`\text{prefactor}\sqrt{2T\boldsymbol{\mathcal{M}}}d\boldsymbol{W}`, where :math:`\boldsymbol{\mathcal{M}}` is the mobility matrix and :math:`d\boldsymbol{W}` is a Wiener process.

It is required that :py:mod:`setPositions` has been called before calling this function.
Both inputs must have precision given by the precision attribute of the module.
Both inputs must have size 3*N, where N is the number of particles.
The arrays are ordered as :code:`[f0x, f0y, f0z, f1x, f1y, f1z, ...]`.

Parameters
----------
result : array_like,
		Where the result will be stored. The result will have the same format as the forces array.
prefactor : float, optional
		Prefactor to multiply the result by. Default is 1.0.
)pbdoc";



const char *hydrodynamicvelocities_docstring = R"pbdoc(
Computes the hydrodynamic (deterministic and stochastic) velocities.

.. math::
        \boldsymbol{\mathcal{M}}\boldsymbol{F} + \text{prefactor}\sqrt{2T\boldsymbol{\mathcal{M}}}d\boldsymbol{W}

If the forces are ommited only the stochastic part is computed.
If the temperature is zero the stochastic part is ommited.
Calling this function is equivalent to calling :py:mod:`Mdot` and :py:mod:`sqrtMdotW` in sequence, but in some solvers this can be done more efficiently.

Parameters
----------
forces : array_like, optional
		Forces acting on the particles.
result : array_like
		Where the result will be stored.
prefactor : float, optional
		Prefactor to multiply the result by. Default is 1.0.
)pbdoc";


#define xMOBILITY_PYTHONIFY(MODULENAME, EXTRACODE, documentation)	\
  PYBIND11_MODULE(MODULENAME, m){		      \
  using real = libmobility::real;		      \
  using Parameters = libmobility::Parameters;				\
  using Configuration = libmobility::Configuration;			\
  auto solver = py::class_<MODULENAME>(m, MOBILITYSTR(MODULENAME), documentation); \
  solver.def(py::init([](std::string perx, std::string pery, std::string perz){	\
    return std::unique_ptr<MODULENAME>(new MODULENAME(createConfiguration(perx, pery, perz))); }), \
    constructor_docstring, "periodicityX"_a, "periodicityY"_a, "periodicityZ"_a). \
  def("initialize", [](MODULENAME &myself, real T, real eta, real a, int N){ \
    Parameters par;							\
    par.temperature = T;						\
    par.viscosity = eta;						\
    par.hydrodynamicRadius = {a};					\
    par.numberParticles = N;						\
    myself.initialize(par);						\
  },									\
    initialize_docstring,		\
    "temperature"_a, "viscosity"_a,					\
    "hydrodynamicRadius"_a,						\
    "numberParticles"_a).						\
    def("setPositions", [](MODULENAME &myself, pyarray pos){myself.setPositions(cast_to_const_real(pos));}, \
	"The module will compute the mobility according to this set of positions.", \
	"positions"_a).							\
    def("Mdot", [](MODULENAME &myself, pyarray forces, pyarray result){\
      auto f = forces.size()?cast_to_const_real(forces):nullptr;	\
      myself.Mdot(f, cast_to_real(result));},	\
      mdot_docstring,				\
      "forces"_a = pyarray(), "result"_a).	\
    def("sqrtMdotW", [](MODULENAME &myself, pyarray result, libmobility::real prefactor){ \
      myself.sqrtMdotW(cast_to_real(result), prefactor);},		\
      sqrtMdotW_docstring,						\
      "result"_a = pyarray(), "prefactor"_a = 1.0).			\
    def("hydrodynamicVelocities", [](MODULENAME &myself, pyarray forces,\
					       pyarray result, libmobility::real prefactor){ \
      auto f = forces.size()?cast_to_const_real(forces):nullptr;	\
      myself.hydrodynamicVelocities(f, cast_to_real(result), prefactor);}, \
	hydrodynamicvelocities_docstring,				\
	"forces"_a = pyarray(), "result"_a  = pyarray(), "prefactor"_a = 1). \
    def("clean", &MODULENAME::clean, "Frees any memory allocated by the module."). \
    def_property_readonly_static("precision", [](py::object){return MODULENAME::precision;}, R"pbdoc(Compilation precision, a string holding either float or double.)pbdoc"); \
  EXTRACODE\
}
#define MOBILITY_PYTHONIFY(MODULENAME, documentationPrelude) xMOBILITY_PYTHONIFY(MODULENAME,; ,documentationPrelude)
#define MOBILITY_PYTHONIFY_WITH_EXTRA_CODE(MODULENAME, EXTRA, documentationPrelude) xMOBILITY_PYTHONIFY(MODULENAME,  EXTRA, documentationPrelude)
#endif
