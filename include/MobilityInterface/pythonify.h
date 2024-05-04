/*Raul P. Pelaez 2021.
The MOBILITY_PYTHONIFY(className, description) macro creates a pybind11 module
from a class (called className) that inherits from libmobility::Mobility.
"description" is a string that will be printed when calling help(className) from
python (accompanied by the default documentation of the mobility interface.
 */
#include <stdexcept>
#ifndef MOBILITY_PYTHONIFY_H
#include "MobilityInterface.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;
using namespace pybind11::literals;
using pyarray = py::array;
using pyarray_c =
    py::array_t<libmobility::real, py::array::c_style | py::array::forcecast>;

#define MOBILITYSTR(s) xMOBILITYSTR(s)
#define xMOBILITYSTR(s) #s

inline auto string2Periodicity(std::string per) {
  using libmobility::periodicity_mode;
  if (per == "open")
    return periodicity_mode::open;
  else if (per == "unspecified")
    return periodicity_mode::unspecified;
  else if (per == "single_wall")
    return periodicity_mode::single_wall;
  else if (per == "two_walls")
    return periodicity_mode::two_walls;
  else if (per == "periodic")
    return periodicity_mode::periodic;
  else
    throw std::runtime_error("[libMobility] Invalid periodicity");
}

inline auto createConfiguration(std::string perx, std::string pery,
                                std::string perz) {
  libmobility::Configuration conf;
  conf.periodicityX = string2Periodicity(perx);
  conf.periodicityY = string2Periodicity(pery);
  conf.periodicityZ = string2Periodicity(perz);
  return conf;
}

template <typename T, class Array> void check_dtype(Array &arr) {
  if (not py::isinstance<py::array_t<T>>(arr)) {
    throw std::runtime_error("Input array must have the correct data type.");
  }
  if (not py::isinstance<
          py::array_t<T, py::array::c_style | py::array::forcecast>>(arr)) {
    throw std::runtime_error(
        "The input array is not contiguous and cannot be used as a buffer.");
  }
}

template <class Array> libmobility::real *cast_to_real(Array &arr) {
  check_dtype<libmobility::real>(arr);
  return static_cast<libmobility::real *>(arr.mutable_data());
}

template <class Array> const libmobility::real *cast_to_const_real(Array &arr) {
  check_dtype<const libmobility::real>(arr);
  return static_cast<const libmobility::real *>(arr.data());
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
tolerance : float, optional
		Tolerance, used for approximate methods and also for Lanczos (default fluctuation computation). Default is 1e-4.
)pbdoc";

template <class Solver>
auto call_sqrtMdotW(Solver &solver, libmobility::real prefactor) {
  auto result =
      py::array_t<libmobility::real>({solver.getNumberParticles() * 3});
  solver.sqrtMdotW(cast_to_real(result), prefactor);
  return result.reshape({solver.getNumberParticles(), 3});
}

const char *sqrtMdotW_docstring = R"pbdoc(
Computes the stochastic contribution, :math:`\text{prefactor}\sqrt{2T\boldsymbol{\mathcal{M}}}d\boldsymbol{W}`, where :math:`\boldsymbol{\mathcal{M}}` is the mobility matrix and :math:`d\boldsymbol{W}` is a Wiener process.

It is required that :py:mod:`setPositions` has been called before calling this function.

Parameters
----------

prefactor : float, optional
		Prefactor to multiply the result by. Default is 1.0.

Returns
-------
array_like
		The resulting fluctuations. Shape is (N, 3), where N is the number of particles.

)pbdoc";

template <class Solver> auto call_mdot(Solver &myself, pyarray_c &forces) {
  int N = myself.getNumberParticles();
  if (forces.size() < 3 * N and forces.size() > 0) {
    throw std::runtime_error("The forces array must have size 3*N.");
  }
  auto f = forces.size() ? cast_to_const_real(forces) : nullptr;
  auto result =
      py::array_t<libmobility::real>(py::array::ShapeContainer({3 * N}));
  result.attr("fill")(0);
  myself.Mdot(f, cast_to_real(result));
  return result.reshape({N, 3});
}

const char *mdot_docstring = R"pbdoc(
Computes the product of the Mobility matrix with a group of forces, :math:`\boldsymbol{\mathcal{M}}\boldsymbol{F}`.

It is required that :py:mod:`setPositions` has been called before calling this function.

Parameters
----------
forces : array_like,
		Forces acting on the particles. Must have shape (N, 3), where N is the number of particles.

Returns
-------
array_like
		The result of the product. The result will have the same format as the forces array.
)pbdoc";

template <class Solver>
void call_initialize(Solver &myself, libmobility::real T, libmobility::real eta,
                     libmobility::real a, int N, libmobility::real tol) {
  libmobility::Parameters par;
  par.temperature = T;
  par.viscosity = eta;
  par.hydrodynamicRadius = {a};
  par.tolerance = tol;
  par.numberParticles = N;
  myself.initialize(par);
}

template <class Solver> void call_setPositions(Solver &myself, pyarray_c &pos) {
  myself.setPositions(cast_to_const_real(pos));
}

template <class Solver>
auto call_hydrodynamicVelocities(Solver &myself, pyarray_c &forces,
                                 libmobility::real prefactor) {
  int N = myself.getNumberParticles();
  if (forces.size() < 3 * N and forces.size() > 0) {
    throw std::runtime_error("The forces array must have size 3*N.");
  }
  auto f = forces.size() ? cast_to_const_real(forces) : nullptr;
  auto result = py::array_t<libmobility::real>({3 * N});
  result.attr("fill")(0);
  myself.hydrodynamicVelocities(f, cast_to_real(result), prefactor);
  return result.reshape({N, 3});
}

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

prefactor : float, optional
		Prefactor to multiply the result by. Default is 1.0.

Returns
-------
array_like
		The resulting velocities. Shape is (N, 3), where N is the number of particles.
)pbdoc";

template <class Solver>
auto call_construct(std::string perx, std::string pery, std::string perz) {
  return std::unique_ptr<Solver>(
      new Solver(createConfiguration(perx, pery, perz)));
}

#define xMOBILITY_PYTHONIFY(MODULENAME, EXTRACODE, documentation)                            \
  PYBIND11_MODULE(MODULENAME, m) {                                                           \
    using real = libmobility::real;                                                          \
    using Parameters = libmobility::Parameters;                                              \
    using Configuration = libmobility::Configuration;                                        \
    auto solver =                                                                            \
        py::class_<MODULENAME>(m, MOBILITYSTR(MODULENAME), documentation);                   \
    solver                                                                                   \
        .def(py::init(&call_construct<MODULENAME>), constructor_docstring,                   \
             "periodicityX"_a, "periodicityY"_a, "periodicityZ"_a)                           \
        .def("initialize", call_initialize<MODULENAME>, initialize_docstring,                \
             "temperature"_a, "viscosity"_a, "hydrodynamicRadius"_a,                         \
             "numberParticles"_a, "tolerance"_a = 1e-4)                                      \
        .def("setPositions", call_setPositions<MODULENAME>,                                  \
             "The module will compute the mobility according to this set of "                \
             "positions.",                                                                   \
             "positions"_a)                                                                  \
        .def("Mdot", call_mdot<MODULENAME>, mdot_docstring,                                  \
             "forces"_a = pyarray())                                                         \
        .def("sqrtMdotW", call_sqrtMdotW<MODULENAME>, sqrtMdotW_docstring,                   \
             "prefactor"_a = 1.0)                                                            \
        .def("hydrodynamicVelocities",                                                       \
             call_hydrodynamicVelocities<MODULENAME>,                                        \
             hydrodynamicvelocities_docstring, "forces"_a = pyarray_c(),                     \
             "prefactor"_a = 1)                                                              \
        .def("clean", &MODULENAME::clean,                                                    \
             "Frees any memory allocated by the module.")                                    \
        .def_property_readonly_static(                                                       \
            "precision", [](py::object) { return MODULENAME::precision; },                   \
            R"pbdoc(Compilation precision, a string holding either float or double.)pbdoc"); \
    EXTRACODE                                                                                \
  }
#define MOBILITY_PYTHONIFY(MODULENAME, documentationPrelude)                   \
  xMOBILITY_PYTHONIFY(MODULENAME, ;, documentationPrelude)
#define MOBILITY_PYTHONIFY_WITH_EXTRA_CODE(MODULENAME, EXTRA,                  \
                                           documentationPrelude)               \
  xMOBILITY_PYTHONIFY(MODULENAME, EXTRA, documentationPrelude)
#endif
