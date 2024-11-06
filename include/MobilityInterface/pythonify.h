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

template <class Solver>
auto setup_arrays(Solver &myself, pyarray_c &forces, pyarray_c &torques) {
  int N = myself.getNumberParticles();
  if (forces.size() < 3 * N and forces.size() > 0) {
    throw std::runtime_error("The forces array must have size 3*N.");
  }
  if (torques.size() < 3 * N and torques.size() > 0) {
    throw std::runtime_error("The torques array must have size 3*N.");
  }
  auto f = forces.size() ? cast_to_const_real(forces) : nullptr;
  auto t = torques.size() ? cast_to_const_real(torques) : nullptr;
  auto mf = py::array_t<libmobility::real>();
  auto mt = py::array_t<libmobility::real>();
  mf.resize({3 * N});
  mf.attr("fill")(0);
  if (t) {
    if(!myself.getNeedsTorque()){
      throw std::runtime_error("The was configured without torques. Set needsTorque to true in the constructor if you want to use torques");
    }
    mt.resize({3 * N});
    mt.attr("fill")(0);
  }

  return std::make_tuple(f, t, mf, mt);
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
needsTorque : bool, optional
		Whether the solver needs torques. Default is false.
tolerance : float, optional
		Tolerance, used for approximate methods and also for Lanczos (default fluctuation computation). Default is 1e-4.
)pbdoc";

template <class Solver>
auto call_sqrtMdotW(Solver &solver, libmobility::real prefactor) {
  int N = solver.getNumberParticles();
  auto linear = py::array_t<libmobility::real>({N * 3});
  linear.attr("fill")(0);
  auto angular = py::array_t<libmobility::real>();
  if (solver.getNeedsTorque()) {
    angular.resize({3 * N});
    angular.attr("fill")(0);
    solver.sqrtMdotW(cast_to_real(linear), cast_to_real(angular), prefactor);
    angular = angular.reshape({N, 3});
  } else {
    solver.sqrtMdotW(cast_to_real(linear), nullptr, prefactor);
  }
  linear = linear.reshape({N, 3});
  return std::make_pair(linear, angular);
}

const char *sqrtMdotW_docstring = R"pbdoc(
Computes the stochastic contribution, :math:`\text{prefactor}\sqrt{2T\boldsymbol{\mathcal{M}}}d\boldsymbol{W}`, where :math:`\boldsymbol{\mathcal{M}}` is the grand mobility matrix and :math:`d\boldsymbol{W}` is a Wiener process.

It is required that :py:mod:`setPositions` has been called before calling this function.

Parameters
----------

prefactor : float, optional
		Prefactor to multiply the result by. Default is 1.0.

Returns
-------
array_like
		The resulting linear fluctuations. Shape is (N, 3), where N is the number of particles.
array_like
		The resulting angular fluctuations. Shape is (N, 3), where N is the number of particles.


)pbdoc";

template <class Solver>
auto call_mdot(Solver &myself, pyarray_c &forces, pyarray_c &torques) {
  auto [f, t, mf, mt] = setup_arrays(myself, forces, torques);
  int N = myself.getNumberParticles();
  auto mf_ptr = mf.size() ? cast_to_real(mf) : nullptr;
  auto mt_ptr = mt.size() ? cast_to_real(mt) : nullptr;
  myself.Mdot(f, t, mf_ptr, mt_ptr);
  if (mf_ptr)
    mf = mf.reshape({N, 3});
  if (mt_ptr)
    mt = mt.reshape({N, 3});
  return std::make_pair(mf, mt);
}

const char *mdot_docstring = R"pbdoc(
Computes the product of the Mobility matrix with a group of forces and/or torques, :math:`\boldsymbol{\mathcal{M}}\begin{bmatrix}\boldsymbol{F}\\\boldsymbol{T}\end{bmatrix}`.

It is required that :py:mod:`setPositions` has been called before calling this function.

At least one of the forces or torques must be provided.

Parameters
----------
forces : array_like, optional
		Forces acting on the particles. Must have shape (N, 3), where N is the number of particles.
torques : array_like, optional
		Torques acting on the particles. Must have shape (N, 3), where N is the number of particles. The solver must have been initialized with needsTorque=True.

Returns
-------
array_like
		The linear displacements. The result will have the same format as the forces array.
array_like
		The angular displacements. The result will have the same format as the torques array. This array will be empty if the solver was initialized with needsTorque=False.

)pbdoc";

template <class Solver>
void call_initialize(Solver &myself, libmobility::real T, libmobility::real eta,
                     libmobility::real a, int N, bool needsTorque,
                     libmobility::real tol) {
  libmobility::Parameters par;
  par.temperature = T;
  par.viscosity = eta;
  par.hydrodynamicRadius = {a};
  par.tolerance = tol;
  par.numberParticles = N;
  par.needsTorque = needsTorque;
  myself.initialize(par);
}

template <class Solver> void call_setPositions(Solver &myself, pyarray_c &pos) {
  myself.setPositions(cast_to_const_real(pos));
}


template <class Solver>
auto call_hydrodynamicVelocities(Solver &myself, pyarray_c &forces,
                                 pyarray_c &torques,
                                 libmobility::real prefactor) {
  auto [f, t, mf, mt] = setup_arrays(myself, forces, torques);
  auto mf_ptr = mf.size() ? cast_to_real(mf) : nullptr;
  auto mt_ptr = mt.size() ? cast_to_real(mt) : nullptr;
  myself.hydrodynamicVelocities(f, t, mf_ptr, mt_ptr, prefactor);
  int N = myself.getNumberParticles();
  if (mf_ptr)
    mf = mf.reshape({N, 3});
  if (mt_ptr)
    mt = mt.reshape({N, 3});
  return std::make_pair(mf, mt);
}

const char *hydrodynamicvelocities_docstring = R"pbdoc(
Computes the hydrodynamic (deterministic and stochastic) velocities.

.. math::
        \boldsymbol{\mathcal{M}}\begin{bmatrix}\boldsymbol{F}\\\boldsymbol{T}\end{bmatrix} + \text{prefactor}\sqrt{2T\boldsymbol{\mathcal{M}}}d\boldsymbol{W}

If the forces are omitted only the stochastic part is computed.
If the temperature is zero the stochastic part is omitted.
Calling this function is equivalent to calling :py:mod:`Mdot` and :py:mod:`sqrtMdotW` in sequence, but in some solvers this can be done more efficiently.

Parameters
----------
forces : array_like, optional
		Forces acting on the particles.
torques : array_like, optional
		Torques acting on the particles. The solver must have been initialized with needsTorque=True.
prefactor : float, optional
		Prefactor to multiply the result by. Default is 1.0.

Returns
-------
array_like
		The resulting linear displacements. Shape is (N, 3), where N is the number of particles.
array_like
		The resulting angular displacements. Shape is (N, 3), where N is the number of particles. This array will be empty if the solver was initialized with needsTorque=False.
)pbdoc";

template <class Solver>
auto call_construct(std::string perx, std::string pery, std::string perz) {
  return std::unique_ptr<Solver>(
      new Solver(createConfiguration(perx, pery, perz)));
}

using real = libmobility::real;
using Parameters = libmobility::Parameters;
using Configuration = libmobility::Configuration;

template <typename MODULENAME>
auto define_module_content(
    py::module &m, const char *name, const char *documentation,
    const std::function<void(py::class_<MODULENAME> &)> &extra_code) {

  auto solver = py::class_<MODULENAME>(m, name, documentation);

  solver
      .def(py::init(&call_construct<MODULENAME>), constructor_docstring,
           "periodicityX"_a, "periodicityY"_a, "periodicityZ"_a)
      .def("initialize", call_initialize<MODULENAME>, initialize_docstring,
           "temperature"_a, "viscosity"_a, "hydrodynamicRadius"_a,
           "numberParticles"_a, "needsTorque"_a = false, "tolerance"_a = 1e-4)
      .def("setPositions", call_setPositions<MODULENAME>,
           "The module will compute the mobility according to this set of "
           "positions.",
           "positions"_a)
      .def("Mdot", call_mdot<MODULENAME>, mdot_docstring,
           "forces"_a = pyarray(), "torques"_a = pyarray())
      .def("sqrtMdotW", call_sqrtMdotW<MODULENAME>, sqrtMdotW_docstring,
           "prefactor"_a = 1.0)
      .def("hydrodynamicVelocities", call_hydrodynamicVelocities<MODULENAME>,
           hydrodynamicvelocities_docstring, "forces"_a = pyarray_c(),
           "torques"_a = pyarray_c(), "prefactor"_a = 1)
      .def("clean", &MODULENAME::clean,
           "Frees any memory allocated by the module.")
      .def_property_readonly_static(
          "precision", [](py::object) { return MODULENAME::precision; },
          R"pbdoc(Compilation precision, a string holding either float or double.)pbdoc");

  extra_code(solver);
  return solver;
}

#define MOBILITY_PYTHONIFY_WITH_EXTRA_CODE(MODULENAME, EXTRA, documentation)   \
  PYBIND11_MODULE(MODULENAME, m) {                                             \
    auto solver = define_module_content<MODULENAME>(                           \
        m, MOBILITYSTR(MODULENAME), documentation,                             \
        [](py::class_<MODULENAME> &solver) { EXTRA });                         \
  }

#define MOBILITY_PYTHONIFY(MODULENAME, documentation)                          \
  MOBILITY_PYTHONIFY_WITH_EXTRA_CODE(MODULENAME, {}, documentation)

#endif
