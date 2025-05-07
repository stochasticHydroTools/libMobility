/*Raul P. Pelaez 2021-2025.
The MOBILITY_PYTHONIFY(className, description) macro creates a python module
from a class (called className) that inherits from libmobility::Mobility.
"description" is a string that will be printed when calling help(className) from
python (accompanied by the default documentation of the mobility interface.
 */
#ifndef MOBILITY_PYTHONIFY_H
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>

#include "MobilityInterface/MobilityInterface.h"
#include "memory/python_tensor.h"
namespace nb = nanobind;
using namespace nb::literals;
namespace py = nb;
using pyarray = nb::ndarray<nb::c_contig>;
using pyarray_c = nb::ndarray<libmobility::real, nb::c_contig>;
namespace lp = libmobility::python;
using libmobility::real;
using Parameters = libmobility::Parameters;
using Configuration = libmobility::Configuration;

#define MOBILITYSTR(s) xMOBILITYSTR(s)
#define xMOBILITYSTR(s) #s

static lp::framework last_framework = lp::framework::numpy;
static int last_device = nb::device::cpu::value;
static std::vector<size_t> last_shape;

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

inline libmobility::device get_device(pyarray_c &arr) {
  switch (arr.device_type()) {
  case nb::device::cpu::value:
    return libmobility::device::cpu;
  case nb::device::cuda::value:
    return libmobility::device::cuda;
  default:
    return libmobility::device::unknown;
  }
}
inline libmobility::device_span<real> cast_to_real(pyarray_c &arr) {
  auto dev = get_device(arr);
  return {{arr.data(), arr.size()}, dev};
}

inline libmobility::device_span<const real> cast_to_const_real(pyarray_c &arr) {
  auto dev = get_device(arr);
  return {{arr.data(), arr.size()}, dev};
}

auto check_and_get_shape(pyarray_c &arr) {
  if (arr.size() == 0) {
    return std::vector<size_t>{};
  }
  if (arr.size() % 3 != 0) {
    throw std::runtime_error(
        "[libMobility] The input arrays for positions, forces, and torques "
        "must each have total size 3*N.");
  }
  int N = arr.size() / 3;
  int err = 0;
  if (arr.ndim() == 1 && arr.shape(0) != 3 * N)
    err = 1;
  else if (arr.ndim() == 2 && arr.shape(1) != 3)
    err = 1;
  else if (arr.ndim() != 1 && arr.ndim() != 2)
    err = 1;
  if (err) {
    throw std::runtime_error(
        "[libMobility] The input arrays for positions, forces, and torques "
        "must each have shape (N, 3) or (3*N).");
  }
  std::vector<size_t> shape(arr.shape_ptr(), arr.shape_ptr() + arr.ndim());
  return shape;
}

template <class Solver>
auto setup_arrays(Solver &myself, pyarray_c &forces, pyarray_c &torques) {
  size_t N = myself.getNumberParticles();

  if (forces.size() < 3 * N and forces.size() > 0) {
    throw std::runtime_error("The forces array must have size 3*N.");
  }
  if (torques.size() < 3 * N and torques.size() > 0) {
    throw std::runtime_error("The torques array must have size 3*N.");
  }
  auto f = cast_to_const_real(forces);
  auto t = cast_to_const_real(torques);
  auto framework = lp::get_framework(forces);
  last_framework = framework;
  int device = f.empty() ? torques.device_type() : forces.device_type();
  if (device == nb::device::none::value)
    device = nb::device::cpu::value;
  last_device = device;
  auto f_shape = check_and_get_shape(forces);
  auto mf = lp::create_with_framework<real>(f_shape, device, framework);
  auto mt = nb::ndarray<real, nb::c_contig>();
  if (!t.empty()) {
    if (!myself.getNeedsTorque()) {
      throw std::runtime_error(
          "The solver was configured without torques. Set "
          "needsTorque to true when initializing if you want to use torques");
    }
    auto t_shape = check_and_get_shape(torques);
    mt = lp::create_with_framework<real>(t_shape, device, framework);
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
needsTorque : bool, optional
		Whether the solver needs torques. Default is false.
tolerance : float, optional
		Tolerance, used for approximate methods and also for Lanczos (default fluctuation computation). Default is 1e-4.
)pbdoc";

template <class Solver> auto call_sqrtMdotW(Solver &myself, real prefactor) {
  size_t N = myself.getNumberParticles();

  auto linear =
      lp::create_with_framework<real>(last_shape, last_device, last_framework);
  auto angular = nb::ndarray<real, nb::c_contig>();
  if (myself.getNeedsTorque()) {
    angular = lp::create_with_framework<real>(last_shape, last_device,
                                              last_framework);
    myself.sqrtMdotW(cast_to_real(linear), cast_to_real(angular), prefactor);
  } else {
    auto empty = libmobility::device_span<real>({}, libmobility::device::cpu);
    myself.sqrtMdotW(cast_to_real(linear), empty, prefactor);
  }
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
  auto mf_ptr = cast_to_real(mf);
  auto mt_ptr = cast_to_real(mt);
  myself.Mdot(f, t, mf_ptr, mt_ptr);
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
void call_initialize(Solver &myself, real T, real eta, real a, bool needsTorque,
                     real tol) {
  libmobility::Parameters par;
  par.temperature = T;
  par.viscosity = eta;
  par.hydrodynamicRadius = {a};
  par.tolerance = tol;
  par.needsTorque = needsTorque;
  myself.initialize(par);
}

template <class Solver> void call_setPositions(Solver &myself, pyarray_c &pos) {
  last_shape = check_and_get_shape(pos);
  last_framework = lp::get_framework(pos);
  last_device = pos.device_type();
  myself.setPositions(cast_to_const_real(pos));
}

template <class Solver>
auto call_hydrodynamicVelocities(Solver &myself, pyarray_c &forces,
                                 pyarray_c &torques, real prefactor) {
  auto [f, t, mf, mt] = setup_arrays(myself, forces, torques);

  if (forces.size() == 0) // must check because this can be called without
                          // forces (for sqrtMdotW)
  {
    mf = lp::create_with_framework<real>(last_shape, last_device,
                                         last_framework);
  }
  if (myself.getNeedsTorque() && torques.size() == 0) {
    mt = lp::create_with_framework<real>(last_shape, last_device,
                                         last_framework);
  }

  auto mf_ptr = cast_to_real(mf);
  auto mt_ptr = cast_to_real(mt);
  myself.hydrodynamicVelocities(f, t, mf_ptr, mt_ptr, prefactor);
  int N = myself.getNumberParticles();
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
std::unique_ptr<Solver> call_construct(std::string perx, std::string pery,
                                       std::string perz) {
  return std::make_unique<Solver>(createConfiguration(perx, pery, perz));
}

template <class Solver> auto call_thermalDrift(Solver &solver, real prefactor) {
  const size_t N = solver.getNumberParticles();
  if (N <= 0) {
    throw std::runtime_error(
        "[libMobility] The number of particles is not set. Did you "
        "forget to call setPositions?");
  }
  auto linear =
      lp::create_with_framework<real>(last_shape, last_device, last_framework);
  auto angular = nb::ndarray<real, nb::c_contig>();
  if (solver.getNeedsTorque()) {
    angular = lp::create_with_framework<real>(last_shape, last_device,
                                              last_framework);
  }
  solver.thermalDrift(cast_to_real(linear), cast_to_real(angular), prefactor);
  return std::make_pair(linear, angular);
}

const char *thermaldrift_docstring = R"pbdoc(
Computes the thermal drift, :math:`k_BT\boldsymbol{\partial}_\boldsymbol{x}\cdot \boldsymbol{\mathcal{M}}`.
It is required that :py:mod:`setPositions` has been called before calling this function.
Parameters
----------
prefactor : float, optional
		Prefactor to multiply the result by. Default is 1.0.
Returns
-------
array_like
		The resulting linear displacements. Shape is (N, 3), where N is the number of particles.
array_like
		The resulting angular displacements. Shape is (N, 3), where N is the number of particles. This array will be empty if the solver was initialized with needsTorque=False.
)pbdoc";

template <typename MODULENAME>
auto define_module_content(
    py::module_ &m, const char *name, const char *documentation,
    const std::function<void(py::class_<MODULENAME> &)> &extra_code) {
  auto solver = py::class_<MODULENAME>(m, name, documentation);

  solver
      .def(nb::new_(&call_construct<MODULENAME>), constructor_docstring,
           "periodicityX"_a, "periodicityY"_a, "periodicityZ"_a)
      .def("initialize", call_initialize<MODULENAME>, initialize_docstring,
           "temperature"_a, "viscosity"_a, "hydrodynamicRadius"_a,
           "needsTorque"_a = false, "tolerance"_a = 1e-4)
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
      .def("thermalDrift", call_thermalDrift<MODULENAME>,
           thermaldrift_docstring, "prefactor"_a = 1)
      .def("clean", &MODULENAME::clean,
           "Frees any memory allocated by the module.")
      .def_prop_ro_static(
          "precision", [](py::object) { return MODULENAME::precision; },
          R"pbdoc(Compilation precision, a string holding either float or double.)pbdoc");

  extra_code(solver);
  return solver;
}

#define MOBILITY_PYTHONIFY_WITH_EXTRA_CODE(MODULENAME, EXTRA, documentation)   \
  NB_MODULE(MODULENAME, m) {                                                   \
    auto solver = define_module_content<MODULENAME>(                           \
        m, MOBILITYSTR(MODULENAME), documentation,                             \
        [](py::class_<MODULENAME> &solver) { EXTRA });                         \
  }

#define MOBILITY_PYTHONIFY(MODULENAME, documentation)                          \
  MOBILITY_PYTHONIFY_WITH_EXTRA_CODE(MODULENAME, {}, documentation)

#endif
