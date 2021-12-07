/*Raul P. Pelaez 2021.
The MOBILITY_PYTHONIFY(className, description) macro creates a pybind11 module from a class (called className) that inherits from libmobility::Mobility. "description" is a string that will be printed when calling help(className) from python (accompanied by the default documentation of the mobility interface.
 */
#ifndef MOBILITY_PYTHONIFY_H
#include"MobilityInterface.h"
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
namespace py = pybind11;
using namespace pybind11::literals;
using pyarray = py::array_t<libmobility::real>;

#define MOBILITYSTR(s) xMOBILITYSTR(s)
#define xMOBILITYSTR(s) #s

#define xMOBILITY_PYTHONIFY(MODULENAME, documentation)\
  PYBIND11_MODULE(MODULENAME, m){				\
  using real = libmobility::real;\
  using Parameters = MODULENAME::Parameters;\
  auto pyclass = py::class_<MODULENAME>(m, MOBILITYSTR(MODULENAME), documentation);	\
  pyclass.def(py::init<>(), "Class constructor, does not require any arguments."). \
    def("initialize", &MODULENAME::initialize,	\
	"Initialize the module with a given set of parameters.",\
	"Parameters"_a).\
    def("setPositions",\
	[](MODULENAME &myself, pyarray pos, int N){myself.setPositions(pos.data(), N);}, \
	"The module will compute the mobility according to this set of positions.",\
	"positions"_a, "numberParticles"_a).		\
    def("Mdot", [](MODULENAME &myself, pyarray forces, pyarray torques, pyarray result){\
      auto f = forces.size()?forces.data():nullptr;			\
      auto t = torques.size()?torques.data():nullptr;			\
      myself.Mdot(f, t, result.mutable_data());}, \
      "Computes the product of the RPY Mobility matrix with a group of forces and/or torques.",	\
      "forces"_a = pyarray(), "torques"_a = pyarray(), "result"_a).	\
    def("stochasticDisplacements", [](MODULENAME &myself, pyarray result, libmobility::real prefactor){ \
      myself.stochasticDisplacements(result.mutable_data(), prefactor);},	\
	"Computes the stochastic contribution, sqrt(2*T*M) dW, where M is the mobility and dW is a Weiner process.",\
	"result"_a = pyarray(), "prefactor"_a = 1.0).							\
    def("computeHydrodynamicDisplacements", [](MODULENAME &myself, pyarray forces,\
					       pyarray torques, pyarray result, libmobility::real prefactor){ \
            auto f = forces.size()?forces.data():nullptr;			\
      auto t = torques.size()?torques.data():nullptr;			\
      myself.hydrodynamicDisplacements(f, t, result.mutable_data(), prefactor);}, \
	"Computes the hydrodynamic (deterministic and stochastic) displacements. If the forces/torques are ommited only the stochastic part is computed. If the temperature is zero (default) the stochastic part is ommited. The result is equivalent to calling Mdot followed by stochasticDisplacements.",\
	"forces"_a = pyarray(), "torques"_a = pyarray(), "result"_a  = pyarray(), "prefactor"_a = 1).	\
    def("clean", &MODULENAME::clean, "Frees any memory allocated by the module.").\
    def_property_readonly_static("precision", [](py::object){return MODULENAME::precision;}, "Compilation precision, a string holding either float or double."); \
  auto period = py::class_<libmobility::Periodicity>(m, "Periodicity", py::module_local()); \
    period.def(py::init<bool, bool, bool>(), "x"_a, "y"_a, "z"_a).\
      def_readwrite("x",&libmobility::Periodicity::x).\
      def_readwrite("y",&libmobility::Periodicity::y).\
      def_readwrite("z",&libmobility::Periodicity::z);\
    auto box = py::class_<libmobility::BoxSize>(m, "BoxSize", py::module_local());		\
    box.def(py::init<libmobility::real,libmobility::real,libmobility::real>(), "x"_a, "y"_a, "z"_a).\
      def_readwrite("x",&libmobility::BoxSize::x).\
      def_readwrite("y",&libmobility::BoxSize::y).\
      def_readwrite("z",&libmobility::BoxSize::z);\
    auto params = py::class_<Parameters>(m, "Parameters", py::module_local()); \
    params.def(py::init([](real temperature,\
		    real viscosity,\
		    real hydrodynamicRadius,\
		    libmobility::BoxSize boxSize,\
		    libmobility::Periodicity periodicity,\
			   real tolerance) { \
      auto tmp = std::unique_ptr<Parameters>(new Parameters); \
      tmp->temperature = temperature;\
      tmp->viscosity = viscosity;\
      tmp->hydrodynamicRadius = hydrodynamicRadius;\
      tmp->boxSize = boxSize;\
      tmp->tolerance = tolerance;\
      tmp->periodicity = periodicity;     \
      return tmp;	\
    }),"temperature"_a = 0.0,"viscosity"_a  = 1.0,"hydrodynamicRadius"_a = 1.0,\
	"boxSize"_a = libmobility::BoxSize(), "periodicity"_a = libmobility::Periodicity(),\
	"tolerance"_a = 1e-4).\
      def_readwrite("temperature", &Parameters::temperature).\
      def_readwrite("viscosity", &Parameters::viscosity).\
      def_readwrite("hydrodynamicRadius", &Parameters::hydrodynamicRadius).\
      def_readwrite("boxSize", &Parameters::boxSize).\
      def_readwrite("periodicity", &Parameters::periodicity).\
      def_readwrite("tolerance", &Parameters::tolerance).\
      def("__str__", [](const libmobility::Parameters &p){\
	return "temperature = "+std::to_string(p.temperature)+"\n"+\
	  "viscosity = " + std::to_string(p.viscosity) +"\n"+\
	  "hydrodynamicRadius = " + std::to_string(p.hydrodynamicRadius)+"\n"+\
	  "box (L = " + std::to_string(p.boxSize.x) +\
	  "," + std::to_string(p.boxSize.y) + "," + std::to_string(p.boxSize.z) + ")\n"+\
	  "periodicity (x = " + std::to_string(p.periodicity.x) +\
	  ", y =" + std::to_string(p.periodicity.y) + ", z = " + std::to_string(p.periodicity.z) + ")\n"+\
	  "tolerance = " + std::to_string(p. tolerance)+ "\n";\
      });\
  }
#define MOBILITY_PYTHONIFY(MODULENAME, documentationPrelude) xMOBILITY_PYTHONIFY(MODULENAME, documentationPrelude)
#endif

