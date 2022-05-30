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
using pyarray = py::array_t<libmobility::real>;

#define MOBILITYSTR(s) xMOBILITYSTR(s)
#define xMOBILITYSTR(s) #s

inline auto string2Periodicity(std::string per){
  using libmobility::periodicity_mode;
  if(per == "open") return periodicity_mode::open;
  else if(per == "unspecified") return periodicity_mode::unspecified;
  else if(per == "triply_periodic") return periodicity_mode::triply_periodic;
  else if(per == "doubly_periodic") return periodicity_mode::doubly_periodic;
  else if(per == "single_wall") return periodicity_mode::single_wall;
  else throw std::runtime_error("[libMobility] Invalid periodicity");   
}

inline auto string2Device(std::string dev){
  using libmobility::device;
  if(dev == "cpu") return device::cpu;
  else if(dev == "gpu") return device::gpu;
  else if(dev == "automatic") return device::automatic;
  else throw std::runtime_error("[libMobility] Invalid device");
}

inline auto createConfiguration(int dim, std::string per, std::string dev){
  libmobility::Configuration conf;
  conf.dimensions = dim;
  conf.periodicity = string2Periodicity(per);
  conf.dev = string2Device(dev);
  return conf;
}

#define xMOBILITY_PYTHONIFY(MODULENAME, EXTRACODE, documentation)	\
  PYBIND11_MODULE(MODULENAME, m){		      \
  using real = libmobility::real;		      \
  using Parameters = libmobility::Parameters;				\
  using Configuration = libmobility::Configuration;			\
  auto solver = py::class_<MODULENAME>(m, MOBILITYSTR(MODULENAME), documentation); \
  solver.def(py::init([](int dim, std::string per, std::string dev){ \
    return std::unique_ptr<MODULENAME>(new MODULENAME(createConfiguration(dim,per,dev))); }),\
    "Class constructor.", "dimension"_a, "periodicity"_a, "device"_a). \
  def("initialize", [](MODULENAME &myself, real T, real eta, real a, int N){ \
    Parameters par;							\
    par.temperature = T;						\
    par.viscosity = eta;						\
    par.hydrodynamicRadius = {a};					\
    par.numberParticles = N;						\
    myself.initialize(par);						\
  },									\
    "Initialize the module with a given set of parameters.",		\
    "temperature"_a, "viscosity"_a,					\
    "hydrodynamicRadius"_a,						\
    "numberParticles"_a).						\
    def("setPositions", [](MODULENAME &myself, pyarray pos){myself.setPositions(pos.data());}, \
	"The module will compute the mobility according to this set of positions.", \
	"positions"_a).							\
    def("Mdot", [](MODULENAME &myself, pyarray forces, pyarray torques, pyarray result){\
      auto f = forces.size()?forces.data():nullptr;			\
      auto t = torques.size()?torques.data():nullptr;			\
      myself.Mdot(f, t, result.mutable_data());},			\
      "Computes the product of the RPY Mobility matrix with a group of forces and/or torques.",	\
      "forces"_a = pyarray(), "torques"_a = pyarray(), "result"_a).	\
    def("stochasticDisplacements", [](MODULENAME &myself, pyarray result, libmobility::real prefactor){ \
      myself.stochasticDisplacements(result.mutable_data(), prefactor);}, \
      "Computes the stochastic contribution, sqrt(2*T*M) dW, where M is the mobility and dW is a Weiner process.", \
      "result"_a = pyarray(), "prefactor"_a = 1.0).			\
    def("computeHydrodynamicDisplacements", [](MODULENAME &myself, pyarray forces,\
					       pyarray torques, pyarray result, libmobility::real prefactor){ \
      auto f = forces.size()?forces.data():nullptr;			\
      auto t = torques.size()?torques.data():nullptr;			\
      myself.hydrodynamicDisplacements(f, t, result.mutable_data(), prefactor);}, \
	"Computes the hydrodynamic (deterministic and stochastic) displacements. If the forces/torques are ommited only the stochastic part is computed. If the temperature is zero (default) the stochastic part is ommited. The result is equivalent to calling Mdot followed by stochasticDisplacements.", \
	"forces"_a = pyarray(), "torques"_a = pyarray(), "result"_a  = pyarray(), "prefactor"_a = 1). \
    def("clean", &MODULENAME::clean, "Frees any memory allocated by the module."). \
    def_property_readonly_static("precision", [](py::object){return MODULENAME::precision;}, "Compilation precision, a string holding either float or double."); \
  EXTRACODE\
}
#define MOBILITY_PYTHONIFY(MODULENAME, documentationPrelude) xMOBILITY_PYTHONIFY(MODULENAME,; ,documentationPrelude)
#define MOBILITY_PYTHONIFY_WITH_EXTRA_CODE(MODULENAME, EXTRA, documentationPrelude) xMOBILITY_PYTHONIFY(MODULENAME,  EXTRA, documentationPrelude)
#endif

