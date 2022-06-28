/* Raul P. Pelaez 2020-2021. Python wrapper for the batched RPY Nbody evaluator.
   Three algorithms are provided:
     Fast: Leverages shared memory to hide bandwidth latency
     Naive: A dumb thread-per-partice parallelization of the N^2 double loop
     Block: Assigns a block to each particle, the first thread then reduces the result of the whole block.
 */

#include"interface.h"
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
namespace py = pybind11;
using namespace nbody_rpy;
auto adviseAlgorithm(int Nbatches, int NperBatch){
  int N = Nbatches*NperBatch;
  if(N<5e3){
    return algorithm::block;
  }
  if(N<50e3){
    return algorithm::naive;
  }
  return algorithm::fast;
}

auto stringToAlgorithm(std::string alg_str){
  if(alg_str == "advise") return algorithm::advise;
  else if(alg_str == "block") return algorithm::block;
  else if(alg_str == "fast") return algorithm::fast;
  else if(alg_str == "naive") return algorithm::naive;
  else{
    throw std::runtime_error("Invalid algorithm, choose from (fast, naive, block, advise)");
  }
  return algorithm::advise;
}

void computeMdot(py::array_t<real> h_pos, py::array_t<real> h_forces,
                 py::array_t<real> h_MF, int Nbatches, int NperBatch,
                 real selfMobility, real hydrodynamicRadius, std::string alg_str){
  algorithm alg;
  if(alg_str == "advise")
    alg = adviseAlgorithm(Nbatches, NperBatch);
  else
    alg = stringToAlgorithm(alg_str);
  callBatchedNBodyRPY(h_pos.data(), h_forces.data(),
		      h_MF.mutable_data(),
		      Nbatches, NperBatch,
		      selfMobility, hydrodynamicRadius,
		      alg);
}


auto getPrecision(){
#ifndef DOUBLE_PRECISION
  constexpr auto precision = "float";
#else
  constexpr auto precision = "double";
#endif
  return precision;
}
using namespace pybind11::literals;
#ifndef MODULENAME
#define MODULENAME BatchedNBodyRPY
#endif
PYBIND11_MODULE(MODULENAME, m) {
  m.doc() = "NBody Batched RPY evaluator\nUSAGE:\n\tcall computeMdot, the input/output must have interleaved coordinates and each batch is placed after the previous one. [x_1_1 y_1_1 z_1_1,...x_1_NperBatch,...x_Nbatches_NperBatch]\nThree algorithms are available (block, naive and fast), the optional parameter 'algorithm' allows to select one (otherwise it will be automatically chosen according to the problem size).";  
  m.def("computeMdot", &computeMdot,
	"pos"_a, "force"_a, "MF"_a, "Nbatches"_a, "NperBatch"_a, "selfMobility"_a, "hydrodynamicRadius"_a,
	"algorithm"_a = "advise");
  m.def("getPrecision", &getPrecision);
}
