/*Raul P. Pelaez 2021-2025. Python wrapper for the NBody module
 */
#include "mobility.h"
#include <MobilityInterface/pythonify.h>

namespace nbody_rpy {
auto string2NBodyAlgorithm(std::string algo) {
  if (algo == "naive")
    return nbody_rpy::algorithm::naive;
  else if (algo == "fast")
    return nbody_rpy::algorithm::fast;
  else if (algo == "block")
    return nbody_rpy::algorithm::block;
  else if (algo == "advise")
    return nbody_rpy::algorithm::advise;
  else {
    throw std::runtime_error("Invalid algorithm selected");
  }
}
} // namespace nbody_rpy

MOBILITY_PYTHONIFY_WITH_EXTRA_CODE(
    NBody,
    solver.def(
        "setParameters",
        [](NBody &myself, std::string algo, int NBatch, int NperBatch, real wallHeight)
        {
            myself.setParametersNBody(
                {nbody_rpy::string2NBodyAlgorithm(algo), NBatch, NperBatch, wallHeight});
        },
        "algorithm"_a = "advise", "Nbatch"_a = -1, "NperBatch"_a = -1, "wallHeight"_a = 0.0);
    , "This module computes the RPY mobility using an N^2 algorithm in the "
      "GPU. Different hydrodynamic kernels can be chosen.");
