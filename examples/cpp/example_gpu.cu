/*Raul P. Pelaez 2021-2025. Example usage of the NBody mobility solver.
 All available solvers are used in a similar way, providing, in each case, the
 required parameters. For instance, a triply periodic algorithm will need at
 least a box size. Identical to example.cpp, but uses thrust::device_vector
 instead of std::vector
 */
#include "MobilityInterface/MobilityInterface.h"
#include "solvers/NBody/mobility.h"
#include "solvers/PSE/mobility.h"
#include <algorithm>
#include <iostream>
#include <random>
#include <thrust/device_vector.h>
#include <type_traits>
#include <vector>
using namespace std;

using scalar = libmobility::real;
using MobilityBase = libmobility::Mobility;
using Configuration = libmobility::Configuration;
using libmobility::Parameters;
// Configures, initializes any solver (between PSE and NBody)
// The same function can be extended to create any solver.
// We need it to desambiguate by calling the solver-dependent setParameters
// function when necessary. For instance, see PSE below
template <class Solver> auto initializeSolver(Parameters par, int numberParticles) {
  std::shared_ptr<MobilityBase> solver;
  if (std::is_same<Solver, NBody>::value) {
    auto nbody = std::make_shared<NBody>(
        Configuration{.periodicityX = libmobility::periodicity_mode::open,
                      .periodicityY = libmobility::periodicity_mode::open,
                      .periodicityZ = libmobility::periodicity_mode::open});
    nbody->setParametersNBody(
        {nbody_rpy::algorithm::advise, 1, numberParticles});
    solver = nbody;
  }
  if (std::is_same<Solver, PSE>::value) {
    auto pse = std::make_shared<PSE>(
        Configuration{.periodicityX = libmobility::periodicity_mode::periodic,
                      .periodicityY = libmobility::periodicity_mode::periodic,
                      .periodicityZ = libmobility::periodicity_mode::periodic});
    scalar lx, ly, lz;
    lx = ly = lz = 128;
    scalar split = 1.0;
    scalar shearStrain = 0.0;
    pse->setParametersPSE({split, lx, ly, lz, shearStrain});
    solver = pse;
  }
  solver->initialize(par);
  return solver;
}

// An example of a function that works for any solver
auto computeMFWithSolver(std::shared_ptr<MobilityBase> solver,
                         thrust::device_vector<scalar> &ipos,
                         thrust::device_vector<scalar> &iforces) {
  thrust::device_vector<scalar> result(ipos.size(), 0);
  solver->setPositions(ipos);
  solver->Mdot(iforces, {}, result, {});
  return result;
}

// Lets compute the deterministic and stochastic displacements of a group of
// particles
int main() {
  {
    // Create some arbitrary positions and forces
    int numberParticles = 10;
    std::vector<scalar> h_pos(3 * numberParticles);
    auto h_forces = h_pos;
    mt19937 mersenne_engine{1234};
    uniform_real_distribution<scalar> dist{-10, 10};
    std::generate(h_pos.begin(), h_pos.end(),
                  [&]() { return dist(mersenne_engine); });
    std::generate(h_forces.begin(), h_forces.end(),
                  [&]() { return dist(mersenne_engine); });

    thrust::device_vector<scalar> pos = h_pos;
    thrust::device_vector<scalar> forces = h_forces;
    // Set up parameters generic to any solver
    Parameters par;
    par.hydrodynamicRadius = {1};
    par.viscosity = 1;
    par.tolerance = 1e-4;
    par.temperature = 1.0;
    par.needsTorque = false;

    // Create two different solvers
    auto solver_pse = initializeSolver<PSE>(par, numberParticles);
    auto solver_nbody = initializeSolver<NBody>(par, numberParticles);

    // Compute the displacements
    auto resultNBody = computeMFWithSolver(solver_nbody, pos, forces);
    auto resultPSE = computeMFWithSolver(solver_pse, pos, forces);

    // The solvers can be used to compute stochastic displacements, even if they
    // do not provide a specific way to compute them (defaults to using the
    // lanczos algorithm
    thrust::device_vector<scalar> noiseNBody(pos.size(), 0);
    scalar prefactor = 1.0;
    solver_nbody->sqrtMdotW(noiseNBody, {}, prefactor);

    // Remember to clean up when done
    solver_nbody->clean();
    solver_pse->clean();

    std::cout << "NBody\tPSE\n";
    for (int i = 0; i < resultNBody.size(); i++) {
      std::cout << resultNBody[i] << "\t" << resultPSE[i] << std::endl;
    }
  }
  return 0;
}
