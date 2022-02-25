/*Raul P. Pelaez 2021. Example usage of the NBody mobility solver.
 All available solvers are used in a similar way, providing, in each case, the required parameters.
 For instance, a triply periodic algorithm will need at least a box size.
 */
#include"NBody/mobility.h"
#include"PSE/mobility.h"

#include <type_traits>
#include<vector>
#include<random>
#include<algorithm>
#include<iostream>
using namespace std;
using scalar = libmobility::real;

using MobilityBase = libmobility::Mobility;
using libmobility::Parameters;


//Configures, initializes and returns a NBody solver
std::shared_ptr<NBody> initializeNBody(Parameters par){
  libmobility::Configuration conf{.dimensions = 3,
    .numberSpecies = 1,
    .periodicity = libmobility::periodicity_mode::open,
    .dev = libmobility::device::automatic};
    auto solver = std::make_shared<NBody>(conf);
    solver->initialize(par);
    return solver;
}

//Configures, initializes and returns a PSE solver
std::shared_ptr<PSE> initializePSE(Parameters par){
  libmobility::Configuration conf{.dimensions = 3,
    .numberSpecies = 1,
    .periodicity = libmobility::periodicity_mode::triply_periodic,
    .dev = libmobility::device::automatic};
    auto solver = std::make_shared<PSE>(conf);
    solver->initialize(par);
    return solver;
}

//Configures, initializes any solver (between PSE and NBody)
//The same function can create any solver, this function also calls the solver-dependent setParameters function when necessary
template<class Solver>
std::shared_ptr<MobilityBase> initializeSolver(Parameters par){
  if(std::is_same_v<Solver,NBody>){
    auto solver = initializeNBody(par);    
    return solver;
  }
  if(std::is_same_v<Solver,PSE>){
    scalar lx,ly,lz;
    lx=ly=lz=128;
    scalar split = 1.0;
    auto solver = initializePSE(par);
    solver->setParametersPSE(split, lx,ly,lz);
    return solver;
  }
}

// Donev: I don't really understand why you are doing templates above and why we need computeMFWithSolver
// Isn't the whole point that oen can simply call solver->Mdot inside main. Why do we need yet another wrapper here
// Let's discuss

//This function works for any solver
auto computeMFWithSolver(std::shared_ptr<MobilityBase> solver,
			 std::vector<scalar> &pos,
			 std::vector<scalar> &forces){
  std::vector<scalar> result(pos.size(), 0);
  solver->Mdot(forces.data(), nullptr, result.data());
  return result;
}

//Lets compute the deterministic and stochastic displacements of a group of particles
int main(){

  //Create some arbitrary positions and forces
  int numberParticles = 10;
  std::vector<scalar> pos(3*numberParticles);
  auto forces = pos;
  mt19937 mersenne_engine {1234};
  uniform_real_distribution<scalar> dist {-10, 10};
  std::generate(pos.begin(), pos.end(),[&](){return dist(mersenne_engine);});
  std::generate(forces.begin(), forces.end(),[&](){return dist(mersenne_engine);});

  
  //Set up parameters generic to any solver
  Parameters par;
  par.hydrodynamicRadius = {1};
  par.viscosity = 1;
  par.numberParticles = numberParticles;
  par.tolerance = 1e-4;
  par.temperature = 1.0;

  //Create two different solvers
  auto solver_pse = initializeSolver<PSE>(par);
  auto solver_nbody = initializeSolver<NBody>(par);

  //Compute the displacements
  auto resultNBody = computeMFWithSolver(solver_nbody, pos, forces);
  auto resultPSE = computeMFWithSolver(solver_pse, pos, forces);

  //The solvers can be used to compute stochastic displacements, even if they do not provide a specific way to compute them (defaults to using the lanczos algorithm
  std::vector<scalar> noiseNBody(pos.size(), 0);
  scalar prefactor = 1.0;
  solver_nbody->stochasticDisplacements(noiseNBody.data(), prefactor);

  //Remember to clean up when done
  solver_nbody->clean();
  solver_pse->clean();
  
  std::cout<<"NBody\tPSE\n";
  for(int i = 0;  i<resultNBody.size(); i++){
    std::cout<<resultNBody[i]<<"\t"<<resultPSE[i]<<std::endl;
  }
  return 0;
}
