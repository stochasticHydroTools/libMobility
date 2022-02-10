/*Raul P. Pelaez 2021. Example usage of the NBody mobility solver.
 All available solvers are used in a similar way, providing, in each case, the required parameters.
 For instance, a triply periodic algorithm will need at least a box size.
 This example computes the deterministic displacements of a group of particles with forces acting on them (AKA applies the mobility operator).
 */
#include"NBody/mobility.h"
#include"PSE/mobility.h"

#include<vector>
#include<random>
#include<algorithm>
#include<iostream>
using namespace std;
using scalar = libmobility::real;

int main(){

  int numberParticles = 10;
  std::vector<scalar> pos(3*numberParticles);
  auto forces = pos;
  mt19937 mersenne_engine {1234};
  uniform_real_distribution<scalar> dist {-10, 10};
  std::generate(pos.begin(), pos.end(),[&](){return dist(mersenne_engine);});
  std::generate(forces.begin(), forces.end(),[&](){return dist(mersenne_engine);});

  std::vector<scalar> resultNBody(pos.size(), 0);
  std::vector<scalar> noiseNBody(pos.size(), 0);
  
  auto resultPSE = resultNBody;
  
  libmobility::Parameters par;
  par.hydrodynamicRadius = {1};
  par.viscosity = 1;
  par.numberParticles = numberParticles;
  par.tolerance = 1e-4;
  par.temperature = 1.0;

  
  {
    libmobility::Configuration conf{.dimensions = 3,
      .numberSpecies = 1,
      .periodicity = libmobility::periodicity_mode::open,    
      .dev = libmobility::device::automatic};

    NBody nb(conf);
    nb.initialize(par);
    nb.setPositions(pos.data());
    nb.Mdot(forces.data(), nullptr, resultNBody.data());
    nb.stochasticDisplacements(noiseNBody.data());
    nb.clean();
  }
  
  {
    libmobility::Configuration conf{.dimensions = 3,
      .numberSpecies = 1,
      .periodicity = libmobility::periodicity_mode::triply_periodic,    
      .dev = libmobility::device::automatic};

    PSE pse(conf);
    pse.initialize(par);
    pse.setPositions(pos.data());
    pse.Mdot(forces.data(), nullptr, resultPSE.data());
    pse.clean();
  }
  std::cout<<"NBody\tPSE\n";
  for(int i = 0;  i<resultNBody.size(); i++){
    std::cout<<resultNBody[i]<<"\t"<<resultPSE[i]<<std::endl;
  }
  return 0;
}
