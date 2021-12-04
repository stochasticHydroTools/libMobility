/*Raul P. Pelaez 2021. Example usage of the NBody mobility solver.
 All available solvers are used in a similar way, providing, in each case, the required parameters.
 For instance, a triply periodic algorithm will need at least a box size.
 This example computes the deterministic displacements of a group of particles with forces acting on them (AKA applies the mobility operator).
 */
#include"../solvers/NBody/mobility.h"
#include"../solvers/PSE/mobility.h"

#include<vector>
#include<random>
#include<algorithm>
#include<iostream>
using namespace std;
using real = libmobility::real;

int main(){

  int numberParticles = 10;
  std::vector<real> pos(3*numberParticles);
  auto forces = pos;
  mt19937 mersenne_engine {1234};
  uniform_real_distribution<real> dist {-10, 10};
  std::generate(pos.begin(), pos.end(),[&](){return dist(mersenne_engine);});
  std::generate(forces.begin(), forces.end(),[&](){return dist(mersenne_engine);});

  std::vector<real> resultNBody(pos.size(), 0);
  auto resultPSE = resultNBody;
  {
    NBody nb;
    libmobility::Parameters par;
    par.hydrodynamicRadius = 1;
    par.viscosity = 1;
    nb.initialize(par);
    nb.setPositions(pos.data(), numberParticles);
    nb.Mdot(forces.data(), nullptr, resultNBody.data());
    nb.clean();
  }
  {
    PSE pse;
    libmobility::Parameters par;
    par.hydrodynamicRadius = 1;
    par.viscosity = 1;
    par.boxSize = {128, 128, 128};
    pse.initialize(par);
    pse.setPositions(pos.data(), numberParticles);
    pse.Mdot(forces.data(), nullptr, resultPSE.data());
    pse.clean();
  }
  std::cout<<"NBody\tPSE\n";
  for(int i = 0;  i<resultNBody.size(); i++){
    std::cout<<resultNBody[i]<<"\t"<<resultPSE[i]<<std::endl;
  }
  return 0;
}
