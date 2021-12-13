
#ifndef MOBILITY_PSE_H
#define MOBILITY_PSE_H
#include<MobilityInterface/MobilityInterface.h>
#include"UAMMD_PSE_Python/uammd_interface.h"
#include<vector>
#include<cmath>
#include<type_traits>

static_assert(std::is_same<libmobility::real, uammd_pse::real>::value,
	      "Trying to compile PSE with a different precision to MobilityInterface.h");

class PSE: public libmobility::Mobility{
  using real = libmobility::real;
  std::vector<real> positions;
  std::shared_ptr<uammd_pse::UAMMD_PSE_Glue> pse;
  uammd_pse::PyParameters psepar;
  int currentNumberParticles = 0;
  real temperature;
public:
  using Parameters = libmobility::Parameters;
  
  virtual void initialize(Parameters ipar) override{
    this->temperature = ipar.temperature;
    psepar.viscosity = ipar.viscosity;
    psepar.hydrodynamicRadius = ipar.hydrodynamicRadius;
    psepar.Lx = ipar.boxSize.x;
    psepar.Ly = ipar.boxSize.y;
    psepar.Lz = ipar.boxSize.z;
    psepar.tolerance = ipar.tolerance;
    psepar.psi = 0.5/ipar.hydrodynamicRadius;
  }

  virtual void setPositions(const real* ipositions, int numberParticles) override{
    if(not pse or currentNumberParticles != numberParticles){
      this->currentNumberParticles = numberParticles;
      pse = std::make_shared<uammd_pse::UAMMD_PSE_Glue>(psepar, numberParticles);
    }
    positions.resize(3*numberParticles);
    std::copy(ipositions, ipositions + 3*numberParticles, positions.begin());    
  }

  virtual void Mdot(const real* forces, const real *torques, real* result) override{
    if(torques) throw std::runtime_error("PSE can only compute monopole displacements");
    pse->computeHydrodynamicDisplacements(positions.data(), forces, result, 0, 0);
  }

  virtual void stochasticDisplacements(real* result, real prefactor = 1) override{
    pse->computeHydrodynamicDisplacements(positions.data(), nullptr, result,
					  temperature, prefactor);
  }

  virtual void hydrodynamicDisplacements(const real* forces, const real *torques,
					 real* result, real prefactor = 1) override{
    pse->computeHydrodynamicDisplacements(positions.data(), forces, result,
					  temperature, prefactor);
  }

};
#endif
