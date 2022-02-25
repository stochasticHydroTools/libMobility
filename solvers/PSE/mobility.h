// Donev: This looks nice, clean, and readable to me ;-)

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
  using device = libmobility::device;
  using periodicity_mode = libmobility::periodicity_mode;
  using Configuration = libmobility::Configuration;
  using Parameters = libmobility::Parameters;
  using real = libmobility::real;
  std::vector<real> positions;
  std::shared_ptr<uammd_pse::UAMMD_PSE_Glue> pse;
  uammd_pse::PyParameters psepar;
  int currentNumberParticles = 0;
  real temperature;
public:

  PSE(Configuration conf){
    if(conf.numberSpecies!=1) // Donev: Doesn't UAMMD support different species at all?
      throw std::runtime_error("[Mobility] I can only deal with one species");
    if(conf.dev == device::cpu)
      throw std::runtime_error("[Mobility] This is a GPU-only solver");
    if(conf.periodicity != periodicity_mode::triply_periodic)
      throw std::runtime_error("[Mobility] This is a triply periodic solver");
  }
  
  void initialize(Parameters ipar) override{
    this->temperature = ipar.temperature;
    psepar.viscosity = ipar.viscosity;
    psepar.hydrodynamicRadius = ipar.hydrodynamicRadius[0];
    psepar.tolerance = ipar.tolerance;
    this->currentNumberParticles = ipar.numberParticles;
  }

  void setParametersPSE(real psi, real Lx, real Ly, real Lz){
    psepar.psi = psi;
    psepar.Lx = Lx;
    psepar.Ly = Ly;
    psepar.Lz = Lz;    
    pse = std::make_shared<uammd_pse::UAMMD_PSE_Glue>(psepar, this->currentNumberParticles);
  }
  
  void setPositions(const real* ipositions) override{
    int numberParticles = currentNumberParticles;
    if(numberParticles == 0){
      throw std::runtime_error("[PSE] You must call setParametersPSE() after initialize()");
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
