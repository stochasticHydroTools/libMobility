// Raul P. Pelaez 2021-2022. libmobility interface for UAMMD PSE module
#ifndef MOBILITY_PSE_H
#define MOBILITY_PSE_H
#include<MobilityInterface/MobilityInterface.h>
#include"extra/uammd_interface.h"
#include<vector>
#include<cmath>
#include<type_traits>

static_assert(std::is_same<libmobility::real, uammd_pse::real>::value,
	      "Trying to compile PSE with a different precision to MobilityInterface.h");

class PSE: public libmobility::Mobility{
  using periodicity_mode = libmobility::periodicity_mode;
  using Configuration = libmobility::Configuration;
  using Parameters = libmobility::Parameters;
  using real = libmobility::real;
  std::vector<real> positions;
  std::shared_ptr<uammd_pse::UAMMD_PSE_Glue> pse;
  uammd_pse::PyParameters psepar, currentpsepar;
  int currentNumberParticles = 0;
  real temperature;

public:

  PSE(Configuration conf){
    if(conf.periodicityX != periodicity_mode::periodic or
       conf.periodicityY != periodicity_mode::periodic or
       conf.periodicityZ != periodicity_mode::periodic)
      throw std::runtime_error("[Mobility] This is a triply periodic solver");
  }

  //If the initialize function is called two times only changing the shear strain the module is not reinitialized entirely
  void initialize(Parameters ipar) override{
    if(pse and onlyShearStrainChanged(ipar)){
      pse->setShearStrain(psepar.shearStrain);
    }
    else{
      this->temperature = ipar.temperature;
      psepar.viscosity = ipar.viscosity;
      psepar.hydrodynamicRadius = ipar.hydrodynamicRadius[0];
      psepar.tolerance = ipar.tolerance;
      this->currentNumberParticles = ipar.numberParticles;
      Mobility::initialize(ipar);
      pse = std::make_shared<uammd_pse::UAMMD_PSE_Glue>(psepar, this->currentNumberParticles);
    }
    currentpsepar = psepar;
    if(ipar.needsTorque)
      throw std::runtime_error("[PSE] Torque is not implemented");
  }

  struct PSEParameters{
    real psi, Lx, Ly, Lz, shearStrain;
  };

  void setParametersPSE(PSEParameters i_par){
    psepar.psi = i_par.psi;
    psepar.Lx = i_par.Lx;
    psepar.Ly = i_par.Ly;
    psepar.Lz = i_par.Lz;
    psepar.shearStrain = i_par.shearStrain;
  }

  void setPositions(const real* ipositions) override{
    int numberParticles = currentNumberParticles;
    if(numberParticles == 0){
      throw std::runtime_error("[PSE] You must call setParametersPSE() after initialize()");
    }
    positions.resize(3*numberParticles);
    std::copy(ipositions, ipositions + 3*numberParticles, positions.begin());
  }

  virtual void Mdot(const real* forces, const real* torques,
		    real* linear, real* angular) override{
    if(torques)
      throw std::runtime_error("[PSE] Torque is not implemented");
    pse->computeHydrodynamicDisplacements(positions.data(), forces, linear, 0, 0);
  }

  virtual void sqrtMdotW(real* linear, real *angular, real prefactor = 1) override{
    if(angular)
      throw std::runtime_error("[PSE] Torque is not implemented");
    pse->computeHydrodynamicDisplacements(positions.data(), nullptr, linear,
					  temperature, prefactor);
  }

  virtual void hydrodynamicVelocities(const real* forces,
				      const real* torques,
				      real* linear,
				      real* angular,
				      real prefactor = 1) override{
    if(angular)
      throw std::runtime_error("[PSE] Torque is not implemented");
    pse->computeHydrodynamicDisplacements(positions.data(), forces, linear,
					  temperature, prefactor);
  }

private:

  bool onlyShearStrainChanged(Parameters i_par){
    if(currentpsepar.psi != psepar.psi or
       currentpsepar.Lx != psepar.Lx or
       currentpsepar.Ly != psepar.Ly or
       currentpsepar.Lz != psepar.Lz)
      return false;
    if(this->temperature != i_par.temperature or
       psepar.viscosity != i_par.viscosity or
       psepar.hydrodynamicRadius != i_par.hydrodynamicRadius[0] or
       psepar.tolerance != i_par.tolerance or
       this->currentNumberParticles != i_par.numberParticles)
      return false;
    return true;
  }

};
#endif
