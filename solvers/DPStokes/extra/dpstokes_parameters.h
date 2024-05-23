/*Ryker Fish, Raul P. Pelaez 2024.
  TODO description
*/

#ifndef DPSTOKES_PARAMETERS_H
#define DPSTOKES_PARAMETERS_H

#include<vector>
#include<algorithm>
#include<cmath>
#include<MobilityInterface/MobilityInterface.h>
#include"uammd_interface.h"

namespace dpstokes_parameters{
  using Parameters = libmobility::Parameters;
  using DPStokesParameters = uammd_dpstokes::PyParameters;
  using real = uammd_dpstokes::real;

  double configure_grid_and_kernels_xy(Parameters, DPStokesParameters&);
  void configure_grid_and_kernels_z(real, std::string, DPStokesParameters&, double fac=1.5);

}

#endif