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

  bool isValid(int num);

  int findBest(int N);

  std::vector<int> fft_friendly_sizes(int N, int sep, int count);

  double polyEval(std::vector<double> polyCoeffs, double x);

  double linearInterp(std::vector<double> f, std::vector<double> x, double v);

}

#endif