/*Ryker Fish, Raul P. Pelaez 2024.
  TODO description
*/

#include<vector>
#include<algorithm>
#include<cmath>
#include<MobilityInterface/MobilityInterface.h>
#include"uammd_interface.h"
#include"dpstokes_parameters.h"
#include"polyFits.h"

namespace dpstokes_parameters{

  double configure_grid_and_kernels_xy(Parameters ipar, DPStokesParameters &dppar){
    double hydroRadius = ipar.hydrodynamicRadius[0];

    // STEP 1: set w & beta based on kernel type (Table 1/2)
    double w = 6.0; // note: should be an int but I'm scared of the consequences
    double beta = 1.714;

    // STEP 2: use c(beta*w) polyfit to get c
      // note: polyfit is to c(beta*w), not just beta
    double c_beta = polyEval(polyFits::cbetam, beta*w);
    double h = hydroRadius / (w*c_beta); // compute h using h = Rh/(w*c) and nx = Lx/h
    double nx_unsafe = dppar.Lx/h; // not necessarily fft friendly

    // STEP 3: find candidates for fft friendly nx
    std::vector<int> nx_safe = fft_friendly_sizes(floor(nx_unsafe), 100, 10);
    int m = nx_safe.size();

    std::vector<double> errors(m);
    std::vector<double> h_candidates(m);
    std::vector<double> beta_candidates(m);

    int n = 1000;
    double start = w; // interpolation range
    double end = 3*w;
    std::vector<double> err_axis(n); // creates a linspace vector
    h = (end-start)/(n-1);
    for(int i = 0; i < n; i++){
      err_axis[i] = start + i*h;
    }

    // STEP 4: for each candidate, compute corresponding h and use c^{-1} polyfits to get beta
    // then, linearly interpolate the error spline using beta*w to get error
    for(int i = 0; i < m; i++){
      h_candidates[i] = dppar.Lx/(1.0*nx_safe[i]); // double cast
      beta_candidates[i] = polyEval(polyFits::cbetam_inv, hydroRadius/(w*h_candidates[i])) / w;

      double beta_w = beta_candidates[i]*w;
      if (beta_w < w || beta_w > 3*w){ // out of range of interpolation
        errors[i] = -1;
        continue;
      }
      errors[i] = linearInterp(polyFits::errmw6,err_axis,beta_w);
    }

    // pick parameters with minimum error
    int best_index = 0;
    for(int i = 1; i < m; i++){
      if(errors[i] != -1 && errors[i] < errors[best_index]){
        best_index = i;
      }
    }

    // set smallest h, beta
    dppar.nx = nx_safe[best_index]; 
    dppar.ny = nx_safe[best_index]; // assumes Lx=Ly
    dppar.w = w;
    dppar.beta = beta_candidates[best_index]*w;
    h = h_candidates[best_index];

    return h;
  }

  void configure_grid_and_kernels_z(real h, std::string wallmode, DPStokesParameters &dppar, double fac){
    // fac: safety factor

    // Add a buffer of fac*w*h/2 when there is an open boundary
    double sep_up = 0;
    double sep_down = 0;
    if(wallmode == "nowall"){
      sep_up = fac*dppar.w*h/2;
      sep_down = fac*dppar.w*h/2;
      dppar.zmax += sep_up;
      dppar.zmin -= sep_down;
    }
    if(wallmode == "bottom"){
      sep_up = fac*dppar.w*h/2;
      dppar.zmax += sep_up;
    }

    double Lz = dppar.zmax - dppar.zmin;
    double H = Lz/2;

    int nz = ceil(M_PI/ (acos(-h/H) - M_PI_2) );

    // correction so 2(Nz-1) is fft friendly
    dppar.nz = fft_friendly_sizes(nz, 100, 1)[0] + 1;
  }

  bool isValid(int num) {
    while (num % 2 == 0) num /= 2;
    while (num % 3 == 0) num /= 3;
    while (num % 5 == 0) num /= 5;
    while (num % 7 == 0) num /= 7;
    return (num == 1);
  }

  int findBest(int N) {
    for (int i = 0; i < N; ++i) {
      if (isValid(N + i)) {
        return N + i;
      }
      if (isValid(N - i)) {
        return N - i;
      }
    }
    return 0;
  }

  std::vector<int> fft_friendly_sizes(int N, int sep, int count) {
      std::vector<int> Ns;
      int c = 0;
      for (int i = N; i < N + sep; ++i) {
          int _N = findBest(i);
          if (std::find(Ns.begin(), Ns.end(), _N) == Ns.end()) {
              if (c == count) {
                  break;
              }
              Ns.push_back(_N);
              ++c;
          }
      }
      return Ns;
  }

  double polyEval(std::vector<double> polyCoeffs, double x){

    int order = polyCoeffs.size() - 1;
    double accumulator = polyCoeffs[order];
    double current_x = x;
    for(int i = 1; i <= order; i++){
      accumulator += polyCoeffs[order-i]*current_x;
      current_x *= x;
    }

    return accumulator;
  }

  double linearInterp(std::vector<double> f, std::vector<double> x, double v){

    // assume x sorted (since it comes from a linspace)
    // f is some function evaluated on x

    double h = x[1]-x[0];
    int j = floor( (v-x[0])/h ); // index such that x[j] <= v < x[j+1]

    double x0 = x[j];
    double x1 = x[j+1];
    double y0 = f[j];
    double y1 = f[j+1];

    double f_v = y0 + (v-x0)*( (y1-y0)/(x1-x0) ); // linear interp formula

    return f_v;
  }
}