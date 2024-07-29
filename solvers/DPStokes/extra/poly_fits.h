#include <vector>
  
namespace dpstokes_polys{

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

  std::vector<double> cbetam_inv = {
  4131643418.193291,
  -10471683395.26777,
  11833009228.6429,
  -7851132955.882548,
  3388121732.651829,
  -994285251.2185925,
  201183449.7086889,
  -27776767.88241613,
  2515647.646492857,
  -136305.2970161326,
  3445.959503226691};
}