#ifndef LIBMOBILITY_LANCZOS_ADAPTOR_H
#define LIBMOBILITY_LANCZOS_ADAPTOR_H
#include"../../third_party/LanczosAlgorithm/include/LanczosAlgorithm.h"
#include<random>
#include<algorithm>
template<class Foo>
struct MatrixDotAdaptor: public lanczos::MatrixDot{
  Foo foo;
  MatrixDotAdaptor(Foo foo):foo(foo){}
  void dot(lanczos::real* v, lanczos::real* Mv) override{
    foo(v,Mv);
  }
};

template<class Foo>
auto createMatrixDotAdaptor(Foo foo){
  return MatrixDotAdaptor<Foo>(foo);
}

class LanczosStochasticDisplacements{
  using real = lanczos::real;
  lanczos::Solver lanczos;
  std::vector<real> lanczosNoise;
  real lanczosTolerance = 1e-3; //Default tolerance
  int numberParticles;
  std::mt19937 engine;
  real temperature;
public:

  LanczosStochasticDisplacements(int N, real T, real tol){
    this->numberParticles = N;
    this->temperature = T;
    this->lanczosTolerance = tol;
    std::random_device rnd_device;
    engine = std::mt19937{rnd_device()};
  }

  template<class Foo>
  void stochasticDisplacements(Foo foo, real* result, real prefactor = 1){
    std::normal_distribution<real> dist {0, 1};
    auto gen = [&](){return dist(engine);};
    lanczosNoise.resize(3*numberParticles);
    std::generate(lanczosNoise.begin(), lanczosNoise.end(), gen);
    auto dot = createMatrixDotAdaptor([&foo](const real *v, real *Mv){foo(v, Mv);});    
    lanczos.run(dot, result, lanczosNoise.data(), lanczosTolerance, 3*numberParticles);
  }

};

#endif
