/* Raul P. Pelaez 2021. Doubly Periodic Stokes UAMMD wrapper
   Allows to call the DPStokes or TP FCM modules from via a simple contained class to compute the product between the mobility tensor and a list forces and torques acting on a group of positions.

   Additionally, a glue class is provided to ease separate compilation between GPU code (this source) and another code. For instance, the python wrapper in uammd_python.cpp
*/
#include <uammd.cuh>
//Doubly Periodic FCM implementation (currently without noise)
#include <Integrator/BDHI/DoublyPeriodic/DPStokesSlab.cuh>
//Triply Periodic FCM implementation
#include <Integrator/BDHI/BDHI_FCM.cuh>
#include"uammd_interface.h"
// Some convenient aliases
namespace uammd_dpstokes{
  using FCM_BM = uammd::BDHI::FCM_ns::Kernels::BarnettMagland;
  using FCM = uammd::BDHI::FCM_impl<FCM_BM, FCM_BM>;
  using DPStokesSlab = uammd::DPStokesSlab_ns::DPStokes;
  using uammd::DPStokesSlab_ns::WallMode;
  using uammd::System;

  //Helper functions and objects
  struct Real3ToReal4{
    __host__ __device__ uammd::real4 operator()(uammd::real3 i){
      auto pr4 = uammd::make_real4(i);
      return pr4;
    }
  };
  struct Real4ToReal3{
    __host__ __device__ uammd::real3 operator()(uammd::real4 i){
      auto pr3 = uammd::make_real3(i);
      return pr3;
    }
  };

  struct Real3ToReal4SubstractOriginZ{
    real origin;
    Real3ToReal4SubstractOriginZ(real origin):origin(origin){}
    __host__ __device__ uammd::real4 operator()(uammd::real3 i){
      auto pr4 = uammd::make_real4(i);
      pr4.z -= origin;
      return pr4;
    }
  };

  auto createFCMParameters(PyParameters pypar){
    FCM::Parameters par;
    par.temperature = 0; //FCM can compute fluctuations, but they are turned off here
    par.viscosity = pypar.viscosity;
    par.tolerance = pypar.tolerance;
    par.box = uammd::Box({pypar.Lx, pypar.Ly, pypar.zmax- pypar.zmin});
    par.cells = {pypar.nx, pypar.ny, pypar.nz};
    par.kernel = std::make_shared<FCM_BM>(pypar.w, pypar.alpha, pypar.beta, pypar.Lx/pypar.nx);
    par.kernelTorque = std::make_shared<FCM_BM>(pypar.w_d, pypar.alpha_d, pypar.beta_d, pypar.Lx/pypar.nx);
    return par;
  }

  WallMode stringToWallMode(std::string str){
    if(str.compare("nowall") == 0){
      return WallMode::none;
    }
    else if(str.compare("slit") == 0){
      return WallMode::slit;
    }
    else if(str.compare("bottom") == 0){
      return WallMode::bottom;
    }
    else return WallMode::none;
  }

  auto createDPStokesParameters(PyParameters pypar){
    DPStokesSlab::Parameters par;
    par.nx         = pypar.nx;
    par.ny         = pypar.ny;
    par.nz	  = pypar.nz;
    par.dt	  = pypar.dt;
    par.viscosity	  = pypar.viscosity;
    par.Lx	  = pypar.Lx;
    par.Ly	  = pypar.Ly;
    par.H		  = pypar.zmax-pypar.zmin;
    par.w = pypar.w;
    par.w_d = pypar.w_d;
    par.hydrodynamicRadius = pypar.hydrodynamicRadius;
    par.beta = pypar.beta;
    par.beta_d = pypar.beta_d;
    par.alpha = pypar.alpha;
    par.alpha_d = pypar.alpha_d;
    par.mode = stringToWallMode(pypar.mode);
    return par;
  }

  //Wrapper to UAMMD's TP and DP hydrodynamic modules
  struct DPStokesUAMMD {
  private:
    auto computeHydrodynamicDisplacements(bool useTorque){
      auto force = pd->getForce(uammd::access::gpu, uammd::access::read);
      auto pos = pd->getPos(uammd::access::gpu, uammd::access::read);
      auto torque = pd->getTorqueIfAllocated(uammd::access::gpu, uammd::access::read);
      auto d_torques_ptr = useTorque?torque.raw():nullptr;
      if(fcm){
	return fcm->computeHydrodynamicDisplacements(pos.raw(), force.raw(),  d_torques_ptr,
						     numberParticles, 0.0, 0.0, st);
      }
      else if(dpstokes){
	return dpstokes->Mdot(pos.raw(), force.raw(),
			      d_torques_ptr, numberParticles, st);
      }
    }
  public:
    std::shared_ptr<DPStokesSlab> dpstokes;
    std::shared_ptr<FCM> fcm;
    std::shared_ptr<uammd::System> sys;
    std::shared_ptr<uammd::ParticleData> pd;
    int numberParticles;
    cudaStream_t st;
    thrust::device_vector<uammd::real3> tmp;
    real zOrigin;

    DPStokesUAMMD(PyParameters pypar, int numberParticles): numberParticles(numberParticles){
      this->sys = std::make_shared<uammd::System>();
      this->pd = std::make_shared<uammd::ParticleData>(numberParticles, sys);
      if(pypar.mode.compare("periodic")==0){
	auto par = createFCMParameters(pypar);
	this->fcm = std::make_shared<FCM>(par);
	zOrigin = 0;
      }
      else{
	auto par = createDPStokesParameters(pypar);
	this->dpstokes = std::make_shared<DPStokesSlab>(par);
	zOrigin = pypar.zmin + par.H*0.5;
      }
      CudaSafeCall(cudaStreamCreate(&st));
    }

    //Copy positions to UAMMD's ParticleData
    void setPositions(const real* h_pos){
      tmp.resize(numberParticles);
      auto pos = pd->getPos(uammd::access::gpu, uammd::access::write);
      thrust::copy((uammd::real3*)h_pos, (uammd::real3*)h_pos + numberParticles,
		   tmp.begin());
      thrust::transform(thrust::cuda::par.on(st), tmp.begin(), tmp.end(),
			pos.begin(), Real3ToReal4SubstractOriginZ(zOrigin));
    }

    //Compute the hydrodynamic displacements due to a series of forces and/or torques acting on the particles
    void Mdot(const real* h_forces, const real* h_torques,
	      real* h_MF,
	      real* h_MT){
      tmp.resize(numberParticles);
      bool useTorque = h_torques;//.size() != 0;
      {
	auto force = pd->getForce(uammd::access::gpu, uammd::access::write);
	thrust::copy((uammd::real3*)h_forces, (uammd::real3*)h_forces + numberParticles, tmp.begin());
	thrust::transform(thrust::cuda::par.on(st),
			  tmp.begin(), tmp.end(), force.begin(), Real3ToReal4());
      }
      if(useTorque){
	auto torque = pd->getTorque(uammd::access::gpu, uammd::access::write);
	thrust::copy((uammd::real3*)h_torques, (uammd::real3*)h_torques + numberParticles, tmp.begin());
	thrust::transform(thrust::cuda::par, tmp.begin(), tmp.end(), torque.begin(), Real3ToReal4());
      }
      auto mob = this->computeHydrodynamicDisplacements(useTorque);
      thrust::copy(mob.first.begin(), mob.first.end(), (uammd::real3*)h_MF);
      if(mob.second.size()){
	thrust::copy(mob.second.begin(), mob.second.end(), (uammd::real3*)h_MT);
      }
    }

    ~DPStokesUAMMD(){
      cudaDeviceSynchronize();
      cudaStreamDestroy(st);
    }

  };

  //Initialize the modules with a certain set of parameters
  //Reinitializes if the module was already initialized
  void DPStokesGlue::initialize(PyParameters pypar, int numberParticles){
    dpstokes = std::make_shared<DPStokesUAMMD>(pypar, numberParticles);
  }

  //Clears all memory allocated by the module.
  //This leaves the module in an unusable state until initialize is called again.
  void DPStokesGlue::clear(){
    dpstokes->sys->finish();
    dpstokes.reset();
  }

  //Set positions to compute mobility matrix
  void DPStokesGlue::setPositions(const real* h_pos){
    throwIfInvalid();
    dpstokes->setPositions(h_pos);
    this->numberParticles = dpstokes->numberParticles;
  }

  //Compute the dot product of the mobility matrix with the forces and/or torques acting on the previously provided positions
  void DPStokesGlue::Mdot(const real* h_forces, const real* h_torques,
			  real* h_MF,
			  real* h_MT){
    throwIfInvalid();
    dpstokes->Mdot(h_forces, h_torques, h_MF, h_MT);
  }


  void DPStokesGlue::throwIfInvalid(){
    if(not dpstokes){
      throw std::runtime_error("DPStokes is not initialized. Call Initialize first");
    }
  }

  std::string getPrecision() {
#ifndef DOUBLE_PRECISION
    return "single";
#else
    return "double";
#endif
  }

  struct precision_type{
#ifndef DOUBLE_PRECISION
    using type = float;
#else
    using type = double;
#endif
  };

}
