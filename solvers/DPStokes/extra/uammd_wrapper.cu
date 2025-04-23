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
    auto computeHydrodynamicDisplacements(const auto* d_pos,
					  const uammd::real4* d_force,
					  const uammd::real4* d_torques,
					  int numberParticles, real dt, real dtTorque,
					  cudaStream_t st){
      if(fcm){
	return fcm->computeHydrodynamicDisplacements((uammd::real4*)(d_pos),
						     (uammd::real4*)(d_force),
						     (uammd::real4*)(d_torques),
						     numberParticles, 0.0, 0.0, st);
      }
      else if(dpstokes){
	return dpstokes->Mdot(reinterpret_cast<const uammd::real4*>(d_pos),
			      reinterpret_cast<const uammd::real4*>(d_force),
			      reinterpret_cast<const uammd::real4*>(d_torques),
			      numberParticles, st);
      }
    }
  public:
    std::shared_ptr<DPStokesSlab> dpstokes;
    std::shared_ptr<FCM> fcm;
    cudaStream_t st;
    thrust::device_vector<uammd::real3> tmp3;
    thrust::device_vector<uammd::real4> force4;
    thrust::device_vector<uammd::real4> torque4;
    thrust::device_vector<uammd::real4> stored_positions;
    real zOrigin;

    DPStokesUAMMD(PyParameters pypar){
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
    void setPositions(const real* d_pos, int numberParticles){
      stored_positions.resize(numberParticles);
      thrust::transform(thrust::cuda::par.on(st),
			reinterpret_cast<const uammd::real3*>(d_pos),
			reinterpret_cast<const uammd::real3*>(d_pos)+ numberParticles,
			stored_positions.begin(), Real3ToReal4SubstractOriginZ(zOrigin));
    }

    //Compute the hydrodynamic displacements due to a series of forces and/or torques acting on the particles
    void Mdot(const real* h_forces,
	      const real* h_torques,
	      real* h_MF,
	      real* h_MT, int numberParticles){
      force4.resize(numberParticles);
      bool useTorque = h_torques;
      force4.resize(numberParticles);
      thrust::transform(thrust::cuda::par.on(st),
			reinterpret_cast<const uammd::real3*>(h_forces), reinterpret_cast<const uammd::real3*>(h_forces) + numberParticles,
			force4.begin(), Real3ToReal4());
      if(useTorque){
	torque4.resize(numberParticles);
	thrust::transform(thrust::cuda::par.on(st),
			  reinterpret_cast<const uammd::real3*>(h_torques), reinterpret_cast<const uammd::real3*>(h_torques) + numberParticles,
			  torque4.begin(), Real3ToReal4());
      }
      auto mob = this->computeHydrodynamicDisplacements(stored_positions.data().get(),
							force4.data().get(),
							useTorque?torque4.data().get():nullptr,
							numberParticles, 0.0, 0.0, st);
      thrust::copy(thrust::cuda::par.on(st), mob.first.begin(), mob.first.end(), (uammd::real3*)h_MF);
      if(mob.second.size()){
	thrust::copy(thrust::cuda::par.on(st), mob.second.begin(), mob.second.end(), (uammd::real3*)h_MT);
      }
    }

    ~DPStokesUAMMD(){
      cudaDeviceSynchronize();
      cudaStreamDestroy(st);
    }

  };

  //Initialize the modules with a certain set of parameters
  //Reinitializes if the module was already initialized
  void DPStokesGlue::initialize(PyParameters pypar){
    dpstokes = std::make_shared<DPStokesUAMMD>(pypar);
  }

  //Clears all memory allocated by the module.
  //This leaves the module in an unusable state until initialize is called again.
  void DPStokesGlue::clear(){
    dpstokes.reset();
  }

  //Set positions to compute mobility matrix
  void DPStokesGlue::setPositions(const real* h_pos, int numberParticles){
    throwIfInvalid();
    dpstokes->setPositions(h_pos, numberParticles);
  }

  //Compute the dot product of the mobility matrix with the forces and/or torques acting on the previously provided positions
  void DPStokesGlue::Mdot(const real* h_forces, const real* h_torques,
			  real* h_MF,
			  real* h_MT, int numberParticles){
    throwIfInvalid();
    dpstokes->Mdot(h_forces, h_torques, h_MF, h_MT, numberParticles);
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
