/* Raul P. Pelaez 2020-2021. Batched Nbody evaluation of RPY kernels,
   Given N batches of particles (all with the same size NperBatch) the kernel nbodyBatchRPYGPU evaluates the matrix product RPY(ri, rj)*F ((NperBatchxNperBatch)*(3xNperBatch) size) for all particles inside each batch.
   If DOUBLE_PRECISION is defined the code will be compiled in double.
   Three algorithms are provided:
     Fast: Leverages shared memory to hide bandwidth latency
     Naive: A dumb thread-per-partice parallelization of the N^2 double loop
     Block: Assigns a block to each particle, the first thread then reduces the result of the whole block.

 */
#ifndef NBODY_RPY_CUH
#define NBODY_RPY_CUH
#include<thrust/device_vector.h>
#include<iostream>
#include"allocator.h"
#include "interface.h"
#include "vector.cuh"
#include "hydrodynamicKernels.cuh"

//These lines set up a special cached allocator container that effectively makes allocationg GPU memory free.
using resource = nbody_rpy::device_memory_resource;
using device_temporary_memory_resource = nbody_rpy::pool_memory_resource_adaptor<resource>;
template<class T> using allocator_thrust = nbody_rpy::polymorphic_allocator<T, device_temporary_memory_resource, thrust::cuda::pointer<T>>;
template<class T>  using cached_vector = thrust::device_vector<T, allocator_thrust<T>>;


namespace nbody_rpy{


  //Reference: Fast N-Body Simulation with CUDA. Chapter 31 of GPU Gems 3
  //This kernel loads batches of particles into shared memory to speed up the computation.
  //Threads will tipically read one value from global memory but blockDim.x from shared memory.
  template<class HydrodynamicKernel, class vecType>
  __global__ void computeRPYBatchedFastGPU(const vecType* pos,
					   const vecType* forces,
             const vecType* torques,
					   real3* Mv,
             real3* Mw,
					   HydrodynamicKernel kernel,
					   int Nbatches,
					   int NperBatch){
    const int tid = blockIdx.x*blockDim.x+threadIdx.x;
    const int N = Nbatches*NperBatch;
    const bool active = tid < N;
    const int id = tid;
    const int fiber_id = thrust::min(tid/NperBatch, Nbatches-1);
    const int blobsPerTile = blockDim.x;
    const int firstId = blockIdx.x*blobsPerTile;
    const int lastId =((firstId+blockDim.x)/NperBatch + 1)*NperBatch;
    const int fiberOfFirstId = firstId/NperBatch;
    const int tileOfFirstParticle = fiberOfFirstId*NperBatch/blobsPerTile;
    const int numberTiles = N/blobsPerTile;
    const int tileOfLastParticle = thrust::min(lastId/blobsPerTile, numberTiles);
    extern __shared__ char shMem[];
    vecType *shPos = (vecType*) (shMem);
    vecType *shForce = (vecType*) (shMem+blockDim.x*sizeof(vecType));
    const real3 pi= active?make_real3(pos[id]):real3();
    real3 MF = real3();
    for(int tile = tileOfFirstParticle; tile<=tileOfLastParticle; tile++){
      //Load tile to shared memory
      int i_load = tile*blockDim.x + threadIdx.x;
      if(i_load<N){
	shPos[threadIdx.x] = make_real3(pos[i_load]);
	shForce[threadIdx.x] = make_real3(forces[i_load]);
      }
      __syncthreads();
      //Compute interaction with all particles in tile
      if(active){
#pragma unroll 8
	for(uint counter = 0; counter<blockDim.x; counter++){
	  const int cur_j = tile*blockDim.x + counter;
	  const int fiber_j = cur_j/NperBatch;
	  if(fiber_id == fiber_j and cur_j<N){
	    const real3 fj = shForce[counter];
	    const real3 pj = shPos[counter];
	    MF += kernel.dotProduct_UF(pi, pj, fj);
      // TODO update function for torques
	  }
	}
      }
      __syncthreads();
    }
    if(active)
      Mv[id] = MF;
  }

  template<class HydrodynamicKernel, class vecType>
  void computeRPYBatchedFast(vecType* pos, vecType* force, vecType* torque, real3 *Mv, real3* Mw,
			     int Nbatches, int NperBatch,
			     HydrodynamicKernel &hydrodynamicKernel){
    int N = Nbatches*NperBatch;
    int nearestWarpMultiple = ((NperBatch+16)/32)*32;
    int minBlockSize = std::max(nearestWarpMultiple, 32);
    int Nthreads = std::min(std::min(minBlockSize, N), 256);
    int Nblocks  = (N+Nthreads-1)/Nthreads;
    computeRPYBatchedFastGPU<<<Nblocks, Nthreads, 2*Nthreads*sizeof(real3)>>>(pos,
									      force,
                        torque,
									      Mv,
                        Mw,
									      hydrodynamicKernel,
									      Nbatches, NperBatch);
    cudaDeviceSynchronize();
  }

  template<class HydrodynamicKernel, class vecType>
  //Naive N^2 algorithm (looks like x20 times slower than the fast kernel
  __global__ void computeRPYBatchedNaiveGPU(const vecType* pos,
					    const vecType* forces,
              const vecType* torques,
					    real3* Mv,
              real3* Mw,
					    HydrodynamicKernel kernel,
					    int Nbatches,
					    int NperBatch){
    const int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=Nbatches*NperBatch) return;
    real3 pi = make_real3(pos[tid]);
    real3 MF = real3();
    real3 MT = real3();
    int fiber_id = tid/NperBatch;
    for(int i= fiber_id*NperBatch; i<(fiber_id+1)*NperBatch; i++){
      if(i>=Nbatches*NperBatch) break;
      real3 pj = make_real3(pos[i]);
      real3 fj = make_real3(forces[i]);
      real3 tj = make_real3(torques[i]);
      MF += kernel.dotProduct_UF(pi, pj, fj);
      MT += kernel.dotProduct_WT(pi, pj, tj);
      // TODO add other dot products and modify above to add torques
    }
    Mv[tid] = MF;
    Mw[tid] = MT;
  }

  template<class HydrodynamicKernel, class vecType>
  void computeRPYBatchedNaive(vecType* pos, vecType* force, vecType* torque, real3 *Mv, real3 *Mw,
			      int Nbatches, int NperBatch,
			      HydrodynamicKernel &hydrodynamicKernel){
    int N = Nbatches*NperBatch;
    int minBlockSize = 128;
    int Nthreads = minBlockSize<N?minBlockSize:N;
    int Nblocks  = N/Nthreads+1;
    computeRPYBatchedNaiveGPU<<<Nblocks, Nthreads>>>(pos,
						     force,
                 torque,
						     Mv,
                 Mw,
						     hydrodynamicKernel,
						     Nbatches, NperBatch);
    cudaDeviceSynchronize();
  }


  template<class HydrodynamicKernel, class vecType>
  //NaiveBlock N^2 algorithm (looks like x20 times slower than the fast kernel
  __global__ void computeRPYBatchedNaiveBlockGPU(const vecType* pos,
						 const vecType* forces,
             const vecType* torque,
						 real3* Mv,
             real3* Mw,
						 HydrodynamicKernel &kernel,
						 int Nbatches,
						 int NperBatch){
    const int tid = blockIdx.x;
    if(tid>=Nbatches*NperBatch) return;
    real3 pi = make_real3(pos[tid]);
    extern __shared__ real3 MFshared[];
    real3 MF = real3();
    // real3 MT = real3();
    int fiber_id = tid/NperBatch;
    int last_id = thrust::min((fiber_id+1)*NperBatch, Nbatches*NperBatch);
    for(int i= fiber_id*NperBatch+threadIdx.x; i<last_id; i+=blockDim.x){
      real3 pj = make_real3(pos[i]);
      real3 fj = make_real3(forces[i]);
      // real3 tj = make_real3(torque[i]);
      MF += kernel.dotProduct_UF(pi, pj, fj);
      // MT += kernel.dotProduct_WT(pi, pj, tj);
    }
    MFshared[threadIdx.x] = MF;
    // MFshared[threadIdx.x + blockDim.x] = MT;
    __syncthreads();
    if(threadIdx.x == 0){
      auto MFTot = real3();
      for(int i =0; i<blockDim.x; i++){
	MFTot += MFshared[i];
      }
      Mv[tid] = MFTot;
    }

  }

  template<class HydrodynamicKernel, class vecType>
  void computeRPYBatchedNaiveBlock(vecType* pos, vecType* force, vecType* torque,
           real3 *Mv, real3 *Mw,
				   int Nbatches, int NperBatch,
				   HydrodynamicKernel &hydrodynamicKernel){
    int N = Nbatches*NperBatch;
    int minBlockSize = 128;
    int Nthreads = minBlockSize<N?minBlockSize:N;
    int Nblocks  = N;
    computeRPYBatchedNaiveBlockGPU<<<Nblocks, Nthreads, 4*Nthreads*sizeof(real3)>>>(pos,
										    force,
                        torque,
										    Mv,
                        Mw,
										    hydrodynamicKernel,
										    Nbatches, NperBatch);
    cudaDeviceSynchronize();
  }

  using LayoutType = real3;

  template<class HydrodynamicKernel>
  void callBatchedNBody(const real* h_pos, const real* h_forces, const real* h_torques,
				       real* h_MF, real* h_MT, int Nbatches, int NperBatch,
			HydrodynamicKernel &hydrodynamicKernel, algorithm alg){
    constexpr size_t elementsPerValue = sizeof(LayoutType)/sizeof(real);
    const int numberParticles = Nbatches * NperBatch;
    cached_vector<real> pos(h_pos, h_pos + elementsPerValue*numberParticles);
    cached_vector<real> forces(h_forces, h_forces + elementsPerValue*numberParticles);
    cached_vector<real> torques(h_torques, h_torques + elementsPerValue*numberParticles);
    cached_vector<real> Mv(elementsPerValue * numberParticles);
    cached_vector<real> Mw(elementsPerValue * numberParticles);
    auto kernel = computeRPYBatchedFast<HydrodynamicKernel, LayoutType>;
    if(alg==algorithm::naive)
      kernel = computeRPYBatchedNaive<HydrodynamicKernel, LayoutType>;
    else if(alg==algorithm::block)
      kernel = computeRPYBatchedNaiveBlock<HydrodynamicKernel, LayoutType>;
    kernel((LayoutType *)thrust::raw_pointer_cast(pos.data()),
	   (LayoutType *)thrust::raw_pointer_cast(forces.data()),
     (LayoutType *)thrust::raw_pointer_cast(torques.data()),
	   (LayoutType *)thrust::raw_pointer_cast(Mv.data()),
     (LayoutType *)thrust::raw_pointer_cast(Mw.data()),
	   Nbatches, NperBatch, hydrodynamicKernel);
    thrust::copy(Mv.begin(), Mv.end(), h_MF);
    thrust::copy(Mw.begin(), Mw.end(), h_MT);
  }

// Donev: These two seem identical to me except for the using HydrodynamicKernel = ???; line. Why can't there just be one routine callBatchedNBody that dispatches the right routine based on if(kernel == kernel_type::bottom_wall)? This doubling of code seems redundant
  void callBatchedNBodyOpenBoundaryRPY(const real* h_pos, const real* h_forces, const real* h_torques,
				       real* h_MF, real* h_MT, int Nbatches, int NperBatch,
				       real selfMobility, real rotMobility, real hydrodynamicRadius, algorithm alg){
    using HydrodynamicKernel = OpenBoundary;
    HydrodynamicKernel hydrodynamicKernel(selfMobility, rotMobility, hydrodynamicRadius);
    callBatchedNBody(h_pos, h_forces, h_torques, h_MF, h_MT, Nbatches, NperBatch, hydrodynamicKernel, alg);
  }

  void callBatchedNBodyBottomWallRPY(const real* h_pos, const real* h_forces, const real* h_torques,
				       real* h_MF, real* h_MT, int Nbatches, int NperBatch,
				       real selfMobility, real rotMobility, real hydrodynamicRadius, algorithm alg){
    using HydrodynamicKernel = BottomWall;
    HydrodynamicKernel hydrodynamicKernel(selfMobility, rotMobility, hydrodynamicRadius);
    callBatchedNBody(h_pos, h_forces, h_torques, h_MF, h_MT, Nbatches, NperBatch, hydrodynamicKernel, alg);
  }

}
#endif
