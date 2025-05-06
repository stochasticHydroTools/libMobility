/* Raul P. Pelaez 2020-2021. Batched Nbody evaluation of RPY kernels,
   Given N batches of particles (all with the same size NperBatch) the kernel
   nbodyBatchRPYGPU evaluates the matrix product RPY(ri, rj)*F
   ((NperBatchxNperBatch)*(3xNperBatch) size) for all particles inside each
   batch. If DOUBLE_PRECISION is defined the code will be compiled in double.
   Three algorithms are provided:
     Fast: Leverages shared memory to hide bandwidth latency
     Naive: A dumb thread-per-partice parallelization of the N^2 double loop
     Block: Assigns a block to each particle, the first thread then reduces the
   result of the whole block.

 */
#ifndef NBODY_RPY_CUH
#define NBODY_RPY_CUH
#include "hydrodynamicKernels.cuh"
#include "interface.h"
#include "vector.cuh"

namespace nbody_rpy {

// Reference: Fast N-Body Simulation with CUDA. Chapter 31 of GPU Gems 3
// This kernel loads batches of particles into shared memory to speed up the
// computation. Threads will tipically read one value from global memory but
// blockDim.x from shared memory.
template <class HydrodynamicKernel, class vecType, class Policy>
__global__ void computeRPYBatchedFastGPU(const vecType *pos,
                                         const vecType *forces,
                                         const vecType *torques, real3 *Mv,
                                         real3 *Mw, Constants constants,
                                         int Nbatches, int NperBatch) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int N = Nbatches * NperBatch;
  const bool active = tid < N;
  const int id = tid;
  const int fiber_id = thrust::min(tid / NperBatch, Nbatches - 1);
  const int blobsPerTile = blockDim.x;
  const int firstId = blockIdx.x * blobsPerTile;
  const int lastId = ((firstId + blockDim.x) / NperBatch + 1) * NperBatch;
  const int fiberOfFirstId = firstId / NperBatch;
  const int tileOfFirstParticle = fiberOfFirstId * NperBatch / blobsPerTile;
  const int numberTiles = N / blobsPerTile;
  const int tileOfLastParticle =
      thrust::min(lastId / blobsPerTile, numberTiles);
  extern __shared__ char shMem[];
  vecType *shPos = (vecType *)(shMem);
  vecType *shForce = nullptr;
  if (forces) {
      shForce = (vecType *)(shMem + blockDim.x * sizeof(vecType));
  }
  vecType *shTorque = nullptr;
  if (torques) {
    shTorque = (vecType *)(shMem + 2 * blockDim.x * sizeof(vecType));
  }

  const real3 pi = active ? make_real3(pos[id]) : real3();
  real3 MF = real3();
  real3 MT = real3();
  for (int tile = tileOfFirstParticle; tile <= tileOfLastParticle; tile++) {
    // Load tile to shared memory
    int i_load = tile * blockDim.x + threadIdx.x;
    if (i_load < N) {
      shPos[threadIdx.x] = make_real3(pos[i_load]);
      if (forces)
        shForce[threadIdx.x] = make_real3(forces[i_load]);
      if (torques)
        shTorque[threadIdx.x] = make_real3(torques[i_load]);
    }
    __syncthreads();
    // Compute interaction with all particles in tile
    if (active) {
#pragma unroll 8
      for (uint counter = 0; counter < blockDim.x; counter++) {
        const int cur_j = tile * blockDim.x + counter;
        const int fiber_j = cur_j / NperBatch;
        if (fiber_id == fiber_j and cur_j < N) {
          const real3 pj = shPos[counter];
          const real3 fj = forces ? shForce[counter] : real3();
          const real3 tj = torques ? shTorque[counter] : real3();
          const mdot_result dot = HydrodynamicKernel::template dotProduct<Policy>(pi, pj, fj, tj, constants);
          MF += dot.MF;
          MT += dot.MT;
        }
      }
    }
    __syncthreads();
  }
  if (active) {
    if(Mv)
      Mv[id] = MF;
    if (Mw)
      Mw[id] = MT;
  }
}

template <class HydrodynamicKernel, class vecType, class Policy>
void computeRPYBatchedFast(const vecType *pos, const vecType *force,
                           const vecType *torque, real3 *Mv, real3 *Mw,
                           int Nbatches, int NperBatch,
                           Constants &constants) {
  int N = Nbatches * NperBatch;
  int nearestWarpMultiple = ((NperBatch + 16) / 32) * 32;
  int minBlockSize = std::max(nearestWarpMultiple, 32);
  int Nthreads = std::min(std::min(minBlockSize, N), 256);
  int Nblocks = (N + Nthreads - 1) / Nthreads;
  int sharedMemoryFactor = torque != nullptr ? 3 : 2;
  computeRPYBatchedFastGPU<HydrodynamicKernel, vecType, Policy>
      <<<Nblocks, Nthreads, sharedMemoryFactor * Nthreads * sizeof(real3)>>>(
          pos, force, torque, Mv, Mw, constants, Nbatches, NperBatch);
}

template <class HydrodynamicKernel, class vecType, class Policy>
// Naive N^2 algorithm (looks like x20 times slower than the fast kernel
__global__ void computeRPYBatchedNaiveGPU(const vecType *pos,
                                          const vecType *forces,
                                          const vecType *torques, real3 *Mv,
                                          real3 *Mw, Constants constants,
                                          int Nbatches, int NperBatch) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= Nbatches * NperBatch)
    return;
  real3 pi = make_real3(pos[tid]);
  real3 MF = real3();
  real3 MT = real3();
  int fiber_id = tid / NperBatch;
  for (int i = fiber_id * NperBatch; i < (fiber_id + 1) * NperBatch; i++) {
    if (i >= Nbatches * NperBatch)
      break;
    real3 pj = make_real3(pos[i]);
    const real3 fj = forces ? make_real3(forces[i]) : real3();
    // real3 fj = make_real3(forces[i]);
    const real3 tj = torques ? make_real3(torques[i]) : real3();
    mdot_result dot = HydrodynamicKernel::template dotProduct<Policy>(pi, pj, fj, tj, constants);
    if(forces)
      MF += dot.MF;
    if (torques)
      MT += dot.MT;
  }
  if(forces)
    Mv[tid] = MF;
  if (Mw)
    Mw[tid] = MT;
}

template <class HydrodynamicKernel, class vecType, class Policy>
void computeRPYBatchedNaive(const vecType *pos, const vecType *force,
                            const vecType *torque, real3 *Mv, real3 *Mw,
                            int Nbatches, int NperBatch,
                            Constants &constants) {
  int N = Nbatches * NperBatch;
  int minBlockSize = 128;
  int Nthreads = minBlockSize < N ? minBlockSize : N;
  int Nblocks = N / Nthreads + 1;
  computeRPYBatchedNaiveGPU<HydrodynamicKernel, vecType, Policy><<<Nblocks, Nthreads>>>(
      pos, force, torque, Mv, Mw, constants, Nbatches, NperBatch);
}

template <class HydrodynamicKernel, class vecType, class Policy>
// NaiveBlock N^2 algorithm (looks like x20 times slower than the fast kernel
__global__ void computeRPYBatchedNaiveBlockGPU(
    const vecType *pos, const vecType *forces, const vecType *torque, real3 *Mv,
    real3 *Mw, Constants constants, int Nbatches, int NperBatch) {
  const int tid = blockIdx.x;
  if (tid >= Nbatches * NperBatch)
    return;
  real3 pi = make_real3(pos[tid]);
  extern __shared__ real3 sharedMemory[];
  real3 MF = real3();
  real3 MT = real3();
  int fiber_id = tid / NperBatch;
  int last_id = thrust::min((fiber_id + 1) * NperBatch, Nbatches * NperBatch);
  for (int i = fiber_id * NperBatch + threadIdx.x; i < last_id;
       i += blockDim.x) {
    real3 pj = make_real3(pos[i]);
    // real3 fj = make_real3(forces[i]);
    const real3 fj = forces ? make_real3(forces[i]) : real3();
    const real3 tj = torque ? make_real3(torque[i]) : real3();
    mdot_result dot = HydrodynamicKernel::template dotProduct<Policy>(pi, pj, fj, tj, constants);
    MF += dot.MF;
    MT += dot.MT;
  }
  if(forces)
    sharedMemory[threadIdx.x] = MF;
  if (torque)
    sharedMemory[threadIdx.x + blockDim.x] = MT;
  __syncthreads();
  if (threadIdx.x == 0) {
    auto MFTot = real3();
    auto MTTot = real3();
    for (int i = 0; i < blockDim.x; i++) {
      if(forces)
        MFTot += sharedMemory[i];
      if (torque)
        MTTot += sharedMemory[i + blockDim.x];
    }
    if(forces)
      Mv[tid] = MFTot;
    if (torque)
      Mw[tid] = MTTot;
  }
}

template <class HydrodynamicKernel, class vecType, class Policy>
void computeRPYBatchedNaiveBlock(const vecType *pos, const vecType *force,
                                 const vecType *torque, real3 *Mv, real3 *Mw,
                                 int Nbatches, int NperBatch,
                                 Constants &constants) {
  int N = Nbatches * NperBatch;
  int minBlockSize = 128;
  int Nthreads = minBlockSize < N ? minBlockSize : N;
  int Nblocks = N;
  int sharedMemoryFactor = torque ? 2 : 1;
  computeRPYBatchedNaiveBlockGPU<HydrodynamicKernel, vecType, Policy>
      <<<Nblocks, Nthreads, sharedMemoryFactor * Nthreads * sizeof(real3)>>>(
          pos, force, torque, Mv, Mw, constants, Nbatches, NperBatch);
}

using LayoutType = real3;

template <class HydrodynamicKernel, class Policy>
void batchedNBody(device_span<const real> ipos, device_span<const real> iforces,
                  device_span<const real> itorques, device_span<real> iMF,
                  device_span<real> iMT, int Nbatches, int NperBatch,
                  Constants& constants, algorithm alg) {
  if (ipos.size() < Nbatches * NperBatch * 3)
    throw std::runtime_error("Not enough space in pos");
  device_adapter<const real> pos(ipos, device::cuda);
  device_adapter<const real> forces(iforces, device::cuda);
  device_adapter<const real> torques(itorques, device::cuda);
  device_adapter<real> MF(iMF, device::cuda);
  device_adapter<real> MT(iMT, device::cuda);
  auto kernel = computeRPYBatchedFast<HydrodynamicKernel, LayoutType, Policy>;
  if (alg == algorithm::naive)
    kernel = computeRPYBatchedNaive<HydrodynamicKernel, LayoutType, Policy>;
  else if (alg == algorithm::block)
    kernel = computeRPYBatchedNaiveBlock<HydrodynamicKernel, LayoutType, Policy>;
  kernel(reinterpret_cast<const LayoutType *>(pos.data()),
         reinterpret_cast<const LayoutType *>(forces.data()),
         reinterpret_cast<const LayoutType *>(torques.data()),
         reinterpret_cast<LayoutType *>(MF.data()),
         reinterpret_cast<LayoutType *>(MT.data()), Nbatches, NperBatch,
         constants);
}

void callBatchedNBody(device_span<const real> pos,
                      device_span<const real> forces,
                      device_span<const real> torques, device_span<real> MF,
                      device_span<real> MT, int Nbatches, int NperBatch,
                      real selfMobility, real rotMobility,
                      real transRotMobility, real hydrodynamicRadius, bool needsTorque,
                      algorithm alg, kernel_type kernel) {

    Constants constants(selfMobility, rotMobility, transRotMobility, hydrodynamicRadius, needsTorque);

    if (kernel == kernel_type::bottom_wall)
    {
        if (forces.size() > 0 && torques.size() > 0)
            batchedNBody<BottomWall, ForcesAndTorques>(pos, forces, torques, MF, MT, Nbatches, NperBatch,
                        constants, alg);
        else if (forces.size() > 0)
            batchedNBody<BottomWall, ForcesOnlyWall>(pos, forces, torques, MF, MT, Nbatches, NperBatch,
                        constants, alg);
        else if (torques.size() > 0)
            batchedNBody<BottomWall, TorquesOnlyWall>(pos, forces, torques, MF, MT, Nbatches, NperBatch,
                        constants, alg);
    }
    else if (kernel == kernel_type::open_rpy)
    {
        if (forces.size() > 0 && torques.size() > 0)
            batchedNBody<OpenBoundary, ForcesAndTorques>(pos, forces, torques, MF,
                MT, Nbatches, NperBatch,constants, alg);
        else if (forces.size() > 0)
            batchedNBody<OpenBoundary, ForcesOnlyWall>(pos, forces, torques, MF, MT,
                Nbatches, NperBatch,constants, alg);
        else if (torques.size() > 0)
            batchedNBody<OpenBoundary, TorquesOnlyWall>(pos, forces, torques, MF,
                MT, Nbatches, NperBatch, constants, alg);
    }
    else
    {
        throw std::runtime_error("Unknown kernel type");
    }
}

} // namespace nbody_rpy
#endif
