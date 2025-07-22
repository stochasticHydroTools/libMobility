/*Raul P. Pelaez 2019-2022. Some utilities for debugging GPU code
 */
#ifndef DEBUGTOOLS_H
#define DEBUGTOOLS_H

#define CUDA_ERROR_CHECK

#ifdef LANCZOS_DEBUG
#define CUDA_ERROR_CHECK_SYNC
#endif
#define CudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
#define CudaCheckError() __cudaCheckError(__FILE__, __LINE__)

#include<string>
#include<exception>
#include<stdexcept>

inline void __cudaSafeCall(cudaError err, const char *file, const int line){
  #ifdef CUDA_ERROR_CHECK
  if (cudaSuccess != err){
    cudaGetLastError(); //Reset CUDA error status
    throw std::runtime_error("CudaSafeCall() failed at "+
			     std::string(file) + ":" + std::to_string(line)+
			     " with error " + std::to_string(err));
  }
  #endif
}

inline void __cudaCheckError(const char *file, const int line){
  cudaError err;
#ifdef CUDA_ERROR_CHECK_SYNC
  err = cudaDeviceSynchronize();
  if(cudaSuccess != err){
    throw std::runtime_error("CudaCheckError() with sync failed at "+
			     std::string(file) + ":" + std::to_string(line)+
			     " with error " + std::to_string(err));
  }
#endif
  err = cudaGetLastError();
  if(cudaSuccess != err){
    throw std::runtime_error("CudaSafeCall() failed at "+
			     std::string(file) + ":" + std::to_string(line)+
			     " with error " + std::to_string(err));
  }
}

#endif
