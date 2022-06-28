# NBody solver

Raul P. Pelaez 2020-2022.  

Computes the RPY mobility using the GPU with a O(N^2) evaluation of the RPY tensor.  
	
Multiple hydrodynamic kernels are available (see extra/hydrodynamicKernels.cuh for a list).  
They are chosen via the libmobility configuration periodicity option at creation (see mobility.h).  

This module can work on batches of particles (hydrodynamically independent groups of particles), see setParametersNBody in mobility.h for more info.  


## About the GPU Batched RPY evaluator

Given a set of positions and forces acting on each particle, this module computes the product between the RPY tensor and the forces in the GPU.  

This module can work on batches of particles, all batches must have the same size. Note that a single batch is equivalent to every particle interacting to every other.  

Only the elements of the mobility matrix that correspond to pairs that belong to the same batch are non zero. It is equivalent to computing an NPerBatch^2 matrix-vector products for each batch separately.  
The data layout is 3 interleaved coordinates with each batch placed after the previous one: ```[x_1_1, y_1_1, z_1_1,...x_1_NperBatch,...x_Nbatches_NperBatch]```  

Three algorithms are provided, each one works better in different regimes (a.i. number of particles and batch size).  

	* Fast: Leverages shared memory to hide bandwidth latency  
	* Naive: A naive thread-per-partice parallelization of the N^2 double loop  
	* Block: Assigns a block to each particle, the first thread then reduces the result of the whole block.  
	
In general, testing suggests that Block is better for low sizes (less than 10K particles), Naive works best for intermediate sizes (10K-50K) and after that Fast is the better choice.  
As a side note, the reduction performed by Block is more accurate than the others, so while the results of Naive and Fast will be numerically identical, some differences due to numerical errors are expected between the latter and the former.  
