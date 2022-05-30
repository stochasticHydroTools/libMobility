### Raul P. Pelaez 2021. Python interface to some UAMMD FCM hydrodynamic modules (Triply and doubly periodic)
Compile this python wrapper by running make at the root level of the repository.

Once compiled, a .so library will be available that can be used by python directly, see examples under python_interface.  

This interface allows to compute the hydrodynamic displacements of a group of particles due to forces and/or torques acting on them:  

 [Displacements; AngularDisplacements] = Mobility·[Forces; Torques]  
 
Hydrodynamic displacements can be computed in several domain geometries.  
In particular, this interface allows to access two uammd hydrodynamics modules:  
 -DPStokes for doubly periodic (DP) hydrodynamics with either no walls (nowall), a bottom wall (bottom) or two walls (slit)  
 -FCM for triply periodic (TP) hydrodynamics  

Both algorithms are grid based. Communication of the particles' forces and torques is carried out via the ES kernel.  
In TP, the grid is regular in all directions, while in DP the grid is defined in the Chebishev basis in the third direction.  

## USAGE:
### 0- Compile the library. 
It is required to pass arrays to the interface with the same precision as the library was compiled in. By default, the Makefile will compile UAMMD in single precision (corresponding to np.float32).
### 1- Import the uammd python wrapper:
  ```python
  import uammd
  ```
You will need numpy for passing data to uammd:  
```python 
 import numpy as np
 ```
### 2- Create the DPStokes object:  
```python 
 dpstokes = uammd.DPStokes()
 ```
### 3- Encapsulate the parameters in the uammd.StokesParameters object: 
 A list of parameters and their meaning can be found below  
 ```python 
 par = uammd.StokesParameters(viscosity=viscosity,
                             Lx=Lx,Ly=Ly,
                             zmin=zmin, zmax=zmax,
                             w=w, w_d=w_d,
                             alpha=alpha, alpha_d=alpha_d,
                             beta=beta, beta_d=beta_d,
                             nx=nx, ny=ny, nz=nz, mode=mode)
```
### 4- Initialize DPStokes with the parameters and the number of particles:  
```python 
 dpstokes.initialize(par, numberParticles)
```
 If some parameters need to change, simply call initialize again with the new ones.  
 Mind you, initialization is in general a really slow operation.  
### 5- Set the positions to construct the mobility matrix with:  
 All arrays have the format [x0 y0 z0 x1 y1 z1 ...] (interleaved)  
 For instance:  
 ```python 
 positions = np.array([0, 0, -3, 0, 0, 3])
 ```
 Corresponds to two particles, the first one located at (0,0,-3) and the second at (0,0,3)  
 ```python 
 dpstokes.setPositions(positions)
 ```
### 6- Compute hydrodynamic displacements for a group of forces and/or torques acting on the previously set positions  
```python 
 dpstokes.Mdot(forces=forces, torques=torques, velocities=MF, angularVelocities=MT)
 ```
 The torques and angularVelocities can be omited, in which case they are assumed to be zero:  
 ```python 
 dpstokes.Mdot(forces=forces, velocities=MF)
 ```
 Both MF and MT are output arguemnts and must be numpy arrays with the precision for which the library was compiled for.  
 Otherwise the output will be ignored by python.  
 The contents of both arrays will be overwritten.  
### 7- Clean up any memory allocated by the module, which will remain in an unusable state until initialization is called again  
```python 
 dpstokes.clear()
```



## LIST OF PARAMETERS:

* viscosity: The viscosity of the solvent  

* mode:      The domain geometry, can be any of the following:  
       * "periodic": TP FCM  
       * "nowall":   DP, open boundaries in the third direction  
       * "bottom":   DP, there is a wall at the bottom of the domain  
       * "slit":     DP, there are walls at both the bottom and top of the domain  

* Lx,Ly:     The domain size in the plane (which is always periodic). Periodic wrapping is carried out by UAMMD, so the positions' origin in the plane is irrelevant.  

* zmin, zmax: The domain size in the Z direction. The meaning of this parameter changes with each mode:  
             * In TP the domain is periodic in z with a size Lz=zmax-zmin (the origin of the particles' positions is irrelevant).  
             * In DP zmin and zmax denote the allowed range for the heights of the particles. In the DP modes particles must always be contained between zmin and zmax.  
                *For "nowall" this range is not physical and can be considered a requirement of the implementation (and similarly for zmax in the "bottom" wall case).  
                *For walled modes ("bottom" and "slit") zmin/zmax correspond to the locations of the bottom/top walls respectively.  

* w, alpha, beta: Parameters of the ES kernel to spread forces  
* w_d, alpha_d, beta_d: Parameters of the ES kernel to spread torques.  
     In both TP and DP torques are spreaded by performing the curl in fourier space, allowing to spread the torques using the same kernel as for the forces (instead of its derivative).  
     In all cases alpha defaults to w*h/2, where h is the grid size (Lx/nx).  

* nx, ny, nz: Number of grid cells in each direction.  
