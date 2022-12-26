### Raul P. Pelaez 2021. Python interface to some UAMMD FCM hydrodynamic modules (Triply and doubly periodic)

This wrapper allows to compute the hydrodynamic displacements of a group of particles due to forces and/or torques acting on them: 
<!--- Donev: mobility.h does not have torques any more ;-) Does this python code work with torques also (nice if it did)? ---> 

 [Displacements; AngularDisplacements] = Mobility·[Forces; Torques]  
 
Hydrodynamic displacements can be computed in several domain geometries.  
In particular, this interface allows to access two uammd hydrodynamics modules:  
 -DPStokes for doubly periodic (DP) hydrodynamics with either no walls (nowall), a bottom wall (bottom) or two walls (slit)  
 -FCM for triply periodic (TP) hydrodynamics  

Both algorithms are grid based. Communication of the particles' forces and torques is carried out via the ES kernel.  
In TP, the grid is regular in all directions, while in DP the grid is defined in the Chebyshev basis in the third direction.  


## LIST OF PARAMETERS:

* viscosity: The viscosity of the solvent  

* mode:      The domain geometry, can be any of the following:  
       * "periodic": TP FCM  
       * "nowall":   DP, open boundaries in the third direction  
       * "bottom":   DP, there is a wall at the bottom of the domain  
       * "slit":     DP, there are walls at both the bottom and top of the domain  

* Lx,Ly:     The domain size in the plane (which is always periodic). Periodic wrapping is carried out by UAMMD, so the positions' origin in the plane is irrelevant.
<!--- Donev: Please confirm this is not [-Lx,Lx] i.e. length is 2*Lx as in the DPStokes paper --->  

* zmin, zmax: The domain size in the Z direction. The meaning of this parameter changes with each mode:  
             * In TP the domain is periodic in z with a size Lz=zmax-zmin (the origin of the particles' positions is irrelevant). 
             * In DP zmin and zmax denote the allowed range for the heights of the particles. In the DP modes particles must always be contained between zmin and zmax.  
                *For "nowall" this range is not physical and can be considered a requirement of the implementation (and similarly for zmax in the "bottom" wall case).  
                *For walled modes ("bottom" and "slit") zmin/zmax correspond to the locations of the bottom/top walls respectively.  

     <!--- Donev: Maybe it would be nice to point here to the script that determines these algorithm written by Sachin? ---> 

* w, alpha, beta: Parameters of the ES kernel to spread forces  
* w_d, alpha_d, beta_d: Parameters of the ES kernel to spread torques.  
     In both TP and DP torques are spreaded by performing the curl in fourier space, allowing to spread the torques using the same kernel as for the forces (instead of its derivative).  
     In all cases alpha defaults to w*h/2, where h is the grid size (Lx/nx). 

* nx, ny, nz: Number of grid cells in each direction.  


