# Positively Split Ewald

A spectral method for computing the RPY mobility in triply periodic domains. Offers Ewald splitting, but falls back to a non-ewald force coupling method (FCM) (note importantly that a variant of FCM is also provided in solvers/DPStokes when periodic in all directions more efficiently, since a non-Gaussian kernel is used) when the splitting is turned off.  


### Arguments to setParametersPSE  

* **psi**: Splitting parameter. Defaults to 0 (not split)
* **Lx, Ly, Lz**: Box dimensions
* **shearStrain**: Shear strain parameter for the box along the y direction only. Defaults to 0

