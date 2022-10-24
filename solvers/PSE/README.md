# Positively Split Ewald

A pseudo spectral method for computing the RPY mobility in triply periodic domains. Offers Ewald splitting, but falls back to a non-ewald force coupling method when the splitting is turned off.  


### Arguments to setParametersPSE  

* **psi**: Splitting parameter. Defaults to 0 (not split)
* **Lx, Ly, Lz**: Box dimensions
* **shearStrain**: Shear strain parameter. Defaults to 0

