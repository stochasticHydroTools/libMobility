## Usage example for the libMobility C++ interface 

The code example.cpp showcases how a solver can be included and used with libMobility.  


### Compilation

After [installing libMobility](https://libmobility.readthedocs.io) you can use CMake to compile this file:

```bash
	mkdir build
	cd build
	cmake ..
	make
```


### Running
A valid CUDA environment and a GPU are needed to run the example.  
After succesfully running, the mobilities of a group of randomly placed particles (with random forces acting on them) will be computed via an open boundary solver (NBody) and a triply periodic one (PSE) with a large box size. The displacements obtained by both methods are then printed.  
