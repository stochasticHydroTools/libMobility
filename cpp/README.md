## Usage example for the libMobility C++ interface 

The code example.cpp showcases how a solver can be included and used with libMobility.  


### Compilation
In order to succesfully compile example.cpp, ensure the Makefile in the root directory of the repo has generated the libraries for all the modules used (in this case ../solvers/NBody/mobility.so and ../solvers/PSE/mobility.so).  

It is required to compile using the same precision as the libraries, so modify the Makefile if necessary (single precision is the default).  
Running the Makefile at the root folder, instead of the one directly in this folder, will take care of that.

### Running
A valid CUDA environment and a GPU are needed to run the example.  
After succesfully running, the mobilities of a group of randomly placed particles (with random forces acting on them) will be computed via an open boundary solver (NBody) and a triply periodic one (PSE) with a large box size. The displacements obtained by both methods are then printed.  
