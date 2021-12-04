PYTHON=python3
PYBIND_ROOT=UAMMD_PSE_Python/pybind11/
INCLUDE_FLAGS:= `$(PYTHON)-config --includes` -I $(PYBIND_ROOT)/include/ -I../../include
CXX=g++
NVCC=nvcc
MODULENAME=PSE
LIBRARY_NAME:=$(MODULENAME)$(shell $(PYTHON)-config --extension-suffix)

all: $(LIBRARY_NAME)

$(LIBRARY_NAME): python_wrapper.o base
	$(NVCC) -O3 -w -shared UAMMD_PSE_Python/uammd_wrapper.o  $< -o $@ -lcufft -lcublas -lmkl_rt -lpthread -ldl

python_wrapper.o: python_wrapper.cpp mobility.h
	$(CXX) -O3 -c -fPIC -std=c++14 $(DOUBLEPRECISION) $(INCLUDE_FLAGS) $<  -o $@
base:
	make -C UAMMD_PSE_Python
clean:
	make -C UAMMD_PSE_Python clean
	rm -f $(LIBRARY_NAME)
