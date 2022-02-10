PYTHON=python3
PYBIND_ROOT=../../third_party/pybind11/
INCLUDE_FLAGS:= `$(PYTHON)-config --includes` -I $(PYBIND_ROOT)/include/ -I ../../include $(LAPACK_INCLUDE)
CXX=g++
NVCC=nvcc
MODULENAME=NBody_wall
LIBRARY_NAME:=$(MODULENAME)$(shell $(PYTHON)-config --extension-suffix)

BASE=Mobility_NBody_wall
all: $(LIBRARY_NAME) mobility.so

$(LIBRARY_NAME): python_wrapper.o base
	$(NVCC) -O3 -w -shared $(BASE)/mobility_cuda.o  $< -o $@ $(LAPACK_LIBS)


mobility.so: $(BASE)/mobility_cuda.o base
	$(NVCC) -O3 -w -shared $< -o $@ $(LAPACK_LIBS)

python_wrapper.o: python_wrapper.cpp mobility.h
	$(CXX) -O3 -c -fPIC -std=c++14 $(DOUBLEPRECISION) $(INCLUDE_FLAGS) $<  -o $@

base:
	make -C $(BASE)
clean:
	make -C $(BASE) clean
	rm -f $(LIBRARY_NAME)
