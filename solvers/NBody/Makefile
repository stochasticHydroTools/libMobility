PYTHON=python3
PYBIND_ROOT=BatchedNBodyRPY/source/pybind11/
INCLUDE_FLAGS:= `$(PYTHON)-config --includes` -I $(PYBIND_ROOT)/include/ -I ../../include $(LAPACK_INCLUDE)
CXX=g++
NVCC=nvcc -ccbin=$(CXX)
MODULENAME=NBody
LIBRARY_NAME:=$(MODULENAME)$(shell $(PYTHON)-config --extension-suffix)

all: $(LIBRARY_NAME) mobility.so

$(LIBRARY_NAME): python_wrapper.o base
	$(NVCC) -O3 -w -shared BatchedNBodyRPY/source/NbodyRPY.o  $< -o $@ $(LAPACK_LIBS)


mobility.so: BatchedNBodyRPY/source/NbodyRPY.o base
	$(NVCC) -O3 -w -shared $< -o $@ $(LAPACK_LIBS)

python_wrapper.o: python_wrapper.cpp mobility.h
	$(CXX) -O3 -c -fPIC -std=c++14 $(DOUBLEPRECISION) $(INCLUDE_FLAGS) $<  -o $@

base:
	make -C BatchedNBodyRPY
clean:
	make -C BatchedNBodyRPY clean
	rm $(LIBRARY_NAME)
