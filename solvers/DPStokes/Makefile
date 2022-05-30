PYTHON=python3
INCLUDE_FLAGS:= `$(PYTHON)-config --includes`   -I ../../include -I ../../third_party/pybind11/include $(LAPACK_INCLUDE)
CXX=g++
NVCC=nvcc
MODULENAME=DPStokes
LIBRARY_NAME:=$(MODULENAME)$(shell $(PYTHON)-config --extension-suffix)

all: base $(LIBRARY_NAME) mobility.so

$(LIBRARY_NAME): python_wrapper.o
	$(CXX) -O3 -w -shared   $< -o $@ $(LAPACK_LIBS)


python_wrapper.o: python_wrapper.cpp mobility.h
	$(CXX) -O3 -c -fPIC -std=c++14 $(DOUBLEPRECISION) $(INCLUDE_FLAGS) $<  -o $@

mobility.so: DPStokes/uammd_gpu.o
	$(NVCC) -ccbin=$(CXX) -O3 -w -shared $< -o $@  -lcufft -lcublas $(LAPACK_LIBS)

base:
	make -C DPStokes
clean:
	make -C DPStokes clean
	rm -f $(LIBRARY_NAME) mobility.so python_wrapper.o
