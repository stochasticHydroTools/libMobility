CXX=g++
NVCC=nvcc
PYTHON=python3
MODULENAME=DPStokes
VERBOSITY=1

INCLUDE_FLAGS:= `$(PYTHON)-config --includes`   -I ../../include -I ../../third_party/pybind11/include $(LAPACK_INCLUDE)
UAMMD_ROOT=../../third_party/uammd
GPU_INCLUDE_FLAGS   = -I$(UAMMD_ROOT)/src -I$(UAMMD_ROOT)/src/third_party $(LAPACK_INCLUDE)
NVCCFLAGS=-ccbin=$(CXX) -O3 -DMAXLOGLEVEL=$(VERBOSITY) $(DOUBLEPRECISION) $(GPU_INCLUDE_FLAGS) -Xcompiler "-fPIC -w"
LIBRARY_NAME:=$(MODULENAME)$(shell $(PYTHON)-config --extension-suffix)

all: $(LIBRARY_NAME) mobility.so

$(LIBRARY_NAME): python_wrapper.cpp mobility.h
	$(CXX) -O3 -shared -fPIC -std=c++14 $(DOUBLEPRECISION) $(INCLUDE_FLAGS) $<  -o $@ $(LAPACK_LIBS)

mobility.so: extra/uammd_wrapper.cu
	$(NVCC) -shared -std=c++14 $(NVCCFLAGS) $< -o $@  -lcufft -lcublas  $(LAPACK_LIBS)

clean:
	rm -f $(LIBRARY_NAME) mobility.so
