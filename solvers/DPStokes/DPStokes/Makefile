
UAMMD_ROOT=uammd

GPU_INCLUDE_FLAGS   = -I$(UAMMD_ROOT)/src -I$(UAMMD_ROOT)/src/third_party

PYBIND11=-I../../../third_party/pybind11/include

GPU_OPTIMIZATION    = -O3

PYTHON3=python3
GPU_LIBRARY_NAME    = dpstokes$(shell $(PYTHON3)-config --extension-suffix)

LAPACKE_FLAGS = $(LAPACKE_INCLUDE) $(LAPACKE_LIBS)
VERBOSITY=1

CXX=g++
NVCC=nvcc

all: dpstokesGPU

dpstokesGPU: $(GPU_LIBRARY_NAME)

uammd_gpu.o: uammd_wrapper.cu uammd_interface.h
	$(NVCC) -w -std=c++14 -ccbin=$(CXX) -DMAXLOGLEVEL=$(VERBOSITY) $(GPU_OPTIMIZATION) $(DOUBLEPRECISION) $(GPU_INCLUDE_FLAGS) -Xcompiler "-fPIC -w" -c $< -o $@  $(LAPACKE_FLAGS)

uammd_python.o: uammd_python.cpp uammd_interface.h
	$(CXX) -std=c++14 -O3 $(DOUBLEPRECISION) $(CPU_DEBUG) $(CPU_OPTIMIZATION) -fPIC -w `$(PYTHON3)-config --includes` -I $(PYBIND_ROOT)/include/ -c $< -o $@ $(PYBIND11)

$(GPU_LIBRARY_NAME): uammd_gpu.o uammd_python.o
	$(NVCC) -ccbin=$(CXX) $(DOUBLEPRECISION) $(GPU_OPTIMIZATION) $(GPU_DEBUG) -w -shared $^ -o $@ -lcufft -lnvToolsExt $(LAPACKE_LIBS)

clean:
	rm -f *.o *.so
