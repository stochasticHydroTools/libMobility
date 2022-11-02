PYTHON=python3
PYBIND_ROOT=../../third_party/pybind11
INCLUDE_FLAGS:= `$(PYTHON)-config --includes` -I $(PYBIND_ROOT)/include/ -I ../../include $(LAPACK_INCLUDE)
CXX=g++
NVCC=nvcc -ccbin=$(CXX)
MODULENAME=NBody
LIBRARY_NAME:=$(MODULENAME)$(shell $(PYTHON)-config --extension-suffix)

all: $(LIBRARY_NAME) mobility.so

$(LIBRARY_NAME): python_wrapper.o extra/NbodyRPY.o
	$(NVCC) -O3 -w -shared $^ -o $@ $(LAPACK_LIBS)

mobility.so: extra/NbodyRPY.o
	$(NVCC) -O3 -w -shared $< -o $@ $(LAPACK_LIBS)

python_wrapper.o: python_wrapper.cpp mobility.h
	$(CXX) -O3 -c -fPIC -std=c++14 $(DOUBLEPRECISION) $(INCLUDE_FLAGS) $<  -o $@

extra/NbodyRPY.o: extra/NbodyRPY.cu
	$(NVCC) -O3 -c -Xcompiler=-fPIC -std=c++14 $(DOUBLEPRECISION) $(INCLUDE_FLAGS) $< -o $@

clean:
	rm -f python_wrapper.o extra/NbodyRPY.o $(LIBRARY_NAME) mobility.so
